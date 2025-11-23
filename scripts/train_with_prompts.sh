#!/usr/bin/env bash
# Train PPO by sending multi-step prompts (via planner -> router -> LLMs)
# Usage:
#   ./scripts/train_with_prompts.sh [train_file] [--epochs N] [--max-tokens N] [--save-every K] [--shuffle]
# Defaults:
#   train_file   = train_prompts.txt
#   epochs       = 1
#   max_tokens   = 200
#   save_every   = 50   (save PPO checkpoint every K prompts)
#   shuffle      = off  (use --shuffle to randomize prompt order each epoch)

set -euo pipefail

TRAIN_FILE="${1:-train_prompts.txt}"
shift || true

EPOCHS=1
MAX_TOKENS=200
SAVE_EVERY=50
SHUFFLE=0

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)      EPOCHS="${2}"; shift 2 ;;
    --max-tokens)  MAX_TOKENS="${2}"; shift 2 ;;
    --save-every)  SAVE_EVERY="${2}"; shift 2 ;;
    --shuffle)     SHUFFLE=1; shift ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PLANNER_URL="${PLANNER_URL:-http://localhost:9000}"
ROUTER_URL="${ROUTER_URL:-http://localhost:8080}"

echo "Planner:  ${PLANNER_URL}"
echo "Router:   ${ROUTER_URL}"
echo "Train:    ${TRAIN_FILE}"
echo "Epochs:   ${EPOCHS}"
echo "Max tok:  ${MAX_TOKENS}"
echo "Save every ${SAVE_EVERY} prompts"
echo "Shuffle:  $([[ $SHUFFLE -eq 1 ]] && echo yes || echo no)"
echo

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "ERROR: $TRAIN_FILE not found." >&2
  exit 1
fi

# --- helpers ---------------------------------------------------------------

wait_for() {
  local url="$1"
  local name="$2"
  echo "Waiting for $name at $url ..."
  for i in {1..60}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is up."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $name did not become healthy: $url" >&2
  exit 1
}

save_policy() {
  # Safe to call for epsilon too (router will 400); ignore failures.
  curl -fsS -X POST "${ROUTER_URL}/policy/save" >/dev/null 2>&1 || true
}

cleanup() {
  echo
  echo "Caught signal; saving PPO checkpoint..."
  save_policy
  echo "Saved. Exiting."
  exit 0
}

trap cleanup INT TERM

# --- wait for services -----------------------------------------------------
wait_for "${PLANNER_URL}/health" "planner"
wait_for "${ROUTER_URL}/health"  "router"

# --- split file into blocks separated by blank lines -----------------------
# Each block should contain the 3-step prompt (lines starting with 1) 2) 3))
# We’ll keep empty-line separators intact to detect blocks.
mapfile -t BLOCKS < <(awk -v RS= '{gsub(/\r/,""); print}' "$TRAIN_FILE")

total_blocks="${#BLOCKS[@]}"
if [[ $total_blocks -eq 0 ]]; then
  echo "No blocks detected in $TRAIN_FILE (expected blank-line-separated 3-step prompts)." >&2
  exit 1
fi

# --- training loop ---------------------------------------------------------
COUNT=0
for epoch in $(seq 1 "$EPOCHS"); do
  echo
  echo "===== EPOCH $epoch / $EPOCHS ====="
  # Build an index array and optionally shuffle it
  IDX=()
  for i in "${!BLOCKS[@]}"; do IDX+=("$i"); done
  if [[ $SHUFFLE -eq 1 ]]; then
    # requires coreutils 'shuf'
    mapfile -t IDX < <(printf "%s\n" "${IDX[@]}" | shuf)
  fi

  for i in "${IDX[@]}"; do
    block="${BLOCKS[$i]}"
    # Skip if block is empty after trimming
    if [[ -z "$(echo "$block" | tr -d ' \t\n\r')" ]]; then
      continue
    fi

    COUNT=$((COUNT+1))
    # Build JSON safely using jq -Rs to escape newlines etc.
    payload=$(jq -Rs --arg mt "$MAX_TOKENS" '{prompt: ., max_new_tokens: ($mt|tonumber)}' <<< "$block")

    # Send to planner (which calls router for each step)
    # If there’s a transient error, retry a couple of times.
    ok=0
    for attempt in 1 2 3; do
      resp=$(curl -sS -X POST "${PLANNER_URL}/plan_run" -H "content-type: application/json" -d "$payload" || true)
      if [[ -n "$resp" ]] && jq -e 'has("final_answer")' >/dev/null 2>&1 <<<"$resp"; then
        ok=1
        break
      else
        echo "WARN: plan_run failed (attempt ${attempt}). Response:"
        echo "$resp" | sed 's/^/  /'
        sleep 2
      fi
    done

    if [[ $ok -ne 1 ]]; then
      echo "ERROR: failed to process block index $i after retries; continuing…" >&2
    else
      # Print a tiny one-line summary (chosen backends and latencies)
      summary=$(jq -r '[.steps[] | {b:.chosen_backend,lat:(.latency_s//null)}] | tostring' <<<"$resp" 2>/dev/null || echo "[]")
      echo "[$COUNT/$total_blocks @ epoch $epoch] steps: $summary"
    fi

    # Save periodically (router also may autosave if configured)
    if (( COUNT % SAVE_EVERY == 0 )); then
      echo "Saving PPO checkpoint at step $COUNT..."
      save_policy
    fi
  done
done

echo
echo "Training passes complete. For safety, saving PPO checkpoint one more time…"
save_policy
echo "Done."
