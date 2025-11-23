#!/usr/bin/env bash
# Evaluate prompts from scripts/eval_prompts.txt through planner → router.
# Safely JSON-encodes each line, handles CRLF, and shows per-request status.

set -u

PLANNER_URL=${PLANNER_URL:-"http://localhost:9000"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${1:-$SCRIPT_DIR/eval_prompts.txt}"

ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$ROOT_DIR/logs/route_log.jsonl"

# --- checks ---
if [ ! -f "$PROMPT_FILE" ]; then
  echo "❌ Prompt file not found: $PROMPT_FILE"
  exit 1
fi

if ! curl -fs "$PLANNER_URL/health" >/dev/null; then
  echo "❌ Planner not reachable at $PLANNER_URL"
  exit 1
fi

mkdir -p "$ROOT_DIR/logs"

echo "🚀 Evaluating prompts from: $PROMPT_FILE"
echo "🧠 Planner URL: $PLANNER_URL"
echo "🪵 Log file:    $LOG_FILE"
echo

LINE=0
# Use LC_ALL=C to avoid locale surprises; strip CR if present; skip empty lines.
while IFS= read -r raw || [ -n "$raw" ]; do
  # Strip trailing CR from Windows-encoded files
  prompt="${raw%$'\r'}"
  # Skip blank lines
  if [ -z "$prompt" ]; then
    continue
  fi

  ((LINE++))
  # Build JSON safely: jq -Rs JSON-encodes the entire line as a string.
  DATA=$(jq -Rs --argjson max 140 '{prompt:., max_new_tokens:$max}' <<<"$prompt")

  # Send and display concise status (HTTP code + short message)
  HTTP_CODE=$(curl -sS -o /tmp/plan_run.out -w "%{http_code}" \
    -X POST "$PLANNER_URL/plan_run" \
    -H "content-type: application/json" \
    --data "$DATA")

  if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ [$LINE] OK"
  else
    echo "❗ [$LINE] HTTP $HTTP_CODE"
    head -c 200 /tmp/plan_run.out | sed 's/$/…/' || true
    echo
  fi

  # Gentle pacing so you can watch logs moving
  sleep 0.3
done < "$PROMPT_FILE"

echo
echo "✅ Done. Watch live:"
echo "  tail -f \"$LOG_FILE\" | jq -c '{step:.step_tag,backend:.backend,lat:.latency_s,reward:.reward}'"
