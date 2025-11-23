#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <prompts_file> <tag: code|explain|tests>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPTS_FILE="$1"
TAG="$2"

# Bring up router in epsilon mode
docker compose -f "$ROOT_DIR/docker-compose.yml" -f "$ROOT_DIR/docker-compose.epsilon.yml" up -d --build rl_router

echo "[EPS] warm-up 3 requests…"
for i in 1 2 3; do
  curl -s -X POST http://localhost:8080/infer \
    -H "content-type: application/json" \
    -d '{"prompt":"Warm up: say hello","max_new_tokens":32,"step_tag":"'"$TAG"'"}' >/dev/null || true
  sleep 0.7
done

mkdir -p "$ROOT_DIR/logs"

echo "[EPS] running dataset $PROMPTS_FILE with tag=$TAG"
MAX_REQUESTS="${MAX_REQUESTS:-0}"  # 0 = no cap, otherwise stop after N lines
count=0
while IFS= read -r line || [[ -n "$line" ]]; do
  # Optional cap
  if [[ "$MAX_REQUESTS" -gt 0 && "$count" -ge "$MAX_REQUESTS" ]]; then
    echo "[INFO] Reached MAX_REQUESTS=$MAX_REQUESTS — stopping."
    break
  fi

  # Escape JSON safely
  esc=$(printf '%s' "$line" | jq -Rsa . | sed 's/^"//;s/"$//')

  # Send via planner so step_tag flows through
  curl -s -X POST http://localhost:9000/plan_run \
    -H "content-type: application/json" \
    -d "{\"prompt\":\"$esc\",\"max_new_tokens\":200}" >/dev/null || true

  count=$((count+1))
  sleep 0.3
done < "$PROMPTS_FILE"

# Snapshot router log atomically with timestamp
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
SRC="$LOG_DIR/route_log_epsilon.jsonl"          # <-- matches ROUTER_JSONL
DST="$LOG_DIR/route_log_epsilon_${TAG}_${TS}.jsonl"

if [[ -f "$SRC" ]]; then
  cp "$SRC" "$DST"
  echo "[PPO] saved $DST"
else
  echo "⚠️  $SRC not found — did requests hit the router?"
fi
