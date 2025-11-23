#!/usr/bin/env bash
# Run evaluation using the epsilon-greedy baseline (no PPO learning).
# Uses docker-compose.epsilon.yml to temporarily switch the router policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="${1:-$SCRIPT_DIR/eval_prompts.txt}"

PLANNER_URL=${PLANNER_URL:-"http://localhost:9000"}
EPS_LOG="$ROOT_DIR/logs/route_log_epsilon.jsonl"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "❌ Prompt file not found: $PROMPT_FILE"
  exit 1
fi

echo "🔁 Switching router to epsilon baseline…"
docker compose up -d --build rl_router

echo "⏳ Waiting for router health…"
for i in {1..30}; do
  if curl -fs http://localhost:8080/health >/dev/null; then break; fi
  sleep 1
done
curl -s http://localhost:8080/health | jq

echo "🔥 Warm-up (to avoid first-call cold start)…"
for i in {1..3}; do
  curl -s -X POST "$PLANNER_URL/plan_run" \
    -H "content-type: application/json" \
    -d '{"prompt":"1) Write a tiny Python function.\n2) Explain it.\n3) Add 1 test.","max_new_tokens":80}' > /dev/null
  sleep 0.5
done

echo "🧪 Running evaluation with epsilon on: $PROMPT_FILE"
bash "$SCRIPT_DIR/eval_run.sh" "$PROMPT_FILE"

echo
echo "📈 Epsilon log file: $EPS_LOG"
echo "   tail -n 5 $EPS_LOG | jq -C"
echo
echo "📝 To switch back to PPO:"
echo "   docker compose -f docker-compose.yml up -d rl_router"
