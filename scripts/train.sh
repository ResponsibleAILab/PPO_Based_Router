#!/usr/bin/env bash
set -euo pipefail

ROUNDS="${1:-500}"        # how many composite prompts to send
SAVE_EVERY="${2:-100}"    # call /policy/save every N prompts
HOST="${HOST:-http://localhost}"

prompts=(
  $'1) Write a Python function to reverse a list.\n2) Explain how it works.\n3) Create 2 unit tests.'
  $'1) Implement factorial(n).\n2) Explain iterative vs recursive.\n3) Add 2 unit tests.'
  $'1) Implement fizzbuzz.\n2) Summarize time complexity.\n3) Add tests for multiples of 3 and 5.'
  $'1) Implement is_prime.\n2) Explain your algorithm.\n3) Add tests for edge cases.'
  $'1) Implement quicksort.\n2) Explain best/worst case.\n3) Add 3 unit tests.'
  $'1) Write a function to merge two sorted lists.\n2) Explain it simply.\n3) Add two tests.'
  $'1) Implement binary_search.\n2) Explain invariants.\n3) Provide 2 tests.'
)

echo "Training PPO by driving planner at ${HOST}:9000 for ${ROUNDS} prompts…"
for ((i=1; i<=ROUNDS; i++)); do
  p="${prompts[$RANDOM % ${#prompts[@]}]}"
  curl -s -X POST "${HOST}:9000/plan_run" \
    -H "content-type: application/json" \
    -d "$(jq -n --arg prompt "$p" --argjson max 160 '{prompt:$prompt, max_new_tokens:$max}')" >/dev/null || true

  # occasionally save PPO
  if (( i % SAVE_EVERY == 0 )); then
    echo "[${i}] saving PPO checkpoint…"
    curl -s -X POST "${HOST}:8080/policy/save" >/dev/null || true
  fi
done

echo "Final save…"
curl -s -X POST "${HOST}:8080/policy/save" >/dev/null || true
echo "Done. Check ./checkpoints/ppo.pt and logs/route_log.jsonl"
