#!/usr/bin/env bash
set -euo pipefail

# Default cap per run; can be overridden by exporting MAX_REQUESTS before calling
export MAX_REQUESTS="${MAX_REQUESTS:-300}"
export ALPHA_QUALITY_OVERRIDE="${ALPHA_QUALITY_OVERRIDE:-${ALPHA_QUALITY:-1.0}}"
export BETA_LAT_OVERRIDE="${BETA_LAT_OVERRIDE:-${BETA_LAT:-1.0}}"
export GAMMA_COST_OVERRIDE="${GAMMA_COST_OVERRIDE:-${GAMMA_COST:-0.001}}"

# Root of the project (one level above scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets"
LOG_DIR="$ROOT_DIR/logs"

# Datasets and tags in parallel arrays
PROMPTS=( "code_prompts.txt" "explain_prompts.txt" "tests_prompts.txt" )
TAGS=(    "code"            "explain"             "tests"            )

# Only run these three methods – PPO/epsilon handled elsewhere
METHODS=( "epsilon" "ppo" "softmax" "ucb" "thompson" "moucb" )
#METHODS=( "moucb" )

# Name of the Ollama container (from docker ps)
OLLAMA_CONTAINER="rl_mcp_project-ollama-1"

mkdir -p "$LOG_DIR"

echo "Project root: $ROOT_DIR"
echo "Datasets dir: $DATA_DIR"
echo "Methods: ${METHODS[*]}"
echo "Datasets: ${PROMPTS[*]}"
echo "Reward weights: alpha=${ALPHA_QUALITY_OVERRIDE} beta=${BETA_LAT_OVERRIDE} gamma=${GAMMA_COST_OVERRIDE}"
echo

check_router_policy() {
  local expected="$1"
  local got
  got="$(curl -s http://localhost:8080/health | sed -n 's/.*"policy"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  echo "🔎 Router policy reports: ${got:-unknown} (expected: $expected)"
  if [[ "$got" != "$expected" ]]; then
    echo "❌ Router policy mismatch. Expected '$expected' but got '${got:-unknown}'."
    exit 1
  fi
}

cleanup_route_log() {
  local method="$1"
  local log_file="$LOG_DIR/route_log_${method}.jsonl"

  if [[ -f "$log_file" ]]; then
    echo "🧹 Removing existing router log for ${method}: $log_file"
    rm -f "$log_file"
  else
    echo "🧹 No existing router log for ${method} to remove: $log_file"
  fi
}

wait_for_ollama_health() {
  local max_tries=60
  local delay=5

  echo "⏳ Waiting for Ollama container health: $OLLAMA_CONTAINER"

  for i in $(seq 1 "$max_tries"); do
    if docker inspect "$OLLAMA_CONTAINER" &>/dev/null; then
      local status
      status="$(docker inspect -f '{{.State.Health.Status}}' "$OLLAMA_CONTAINER" 2>/dev/null || echo "unknown")"
      echo "   Attempt $i/$max_tries: health=$status"

      if [[ "$status" == "healthy" ]]; then
        echo "✅ Ollama container is healthy."
        return 0
      fi
    else
      echo "   Attempt $i/$max_tries: container not found yet."
    fi
    sleep "$delay"
  done

  echo "⚠️  Ollama container did not report healthy in time; continuing anyway."
  return 1
}

wait_for_llms_ready() {
  local max_tries=60
  local delay=300

  echo "⏳ Verifying that at least 3 LLMs are loaded in $OLLAMA_CONTAINER…"

  for i in $(seq 1 "$max_tries"); do
    if docker exec "$OLLAMA_CONTAINER" ollama list >/tmp/ollama_list 2>/dev/null; then
      # Count models (skip header line)
      local count
      count=$(docker exec "$OLLAMA_CONTAINER" ollama list 2>/dev/null | tail -n +2 | wc -l || echo 0)

      echo "   Attempt $i/$max_tries: found $count models"

      if (( count >= 3 )); then
        echo "✅ Detected $count models in Ollama — assuming 3+ LLMs are ready."
        return 0
      fi
    else
      echo "   Attempt $i/$max_tries: ollama list not ready yet."
    fi
    sleep "$delay"
  done

  echo "⚠️  LLMs did not report ready in time; continuing anyway."
  return 1
}

for METHOD in "${METHODS[@]}"; do
  echo "============================================================"
  echo "🏁 Starting method: $METHOD"
  echo "============================================================"

  # Method-specific eval script
  EVAL_SCRIPT="$SCRIPT_DIR/run_eval_${METHOD}.sh"
  if [[ ! -x "$EVAL_SCRIPT" ]]; then
    echo "❌ Eval script not found or not executable: $EVAL_SCRIPT"
    exit 1
  fi

  OVERRIDE_FILE="$ROOT_DIR/docker-compose.${METHOD}.yml"
  if [[ ! -f "$OVERRIDE_FILE" ]]; then
    echo "❌ Missing override file: $OVERRIDE_FILE"
    exit 1
  fi

  # Loop over each dataset/tag
  for idx in "${!PROMPTS[@]}"; do
    PROMPT_FILE="$DATA_DIR/${PROMPTS[$idx]}"
    TAG="${TAGS[$idx]}"

    if [[ ! -f "$PROMPT_FILE" ]]; then
      echo "❌ Prompt file missing: $PROMPT_FILE"
      exit 1
    fi

    echo
    echo "========================================================"
    echo "🚀 Method: $METHOD   Dataset: $PROMPT_FILE   Tag: $TAG"
    echo "========================================================"

    cd "$ROOT_DIR"

    # 1) Ensure fresh main router log for this method
    cleanup_route_log "$METHOD"

    # 2) Tear everything down (containers + volumes + LLM weights/caches)
    echo "🔻 docker compose down -v (clearing containers + volumes)…"
    docker compose down -v || true

    # 3) Bring stack back up fresh with this method's override
    echo "🔺 docker compose up -d with override: docker-compose.${METHOD}.yml"
    docker compose -f docker-compose.yml -f "docker-compose.${METHOD}.yml" up -d --build

    # 4) Wait for Ollama health + verify that 3+ LLMs are present
    wait_for_ollama_health
    wait_for_llms_ready

    # (Optional: small grace pause)
    sleep 5
    check_router_policy "$METHOD"

    # 5) Run the evaluation for this method/dataset pair
    echo "🧪 Running eval: $METHOD on $PROMPT_FILE (tag=$TAG)"
    "$EVAL_SCRIPT" "$PROMPT_FILE" "$TAG"

    echo "✅ Finished method=$METHOD, dataset=$PROMPT_FILE (tag=$TAG)"
    echo
  done

  echo "🎯 Completed all datasets for method: $METHOD"
  echo
done

echo "🎉 All methods completed across all datasets."
