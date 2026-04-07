#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <prompts_file|all> [tag: code|explain|tests]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets"

run_one() {
  local prompts_file="$1"
  local tag="$2"
  local label="$3"
  local alpha="$4"
  local beta="$5"
  local gamma="$6"
  echo "============================================================"
  echo "[SWEEP][$label] alpha=$alpha beta=$beta gamma=$gamma tag=$tag"
  echo "============================================================"
  export ALPHA_QUALITY_OVERRIDE="$alpha"
  export BETA_LAT_OVERRIDE="$beta"
  export GAMMA_COST_OVERRIDE="$gamma"
  export LOG_LABEL="$label"
  "$SCRIPT_DIR/run_eval_ppo.sh" "$prompts_file" "$tag"
}

# Recommended small sensitivity set
# 1) paper default
# 2) slightly quality-favoring
# 3) slightly latency-favoring
# 4) balanced quality-favoring
CONFIGS=(
  "paper 1.0 1.0 0.001"
  "quality_up 1.5 1.0 0.001"
  "latency_up 1.0 1.25 0.001"
  "balanced_q 1.25 1.0 0.001"
)

if [[ "$1" == "all" ]]; then
  PROMPTS=("$DATA_DIR/code_prompts.txt" "$DATA_DIR/explain_prompts.txt" "$DATA_DIR/tests_prompts.txt")
  TAGS=("code" "explain" "tests")
  for idx in "${!PROMPTS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      read -r label alpha beta gamma <<< "$cfg"
      run_one "${PROMPTS[$idx]}" "${TAGS[$idx]}" "$label" "$alpha" "$beta" "$gamma"
    done
  done
else
  if [[ $# -ne 2 ]]; then
    echo "When not using all, provide both <prompts_file> and <tag>."
    exit 1
  fi
  for cfg in "${CONFIGS[@]}"; do
    read -r label alpha beta gamma <<< "$cfg"
    run_one "$1" "$2" "$label" "$alpha" "$beta" "$gamma"
  done
fi
