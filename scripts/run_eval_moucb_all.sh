#!/usr/bin/env bash
set -euo pipefail

export MAX_REQUESTS="${MAX_REQUESTS:-300}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets"
LOG_DIR="$ROOT_DIR/logs"
PROMPTS=( "code_prompts.txt" "explain_prompts.txt" "tests_prompts.txt" )
TAGS=(    "code"            "explain"             "tests"            )
mkdir -p "$LOG_DIR"
for idx in "${!PROMPTS[@]}"; do
  PROMPT_FILE="$DATA_DIR/${PROMPTS[$idx]}"
  TAG="${TAGS[$idx]}"
  "$SCRIPT_DIR/run_eval_moucb.sh" "$PROMPT_FILE" "$TAG"
done
python3 "$SCRIPT_DIR/analyze_results.py" --auto "$LOG_DIR" --outdir "$ROOT_DIR/analysis_moucb"
