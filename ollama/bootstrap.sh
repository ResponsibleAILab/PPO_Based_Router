#!/bin/sh
set -eu

# 1) Start Ollama server in the background
/bin/ollama serve &
OLLAMA_PID=$!

# 2) Wait until the server responds to CLI
echo "Waiting for Ollama to be ready ..."
for i in $(seq 1 60); do
  if /bin/ollama list >/dev/null 2>&1; then
    echo "Ollama is ready."
    break
  fi
  sleep 2
done

# 3) Auto-pull models (CLI is simplest inside the official image)
MODELS="${OLLAMA_MODELS:-mistral:7b-instruct,llama3:8b,codellama:7b-instruct}"
echo "Auto-pulling models: $MODELS"

IFS=,
for m in $MODELS; do
  echo "Pulling $m ..."
  /bin/ollama pull "$m" || true
done
unset IFS

echo "All models pulled. Handing over to Ollama (PID=$OLLAMA_PID)."
wait "$OLLAMA_PID"
