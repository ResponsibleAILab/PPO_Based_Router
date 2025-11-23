#!/usr/bin/env bash
# Purpose: Clean up Docker/WSL bind-mount state before an experiment run.
# Safe: does NOT remove the named `ollama_models` volume.
set -euo pipefail

# Project root (folder containing docker-compose*.yml)
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "🧹 Cleaning Docker compose stacks (down -v, remove orphans)…"
docker compose down -v --remove-orphans || true

echo "🧼 Pruning unused Docker resources (images/containers/networks, with volumes)…"
docker system prune -af --volumes || true

# WSL/Docker Desktop bind-mount cache fix (no-op on non-WSL)
WSL_MOUNTS="/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts"
if [[ -d "$WSL_MOUNTS" ]]; then
  echo "🧽 Clearing stale WSL bind-mount cache at: $WSL_MOUNTS"
  # Needs sudo because this path is owned by root inside WSL
  sudo rm -rf "${WSL_MOUNTS:?}/"* || true
fi

echo "✅ Base cleanup done."

# Optional: --hard flag to suggest/perform a Docker Desktop/WSL restart
if [[ "${1-}" == "--hard" ]]; then
  if command -v wsl.exe >/dev/null 2>&1; then
    echo "🔁 HARD mode: shutting down WSL (this will terminate your shell)…"
    wsl.exe --shutdown || true
    echo "ℹ️ Reopen your WSL terminal, then re-run your compose command."
  else
    echo "ℹ️ HARD mode: please restart Docker Desktop manually."
  fi
else
  echo "ℹ️ If you still hit mount errors, re-run with:  scripts/clean_environment.sh --hard"
fi
