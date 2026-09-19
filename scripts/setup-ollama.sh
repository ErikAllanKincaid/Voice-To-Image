#!/bin/bash
# Ollama host prerequisite for Voice-To-Image, Docker or native.
#
# Idempotent: makes no assumption about what is already on the machine. Safe to run
# repeatedly -- each step checks current state first and skips work already done.
#
# What it does, in order:
#   1. Installs Ollama natively if the `ollama` command is not already present.
#   2. Waits for the Ollama API to answer on localhost:11434 (systemd starts it after
#      install; if it was already installed and already running, this returns instantly).
#   3. Pulls every model server.py's presets reference: llama3.2:1b, llama3.2, qwen3.5:9b.
#      `ollama pull` is itself idempotent -- it checks the local manifest and does nothing
#      if the model is already present, so re-running this script re-downloads nothing.
#
# Run this once before `docker compose up -d`, or before running server.py natively.
# Exit code 0 = Ollama is installed, running, and has all required models.

set -euo pipefail

API="http://localhost:11434"
MODELS="llama3.2:1b llama3.2 qwen3.5:9b"
TIMEOUT=60

section() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

section "1. Ollama binary"
if command -v ollama >/dev/null 2>&1; then
    echo "Already installed: $(ollama --version 2>&1 | head -1)"
else
    echo "Not found -- installing via https://ollama.com/install.sh"
    curl -fsSL https://ollama.com/install.sh | sh
fi

section "2. Ollama API reachable"
waited=0
until curl -sf "$API/api/version" >/dev/null 2>&1; do
    if [ "$waited" -ge "$TIMEOUT" ]; then
        echo "Ollama did not come up on $API within ${TIMEOUT}s." >&2
        echo "If it is installed but not running: sudo systemctl start ollama" >&2
        exit 1
    fi
    sleep 2
    waited=$((waited + 2))
done
echo "Reachable at $API ($(curl -sf "$API/api/version"))"

section "3. Required models"
have_model() {
    curl -sf "$API/api/tags" | grep -qE "\"name\":\"${1//./\\.}(:latest)?\""
}
for m in $MODELS; do
    if have_model "$m"; then
        echo "Already present: $m"
    else
        echo "Pulling: $m"
        ollama pull "$m"
    fi
done

section "Done"
echo "Ollama is installed, running, and has: $MODELS"
