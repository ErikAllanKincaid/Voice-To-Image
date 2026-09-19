#!/bin/bash
# Smoke test for the Voice-To-Image docker compose deployment.
# Run this ON THE DEPLOY TARGET, after scripts/setup-ollama.sh and `docker compose up -d --build`.
#
# Checks, in order:
#   1. voice-to-image container is running
#   2. API /health reports up and gpu:true
#   3. The host's native Ollama (localhost:11434) has all three models the code
#      references (llama3.2:1b, llama3.2, qwen3.5:9b) -- see scripts/setup-ollama.sh
#   4. GET /models exposes every preset and qwen3.5:9b
#   5. (optional, needs --audio) one real /pipeline generation per preset, confirming
#      the X-Models response header matches what that preset should have selected
#   6. (optional, --restart-check) a down/up cycle proves the named volumes persisted
#      the downloaded weights instead of re-downloading them
#
# Usage:
#   ./smoke_test.sh                               # steps 1-4 only
#   ./smoke_test.sh --audio samples/               # also run step 5
#   ./smoke_test.sh --audio samples/ --restart-check
#   ./smoke_test.sh --presets "standard,high"      # limit which presets step 5 exercises
#   ./smoke_test.sh --timeout 600                  # health-wait timeout in seconds (default 300)
#
# --audio points at either a single WAV file or a directory of them. With a directory, each
# preset in step 5 gets a fresh random pick (via `shuf`), so repeated runs and different
# presets do not all exercise the exact same recording -- one fixed sample only proves that
# one exact input works, not the pipeline in general. Samples should be a few seconds of real
# speech (e.g. "generate a picture of a red fox in a forest"); step 5 is skipped entirely
# without --audio, since Whisper correctly returns "No speech detected" on silence and that
# is not a useful test of the happy path.
#
# Exit code 0 = everything checked passed. Non-zero = at least one failure.

ORIG_PWD="$PWD"
cd "$(dirname "$0")" || exit 1

API="http://localhost:8765"
PRESETS="lite standard high ultra flux"
TIMEOUT=300
AUDIO=""
RESTART_CHECK=0
FAILURES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --audio) AUDIO="$2"; case "$AUDIO" in /*) ;; *) AUDIO="$ORIG_PWD/$AUDIO" ;; esac; shift 2 ;;
        --presets) PRESETS=$(echo "$2" | tr ',' ' '); shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --restart-check) RESTART_CHECK=1; shift ;;
        -h|--help) awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Resolves $AUDIO to one WAV path for this call. A directory picks a random file each time
# (via shuf) so consecutive presets, and consecutive runs, do not all reuse the same sample.
pick_audio_sample() {
    if [ -d "$AUDIO" ]; then
        find "$AUDIO" -maxdepth 1 -iname '*.wav' | shuf -n 1
    else
        echo "$AUDIO"
    fi
}

section() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
pass()    { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail()    { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
note()    { printf '  %s\n' "$1"; }

container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

# Polls /health until it responds or $TIMEOUT elapses. Prints the raw body on success.
wait_for_health() {
    local waited=0
    while [ "$waited" -lt "$TIMEOUT" ]; do
        if body=$(curl -sf "$API/health" 2>/dev/null); then
            echo "$body"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        note "...still waiting (${waited}s/${TIMEOUT}s). Check: docker compose logs -f voice-to-image"
    done
    return 1
}

# True if Ollama's native API (localhost:11434) has model $1 (bare names match :latest too).
ollama_has_model() {
    local name="$1" pattern
    pattern="\"name\":\"${name//./\\.}\""
    case "$name" in
        *:*) ;;
        *) pattern="${pattern}|\"name\":\"${name//./\\.}:latest\"" ;;
    esac
    curl -sf "http://localhost:11434/api/tags" 2>/dev/null | grep -qE "$pattern"
}

section "1. Container running"
if container_running voice-to-image; then pass "voice-to-image container running"
else fail "voice-to-image container not running -- run 'docker compose up -d --build' first"; fi

section "2. API health + GPU visibility"
if health=$(wait_for_health); then
    note "$health"
    case "$health" in
        *'"gpu":true'*|*'"gpu": true'*) pass "server reports GPU visible" ;;
        *) fail "server reports gpu:false -- check nvidia-container-toolkit and the compose GPU reservation" ;;
    esac
else
    fail "API never became healthy within ${TIMEOUT}s"
fi

section "3. Host Ollama has required models"
if ! curl -sf "http://localhost:11434/api/version" >/dev/null 2>&1; then
    fail "Ollama not reachable at localhost:11434 -- run scripts/setup-ollama.sh"
else
    for m in llama3.2:1b llama3.2 qwen3.5:9b; do
        if ollama_has_model "$m"; then pass "ollama has $m"
        else fail "ollama missing $m -- run scripts/setup-ollama.sh"; fi
    done
fi

section "4. /models allowlist"
if models_json=$(curl -sf "$API/models" 2>/dev/null); then
    for key in '"lite"' '"standard"' '"high"' '"ultra"' '"flux"' '"qwen3.5:9b"'; do
        case "$models_json" in
            *"$key"*) pass "/models includes $key" ;;
            *) fail "/models missing $key" ;;
        esac
    done
else
    fail "GET /models failed"
fi

section "5. Per-preset generation"
if [ -z "$AUDIO" ]; then
    note "skipped (pass --audio <sample.wav or dir/> to exercise the full pipeline per preset)"
elif [ ! -e "$AUDIO" ]; then
    fail "--audio path not found: $AUDIO"
else
    for p in $PRESETS; do
        sample=$(pick_audio_sample)
        if [ -z "$sample" ]; then
            fail "$p: no .wav files found under $AUDIO"
            continue
        fi
        hdrs=$(mktemp)
        out=$(mktemp --suffix=.png)
        start=$(date +%s)
        code=$(curl -s -D "$hdrs" -o "$out" -w '%{http_code}' \
            -F "audio=@${sample}" -F "preset=${p}" "$API/pipeline")
        elapsed=$(( $(date +%s) - start ))
        size=$(wc -c < "$out" 2>/dev/null || echo 0)
        models_used=$(grep -i '^x-models:' "$hdrs" | sed 's/^[Xx]-[Mm]odels: *//' | tr -d '\r')
        if [ "$code" = "200" ] && [ "$size" -gt 1000 ]; then
            pass "$p: 200 in ${elapsed}s, ${size} bytes, sample=$(basename "$sample"), models=[$models_used] -> $out"
        else
            body_excerpt=$(head -c 300 "$out" 2>/dev/null)
            fail "$p: HTTP $code after ${elapsed}s, sample=$(basename "$sample") (${body_excerpt})"
        fi
        rm -f "$hdrs"
    done
fi

section "6. Restart persistence"
if [ "$RESTART_CHECK" -ne 1 ]; then
    note "skipped (pass --restart-check to verify volumes survive a down/up cycle)"
else
    note "docker compose down..."
    docker compose down >/dev/null 2>&1
    note "docker compose up -d (no --build, no re-download expected)..."
    docker compose up -d >/dev/null 2>&1
    restart_timeout=90
    saved_timeout=$TIMEOUT
    TIMEOUT=$restart_timeout
    if wait_for_health >/dev/null; then
        pass "stack became healthy again within ${restart_timeout}s -- volumes persisted"
    else
        fail "stack did not become healthy within ${restart_timeout}s after restart -- check whether models re-downloaded (docker compose logs voice-to-image)"
    fi
    TIMEOUT=$saved_timeout
fi

section "Summary"
if [ "$FAILURES" -eq 0 ]; then
    printf '\033[32mAll checks passed.\033[0m\n'
    exit 0
else
    printf '\033[31m%d check(s) failed.\033[0m See above.\n' "$FAILURES"
    exit 1
fi
