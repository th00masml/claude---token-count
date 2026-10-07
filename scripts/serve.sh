#!/usr/bin/env bash
# Start a vLLM server for one configured model, wait for /health, run a command, stop it.
#
#   scripts/serve.sh <model> [--lora NAME=PATH ...] -- <command ...>
#   scripts/serve.sh qwen2.5-coder-7b -- uv run t2sbench run --stage stage3 --model qwen2.5-coder-7b
#
# Without a command the server stays up until Ctrl-C. Exit codes: 3 = server died while
# starting (e.g. not enough memory for the KV cache), 4 = /health timeout.
# Env: VLLM_BIN (default vllm), PORT (8000), HEALTH_TIMEOUT (1200 s), GPU_FREE_MB (1500),
#      T2SBENCH_CMD (default "uv run --quiet t2sbench").
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: serve.sh <model> [--lora NAME=PATH ...] -- <command>}"
shift
EXTRA=()
while [[ $# -gt 0 && "$1" != "--" ]]; do EXTRA+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift

PORT="${PORT:-8000}"
LOG="logs/vllm_${MODEL}.log"
mkdir -p logs
read -r -a T2S <<<"${T2SBENCH_CMD:-uv run --quiet t2sbench}"
ARGS_TXT="$("${T2S[@]}" vllm-args "$MODEL" --port "$PORT" "${EXTRA[@]}")"  # fails loudly under set -e
mapfile -t ARGS <<<"$ARGS_TXT"

echo "[serve] $(date -Is) starting ${VLLM_BIN:-vllm} ${ARGS[*]}" | tee -a logs/serve.log
"${VLLM_BIN:-vllm}" "${ARGS[@]}" >"$LOG" 2>&1 &
PID=$!

gpu_used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0; }

cleanup() {
  if kill -0 "$PID" 2>/dev/null; then
    # SIGTERM, not SIGINT: background jobs of a non-interactive shell ignore SIGINT
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$PID" 2>/dev/null || true
  fi
  wait "$PID" 2>/dev/null || true
  # free the GPU before the next model is loaded
  for _ in $(seq 1 180); do
    [[ "$(gpu_used_mb)" -lt "${GPU_FREE_MB:-1500}" ]] && break
    sleep 1
  done
  echo "[serve] $(date -Is) stopped $MODEL (GPU used: $(gpu_used_mb) MB)" | tee -a logs/serve.log
}
trap cleanup EXIT INT TERM

deadline=$(( $(date +%s) + ${HEALTH_TIMEOUT:-1200} ))
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[serve] vLLM for $MODEL died during startup; last log lines:" >&2
    tail -n 40 "$LOG" >&2
    exit 3
  fi
  if [[ $(date +%s) -gt $deadline ]]; then
    echo "[serve] /health timeout for $MODEL" >&2
    exit 4
  fi
  sleep 5
done
echo "[serve] $(date -Is) $MODEL healthy on :$PORT" | tee -a logs/serve.log

export T2S_VLLM_LOG="$LOG"
export T2S_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
if [[ $# -gt 0 ]]; then
  "$@"
else
  wait "$PID"
fi
