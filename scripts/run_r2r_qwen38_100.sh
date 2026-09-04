#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-17892}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${VLLM_HOST},localhost"
export no_proxy="$NO_PROXY"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if ! curl -fsS --max-time "${VLLM_PREFLIGHT_TIMEOUT:-5}" \
  "${OPENAI_BASE_URL%/}/models" >/dev/null; then
  printf 'ERROR: vLLM API is unavailable at %s\n' "$OPENAI_BASE_URL" >&2
  exit 1
fi

cd "$ROOT"
exec python -m cli run --runner config/runners/r2r_qwen38_100.yaml "$@"
