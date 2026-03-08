#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

# vLLM LLM server
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_LLM_PORT="${VLLM_LLM_PORT:-8080}"
LLM_MODEL="${LLM_MODEL:-BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

# vLLM embedding server
START_EMBED_SERVER="${START_EMBED_SERVER:-1}"
VLLM_EMBED_PORT="${VLLM_EMBED_PORT:-8081}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-ai/nomic-embed-text-v1.5}"

# LightRAG server
LIGHTRAG_HOST="${LIGHTRAG_HOST:-0.0.0.0}"
LIGHTRAG_PORT="${LIGHTRAG_PORT:-9621}"
LIGHTRAG_WORKING_DIR="${LIGHTRAG_WORKING_DIR:-$REPO_ROOT/rag_storage}"
LIGHTRAG_INPUT_DIR="${LIGHTRAG_INPUT_DIR:-$REPO_ROOT/inputs}"

mkdir -p "$LOG_DIR" "$LIGHTRAG_WORKING_DIR" "$LIGHTRAG_INPUT_DIR"
cd "$REPO_ROOT"

# LightRAG asks for .env in startup directory for multi-instance support.
if [[ ! -f ".env" ]]; then
  touch .env
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

wait_for_http() {
  local url="$1"
  local name="$2"
  local retries="${3:-120}"
  local sleep_sec="${4:-1}"
  local i

  for ((i = 1; i <= retries; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep "$sleep_sec"
  done

  echo "Timed out waiting for $name at $url" >&2
  return 1
}

cleanup() {
  set +e
  echo
  echo "Stopping services..."

  if [[ -n "${LIGHTRAG_PID:-}" ]] && kill -0 "$LIGHTRAG_PID" >/dev/null 2>&1; then
    kill "$LIGHTRAG_PID" >/dev/null 2>&1 || true
    wait "$LIGHTRAG_PID" 2>/dev/null || true
  fi

  if [[ -n "${EMBED_PID:-}" ]] && kill -0 "$EMBED_PID" >/dev/null 2>&1; then
    kill "$EMBED_PID" >/dev/null 2>&1 || true
    wait "$EMBED_PID" 2>/dev/null || true
  fi

  if [[ -n "${LLM_PID:-}" ]] && kill -0 "$LLM_PID" >/dev/null 2>&1; then
    kill "$LLM_PID" >/dev/null 2>&1 || true
    wait "$LLM_PID" 2>/dev/null || true
  fi

  echo "All services stopped."
}

trap cleanup EXIT INT TERM

require_cmd python
require_cmd curl
require_cmd lightrag-server

LLM_LOG="$LOG_DIR/vllm_llm.log"
EMBED_LOG="$LOG_DIR/vllm_embed.log"
LIGHTRAG_LOG="$LOG_DIR/lightrag.log"

echo "Starting vLLM LLM server on port $VLLM_LLM_PORT ..."
python -m vllm.entrypoints.openai.api_server \
  --model "$LLM_MODEL" \
  --quantization "$VLLM_QUANTIZATION" \
  --host "$VLLM_HOST" \
  --port "$VLLM_LLM_PORT" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  >"$LLM_LOG" 2>&1 &
LLM_PID=$!

if [[ "$START_EMBED_SERVER" == "1" ]]; then
  echo "Starting vLLM embedding server on port $VLLM_EMBED_PORT ..."
  python -m vllm.entrypoints.openai.api_server \
    --model "$EMBEDDING_MODEL" \
    --task embedding \
    --host "$VLLM_HOST" \
    --port "$VLLM_EMBED_PORT" \
    >"$EMBED_LOG" 2>&1 &
  EMBED_PID=$!
fi

wait_for_http "http://127.0.0.1:${VLLM_LLM_PORT}/v1/models" "vLLM LLM"
if [[ "$START_EMBED_SERVER" == "1" ]]; then
  wait_for_http "http://127.0.0.1:${VLLM_EMBED_PORT}/v1/models" "vLLM embedding"
fi

export LLM_BINDING="openai"
export EMBEDDING_BINDING="openai"
export LLM_BINDING_HOST="http://127.0.0.1:${VLLM_LLM_PORT}/v1"
export LLM_BINDING_API_KEY="none"
export LLM_MODEL="$LLM_MODEL"

if [[ "$START_EMBED_SERVER" == "1" ]]; then
  export EMBEDDING_BINDING_HOST="http://127.0.0.1:${VLLM_EMBED_PORT}/v1"
  export EMBEDDING_BINDING_API_KEY="none"
  export EMBEDDING_MODEL="$EMBEDDING_MODEL"
fi

echo "Starting LightRAG server on port $LIGHTRAG_PORT ..."
lightrag-server \
  --host "$LIGHTRAG_HOST" \
  --port "$LIGHTRAG_PORT" \
  --working-dir "$LIGHTRAG_WORKING_DIR" \
  --input-dir "$LIGHTRAG_INPUT_DIR" \
  --llm-binding openai \
  --embedding-binding openai \
  >"$LIGHTRAG_LOG" 2>&1 &
LIGHTRAG_PID=$!

wait_for_http "http://127.0.0.1:${LIGHTRAG_PORT}/docs" "LightRAG"

echo
echo "Services are up:"
echo "- vLLM LLM:      http://127.0.0.1:${VLLM_LLM_PORT}/v1"
if [[ "$START_EMBED_SERVER" == "1" ]]; then
  echo "- vLLM Embedding: http://127.0.0.1:${VLLM_EMBED_PORT}/v1"
fi
echo "- LightRAG API:   http://127.0.0.1:${LIGHTRAG_PORT}/docs"
echo
echo "Logs:"
echo "- $LLM_LOG"
if [[ "$START_EMBED_SERVER" == "1" ]]; then
  echo "- $EMBED_LOG"
fi
echo "- $LIGHTRAG_LOG"
echo
echo "Press Ctrl+C to stop all services."

wait "$LIGHTRAG_PID"
