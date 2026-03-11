#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${VLLM_PYTHON_BIN:-${PYTHON_BIN:-}}"
if [[ -z "$PYTHON_BIN" ]] && [[ -x "$HOME/.venv-vllm-metal/bin/python" ]]; then
	PYTHON_BIN="$HOME/.venv-vllm-metal/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
	PYTHON_BIN="$(command -v python3 || command -v python)"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/start_lightrag_server.py" "$@"
