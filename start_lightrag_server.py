#!/usr/bin/env python3
"""Start a LightRAG server with configurable defaults and CLI overrides."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR
DEFAULT_LOG_DIR = DEFAULT_REPO_ROOT / "logs"
DEFAULT_LOG_FILE = "lightrag.log"

DEFAULT_LIGHTRAG_HOST = "0.0.0.0"
DEFAULT_LIGHTRAG_PORT = 9621
DEFAULT_LIGHTRAG_WORKING_DIR = DEFAULT_REPO_ROOT / "rag_storage"
DEFAULT_LIGHTRAG_INPUT_DIR = DEFAULT_REPO_ROOT / "inputs"
DEFAULT_VLLM_LLM_PORT = 8080
DEFAULT_VLLM_EMBED_PORT = 8081
DEFAULT_LLM_MODEL = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_WAIT_RETRIES = 120
DEFAULT_WAIT_SLEEP_SECONDS = 1.0


def get_config_value(
    arg_value: Any,
    env_name: str,
    default: Any,
    caster: Callable[[str], Any] | None = None,
) -> Any:
    if arg_value is not None:
        return arg_value

    raw = os.getenv(env_name)
    if raw is None:
        return default

    if caster is None:
        return raw

    return caster(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start LightRAG server after waiting for both vLLM endpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--repo-root", default=None, help="Repository root working directory")
    parser.add_argument("--log-dir", default=None, help="Directory for log files")
    parser.add_argument("--log-file", default=None, help="Log file name")

    parser.add_argument("--host", default=None, help="LightRAG server host")
    parser.add_argument("--port", type=int, default=None, help="LightRAG server port")
    parser.add_argument("--working-dir", default=None, help="LightRAG working directory")
    parser.add_argument("--input-dir", default=None, help="LightRAG input directory")
    parser.add_argument("--llm-port", type=int, default=None, help="vLLM LLM server port")
    parser.add_argument("--embed-port", type=int, default=None, help="vLLM embedding server port")
    parser.add_argument("--llm-model", default=None, help="LLM model name exposed to LightRAG")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name exposed to LightRAG",
    )
    parser.add_argument(
        "--wait-retries",
        type=int,
        default=None,
        help="Maximum HTTP readiness polling attempts per server",
    )
    parser.add_argument(
        "--wait-sleep-seconds",
        type=float,
        default=None,
        help="Seconds to sleep between readiness checks",
    )

    return parser


def require_command(name: str) -> str:
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)

    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"Missing required command: {name}")
    return resolved


def wait_for_http(url: str, name: str, retries: int, sleep_seconds: float) -> None:
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(f"{name} is ready: {url}")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(sleep_seconds)

    raise SystemExit(f"Timed out waiting for {name} at {url}")


def stream_process_to_console_and_file(process: subprocess.Popen[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log_fp:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_fp.write(line)

    return process.wait()


def main() -> int:
    args = build_parser().parse_args()

    repo_root = Path(
        get_config_value(args.repo_root, "REPO_ROOT", str(DEFAULT_REPO_ROOT), str)
    ).resolve()
    log_dir = Path(get_config_value(args.log_dir, "LOG_DIR", str(DEFAULT_LOG_DIR), str)).resolve()
    log_file = get_config_value(args.log_file, "LIGHTRAG_LOG_FILE", DEFAULT_LOG_FILE, str)

    host = get_config_value(args.host, "LIGHTRAG_HOST", DEFAULT_LIGHTRAG_HOST, str)
    port = get_config_value(args.port, "LIGHTRAG_PORT", DEFAULT_LIGHTRAG_PORT, int)
    working_dir = Path(
        get_config_value(args.working_dir, "LIGHTRAG_WORKING_DIR", str(DEFAULT_LIGHTRAG_WORKING_DIR), str)
    ).resolve()
    input_dir = Path(
        get_config_value(args.input_dir, "LIGHTRAG_INPUT_DIR", str(DEFAULT_LIGHTRAG_INPUT_DIR), str)
    ).resolve()
    llm_port = get_config_value(args.llm_port, "VLLM_LLM_PORT", DEFAULT_VLLM_LLM_PORT, int)
    embed_port = get_config_value(args.embed_port, "VLLM_EMBED_PORT", DEFAULT_VLLM_EMBED_PORT, int)
    llm_model = get_config_value(args.llm_model, "LLM_MODEL", DEFAULT_LLM_MODEL, str)
    embedding_model = get_config_value(
        args.embedding_model,
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
        str,
    )
    wait_retries = get_config_value(
        args.wait_retries,
        "LIGHTRAG_WAIT_RETRIES",
        DEFAULT_WAIT_RETRIES,
        int,
    )
    wait_sleep_seconds = get_config_value(
        args.wait_sleep_seconds,
        "LIGHTRAG_WAIT_SLEEP_SECONDS",
        DEFAULT_WAIT_SLEEP_SECONDS,
        float,
    )

    lightrag_server_path = require_command("lightrag-server")

    log_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    repo_root.joinpath(".env").touch(exist_ok=True)
    log_path = log_dir / log_file

    llm_models_url = f"http://127.0.0.1:{llm_port}/v1/models"
    embed_models_url = f"http://127.0.0.1:{embed_port}/v1/models"

    print("Waiting for vLLM servers to be ready...")
    wait_for_http(llm_models_url, "vLLM LLM", wait_retries, wait_sleep_seconds)
    wait_for_http(embed_models_url, "vLLM embedding", wait_retries, wait_sleep_seconds)

    child_env = os.environ.copy()
    child_env.update(
        {
            "LLM_BINDING": "openai",
            "EMBEDDING_BINDING": "openai",
            "LLM_BINDING_HOST": f"http://127.0.0.1:{llm_port}/v1",
            "LLM_BINDING_API_KEY": "none",
            "LLM_MODEL": llm_model,
            "EMBEDDING_BINDING_HOST": f"http://127.0.0.1:{embed_port}/v1",
            "EMBEDDING_BINDING_API_KEY": "none",
            "EMBEDDING_MODEL": embedding_model,
        }
    )

    cmd = [
        lightrag_server_path,
        "--host",
        host,
        "--port",
        str(port),
        "--working-dir",
        str(working_dir),
        "--input-dir",
        str(input_dir),
        "--llm-binding",
        "openai",
        "--embedding-binding",
        "openai",
    ]

    print(f"Starting LightRAG server on port {port} ...")
    print(f"Log file: {log_path}")

    process = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        return stream_process_to_console_and_file(process, log_path)
    except KeyboardInterrupt:
        process.terminate()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())