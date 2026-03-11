#!/usr/bin/env python3
"""Start a vLLM LLM server with configurable defaults and CLI overrides."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR
DEFAULT_LOG_DIR = DEFAULT_REPO_ROOT / "logs"
DEFAULT_LOG_FILE = "vllm_llm.log"

DEFAULT_VLLM_HOST = "0.0.0.0"
DEFAULT_VLLM_LLM_PORT = 8080
DEFAULT_LLM_MODEL = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
DEFAULT_VLLM_QUANTIZATION = "awq_marlin"
DEFAULT_VLLM_MAX_MODEL_LEN = 4096
DEFAULT_VLLM_GPU_MEM_UTIL = 0.70
DEFAULT_LLM_DEVICE = "gpu"
DEFAULT_CPU_DTYPE = "half"
DEFAULT_TRUST_REMOTE_CODE = True


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


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
        description="Start vLLM OpenAI-compatible LLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--repo-root", default=None, help="Repository root working directory")
    parser.add_argument("--log-dir", default=None, help="Directory for log files")
    parser.add_argument("--log-file", default=None, help="Log file name")

    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="LLM server port")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--quantization", default=None, help="Quantization mode for GPU")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum model length")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory utilization fraction for vLLM when device=gpu",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default=None,
        help="Device for LLM server",
    )
    parser.add_argument(
        "--cpu-dtype",
        default=None,
        help="Dtype passed to vLLM when device=cpu",
    )

    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Enable --trust-remote-code",
    )
    trust_group.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable --trust-remote-code",
    )
    parser.set_defaults(trust_remote_code=None)

    return parser


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
    log_file = get_config_value(args.log_file, "LLM_LOG_FILE", DEFAULT_LOG_FILE, str)

    host = get_config_value(args.host, "VLLM_HOST", DEFAULT_VLLM_HOST, str)
    port = get_config_value(args.port, "VLLM_LLM_PORT", DEFAULT_VLLM_LLM_PORT, int)
    model = get_config_value(args.model, "LLM_MODEL", DEFAULT_LLM_MODEL, str)
    quantization = get_config_value(
        args.quantization,
        "VLLM_QUANTIZATION",
        DEFAULT_VLLM_QUANTIZATION,
        str,
    )
    max_model_len = get_config_value(
        args.max_model_len,
        "VLLM_MAX_MODEL_LEN",
        DEFAULT_VLLM_MAX_MODEL_LEN,
        int,
    )
    gpu_mem_util = get_config_value(
        args.gpu_memory_utilization,
        "VLLM_GPU_MEM_UTIL",
        DEFAULT_VLLM_GPU_MEM_UTIL,
        float,
    )
    device = get_config_value(args.device, "LLM_DEVICE", DEFAULT_LLM_DEVICE, str).lower()
    cpu_dtype = get_config_value(args.cpu_dtype, "CPU_DTYPE", DEFAULT_CPU_DTYPE, str)
    trust_remote_code = get_config_value(
        args.trust_remote_code,
        "TRUST_REMOTE_CODE",
        DEFAULT_TRUST_REMOTE_CODE,
        parse_bool,
    )

    if device not in {"gpu", "cpu"}:
        raise ValueError(f"Invalid device: {device}. Expected 'gpu' or 'cpu'.")

    if platform.system() == "Darwin" and device == "gpu":
        print(
            "WARNING: macOS detected with LLM_DEVICE=gpu. "
            "This launcher is configured for CUDA-style GPU execution and AWQ quantization. "
            "Falling back to CPU mode automatically. "
            "Set LLM_DEVICE=cpu explicitly to suppress this warning."
        )
        device = "cpu"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
    ]

    if device == "cpu":
        cmd.extend(["--dtype", cpu_dtype])
    else:
        cmd.extend(
            [
                "--quantization",
                quantization,
                "--gpu-memory-utilization",
                str(gpu_mem_util),
            ]
        )

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    child_env = os.environ.copy()
    if device == "cpu":
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        if platform.system() == "Darwin":
            child_env["VLLM_PLUGINS"] = ""

    print(f"Starting vLLM LLM server on port {port} (device: {device}) ...")
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