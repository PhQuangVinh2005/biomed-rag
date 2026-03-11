"""
LightRAG configuration for vLLM-served BioMistral + nomic-embed-text.

Servers are started by start_lightrag_vllm.sh:
  - LLM  : http://localhost:8080/v1  (BioMistral-7B-AWQ-QGS128-W4-GEMM)
  - Embed : http://localhost:8081/v1  (nomic-ai/nomic-embed-text-v1.5)
"""

import os

# ── Server endpoints (override via env vars if needed) ───────────────────────
LLM_BASE_URL   = os.getenv("LLM_BINDING_HOST",       "http://localhost:8080/v1")
EMBED_BASE_URL = os.getenv("EMBEDDING_BINDING_HOST",  "http://localhost:8081/v1")
LLM_API_KEY    = os.getenv("LLM_BINDING_API_KEY",    "none")
EMBED_API_KEY  = os.getenv("EMBEDDING_BINDING_API_KEY", "none")

LLM_MODEL   = os.getenv("LLM_MODEL",        "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL",  "nomic-ai/nomic-embed-text-v1.5")
EMBED_DIM   = int(os.getenv("EMBEDDING_DIM", "768"))
EMBED_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_TOKENS", "512"))

WORKING_DIR = os.getenv(
    "LIGHTRAG_WORKING_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "vectorstore", "lightrag_storage"),
)

# ── Async functions passed to LightRAG ───────────────────────────────────────
from lightrag.llm.openai import openai_complete_if_cache, openai_embed


async def llm_fn(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Async LLM completion via vLLM OpenAI-compatible API."""
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        **kwargs,
    )


async def embed_fn(texts):
    """Async embedding via vLLM OpenAI-compatible API."""
    return await openai_embed(
        texts,
        model=EMBED_MODEL,
        base_url=EMBED_BASE_URL,
        api_key=EMBED_API_KEY,
    )
