import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt 
# import seaborn as sns
import torch
from functools import partial
from openai import AsyncOpenAI

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

from shared_functions.gg_sheet_drive import *
from prompt import *

import ast

# Read model config from environment
LLM_MODEL = os.environ["LLM_MODEL"]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8081/v1")

print(f"LLM_MODEL     : {LLM_MODEL}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"LLM_BASE_URL  : {LLM_BASE_URL}")
print(f"EMBED_BASE_URL : {EMBED_BASE_URL}")

import asyncio
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.hf import hf_embed
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed, openai_complete
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

class BioMed_RAG:
    def __init__(self):
        self.rag = None  # Lazy initialization

    async def llm_complete(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        """LLM completion using the model configured in .env (LLM_MODEL)."""
        kwargs.update({
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 1024,
        })

        return await openai_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )

    async def initialize_model(self):
        if self.rag is not None:
            return

        self.rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=self.llm_complete,
                llm_model_name=LLM_MODEL,
                llm_model_max_async=4,
                llm_model_kwargs={
                    "base_url": LLM_BASE_URL,
                    "api_key": "none"
                },
                chunk_token_size=1200,
                entity_extract_max_gleaning=0,
                default_embedding_timeout=120,
                
                embedding_func=EmbeddingFunc(
                    embedding_dim=768, 
                    max_token_size=8192,
                    func=partial(
                        openai_embed.func,
                        base_url=EMBED_BASE_URL,
                        api_key="none",
                        model=EMBEDDING_MODEL
                    )
                )
            )

        await self.rag.initialize_storages()

    async def call_llm(self, input):
        TEST_SYSTEM = SHORT_PROMPT
        llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="none")

        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": TEST_SYSTEM},
                {"role": "user",   "content": input},
            ],
            temperature=0.1,
            max_tokens=512,
        )

        return response

    async def biomedical_rag(self, user_query: str):
        """
        Combined Pipeline (Balanced Context):
        Ensures that text chunks are prioritized over entity definitions.
        """
        # Efficiently initialize only once
        if self.rag is None:
            await self.initialize_model()
        
        retrieval_result = await self.rag.aquery(
            user_query, 
            param=QueryParam(
                mode="hybrid", 
                only_need_context=True,
                max_total_tokens=10000,   
                max_entity_tokens=2000,  
                max_relation_tokens=2000,

                top_k=15,                
                chunk_top_k=8,           # Use more chunks
                enable_rerank=False
            )
        )
        
        context_text = retrieval_result if isinstance(retrieval_result, str) else str(retrieval_result)
        
        # Manual safety clip
        if len(context_text) > 15000:
            context_text = context_text[:15000] + "\n... [context truncated]"
        
        formatted_prompt = f"""Use the following retrieved data to answer the clinical case.
            --- RETRIEVED CONTEXT ---
            {context_text}
            --- END OF CONTEXT ---

            Question: {user_query}
            """

        response = await self.call_llm(formatted_prompt)

        if hasattr(response, 'choices'):
            return response.choices[0].message.content, retrieval_result
        
        return response, retrieval_result

