# %%
# %%
import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt 
# import seaborn as sns
import torch
from functools import partial
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.hf import hf_embed
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed, openai_complete
from lightrag.utils import setup_logger

# %%
setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# %%
async def hf_model_complete(prompt: str, system_prompt=None, history_messages=[], **kwargs):
    device = model.device
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Use chat template to wrap the complex LightRAG instructions
    tokenized_chat = llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = {"input_ids": tokenized_chat.to(device)}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )

    decoded = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return decoded.strip()

# %%
async def bio_mistral_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    # Update kwargs to be more strict for extraction
    kwargs.update({
        "temperature": 0.1,      # Be deterministic
        "top_p": 0.95,
        "max_tokens": 1024,      # Ensure it doesn't cut off mid-list
    })

    if system_prompt:
        # Prepend system prompt and emphasize the separator format
        prompt = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            f"IMPORTANT: You MUST use the exact separator <SEP> between fields and "
            f"END the response with the ############################# delimiter.\n\n"
            f"USER INPUT:\n{prompt}"
        )
        system_prompt = None
        
    return await openai_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

# %%
# Using vLLM

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=bio_mistral_complete,
    llm_model_name="BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM",
    llm_model_max_async=4,
    llm_model_kwargs={
        "base_url": "http://127.0.0.1:8080/v1", 
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
            base_url="http://127.0.0.1:8081/v1",  # Updated port
            api_key="none",
            model="nomic-ai/nomic-embed-text-v1.5"
        )
    )
)

# %%
async def main():
    await rag.initialize_storages()

    # %%
    # data_dir = os.path.join(project_root, 'biomed-rag', 'data', 'external', 'medqa')
    # Use path relative to the script for robustness
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'external', 'medqa')
    sample_textbook = os.path.join(data_dir, 'textbooks', 'Anatomy_Gray.txt')

    with open(sample_textbook, 'r') as f:
        text = f.read()

    # %%
    await rag.ainsert(text)

if __name__ == "__main__":
    asyncio.run(main())


