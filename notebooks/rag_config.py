"""
Shared RAG setup — imported by ingest.py and query.ipynb.
All tuneable config lives at the top of this file.
"""
import sys, os
from functools import partial

# ── resolve project root (always repo root, regardless of cwd) ────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # notebooks/
project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # repo root
if project_root not in sys.path:
    sys.path.append(project_root)

# ── load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

# RAG CONFIGURATION (tweak here)
LLM_MODEL           = os.environ["LLM_MODEL"]
LLM_MAX_TOKENS      = 1024     # Match try.ipynb
LLM_TEMPERATURE     = 0.1
LLM_TOP_P           = 0.95

# Repetition penalties disabled (match try.ipynb)
FREQUENCY_PENALTY   = 0.0
PRESENCE_PENALTY    = 0.0

CHUNK_TOKEN_SIZE    = 1200     # Match try.ipynb
USE_CUSTOM_ENTITIES = False
USE_CUSTOM_PROMPTS  = False    # Switch back to LightRAG defaults (match try.ipynb)

DEBUG_OUTPUT_FILE = 'debug.log' #temp

# ── Custom Prompts ────────────────────────────────────────────────────────────

BIOMED_GRAPH_EXTRACTION_PROMPT = """---Role---
You are a Biomedical Knowledge Graph Specialist. Your task is to extract entities and concise relationships from medical text.

---Instructions---
1. **Entity Extraction**:
   - Identify medical entities (Anatomy, Disease, Gene, etc.).
   - Format: entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}concise_description

2. **Relationship Extraction (CRITICAL)**:
   - Identify direct relationships between entities.
   - **MAX LENGTH**: Relationship descriptions MUST be shorter than 20 words.
   - **NO PARAGRAPHS**: Do not summarize large sections into one relationship.
   - Format: relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}keywords{tuple_delimiter}concise_description

3. **Strict Delimiter Protocol**:
   - The {tuple_delimiter} is a literal atomic separator. Do not repeat it or use it within fields.
   - Output all entities first, then relationships.
   - End with {completion_delimiter}.

---Data to be Processed---
<Entity_types>
{entity_types}

<Output>
"""

# ── debug logging ─────────────────────────────────────────────────────────────
DEBUG_LLM         = True   # set False to disable prompt/response logging
DEBUG_LOG_FILE    = os.path.join(project_root, "debug_llm_output.txt")
# ─────────────────────────────────────────────────────────────────────────────

# ── from .env ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
LLM_BASE_URL    = os.environ.get("LLM_BASE_URL",  "http://127.0.0.1:8080/v1")
EMBED_BASE_URL   = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8081/v1")

# Use local storage inside notebooks/ (mirroring try.ipynb's behavior)
WORKING_DIR     = os.environ.get("RAG_WORKING_DIR", os.path.join(SCRIPT_DIR, "rag_storage"))

# ── lightrag imports ──────────────────────────────────────────────────────────
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, openai_complete

setup_logger("lightrag", level="INFO")

os.makedirs(WORKING_DIR, exist_ok=True)

# ── LLM completion function ───────────────────────────────────────────────────
async def llm_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    """OpenAI-compatible completion routed to the local vLLM server."""
    kwargs.update({
        "temperature":       LLM_TEMPERATURE,
        "top_p":             LLM_TOP_P,
        "max_tokens":        LLM_MAX_TOKENS,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty":  PRESENCE_PENALTY,
    })
    
    # Debug logging
    if DEBUG_LLM:
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write("\n=== PROMPT ===\n")
            f.write(f"SYSTEM: {system_prompt}\n")
            f.write(f"USER: {prompt}\n")
        
    result = await openai_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    if DEBUG_LLM:
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write("\n=== RESPONSE ===\n")
            f.write(result)
            f.write("\n" + "="*50 + "\n")
        
    return result

# ── RAG instance ──────────────────────────────────────────────────────────────

# Custom entity types if enabled
BIOMED_ENTITY_TYPES = ["Anatomy", "Biological_Process", "Cell", "Cellular_Component", "Chemical", "Disease", "Gene", "Organism", "Pathway", "Variant"]

def build_rag(working_dir=WORKING_DIR) -> LightRAG:
    """Builds and returns a LightRAG instance based on centralized config."""
    
    # Use custom entities if toggled
    addon_params = {}
    if USE_CUSTOM_ENTITIES:
        addon_params["entity_types"] = BIOMED_ENTITY_TYPES
    
    # Inject Custom Prompt if toggled
    if USE_CUSTOM_PROMPTS:
        import lightrag.prompt
        # Override the system prompt with our length-constrained version
        lightrag.prompt.PROMPTS["entity_extraction_system_prompt"] = BIOMED_GRAPH_EXTRACTION_PROMPT

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_complete,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=4,
        addon_params=addon_params,
        llm_model_kwargs={
            "base_url": LLM_BASE_URL,
            "api_key":  "none",
        },
        chunk_token_size=CHUNK_TOKEN_SIZE,
        entity_extract_max_gleaning=0,
        default_embedding_timeout=120,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                base_url=EMBED_BASE_URL,
                api_key="none",
                model=EMBEDDING_MODEL,
            ),
        ),
    )
