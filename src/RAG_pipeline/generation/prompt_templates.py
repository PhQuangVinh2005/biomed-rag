"""Simple prompt templates."""

def create_rag_prompt(query: str, context: list[str]) -> str:
    """Create a RAG prompt with query and context."""
    context_text = "\n\n".join(context)
    return f"""Answer the question based on the context below.

Context:
{context_text}

Question: {query}

Answer:"""
