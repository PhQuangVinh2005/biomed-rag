"""Retrieve relevant documents."""

class Retriever:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve top-k relevant documents for a query."""
        query_embedding = self.embedding_model.embed([query])[0]
        return self.vector_store.search(query_embedding, k)
