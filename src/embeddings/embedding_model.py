"""Generate embeddings for text."""
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: list[str]) -> list:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts)
