"""Simple vector store using FAISS."""
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
    
    def add(self, embeddings: list, texts: list[str]):
        """Add embeddings and texts to the store."""
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.texts.extend(texts)
    
    def search(self, query_embedding: list, k: int = 5) -> list[str]:
        """Search for similar texts."""
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, k)
        return [self.texts[i] for i in indices[0]]
