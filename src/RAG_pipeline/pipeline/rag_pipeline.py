"""Simple RAG pipeline."""
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.vector_db import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.llm import LLM
from src.generation.prompt_templates import create_rag_prompt


class RAGPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, self.embedding_model)
        self.llm = LLM()
    
    def add_documents(self, texts: list[str]):
        """Add documents to the knowledge base."""
        embeddings = self.embedding_model.embed(texts)
        self.vector_store.add(embeddings, texts)
    
    def query(self, question: str) -> str:
        """Answer a question using RAG."""
        context = self.retriever.retrieve(question)
        prompt = create_rag_prompt(question, context)
        return self.llm.generate(prompt)
