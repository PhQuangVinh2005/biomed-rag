"""Simple RAG example."""
from src.pipeline.rag_pipeline import RAGPipeline
from src.chunking.text_splitter import split_text
from src.ingestion.document_loader import load_text_file


def main():
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Load and add documents
    # text = load_text_file("data/raw/example.txt")
    # chunks = split_text(text)
    # rag.add_documents(chunks)
    
    # Query the system
    # answer = rag.query("What is...?")
    # print(answer)
    
    print("RAG pipeline ready!")


if __name__ == "__main__":
    main()
