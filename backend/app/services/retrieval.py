from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class RetrievalService:
    def __init__(self, vector_store_dir: Path, embedding_model: str, top_k: int = 5):
        self.top_k = top_k
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            collection_name="rag_core",
            embedding_function=self.embeddings,
            persist_directory=str(vector_store_dir),
        )

    def retrieve(self, question: str, k: int | None = None):
        return self.vectorstore.similarity_search(
            question,
            k=k or self.top_k,
        )
