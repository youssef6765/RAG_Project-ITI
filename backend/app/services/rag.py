from .generation import GenerationService
from .retrieval import RetrievalService


class RAGService:
    def __init__(
        self,
        vector_store_dir,
        embedding_model,
        llm_model,
        top_k=5,
    ):
        self.retrieval = RetrievalService(
            vector_store_dir=vector_store_dir,
            embedding_model=embedding_model,
            top_k=top_k,
        )
        self.generation = GenerationService(model_name=llm_model)

    def query(self, question: str):
        docs = self.retrieval.retrieve(question)
        answer, sources = self.generation.generate(question, docs)
        return answer, sources
