import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_ollama import OllamaEmbeddings


class RetrievalService:
    def __init__(self, vector_store_dir: Path, embedding_model: str, top_k: int = 5):
        self.top_k = top_k
        vector_store_dir = Path(vector_store_dir)

        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            collection_name="multi_modal_rag_v2",                    # must match the notebook
            embedding_function=self.embeddings,
            persist_directory=str(vector_store_dir / "chroma"),   # notebook's chroma subfolder
        )

        count = self.vectorstore._collection.count()
        print(f"[DEBUG] Loaded Chroma collection 'multi_modal_rag_v2' with {count} embeddings")
        if count == 0:
            raise RuntimeError(
                f"Chroma collection 'multi_modal_rag_v2' at {vector_store_dir / 'chroma'} is empty."
            )

        parent_store_path = vector_store_dir / "parent_store.pkl"
        with open(parent_store_path, "rb") as f:
            parent_data = pickle.load(f)

        store = InMemoryStore()
        store.mset(list(parent_data.items()))

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key="doc_id",
            search_kwargs={"k": top_k},
        )

    def retrieve(self, question: str, k: int | None = None):
        if k is not None:
            self.retriever.search_kwargs["k"] = k
        return self.retriever.invoke(question)