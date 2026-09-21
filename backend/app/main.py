from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.query import router
from .core.config import get_settings
from .services.rag import RAGService
from .utils.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    vector_store_dir = settings.resolved_vector_store_dir()
    if not vector_store_dir.exists():
        raise RuntimeError(
            f"Persisted vector store not found: {vector_store_dir}. "
            "Run Phase 2 notebook first."
        )

    app.state.rag_service = RAGService(
        vector_store_dir=vector_store_dir,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        top_k=settings.top_k,
    )

    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Grounded RAG assistant over the project PDF knowledge base.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
