from fastapi import APIRouter, Request

from ...schemas.query import QueryRequest, QueryResponse


router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "service": "rag-document-assistant",
        "rag_loaded": hasattr(request.app.state, "rag_service"),
    }


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request):
    answer, sources = request.app.state.rag_service.query(payload.question)
    return QueryResponse(answer=answer, sources=sources)
