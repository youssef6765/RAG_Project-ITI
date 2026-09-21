from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the documents")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
