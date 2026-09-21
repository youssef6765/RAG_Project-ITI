import app.main as main_module
from fastapi.testclient import TestClient


class FakeRAGService:
    def query(self, question: str):
        return (
            f"Grounded test answer for: {question}",
            ["Attention.pdf p.1"],
        )


def test_health_and_query_happy_path(monkeypatch):
    # Prevent tests from requiring a running Ollama server or a real Chroma DB.
    monkeypatch.setattr(
        main_module,
        "RAGService",
        lambda **kwargs: FakeRAGService(),
    )

    with TestClient(main_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["rag_loaded"] is True

        response = client.post(
            "/query",
            json={"question": "What is attention?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"]
        assert data["sources"] == ["Attention.pdf p.1"]


def test_query_invalid_input(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "RAGService",
        lambda **kwargs: FakeRAGService(),
    )

    with TestClient(main_module.app) as client:
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422
