# tests/test_rag.py
from fastapi.testclient import TestClient
from src.rag.api import app

def test_health_check():
    # Usar o 'with' garante que o lifespan (startup/shutdown) seja executado
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_ask_endpoint_smoke_test():
    """Testa se o endpoint /ask responde corretamente a uma pergunta válida."""
    # Usar o 'with' garante que o lifespan (startup/shutdown) seja executado
    with TestClient(app) as client:
        # Este teste depende que o `make run-rag-ingest` tenha sido executado
        response = client.post(
            "/ask",
            json={"question": "Qual a temperatura de operação?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert len(data["answer"]) > 0