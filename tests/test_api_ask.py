from __future__ import annotations

from src.api.app import create_app
from src.rag.types import RAGResult, RAGSource


class _FakeIndexManager:
    is_busy = False


class _FakeRAGService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, bool]] = []

    def ask(self, question: str, top_k: int | None = None, debug: bool = False) -> RAGResult:
        self.calls.append((question, top_k, debug))
        return RAGResult(
            question=question,
            answer="Reponse test",
            sources=[
                RAGSource(
                    event_id="evt-1",
                    title="Concert jazz",
                    start_datetime="2026-02-20T19:00:00Z",
                    city="Montpellier",
                    location_name="Salle Victoire",
                    url="https://example.org/evt-1",
                    snippet="Extrait test",
                )
            ],
            meta={"retriever_top_k": top_k, "filters_applied": [], "warnings": []},
        )


def _make_client():
    app = create_app(
        {
            "TESTING": True,
            "ADMIN_TOKEN": "secret-token",
            "DEFAULT_TOP_K": 6,
            "MAX_TOP_K": 10,
        }
    )
    return app.test_client()


def test_ask_happy_path_and_top_k_clamped(monkeypatch):
    client = _make_client()
    fake_service = _FakeRAGService()

    monkeypatch.setattr("src.api.routes.index_available", lambda: True)
    monkeypatch.setattr("src.api.routes.get_index_manager", lambda: _FakeIndexManager())
    monkeypatch.setattr("src.api.routes.get_rag_service", lambda: fake_service)

    response = client.post(
        "/ask",
        json={
            "question": "Quels evenements jazz dans l Herault ?",
            "top_k": 999,
            "debug": True,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["question"] == "Quels evenements jazz dans l Herault ?"
    assert payload["meta"]["retriever_top_k"] == 10
    assert len(payload["sources"]) == 1
    assert fake_service.calls[0][1] == 10


def test_ask_question_empty_returns_400():
    client = _make_client()

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_ask_index_unavailable_returns_503(monkeypatch):
    client = _make_client()

    monkeypatch.setattr("src.api.routes.index_available", lambda: False)
    monkeypatch.setattr("src.api.routes.get_index_manager", lambda: _FakeIndexManager())

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"]["code"] == "INDEX_UNAVAILABLE"
