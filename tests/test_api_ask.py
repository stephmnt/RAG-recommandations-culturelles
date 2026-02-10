from __future__ import annotations

from typing import Any

from src.api.app import create_app


class FakeDeps:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    def ask(self, *, question: str, top_k: int, debug: bool, filters: dict[str, Any] | None = None):
        self.last_call = {
            "question": question,
            "top_k": top_k,
            "debug": debug,
            "filters": filters,
        }
        return {
            "question": question,
            "answer": "Reponse mockee",
            "sources": [
                {
                    "event_id": "evt-1",
                    "title": "Concert jazz",
                    "start_datetime": "2026-02-14T20:00:00Z",
                    "end_datetime": None,
                    "city": "Montpellier",
                    "location_name": "Salle A",
                    "url": "https://example.org/evt-1",
                    "score": 0.21,
                    "snippet": "Concert jazz a Montpellier",
                }
            ],
            "meta": {
                "retriever_top_k": top_k,
                "retrieved_chunks": 1,
                "returned_events": 1,
                "latency_ms": {"load_index": 1, "retrieval": 2, "generation": 3},
                "model": "mistral-small-latest",
                "prompt_version": "v1",
                "timestamp": "2026-02-10T12:00:00Z",
                "warnings": [],
            },
        }

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": True,
            "mistral_configured": True,
            "version": "0.1.0",
            "timestamp": "2026-02-10T12:00:00Z",
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        return {
            "index": {
                "path": "artifacts/faiss_index",
                "build_date": "2026-02-10T10:00:00Z",
                "num_events": 12,
                "num_chunks": 24,
                "embedding_model": "intfloat/multilingual-e5-small",
                "dataset_hash": "abc",
            },
            "rag": {
                "default_top_k": 6,
                "max_top_k": 10,
                "prompt_version": "v1",
                "llm_model": "mistral-small-latest",
            },
        }


def _build_client(fake_deps: FakeDeps):
    app = create_app(
        config_overrides={
            "ADMIN_TOKEN": "secret-token",
            "MAX_TOP_K": 10,
            "DEFAULT_TOP_K": 6,
        },
        deps_override=fake_deps,
    )
    return app.test_client()


def test_ask_happy_path_returns_rag_result():
    fake_deps = FakeDeps()
    client = _build_client(fake_deps)

    response = client.post(
        "/ask",
        json={
            "question": "Quels concerts jazz dans l'Herault ?",
            "top_k": 4,
            "debug": True,
            "filters": {"city": "Montpellier"},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["question"] == "Quels concerts jazz dans l'Herault ?"
    assert payload["answer"] == "Reponse mockee"
    assert len(payload["sources"]) == 1
    assert fake_deps.last_call is not None
    assert fake_deps.last_call["top_k"] == 4
    assert fake_deps.last_call["filters"] == {"city": "Montpellier"}


def test_ask_question_empty_returns_400():
    fake_deps = FakeDeps()
    client = _build_client(fake_deps)

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_ask_top_k_is_clamped_to_max():
    fake_deps = FakeDeps()
    client = _build_client(fake_deps)

    response = client.post(
        "/ask",
        json={
            "question": "Donne-moi des sorties culturelles",
            "top_k": 999,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert fake_deps.last_call is not None
    assert fake_deps.last_call["top_k"] == 10
    warnings = payload["meta"]["warnings"]
    assert any(str(item).startswith("top_k_clamped") for item in warnings)
