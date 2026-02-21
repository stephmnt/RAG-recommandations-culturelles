from __future__ import annotations

from src.api.app import create_app


def _client():
    app = create_app(
        {
            "TESTING": True,
            "INDEX_PATH": "artifacts/faiss_index",
            "MISTRAL_MODEL": "mistral-small-latest",
            "DEFAULT_TOP_K": 6,
            "MAX_TOP_K": 10,
            "PROMPT_VERSION": "v1",
        }
    )
    return app.test_client()


def test_health_returns_expected_fields(monkeypatch):
    client = _client()

    monkeypatch.setattr("src.api.routes.index_available", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["api"] == "up"
    assert payload["index_loaded"] is True
    assert "timestamp" in payload


def test_metadata_returns_expected_fields(monkeypatch):
    client = _client()
    fake_metadata = {
        "built_at_utc": "2026-02-20T10:00:00Z",
        "events_valid": 123,
        "chunks_count": 456,
        "dataset_hash": "abc123",
        "embeddings": {"huggingface_model": "intfloat/multilingual-e5-small"},
    }

    monkeypatch.setattr("src.api.routes.get_index_metadata", lambda: fake_metadata)

    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["index"]["num_events"] == 123
    assert payload["index"]["num_chunks"] == 456
    assert payload["rag"]["default_top_k"] == 6
    assert payload["rag"]["prompt_version"] == "v1"
