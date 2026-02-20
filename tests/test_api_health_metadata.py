from __future__ import annotations

from typing import Any

from src.api.app import create_app


class FakeDeps:
    def ask(self, *, question: str, top_k: int, debug: bool, filters: dict[str, Any] | None = None):
        del question, top_k, debug, filters
        return {
            "question": "q",
            "answer": "a",
            "sources": [],
            "meta": {},
        }

    def rebuild_index(self, *, dataset_path: str | None = None, index_path: str | None = None):
        del dataset_path, index_path
        return {
            "status": "ok",
            "mode": "rebuild",
            "index_path": "artifacts/faiss_index",
            "dataset_path": "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 10},
            "timings_ms": {"build": 4, "reload": 2, "total": 6},
        }

    def reload_index(self, *, index_path: str | None = None):
        del index_path
        return {
            "status": "ok",
            "mode": "reload",
            "index_path": "artifacts/faiss_index",
            "dataset_path": "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 10},
            "timings_ms": {"total": 2},
        }

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": True,
            "mistral_configured": False,
            "version": "0.1.0",
            "timestamp": "2026-02-10T12:00:00Z",
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        return {
            "index": {
                "path": "artifacts/faiss_index",
                "build_date": "2026-02-10T10:00:00Z",
                "num_events": 123,
                "num_chunks": 456,
                "embedding_model": "intfloat/multilingual-e5-small",
                "dataset_hash": "abc123",
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
        },
        deps_override=fake_deps,
    )
    return app.test_client()


def test_health_returns_expected_fields():
    client = _build_client(FakeDeps())

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["api"] == "up"
    assert "index_loaded" in payload
    assert "mistral_configured" in payload
    assert "timestamp" in payload


def test_metadata_returns_index_and_rag_sections():
    client = _build_client(FakeDeps())

    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.get_json()
    assert "index" in payload
    assert "rag" in payload
    assert payload["index"]["num_chunks"] == 456
    assert payload["rag"]["default_top_k"] == 6
