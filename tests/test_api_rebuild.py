from __future__ import annotations

from typing import Any

from src.api.app import create_app


class FakeDeps:
    def __init__(self) -> None:
        self.rebuild_calls = 0
        self.reload_calls = 0
        self.last_payload: dict[str, Any] | None = None

    def ask(self, *, question: str, top_k: int, debug: bool, filters: dict[str, Any] | None = None):
        del question, top_k, debug, filters
        return {
            "question": "q",
            "answer": "a",
            "sources": [],
            "meta": {},
        }

    def rebuild_index(self, *, dataset_path: str | None = None, index_path: str | None = None):
        self.rebuild_calls += 1
        self.last_payload = {
            "dataset_path": dataset_path,
            "index_path": index_path,
        }
        return {
            "status": "ok",
            "mode": "rebuild",
            "index_path": index_path or "artifacts/faiss_index",
            "dataset_path": dataset_path or "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 42},
            "timings_ms": {"build": 12, "reload": 4, "total": 16},
        }

    def reload_index(self, *, index_path: str | None = None):
        self.reload_calls += 1
        self.last_payload = {
            "index_path": index_path,
        }
        return {
            "status": "ok",
            "mode": "reload",
            "index_path": index_path or "artifacts/faiss_index",
            "dataset_path": "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 42},
            "timings_ms": {"total": 3},
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
                "build_date": None,
                "num_events": None,
                "num_chunks": None,
                "embedding_model": "",
                "dataset_hash": None,
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


def test_rebuild_without_token_returns_401():
    client = _build_client(FakeDeps())

    response = client.post("/rebuild", json={"mode": "reload"})

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"]["code"] == "MISSING_ADMIN_TOKEN"


def test_rebuild_with_invalid_token_returns_403():
    client = _build_client(FakeDeps())

    response = client.post(
        "/rebuild",
        json={"mode": "reload"},
        headers={"X-ADMIN-TOKEN": "wrong-token"},
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_ADMIN_TOKEN"


def test_rebuild_mode_invalid_returns_400():
    client = _build_client(FakeDeps())

    response = client.post(
        "/rebuild",
        json={"mode": "invalid-mode"},
        headers={"X-ADMIN-TOKEN": "secret-token"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_rebuild_with_token_ok_returns_200():
    fake_deps = FakeDeps()
    client = _build_client(fake_deps)

    response = client.post(
        "/rebuild",
        json={"mode": "rebuild", "dataset_path": "data/processed/events_processed.parquet"},
        headers={"X-ADMIN-TOKEN": "secret-token"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "rebuild"
    assert fake_deps.rebuild_calls == 1
