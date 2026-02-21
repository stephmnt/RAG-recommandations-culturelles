from __future__ import annotations

from src.api.app import create_app


class _FakeIndexManager:
    def __init__(self) -> None:
        self.is_busy = False
        self.last_mode = ""

    def rebuild_index(self, *, dataset_path: str, index_path: str):
        self.last_mode = "rebuild"
        return {
            "mode": "rebuild",
            "dataset_path": dataset_path,
            "index_path": index_path,
            "index_metadata": {"chunks_count": 42},
            "timings_ms": {"total": 12},
        }

    def reload_index(self, *, index_path: str, dataset_path: str):
        self.last_mode = "reload"
        return {
            "mode": "reload",
            "dataset_path": dataset_path,
            "index_path": index_path,
            "index_metadata": {"chunks_count": 42},
            "timings_ms": {"total": 3},
        }


def _client():
    app = create_app(
        {
            "TESTING": True,
            "ADMIN_TOKEN": "secret-token",
            "DATASET_PATH": "tests/data/processed.parquet",
            "INDEX_PATH": "tests/artifacts/faiss_index",
        }
    )
    return app.test_client()


def test_rebuild_requires_token():
    client = _client()

    response = client.post("/rebuild", json={"mode": "reload"})

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"]["code"] == "UNAUTHORIZED"


def test_rebuild_invalid_token_forbidden():
    client = _client()

    response = client.post(
        "/rebuild",
        json={"mode": "reload"},
        headers={"X-ADMIN-TOKEN": "bad-token"},
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["error"]["code"] == "FORBIDDEN"


def test_rebuild_mode_reload_ok(monkeypatch):
    client = _client()
    fake_manager = _FakeIndexManager()

    monkeypatch.setattr("src.api.routes.get_index_manager", lambda: fake_manager)
    monkeypatch.setattr("src.api.routes.reset_rag_service", lambda: None)

    response = client.post(
        "/rebuild",
        json={"mode": "reload"},
        headers={"X-ADMIN-TOKEN": "secret-token"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "reload"


def test_rebuild_mode_rebuild_ok(monkeypatch, tmp_path):
    dataset_path = tmp_path / "events_processed.parquet"
    dataset_path.write_text("placeholder", encoding="utf-8")
    client = _client()
    fake_manager = _FakeIndexManager()

    monkeypatch.setattr("src.api.routes.get_index_manager", lambda: fake_manager)
    monkeypatch.setattr("src.api.routes.reset_rag_service", lambda: None)

    response = client.post(
        "/rebuild",
        json={
            "mode": "rebuild",
            "dataset_path": str(dataset_path),
            "index_path": str(tmp_path / "faiss_index"),
        },
        headers={"X-ADMIN-TOKEN": "secret-token"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "rebuild"
