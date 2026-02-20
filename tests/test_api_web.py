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
            "index": {"path": "artifacts/faiss_index"},
            "rag": {"default_top_k": 6},
        }


def _build_client():
    app = create_app(
        config_overrides={"ADMIN_TOKEN": "secret-token"},
        deps_override=FakeDeps(),
    )
    return app.test_client()


def test_home_page_is_served():
    client = _build_client()
    response = client.get("/")

    assert response.status_code == 200
    assert response.content_type.startswith("text/html")
    body = response.get_data(as_text=True)
    assert "Puls-Events" in body
    assert 'id="signup-form"' in body


def test_app_alias_is_served():
    client = _build_client()
    response = client.get("/app")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="question"' in body
    assert "top_k (nb de sources explorees)" in body


def test_html5up_assets_are_served():
    client = _build_client()
    response = client.get("/assets/css/main.css")

    assert response.status_code == 200
    assert response.content_type.startswith("text/css")
