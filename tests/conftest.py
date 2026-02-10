"""Test configuration helpers."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _MockResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = int(payload.get("status_code", 200))
        self._json_payload = payload.get("json")
        self.text = str(payload.get("text", ""))
        if not self.text and self._json_payload is not None:
            self.text = json.dumps(self._json_payload, ensure_ascii=False)

    def json(self) -> Any:
        if self._json_payload is None:
            raise ValueError("No JSON payload configured for this mocked response.")
        return self._json_payload


class _RequestsMockFallback:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.call_count = 0
        self._routes: dict[str, deque[dict[str, Any]]] = {}

        def _session_get(
            session: requests.Session,
            url: str,
            params: dict[str, Any] | None = None,
            timeout: int | float | None = None,
            **kwargs: Any,
        ) -> _MockResponse:
            del session
            return self._handle_get(url=url, params=params, timeout=timeout, **kwargs)

        monkeypatch.setattr(requests.sessions.Session, "get", _session_get)

    def get(self, url: str, responses: dict[str, Any] | list[dict[str, Any]]) -> None:
        if isinstance(responses, list):
            self._routes[url] = deque(responses)
        else:
            self._routes[url] = deque([responses])

    def _handle_get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: int | float | None = None,
        **kwargs: Any,
    ) -> _MockResponse:
        del params, timeout, kwargs
        route = self._routes.get(url)
        if not route:
            raise AssertionError(f"Unexpected request URL in test: {url}")
        payload = route.popleft()
        self.call_count += 1
        return _MockResponse(payload)


@pytest.fixture
def requests_mock(monkeypatch: pytest.MonkeyPatch):
    """Fallback fixture compatible with test expectations when plugin is absent."""

    return _RequestsMockFallback(monkeypatch)
