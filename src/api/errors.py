"""Error handlers with stable JSON payloads."""

from __future__ import annotations

import logging
import json
from typing import Any

from flask import Flask, g, jsonify

from src.api.exceptions import APIError

LOGGER = logging.getLogger(__name__)


def _request_id() -> str:
    value = getattr(g, "request_id", "")
    return str(value or "")


def _error_payload(
    *,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
):
    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": _request_id(),
        }
    }
    if details:
        payload["error"]["details"] = json.loads(
            json.dumps(details, ensure_ascii=False, default=str)
        )
    return jsonify(payload), status_code


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError):
        return _error_payload(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )

    @app.errorhandler(404)
    def handle_not_found(_: Exception):
        return _error_payload(
            code="NOT_FOUND",
            message="Endpoint not found.",
            status_code=404,
        )

    @app.errorhandler(405)
    def handle_method_not_allowed(_: Exception):
        return _error_payload(
            code="METHOD_NOT_ALLOWED",
            message="Method not allowed for this endpoint.",
            status_code=405,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        LOGGER.exception("Unhandled API error: %s", exc)
        return _error_payload(
            code="INTERNAL_ERROR",
            message="Internal server error.",
            status_code=500,
        )
