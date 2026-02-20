"""Standardized JSON error responses for Flask API."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from flask import Flask, jsonify, g
from pydantic import ValidationError

LOGGER = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def _request_id() -> str:
    rid = getattr(g, "request_id", "")
    if rid:
        return str(rid)
    fallback = str(uuid.uuid4())
    g.request_id = fallback
    return fallback


def _error_payload(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": _request_id(),
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return payload


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError):  # type: ignore[no-untyped-def]
        return jsonify(
            _error_payload(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        ), exc.status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(exc: ValidationError):  # type: ignore[no-untyped-def]
        return jsonify(
            _error_payload(
                code="INVALID_SCHEMA",
                message="Request schema validation failed.",
                details=exc.errors(),
            )
        ), 422

    @app.errorhandler(404)
    def handle_not_found(exc):  # type: ignore[no-untyped-def]
        del exc
        return jsonify(
            _error_payload(
                code="NOT_FOUND",
                message="Endpoint not found.",
            )
        ), 404

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):  # type: ignore[no-untyped-def]
        request_id = _request_id()
        LOGGER.exception("Unhandled API exception request_id=%s error=%s", request_id, exc)
        return jsonify(
            _error_payload(
                code="INTERNAL_ERROR",
                message="Unexpected server error.",
            )
        ), 500


__all__ = ["APIError", "register_error_handlers"]
