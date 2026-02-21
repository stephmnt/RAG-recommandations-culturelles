"""Flask application factory for Puls-Events API."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from flask import Flask, g, request

from src.api.config import APISettings
from src.api.deps import configure, get_settings
from src.api.errors import register_error_handlers
from src.api.routes import api_bp

LOGGER = logging.getLogger("puls_events_api")


def _setup_logging(log_level: str) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    LOGGER.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    settings: APISettings = configure(config_overrides)
    _setup_logging(settings.log_level)

    app = Flask(__name__)
    app.config.update(settings.to_app_dict())
    app.register_blueprint(api_bp)
    register_error_handlers(app)

    @app.before_request
    def _before_request() -> None:
        g.request_id = str(uuid.uuid4())
        g.started_at = time.perf_counter()

    @app.after_request
    def _after_request(response):
        elapsed_ms = int((time.perf_counter() - getattr(g, "started_at", time.perf_counter())) * 1000)
        request_id = getattr(g, "request_id", "")
        endpoint = request.path
        method = request.method

        LOGGER.info(
            "api.request endpoint=%s method=%s status=%s latency_ms=%s request_id=%s",
            endpoint,
            method,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    return app


def app_settings() -> APISettings:
    return get_settings()
