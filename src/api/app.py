"""Flask application factory for Step-5 API."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, g, request

from src.api.config import APISettings
from src.api.deps import AppDependencies, init_dependencies
from src.api.errors import register_error_handlers
from src.api.routes import api_bp


def _configure_logging(level_name: str) -> logging.Logger:
    logger = logging.getLogger("puls_events_api")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def create_app(
    config_overrides: dict[str, Any] | None = None,
    deps_override: AppDependencies | None = None,
) -> Flask:
    """Create configured Flask app with routes, errors and dependency container."""

    load_dotenv(override=False)
    settings = APISettings.from_env(config_overrides)

    app = Flask(__name__)
    app.config["API_SETTINGS"] = settings

    logger = _configure_logging(settings.log_level)
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)

    register_error_handlers(app)
    init_dependencies(app, deps_override=deps_override)
    app.register_blueprint(api_bp)

    @app.before_request
    def _before_request() -> None:
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.request_started_at = time.perf_counter()

    @app.after_request
    def _after_request(response: Response) -> Response:
        started = getattr(g, "request_started_at", None)
        latency_ms = int((time.perf_counter() - started) * 1000) if started else -1
        request_id = getattr(g, "request_id", "")

        response.headers["X-Request-ID"] = str(request_id)
        logger.info(
            "api.request endpoint=%s method=%s status=%s latency_ms=%s request_id=%s",
            request.path,
            request.method,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response

    return app


__all__ = ["create_app"]
