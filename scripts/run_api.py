#!/usr/bin/env python3
"""Run Flask API for Puls-Events RAG (Step 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flask API locally.")
    parser.add_argument("--host", default="", help="Host override (default from env HOST)")
    parser.add_argument("--port", type=int, default=0, help="Port override (default from env PORT)")
    parser.add_argument(
        "--log-level",
        default="",
        help="Logging level override (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    overrides: dict[str, object] = {}
    if args.host:
        overrides["HOST"] = args.host
    if args.port:
        overrides["PORT"] = args.port
    if args.log_level:
        overrides["LOG_LEVEL"] = args.log_level
    if args.debug:
        overrides["DEBUG"] = True

    app = create_app(config_overrides=overrides)
    settings = app.config["API_SETTINGS"]

    app.logger.info(
        "Starting Flask API host=%s port=%s env=%s",
        settings.host,
        settings.port,
        settings.flask_env,
    )
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
