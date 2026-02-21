#!/usr/bin/env python3
"""Run Flask API locally."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Puls-Events Flask API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    config_overrides = {
        "HOST": args.host,
        "PORT": args.port,
        "LOG_LEVEL": args.log_level,
    }
    app = create_app(config_overrides=config_overrides)
    logger = logging.getLogger("src.api.app")
    logger.info(
        "Starting Flask API host=%s port=%s env=%s",
        args.host,
        args.port,
        app.config.get("FLASK_ENV", "dev"),
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
