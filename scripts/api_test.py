#!/usr/bin/env python3
"""Manual smoke test for local Flask API endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test local Flask API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--offline", action="store_true", help="Skip /ask request")
    parser.add_argument(
        "--admin-token",
        default="",
        help="Optional token for /rebuild reload smoke request.",
    )
    return parser.parse_args()


def pretty_print(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def call_endpoint(
    *,
    method: str,
    url: str,
    timeout: int,
    json_payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    response = requests.request(
        method=method,
        url=url,
        json=json_payload,
        headers=headers,
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    return response.status_code, payload


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    base_url = args.base_url.rstrip("/")
    admin_token = args.admin_token.strip() or os.getenv("ADMIN_TOKEN", "").strip()

    try:
        status, payload = call_endpoint(
            method="GET",
            url=f"{base_url}/health",
            timeout=args.timeout,
        )
        pretty_print(f"GET /health [{status}]", payload)

        status, payload = call_endpoint(
            method="GET",
            url=f"{base_url}/metadata",
            timeout=args.timeout,
        )
        pretty_print(f"GET /metadata [{status}]", payload)

        if not args.offline:
            ask_payload = {
                "question": "Quels evenements jazz dans l'Herault cette semaine ?",
                "top_k": 6,
                "debug": False,
            }
            status, payload = call_endpoint(
                method="POST",
                url=f"{base_url}/ask",
                timeout=args.timeout,
                json_payload=ask_payload,
            )
            pretty_print(f"POST /ask [{status}]", payload)
        else:
            print("\n[offline] /ask ignore (aucun appel generation live).")

        if admin_token:
            status, payload = call_endpoint(
                method="POST",
                url=f"{base_url}/rebuild",
                timeout=args.timeout,
                json_payload={"mode": "reload"},
                headers={"X-ADMIN-TOKEN": admin_token},
            )
            pretty_print(f"POST /rebuild mode=reload [{status}]", payload)
        else:
            print("\nADMIN_TOKEN absent: /rebuild non teste.")

        return 0
    except requests.RequestException as exc:
        print(f"API smoke test failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
