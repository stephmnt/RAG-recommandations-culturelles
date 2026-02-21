#!/usr/bin/env python3
"""Manual API smoke test for Step 5/6."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test local Puls-Events API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--admin-token", default="", help="Optional admin token for /rebuild")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip /ask call when no index/model is ready.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    return parser.parse_args()


def _print_response(name: str, response: requests.Response) -> None:
    print(f"\n=== {name} ===")
    print(f"status: {response.status_code}")
    try:
        payload = response.json()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)


def _request(
    method: str,
    url: str,
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> requests.Response:
    return requests.request(method, url, timeout=timeout, headers=headers, json=json_payload)


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    timeout = args.timeout

    try:
        health = _request("GET", f"{base_url}/health", timeout=timeout)
        _print_response("GET /health", health)
        if health.status_code != 200:
            return 1

        metadata = _request("GET", f"{base_url}/metadata", timeout=timeout)
        _print_response("GET /metadata", metadata)
        if metadata.status_code != 200:
            return 1

        if not args.offline:
            ask_payload = {
                "question": "Quels evenements jazz dans l Herault cette semaine ?",
                "top_k": 6,
                "debug": True,
            }
            ask_response = _request(
                "POST",
                f"{base_url}/ask",
                timeout=timeout,
                headers={"Content-Type": "application/json"},
                json_payload=ask_payload,
            )
            _print_response("POST /ask", ask_response)
            if ask_response.status_code not in {200, 503}:
                return 1
        else:
            print("\n[INFO] offline mode active: /ask skipped.")

        if args.admin_token:
            rebuild_response = _request(
                "POST",
                f"{base_url}/rebuild",
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "X-ADMIN-TOKEN": args.admin_token,
                },
                json_payload={"mode": "reload"},
            )
            _print_response("POST /rebuild (reload)", rebuild_response)
            if rebuild_response.status_code != 200:
                return 1
        else:
            print("\n[INFO] admin token not provided: /rebuild skipped.")

        print("\nAPI smoke test: OK")
        return 0
    except requests.RequestException as exc:
        print(f"API smoke test failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
