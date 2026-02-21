#!/usr/bin/env python3
"""Smoke test for the local environment."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_versions() -> None:
    print("=== Environment versions ===")
    print(f"python: {platform.python_version()}")
    print(f"langchain: {get_version('langchain')}")
    print(f"faiss-cpu: {get_version('faiss-cpu')}")
    print(f"mistralai: {get_version('mistralai')}")
    print(f"pandas: {get_version('pandas')}")
    print(f"requests: {get_version('requests')}")
    print(f"flask: {get_version('flask')}")


def run_import_checks() -> list[str]:
    failures: list[str] = []

    try:
        import faiss  # noqa: F401
        print("[OK] import faiss")
    except Exception as exc:  # pragma: no cover - intentional broad catch for smoke test
        failures.append(f"import faiss -> {exc}")

    try:
        from langchain.vectorstores import FAISS  # noqa: F401
        print("[OK] from langchain.vectorstores import FAISS")
    except Exception as exc:  # pragma: no cover
        failures.append(f"from langchain.vectorstores import FAISS -> {exc}")

    try:
        from langchain.embeddings import HuggingFaceEmbeddings  # noqa: F401
        print("[OK] from langchain.embeddings import HuggingFaceEmbeddings")
    except Exception as exc:  # pragma: no cover
        failures.append(
            "from langchain.embeddings import HuggingFaceEmbeddings -> "
            f"{exc}"
        )

    try:
        from mistral import MistralClient  # noqa: F401
        print("[OK] from mistral import MistralClient")
    except Exception as exc:  # pragma: no cover
        failures.append(f"from mistral import MistralClient -> {exc}")

    try:
        import flask  # noqa: F401
        print("[OK] import flask")
    except Exception as exc:  # pragma: no cover
        failures.append(f"import flask -> {exc}")

    return failures


def main() -> int:
    print_versions()
    major, minor = sys.version_info[:2]
    if major != 3 or minor not in (10, 11):
        print(
            "\n[ERROR] Unsupported Python version for this project lock: "
            f"{major}.{minor}. Use Python 3.10 or 3.11."
        )
        return 1

    print("\n=== Import checks ===")
    failures = run_import_checks()
    if failures:
        print("\n[ERROR] Import checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\n[SUCCESS] Environment smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
