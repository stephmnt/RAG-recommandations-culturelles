#!/usr/bin/env python3
"""CLI demo for Step-4 RAG engine (local retrieval + Mistral generation)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.service import RAGService
from src.rag.types import RAGConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask one independent question to local RAG engine.")
    parser.add_argument("--query", required=True, help="Question utilisateur.")
    parser.add_argument("--top_k", type=int, default=6, help="Nombre de chunks recuperes.")
    parser.add_argument("--debug", action="store_true", help="Affiche les metadonnees debug.")
    parser.add_argument(
        "--index_path",
        default="artifacts/faiss_index",
        help="Dossier de l'index FAISS local.",
    )
    parser.add_argument(
        "--prompt_version",
        default="v1",
        help="Version du prompt a utiliser.",
    )
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ask_local")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _print_result(result: dict) -> None:
    print("\n=== Question ===")
    print(result["question"])

    print("\n=== Reponse ===")
    print(result["answer"])

    print(f"\n=== Sources ({len(result.get('sources', []))}) ===")
    for idx, source in enumerate(result.get("sources", []), start=1):
        print(f"\n[{idx}] event_id={source.get('event_id', '')} score={source.get('score')}")
        print(f"Titre : {source.get('title', '')}")
        print(f"Date  : {source.get('start_datetime', '')}")
        print(f"Lieu  : {source.get('location_name', '')} ({source.get('city', '')})")
        print(f"URL   : {source.get('url', '')}")
        print(f"Extrait: {source.get('snippet', '')}")

    print("\n=== Meta ===")
    print(json.dumps(result.get("meta", {}), ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        index_path = Path(args.index_path).resolve()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index introuvable: {index_path}. "
                "Construisez d'abord l'index avec scripts/build_index.py."
            )

        config = RAGConfig(
            index_path=str(index_path),
            prompt_version=args.prompt_version,
            retriever_top_k=args.top_k,
        )
        service = RAGService(config=config, logger=logger)
        result = service.ask(args.query, top_k=args.top_k, debug=args.debug)
        payload = result.model_dump()

        _print_result(payload)
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("ask_local failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
