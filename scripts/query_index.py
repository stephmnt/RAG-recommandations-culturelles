#!/usr/bin/env python3
"""CLI script for local FAISS similarity search smoke test."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.build_index import load_indexing_config
from src.indexing.search import search_similar_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local FAISS index.")
    parser.add_argument(
        "--query",
        required=True,
        help="User query to search in indexed chunks.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k results.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Path to indexing YAML config.",
    )
    parser.add_argument(
        "--index-dir",
        default="",
        help="Optional index directory override.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("query_index")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _print_results(query: str, results: list[dict]) -> None:
    print("\n=== Query ===")
    print(query)
    print(f"\n=== Top {len(results)} Results ===")

    for item in results:
        metadata = item.get("metadata", {}) or {}
        score = item.get("score")
        score_display = "n/a" if score is None else f"{score:.6f}"
        content = (item.get("content") or "").replace("\n", " ").strip()
        content_preview = content[:240] + ("..." if len(content) > 240 else "")

        print(f"\n[{item.get('rank')}] score={score_display}")
        print(f"event_id      : {metadata.get('event_id', '')}")
        print(f"start_datetime: {metadata.get('start_datetime', '')}")
        print(f"city          : {metadata.get('city', '')}")
        print(f"url           : {metadata.get('url', '')}")
        print(f"chunk_id      : {metadata.get('chunk_id', '')}")
        print(f"content       : {content_preview}")


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        config = load_indexing_config(args.config)
        index_dir = Path(args.index_dir or config["paths"]["output_dir"]).resolve()
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}. Build it with scripts/build_index.py first."
            )

        logger.info("Loading index from %s", index_dir)
        results = search_similar_chunks(
            query=args.query,
            index_dir=index_dir,
            k=args.k,
            config=config,
            logger=logger,
        )
        _print_results(args.query, results)
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("Query failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
