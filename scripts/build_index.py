#!/usr/bin/env python3
"""CLI script to rebuild FAISS index from processed dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.build_index import build_faiss_index, load_indexing_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from processed events.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Path to indexing YAML config.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional input dataset override (parquet/jsonl/csv).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output directory override for FAISS artifacts.",
    )
    parser.add_argument(
        "--provider",
        default="",
        choices=["", "huggingface", "mistral"],
        help="Optional embedding provider override.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Optional embedding model override (provider-specific).",
    )
    parser.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs/build_index.log"),
        help="Log file path.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_index")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    cfg_paths = dict(cfg.get("paths", {}))
    cfg_embeddings = dict(cfg.get("embeddings", {}))

    if args.input:
        cfg_paths["input_dataset"] = args.input
    if args.output:
        cfg_paths["output_dir"] = args.output

    if args.provider:
        cfg_embeddings["provider"] = args.provider
        os.environ["EMBEDDING_PROVIDER"] = args.provider

    if args.embedding_model:
        if cfg_embeddings.get("provider", "huggingface") == "mistral":
            cfg_embeddings["mistral_model"] = args.embedding_model
            os.environ["MISTRAL_EMBEDDING_MODEL"] = args.embedding_model
        else:
            cfg_embeddings["huggingface_model"] = args.embedding_model
            os.environ["EMBEDDING_MODEL"] = args.embedding_model

    cfg["paths"] = cfg_paths
    cfg["embeddings"] = cfg_embeddings
    return cfg


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== FAISS build summary ===")
    print(f"Events input            : {summary['events_input']}")
    print(f"Events valid            : {summary['events_valid']}")
    print(f"Events invalid filtered : {summary['events_invalid']}")
    print(f"Chunks indexed          : {summary['chunks_count']}")
    print(f"Chunking time (s)       : {summary['chunking_seconds']}")
    print(f"Index build time (s)    : {summary['index_build_seconds']}")
    print(f"Total time (s)          : {summary['total_seconds']}")
    print(f"Output dir              : {summary['output_dir']}")
    print(f"Index metadata          : {summary['metadata_path']}")
    print(f"Dataset hash            : {summary['dataset_hash']}")


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file).resolve()
    logger = setup_logging(log_path)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        config = load_indexing_config(args.config)
        config = apply_overrides(config, args)

        input_dataset = Path(config["paths"]["input_dataset"]).resolve()
        output_dir = Path(config["paths"]["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building index with config=%s", args.config)
        logger.info("Input dataset=%s", input_dataset)
        logger.info("Output dir=%s", output_dir)
        logger.info("Embedding provider=%s", config.get("embeddings", {}).get("provider", "huggingface"))

        result = build_faiss_index(
            input_path=input_dataset,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )
        summary = result.to_dict()
        logger.info("Build summary: %s", summary)
        _print_summary(summary)
        return 0

    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("Index build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
