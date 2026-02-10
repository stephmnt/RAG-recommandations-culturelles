#!/usr/bin/env python3
"""Offline-friendly smoke evaluation for Step-4 RAG engine."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.llm import FakeLLMClient
from src.rag.service import RAGService
from src.rag.types import RAGConfig

TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small smoke evaluation on local RAG service.")
    parser.add_argument(
        "--input",
        default="data/eval/smoke_eval.jsonl",
        help="Chemin du jeu d'evaluation smoke (jsonl).",
    )
    parser.add_argument(
        "--output",
        default="reports/smoke_eval_report.json",
        help="Fichier JSON de sortie du rapport.",
    )
    parser.add_argument(
        "--index_path",
        default="artifacts/faiss_index",
        help="Chemin de l'index FAISS local.",
    )
    parser.add_argument("--top_k", type=int, default=6, help="Top-k retrieval.")
    parser.add_argument("--offline", action="store_true", help="Utilise FakeLLM (sans appel reseau).")
    parser.add_argument("--limit", type=int, default=0, help="Limite le nombre de questions traitees.")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("evaluate_smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def load_eval_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if "question" not in payload:
                raise ValueError(f"Missing 'question' in eval line {line_number}")
            records.append(payload)
    return records


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token}


def keyword_overlap(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_tokens = _tokenize(answer)
    expected_tokens = _tokenize(" ".join(expected_keywords))
    if not answer_tokens or not expected_tokens:
        return 0.0
    intersection = answer_tokens.intersection(expected_tokens)
    union = answer_tokens.union(expected_tokens)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def contains_expected_urls(source_urls: list[str], expected_urls: list[str]) -> bool | None:
    if not expected_urls:
        return None
    normalized_sources = {str(url).strip() for url in source_urls if str(url).strip()}
    normalized_expected = [str(url).strip() for url in expected_urls if str(url).strip()]
    if not normalized_expected:
        return None
    return all(url in normalized_sources for url in normalized_expected)


def _build_service(args: argparse.Namespace, logger: logging.Logger) -> RAGService:
    config = RAGConfig(
        index_path=str(Path(args.index_path).resolve()),
        retriever_top_k=args.top_k,
    )
    llm_client = None
    if args.offline:
        llm_client = FakeLLMClient(
            fixed_answer=(
                "Mode offline: reponse simplifiee. "
                "Verifier les sources pour les details factuels (date, lieu, URL)."
            )
        )
    return RAGService(config=config, llm_client=llm_client, logger=logger)


def evaluate(
    *,
    records: list[dict[str, Any]],
    service: RAGService,
    top_k: int,
    limit: int,
) -> dict[str, Any]:
    eval_records = records[:limit] if limit and limit > 0 else records

    details: list[dict[str, Any]] = []
    with_expected_urls = 0
    matched_urls = 0
    overlap_values: list[float] = []

    for idx, record in enumerate(eval_records, start=1):
        question = str(record.get("question", "")).strip()
        expected_urls = list(record.get("expected_urls", []) or [])
        expected_keywords = list(record.get("expected_keywords", []) or [])

        result = service.ask(question=question, top_k=top_k, debug=False)
        payload = result.model_dump()
        source_urls = [item.get("url", "") for item in payload.get("sources", [])]

        url_match = contains_expected_urls(source_urls, expected_urls)
        if url_match is not None:
            with_expected_urls += 1
            if url_match:
                matched_urls += 1

        overlap = keyword_overlap(payload.get("answer", ""), expected_keywords)
        overlap_values.append(overlap)

        details.append(
            {
                "id": idx,
                "question": question,
                "expected_urls": expected_urls,
                "source_urls": source_urls,
                "contains_expected_urls": url_match,
                "keyword_overlap": round(overlap, 4),
                "returned_events": len(payload.get("sources", [])),
                "answer_preview": (payload.get("answer", "")[:240] + "...")
                if len(payload.get("answer", "")) > 240
                else payload.get("answer", ""),
            }
        )

    avg_keyword_overlap = sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
    url_rate = (matched_urls / with_expected_urls) if with_expected_urls else None

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_questions": len(eval_records),
        "with_expected_urls": with_expected_urls,
        "contains_expected_urls_rate": round(url_rate, 4) if url_rate is not None else None,
        "avg_keyword_overlap": round(avg_keyword_overlap, 4),
        "details": details,
    }


def print_summary(report: dict[str, Any], output_path: Path, offline: bool) -> None:
    print("\n=== Smoke Evaluation Summary ===")
    print(f"Mode offline               : {offline}")
    print(f"Questions traitees         : {report['total_questions']}")
    print(f"Questions avec URLs attendues : {report['with_expected_urls']}")
    print(f"contains_expected_urls_rate: {report['contains_expected_urls_rate']}")
    print(f"avg_keyword_overlap        : {report['avg_keyword_overlap']}")
    print(f"Rapport JSON               : {output_path}")


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    try:
        eval_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()
        index_path = Path(args.index_path).resolve()

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index introuvable: {index_path}. Construire d'abord via scripts/build_index.py."
            )

        records = load_eval_records(eval_path)
        service = _build_service(args, logger)
        report = evaluate(
            records=records,
            service=service,
            top_k=args.top_k,
            limit=args.limit,
        )
        report["offline"] = bool(args.offline)
        report["index_path"] = str(index_path)
        report["top_k"] = args.top_k

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(report, output_path=output_path, offline=bool(args.offline))
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("evaluate_smoke failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
