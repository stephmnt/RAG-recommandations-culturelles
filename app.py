#!/usr/bin/env python3
"""Unified CLI entrypoint for Puls-Events project."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "data/processed/events_processed.parquet"
DEFAULT_INDEX = PROJECT_ROOT / "artifacts/faiss_index"
TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _load_dotenv() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _setup_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# -----------------------------
# check-env command
# -----------------------------
def _get_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _print_versions() -> None:
    print("=== Environment versions ===")
    print(f"python: {platform.python_version()}")
    print(f"langchain: {_get_version('langchain')}")
    print(f"faiss-cpu: {_get_version('faiss-cpu')}")
    print(f"mistralai: {_get_version('mistralai')}")
    print(f"pandas: {_get_version('pandas')}")
    print(f"requests: {_get_version('requests')}")
    print(f"flask: {_get_version('flask')}")


def _run_import_checks() -> list[str]:
    failures: list[str] = []

    try:
        import faiss  # noqa: F401

        print("[OK] import faiss")
    except Exception as exc:  # pragma: no cover
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
        failures.append(f"from langchain.embeddings import HuggingFaceEmbeddings -> {exc}")

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


def cmd_check_env(_: argparse.Namespace) -> int:
    _print_versions()
    major, minor = sys.version_info[:2]
    if major != 3 or minor not in (10, 11):
        print(
            "\n[ERROR] Unsupported Python version for this project lock: "
            f"{major}.{minor}. Use Python 3.10 or 3.11."
        )
        return 1

    print("\n=== Import checks ===")
    failures = _run_import_checks()
    if failures:
        print("\n[ERROR] Import checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\n[SUCCESS] Environment smoke test passed.")
    return 0


# -----------------------------
# build-dataset command
# -----------------------------
def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("config.yaml must contain a mapping at root level.")
    return payload


def _apply_env_overrides(config: dict[str, Any]) -> None:
    _load_dotenv()
    openagenda_key = os.getenv("OPENAGENDA_API_KEY", "").strip()
    if openagenda_key:
        config.setdefault("openagenda", {}).setdefault("auth", {})["api_key"] = openagenda_key


def _resolve_time_window(config: dict[str, Any]) -> tuple[str, str]:
    today = date.today()
    default_start = today - timedelta(days=365)
    default_end = today + timedelta(days=90)

    openagenda = config.setdefault("openagenda", {})
    time_window = openagenda.setdefault("time_window", {})

    start_date = (time_window.get("start_date") or str(default_start)).strip()
    end_date = (time_window.get("end_date") or str(default_end)).strip()

    time_window["start_date"] = start_date
    time_window["end_date"] = end_date
    return start_date, end_date


def _write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_parquet(
    output_path: Path,
    records: list[dict[str, Any]],
    record_fields: list[str],
) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        dataframe = pd.DataFrame(records)
    else:
        dataframe = pd.DataFrame(columns=record_fields)
    dataframe.to_parquet(output_path, index=False)


def _print_dataset_summary(summary: dict[str, Any]) -> None:
    print("\n=== Build summary ===")
    print(f"Agendas found            : {summary['agendas_found']}")
    print(f"Agendas scanned          : {summary['agendas_scanned']}")
    print(f"Legacy fallback used     : {summary['legacy_fallback_used']}")
    print(f"Raw events fetched       : {summary['raw_events']}")
    print(f"Outside geo scope        : {summary['outside_geo_scope']}")
    print(f"After period filtering   : {summary['after_period_filter']}")
    print(f"Duplicates removed       : {summary['duplicates_removed']}")
    print(f"Invalid records dropped  : {summary['invalid_records']}")
    print(f"Final processed records  : {summary['processed_events']}")
    print(f"Geo scope mode           : {summary['geo_scope_mode']}")
    print(f"Geo strict mode          : {summary['geo_scope_strict']}")
    print(f"Top external cities      : {summary['external_city_counts']}")
    print(f"Events by agenda         : {summary['events_by_agenda']}")
    print(f"Raw output               : {summary['raw_output']}")
    print(f"Processed output         : {summary['processed_output']}")
    print(f"Log file                 : {summary['log_file']}")


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from src.openagenda.client import OpenAgendaConfig, fetch_events_with_stats
    from src.preprocess.cleaning import clean_events
    from src.preprocess.schema import EVENT_RECORD_FIELDS

    log_path = Path(args.log_file).resolve()
    logger = _setup_logger("build_dataset", log_path)

    try:
        config_path = Path(args.config).resolve()
        raw_output = Path(args.raw_output).resolve()
        processed_output = Path(args.processed_output).resolve()

        logger.info("Loading configuration from %s", config_path)
        config = _load_yaml_config(config_path)
        _apply_env_overrides(config)
        start_date, end_date = _resolve_time_window(config)

        logger.info("Using time window start_date=%s end_date=%s", start_date, end_date)
        oa_config = OpenAgendaConfig.from_dict(config)
        oa_config.start_date = start_date
        oa_config.end_date = end_date

        if not oa_config.api_key:
            logger.warning("OPENAGENDA_API_KEY is missing. Requests are likely to fail with HTTP 401/403.")

        logger.info("Fetching events from OpenAgenda (multi-agendas flow)")
        raw_events, ingestion_stats = fetch_events_with_stats(oa_config)
        logger.info(
            "OpenAgenda stats agendas_found=%s agendas_scanned=%s total_events=%s legacy_fallback=%s",
            ingestion_stats.get("agendas_found", 0),
            ingestion_stats.get("agendas_scanned", 0),
            ingestion_stats.get("total_events", 0),
            ingestion_stats.get("legacy_fallback_used", False),
        )
        logger.info("OpenAgenda events_by_agenda=%s", ingestion_stats.get("events_by_agenda", {}))

        _write_jsonl(raw_output, raw_events)
        logger.info("Raw events written to %s", raw_output)

        logger.info("Cleaning and validating events")
        geo_scope = config.get("openagenda", {}).get("geo_scope", {})
        records, stats = clean_events(
            raw_events=raw_events,
            start_date=start_date,
            end_date=end_date,
            language=oa_config.language,
            source="openagenda",
            geo_scope=geo_scope if isinstance(geo_scope, dict) else {},
        )
        _write_parquet(processed_output, records, EVENT_RECORD_FIELDS)
        logger.info("Processed events written to %s", processed_output)

        summary = {
            "agendas_found": ingestion_stats.get("agendas_found", 0),
            "agendas_scanned": ingestion_stats.get("agendas_scanned", 0),
            "legacy_fallback_used": ingestion_stats.get("legacy_fallback_used", False),
            "raw_events": len(raw_events),
            "outside_geo_scope": stats["outside_geo_scope"],
            "after_period_filter": stats["after_period_filter"],
            "duplicates_removed": stats["duplicates_removed"],
            "invalid_records": stats["invalid_records"],
            "processed_events": stats["processed_events"],
            "geo_scope_mode": stats.get("geo_scope_mode", "none"),
            "geo_scope_strict": stats.get("geo_scope_strict", True),
            "external_city_counts": stats.get("external_city_counts", {}),
            "events_by_agenda": ingestion_stats.get("events_by_agenda", {}),
            "raw_output": str(raw_output),
            "processed_output": str(processed_output),
            "log_file": str(log_path),
        }
        logger.info("Build summary: %s", summary)
        _print_dataset_summary(summary)
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("Dataset build failed: %s", exc)
        return 1


# -----------------------------
# build-index command
# -----------------------------
def _apply_index_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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


def _print_index_summary(summary: dict[str, Any]) -> None:
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


def cmd_build_index(args: argparse.Namespace) -> int:
    from src.indexing.build_index import build_faiss_index, load_indexing_config

    _load_dotenv()
    log_path = Path(args.log_file).resolve()
    logger = _setup_logger("build_index", log_path)

    try:
        config = load_indexing_config(args.config)
        config = _apply_index_overrides(config, args)

        input_dataset = Path(config["paths"]["input_dataset"]).resolve()
        output_dir = Path(config["paths"]["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building index with config=%s", args.config)
        logger.info("Input dataset=%s", input_dataset)
        logger.info("Output dir=%s", output_dir)
        logger.info(
            "Embedding provider=%s", config.get("embeddings", {}).get("provider", "huggingface")
        )

        result = build_faiss_index(
            input_path=input_dataset,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )
        summary = result.to_dict()
        logger.info("Build summary: %s", summary)
        _print_index_summary(summary)
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("Index build failed: %s", exc)
        return 1


# -----------------------------
# query-index command
# -----------------------------
def _print_query_results(query: str, results: list[dict[str, Any]]) -> None:
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


def cmd_query_index(args: argparse.Namespace) -> int:
    from src.indexing.build_index import load_indexing_config
    from src.indexing.search import search_similar_chunks

    _load_dotenv()
    logger = _setup_logger("query_index")
    try:
        config = load_indexing_config(args.config)
        index_dir = Path(args.index_dir or config["paths"]["output_dir"]).resolve()
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}. Build it with `python app.py build-index` first."
            )

        logger.info("Loading index from %s", index_dir)
        results = search_similar_chunks(
            query=args.query,
            index_dir=index_dir,
            k=args.k,
            config=config,
            logger=logger,
        )
        _print_query_results(args.query, results)
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("Query failed: %s", exc)
        return 1


# -----------------------------
# ask-local command
# -----------------------------
def _print_ask_result(result: dict[str, Any]) -> None:
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


def cmd_ask_local(args: argparse.Namespace) -> int:
    from src.rag.service import RAGService
    from src.rag.types import RAGConfig

    _load_dotenv()
    logger = _setup_logger("ask_local")
    try:
        index_path = Path(args.index_path).resolve()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index introuvable: {index_path}. Construisez d'abord l'index avec `python app.py build-index`."
            )

        config = RAGConfig(
            index_path=str(index_path),
            prompt_version=args.prompt_version,
            retriever_top_k=args.top_k,
        )
        service = RAGService(config=config, logger=logger)
        result = service.ask(args.query, top_k=args.top_k, debug=args.debug)
        payload = result.model_dump()
        _print_ask_result(payload)
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("ask-local failed: %s", exc)
        return 1


# -----------------------------
# evaluate-smoke command
# -----------------------------
def _load_eval_records(path: Path) -> list[dict[str, Any]]:
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


def _keyword_overlap(answer: str, expected_keywords: list[str]) -> float:
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


def _contains_expected_urls(source_urls: list[str], expected_urls: list[str]) -> bool | None:
    if not expected_urls:
        return None
    normalized_sources = {str(url).strip() for url in source_urls if str(url).strip()}
    normalized_expected = [str(url).strip() for url in expected_urls if str(url).strip()]
    if not normalized_expected:
        return None
    return all(url in normalized_sources for url in normalized_expected)


def _build_eval_service(args: argparse.Namespace, logger: logging.Logger) -> Any:
    from src.rag.llm import FakeLLMClient
    from src.rag.service import RAGService
    from src.rag.types import RAGConfig

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


def _evaluate_smoke(
    *,
    records: list[dict[str, Any]],
    service: Any,
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

        url_match = _contains_expected_urls(source_urls, expected_urls)
        if url_match is not None:
            with_expected_urls += 1
            if url_match:
                matched_urls += 1

        overlap = _keyword_overlap(payload.get("answer", ""), expected_keywords)
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


def _print_eval_summary(report: dict[str, Any], output_path: Path, offline: bool) -> None:
    print("\n=== Smoke Evaluation Summary ===")
    print(f"Mode offline                  : {offline}")
    print(f"Questions traitees            : {report['total_questions']}")
    print(f"Questions avec URLs attendues : {report['with_expected_urls']}")
    print(f"contains_expected_urls_rate   : {report['contains_expected_urls_rate']}")
    print(f"avg_keyword_overlap           : {report['avg_keyword_overlap']}")
    print(f"Rapport JSON                  : {output_path}")


def cmd_evaluate_smoke(args: argparse.Namespace) -> int:
    _load_dotenv()
    logger = _setup_logger("evaluate_smoke")

    try:
        eval_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()
        index_path = Path(args.index_path).resolve()

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index introuvable: {index_path}. Construire d'abord via `python app.py build-index`."
            )

        records = _load_eval_records(eval_path)
        service = _build_eval_service(args, logger)
        report = _evaluate_smoke(
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
        _print_eval_summary(report, output_path=output_path, offline=bool(args.offline))
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("evaluate-smoke failed: %s", exc)
        return 1


# -----------------------------
# run-api command
# -----------------------------
def cmd_run_api(args: argparse.Namespace) -> int:
    from src.api.app import create_app

    _load_dotenv()

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


# -----------------------------
# api-test command
# -----------------------------
def _pretty_print(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def _call_endpoint(
    *,
    method: str,
    url: str,
    timeout: int,
    json_payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    import requests

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


def cmd_api_test(args: argparse.Namespace) -> int:
    import requests

    _load_dotenv()

    base_url = args.base_url.rstrip("/")
    admin_token = args.admin_token.strip() or os.getenv("ADMIN_TOKEN", "").strip()

    try:
        status, payload = _call_endpoint(
            method="GET",
            url=f"{base_url}/health",
            timeout=args.timeout,
        )
        _pretty_print(f"GET /health [{status}]", payload)

        status, payload = _call_endpoint(
            method="GET",
            url=f"{base_url}/metadata",
            timeout=args.timeout,
        )
        _pretty_print(f"GET /metadata [{status}]", payload)

        if not args.offline:
            ask_payload = {
                "question": "Quels evenements jazz en Occitanie cette semaine ?",
                "top_k": 6,
                "debug": False,
            }
            status, payload = _call_endpoint(
                method="POST",
                url=f"{base_url}/ask",
                timeout=args.timeout,
                json_payload=ask_payload,
            )
            _pretty_print(f"POST /ask [{status}]", payload)
        else:
            print("\n[offline] /ask ignore (aucun appel generation live).")

        if admin_token:
            status, payload = _call_endpoint(
                method="POST",
                url=f"{base_url}/rebuild",
                timeout=args.timeout,
                json_payload={"mode": "reload"},
                headers={"X-ADMIN-TOKEN": admin_token},
            )
            _pretty_print(f"POST /rebuild mode=reload [{status}]", payload)
        else:
            print("\nADMIN_TOKEN absent: /rebuild non teste.")

        return 0
    except requests.RequestException as exc:
        print(f"API smoke test failed: {exc}")
        return 1


# -----------------------------
# bootstrap command
# -----------------------------
def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m{rem:05.2f}s"


def _run_step(
    *,
    name: str,
    command: Sequence[str],
    durations: list[tuple[str, float]],
    cwd: Path,
) -> None:
    print(f"\n=== {name} ===")
    print("$ " + " ".join(command))
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd)
    elapsed = time.perf_counter() - started
    durations.append((name, elapsed))

    if completed.returncode != 0:
        raise RuntimeError(f"Step failed ({name}) with exit code {completed.returncode}.")

    print(f"[OK] {name} termine en {_format_duration(elapsed)}")


def _wait_for_health(base_url: str, timeout_seconds: int) -> float:
    import requests

    health_url = f"{base_url.rstrip('/')}/health"
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last_error = ""

    while time.perf_counter() < deadline:
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return time.perf_counter() - started
            last_error = f"status={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)

    raise TimeoutError(f"API did not become healthy within {timeout_seconds}s ({last_error}).")


def _print_duration_summary(durations: list[tuple[str, float]]) -> None:
    print("\n=== Resume des durees ===")
    total = 0.0
    for name, elapsed in durations:
        total += elapsed
        print(f"- {name:<28} {_format_duration(elapsed)}")
    print(f"- {'TOTAL':<28} {_format_duration(total)}")


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _dataset_has_minimum_content(dataset_path: Path) -> tuple[bool, str]:
    import pandas as pd

    if not dataset_path.exists():
        return False, "missing file"

    try:
        suffix = dataset_path.suffix.lower()
        if suffix == ".parquet":
            dataframe = pd.read_parquet(dataset_path)
        elif suffix == ".jsonl":
            dataframe = pd.read_json(dataset_path, lines=True)
        elif suffix == ".csv":
            dataframe = pd.read_csv(dataset_path)
        else:
            return False, f"unsupported dataset format: {suffix}"
    except Exception as exc:
        return False, f"failed to read dataset: {exc}"

    if dataframe.empty:
        return False, "dataset is empty"
    if "document_text" not in dataframe.columns:
        return False, "missing required column: document_text"

    non_empty_docs = dataframe["document_text"].astype(str).str.strip().ne("").sum()
    if int(non_empty_docs) <= 0:
        return False, "all document_text values are empty"

    return True, ""


def cmd_bootstrap(args: argparse.Namespace) -> int:
    _load_dotenv()

    python_bin = sys.executable
    durations: list[tuple[str, float]] = []

    dataset_path = Path(args.dataset_path).resolve()
    index_path = Path(args.index_path).resolve()

    try:
        if not args.skip_env_check:
            _run_step(
                name="check_env",
                command=[python_bin, str(PROJECT_ROOT / "app.py"), "check-env"],
                durations=durations,
                cwd=PROJECT_ROOT,
            )

        should_build_dataset = False
        if not args.offline:
            dataset_ok, dataset_reason = _dataset_has_minimum_content(dataset_path)
            should_build_dataset = args.force_dataset or (not dataset_ok)
            if should_build_dataset:
                if dataset_path.exists() and not args.force_dataset:
                    print(
                        "\n[WARN] Dataset present but not usable, rebuild required: "
                        f"{dataset_reason} ({dataset_path})"
                    )
                _run_step(
                    name="build_dataset",
                    command=[
                        python_bin,
                        str(PROJECT_ROOT / "app.py"),
                        "build-dataset",
                        "--config",
                        str(Path(args.config).resolve()),
                        "--processed-output",
                        str(dataset_path),
                    ],
                    durations=durations,
                    cwd=PROJECT_ROOT,
                )
            else:
                print(f"\n[INFO] Dataset deja present, build_dataset saute ({dataset_path}).")
        else:
            print("\n[INFO] Mode offline actif: build_dataset saute.")

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Processed dataset introuvable. "
                f"Attendu: {dataset_path}. "
                "Lance build-dataset (ou retire --offline)."
            )

        dataset_ok, dataset_reason = _dataset_has_minimum_content(dataset_path)
        if not dataset_ok:
            raise RuntimeError(
                "Processed dataset is not usable for index build "
                f"({dataset_reason}) at {dataset_path}. "
                "Run bootstrap without --offline or use --force-dataset."
            )

        if not args.skip_index_build:
            _run_step(
                name="build_index",
                command=[
                    python_bin,
                    str(PROJECT_ROOT / "app.py"),
                    "build-index",
                    "--config",
                    str(Path(args.index_config).resolve()),
                    "--input",
                    str(dataset_path),
                    "--output",
                    str(index_path),
                ],
                durations=durations,
                cwd=PROJECT_ROOT,
            )
        else:
            print("\n[INFO] build_index saute (--skip-index-build).")

        if args.prepare_only:
            _print_duration_summary(durations)
            print("\nPreparation terminee. API non demarree (--prepare-only).")
            return 0

        run_api_cmd = [
            python_bin,
            str(PROJECT_ROOT / "app.py"),
            "run-api",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--log-level",
            args.log_level,
        ]

        print("\n=== run_api ===")
        print("$ " + " ".join(run_api_cmd))
        api_process = subprocess.Popen(run_api_cmd, cwd=PROJECT_ROOT)

        try:
            startup_elapsed = _wait_for_health(
                base_url=f"http://{args.host}:{args.port}",
                timeout_seconds=args.startup_timeout,
            )
        except Exception:
            _terminate_process(api_process)
            raise

        durations.append(("run_api_startup", startup_elapsed))
        print(f"[OK] API healthy en {_format_duration(startup_elapsed)}")

        if not args.skip_api_smoke:
            smoke_cmd = [
                python_bin,
                str(PROJECT_ROOT / "app.py"),
                "api-test",
                "--base-url",
                f"http://{args.host}:{args.port}",
            ]
            if not os.getenv("MISTRAL_API_KEY", "").strip():
                smoke_cmd.append("--offline")

            admin_token = os.getenv("ADMIN_TOKEN", "").strip()
            if admin_token:
                smoke_cmd.extend(["--admin-token", admin_token])

            _run_step(
                name="api_smoke",
                command=smoke_cmd,
                durations=durations,
                cwd=PROJECT_ROOT,
            )
        else:
            print("\n[INFO] api-test saute (--skip-api-smoke).")

        _print_duration_summary(durations)

        if args.exit_after_smoke:
            _terminate_process(api_process)
            print("\nAPI arretee (--exit-after-smoke).")
            return 0

        print("\nApp operationnelle. API en cours d'execution.")
        print("Arret: Ctrl+C")

        try:
            while True:
                return_code = api_process.poll()
                if return_code is not None:
                    print(f"\nAPI terminee avec code {return_code}.")
                    return return_code
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nArret demande, fermeture de l'API...")
            _terminate_process(api_process)
            return 0

    except Exception as exc:  # pragma: no cover
        print(f"\n[ERROR] {exc}")
        _print_duration_summary(durations)
        return 1


# -----------------------------
# step6-docker-demo command
# -----------------------------
def _run_external_command(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    print("$ " + " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({name}) with exit code {result.returncode}.")


def cmd_step6_docker_demo(args: argparse.Namespace) -> int:
    python_bin = args.python_bin or sys.executable
    image_name = args.image_name

    try:
        print("== Step 6 demo bootstrap ==")
        print(f"root: {PROJECT_ROOT}")

        _run_external_command(
            "build_index",
            [
                python_bin,
                str(PROJECT_ROOT / "app.py"),
                "build-index",
                "--config",
                str(Path(args.index_config).resolve()),
                "--input",
                str(Path(args.input_dataset).resolve()),
                "--output",
                str(Path(args.output_index).resolve()),
            ],
        )

        _run_external_command("docker_build", ["docker", "build", "-t", image_name, str(PROJECT_ROOT)])

        docker_run_cmd = [
            "docker",
            "run",
            "--rm",
            "-p",
            f"{args.port}:8000",
            "--env-file",
            str(PROJECT_ROOT / ".env"),
            "-e",
            f"HOST={args.host}",
            "-e",
            "PORT=8000",
            "-v",
            f"{PROJECT_ROOT / 'artifacts'}:/app/artifacts",
            "-v",
            f"{PROJECT_ROOT / 'data'}:/app/data",
            "-v",
            f"{PROJECT_ROOT / 'logs'}:/app/logs",
            image_name,
        ]
        _run_external_command("docker_run", docker_run_cmd)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}")
        return 1


# -----------------------------
# parser wiring
# -----------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for Puls-Events project (data, index, RAG, API, bootstrap)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_check = subparsers.add_parser("check-env", help="Run environment smoke test.")
    p_check.set_defaults(func=cmd_check_env)

    p_dataset = subparsers.add_parser("build-dataset", help="Build OpenAgenda dataset.")
    p_dataset.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Path to YAML config.")
    p_dataset.add_argument(
        "--raw-output",
        default=str(PROJECT_ROOT / "data/raw/events_raw.jsonl"),
        help="Path to raw JSONL output.",
    )
    p_dataset.add_argument(
        "--processed-output",
        default=str(PROJECT_ROOT / "data/processed/events_processed.parquet"),
        help="Path to processed Parquet output.",
    )
    p_dataset.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs/build_dataset.log"),
        help="Path to log file.",
    )
    p_dataset.set_defaults(func=cmd_build_dataset)

    p_index = subparsers.add_parser("build-index", help="Build FAISS index from processed dataset.")
    p_index.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Path to indexing YAML config.",
    )
    p_index.add_argument("--input", default="", help="Optional input dataset override.")
    p_index.add_argument("--output", default="", help="Optional output directory override.")
    p_index.add_argument(
        "--provider",
        default="",
        choices=["", "huggingface", "mistral"],
        help="Optional embedding provider override.",
    )
    p_index.add_argument("--embedding-model", default="", help="Optional embedding model override.")
    p_index.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs/build_index.log"),
        help="Log file path.",
    )
    p_index.set_defaults(func=cmd_build_index)

    p_query = subparsers.add_parser("query-index", help="Run similarity search on local FAISS index.")
    p_query.add_argument("--query", required=True, help="User query to search.")
    p_query.add_argument("--k", type=int, default=5, help="Top-k results.")
    p_query.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Path to indexing YAML config.",
    )
    p_query.add_argument("--index-dir", default="", help="Optional index directory override.")
    p_query.set_defaults(func=cmd_query_index)

    p_ask = subparsers.add_parser("ask-local", help="Ask one question to local RAG engine.")
    p_ask.add_argument("--query", required=True, help="Question utilisateur.")
    p_ask.add_argument("--top_k", type=int, default=6, help="Nombre de chunks recuperes.")
    p_ask.add_argument("--debug", action="store_true", help="Affiche les metadonnees debug.")
    p_ask.add_argument("--index_path", default="artifacts/faiss_index", help="Dossier index FAISS.")
    p_ask.add_argument("--prompt_version", default="v1", help="Version du prompt.")
    p_ask.set_defaults(func=cmd_ask_local)

    p_eval = subparsers.add_parser("evaluate-smoke", help="Run smoke evaluation for RAG.")
    p_eval.add_argument("--input", default="data/eval/smoke_eval.jsonl", help="Input smoke eval jsonl.")
    p_eval.add_argument("--output", default="reports/smoke_eval_report.json", help="Output report JSON.")
    p_eval.add_argument("--index_path", default="artifacts/faiss_index", help="Path to FAISS index.")
    p_eval.add_argument("--top_k", type=int, default=6, help="Top-k retrieval.")
    p_eval.add_argument("--offline", action="store_true", help="Use FakeLLM without network.")
    p_eval.add_argument("--limit", type=int, default=0, help="Limit number of questions.")
    p_eval.set_defaults(func=cmd_evaluate_smoke)

    p_run_api = subparsers.add_parser("run-api", help="Run Flask API locally.")
    p_run_api.add_argument("--host", default="", help="Host override (default from env HOST).")
    p_run_api.add_argument("--port", type=int, default=0, help="Port override (default from env PORT).")
    p_run_api.add_argument("--log-level", default="", help="Logging level override.")
    p_run_api.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    p_run_api.set_defaults(func=cmd_run_api)

    p_api_test = subparsers.add_parser("api-test", help="Smoke test API endpoints.")
    p_api_test.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL.")
    p_api_test.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    p_api_test.add_argument("--offline", action="store_true", help="Skip /ask request.")
    p_api_test.add_argument("--admin-token", default="", help="Optional token for /rebuild request.")
    p_api_test.set_defaults(func=cmd_api_test)

    p_bootstrap = subparsers.add_parser("bootstrap", help="End-to-end local bootstrap.")
    p_bootstrap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Dataset config file.")
    p_bootstrap.add_argument(
        "--index-config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Index config file.",
    )
    p_bootstrap.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET),
        help="Processed dataset path used by index build.",
    )
    p_bootstrap.add_argument("--index-path", default=str(DEFAULT_INDEX), help="FAISS index directory.")
    p_bootstrap.add_argument("--host", default="127.0.0.1", help="API host")
    p_bootstrap.add_argument("--port", type=int, default=8000, help="API port")
    p_bootstrap.add_argument("--log-level", default="INFO", help="API log level")
    p_bootstrap.add_argument("--offline", action="store_true", help="Skip dataset fetch/build.")
    p_bootstrap.add_argument("--force-dataset", action="store_true", help="Force dataset rebuild.")
    p_bootstrap.add_argument("--skip-env-check", action="store_true", help="Skip check-env.")
    p_bootstrap.add_argument("--skip-index-build", action="store_true", help="Skip build-index.")
    p_bootstrap.add_argument("--prepare-only", action="store_true", help="Prepare without starting API.")
    p_bootstrap.add_argument("--skip-api-smoke", action="store_true", help="Skip api-test.")
    p_bootstrap.add_argument(
        "--exit-after-smoke",
        action="store_true",
        help="Start API, run smoke test, then stop API and exit.",
    )
    p_bootstrap.add_argument(
        "--startup-timeout",
        type=int,
        default=45,
        help="Seconds to wait for API /health readiness.",
    )
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_step6 = subparsers.add_parser("step6-docker-demo", help="Run step 6 docker demo workflow.")
    p_step6.add_argument("--python-bin", default=sys.executable, help="Python binary to invoke app.py.")
    p_step6.add_argument("--image-name", default="puls-events-rag:step6", help="Docker image name.")
    p_step6.add_argument("--port", type=int, default=8000, help="Host port bound to container 8000.")
    p_step6.add_argument("--host", default="0.0.0.0", help="HOST env passed to container.")
    p_step6.add_argument(
        "--index-config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Indexing config path.",
    )
    p_step6.add_argument(
        "--input-dataset",
        default=str(PROJECT_ROOT / "data/processed/events_processed.parquet"),
        help="Input dataset for index build.",
    )
    p_step6.add_argument(
        "--output-index",
        default=str(PROJECT_ROOT / "artifacts/faiss_index"),
        help="Output FAISS index directory.",
    )
    p_step6.set_defaults(func=cmd_step6_docker_demo)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
