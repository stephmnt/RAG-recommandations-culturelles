#!/usr/bin/env python3
"""Build OpenAgenda dataset for Step 2 (raw + cleaned)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.openagenda.client import OpenAgendaConfig, fetch_events_with_stats
from src.preprocess.cleaning import clean_events
from src.preprocess.schema import EVENT_RECORD_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAgenda dataset.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--raw-output",
        default=str(PROJECT_ROOT / "data/raw/events_raw.jsonl"),
        help="Path to raw JSONL output.",
    )
    parser.add_argument(
        "--processed-output",
        default=str(PROJECT_ROOT / "data/processed/events_processed.parquet"),
        help="Path to processed Parquet output.",
    )
    parser.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs/build_dataset.log"),
        help="Path to log file.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_dataset")
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


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("config.yaml must contain a mapping at root level.")
    return payload


def apply_env_overrides(config: dict[str, Any]) -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    openagenda_key = os.getenv("OPENAGENDA_API_KEY", "").strip()
    if openagenda_key:
        config.setdefault("openagenda", {}).setdefault("auth", {})["api_key"] = openagenda_key


def resolve_time_window(config: dict[str, Any]) -> tuple[str, str]:
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


def write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_parquet(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        dataframe = pd.DataFrame(records)
    else:
        dataframe = pd.DataFrame(columns=EVENT_RECORD_FIELDS)
    dataframe.to_parquet(output_path, index=False)


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Build summary ===")
    print(f"Agendas found            : {summary['agendas_found']}")
    print(f"Agendas scanned          : {summary['agendas_scanned']}")
    print(f"Legacy fallback used     : {summary['legacy_fallback_used']}")
    print(f"Raw events fetched        : {summary['raw_events']}")
    print(f"After period filtering    : {summary['after_period_filter']}")
    print(f"Duplicates removed        : {summary['duplicates_removed']}")
    print(f"Invalid records dropped   : {summary['invalid_records']}")
    print(f"Final processed records   : {summary['processed_events']}")
    print(f"Events by agenda          : {summary['events_by_agenda']}")
    print(f"Raw output                : {summary['raw_output']}")
    print(f"Processed output          : {summary['processed_output']}")
    print(f"Log file                  : {summary['log_file']}")


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file).resolve()
    logger = setup_logging(log_path)

    try:
        config_path = Path(args.config).resolve()
        raw_output = Path(args.raw_output).resolve()
        processed_output = Path(args.processed_output).resolve()

        logger.info("Loading configuration from %s", config_path)
        config = load_config(config_path)
        apply_env_overrides(config)
        start_date, end_date = resolve_time_window(config)

        logger.info("Using time window start_date=%s end_date=%s", start_date, end_date)
        oa_config = OpenAgendaConfig.from_dict(config)
        oa_config.start_date = start_date
        oa_config.end_date = end_date

        if not oa_config.api_key:
            logger.warning(
                "OPENAGENDA_API_KEY is missing. Requests are likely to fail with HTTP 401/403."
            )

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
        write_jsonl(raw_output, raw_events)
        logger.info("Raw events written to %s", raw_output)

        logger.info("Cleaning and validating events")
        records, stats = clean_events(
            raw_events=raw_events,
            start_date=start_date,
            end_date=end_date,
            language=oa_config.language,
            source="openagenda",
        )
        write_parquet(processed_output, records)
        logger.info("Processed events written to %s", processed_output)

        summary = {
            "agendas_found": ingestion_stats.get("agendas_found", 0),
            "agendas_scanned": ingestion_stats.get("agendas_scanned", 0),
            "legacy_fallback_used": ingestion_stats.get("legacy_fallback_used", False),
            "raw_events": len(raw_events),
            "after_period_filter": stats["after_period_filter"],
            "duplicates_removed": stats["duplicates_removed"],
            "invalid_records": stats["invalid_records"],
            "processed_events": stats["processed_events"],
            "events_by_agenda": ingestion_stats.get("events_by_agenda", {}),
            "raw_output": str(raw_output),
            "processed_output": str(processed_output),
            "log_file": str(log_path),
        }
        logger.info("Build summary: %s", summary)
        print_summary(summary)
        return 0

    except Exception as exc:  # pragma: no cover - CLI entrypoint guard
        logger.exception("Dataset build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
