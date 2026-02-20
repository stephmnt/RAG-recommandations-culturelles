"""Cleaning and normalization pipeline for OpenAgenda events."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from hashlib import sha1
from typing import Any

from src.preprocess.schema import validate_record

LOGGER = logging.getLogger(__name__)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_url(value: Any, language: str = "fr") -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, dict):
        candidate = _pick_localized_text(value, language=language)
        text = candidate.strip()
    else:
        text = str(value).strip()

    if text in {"{}", "[]", "None", "null", "#"}:
        return ""

    # Keep only URL-like values; avoid persisting dict string dumps as URLs.
    if "://" in text or text.startswith("/"):
        return text
    return ""


def _pick_localized_text(value: Any, language: str = "fr") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        preferred = value.get(language)
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
        for candidate in value.values():
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    if isinstance(value, list):
        chunks = [_pick_localized_text(item, language) for item in value]
        return " ".join(chunk for chunk in chunks if chunk).strip()
    return str(value).strip()


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def normalize_iso_datetime(value: Any) -> str:
    dt = _parse_datetime(value)
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError(f"Unable to parse date value: {value}")
    return parsed.date()


def _extract_start_end(raw_event: dict[str, Any]) -> tuple[str, str]:
    first_timing = raw_event.get("firstTiming")
    if not isinstance(first_timing, dict):
        first_timing = {}

    timings = raw_event.get("timings")
    first_list_timing: dict[str, Any] = {}
    if isinstance(timings, list) and timings and isinstance(timings[0], dict):
        first_list_timing = timings[0]

    start_candidates = [
        raw_event.get("start"),
        raw_event.get("begin"),
        raw_event.get("startDate"),
        first_timing.get("begin"),
        first_list_timing.get("begin"),
    ]
    end_candidates = [
        raw_event.get("end"),
        raw_event.get("endDate"),
        first_timing.get("end"),
        first_list_timing.get("end"),
    ]

    start_datetime = ""
    for candidate in start_candidates:
        start_datetime = normalize_iso_datetime(candidate)
        if start_datetime:
            break

    end_datetime = ""
    for candidate in end_candidates:
        end_datetime = normalize_iso_datetime(candidate)
        if end_datetime:
            break

    return start_datetime, end_datetime


def _normalize_tags(raw_tags: Any, language: str = "fr") -> list[str]:
    if raw_tags in (None, ""):
        return []
    if isinstance(raw_tags, str):
        return [token.strip() for token in raw_tags.split(",") if token.strip()]

    if isinstance(raw_tags, list):
        normalized: list[str] = []
        for item in raw_tags:
            if isinstance(item, dict):
                label = (
                    _pick_localized_text(item.get("label"), language)
                    or _pick_localized_text(item.get("name"), language)
                    or _pick_localized_text(item, language)
                )
            else:
                label = _pick_localized_text(item, language)
            if label:
                normalized.append(label)
        return normalized

    if isinstance(raw_tags, dict):
        label = _pick_localized_text(raw_tags, language)
        return [label] if label else []

    return []


def _normalize_location(
    raw_event: dict[str, Any], language: str = "fr"
) -> tuple[str, str, str, float | None, float | None]:
    location = raw_event.get("location")
    if not isinstance(location, dict):
        location = {}

    location_name = (
        _pick_localized_text(location.get("name"), language)
        or _pick_localized_text(raw_event.get("locationName"), language)
        or _pick_localized_text(raw_event.get("venue"), language)
    )
    address = (
        _pick_localized_text(location.get("address"), language)
        or _pick_localized_text(raw_event.get("address"), language)
    )
    city = (
        _pick_localized_text(location.get("city"), language)
        or _pick_localized_text(raw_event.get("city"), language)
    )
    latitude = _to_optional_float(
        location.get("latitude") or location.get("lat") or raw_event.get("latitude")
    )
    longitude = _to_optional_float(
        location.get("longitude") or location.get("lng") or raw_event.get("longitude")
    )
    return city, location_name, address, latitude, longitude


def _resolve_event_id(
    raw_event: dict[str, Any],
    *,
    title: str,
    start_datetime: str,
    city: str,
    url: str,
) -> str:
    for candidate_key in ("uid", "id", "uuid", "slug"):
        value = raw_event.get(candidate_key)
        if value not in (None, ""):
            return str(value).strip()
    if url:
        return url
    fingerprint = sha1(f"{title}|{start_datetime}|{city}".encode("utf-8")).hexdigest()
    return f"openagenda-{fingerprint[:16]}"


def build_document_text(record: dict[str, Any]) -> str:
    parts = [
        f"Titre: {record.get('title', '')}".strip(),
        f"Description: {record.get('description', '')}".strip(),
        f"Lieu: {record.get('location_name', '')}".strip(),
        f"Ville: {record.get('city', '')}".strip(),
        f"Date debut: {record.get('start_datetime', '')}".strip(),
    ]
    return "\n".join(part for part in parts if ":" in part and part.split(":", 1)[1].strip())


def _build_retrieval_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": record.get("event_id", ""),
        "city": record.get("city", ""),
        "start_datetime": record.get("start_datetime", ""),
        "end_datetime": record.get("end_datetime", ""),
        "source": record.get("source", "openagenda"),
        "url": record.get("url", ""),
        "tags": record.get("tags", []),
    }


def select_and_normalize_fields(
    raw_event: dict[str, Any],
    *,
    language: str = "fr",
    source: str = "openagenda",
) -> dict[str, Any]:
    title = _pick_localized_text(raw_event.get("title"), language)
    description = _pick_localized_text(raw_event.get("description"), language)
    start_datetime, end_datetime = _extract_start_end(raw_event)
    city, location_name, address, latitude, longitude = _normalize_location(
        raw_event, language=language
    )
    url = _normalize_url(
        raw_event.get("canonicalUrl")
        or raw_event.get("url")
        or raw_event.get("link"),
        language=language,
    )
    tags = _normalize_tags(raw_event.get("tags"), language)

    event_id = _resolve_event_id(
        raw_event,
        title=title,
        start_datetime=start_datetime,
        city=city,
        url=url,
    )

    record = {
        "event_id": event_id,
        "title": title,
        "description": description,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "city": city,
        "location_name": location_name,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "url": url,
        "tags": tags,
        "source": source,
        "document_text": "",
        "retrieval_metadata": {},
    }
    record["document_text"] = build_document_text(record)
    record["retrieval_metadata"] = _build_retrieval_metadata(record)
    return record


def _is_in_period(start_datetime: str, start_date: date, end_date: date) -> bool:
    parsed = _parse_datetime(start_datetime)
    if parsed is None:
        return False
    event_date = parsed.date()
    return start_date <= event_date <= end_date


def filter_events_by_period(
    records: list[dict[str, Any]],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> list[dict[str, Any]]:
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    return [
        record
        for record in records
        if _is_in_period(_clean_text(record.get("start_datetime")), start, end)
    ]


def _dedupe_key(record: dict[str, Any]) -> str:
    event_id = _clean_text(record.get("event_id"))
    if event_id:
        return f"id::{event_id}"
    url = _clean_text(record.get("url"))
    if url:
        return f"url::{url}"
    return "fallback::{title}::{start}::{city}".format(
        title=_clean_text(record.get("title")),
        start=_clean_text(record.get("start_datetime")),
        city=_clean_text(record.get("city")),
    )


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0

    for record in records:
        key = _dedupe_key(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduplicated.append(record)

    return deduplicated, duplicates


def clean_events(
    *,
    raw_events: list[dict[str, Any]],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    language: str = "fr",
    source: str = "openagenda",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clean, filter, deduplicate and validate OpenAgenda events."""

    start = _coerce_date(start_date)
    end = _coerce_date(end_date)

    stats = {
        "raw_events": len(raw_events),
        "missing_required": 0,
        "outside_period": 0,
        "after_period_filter": 0,
        "duplicates_removed": 0,
        "invalid_records": 0,
        "processed_events": 0,
    }

    filtered: list[dict[str, Any]] = []
    for raw_event in raw_events:
        record = select_and_normalize_fields(raw_event, language=language, source=source)

        if not record["title"] or not record["start_datetime"]:
            stats["missing_required"] += 1
            continue

        if not _is_in_period(record["start_datetime"], start, end):
            stats["outside_period"] += 1
            continue

        filtered.append(record)

    stats["after_period_filter"] = len(filtered)

    deduplicated, duplicates = deduplicate_records(filtered)
    stats["duplicates_removed"] = duplicates

    validated_records: list[dict[str, Any]] = []
    for record in deduplicated:
        try:
            event = validate_record(record)
        except ValueError as exc:
            stats["invalid_records"] += 1
            LOGGER.error(
                "cleaning.validation_error event_id=%s error=%s",
                record.get("event_id"),
                exc,
            )
            continue
        validated_records.append(event.to_dict())

    stats["processed_events"] = len(validated_records)
    return validated_records, stats
