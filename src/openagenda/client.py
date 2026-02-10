"""OpenAgenda HTTP client with pagination and retry support."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class OpenAgendaConfig:
    base_url: str = "https://api.openagenda.com/v2/events"
    api_key: str = ""
    api_key_param: str = "key"
    department: str = ""
    city: str = ""
    latitude: float | None = None
    longitude: float | None = None
    radius_km: int | None = None
    start_date: str = ""
    end_date: str = ""
    language: str = "fr"
    page_size: int = 100
    max_pages: int = 20
    max_events: int = 1000
    timeout_seconds: int = 20
    retry_attempts: int = 3
    backoff_seconds: float = 1.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "OpenAgendaConfig":
        config = raw_config.get("openagenda", raw_config)
        location = config.get("location", {})
        time_window = config.get("time_window", {})
        request_config = config.get("request", {})
        pagination = config.get("pagination", {})
        auth = config.get("auth", {})
        extra_params = config.get("filters", {})
        if not isinstance(extra_params, dict):
            extra_params = {}
        radius_raw = location.get("radius_km")
        radius_km = _to_int(radius_raw, 20) if radius_raw not in (None, "") else None

        return cls(
            base_url=str(config.get("base_url", cls.base_url)),
            api_key=(auth.get("api_key") or config.get("api_key") or "").strip(),
            api_key_param=str(auth.get("api_key_param", "key")).strip() or "key",
            department=str(location.get("department", "")).strip(),
            city=str(location.get("city", "")).strip(),
            latitude=_to_float(location.get("latitude")),
            longitude=_to_float(location.get("longitude")),
            radius_km=radius_km,
            start_date=str(time_window.get("start_date", "")).strip(),
            end_date=str(time_window.get("end_date", "")).strip(),
            language=str(request_config.get("language", "fr")).strip() or "fr",
            page_size=_to_int(pagination.get("page_size"), 100),
            max_pages=_to_int(pagination.get("max_pages"), 20),
            max_events=_to_int(pagination.get("max_events"), 1000),
            timeout_seconds=_to_int(request_config.get("timeout_seconds"), 20),
            retry_attempts=_to_int(request_config.get("retry_attempts"), 3),
            backoff_seconds=float(request_config.get("backoff_seconds", 1.0)),
            extra_params=extra_params,
        )


def build_query_params(config: OpenAgendaConfig, offset: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "offset": offset,
        "size": config.page_size,
        "lang": config.language,
    }

    if config.api_key:
        params[config.api_key_param] = config.api_key
    if config.department:
        params["department"] = config.department
    if config.city:
        params["city"] = config.city
    if config.latitude is not None:
        params["latitude"] = config.latitude
    if config.longitude is not None:
        params["longitude"] = config.longitude
    if config.radius_km is not None:
        params["radius_km"] = config.radius_km
    if config.start_date:
        params["start"] = config.start_date
    if config.end_date:
        params["end"] = config.end_date

    for key, value in config.extra_params.items():
        if value is None:
            continue
        params[key] = value

    return params


def _extract_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for candidate_key in ("events", "results", "items"):
        candidate = payload.get(candidate_key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _request_page(
    *,
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout_seconds: int,
    retry_attempts: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    for attempt in range(1, retry_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)
        except requests.RequestException as exc:
            LOGGER.error(
                "openagenda.request_error attempt=%s/%s error=%s",
                attempt,
                retry_attempts,
                exc,
            )
            if attempt >= retry_attempts:
                raise RuntimeError(f"OpenAgenda request failed after retries: {exc}") from exc
            time.sleep(backoff_seconds * attempt)
            continue

        if 500 <= response.status_code < 600:
            LOGGER.error(
                "openagenda.server_error attempt=%s/%s status=%s",
                attempt,
                retry_attempts,
                response.status_code,
            )
            if attempt >= retry_attempts:
                raise RuntimeError(
                    f"OpenAgenda server error after retries (HTTP {response.status_code})."
                )
            time.sleep(backoff_seconds * attempt)
            continue

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAgenda request failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenAgenda response is not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("OpenAgenda response must be a JSON object.")
        return payload

    raise RuntimeError("Unexpected request retry flow termination.")


def fetch_events(
    config: OpenAgendaConfig,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch raw events from OpenAgenda with pagination and retry."""

    if not config.api_key:
        LOGGER.info("OPENAGENDA_API_KEY missing in runtime config. Request may return 401/403.")

    own_session = session is None
    session = session or requests.Session()

    events: list[dict[str, Any]] = []
    offset = 0

    try:
        for page_number in range(1, config.max_pages + 1):
            if len(events) >= config.max_events:
                break

            params = build_query_params(config, offset=offset)
            LOGGER.info(
                "openagenda.page_start page=%s offset=%s size=%s",
                page_number,
                offset,
                config.page_size,
            )

            payload = _request_page(
                session=session,
                url=config.base_url,
                params=params,
                timeout_seconds=config.timeout_seconds,
                retry_attempts=config.retry_attempts,
                backoff_seconds=config.backoff_seconds,
            )
            page_events = _extract_events(payload)

            if not page_events:
                LOGGER.info("openagenda.page_empty page=%s", page_number)
                break

            remaining = config.max_events - len(events)
            events.extend(page_events[:remaining])

            LOGGER.info(
                "openagenda.page_done page=%s page_events=%s total_events=%s",
                page_number,
                len(page_events),
                len(events),
            )

            if len(page_events) < config.page_size:
                break

            offset += len(page_events)

    finally:
        if own_session:
            session.close()

    return events
