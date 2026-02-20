"""OpenAgenda HTTP client with agenda discovery, pagination and retry support."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                output.append(text)
        return output
    text = str(value).strip()
    return [text] if text else []


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
    return str(value).strip()


def _normalize_for_compare(value: Any) -> str:
    return str(value or "").strip().casefold()


def _to_openagenda_timing(value: str, *, end_of_day: bool) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    # Common case in config: YYYY-MM-DD.
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        if end_of_day:
            return f"{text}T23:59:59.999Z"
        return f"{text}T00:00:00.000Z"

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class OpenAgendaConfig:
    base_url: str = "https://api.openagenda.com/v2"
    api_key: str = ""
    api_key_param: str = "key"
    send_api_key_in_query: bool = False

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

    agenda_uids: list[str] = field(default_factory=list)
    agenda_search_queries: list[str] = field(default_factory=list)
    max_agendas: int = 20
    agenda_page_size: int = 50
    agenda_max_pages: int = 5

    timeout_seconds: int = 20
    retry_attempts: int = 3
    backoff_seconds: float = 1.0

    extra_params: dict[str, Any] = field(default_factory=dict)
    agenda_extra_params: dict[str, Any] = field(default_factory=dict)

    allow_legacy_events_fallback: bool = True

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "OpenAgendaConfig":
        config = raw_config.get("openagenda", raw_config)
        location = config.get("location", {})
        time_window = config.get("time_window", {})
        request_config = config.get("request", {})
        pagination = config.get("pagination", {})
        auth = config.get("auth", {})
        agenda_search = config.get("agenda_search", {})

        event_filters = config.get("filters", {})
        if not isinstance(event_filters, dict):
            event_filters = {}

        agenda_extra_params = agenda_search.get("extra_params", {})
        if not isinstance(agenda_extra_params, dict):
            agenda_extra_params = {}

        radius_raw = location.get("radius_km")
        radius_km = _to_int(radius_raw, 20) if radius_raw not in (None, "") else None

        return cls(
            base_url=str(config.get("base_url", cls.base_url)).strip() or cls.base_url,
            api_key=(auth.get("api_key") or config.get("api_key") or "").strip(),
            api_key_param=str(auth.get("api_key_param", "key")).strip() or "key",
            send_api_key_in_query=_to_bool(auth.get("send_api_key_in_query"), default=False),
            department=str(location.get("department", "")).strip(),
            city=str(location.get("city", "")).strip(),
            latitude=_to_float(location.get("latitude")),
            longitude=_to_float(location.get("longitude")),
            radius_km=radius_km,
            start_date=str(time_window.get("start_date", "")).strip(),
            end_date=str(time_window.get("end_date", "")).strip(),
            language=str(request_config.get("language", "fr")).strip() or "fr",
            page_size=max(1, _to_int(pagination.get("page_size"), 100)),
            max_pages=max(1, _to_int(pagination.get("max_pages"), 20)),
            max_events=max(1, _to_int(pagination.get("max_events"), 1000)),
            agenda_uids=_to_string_list(config.get("agenda_uids", [])),
            agenda_search_queries=_to_string_list(agenda_search.get("queries", [])),
            max_agendas=max(1, _to_int(agenda_search.get("max_agendas"), 20)),
            agenda_page_size=max(1, _to_int(agenda_search.get("page_size"), 50)),
            agenda_max_pages=max(1, _to_int(agenda_search.get("max_pages"), 5)),
            timeout_seconds=max(1, _to_int(request_config.get("timeout_seconds"), 20)),
            retry_attempts=max(1, _to_int(request_config.get("retry_attempts"), 3)),
            backoff_seconds=float(request_config.get("backoff_seconds", 1.0)),
            extra_params=event_filters,
            agenda_extra_params=agenda_extra_params,
            allow_legacy_events_fallback=_to_bool(
                config.get("allow_legacy_events_fallback"),
                default=True,
            ),
        )

    @property
    def api_root(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/events"):
            base = base[: -len("/events")]
        marker = "/agendas/"
        if marker in base:
            base = base.split(marker, 1)[0]
        return base.rstrip("/")

    @property
    def agendas_url(self) -> str:
        return f"{self.api_root}/agendas"

    def agenda_events_url(self, agenda_uid: str) -> str:
        return f"{self.api_root}/agendas/{agenda_uid}/events"

    @property
    def uses_legacy_events_endpoint(self) -> bool:
        return self.base_url.rstrip("/").endswith("/events")


def _build_request_headers(config: OpenAgendaConfig) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if config.api_key:
        headers["key"] = config.api_key
    return headers


def _apply_query_auth_if_needed(config: OpenAgendaConfig, params: dict[str, Any]) -> None:
    if config.send_api_key_in_query and config.api_key:
        params[config.api_key_param] = config.api_key


def build_agenda_query_params(
    config: OpenAgendaConfig,
    from_cursor: int,
    query: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "from": from_cursor,
        "size": config.agenda_page_size,
        "lang": config.language,
    }
    if query:
        params["q"] = query

    for key, value in config.agenda_extra_params.items():
        if value is None:
            continue
        params[key] = value

    _apply_query_auth_if_needed(config, params)
    return params


def build_event_query_params(config: OpenAgendaConfig, from_cursor: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "from": from_cursor,
        "size": config.page_size,
        "lang": config.language,
        "detailed": 1,
    }

    if config.start_date:
        start_timing = _to_openagenda_timing(config.start_date, end_of_day=False)
        if start_timing:
            params["timings[gte]"] = start_timing
    if config.end_date:
        end_timing = _to_openagenda_timing(config.end_date, end_of_day=True)
        if end_timing:
            params["timings[lte]"] = end_timing

    for key, value in config.extra_params.items():
        if value is None:
            continue
        params[key] = value

    _apply_query_auth_if_needed(config, params)
    return params


def build_legacy_events_query_params(config: OpenAgendaConfig, offset: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "offset": offset,
        "size": config.page_size,
        "lang": config.language,
    }
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

    _apply_query_auth_if_needed(config, params)
    return params


def _extract_objects(payload: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _extract_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_objects(payload, ("events", "results", "items"))


def _extract_agendas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_objects(payload, ("agendas", "results", "items"))


def _extract_agenda_uid(agenda: dict[str, Any]) -> str:
    for key in ("uid", "slug", "id"):
        value = agenda.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _extract_agenda_title(agenda: dict[str, Any], language: str = "fr") -> str:
    for key in ("title", "name", "label"):
        value = agenda.get(key)
        title = _pick_localized_text(value, language=language)
        if title:
            return title
    return ""


def _extract_event_lat_lon(raw_event: dict[str, Any]) -> tuple[float | None, float | None]:
    location = raw_event.get("location")
    if not isinstance(location, dict):
        location = {}

    latitude = _to_float(
        location.get("latitude")
        or location.get("lat")
        or raw_event.get("latitude")
        or raw_event.get("lat")
    )
    longitude = _to_float(
        location.get("longitude")
        or location.get("lng")
        or location.get("lon")
        or raw_event.get("longitude")
        or raw_event.get("lng")
        or raw_event.get("lon")
    )
    return latitude, longitude


def _extract_event_city(raw_event: dict[str, Any], language: str = "fr") -> str:
    location = raw_event.get("location")
    if not isinstance(location, dict):
        location = {}
    return (
        _pick_localized_text(location.get("city"), language=language)
        or _pick_localized_text(raw_event.get("city"), language=language)
    )


def _extract_event_department(raw_event: dict[str, Any], language: str = "fr") -> str:
    location = raw_event.get("location")
    if not isinstance(location, dict):
        location = {}
    return (
        _pick_localized_text(location.get("department"), language=language)
        or _pick_localized_text(raw_event.get("department"), language=language)
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Local import avoids adding an unconditional dependency at module import time.
    from math import asin, cos, radians, sin, sqrt

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    a = sin(d_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius_km = 6371.0088
    return earth_radius_km * c


def _event_matches_geo_scope(config: OpenAgendaConfig, raw_event: dict[str, Any]) -> bool:
    if (
        config.latitude is None
        or config.longitude is None
        or config.radius_km is None
        or config.radius_km <= 0
    ):
        return True

    event_lat, event_lon = _extract_event_lat_lon(raw_event)
    if event_lat is not None and event_lon is not None:
        return _haversine_km(config.latitude, config.longitude, event_lat, event_lon) <= float(
            config.radius_km
        )

    # Fallback when coordinates are missing: use city/department textual hints.
    event_city = _normalize_for_compare(_extract_event_city(raw_event, language=config.language))
    expected_city = _normalize_for_compare(config.city)
    if expected_city and event_city == expected_city:
        return True

    event_department = _normalize_for_compare(
        _extract_event_department(raw_event, language=config.language)
    )
    expected_department = _normalize_for_compare(config.department)
    if expected_department and expected_department in event_department:
        return True

    return False


def _request_page(
    *,
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    retry_attempts: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    for attempt in range(1, retry_attempts + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            LOGGER.error(
                "openagenda.request_error attempt=%s/%s url=%s error=%s",
                attempt,
                retry_attempts,
                url,
                exc,
            )
            if attempt >= retry_attempts:
                raise RuntimeError(f"OpenAgenda request failed after retries: {exc}") from exc
            time.sleep(backoff_seconds * attempt)
            continue

        if 500 <= response.status_code < 600:
            LOGGER.error(
                "openagenda.server_error attempt=%s/%s status=%s url=%s",
                attempt,
                retry_attempts,
                response.status_code,
                url,
            )
            if attempt >= retry_attempts:
                raise RuntimeError(
                    f"OpenAgenda server error after retries (HTTP {response.status_code})."
                )
            time.sleep(backoff_seconds * attempt)
            continue

        if response.status_code in {401, 403}:
            raise RuntimeError(
                "OpenAgenda request failed with HTTP "
                f"{response.status_code}: {response.text[:300]}. "
                "Check API key permissions, send key in header 'key: <API_KEY>', "
                "and use /v2/agendas/{agenda_uid}/events endpoints."
            )

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


def fetch_agendas(
    config: OpenAgendaConfig,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch agenda list from explicit UIDs or search strategy around Montpellier."""

    if config.agenda_uids:
        return [{"uid": uid, "title": ""} for uid in config.agenda_uids if uid]

    queries = [query for query in config.agenda_search_queries if query]
    if not queries:
        if config.city:
            queries.append(config.city)
        if config.department:
            queries.append(config.department)
        if config.department == "34":
            queries.append("Herault")

    # Deduplicate while preserving order.
    dedup_queries: list[str] = []
    seen_query: set[str] = set()
    for query in queries:
        key = query.lower().strip()
        if not key or key in seen_query:
            continue
        dedup_queries.append(query.strip())
        seen_query.add(key)

    if not dedup_queries:
        LOGGER.warning("openagenda.no_queries No agenda_uids and no agenda_search queries configured.")
        return []

    own_session = session is None
    session = session or requests.Session()
    headers = _build_request_headers(config)

    agendas: list[dict[str, Any]] = []
    seen_uids: set[str] = set()

    try:
        for query in dedup_queries:
            if len(agendas) >= config.max_agendas:
                break

            from_cursor = 0
            for page_number in range(1, config.agenda_max_pages + 1):
                params = build_agenda_query_params(config, from_cursor=from_cursor, query=query)
                LOGGER.info(
                    "openagenda.agendas_page_start query=%s page=%s from=%s size=%s",
                    query,
                    page_number,
                    from_cursor,
                    config.agenda_page_size,
                )

                payload = _request_page(
                    session=session,
                    url=config.agendas_url,
                    params=params,
                    headers=headers,
                    timeout_seconds=config.timeout_seconds,
                    retry_attempts=config.retry_attempts,
                    backoff_seconds=config.backoff_seconds,
                )
                page_agendas = _extract_agendas(payload)

                if not page_agendas:
                    LOGGER.info(
                        "openagenda.agendas_page_empty query=%s page=%s",
                        query,
                        page_number,
                    )
                    break

                for agenda in page_agendas:
                    uid = _extract_agenda_uid(agenda)
                    if not uid or uid in seen_uids:
                        continue

                    seen_uids.add(uid)
                    agendas.append(
                        {
                            "uid": uid,
                            "title": _extract_agenda_title(agenda, language=config.language),
                        }
                    )
                    if len(agendas) >= config.max_agendas:
                        break

                LOGGER.info(
                    "openagenda.agendas_page_done query=%s page=%s page_agendas=%s total_agendas=%s",
                    query,
                    page_number,
                    len(page_agendas),
                    len(agendas),
                )

                if len(agendas) >= config.max_agendas:
                    break
                if len(page_agendas) < config.agenda_page_size:
                    break
                from_cursor += len(page_agendas)

    finally:
        if own_session:
            session.close()

    LOGGER.info("openagenda.agendas_found total=%s max_agendas=%s", len(agendas), config.max_agendas)
    return agendas


def fetch_events_for_agenda(
    config: OpenAgendaConfig,
    agenda_uid: str,
    *,
    agenda_title: str = "",
    max_events: int | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch events for one agenda using /v2/agendas/{uid}/events endpoint."""

    if not agenda_uid:
        return []

    own_session = session is None
    session = session or requests.Session()
    headers = _build_request_headers(config)

    events: list[dict[str, Any]] = []
    from_cursor = 0
    events_limit = max_events if max_events is not None else config.max_events
    geo_filtered_out = 0

    try:
        for page_number in range(1, config.max_pages + 1):
            if len(events) >= events_limit:
                break

            params = build_event_query_params(config, from_cursor=from_cursor)
            LOGGER.info(
                "openagenda.events_page_start agenda_uid=%s page=%s from=%s size=%s",
                agenda_uid,
                page_number,
                from_cursor,
                config.page_size,
            )
            LOGGER.debug(
                "openagenda.events_page_request agenda_uid=%s page=%s url=%s params=%s",
                agenda_uid,
                page_number,
                config.agenda_events_url(agenda_uid),
                params,
            )

            payload = _request_page(
                session=session,
                url=config.agenda_events_url(agenda_uid),
                params=params,
                headers=headers,
                timeout_seconds=config.timeout_seconds,
                retry_attempts=config.retry_attempts,
                backoff_seconds=config.backoff_seconds,
            )
            LOGGER.info(
                "openagenda.events_page_response agenda_uid=%s page=%s total=%s keys=%s",
                agenda_uid,
                page_number,
                payload.get("total"),
                list(payload.keys())[:8],
            )
            page_events = _extract_events(payload)

            if not page_events:
                LOGGER.info(
                    "openagenda.events_page_empty agenda_uid=%s page=%s",
                    agenda_uid,
                    page_number,
                )
                break

            accepted_in_page = 0
            for raw_event in page_events:
                if len(events) >= events_limit:
                    break

                record = dict(raw_event)
                record.setdefault("agenda_uid", agenda_uid)
                if agenda_title:
                    record.setdefault("agenda_title", agenda_title)
                if _event_matches_geo_scope(config, record):
                    events.append(record)
                    accepted_in_page += 1
                else:
                    geo_filtered_out += 1

            LOGGER.info(
                "openagenda.events_page_done agenda_uid=%s page=%s page_events=%s accepted=%s total_events=%s geo_filtered_out=%s",
                agenda_uid,
                page_number,
                len(page_events),
                accepted_in_page,
                len(events),
                geo_filtered_out,
            )

            if len(page_events) < config.page_size:
                break
            from_cursor += len(page_events)

    finally:
        if own_session:
            session.close()

    return events


def _fetch_events_legacy_direct(
    config: OpenAgendaConfig,
    *,
    session: requests.Session,
) -> list[dict[str, Any]]:
    """Legacy fallback for older setup hitting /v2/events directly."""

    headers = _build_request_headers(config)
    events: list[dict[str, Any]] = []
    offset = 0

    for page_number in range(1, config.max_pages + 1):
        if len(events) >= config.max_events:
            break

        params = build_legacy_events_query_params(config, offset=offset)
        LOGGER.info(
            "openagenda.legacy_page_start page=%s offset=%s size=%s",
            page_number,
            offset,
            config.page_size,
        )

        payload = _request_page(
            session=session,
            url=config.base_url,
            params=params,
            headers=headers,
            timeout_seconds=config.timeout_seconds,
            retry_attempts=config.retry_attempts,
            backoff_seconds=config.backoff_seconds,
        )
        page_events = _extract_events(payload)

        if not page_events:
            LOGGER.info("openagenda.legacy_page_empty page=%s", page_number)
            break

        remaining = config.max_events - len(events)
        events.extend(page_events[:remaining])

        LOGGER.info(
            "openagenda.legacy_page_done page=%s page_events=%s total_events=%s",
            page_number,
            len(page_events),
            len(events),
        )

        if len(page_events) < config.page_size:
            break
        offset += len(page_events)

    return events


def fetch_events_with_stats(
    config: OpenAgendaConfig,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch events from multiple agendas and return events + ingestion stats."""

    if not config.api_key:
        LOGGER.warning(
            "OPENAGENDA_API_KEY missing in runtime config. "
            "Requests are likely to return 401/403."
        )

    own_session = session is None
    session = session or requests.Session()

    stats: dict[str, Any] = {
        "agendas_found": 0,
        "agendas_scanned": 0,
        "events_by_agenda": {},
        "legacy_fallback_used": False,
        "total_events": 0,
    }

    try:
        agendas = fetch_agendas(config, session=session)
        stats["agendas_found"] = len(agendas)

        if not agendas and config.allow_legacy_events_fallback and config.uses_legacy_events_endpoint:
            LOGGER.warning(
                "openagenda.legacy_fallback enabled because no agenda was discovered and base_url=%s",
                config.base_url,
            )
            events = _fetch_events_legacy_direct(config, session=session)
            stats["legacy_fallback_used"] = True
            stats["total_events"] = len(events)
            return events, stats

        if not agendas:
            LOGGER.warning(
                "openagenda.no_agendas_discovered queries=%s city=%s department=%s",
                config.agenda_search_queries,
                config.city,
                config.department,
            )
            return [], stats

        all_events: list[dict[str, Any]] = []
        for agenda in agendas:
            if len(all_events) >= config.max_events:
                break

            agenda_uid = str(agenda.get("uid", "")).strip()
            agenda_title = str(agenda.get("title", "")).strip()
            if not agenda_uid:
                continue

            remaining = config.max_events - len(all_events)
            agenda_events = fetch_events_for_agenda(
                config,
                agenda_uid,
                agenda_title=agenda_title,
                max_events=remaining,
                session=session,
            )

            stats["agendas_scanned"] += 1
            stats["events_by_agenda"][agenda_uid] = len(agenda_events)
            all_events.extend(agenda_events)

            LOGGER.info(
                "openagenda.agenda_done uid=%s title=%s events=%s cumulative=%s",
                agenda_uid,
                agenda_title,
                len(agenda_events),
                len(all_events),
            )

        stats["total_events"] = len(all_events)
        return all_events, stats

    finally:
        if own_session:
            session.close()


def fetch_events(
    config: OpenAgendaConfig,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible helper: fetch events only."""

    events, _ = fetch_events_with_stats(config, session=session)
    return events
