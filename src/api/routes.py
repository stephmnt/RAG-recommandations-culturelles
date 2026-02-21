"""Flask routes exposing RAG and index operations."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from src.api.deps import (
    get_index_manager,
    get_index_metadata,
    get_rag_service,
    get_settings,
    index_available,
    reset_rag_service,
)
from src.api.exceptions import APIError, IndexUnavailableError
from src.api.schemas import AskFilters, parse_ask_request, parse_rebuild_request
from src.rag.service import FALLBACK_ANSWER

LOGGER = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def _clamp_top_k(raw_top_k: int | None, default_top_k: int, max_top_k: int) -> int:
    if raw_top_k is None:
        return default_top_k
    return max(1, min(int(raw_top_k), max_top_k))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return date.fromisoformat(value)
        except Exception:
            return None


def _apply_source_filters(payload: dict[str, Any], filters: AskFilters | None) -> dict[str, Any]:
    if filters is None:
        return payload

    sources = list(payload.get("sources", []))
    filtered = sources
    filters_applied: list[str] = []

    if filters.city:
        city_requested = filters.city.strip().lower()
        filtered = [item for item in filtered if str(item.get("city", "")).strip().lower() == city_requested]
        filters_applied.append(f"city_eq:{filters.city.strip()}")

    start_bound = _parse_date(filters.date_from)
    end_bound = _parse_date(filters.date_to)
    if start_bound or end_bound:
        def _in_range(source: dict[str, Any]) -> bool:
            source_dt = _parse_date(str(source.get("start_datetime", "")))
            if source_dt is None:
                return False
            if start_bound and source_dt < start_bound:
                return False
            if end_bound and source_dt > end_bound:
                return False
            return True

        filtered = [item for item in filtered if _in_range(item)]
        filters_applied.append("date_range")

    payload["sources"] = filtered
    meta = dict(payload.get("meta", {}))
    current_filters = list(meta.get("filters_applied", []))
    meta["filters_applied"] = current_filters + filters_applied
    meta["returned_events"] = len(filtered)

    if not filtered:
        payload["answer"] = (
            f"{FALLBACK_ANSWER} "
            "Aucun evenement ne correspond aux filtres appliques."
        )

    payload["meta"] = meta
    return payload


@api_bp.get("/")
def root():
    return jsonify(
        {
            "name": "Puls-Events RAG API",
            "status": "ok",
            "endpoints": ["/health", "/metadata", "/ask", "/rebuild"],
        }
    )


@api_bp.get("/health")
def health():
    settings = get_settings()
    response = {
        "status": "ok",
        "api": "up",
        "index_loaded": bool(index_available()),
        "mistral_configured": bool(os.getenv("MISTRAL_API_KEY", "").strip()),
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "env": settings.flask_env,
    }
    return jsonify(response)


@api_bp.get("/metadata")
def metadata():
    settings = get_settings()
    index_meta = get_index_metadata()
    response = {
        "index": {
            "path": settings.index_path,
            "build_date": index_meta.get("built_at_utc", ""),
            "num_events": index_meta.get("events_valid"),
            "num_chunks": index_meta.get("chunks_count"),
            "embedding_model": index_meta.get("embeddings", {}).get("huggingface_model")
            or settings.embedding_model,
            "dataset_hash": index_meta.get("dataset_hash", ""),
        },
        "rag": {
            "default_top_k": settings.default_top_k,
            "max_top_k": settings.max_top_k,
            "prompt_version": settings.prompt_version,
            "llm_model": settings.mistral_model,
        },
    }
    return jsonify(response)


@api_bp.post("/ask")
def ask():
    raw_payload = request.get_json(silent=True)
    if not isinstance(raw_payload, dict):
        raise APIError(
            code="INVALID_REQUEST",
            message="JSON payload is required.",
            status_code=400,
        )

    parsed = parse_ask_request(raw_payload)
    settings = get_settings()
    manager = get_index_manager()
    if manager.is_busy:
        raise APIError(
            code="INDEX_REBUILD_IN_PROGRESS",
            message="Index rebuild is in progress. Try again shortly.",
            status_code=503,
        )
    if not index_available():
        raise IndexUnavailableError(
            f"Index artifacts missing at {settings.index_path}. "
            "Run /rebuild mode='rebuild' first."
        )

    top_k = _clamp_top_k(
        raw_top_k=parsed.top_k,
        default_top_k=settings.default_top_k,
        max_top_k=settings.max_top_k,
    )

    service = get_rag_service()
    result = service.ask(parsed.question, top_k=top_k, debug=parsed.debug)
    payload = result.model_dump()
    payload["meta"]["retriever_top_k"] = top_k

    filtered_payload = _apply_source_filters(payload, parsed.filters)
    return jsonify(filtered_payload)


def _validate_admin_token(header_token: str | None, expected_token: str) -> None:
    if not expected_token:
        raise APIError(
            code="ADMIN_TOKEN_MISSING",
            message="ADMIN_TOKEN is not configured on server.",
            status_code=500,
        )
    if not header_token:
        raise APIError(
            code="UNAUTHORIZED",
            message="Missing X-ADMIN-TOKEN header.",
            status_code=401,
        )
    if header_token != expected_token:
        raise APIError(
            code="FORBIDDEN",
            message="Invalid admin token.",
            status_code=403,
        )


@api_bp.post("/rebuild")
def rebuild():
    settings = get_settings()
    request_token = request.headers.get("X-ADMIN-TOKEN")
    _validate_admin_token(request_token, settings.admin_token)

    raw_payload = request.get_json(silent=True) or {}
    if not isinstance(raw_payload, dict):
        raise APIError(
            code="INVALID_REQUEST",
            message="JSON payload must be an object.",
            status_code=400,
        )
    parsed = parse_rebuild_request(raw_payload)
    index_path = parsed.index_path or settings.index_path
    dataset_path = parsed.dataset_path or settings.dataset_path

    manager = get_index_manager()
    if parsed.mode == "reload":
        summary = manager.reload_index(
            index_path=index_path,
            dataset_path=dataset_path,
        )
        reset_rag_service()
        return jsonify({"status": "ok", **summary})

    if not Path(dataset_path).exists():
        raise APIError(
            code="DATASET_NOT_FOUND",
            message=f"Dataset not found at {dataset_path}",
            status_code=400,
        )

    summary = manager.rebuild_index(
        dataset_path=dataset_path,
        index_path=index_path,
    )
    reset_rag_service()
    return jsonify({"status": "ok", **summary})
