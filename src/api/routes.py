"""Flask routes for Step-5 API endpoints."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, jsonify, request

from src.api.deps import get_dependencies
from src.api.errors import APIError
from src.api.exceptions import IndexUnavailableError, RebuildInProgressError
from src.api.schemas import AskRequest, RebuildRequest, parse_schema

api_bp = Blueprint("api", __name__)


def _model_dump(instance: Any) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(exclude_none=True)
    return instance.dict(exclude_none=True)


def _json_dict_body(required: bool = True) -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        if required:
            raise APIError(
                code="INVALID_REQUEST",
                message="JSON body is required.",
                status_code=400,
            )
        return {}
    if not isinstance(payload, dict):
        raise APIError(
            code="INVALID_REQUEST",
            message="JSON body must be an object.",
            status_code=400,
        )
    return payload


@api_bp.get("/health")
def health() -> Any:
    deps = get_dependencies()
    return jsonify(deps.get_health_payload()), 200


@api_bp.get("/metadata")
def metadata() -> Any:
    deps = get_dependencies()
    return jsonify(deps.get_metadata_payload()), 200


@api_bp.post("/ask")
def ask() -> Any:
    deps = get_dependencies()
    settings = current_app.config["API_SETTINGS"]

    payload = _json_dict_body(required=True)
    question_raw = payload.get("question", "")
    if not str(question_raw).strip():
        raise APIError(
            code="INVALID_REQUEST",
            message="question is required",
            status_code=400,
        )
    ask_req = parse_schema(AskRequest, payload)

    requested_top_k = ask_req.top_k if ask_req.top_k is not None else settings.default_top_k
    clamped_top_k = max(1, min(int(requested_top_k), int(settings.max_top_k)))

    filters_payload: dict[str, Any] | None = None
    if ask_req.filters is not None:
        filters_payload = _model_dump(ask_req.filters)

    try:
        result_payload = deps.ask(
            question=ask_req.question,
            top_k=clamped_top_k,
            debug=bool(ask_req.debug),
            filters=filters_payload,
        )
    except IndexUnavailableError as exc:
        raise APIError(
            code="INDEX_UNAVAILABLE",
            message=str(exc),
            status_code=503,
        ) from exc
    except RebuildInProgressError as exc:
        raise APIError(
            code="REBUILD_IN_PROGRESS",
            message=str(exc),
            status_code=503,
        ) from exc
    except ValueError as exc:
        raise APIError(
            code="INVALID_REQUEST",
            message=str(exc),
            status_code=400,
        ) from exc

    if clamped_top_k != requested_top_k:
        meta = result_payload.setdefault("meta", {})
        meta.setdefault("warnings", []).append(
            f"top_k_clamped:{requested_top_k}->{clamped_top_k}"
        )

    return jsonify(result_payload), 200


@api_bp.post("/rebuild")
def rebuild() -> Any:
    deps = get_dependencies()
    settings = current_app.config["API_SETTINGS"]

    configured_admin_token = str(settings.admin_token or "").strip()
    if not configured_admin_token:
        raise APIError(
            code="ADMIN_TOKEN_NOT_CONFIGURED",
            message="Server misconfiguration: ADMIN_TOKEN is missing.",
            status_code=503,
        )

    provided_token = str(request.headers.get("X-ADMIN-TOKEN", "")).strip()
    if not provided_token:
        raise APIError(
            code="MISSING_ADMIN_TOKEN",
            message="X-ADMIN-TOKEN header is required.",
            status_code=401,
        )
    if provided_token != configured_admin_token:
        raise APIError(
            code="INVALID_ADMIN_TOKEN",
            message="Invalid admin token.",
            status_code=403,
        )

    payload = _json_dict_body(required=True)
    mode_raw = str(payload.get("mode", "")).strip()
    if mode_raw not in {"rebuild", "reload"}:
        raise APIError(
            code="INVALID_REQUEST",
            message="mode must be one of: rebuild, reload",
            status_code=400,
        )
    rebuild_req = parse_schema(RebuildRequest, payload)

    try:
        if rebuild_req.mode == "reload":
            response_payload = deps.reload_index(index_path=rebuild_req.index_path)
        else:
            response_payload = deps.rebuild_index(
                dataset_path=rebuild_req.dataset_path,
                index_path=rebuild_req.index_path,
            )
    except RebuildInProgressError as exc:
        raise APIError(
            code="REBUILD_IN_PROGRESS",
            message=str(exc),
            status_code=503,
        ) from exc
    except FileNotFoundError as exc:
        raise APIError(
            code="INVALID_PATH",
            message=str(exc),
            status_code=400,
        ) from exc
    except IndexUnavailableError as exc:
        raise APIError(
            code="INDEX_UNAVAILABLE",
            message=str(exc),
            status_code=503,
        ) from exc

    return jsonify(response_payload), 200


__all__ = ["api_bp"]
