"""Request schemas for Flask routes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.api.exceptions import APIError

try:  # pydantic v2
    from pydantic import field_validator

    PYDANTIC_V2 = True
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator  # type: ignore

    field_validator = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


class AskFilters(BaseModel):
    city: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., max_length=500)
    top_k: int | None = None
    debug: bool = False
    filters: AskFilters | None = None

    if PYDANTIC_V2:
        @field_validator("question")
        @classmethod
        def _validate_question(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("question is required")
            return value.strip()
    else:  # pragma: no cover - pydantic v1 fallback
        @validator("question")
        def _validate_question(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("question is required")
            return value.strip()


class RebuildRequest(BaseModel):
    mode: Literal["rebuild", "reload"] = "reload"
    dataset_path: str | None = None
    index_path: str | None = None


def _model_validate(model_cls: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    return model_cls.parse_obj(payload)


def parse_ask_request(payload: dict[str, Any]) -> AskRequest:
    try:
        model = _model_validate(AskRequest, payload)
        return model  # type: ignore[return-value]
    except ValidationError as exc:
        raise APIError(
            code="INVALID_REQUEST",
            message="Invalid ask payload.",
            status_code=400,
            details={"validation_errors": exc.errors()},
        ) from exc


def parse_rebuild_request(payload: dict[str, Any]) -> RebuildRequest:
    try:
        model = _model_validate(RebuildRequest, payload)
        return model  # type: ignore[return-value]
    except ValidationError as exc:
        raise APIError(
            code="INVALID_REQUEST",
            message="Invalid rebuild payload.",
            status_code=400,
            details={"validation_errors": exc.errors()},
        ) from exc
