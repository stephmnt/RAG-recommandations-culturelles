"""Request/response schemas for Flask API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

try:  # Pydantic v2+
    from pydantic import ConfigDict, field_validator

    PYDANTIC_V2 = True
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator  # type: ignore

    ConfigDict = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


class _CompatBaseModel(BaseModel):
    """Pydantic v1/v2 compatibility helpers."""

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        if hasattr(super(), "model_dump"):  # pragma: no cover - pydantic v2 path
            return super().model_dump(*args, **kwargs)  # type: ignore[misc]
        return self.dict(*args, **kwargs)

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")  # type: ignore[misc]
    else:  # pragma: no cover - pydantic v1 path
        class Config:
            extra = "forbid"


def parse_schema(model_cls: type[_CompatBaseModel], payload: dict[str, Any]) -> _CompatBaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    return model_cls.parse_obj(payload)  # type: ignore[attr-defined]


def _validate_iso_date(value: str) -> str:
    text = value.strip()
    if not text:
        return text
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return text
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date/datetime: {value}") from exc


class AskFilters(_CompatBaseModel):
    city: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    if PYDANTIC_V2:
        @field_validator("city")
        @classmethod
        def _normalize_city(cls, value: str | None) -> str | None:
            if value is None:
                return None
            text = value.strip()
            return text or None

        @field_validator("date_from", "date_to")
        @classmethod
        def _validate_dates(cls, value: str | None) -> str | None:
            if value in (None, ""):
                return None
            return _validate_iso_date(value)
    else:  # pragma: no cover - pydantic v1 path
        @validator("city")
        def _normalize_city(cls, value: str | None) -> str | None:
            if value is None:
                return None
            text = value.strip()
            return text or None

        @validator("date_from", "date_to")
        def _validate_dates(cls, value: str | None) -> str | None:
            if value in (None, ""):
                return None
            return _validate_iso_date(value)


class AskRequest(_CompatBaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int | None = None
    debug: bool = False
    filters: AskFilters | None = None

    if PYDANTIC_V2:
        @field_validator("question")
        @classmethod
        def _validate_question(cls, value: str) -> str:
            text = value.strip()
            if not text:
                raise ValueError("question is required")
            return text

        @field_validator("top_k")
        @classmethod
        def _validate_top_k(cls, value: int | None) -> int | None:
            if value is None:
                return None
            if value <= 0:
                raise ValueError("top_k must be > 0")
            return value
    else:  # pragma: no cover - pydantic v1 path
        @validator("question")
        def _validate_question(cls, value: str) -> str:
            text = value.strip()
            if not text:
                raise ValueError("question is required")
            return text

        @validator("top_k")
        def _validate_top_k(cls, value: int | None) -> int | None:
            if value is None:
                return None
            if value <= 0:
                raise ValueError("top_k must be > 0")
            return value


class RebuildRequest(_CompatBaseModel):
    mode: Literal["rebuild", "reload"]
    dataset_path: str | None = None
    index_path: str | None = None

    if PYDANTIC_V2:
        @field_validator("dataset_path", "index_path")
        @classmethod
        def _normalize_optional_paths(cls, value: str | None) -> str | None:
            if value is None:
                return None
            text = value.strip()
            return text or None
    else:  # pragma: no cover - pydantic v1 path
        @validator("dataset_path", "index_path")
        def _normalize_optional_paths(cls, value: str | None) -> str | None:
            if value is None:
                return None
            text = value.strip()
            return text or None


__all__ = [
    "AskFilters",
    "AskRequest",
    "RebuildRequest",
    "ValidationError",
    "parse_schema",
]
