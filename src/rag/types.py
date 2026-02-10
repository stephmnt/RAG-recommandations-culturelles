"""Typed schemas for Step-4 RAG engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

try:  # Pydantic v2+
    from pydantic import ConfigDict, field_validator

    PYDANTIC_V2 = True
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator  # type: ignore

    ConfigDict = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


class _CompatBaseModel(BaseModel):
    """Pydantic v1/v2 compatibility for model_dump usage."""

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        if hasattr(super(), "model_dump"):  # pragma: no cover - pydantic v2 path
            return super().model_dump(*args, **kwargs)  # type: ignore[misc]
        return self.dict(*args, **kwargs)

    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[misc]
    else:  # pragma: no cover - pydantic v1 path
        class Config:
            arbitrary_types_allowed = True


class RAGSource(_CompatBaseModel):
    event_id: str = ""
    title: str = ""
    start_datetime: str = ""
    end_datetime: str | None = None
    city: str = ""
    location_name: str = ""
    url: str = ""
    score: float | None = None
    snippet: str = ""


class RAGResult(_CompatBaseModel):
    question: str
    answer: str
    sources: list[RAGSource] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class RAGConfig(_CompatBaseModel):
    index_path: str = "artifacts/faiss_index"
    prompt_version: str = "v1"

    retriever_top_k: int = 6
    max_sources: int = 5
    score_threshold: float | None = None
    min_chunk_chars: int = 40

    context_max_chars: int = 8000
    snippet_max_chars: int = 320

    embedding_provider: str = "huggingface"
    embedding_model: str | None = None

    llm_model: str = "mistral-small-latest"
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    llm_backoff_seconds: float = 1.5

    max_question_chars: int = 500

    if PYDANTIC_V2:
        @field_validator("retriever_top_k")
        @classmethod
        def _validate_top_k(cls, value: int) -> int:
            if value <= 0:
                raise ValueError("retriever_top_k must be > 0")
            return value

        @field_validator("max_sources")
        @classmethod
        def _validate_max_sources(cls, value: int) -> int:
            if value <= 0:
                raise ValueError("max_sources must be > 0")
            return value

        @field_validator("llm_temperature")
        @classmethod
        def _validate_temperature(cls, value: float) -> float:
            if value < 0 or value > 1:
                raise ValueError("llm_temperature must be between 0 and 1")
            return value
    else:  # pragma: no cover - pydantic v1 path
        @validator("retriever_top_k")
        def _validate_top_k(cls, value: int) -> int:
            if value <= 0:
                raise ValueError("retriever_top_k must be > 0")
            return value

        @validator("max_sources")
        def _validate_max_sources(cls, value: int) -> int:
            if value <= 0:
                raise ValueError("max_sources must be > 0")
            return value

        @validator("llm_temperature")
        def _validate_temperature(cls, value: float) -> float:
            if value < 0 or value > 1:
                raise ValueError("llm_temperature must be between 0 and 1")
            return value


@dataclass
class RetrievedChunk:
    """Internal retrieval result used by retriever/context/service."""

    content: str
    metadata: dict[str, Any]
    score: float | None = None
