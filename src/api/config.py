"""Runtime configuration for Flask API (Step 5)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_VERSION = "0.1.0"


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


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class APISettings:
    flask_env: str
    host: str
    port: int
    log_level: str

    mistral_api_key: str
    mistral_model: str

    index_path: str
    dataset_path: str
    indexing_config_path: str

    embedding_provider: str
    embedding_model: str | None

    admin_token: str
    max_top_k: int
    default_top_k: int

    prompt_version: str
    max_question_chars: int
    context_max_chars: int
    max_sources: int

    api_version: str
    debug: bool

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "APISettings":
        overrides = overrides or {}

        def pick(name: str, default: Any = "") -> Any:
            if name in overrides:
                return overrides[name]
            lower = name.lower()
            if lower in overrides:
                return overrides[lower]
            return os.getenv(name, default)

        max_top_k = max(1, _to_int(pick("MAX_TOP_K", 10), 10))
        default_top_k = max(1, _to_int(pick("DEFAULT_TOP_K", 6), 6))
        if default_top_k > max_top_k:
            default_top_k = max_top_k

        max_question_chars = max(100, _to_int(pick("MAX_QUESTION_CHARS", 500), 500))

        prompt_version = str(pick("PROMPT_VERSION", "v1") or "v1").strip() or "v1"
        api_version = str(pick("API_VERSION", DEFAULT_VERSION) or DEFAULT_VERSION).strip() or DEFAULT_VERSION

        indexing_config_path = str(pick("INDEXING_CONFIG_PATH", "configs/indexing.yaml")).strip()
        index_path = str(pick("INDEX_PATH", "artifacts/faiss_index")).strip()
        dataset_path = str(pick("DATASET_PATH", "data/processed/events_processed.parquet")).strip()

        embedding_model = str(pick("EMBEDDING_MODEL", "")).strip() or None

        return cls(
            flask_env=str(pick("FLASK_ENV", "dev")).strip() or "dev",
            host=str(pick("HOST", "127.0.0.1")).strip() or "127.0.0.1",
            port=max(1, _to_int(pick("PORT", 8000), 8000)),
            log_level=str(pick("LOG_LEVEL", "INFO")).strip().upper() or "INFO",
            mistral_api_key=str(pick("MISTRAL_API_KEY", "")).strip(),
            mistral_model=str(pick("MISTRAL_MODEL", "mistral-small-latest")).strip()
            or "mistral-small-latest",
            index_path=index_path,
            dataset_path=dataset_path,
            indexing_config_path=indexing_config_path,
            embedding_provider=str(pick("EMBEDDING_PROVIDER", "huggingface")).strip().lower()
            or "huggingface",
            embedding_model=embedding_model,
            admin_token=str(pick("ADMIN_TOKEN", "")).strip(),
            max_top_k=max_top_k,
            default_top_k=default_top_k,
            prompt_version=prompt_version,
            max_question_chars=max_question_chars,
            context_max_chars=max(1000, _to_int(pick("CONTEXT_MAX_CHARS", 8000), 8000)),
            max_sources=max(1, _to_int(pick("MAX_SOURCES", 5), 5)),
            api_version=api_version,
            debug=_to_bool(pick("DEBUG", False), default=False),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "flask_env": self.flask_env,
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
            "mistral_model": self.mistral_model,
            "index_path": self.index_path,
            "dataset_path": self.dataset_path,
            "indexing_config_path": self.indexing_config_path,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "max_top_k": self.max_top_k,
            "default_top_k": self.default_top_k,
            "prompt_version": self.prompt_version,
            "max_question_chars": self.max_question_chars,
            "context_max_chars": self.context_max_chars,
            "max_sources": self.max_sources,
            "api_version": self.api_version,
            "debug": self.debug,
        }
        return payload

    @property
    def mistral_configured(self) -> bool:
        return bool(self.mistral_api_key)

    @property
    def index_path_obj(self) -> Path:
        return Path(self.index_path)

    @property
    def dataset_path_obj(self) -> Path:
        return Path(self.dataset_path)
