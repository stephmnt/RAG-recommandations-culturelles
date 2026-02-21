"""Runtime configuration for Flask API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _resolve_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


@dataclass(frozen=True)
class APISettings:
    flask_env: str
    host: str
    port: int
    log_level: str
    admin_token: str

    index_path: str
    dataset_path: str
    indexing_config_path: str

    embedding_provider: str
    embedding_model: str
    mistral_model: str

    default_top_k: int
    max_top_k: int
    max_question_chars: int
    prompt_version: str

    def to_app_dict(self) -> dict[str, Any]:
        return {
            "FLASK_ENV": self.flask_env,
            "HOST": self.host,
            "PORT": self.port,
            "LOG_LEVEL": self.log_level,
            "ADMIN_TOKEN": self.admin_token,
            "INDEX_PATH": self.index_path,
            "DATASET_PATH": self.dataset_path,
            "INDEXING_CONFIG_PATH": self.indexing_config_path,
            "EMBEDDING_PROVIDER": self.embedding_provider,
            "EMBEDDING_MODEL": self.embedding_model,
            "MISTRAL_MODEL": self.mistral_model,
            "DEFAULT_TOP_K": self.default_top_k,
            "MAX_TOP_K": self.max_top_k,
            "MAX_QUESTION_CHARS": self.max_question_chars,
            "PROMPT_VERSION": self.prompt_version,
        }


def load_settings(overrides: dict[str, Any] | None = None) -> APISettings:
    overrides = overrides or {}

    flask_env = str(overrides.get("FLASK_ENV", _get_env("FLASK_ENV", "dev"))).strip()
    host = str(overrides.get("HOST", _get_env("HOST", "127.0.0.1"))).strip()
    port = int(overrides.get("PORT", _to_int(_get_env("PORT", "8000"), 8000)))
    log_level = str(overrides.get("LOG_LEVEL", _get_env("LOG_LEVEL", "INFO"))).upper()
    admin_token = str(overrides.get("ADMIN_TOKEN", _get_env("ADMIN_TOKEN", ""))).strip()

    index_path = _resolve_path(str(overrides.get("INDEX_PATH", _get_env("INDEX_PATH", "artifacts/faiss_index"))))
    dataset_path = _resolve_path(
        str(overrides.get("DATASET_PATH", _get_env("DATASET_PATH", "data/processed/events_processed.parquet")))
    )
    indexing_config_path = _resolve_path(
        str(overrides.get("INDEXING_CONFIG_PATH", _get_env("INDEXING_CONFIG_PATH", "configs/indexing.yaml")))
    )

    embedding_provider = str(
        overrides.get("EMBEDDING_PROVIDER", _get_env("EMBEDDING_PROVIDER", "huggingface"))
    ).strip()
    embedding_model = str(
        overrides.get("EMBEDDING_MODEL", _get_env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    ).strip()
    mistral_model = str(
        overrides.get("MISTRAL_MODEL", _get_env("MISTRAL_MODEL", "mistral-small-latest"))
    ).strip()

    default_top_k = int(overrides.get("DEFAULT_TOP_K", _to_int(_get_env("DEFAULT_TOP_K", "6"), 6)))
    max_top_k = int(overrides.get("MAX_TOP_K", _to_int(_get_env("MAX_TOP_K", "10"), 10)))
    max_question_chars = int(
        overrides.get("MAX_QUESTION_CHARS", _to_int(_get_env("MAX_QUESTION_CHARS", "500"), 500))
    )
    prompt_version = str(overrides.get("PROMPT_VERSION", _get_env("PROMPT_VERSION", "v1"))).strip()

    if default_top_k < 1:
        default_top_k = 1
    if max_top_k < 1:
        max_top_k = 1

    return APISettings(
        flask_env=flask_env,
        host=host,
        port=port,
        log_level=log_level,
        admin_token=admin_token,
        index_path=index_path,
        dataset_path=dataset_path,
        indexing_config_path=indexing_config_path,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        mistral_model=mistral_model,
        default_top_k=default_top_k,
        max_top_k=max_top_k,
        max_question_chars=max_question_chars,
        prompt_version=prompt_version,
    )
