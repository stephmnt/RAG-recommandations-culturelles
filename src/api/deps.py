"""Dependency registry for Flask API runtime."""

from __future__ import annotations

import logging
from typing import Any

from src.api.config import APISettings, load_settings
from src.api.index_manager import IndexManager, has_index_artifacts, read_index_metadata
from src.rag.service import RAGService
from src.rag.types import RAGConfig

LOGGER = logging.getLogger(__name__)

_SETTINGS: APISettings | None = None
_RAG_SERVICE: RAGService | None = None
_INDEX_MANAGER: IndexManager | None = None


def configure(overrides: dict[str, Any] | None = None) -> APISettings:
    global _SETTINGS, _RAG_SERVICE, _INDEX_MANAGER
    _SETTINGS = load_settings(overrides=overrides)
    _RAG_SERVICE = None
    _INDEX_MANAGER = None
    return _SETTINGS


def get_settings() -> APISettings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def get_index_manager() -> IndexManager:
    global _INDEX_MANAGER
    if _INDEX_MANAGER is None:
        settings = get_settings()
        _INDEX_MANAGER = IndexManager(
            indexing_config_path=settings.indexing_config_path,
            logger=LOGGER,
        )
    return _INDEX_MANAGER


def reset_rag_service() -> None:
    global _RAG_SERVICE
    _RAG_SERVICE = None


def get_rag_service(force_reload: bool = False) -> RAGService:
    global _RAG_SERVICE
    settings = get_settings()
    if force_reload:
        _RAG_SERVICE = None
    if _RAG_SERVICE is None:
        config = RAGConfig(
            index_path=settings.index_path,
            prompt_version=settings.prompt_version,
            retriever_top_k=settings.default_top_k,
            max_question_chars=settings.max_question_chars,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            llm_model=settings.mistral_model,
        )
        _RAG_SERVICE = RAGService(config=config, logger=LOGGER)
    return _RAG_SERVICE


def index_available() -> bool:
    settings = get_settings()
    return has_index_artifacts(settings.index_path)


def get_index_metadata() -> dict[str, Any]:
    settings = get_settings()
    return read_index_metadata(settings.index_path)
