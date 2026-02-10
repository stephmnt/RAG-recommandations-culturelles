"""Dependency container for API services (RAG + index lifecycle)."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flask import Flask, current_app

from src.api.config import APISettings
from src.rag.retriever import RAGRetriever
from src.rag.service import RAGService
from src.rag.types import RAGConfig

LOGGER = logging.getLogger(__name__)
_EXTENSION_KEY = "api_dependencies"

if TYPE_CHECKING:  # pragma: no cover
    from src.api.index_manager import IndexManager


def _safe_iso_to_date(value: Any):
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _matches_filters(source: dict[str, Any], filters: dict[str, Any]) -> bool:
    city_filter = str(filters.get("city", "")).strip().lower()
    if city_filter:
        source_city = str(source.get("city", "")).strip().lower()
        if source_city != city_filter:
            return False

    date_from = _safe_iso_to_date(filters.get("date_from"))
    date_to = _safe_iso_to_date(filters.get("date_to"))
    if date_from or date_to:
        source_date = _safe_iso_to_date(source.get("start_datetime"))
        if source_date is None:
            return False
        if date_from and source_date < date_from:
            return False
        if date_to and source_date > date_to:
            return False

    return True


class AppDependencies:
    """Singleton-style container for API runtime dependencies."""

    def __init__(
        self,
        settings: APISettings,
        *,
        index_manager: "IndexManager | None" = None,
        rag_service: RAGService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or LOGGER
        if index_manager is None:
            from src.api.index_manager import IndexManager

            index_manager = IndexManager(settings=settings, logger=self.logger)
        self.index_manager = index_manager
        self._rag_service: RAGService | None = rag_service
        self._service_lock = threading.RLock()

    def _build_rag_config(self, index_path: str | Path) -> RAGConfig:
        return RAGConfig(
            index_path=str(Path(index_path)),
            prompt_version=self.settings.prompt_version,
            retriever_top_k=self.settings.default_top_k,
            max_sources=self.settings.max_sources,
            context_max_chars=self.settings.context_max_chars,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.embedding_model,
            llm_model=self.settings.mistral_model,
            max_question_chars=self.settings.max_question_chars,
        )

    def _build_rag_service(self, index_path: str | Path | None = None) -> RAGService:
        target_index_path = Path(index_path or self.settings.index_path).resolve()
        vectorstore, embedding_model = self.index_manager.get_vectorstore(
            index_path=target_index_path,
            force_reload=True,
        )
        rag_config = self._build_rag_config(target_index_path)
        retriever = RAGRetriever(
            config=rag_config,
            vectorstore=vectorstore,
            embedding_model=embedding_model,
            logger=self.logger,
        )
        return RAGService(config=rag_config, retriever=retriever, logger=self.logger)

    def get_rag_service(self) -> RAGService:
        with self._service_lock:
            if self._rag_service is None:
                self._rag_service = self._build_rag_service()
            return self._rag_service

    def reload_rag_service(self, *, index_path: str | Path | None = None) -> RAGService:
        with self._service_lock:
            self._rag_service = self._build_rag_service(index_path=index_path)
            return self._rag_service

    def ask(
        self,
        *,
        question: str,
        top_k: int,
        debug: bool,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        service = self.get_rag_service()
        result = service.ask(question=question, top_k=top_k, debug=debug)
        payload = result.model_dump()

        if self.index_manager.is_rebuilding:
            payload.setdefault("meta", {}).setdefault("warnings", []).append(
                "rebuild_in_progress_using_cached_index"
            )

        if filters:
            original_sources = list(payload.get("sources", []))
            filtered_sources = [
                source for source in original_sources if _matches_filters(source, filters)
            ]
            if len(filtered_sources) != len(original_sources):
                payload["sources"] = filtered_sources
                payload.setdefault("meta", {})["returned_events"] = len(filtered_sources)
                payload.setdefault("meta", {}).setdefault("warnings", []).append(
                    "post_filter_applied_sources_only"
                )
                payload.setdefault("meta", {}).setdefault("filters_applied", []).append(
                    "api_post_filter"
                )

        return payload

    def rebuild_index(
        self,
        *,
        dataset_path: str | None = None,
        index_path: str | None = None,
    ) -> dict[str, Any]:
        payload = self.index_manager.rebuild_index(
            dataset_path=dataset_path,
            index_path=index_path,
        )
        self.reload_rag_service(index_path=payload.get("index_path"))
        return payload

    def reload_index(self, *, index_path: str | None = None) -> dict[str, Any]:
        payload = self.index_manager.reload_index(index_path=index_path)
        self.reload_rag_service(index_path=payload.get("index_path"))
        return payload

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": self.index_manager.is_index_loaded(),
            "mistral_configured": self.settings.mistral_configured,
            "version": self.settings.api_version,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        index_meta = self.index_manager.read_index_metadata()
        embeddings_meta = index_meta.get("embeddings", {}) if isinstance(index_meta, dict) else {}

        embedding_model = (
            embeddings_meta.get("huggingface_model")
            or embeddings_meta.get("mistral_model")
            or self.settings.embedding_model
            or ""
        )

        return {
            "index": {
                "path": str(Path(self.settings.index_path).resolve()),
                "build_date": index_meta.get("built_at_utc"),
                "num_events": index_meta.get("events_valid") or index_meta.get("events_input"),
                "num_chunks": index_meta.get("chunks_count"),
                "embedding_model": embedding_model,
                "dataset_hash": index_meta.get("dataset_hash"),
                "index_loaded": self.index_manager.is_index_loaded(),
                "rebuild_in_progress": self.index_manager.is_rebuilding,
            },
            "rag": {
                "default_top_k": self.settings.default_top_k,
                "max_top_k": self.settings.max_top_k,
                "prompt_version": self.settings.prompt_version,
                "llm_model": self.settings.mistral_model,
            },
        }


def init_dependencies(app: Flask, deps_override: AppDependencies | None = None) -> AppDependencies:
    if deps_override is not None:
        app.extensions[_EXTENSION_KEY] = deps_override
        return deps_override

    settings: APISettings = app.config["API_SETTINGS"]
    deps = AppDependencies(settings=settings)
    app.extensions[_EXTENSION_KEY] = deps
    return deps


def get_dependencies() -> AppDependencies:
    deps = current_app.extensions.get(_EXTENSION_KEY)
    if deps is None:
        raise RuntimeError("API dependencies are not initialized.")
    return deps


__all__ = ["AppDependencies", "get_dependencies", "init_dependencies"]
