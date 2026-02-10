"""Index lifecycle manager for API rebuild/reload operations."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from src.api.config import APISettings
from src.api.exceptions import IndexUnavailableError, RebuildInProgressError
from src.indexing.build_index import (
    build_faiss_index,
    load_faiss_index,
    load_indexing_config,
)
from src.indexing.embeddings import get_embedding_model

LOGGER = logging.getLogger(__name__)


class IndexManager:
    """Handle FAISS index reload/rebuild and in-memory cache."""

    def __init__(
        self,
        settings: APISettings,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or LOGGER

        self._state_lock = threading.RLock()
        self._rebuild_lock = threading.Lock()
        self._is_rebuilding = False

        self._vectorstore: Any | None = None
        self._embedding_model: Any | None = None
        self._cached_index_path: Path | None = None

    @property
    def is_rebuilding(self) -> bool:
        return self._is_rebuilding

    def is_index_loaded(self) -> bool:
        with self._state_lock:
            return self._vectorstore is not None

    def has_persisted_index(self, index_path: str | Path | None = None) -> bool:
        path = self._resolve_index_path(index_path)
        return (path / "index.faiss").exists() and (path / "index.pkl").exists()

    def _resolve_index_path(self, index_path: str | Path | None = None) -> Path:
        raw = index_path if index_path is not None else self.settings.index_path
        return Path(raw).resolve()

    def _resolve_dataset_path(self, dataset_path: str | Path | None = None) -> Path:
        raw = dataset_path if dataset_path is not None else self.settings.dataset_path
        return Path(raw).resolve()

    def _base_indexing_config(self) -> dict[str, Any]:
        config_path = Path(self.settings.indexing_config_path)
        if config_path.exists():
            return load_indexing_config(config_path)
        return load_indexing_config(None)

    def _resolved_indexing_config(
        self,
        *,
        index_path: Path,
        dataset_path: Path | None = None,
    ) -> dict[str, Any]:
        cfg = self._base_indexing_config()

        paths_cfg = dict(cfg.get("paths", {}))
        paths_cfg["output_dir"] = str(index_path)
        if dataset_path is not None:
            paths_cfg["input_dataset"] = str(dataset_path)
        cfg["paths"] = paths_cfg

        embeddings_cfg = dict(cfg.get("embeddings", {}))
        embeddings_cfg["provider"] = self.settings.embedding_provider
        if self.settings.embedding_model:
            if self.settings.embedding_provider == "mistral":
                embeddings_cfg["mistral_model"] = self.settings.embedding_model
            else:
                embeddings_cfg["huggingface_model"] = self.settings.embedding_model
        cfg["embeddings"] = embeddings_cfg

        return cfg

    def _load_vectorstore(
        self,
        *,
        index_path: Path,
        config: dict[str, Any],
        force_reload: bool,
    ) -> tuple[Any, Any]:
        with self._state_lock:
            if (
                not force_reload
                and self._vectorstore is not None
                and self._cached_index_path == index_path
                and self._embedding_model is not None
            ):
                return self._vectorstore, self._embedding_model

        embedding_model = get_embedding_model(
            config=config,
            provider=self.settings.embedding_provider,
            huggingface_model=self.settings.embedding_model,
            logger=self.logger,
        )

        vectorstore = load_faiss_index(
            index_dir=index_path,
            config=config,
            embedding_model=embedding_model,
            logger=self.logger,
        )

        with self._state_lock:
            self._vectorstore = vectorstore
            self._embedding_model = embedding_model
            self._cached_index_path = index_path

        return vectorstore, embedding_model

    def get_vectorstore(
        self,
        *,
        index_path: str | Path | None = None,
        force_reload: bool = False,
    ) -> tuple[Any, Any]:
        target_index = self._resolve_index_path(index_path)
        if not self.has_persisted_index(target_index):
            raise IndexUnavailableError(
                f"Index artifacts missing at {target_index}. Run /rebuild mode='rebuild' first."
            )

        cfg = self._resolved_indexing_config(index_path=target_index)
        try:
            return self._load_vectorstore(
                index_path=target_index,
                config=cfg,
                force_reload=force_reload,
            )
        except FileNotFoundError as exc:
            raise IndexUnavailableError(str(exc)) from exc

    def read_index_metadata(self, index_path: str | Path | None = None) -> dict[str, Any]:
        target_index = self._resolve_index_path(index_path)
        metadata_path = target_index / "index_metadata.json"
        if not metadata_path.exists():
            return {}

        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.logger.warning("Invalid JSON in index metadata: %s", metadata_path)
            return {}

    def reload_index(self, *, index_path: str | Path | None = None) -> dict[str, Any]:
        target_index = self._resolve_index_path(index_path)
        if not self.has_persisted_index(target_index):
            raise IndexUnavailableError(
                f"Cannot reload. Index artifacts not found at {target_index}."
            )

        started = time.perf_counter()
        cfg = self._resolved_indexing_config(index_path=target_index)
        self._load_vectorstore(index_path=target_index, config=cfg, force_reload=True)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return {
            "status": "ok",
            "mode": "reload",
            "index_path": str(target_index),
            "dataset_path": str(self._resolve_dataset_path()),
            "index_metadata": self.read_index_metadata(target_index),
            "timings_ms": {"total": elapsed_ms},
        }

    def rebuild_index(
        self,
        *,
        dataset_path: str | Path | None = None,
        index_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if not self._rebuild_lock.acquire(blocking=False):
            raise RebuildInProgressError("An index rebuild is already in progress.")

        self._is_rebuilding = True
        started = time.perf_counter()
        build_started = time.perf_counter()

        try:
            target_dataset = self._resolve_dataset_path(dataset_path)
            target_index = self._resolve_index_path(index_path)

            if not target_dataset.exists():
                raise FileNotFoundError(f"Dataset not found: {target_dataset}")

            target_index.mkdir(parents=True, exist_ok=True)

            cfg = self._resolved_indexing_config(
                index_path=target_index,
                dataset_path=target_dataset,
            )
            build_result = build_faiss_index(
                input_path=target_dataset,
                output_dir=target_index,
                config=cfg,
                logger=self.logger,
            )
            build_elapsed_ms = int((time.perf_counter() - build_started) * 1000)

            reload_payload = self.reload_index(index_path=target_index)
            total_elapsed_ms = int((time.perf_counter() - started) * 1000)

            return {
                "status": "ok",
                "mode": "rebuild",
                "index_path": str(target_index),
                "dataset_path": str(target_dataset),
                "index_metadata": reload_payload.get("index_metadata", {}),
                "build_summary": build_result.to_dict(),
                "timings_ms": {
                    "build": build_elapsed_ms,
                    "reload": int(reload_payload.get("timings_ms", {}).get("total", 0)),
                    "total": total_elapsed_ms,
                },
            }
        finally:
            self._is_rebuilding = False
            self._rebuild_lock.release()


__all__ = [
    "IndexManager",
    "IndexUnavailableError",
    "RebuildInProgressError",
]
