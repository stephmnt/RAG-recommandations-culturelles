"""Index management helpers for API rebuild/reload operations."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from src.api.exceptions import RebuildBusyError

LOGGER = logging.getLogger(__name__)


def has_index_artifacts(index_path: str | Path) -> bool:
    path = Path(index_path)
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def read_index_metadata(index_path: str | Path) -> dict[str, Any]:
    metadata_path = Path(index_path) / "index_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class IndexManager:
    """Coordinates index rebuild and metadata access with a mutex lock."""

    def __init__(self, *, indexing_config_path: str, logger: logging.Logger | None = None) -> None:
        self.indexing_config_path = indexing_config_path
        self.logger = logger or LOGGER
        self._lock = threading.Lock()
        self._is_rebuilding = False

    @property
    def is_busy(self) -> bool:
        return self._is_rebuilding

    def rebuild_index(self, *, dataset_path: str, index_path: str) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RebuildBusyError()

        self._is_rebuilding = True
        started = time.perf_counter()
        try:
            from src.indexing.build_index import build_faiss_index, load_indexing_config

            config = load_indexing_config(self.indexing_config_path)
            config = dict(config)
            config["paths"] = dict(config.get("paths", {}))
            config["paths"]["input_dataset"] = str(Path(dataset_path).resolve())
            config["paths"]["output_dir"] = str(Path(index_path).resolve())

            result = build_faiss_index(
                input_path=config["paths"]["input_dataset"],
                output_dir=config["paths"]["output_dir"],
                config=config,
                logger=self.logger,
            )
            metadata = read_index_metadata(config["paths"]["output_dir"])
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "mode": "rebuild",
                "dataset_path": config["paths"]["input_dataset"],
                "index_path": config["paths"]["output_dir"],
                "index_metadata": metadata,
                "build_summary": result.to_dict(),
                "timings_ms": {"total": elapsed_ms},
            }
        finally:
            self._is_rebuilding = False
            self._lock.release()

    def reload_index(self, *, index_path: str, dataset_path: str) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RebuildBusyError()
        self._is_rebuilding = True
        started = time.perf_counter()
        try:
            resolved_index_path = str(Path(index_path).resolve())
            metadata = read_index_metadata(resolved_index_path)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "mode": "reload",
                "dataset_path": str(Path(dataset_path).resolve()),
                "index_path": resolved_index_path,
                "index_metadata": metadata,
                "timings_ms": {"total": elapsed_ms},
            }
        finally:
            self._is_rebuilding = False
            self._lock.release()
