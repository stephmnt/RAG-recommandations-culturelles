"""Build and persist FAISS index from processed Step-2 dataset."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.indexing.chunking import ChunkingConfig, build_documents_from_dataframe
from src.indexing.embeddings import get_embedding_model


LOGGER = logging.getLogger(__name__)


def _import_faiss_vectorstore():
    try:
        from langchain_community.vectorstores import FAISS

        return FAISS
    except Exception:
        from langchain.vectorstores import FAISS  # type: ignore

        return FAISS


FAISS = _import_faiss_vectorstore()


DEFAULT_INDEXING_CONFIG: dict[str, Any] = {
    "paths": {
        "input_dataset": "data/processed/events_processed.parquet",
        "output_dir": "artifacts/faiss_index",
    },
    "chunking": {
        "chunk_size": 900,
        "chunk_overlap": 120,
        "min_chunk_size": 50,
        "separators": ["\n\n", "\n", ". ", " ", ""],
    },
    "embeddings": {
        "provider": "huggingface",
        "huggingface_model": "intfloat/multilingual-e5-small",
        "mistral_model": "mistral-embed",
    },
    "faiss": {
        "normalize_L2": True,
    },
    "search": {
        "top_k": 5,
    },
}


@dataclass
class BuildIndexResult:
    events_input: int
    events_valid: int
    events_invalid: int
    chunks_count: int
    chunking_seconds: float
    index_build_seconds: float
    total_seconds: float
    output_dir: str
    metadata_path: str
    dataset_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunking_seconds"] = round(self.chunking_seconds, 4)
        payload["index_build_seconds"] = round(self.index_build_seconds, 4)
        payload["total_seconds"] = round(self.total_seconds, 4)
        return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_indexing_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_INDEXING_CONFIG)
    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Indexing config not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("Indexing config root must be a mapping.")
    return _deep_merge(config, payload)


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataframe = pd.read_parquet(path)
    elif suffix == ".jsonl":
        dataframe = pd.read_json(path, lines=True)
    elif suffix == ".csv":
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Use parquet, jsonl or csv."
        )
    if "event_id" not in dataframe.columns or "document_text" not in dataframe.columns:
        raise ValueError(
            "Dataset must contain at least 'event_id' and 'document_text' columns."
        )
    return dataframe


def _compute_dataset_hash(dataframe: pd.DataFrame, input_path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(input_path.resolve()).encode("utf-8"))
    hasher.update(str(len(dataframe)).encode("utf-8"))
    event_ids = sorted(str(value) for value in dataframe.get("event_id", []).tolist())
    for event_id in event_ids:
        hasher.update(event_id.encode("utf-8"))
    return hasher.hexdigest()


def _save_index_metadata(
    *,
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    metadata_path = output_dir / "index_metadata.json"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def _from_documents_with_compatibility(
    documents: list[Any],
    embeddings: Any,
    normalize_l2: bool,
) -> Any:
    try:
        return FAISS.from_documents(
            documents,
            embeddings,
            normalize_L2=normalize_l2,
        )
    except TypeError:
        return FAISS.from_documents(documents, embeddings)


def build_faiss_index(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    embedding_model: Any | None = None,
    logger: logging.Logger | None = None,
) -> BuildIndexResult:
    logger = logger or LOGGER
    config = config or dict(DEFAULT_INDEXING_CONFIG)

    start_total = time.perf_counter()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed dataset from %s", input_path)
    dataframe = load_dataset(input_path)
    events_input = len(dataframe)
    dataset_hash = _compute_dataset_hash(dataframe, input_path)

    chunking_start = time.perf_counter()
    chunking_config = ChunkingConfig.from_dict(config.get("chunking", {}))
    documents, event_to_chunks, invalid_events = build_documents_from_dataframe(
        dataframe=dataframe,
        chunking_config=chunking_config,
    )
    chunking_seconds = time.perf_counter() - chunking_start

    if not documents:
        raise ValueError("No valid documents generated for indexing.")

    logger.info(
        "Chunking done: events_input=%s events_valid=%s invalid_events=%s chunks=%s",
        events_input,
        len(event_to_chunks),
        invalid_events,
        len(documents),
    )

    if embedding_model is None:
        embedding_model = get_embedding_model(config=config, logger=logger)

    index_cfg = config.get("faiss", {}) if isinstance(config, dict) else {}
    normalize_l2 = bool(index_cfg.get("normalize_L2", True))

    index_build_start = time.perf_counter()
    vectorstore = _from_documents_with_compatibility(
        documents=documents,
        embeddings=embedding_model,
        normalize_l2=normalize_l2,
    )
    index_build_seconds = time.perf_counter() - index_build_start

    vectorstore.save_local(str(output_dir))
    total_seconds = time.perf_counter() - start_total

    build_metadata = {
        "built_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_dataset": str(input_path),
        "dataset_hash": dataset_hash,
        "events_input": events_input,
        "events_valid": len(event_to_chunks),
        "events_invalid": invalid_events,
        "chunks_count": len(documents),
        "chunking": chunking_config.to_dict(),
        "embeddings": config.get("embeddings", {}),
        "faiss": {
            "normalize_L2": normalize_l2,
        },
        "timing_seconds": {
            "chunking": round(chunking_seconds, 4),
            "index_build": round(index_build_seconds, 4),
            "total": round(total_seconds, 4),
        },
    }
    metadata_path = _save_index_metadata(output_dir=output_dir, payload=build_metadata)

    logger.info("Index saved to %s", output_dir)
    logger.info("Index metadata saved to %s", metadata_path)

    return BuildIndexResult(
        events_input=events_input,
        events_valid=len(event_to_chunks),
        events_invalid=invalid_events,
        chunks_count=len(documents),
        chunking_seconds=chunking_seconds,
        index_build_seconds=index_build_seconds,
        total_seconds=total_seconds,
        output_dir=str(output_dir),
        metadata_path=str(metadata_path),
        dataset_hash=dataset_hash,
    )


def _load_local_with_compatibility(
    *,
    index_dir: str | Path,
    embedding_model: Any,
    allow_dangerous_deserialization: bool = True,
) -> Any:
    signature = inspect.signature(FAISS.load_local)
    kwargs: dict[str, Any] = {}
    if "allow_dangerous_deserialization" in signature.parameters:
        kwargs["allow_dangerous_deserialization"] = allow_dangerous_deserialization
    return FAISS.load_local(str(index_dir), embedding_model, **kwargs)


def load_faiss_index(
    *,
    index_dir: str | Path,
    config: dict[str, Any] | None = None,
    embedding_model: Any | None = None,
    logger: logging.Logger | None = None,
    allow_dangerous_deserialization: bool = True,
) -> Any:
    logger = logger or LOGGER
    if embedding_model is None:
        embedding_model = get_embedding_model(config=config or {}, logger=logger)
    return _load_local_with_compatibility(
        index_dir=index_dir,
        embedding_model=embedding_model,
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    )
