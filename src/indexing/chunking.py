"""Chunking utilities for LangChain Document generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - compatibility import
    from langchain.schema import Document  # type: ignore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - compatibility import
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:  # pragma: no cover - lightweight fallback

        class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
            """Small fallback splitter when LangChain splitters are unavailable."""

            def __init__(
                self,
                *,
                chunk_size: int,
                chunk_overlap: int,
                separators: list[str] | None = None,
            ) -> None:
                self.chunk_size = chunk_size
                self.chunk_overlap = max(0, chunk_overlap)
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

            def split_text(self, text: str) -> list[str]:
                if not text:
                    return []
                chunks: list[str] = []
                start = 0
                while start < len(text):
                    end = min(len(text), start + self.chunk_size)
                    chunk = text[start:end]
                    if end < len(text):
                        for separator in self.separators:
                            if not separator:
                                continue
                            split_at = chunk.rfind(separator)
                            if split_at > 0:
                                end = start + split_at + len(separator)
                                chunk = text[start:end]
                                break
                    chunks.append(chunk.strip())
                    if end >= len(text):
                        break
                    start = max(0, end - self.chunk_overlap)
                return [item for item in chunks if item]


DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120
    min_chunk_size: int = 50
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ChunkingConfig":
        payload = payload or {}
        separators_raw = payload.get("separators", list(DEFAULT_SEPARATORS))
        if isinstance(separators_raw, list):
            separators = tuple(str(item) for item in separators_raw)
        elif isinstance(separators_raw, tuple):
            separators = tuple(str(item) for item in separators_raw)
        else:
            separators = DEFAULT_SEPARATORS
        return cls(
            chunk_size=int(payload.get("chunk_size", 900)),
            chunk_overlap=int(payload.get("chunk_overlap", 120)),
            min_chunk_size=int(payload.get("min_chunk_size", 50)),
            separators=separators,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["separators"] = list(self.separators)
        return payload


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_json_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def make_json_serializable(value: Any) -> Any:
    """Convert arbitrary values to JSON-serializable structures."""

    value = _parse_json_if_needed(value)
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    return str(value)


def _merge_short_chunks(chunks: list[str], min_chunk_size: int) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = []
    buffer = ""
    for index, chunk in enumerate(chunks):
        current = f"{buffer} {chunk}".strip() if buffer else chunk
        is_last = index == len(chunks) - 1
        if len(current) < min_chunk_size and not is_last:
            buffer = current
            continue
        merged.append(current)
        buffer = ""
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def split_text_into_chunks(text: str, config: ChunkingConfig) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
    )
    raw_chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk and chunk.strip()]
    return _merge_short_chunks(raw_chunks, min_chunk_size=config.min_chunk_size)


def _base_metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "event_id": _clean_text(row.get("event_id")),
        "title": _clean_text(row.get("title")),
        "start_datetime": _clean_text(row.get("start_datetime")),
        "end_datetime": _clean_text(row.get("end_datetime")),
        "city": _clean_text(row.get("city")),
        "location_name": _clean_text(row.get("location_name")),
        "url": _clean_text(row.get("url")),
        "source": _clean_text(row.get("source")) or "openagenda",
    }
    retrieval_metadata = row.get("retrieval_metadata", {})
    retrieval_metadata = _parse_json_if_needed(retrieval_metadata)
    if isinstance(retrieval_metadata, dict):
        for key, value in retrieval_metadata.items():
            key_clean = str(key)
            if key_clean in metadata:
                continue
            metadata[key_clean] = make_json_serializable(value)
    return make_json_serializable(metadata)


def build_documents_from_dataframe(
    dataframe: pd.DataFrame,
    chunking_config: ChunkingConfig,
) -> tuple[list[Document], dict[str, int], int]:
    """
    Build chunked LangChain documents from processed events dataframe.

    Returns:
      - List of chunk documents
      - Mapping event_id -> chunk count
      - Number of invalid events filtered out
    """

    documents: list[Document] = []
    event_to_chunks: dict[str, int] = {}
    invalid_events = 0

    for row in dataframe.to_dict(orient="records"):
        event_id = _clean_text(row.get("event_id"))
        document_text = _clean_text(row.get("document_text"))
        if not event_id or not document_text:
            invalid_events += 1
            continue

        chunks = split_text_into_chunks(document_text, config=chunking_config)
        if not chunks:
            invalid_events += 1
            continue

        base_metadata = _base_metadata_from_row(row)
        event_to_chunks[event_id] = len(chunks)

        for chunk_id, chunk in enumerate(chunks):
            metadata = dict(base_metadata)
            metadata["chunk_id"] = chunk_id
            metadata["chunk_count"] = len(chunks)
            documents.append(Document(page_content=chunk, metadata=metadata))

    return documents, event_to_chunks, invalid_events
