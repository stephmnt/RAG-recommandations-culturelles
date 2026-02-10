"""Context construction from retrieved chunks."""

from __future__ import annotations

from typing import Any

from src.rag.types import RetrievedChunk


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_snippet(text: str, min_chars: int = 200, max_chars: int = 320) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    trimmed = clean[:max_chars].rsplit(" ", 1)[0]
    if len(trimmed) < min_chars:
        trimmed = clean[:max_chars]
    return trimmed + "..."


def _format_block(chunk: RetrievedChunk, rank: int) -> str:
    metadata = chunk.metadata or {}
    score_display = "n/a" if chunk.score is None else f"{chunk.score:.6f}"
    return (
        f"[RANK={rank} | EVENT_ID={_clean_text(metadata.get('event_id'))} | "
        f"CHUNK={_clean_text(metadata.get('chunk_id'))} | SCORE={score_display}]\n"
        f"Titre: {_clean_text(metadata.get('title'))}\n"
        f"Date debut: {_clean_text(metadata.get('start_datetime'))}\n"
        f"Date fin: {_clean_text(metadata.get('end_datetime'))}\n"
        f"Ville: {_clean_text(metadata.get('city'))}\n"
        f"Lieu: {_clean_text(metadata.get('location_name'))}\n"
        f"URL: {_clean_text(metadata.get('url'))}\n"
        f"Texte: {_clean_text(chunk.content)}\n"
        "---\n"
    )


def build_context(
    chunks: list[RetrievedChunk],
    max_chars: int = 8000,
) -> tuple[str, dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    blocks: list[str] = []
    included_event_ids: list[str] = []
    total_chars = 0
    context_truncated = False

    for rank, chunk in enumerate(chunks, start=1):
        block = _format_block(chunk, rank=rank)
        projected = total_chars + len(block)
        if projected > max_chars:
            context_truncated = True
            if not blocks:
                remaining = max_chars - total_chars
                if remaining > 0:
                    truncated_block = block[:remaining].rstrip()
                    if truncated_block:
                        blocks.append(truncated_block)
                        total_chars += len(truncated_block)
                        included_event_ids.append(_clean_text(chunk.metadata.get("event_id")))
            break
        blocks.append(block)
        total_chars += len(block)
        included_event_ids.append(_clean_text(chunk.metadata.get("event_id")))

    context = "".join(blocks).strip()
    meta = {
        "context_truncated": context_truncated,
        "context_chars": len(context),
        "context_chunks": len(blocks),
        "context_event_ids": [event_id for event_id in included_event_ids if event_id],
    }
    return context, meta
