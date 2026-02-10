"""Retriever wrapper for FAISS with post-processing heuristics."""

from __future__ import annotations

import inspect
import logging
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.indexing.embeddings import get_embedding_model
from src.rag.types import RAGConfig, RetrievedChunk


LOGGER = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().strip()


def _import_faiss_vectorstore():
    try:
        from langchain_community.vectorstores import FAISS

        return FAISS
    except Exception:
        from langchain.vectorstores import FAISS  # type: ignore

        return FAISS


def load_vectorstore(index_path: str | Path, embeddings: Any):
    """Load FAISS index with compatibility across LangChain versions."""

    faiss_cls = _import_faiss_vectorstore()
    signature = inspect.signature(faiss_cls.load_local)
    kwargs: dict[str, Any] = {}
    if "allow_dangerous_deserialization" in signature.parameters:
        kwargs["allow_dangerous_deserialization"] = True
    return faiss_cls.load_local(str(index_path), embeddings, **kwargs)


def _parse_event_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _extract_time_filter(question: str) -> str | None:
    q = _normalize_text(question)
    if "demain" in q:
        return "tomorrow"
    if "ce week-end" in q or "ce weekend" in q or "week-end" in q or "weekend" in q:
        return "weekend"
    if "cette semaine" in q:
        return "this_week"
    if "aujourd'hui" in q or "aujourdhui" in q:
        return "today"
    return None


def _date_matches_filter(event_date: date | None, filter_name: str | None, today: date) -> bool:
    if event_date is None or filter_name is None:
        return False
    if filter_name == "today":
        return event_date == today
    if filter_name == "tomorrow":
        return event_date == today + timedelta(days=1)
    if filter_name == "this_week":
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        return week_start <= event_date <= week_end
    if filter_name == "weekend":
        weekday = today.weekday()
        saturday = today + timedelta(days=(5 - weekday) % 7)
        sunday = saturday + timedelta(days=1)
        return event_date in {saturday, sunday}
    return False


def _extract_requested_city(question: str, candidates: set[str]) -> str | None:
    q = _normalize_text(question)
    for city in sorted(candidates):
        city_norm = _normalize_text(city)
        if not city_norm:
            continue
        if re.search(rf"\\b{re.escape(city_norm)}\\b", q):
            return city
    return None


def _as_retrieved_chunk(document: Any, score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        content=(document.page_content or "").strip(),
        metadata=dict(document.metadata or {}),
        score=float(score) if score is not None else None,
    )


class RAGRetriever:
    def __init__(
        self,
        config: RAGConfig,
        *,
        vectorstore: Any | None = None,
        embedding_model: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.logger = logger or LOGGER

    def load(self) -> None:
        if self.vectorstore is not None:
            return
        if self.embedding_model is None:
            embedding_config = {
                "embeddings": {
                    "provider": self.config.embedding_provider,
                    "huggingface_model": self.config.embedding_model,
                }
            }
            self.embedding_model = get_embedding_model(config=embedding_config, logger=self.logger)
        self.vectorstore = load_vectorstore(self.config.index_path, self.embedding_model)

    def retrieve(
        self,
        *,
        question: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        self.load()

        requested_k = int(top_k or self.config.retriever_top_k)
        raw_chunks: list[RetrievedChunk] = []
        warnings: list[str] = []
        filters_applied: list[str] = []
        scores_available = True

        assert self.vectorstore is not None
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=requested_k)
            for document, score in docs_with_scores:
                raw_chunks.append(_as_retrieved_chunk(document, score=score))
        except Exception:
            scores_available = False
            docs = self.vectorstore.similarity_search(question, k=requested_k)
            for document in docs:
                raw_chunks.append(_as_retrieved_chunk(document))
            warnings.append("scores_non_disponibles")

        cleaned_chunks: list[RetrievedChunk] = []
        for chunk in raw_chunks:
            if len(chunk.content.strip()) < self.config.min_chunk_chars:
                continue
            if self.config.score_threshold is not None and chunk.score is not None:
                if chunk.score > self.config.score_threshold:
                    continue
            cleaned_chunks.append(chunk)

        known_cities = {
            str(chunk.metadata.get("city", "")).strip()
            for chunk in cleaned_chunks
            if str(chunk.metadata.get("city", "")).strip()
        }
        requested_city = _extract_requested_city(question, known_cities)
        if requested_city:
            filters_applied.append(f"city_priority:{requested_city}")

        time_filter = _extract_time_filter(question)
        if time_filter:
            filters_applied.append(f"time_priority:{time_filter}")

        today = date.today()

        def sort_key(chunk: RetrievedChunk) -> tuple[Any, ...]:
            # score behavior for FAISS/L2: lower score is better
            score_key = chunk.score if chunk.score is not None else float("inf")
            city_match = 0
            if requested_city and str(chunk.metadata.get("city", "")).strip() == requested_city:
                city_match = -1

            time_match = 0
            if time_filter:
                event_date = _parse_event_date(chunk.metadata.get("start_datetime"))
                if _date_matches_filter(event_date, time_filter, today=today):
                    time_match = -1
            return (city_match, time_match, score_key)

        cleaned_chunks.sort(key=sort_key)

        deduped_by_event: dict[str, RetrievedChunk] = {}
        for chunk in cleaned_chunks:
            event_id = str(chunk.metadata.get("event_id", "")).strip()
            key = event_id or f"_fallback_{hash(chunk.content)}"
            if key in deduped_by_event:
                continue
            deduped_by_event[key] = chunk

        final_chunks = list(deduped_by_event.values())
        final_chunks = final_chunks[: self.config.max_sources]

        meta = {
            "retriever_top_k": requested_k,
            "raw_chunks": len(raw_chunks),
            "retrieved_chunks": len(final_chunks),
            "scores_available": scores_available,
            "filters_applied": filters_applied,
            "warnings": warnings,
        }
        return final_chunks, meta
