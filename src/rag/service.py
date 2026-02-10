"""RAG service orchestration (retrieve -> context -> generate)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.rag.context import build_context, build_snippet
from src.rag.llm import LLMClientProtocol, MistralLLMClient
from src.rag.prompts import build_user_prompt, get_system_prompt
from src.rag.retriever import RAGRetriever
from src.rag.types import RAGConfig, RAGResult, RAGSource, RetrievedChunk

LOGGER = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "Je ne peux pas répondre avec certitude à partir des données disponibles. "
    "Vous pouvez reformuler en precisant la ville, la periode et le type d'evenement recherche."
)


class RAGService:
    """Main entrypoint for single-turn RAG question answering."""

    def __init__(
        self,
        config: RAGConfig,
        *,
        retriever: RAGRetriever | None = None,
        llm_client: LLMClientProtocol | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER
        self.retriever = retriever or RAGRetriever(config=config, logger=self.logger)
        self.llm_client = llm_client

    def _get_llm_client(self) -> LLMClientProtocol:
        if self.llm_client is None:
            self.llm_client = MistralLLMClient(
                model=self.config.llm_model,
                max_retries=self.config.llm_max_retries,
                backoff_seconds=self.config.llm_backoff_seconds,
                logger=self.logger,
            )
        return self.llm_client

    def _validate_question(self, question: str) -> str:
        question_clean = (question or "").strip()
        if not question_clean:
            raise ValueError("Question cannot be empty.")
        if len(question_clean) > self.config.max_question_chars:
            raise ValueError(
                f"Question too long ({len(question_clean)} chars). "
                f"Maximum allowed: {self.config.max_question_chars}."
            )
        return question_clean

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[RAGSource]:
        deduped: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            metadata = chunk.metadata or {}
            event_id = str(metadata.get("event_id", "")).strip()
            key = event_id or f"_no_event_{hash(chunk.content)}"
            if key in deduped:
                continue
            deduped[key] = chunk

        sources: list[RAGSource] = []
        for chunk in deduped.values():
            metadata = chunk.metadata or {}
            source = RAGSource(
                event_id=str(metadata.get("event_id", "")),
                title=str(metadata.get("title", "")),
                start_datetime=str(metadata.get("start_datetime", "")),
                end_datetime=(
                    str(metadata.get("end_datetime"))
                    if metadata.get("end_datetime") not in (None, "")
                    else None
                ),
                city=str(metadata.get("city", "")),
                location_name=str(metadata.get("location_name", "")),
                url=str(metadata.get("url", "")),
                score=chunk.score,
                snippet=build_snippet(chunk.content, max_chars=self.config.snippet_max_chars),
            )
            sources.append(source)
            if len(sources) >= self.config.max_sources:
                break
        return sources

    def _base_meta(self, *, top_k: int) -> dict[str, Any]:
        return {
            "retriever_top_k": top_k,
            "retrieved_chunks": 0,
            "returned_events": 0,
            "latency_ms": {
                "load_index": 0,
                "retrieval": 0,
                "generation": 0,
            },
            "model": self.config.llm_model,
            "prompt_version": self.config.prompt_version,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "filters_applied": [],
            "warnings": [],
        }

    def ask(self, question: str, top_k: int | None = None, debug: bool = False) -> RAGResult:
        question_clean = self._validate_question(question)
        effective_top_k = int(top_k or self.config.retriever_top_k)
        meta = self._base_meta(top_k=effective_top_k)

        load_start = time.perf_counter()
        if hasattr(self.retriever, "load"):
            self.retriever.load()
        meta["latency_ms"]["load_index"] = int((time.perf_counter() - load_start) * 1000)

        retrieval_start = time.perf_counter()
        chunks, retriever_meta = self.retriever.retrieve(
            question=question_clean,
            top_k=effective_top_k,
        )
        meta["latency_ms"]["retrieval"] = int((time.perf_counter() - retrieval_start) * 1000)

        meta["retrieved_chunks"] = len(chunks)
        meta["filters_applied"] = list(retriever_meta.get("filters_applied", []))
        meta["warnings"] = list(retriever_meta.get("warnings", []))

        if not chunks:
            meta["warnings"].append("no_retrieved_chunks")
            return RAGResult(
                question=question_clean,
                answer=FALLBACK_ANSWER,
                sources=[],
                meta=meta,
            )

        context, context_meta = build_context(chunks, max_chars=self.config.context_max_chars)
        meta.update(context_meta)

        if not context.strip():
            meta["warnings"].append("empty_context")
            return RAGResult(
                question=question_clean,
                answer=FALLBACK_ANSWER,
                sources=[],
                meta=meta,
            )

        system_prompt = get_system_prompt(prompt_version=self.config.prompt_version)
        user_prompt = build_user_prompt(
            question=question_clean,
            context=context,
            prompt_version=self.config.prompt_version,
        )

        generation_start = time.perf_counter()
        try:
            llm = self._get_llm_client()
            answer = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.llm_temperature,
                timeout_seconds=self.config.llm_timeout_seconds,
            ).strip()
        except Exception as exc:
            self.logger.exception("RAG generation failed: %s", exc)
            answer = FALLBACK_ANSWER
            meta["warnings"].append(f"generation_error:{exc}")
        finally:
            meta["latency_ms"]["generation"] = int((time.perf_counter() - generation_start) * 1000)

        sources = self._build_sources(chunks)
        meta["returned_events"] = len(sources)

        if debug:
            meta["debug"] = {
                "question_chars": len(question_clean),
                "chunk_ids": [
                    {
                        "event_id": str(chunk.metadata.get("event_id", "")),
                        "chunk_id": chunk.metadata.get("chunk_id"),
                        "score": chunk.score,
                    }
                    for chunk in chunks
                ],
                "prompt_chars": len(system_prompt) + len(user_prompt),
            }

        return RAGResult(
            question=question_clean,
            answer=answer or FALLBACK_ANSWER,
            sources=sources,
            meta=meta,
        )
