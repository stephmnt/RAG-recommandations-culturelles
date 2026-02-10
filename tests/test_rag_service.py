from __future__ import annotations

import json

from src.rag.llm import FakeLLMClient
from src.rag.service import FALLBACK_ANSWER, RAGService
from src.rag.types import RAGConfig, RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk], meta: dict | None = None) -> None:
        self._chunks = chunks
        self._meta = meta or {
            "retriever_top_k": 6,
            "retrieved_chunks": len(chunks),
            "filters_applied": [],
            "warnings": [],
        }
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def retrieve(self, *, question: str, top_k: int | None = None):
        del question, top_k
        return self._chunks, self._meta


def _chunk(event_id: str, title: str, city: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=(f"{title} a {city}. " * 20).strip(),
        metadata={
            "event_id": event_id,
            "title": title,
            "start_datetime": "2025-07-12T20:00:00Z",
            "end_datetime": "2025-07-12T22:00:00Z",
            "city": city,
            "location_name": "Lieu test",
            "url": f"https://example.org/{event_id}",
            "chunk_id": 0,
            "source": "openagenda",
        },
        score=score,
    )


def test_service_returns_fallback_when_no_docs():
    service = RAGService(
        config=RAGConfig(retriever_top_k=6),
        retriever=FakeRetriever(chunks=[]),
        llm_client=FakeLLMClient(),
    )

    result = service.ask("Quels concerts jazz cette semaine ?")

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == []
    assert result.meta["retrieved_chunks"] == 0
    assert "no_retrieved_chunks" in result.meta["warnings"]


def test_service_happy_path_returns_sources_and_answer():
    retriever = FakeRetriever(
        chunks=[
            _chunk("evt-jazz", "Concert jazz", "Montpellier", 0.11),
            _chunk("evt-photo", "Expo photo", "Beziers", 0.32),
        ],
        meta={
            "retriever_top_k": 6,
            "retrieved_chunks": 2,
            "filters_applied": ["city_priority:Montpellier"],
            "warnings": [],
        },
    )
    llm = FakeLLMClient(
        fixed_answer=(
            "Synthese: voici deux recommandations pertinentes.\n"
            "- Concert jazz | 2025-07-12 | Montpellier | https://example.org/evt-jazz\n"
            "Pourquoi ces choix ? Correspondance avec votre demande."
        )
    )
    service = RAGService(
        config=RAGConfig(retriever_top_k=6, max_sources=5),
        retriever=retriever,
        llm_client=llm,
    )

    result = service.ask("Je cherche un concert a Montpellier", debug=True)

    assert retriever.loaded is True
    assert len(result.sources) == 2
    assert result.sources[0].event_id == "evt-jazz"
    assert "Synthese" in result.answer
    assert result.meta["returned_events"] == 2
    assert result.meta["prompt_version"] == "v1"
    assert set(result.meta["latency_ms"].keys()) == {"load_index", "retrieval", "generation"}
    assert "debug" in result.meta


def test_json_serializable_result():
    retriever = FakeRetriever(chunks=[_chunk("evt-jazz", "Concert jazz", "Montpellier", 0.1)])
    service = RAGService(
        config=RAGConfig(retriever_top_k=4),
        retriever=retriever,
        llm_client=FakeLLMClient(fixed_answer="Reponse test"),
    )

    result = service.ask("Question test")
    payload = result.model_dump()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert serialized
    assert payload["question"] == "Question test"
