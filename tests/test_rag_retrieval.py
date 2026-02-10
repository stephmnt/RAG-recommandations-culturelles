from __future__ import annotations

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

from src.rag.context import build_context
from src.rag.retriever import RAGRetriever
from src.rag.types import RAGConfig, RetrievedChunk


class FakeVectorStoreWithScores:
    def __init__(self, docs_with_scores: list[tuple[Document, float]]) -> None:
        self.docs_with_scores = docs_with_scores

    def similarity_search_with_score(self, query: str, k: int):
        del query
        return self.docs_with_scores[:k]


def _doc(event_id: str, city: str, content: str, chunk_id: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={
            "event_id": event_id,
            "title": f"Titre {event_id}",
            "start_datetime": "2025-07-15T20:00:00Z",
            "end_datetime": "2025-07-15T22:00:00Z",
            "city": city,
            "location_name": "Salle Test",
            "url": f"https://example.org/{event_id}",
            "chunk_id": chunk_id,
            "source": "openagenda",
        },
    )


def test_retriever_deduplicates_by_event_id():
    docs_with_scores = [
        (_doc("evt-jazz", "Montpellier", "concert jazz local", 1), 0.65),
        (_doc("evt-jazz", "Montpellier", "concert jazz local meilleur chunk", 0), 0.12),
        (_doc("evt-theatre", "Sete", "piece theatre contemporain", 0), 0.25),
    ]
    vectorstore = FakeVectorStoreWithScores(docs_with_scores)
    config = RAGConfig(retriever_top_k=6, max_sources=5, min_chunk_chars=5)

    retriever = RAGRetriever(
        config=config,
        vectorstore=vectorstore,
        embedding_model=object(),
    )
    chunks, meta = retriever.retrieve(question="Je cherche du jazz a Montpellier", top_k=6)

    assert len(chunks) == 2
    assert chunks[0].metadata["event_id"] == "evt-jazz"
    assert chunks[0].content == "concert jazz local meilleur chunk"
    assert meta["retrieved_chunks"] == 2


def test_context_builder_limits_length():
    chunks = [
        RetrievedChunk(
            content=("Texte long evenement A. " * 80).strip(),
            metadata={"event_id": "evt-a", "chunk_id": 0, "title": "A", "city": "Montpellier"},
            score=0.2,
        ),
        RetrievedChunk(
            content=("Texte long evenement B. " * 80).strip(),
            metadata={"event_id": "evt-b", "chunk_id": 0, "title": "B", "city": "Sete"},
            score=0.4,
        ),
    ]

    context, meta = build_context(chunks, max_chars=550)

    assert context.strip() != ""
    assert meta["context_truncated"] is True
    assert meta["context_chars"] <= 550
    assert meta["context_chunks"] >= 1
