import hashlib
import time
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("faiss")

from src.indexing.build_index import build_faiss_index, load_faiss_index
from src.indexing.search import search_similar_chunks


class FakeEmbeddings:
    """Deterministic embeddings for fully offline tests."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for index, token in enumerate(text.lower().split()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = digest[0] % self.dimension
            value = (digest[1] / 255.0) + 0.01
            vector[(bucket + index) % self.dimension] += value
        norm = sum(item * item for item in vector) ** 0.5 or 1.0
        return [item / norm for item in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        # Compatibility with langchain versions that call embedding_function(text)
        return self.embed_query(text)


@pytest.fixture
def mini_events_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "evt-jazz",
                "title": "Concert jazz a Montpellier",
                "document_text": (
                    "Concert jazz en plein air a Montpellier avec trio local et jam session."
                ),
                "start_datetime": "2025-07-12T20:00:00Z",
                "end_datetime": "2025-07-12T22:00:00Z",
                "city": "Montpellier",
                "location_name": "Place de la Comedie",
                "url": "https://example.org/evt-jazz",
                "source": "openagenda",
                "retrieval_metadata": {"tags": ["jazz", "musique"], "department": "34"},
            },
            {
                "event_id": "evt-theatre",
                "title": "Piece de theatre",
                "document_text": "Representation de theatre contemporain au centre dramatique.",
                "start_datetime": "2025-07-20T19:00:00Z",
                "end_datetime": "2025-07-20T21:00:00Z",
                "city": "Sete",
                "location_name": "Scene nationale",
                "url": "https://example.org/evt-theatre",
                "source": "openagenda",
                "retrieval_metadata": {"tags": ["theatre"], "department": "34"},
            },
            {
                "event_id": "evt-photo",
                "title": "Expo photo",
                "document_text": "Exposition photographique sur le patrimoine mediterraneen.",
                "start_datetime": "2025-08-04T10:00:00Z",
                "end_datetime": "2025-08-04T18:00:00Z",
                "city": "Beziers",
                "location_name": "Galerie municipale",
                "url": "https://example.org/evt-photo",
                "source": "openagenda",
                "retrieval_metadata": {"tags": ["photo"], "department": "34"},
            },
            {
                "event_id": "evt-invalid",
                "title": "Invalide",
                "document_text": "",
                "start_datetime": "2025-08-04T10:00:00Z",
                "city": "Montpellier",
                "retrieval_metadata": {"department": "34"},
            },
        ]
    )


def _indexing_config(input_path: Path, output_dir: Path) -> dict:
    return {
        "paths": {
            "input_dataset": str(input_path),
            "output_dir": str(output_dir),
        },
        "chunking": {
            "chunk_size": 180,
            "chunk_overlap": 30,
            "min_chunk_size": 40,
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


def test_build_index_counts(tmp_path: Path, mini_events_dataframe: pd.DataFrame):
    input_path = tmp_path / "events.parquet"
    output_dir = tmp_path / "faiss_index"
    mini_events_dataframe.to_parquet(input_path, index=False)

    result = build_faiss_index(
        input_path=input_path,
        output_dir=output_dir,
        config=_indexing_config(input_path, output_dir),
        embedding_model=FakeEmbeddings(),
    )

    assert result.events_input == 4
    assert result.events_invalid == 1
    assert result.events_valid == 3
    assert result.chunks_count >= 3
    assert (output_dir / "index_metadata.json").exists()


def test_save_and_load_index_roundtrip(tmp_path: Path, mini_events_dataframe: pd.DataFrame):
    input_path = tmp_path / "events.parquet"
    output_dir = tmp_path / "faiss_index"
    mini_events_dataframe.to_parquet(input_path, index=False)

    fake_embeddings = FakeEmbeddings()
    build_faiss_index(
        input_path=input_path,
        output_dir=output_dir,
        config=_indexing_config(input_path, output_dir),
        embedding_model=fake_embeddings,
    )

    vectorstore = load_faiss_index(
        index_dir=output_dir,
        config=_indexing_config(input_path, output_dir),
        embedding_model=fake_embeddings,
    )
    docs = vectorstore.similarity_search("concert jazz montpellier", k=2)
    assert len(docs) == 2
    assert all(doc.page_content for doc in docs)


def test_search_returns_relevant_event_id(tmp_path: Path, mini_events_dataframe: pd.DataFrame):
    input_path = tmp_path / "events.parquet"
    output_dir = tmp_path / "faiss_index"
    mini_events_dataframe.to_parquet(input_path, index=False)

    fake_embeddings = FakeEmbeddings()
    build_faiss_index(
        input_path=input_path,
        output_dir=output_dir,
        config=_indexing_config(input_path, output_dir),
        embedding_model=fake_embeddings,
    )

    results = search_similar_chunks(
        query="Je cherche un concert jazz a Montpellier",
        index_dir=output_dir,
        k=3,
        config=_indexing_config(input_path, output_dir),
        embedding_model=fake_embeddings,
    )
    assert len(results) >= 1
    assert results[0]["metadata"]["event_id"] == "evt-jazz"


@pytest.mark.slow
def test_performance_smoke(tmp_path: Path):
    input_path = tmp_path / "events_200.parquet"
    output_dir = tmp_path / "faiss_index"

    rows = []
    for idx in range(200):
        rows.append(
            {
                "event_id": f"evt-{idx:04d}",
                "title": f"Event {idx}",
                "document_text": (
                    f"Evenement culturel numero {idx} dans l'Herault avec musique et exposition."
                ),
                "start_datetime": "2025-09-10T18:00:00Z",
                "end_datetime": "2025-09-10T20:00:00Z",
                "city": "Montpellier",
                "location_name": "Lieu test",
                "url": f"https://example.org/events/{idx}",
                "source": "openagenda",
                "retrieval_metadata": {"department": "34"},
            }
        )
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    started = time.perf_counter()
    result = build_faiss_index(
        input_path=input_path,
        output_dir=output_dir,
        config=_indexing_config(input_path, output_dir),
        embedding_model=FakeEmbeddings(),
    )
    elapsed = time.perf_counter() - started

    assert result.chunks_count >= 200
    assert elapsed < 20.0
