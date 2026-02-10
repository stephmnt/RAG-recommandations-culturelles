import json

import pandas as pd

from src.indexing.chunking import (
    ChunkingConfig,
    build_documents_from_dataframe,
    split_text_into_chunks,
)


def test_chunking_produces_expected_ranges():
    config = ChunkingConfig(
        chunk_size=160,
        chunk_overlap=30,
        min_chunk_size=40,
        separators=(" ", ""),
    )
    text = " ".join(f"mot_{idx}" for idx in range(220))

    chunks = split_text_into_chunks(text, config=config)

    assert len(chunks) >= 3
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) >= config.min_chunk_size for chunk in chunks[:-1])
    assert all(len(chunk) <= (config.chunk_size + config.chunk_overlap + 40) for chunk in chunks)

    overlap_tokens = chunks[0][-35:].split()
    assert any(token in chunks[1] for token in overlap_tokens if token)


def test_documents_metadata_json_serializable():
    dataframe = pd.DataFrame(
        [
            {
                "event_id": "evt-001",
                "title": "Concert jazz",
                "document_text": "Concert jazz au parc de Montpellier. " * 8,
                "start_datetime": "2025-07-01T20:00:00Z",
                "end_datetime": "2025-07-01T22:00:00Z",
                "city": "Montpellier",
                "location_name": "Parc Montcalm",
                "url": "https://example.org/evenements/evt-001",
                "retrieval_metadata": {
                    "tags": ["jazz", "musique"],
                    "nested": {"department": "34"},
                    "score": 0.92,
                },
            }
        ]
    )
    config = ChunkingConfig(chunk_size=120, chunk_overlap=20, min_chunk_size=30)

    documents, event_to_chunks, invalid_events = build_documents_from_dataframe(
        dataframe=dataframe,
        chunking_config=config,
    )

    assert invalid_events == 0
    assert event_to_chunks["evt-001"] == len(documents)
    assert len(documents) > 0

    for document in documents:
        json.dumps(document.metadata)
