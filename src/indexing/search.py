"""Search helpers for persisted FAISS index."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.indexing.build_index import load_faiss_index


LOGGER = logging.getLogger(__name__)


def search_similar_chunks(
    *,
    query: str,
    index_dir: str | Path,
    k: int = 5,
    config: dict[str, Any] | None = None,
    embedding_model: Any | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    logger = logger or LOGGER
    vectorstore = load_faiss_index(
        index_dir=index_dir,
        config=config,
        embedding_model=embedding_model,
        logger=logger,
    )

    results: list[dict[str, Any]] = []
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        for rank, (doc, score) in enumerate(docs_with_scores, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
    except Exception:
        docs = vectorstore.similarity_search(query, k=k)
        for rank, doc in enumerate(docs, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": None,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

    return results
