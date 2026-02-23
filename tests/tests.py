from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

from src.api.app import create_app
from src.indexing.chunking import (
    ChunkingConfig,
    build_documents_from_dataframe,
    split_text_into_chunks,
)
from src.openagenda.client import OpenAgendaConfig, fetch_events
from src.preprocess.cleaning import clean_events, deduplicate_records
from src.preprocess.schema import EVENT_RECORD_FIELDS, validate_record
from src.rag.context import build_context
from src.rag.llm import FakeLLMClient
from src.rag.prompts import build_user_prompt, get_system_prompt
from src.rag.retriever import RAGRetriever
from src.rag.service import FALLBACK_ANSWER, RAGService
from src.rag.types import RAGConfig, RetrievedChunk

FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None
if FAISS_AVAILABLE:
    from src.indexing.build_index import build_faiss_index, load_faiss_index
    from src.indexing.search import search_similar_chunks


# =========================
# API /ask
# =========================
class FakeDepsAsk:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    def ask(
        self,
        *,
        question: str,
        top_k: int,
        debug: bool,
        filters: dict[str, Any] | None = None,
    ):
        self.last_call = {
            "question": question,
            "top_k": top_k,
            "debug": debug,
            "filters": filters,
        }
        return {
            "question": question,
            "answer": "Reponse mockee",
            "sources": [
                {
                    "event_id": "evt-1",
                    "title": "Concert jazz",
                    "start_datetime": "2026-02-14T20:00:00Z",
                    "end_datetime": None,
                    "city": "Montpellier",
                    "location_name": "Salle A",
                    "url": "https://example.org/evt-1",
                    "score": 0.21,
                    "snippet": "Concert jazz a Montpellier",
                }
            ],
            "meta": {
                "retriever_top_k": top_k,
                "retrieved_chunks": 1,
                "returned_events": 1,
                "latency_ms": {"load_index": 1, "retrieval": 2, "generation": 3},
                "model": "mistral-small-latest",
                "prompt_version": "v1",
                "timestamp": "2026-02-10T12:00:00Z",
                "warnings": [],
            },
        }

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": True,
            "mistral_configured": True,
            "version": "0.1.0",
            "timestamp": "2026-02-10T12:00:00Z",
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        return {
            "index": {
                "path": "artifacts/faiss_index",
                "build_date": "2026-02-10T10:00:00Z",
                "num_events": 12,
                "num_chunks": 24,
                "embedding_model": "intfloat/multilingual-e5-small",
                "dataset_hash": "abc",
            },
            "rag": {
                "default_top_k": 6,
                "max_top_k": 10,
                "prompt_version": "v1",
                "llm_model": "mistral-small-latest",
            },
        }


def _build_client_ask(fake_deps: FakeDepsAsk):
    app = create_app(
        config_overrides={
            "ADMIN_TOKEN": "secret-token",
            "MAX_TOP_K": 10,
            "DEFAULT_TOP_K": 6,
        },
        deps_override=fake_deps,
    )
    return app.test_client()


def test_ask_happy_path_returns_rag_result():
    fake_deps = FakeDepsAsk()
    client = _build_client_ask(fake_deps)

    response = client.post(
        "/ask",
        json={
            "question": "Quels concerts jazz dans l'Herault ?",
            "top_k": 4,
            "debug": True,
            "filters": {"city": "Montpellier"},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["question"] == "Quels concerts jazz dans l'Herault ?"
    assert payload["answer"] == "Reponse mockee"
    assert len(payload["sources"]) == 1
    assert fake_deps.last_call is not None
    assert fake_deps.last_call["top_k"] == 4
    assert fake_deps.last_call["filters"] == {"city": "Montpellier"}


def test_ask_question_empty_returns_400():
    fake_deps = FakeDepsAsk()
    client = _build_client_ask(fake_deps)

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_ask_top_k_is_clamped_to_max():
    fake_deps = FakeDepsAsk()
    client = _build_client_ask(fake_deps)

    response = client.post(
        "/ask",
        json={
            "question": "Donne-moi des sorties culturelles",
            "top_k": 999,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert fake_deps.last_call is not None
    assert fake_deps.last_call["top_k"] == 10
    warnings = payload["meta"]["warnings"]
    assert any(str(item).startswith("top_k_clamped") for item in warnings)


# =========================
# API /health and /metadata
# =========================
class FakeDepsHealth:
    def ask(
        self,
        *,
        question: str,
        top_k: int,
        debug: bool,
        filters: dict[str, Any] | None = None,
    ):
        del question, top_k, debug, filters
        return {
            "question": "q",
            "answer": "a",
            "sources": [],
            "meta": {},
        }

    def rebuild_index(self, *, dataset_path: str | None = None, index_path: str | None = None):
        del dataset_path, index_path
        return {
            "status": "ok",
            "mode": "rebuild",
            "index_path": "artifacts/faiss_index",
            "dataset_path": "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 10},
            "timings_ms": {"build": 4, "reload": 2, "total": 6},
        }

    def reload_index(self, *, index_path: str | None = None):
        del index_path
        return {
            "status": "ok",
            "mode": "reload",
            "index_path": "artifacts/faiss_index",
            "dataset_path": "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 10},
            "timings_ms": {"total": 2},
        }

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": True,
            "mistral_configured": False,
            "version": "0.1.0",
            "timestamp": "2026-02-10T12:00:00Z",
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        return {
            "index": {
                "path": "artifacts/faiss_index",
                "build_date": "2026-02-10T10:00:00Z",
                "num_events": 123,
                "num_chunks": 456,
                "embedding_model": "intfloat/multilingual-e5-small",
                "dataset_hash": "abc123",
            },
            "rag": {
                "default_top_k": 6,
                "max_top_k": 10,
                "prompt_version": "v1",
                "llm_model": "mistral-small-latest",
            },
        }


def _build_client_health(fake_deps: FakeDepsHealth):
    app = create_app(
        config_overrides={
            "ADMIN_TOKEN": "secret-token",
        },
        deps_override=fake_deps,
    )
    return app.test_client()


def test_health_returns_expected_fields():
    client = _build_client_health(FakeDepsHealth())

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["api"] == "up"
    assert "index_loaded" in payload
    assert "mistral_configured" in payload
    assert "timestamp" in payload


def test_metadata_returns_index_and_rag_sections():
    client = _build_client_health(FakeDepsHealth())

    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.get_json()
    assert "index" in payload
    assert "rag" in payload
    assert payload["index"]["num_chunks"] == 456
    assert payload["rag"]["default_top_k"] == 6


# =========================
# API /rebuild
# =========================
class FakeDepsRebuild:
    def __init__(self) -> None:
        self.rebuild_calls = 0
        self.reload_calls = 0
        self.last_payload: dict[str, Any] | None = None

    def ask(
        self,
        *,
        question: str,
        top_k: int,
        debug: bool,
        filters: dict[str, Any] | None = None,
    ):
        del question, top_k, debug, filters
        return {
            "question": "q",
            "answer": "a",
            "sources": [],
            "meta": {},
        }

    def rebuild_index(self, *, dataset_path: str | None = None, index_path: str | None = None):
        self.rebuild_calls += 1
        self.last_payload = {
            "dataset_path": dataset_path,
            "index_path": index_path,
        }
        return {
            "status": "ok",
            "mode": "rebuild",
            "index_path": index_path or "artifacts/faiss_index",
            "dataset_path": dataset_path or "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 42},
            "timings_ms": {"build": 12, "reload": 4, "total": 16},
        }

    def reload_index(self, *, index_path: str | None = None):
        self.reload_calls += 1
        self.last_payload = {
            "index_path": index_path,
        }
        return {
            "status": "ok",
            "mode": "reload",
            "index_path": index_path or "artifacts/faiss_index",
            "dataset_path": "data/processed/events_processed.parquet",
            "index_metadata": {"chunks_count": 42},
            "timings_ms": {"total": 3},
        }

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": True,
            "mistral_configured": True,
            "version": "0.1.0",
            "timestamp": "2026-02-10T12:00:00Z",
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        return {
            "index": {
                "path": "artifacts/faiss_index",
                "build_date": None,
                "num_events": None,
                "num_chunks": None,
                "embedding_model": "",
                "dataset_hash": None,
            },
            "rag": {
                "default_top_k": 6,
                "max_top_k": 10,
                "prompt_version": "v1",
                "llm_model": "mistral-small-latest",
            },
        }


def _build_client_rebuild(fake_deps: FakeDepsRebuild):
    app = create_app(
        config_overrides={
            "ADMIN_TOKEN": "secret-token",
        },
        deps_override=fake_deps,
    )
    return app.test_client()


def test_rebuild_without_token_returns_401():
    client = _build_client_rebuild(FakeDepsRebuild())

    response = client.post("/rebuild", json={"mode": "reload"})

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"]["code"] == "MISSING_ADMIN_TOKEN"


def test_rebuild_with_invalid_token_returns_403():
    client = _build_client_rebuild(FakeDepsRebuild())

    response = client.post(
        "/rebuild",
        json={"mode": "reload"},
        headers={"X-ADMIN-TOKEN": "wrong-token"},
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_ADMIN_TOKEN"


def test_rebuild_mode_invalid_returns_400():
    client = _build_client_rebuild(FakeDepsRebuild())

    response = client.post(
        "/rebuild",
        json={"mode": "invalid-mode"},
        headers={"X-ADMIN-TOKEN": "secret-token"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_rebuild_with_token_ok_returns_200():
    fake_deps = FakeDepsRebuild()
    client = _build_client_rebuild(fake_deps)

    response = client.post(
        "/rebuild",
        json={"mode": "rebuild", "dataset_path": "data/processed/events_processed.parquet"},
        headers={"X-ADMIN-TOKEN": "secret-token"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "rebuild"
    assert fake_deps.rebuild_calls == 1


# =========================
# API web page
# =========================
class FakeDepsWeb:
    def ask(
        self,
        *,
        question: str,
        top_k: int,
        debug: bool,
        filters: dict[str, Any] | None = None,
    ):
        del question, top_k, debug, filters
        return {
            "question": "q",
            "answer": "a",
            "sources": [],
            "meta": {},
        }

    def get_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "api": "up",
            "index_loaded": True,
            "mistral_configured": False,
            "version": "0.1.0",
            "timestamp": "2026-02-10T12:00:00Z",
        }

    def get_metadata_payload(self) -> dict[str, Any]:
        return {
            "index": {"path": "artifacts/faiss_index"},
            "rag": {"default_top_k": 6},
        }


def _build_client_web():
    app = create_app(
        config_overrides={"ADMIN_TOKEN": "secret-token"},
        deps_override=FakeDepsWeb(),
    )
    return app.test_client()


def test_home_page_is_served():
    client = _build_client_web()
    response = client.get("/")

    assert response.status_code == 200
    assert response.content_type.startswith("text/html")
    body = response.get_data(as_text=True)
    assert "Puls-Events" in body
    assert 'id="signup-form"' in body


def test_app_alias_is_served():
    client = _build_client_web()
    response = client.get("/app")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="question"' in body
    assert "top_k (nb de sources explorees)" in body


def test_html5up_assets_are_served():
    client = _build_client_web()
    response = client.get("/assets/css/main.css")

    assert response.status_code == 200
    assert response.content_type.startswith("text/css")


# =========================
# OpenAgenda client
# =========================
def test_client_pagination_mocked(requests_mock):
    api_root = "https://api.openagenda.com/v2"
    agendas_url = f"{api_root}/agendas"
    events_url = f"{api_root}/agendas/agenda-1/events"

    requests_mock.get(
        agendas_url,
        [
            {"json": {"agendas": [{"uid": "agenda-1", "title": {"fr": "Agenda Montpellier"}}]}},
            {"json": {"agendas": []}},
        ],
    )
    requests_mock.get(
        events_url,
        [
            {"json": {"events": [{"uid": "evt-1", "title": {"fr": "Evenement 1"}}]}},
            {"json": {"events": [{"uid": "evt-2", "title": {"fr": "Evenement 2"}}]}},
            {"json": {"events": []}},
        ],
    )

    config = OpenAgendaConfig(
        base_url=api_root,
        api_key="fake-key",
        page_size=1,
        max_pages=5,
        max_events=10,
        agenda_search_queries=["Montpellier"],
        agenda_page_size=1,
        agenda_max_pages=5,
        max_agendas=5,
        start_date="2025-01-01",
        end_date="2026-12-31",
    )

    events = fetch_events(config)

    assert [event["uid"] for event in events] == ["evt-1", "evt-2"]
    assert all(event["agenda_uid"] == "agenda-1" for event in events)
    assert requests_mock.call_count == 5
    event_requests = [
        request for request in requests_mock.request_history if request.url.startswith(events_url)
    ]
    assert len(event_requests) == 3
    first_query = event_requests[0].qs
    assert first_query["timings[gte]"] == ["2025-01-01T00:00:00.000Z"]
    assert first_query["timings[lte]"] == ["2026-12-31T23:59:59.999Z"]
    assert first_query["detailed"] == ["1"]


# =========================
# Cleaning
# =========================
def _raw_event(uid: str, title: str, start_iso: str) -> dict:
    return {
        "uid": uid,
        "title": {"fr": title},
        "description": {"fr": "Description de test"},
        "firstTiming": {"begin": start_iso},
        "location": {
            "name": {"fr": "Salle test"},
            "address": "10 rue de la Republique",
            "city": "Montpellier",
            "department": "Hérault",
            "region": "Occitanie",
            "latitude": 43.6119,
            "longitude": 3.8772,
        },
        "canonicalUrl": f"https://example.org/events/{uid}",
        "tags": ["culture", "concert"],
    }


def test_cleaning_filters_period():
    raw_events = [
        _raw_event("evt-old", "Evenement trop ancien", "2024-01-10T19:00:00Z"),
        _raw_event("evt-ok", "Evenement valide", "2025-06-01T19:00:00Z"),
        _raw_event("evt-future", "Evenement trop loin", "2026-03-01T19:00:00Z"),
    ]

    cleaned, stats = clean_events(
        raw_events=raw_events,
        start_date="2025-01-01",
        end_date="2026-01-31",
    )

    assert len(cleaned) == 1
    assert cleaned[0]["event_id"] == "evt-ok"
    assert stats["outside_period"] == 2


def test_deduplication():
    records = [
        {
            "event_id": "evt-1",
            "url": "https://example.org/events/evt-1",
            "title": "A",
            "start_datetime": "2025-06-01T19:00:00Z",
            "city": "Montpellier",
        },
        {
            "event_id": "evt-1",
            "url": "https://example.org/events/evt-1-dup",
            "title": "A",
            "start_datetime": "2025-06-01T19:00:00Z",
            "city": "Montpellier",
        },
        {
            "event_id": "evt-2",
            "url": "https://example.org/events/evt-2",
            "title": "B",
            "start_datetime": "2025-06-03T19:00:00Z",
            "city": "Montpellier",
        },
        {
            "event_id": "evt-2",
            "url": "https://example.org/events/evt-2-other",
            "title": "B",
            "start_datetime": "2025-06-03T19:00:00Z",
            "city": "Montpellier",
        },
    ]

    deduped, duplicates_removed = deduplicate_records(records)

    assert len(deduped) == 2
    assert duplicates_removed == 2


def test_minimum_fields_present():
    raw_events = [_raw_event("evt-schema", "Evenement schema", "2025-07-10T18:30:00Z")]

    cleaned, stats = clean_events(
        raw_events=raw_events,
        start_date="2025-01-01",
        end_date="2026-01-31",
    )

    assert stats["processed_events"] == 1
    assert set(cleaned[0].keys()) == set(EVENT_RECORD_FIELDS)


def test_url_normalization_ignores_empty_dict_url():
    raw_event = _raw_event("evt-url", "Evenement url", "2025-07-10T18:30:00Z")
    raw_event["canonicalUrl"] = {}
    raw_event["url"] = {}
    raw_event["link"] = {}

    cleaned, stats = clean_events(
        raw_events=[raw_event],
        start_date="2025-01-01",
        end_date="2026-01-31",
    )

    assert stats["processed_events"] == 1
    assert cleaned[0]["url"] == ""


def test_cleaning_geo_scope_occitanie_filters_external_city():
    bordeaux = _raw_event("evt-bdx", "Evenement Bordeaux", "2025-07-10T18:30:00Z")
    bordeaux["location"]["city"] = "Bordeaux"
    bordeaux["location"]["department"] = "Gironde"
    bordeaux["location"]["region"] = "Nouvelle-Aquitaine"
    bordeaux["location"]["latitude"] = 44.8378
    bordeaux["location"]["longitude"] = -0.5792

    saint_gaudens = _raw_event("evt-stg", "Evenement Saint-Gaudens", "2025-07-11T18:30:00Z")
    saint_gaudens["location"]["city"] = "Saint-Gaudens"
    saint_gaudens["location"]["department"] = "Haute-Garonne"
    saint_gaudens["location"]["region"] = "Occitanie"
    saint_gaudens["location"]["latitude"] = 43.1086
    saint_gaudens["location"]["longitude"] = 0.7233

    cleaned, stats = clean_events(
        raw_events=[bordeaux, saint_gaudens],
        start_date="2025-01-01",
        end_date="2026-01-31",
        geo_scope={"mode": "occitanie", "strict": True},
    )

    assert stats["outside_geo_scope"] == 1
    assert stats["processed_events"] == 1
    assert cleaned[0]["city"] == "Saint-Gaudens"
    assert stats["external_city_counts"].get("Bordeaux") == 1


# =========================
# Indexing chunking
# =========================
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


# =========================
# Indexing pipeline
# =========================
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


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
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


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
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


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
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
@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
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


# =========================
# Prompting
# =========================
def test_prompt_contains_guardrails():
    system_prompt = get_system_prompt("v1")

    assert "Tu reponds uniquement a partir du CONTEXTE fourni." in system_prompt
    assert "Tu n'inventes jamais" in system_prompt
    assert "Je ne peux pas répondre avec certitude à partir des données disponibles." in system_prompt


def test_user_prompt_contains_required_structure():
    prompt = build_user_prompt(
        question="Quels concerts jazz a Montpellier cette semaine ?",
        context="[EVENT_ID=evt-1] ...",
        prompt_version="v1",
    )

    assert "QUESTION UTILISATEUR" in prompt
    assert "CONTEXTE" in prompt
    assert "Pourquoi ces choix ?" in prompt
    assert "titre, date, lieu, ville, URL" in prompt


def test_unsupported_prompt_version_raises():
    with pytest.raises(ValueError):
        get_system_prompt("v2")


# =========================
# Retriever/context
# =========================
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


# =========================
# RAG service
# =========================
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


# =========================
# Schema
# =========================
def test_schema_document_text():
    payload = {
        "event_id": "evt-123",
        "title": "Concert test",
        "description": "",
        "start_datetime": "2025-08-15T20:00:00Z",
        "end_datetime": "",
        "city": "Montpellier",
        "location_name": "Corum",
        "address": "Esplanade Charles de Gaulle",
        "latitude": 43.6112,
        "longitude": 3.8827,
        "url": "https://example.org/events/evt-123",
        "tags": ["musique"],
        "source": "openagenda",
        "document_text": "",
        "retrieval_metadata": {},
    }

    record = validate_record(payload)

    assert record.document_text.strip() != ""
    assert "Concert test" in record.document_text
