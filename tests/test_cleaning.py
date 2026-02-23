from src.preprocess.cleaning import clean_events, deduplicate_records
from src.preprocess.schema import EVENT_RECORD_FIELDS


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
