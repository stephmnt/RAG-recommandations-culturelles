from src.preprocess.schema import validate_record


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
