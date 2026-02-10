from src.preprocess.schema import validate_record


def test_schema_document_text():
    payload = {
        "event_id": "evt-123",
        "title": "Concert test",
        "description": "",
        "start_datetime": "2025-08-15T20:00:00Z",
        "end_datetime": "",
        "city": "Lyon",
        "location_name": "Le Sucre",
        "address": "50 quai Rambaud",
        "latitude": 45.7337,
        "longitude": 4.8205,
        "url": "https://example.org/events/evt-123",
        "tags": ["musique"],
        "source": "openagenda",
        "document_text": "",
        "retrieval_metadata": {},
    }

    record = validate_record(payload)

    assert record.document_text.strip() != ""
    assert "Concert test" in record.document_text
