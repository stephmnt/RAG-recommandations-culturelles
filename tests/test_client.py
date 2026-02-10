from src.openagenda.client import OpenAgendaConfig, fetch_events


def test_client_pagination_mocked(requests_mock):
    url = "https://api.openagenda.com/v2/events"
    requests_mock.get(
        url,
        [
            {"json": {"events": [{"uid": "evt-1", "title": {"fr": "Evenement 1"}}]}},
            {"json": {"events": [{"uid": "evt-2", "title": {"fr": "Evenement 2"}}]}},
            {"json": {"events": []}},
        ],
    )

    config = OpenAgendaConfig(
        base_url=url,
        api_key="fake-key",
        page_size=1,
        max_pages=5,
        max_events=10,
        start_date="2025-01-01",
        end_date="2026-12-31",
    )

    events = fetch_events(config)

    assert [event["uid"] for event in events] == ["evt-1", "evt-2"]
    assert requests_mock.call_count == 3
