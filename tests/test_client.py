from src.openagenda.client import OpenAgendaConfig, fetch_events


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
