from datetime import datetime, timezone

from app.news import decision_from_events


def test_high_impact_event_blocks_entries_inside_blackout():
    now = datetime(2026, 8, 7, 12, 20, tzinfo=timezone.utc)
    events = [{
        "Date": "2026-08-07T12:30:00",
        "Country": "United States",
        "Event": "Non Farm Payrolls",
        "Importance": 3,
    }]

    result = decision_from_events(events, now)

    assert result["blocking"] is True
    assert result["status"] == "BLACKOUT"
    assert result["event"]["name"] == "Non Farm Payrolls"


def test_low_impact_event_does_not_block_entries():
    now = datetime(2026, 8, 7, 12, 20, tzinfo=timezone.utc)
    events = [{"Date": "2026-08-07T12:30:00", "Event": "Minor release", "Importance": 1}]

    result = decision_from_events(events, now)

    assert result["blocking"] is False
