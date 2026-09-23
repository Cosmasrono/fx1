import asyncio
import copy
import json

import httpx
import pytest
from pydantic import SecretStr

from app import ai_review
from app.config import Settings, settings


def test_alias_and_secret_repr():
    config = Settings(_env_file=None, api_key="test-secret")
    assert config.openrouter_api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(config)


def test_summary_excludes_credentials_and_full_snapshots():
    state = {"symbol": "EUR/USD", "api_key": "secret", "account": {"token": "secret"},
             "last_error": "url?key=secret", "trades": [{"pnl": -10, "signal_snapshot": {"secret": "secret"}}] * 50}
    result = ai_review.summary(state)
    assert "secret" not in json.dumps(result)
    assert len(result["recent_trades"]) == 10
    assert result["engine_has_error"] is True


@pytest.mark.parametrize("status", [200, 401, 429])
def test_review_is_read_only_bounded_and_sanitizes_provider_errors(monkeypatch, status):
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("test-secret"))
    captured = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["verify"] is not False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            captured.append(kwargs)
            return httpx.Response(status, json={"model": "test-model", "choices": [{"message": {"content": "Test explanation"}, "finish_reason": "stop"}], "error": "test-secret"})
    monkeypatch.setattr(ai_review.httpx, "AsyncClient", Client)
    state = {"symbol": "EUR/USD", "latest_signal": {"action": "HOLD"}}
    original = copy.deepcopy(state)
    reviewer = ai_review.Reviewer()
    if status == 200:
        result = asyncio.run(reviewer.review(state))
        assert result["text"] == "Test explanation"
    else:
        with pytest.raises(ai_review.ReviewError) as error:
            asyncio.run(reviewer.review(state))
        assert "test-secret" not in str(error.value)
    assert state == original and not reviewer.busy
    assert "test-secret" not in json.dumps(captured[0]["json"])
    with pytest.raises(ai_review.ReviewError) as error:
        asyncio.run(reviewer.review(state))
    assert error.value.status == 429 and len(captured) == 1


def test_missing_key_never_sends_request(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr(""))
    with pytest.raises(ai_review.ReviewError) as error:
        asyncio.run(ai_review.Reviewer().review({}))
    assert error.value.status == 503


def test_untrusted_browser_cannot_trigger_a_review():
    from fastapi.testclient import TestClient
    from app.main import app
    response = TestClient(app).post("/api/ai-review", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403
