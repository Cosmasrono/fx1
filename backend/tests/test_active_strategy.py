import pytest

from app import active_strategy, smc, strategy
from app.config import settings


def test_dispatcher_routes_to_the_configured_model(monkeypatch):
    monkeypatch.setattr(settings, "strategy", "classic")
    assert active_strategy.MODELS["classic"] is strategy.signal_from
    assert active_strategy.active_name() == "classic"

    monkeypatch.setattr(settings, "strategy", "smc")
    assert active_strategy.MODELS["smc"] is smc.signal_from
    assert active_strategy.active_name() == "smc"


def test_unknown_strategy_fails_loudly_rather_than_trading_the_wrong_model(monkeypatch):
    monkeypatch.setattr(settings, "strategy", "ictt")
    with pytest.raises(RuntimeError, match="Unknown STRATEGY"):
        active_strategy.signal_from(None)
    assert active_strategy.active_name() == "unknown"
