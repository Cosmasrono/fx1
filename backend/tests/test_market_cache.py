import asyncio

import pytest

from app import market


@pytest.fixture(autouse=True)
def isolated_cache():
    market._candle_cache.clear()
    market._candle_locks.clear()
    yield
    market._candle_cache.clear()
    market._candle_locks.clear()


def test_chart_and_engine_share_one_refresh_and_independent_frames(monkeypatch):
    calls = []

    async def provider(outputsize, interval):
        calls.append((outputsize, interval))
        await asyncio.sleep(0)
        return market.synthetic_candles(outputsize, interval), "test"

    monkeypatch.setattr(market, "_fetch_candles_uncached", provider)

    async def scenario():
        chart, engine = await asyncio.gather(market.fetch_candles(160), market.fetch_candles())
        assert len(chart[0]) == 160
        assert len(engine[0]) == 240
        assert chart[0].iloc[-1].close == engine[0].iloc[-1].close
        chart[0].loc[159, "close"] = 999
        fresh, feed = await market.fetch_candles(160)
        assert fresh.iloc[-1].close != 999
        assert feed == "test"

    asyncio.run(scenario())
    assert calls == [(240, market.settings.interval)]


def test_larger_history_and_distinct_intervals_fetch_their_own_windows(monkeypatch):
    calls = []

    async def provider(outputsize, interval):
        calls.append((outputsize, interval))
        return market.synthetic_candles(outputsize, interval), "test"

    monkeypatch.setattr(market, "_fetch_candles_uncached", provider)

    async def scenario():
        await market.fetch_candles(160, "15min")
        large, _ = await market.fetch_candles(5000, "15min")
        small, _ = await market.fetch_candles(240, "15min")
        assert small.iloc[-1].close == large.iloc[-1].close
        await market.fetch_candles(240, "1week")

    asyncio.run(scenario())
    assert calls == [(240, "15min"), (5000, "15min"), (240, "1week")]


def test_expired_snapshot_is_not_served_on_failure_and_can_recover(monkeypatch):
    calls = 0

    async def provider(outputsize, interval):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Provider unavailable")
        return market.synthetic_candles(outputsize, interval), "test"

    monkeypatch.setattr(market, "_fetch_candles_uncached", provider)

    async def scenario():
        await market.fetch_candles()
        for key, (_, capacity, frame, feed) in list(market._candle_cache.items()):
            market._candle_cache[key] = (0, capacity, frame, feed)
        with pytest.raises(RuntimeError, match="Provider unavailable"):
            await market.fetch_candles()
        frame, _ = await market.fetch_candles()
        assert len(frame) == 240

    asyncio.run(scenario())
    assert calls == 3


def test_snapshot_expires_at_candle_boundary(monkeypatch):
    async def provider(outputsize, interval):
        return market.synthetic_candles(outputsize, interval), "test"

    monkeypatch.setattr(market, "_fetch_candles_uncached", provider)
    monkeypatch.setattr(market.time, "time", lambda: 899.0)
    before = market.time.monotonic()
    asyncio.run(market.fetch_candles(240, "15min"))
    expiry = next(iter(market._candle_cache.values()))[0]
    assert before < expiry <= market.time.monotonic() + 1.0
