"""Selects the entry model the engine and backtest both run.

This indirection exists so the two callers can never drift apart: a backtest
that silently evaluated a different model than the live paper engine would
report an edge that paper trading could not reproduce.

``smc`` imports helpers from ``strategy``, so the dispatcher lives here rather
than in either module to keep the import graph acyclic.
"""
from . import judas, smc, strategy
from .config import settings

MODELS = {"classic": strategy.signal_from, "smc": smc.signal_from, "judas": judas.signal_from}


def signal_from(frame):
    """Run the configured entry model against an enriched frame."""
    try:
        return MODELS[settings.strategy](frame)
    except KeyError:
        raise RuntimeError(
            f"Unknown STRATEGY {settings.strategy!r}. Use one of: {', '.join(MODELS)}."
        ) from None


def active_name() -> str:
    return settings.strategy if settings.strategy in MODELS else "unknown"
