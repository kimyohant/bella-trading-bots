"""Agent framework: the `Context` blackboard, the `Agent` base class, a registry.

Agents are pure stages over a shared `Context`:

    guard -> scanner -> sentiment -> whale -> signal -> risk -> executor -> portfolio

Adding your own agent is three lines: subclass `Agent`, set `name`, decorate with
`@register`, then list the name under `[agents] enabled` in `config.toml`.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import Config
from ..data.store import Store
from ..models import (
    Fill,
    FlowBias,
    Market,
    Order,
    Quote,
    SentimentScore,
    Signal,
    Venue,
)
from ..utils import utcnow
from ..venues.base import VenueAdapter
from ..venues.paper import PaperBroker


@dataclass
class Context:
    """Everything an agent may read or write during one tick."""

    cfg: Config
    broker: PaperBroker
    log: logging.Logger
    adapters: dict[Venue, VenueAdapter] = field(default_factory=dict)
    store: Store | None = None
    tick: int = 0
    now: datetime = field(default_factory=utcnow)
    duration_s: float = 0.0

    # filled in by the pipeline, in order
    markets: list[Market] = field(default_factory=list)
    quotes: dict[str, Quote] = field(default_factory=dict)
    sentiment: dict[str, SentimentScore] = field(default_factory=dict)
    flow: dict[str, FlowBias] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # scratch space shared between ticks (price history, cooldowns, counters)
    state: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def price_map(self) -> dict[str, float]:
        return {m.key: m.price for m in self.markets}

    def market_by_key(self) -> dict[str, Market]:
        return {m.key: m for m in self.markets}

    def quote_for(self, market: Market) -> Quote:
        return self.quotes.get(market.key) or market.quote()

    def equity(self) -> float:
        return self.broker.equity()

    def note(self, message: str) -> None:
        self.notes.append(message)

    def reject(self, market: Market | None, reason: str, agent: str = "risk") -> None:
        self.rejections.append(
            {
                "ts": utcnow().isoformat(timespec="seconds"),
                "agent": agent,
                "key": market.key if market else "",
                "question": market.question if market else "",
                "reason": reason,
            }
        )


class Agent(ABC):
    """One stage of the pipeline."""

    name: str = "agent"

    def __init__(self, cfg: Config, **params: Any):
        self.cfg = cfg
        self.params = params
        self.log = logging.getLogger(f"bottrade.{self.name}")

    @abstractmethod
    async def run(self, ctx: Context) -> None:
        """Do this agent's work for one tick."""

    # convenience accessors -------------------------------------------------
    def f(self, path: str, default: float = 0.0) -> float:
        return self.cfg.f(f"{self.name}.{path}", default)

    def i(self, path: str, default: int = 0) -> int:
        return self.cfg.i(f"{self.name}.{path}", default)

    def b(self, path: str, default: bool = False) -> bool:
        return self.cfg.b(f"{self.name}.{path}", default)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name}>"


# --------------------------------------------------------------------- registry

_REGISTRY: dict[str, type[Agent]] = {}


def register(name: str | None = None):
    """Class decorator: `@register("my_agent")` makes an agent config-selectable."""

    def decorator(cls: type[Agent]) -> type[Agent]:
        cls.name = name or getattr(cls, "name", cls.__name__.lower())
        _REGISTRY[cls.name] = cls
        return cls

    return decorator


def registry() -> dict[str, type[Agent]]:
    _load_builtin()
    return dict(_REGISTRY)


def get_agent_class(name: str) -> type[Agent]:
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def _load_builtin() -> None:
    """Import built-in agent modules so their decorators run."""
    for module in ("guard", "scanner", "sentiment", "whale", "signal",
                   "risk", "executor", "portfolio"):
        importlib.import_module(f"{__package__}.{module}")


def build_agents(cfg: Config) -> list[Agent]:
    """Instantiate the pipeline listed in `[agents] enabled`, in that order."""
    _load_builtin()

    for module_name in cfg.list("agents.custom_modules", []):
        importlib.import_module(str(module_name))

    names = cfg.list(
        "agents.enabled",
        ["guard", "scanner", "sentiment", "whale", "signal", "risk",
         "executor", "portfolio"],
    )
    agents: list[Agent] = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        cls = get_agent_class(name)
        agents.append(cls(cfg))
    return agents
