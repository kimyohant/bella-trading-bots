"""Domain objects passed between agents.

Everything is a plain dataclass so it can be logged, stored in sqlite, or
serialised to JSON without ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .utils import hours_between, utcnow


class Venue(str, Enum):
    POLYMARKET = "polymarket"
    PUMPFUN = "pumpfun"

    def __str__(self) -> str:  # nicer f-strings / logs
        return self.value


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    def __str__(self) -> str:
        return self.value


@dataclass(slots=True)
class Quote:
    """Top-of-book snapshot for one tradable token."""

    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    liquidity_usd: float = 0.0
    ts: datetime = field(default_factory=utcnow)

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.ask or self.bid or self.last or 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        if not mid or self.bid is None or self.ask is None:
            return 0.0
        return max(0.0, (self.ask - self.bid) / mid)


@dataclass(slots=True)
class Market:
    """One tradable leg.

    For Polymarket a binary market yields two legs (YES token, NO token), each
    with its own `token_id`; the price of a YES leg *is* the implied probability.
    For pump.fun there is a single SPL mint per token.
    """

    venue: Venue
    market_id: str
    question: str
    token_id: str
    outcome: str = "YES"
    price: float = 0.0
    bid: float | None = None
    ask: float | None = None
    liquidity_usd: float = 0.0
    volume_24h_usd: float = 0.0
    end_date: datetime | None = None
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.venue.value}:{self.token_id}"

    @property
    def spread_pct(self) -> float:
        mid = self.price
        if not mid or self.bid is None or self.ask is None:
            return 0.0
        return max(0.0, (self.ask - self.bid) / mid)

    @property
    def hours_to_resolution(self) -> float | None:
        if not self.end_date:
            return None
        return hours_between(utcnow(), self.end_date)

    def quote(self) -> Quote:
        return Quote(bid=self.bid, ask=self.ask, last=self.price, liquidity_usd=self.liquidity_usd)

    def to_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "venue": self.venue.value,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "question": self.question,
            "outcome": self.outcome,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "liquidity_usd": self.liquidity_usd,
            "volume_24h_usd": self.volume_24h_usd,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


@dataclass(slots=True)
class Signal:
    """A directional opinion with an explicit fair value and edge."""

    market: Market
    side: Side
    fair_price: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    ttl_seconds: int = 900

    @property
    def key(self) -> str:
        return self.market.key

    @property
    def entry_price(self) -> float:
        """Price we would realistically pay/receive right now."""
        if self.side is Side.BUY:
            return self.market.ask or self.market.price
        return self.market.bid or self.market.price

    @property
    def edge(self) -> float:
        """Signed edge per unit of notional, relative to the executable price."""
        price = self.entry_price or 0.0
        if price <= 0:
            return 0.0
        if self.side is Side.BUY:
            return (self.fair_price - price) / price
        return (price - self.fair_price) / price

    @property
    def edge_abs(self) -> float:
        """Absolute price difference — the number Kelly wants for binaries."""
        return abs(self.fair_price - self.entry_price)

    @property
    def expired(self) -> bool:
        return (utcnow() - self.created_at).total_seconds() > self.ttl_seconds


@dataclass(slots=True)
class Order:
    market: Market
    side: Side
    qty: float
    limit_price: float
    reason: str = ""
    edge: float = 0.0
    confidence: float = 0.0
    notional: float = 0.0
    tag: str = "entry"  # entry | exit
    thesis: float = 0.0  # sentiment+flow score at entry, used to detect decay

    @property
    def key(self) -> str:
        return self.market.key


@dataclass(slots=True)
class Fill:
    key: str
    venue: Venue
    question: str
    side: Side
    qty: float
    price: float
    fee: float
    slippage_bps: float
    reason: str = ""
    tag: str = "entry"
    realized_pnl: float = 0.0
    ts: datetime = field(default_factory=utcnow)

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(slots=True)
class Position:
    key: str
    venue: Venue
    token_id: str
    question: str
    qty: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    opened_at: datetime = field(default_factory=utcnow)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    entry_edge: float = 0.0
    entry_thesis: float = 0.0
    high_water: float = 0.0
    partial_taken: bool = False

    @property
    def is_open(self) -> bool:
        return self.qty > 1e-9

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_price

    @property
    def market_value(self) -> float:
        return self.qty * (self.last_price or self.avg_price)

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def return_pct(self) -> float:
        return self.unrealized_pnl / self.cost_basis if self.cost_basis > 0 else 0.0

    @property
    def age_hours(self) -> float:
        return hours_between(self.opened_at, utcnow())

    def to_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "venue": self.venue.value,
            "question": self.question,
            "qty": round(self.qty, 6),
            "avg_price": round(self.avg_price, 6),
            "last_price": round(self.last_price, 6),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "return_pct": round(self.return_pct, 4),
            "age_hours": round(self.age_hours, 2),
        }


@dataclass(slots=True)
class SentimentScore:
    key: str
    score: float = 0.0          # -1 (bearish) .. +1 (bullish)
    confidence: float = 0.0     # 0..1
    samples: int = 0
    source: str = "lexicon"
    headlines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FlowBias:
    """Net aggressive flow for one market, from whale / trade-tape analysis."""

    key: str
    bias: float = 0.0           # -1 .. +1 (taker sell pressure .. taker buy pressure)
    notional_usd: float = 0.0
    wallets: int = 0
    samples: int = 0
    source: str = ""
    detail: str = ""
