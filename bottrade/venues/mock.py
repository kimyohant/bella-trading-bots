"""Synthetic market world used for offline runs, demos and tests.

It generates both venues with a hidden "fair" value that drifts slowly, plus a
noisy quoted price that mean-reverts to it. When the quote dislocates from fair,
the signal agent sees real edge — exactly the situation a live bot trades — so the
whole pipeline can be exercised with no network and no API keys.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ..config import Config
from ..models import Market, Quote, Venue
from ..utils import clamp, utcnow
from .base import VenueAdapter


@dataclass
class _Leg:
    venue: Venue
    market_id: str
    question: str
    token_id: str
    fair: float
    price: float
    sigma: float
    liquidity: float
    volume: float
    end_date: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    momentum: float = 0.0
    last_delta: float = 0.0


PM_QUESTIONS = [
    ("Will BTC close above $120k this month?", 0.58),
    ("Will the Fed cut rates at the next meeting?", 0.41),
    ("Will Team A win the championship?", 0.67),
    ("Will the bill pass before year end?", 0.33),
    ("Will ETH flip $5k before December?", 0.24),
    ("Will the incumbent win the runoff?", 0.72),
    ("Will inflation print below 3% next month?", 0.46),
    ("Will the merger close in Q1?", 0.81),
    ("Will the satellite launch succeed on first try?", 0.63),
    ("Will the sequel gross over $1B?", 0.29),
]

PF_TOKENS = [
    ("MoonRat", "MRAT"),
    ("SolDoge", "SDOGE"),
    ("PixelFrog", "PFRG"),
    ("ChainCat", "CCAT"),
    ("BasedBear", "BBEAR"),
    ("TurboToad", "TTOAD"),
    ("LaserLlama", "LLAMA"),
    ("GigaChad", "GIGA"),
]


class MockWorld:
    """Shared state behind every mock venue, advanced once per tick."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = random.Random(seed if seed is not None else cfg.i("run.seed", 7))
        self.legs: dict[str, _Leg] = {}
        self.tick = 0
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        for index, (question, fair) in enumerate(PM_QUESTIONS):
            market_id = f"mock-pm-{index:03d}"
            for leg, base in (("Yes", fair), ("No", 1.0 - fair)):
                token = f"{market_id}-{leg.lower()}"
                price = clamp(base + self.rng.uniform(-0.03, 0.03), 0.02, 0.98)
                self.legs[token] = _Leg(
                    venue=Venue.POLYMARKET,
                    market_id=market_id,
                    question=question,
                    token_id=token,
                    fair=base,
                    price=price,
                    sigma=self.rng.uniform(0.004, 0.012),
                    liquidity=self.rng.uniform(8_000, 120_000),
                    volume=self.rng.uniform(15_000, 400_000),
                    end_date=utcnow() + timedelta(days=self.rng.uniform(3, 40)),
                    meta={"leg": leg, "group": "mock"},
                )

        for index, (name, symbol) in enumerate(PF_TOKENS):
            mint = f"mockpf{index:02d}" + "x" * 30
            price = self.rng.uniform(1.5e-5, 9e-5)
            self.legs[mint] = _Leg(
                venue=Venue.PUMPFUN,
                market_id=mint,
                question=f"{name} ({symbol})",
                token_id=mint,
                fair=price,
                price=price,
                sigma=self.rng.uniform(0.02, 0.06),
                liquidity=price * 1e9 * 0.35,
                volume=self.rng.uniform(5_000, 90_000),
                meta={
                    "symbol": symbol,
                    "age_minutes": self.rng.uniform(6, 300),
                    "market_cap_usd": price * 1e9,
                    "curve_progress": self.rng.uniform(0.02, 0.55),
                    "complete": False,
                    "sol_usd": 150.0,
                    "last_trade_ts": 0.0,
                    "replies": self.rng.randint(2, 120),
                    "king_of_the_hill": False,
                },
            )

    # ------------------------------------------------------------------- step
    def advance(self) -> None:
        self.tick += 1
        for leg in self.legs.values():
            if leg.venue is Venue.POLYMARKET:
                self._step_binary(leg)
            else:
                self._step_token(leg)

    def _step_binary(self, leg: _Leg) -> None:
        # fair value drifts slowly; the quote mean-reverts and occasionally gaps
        leg.fair = clamp(leg.fair + self.rng.gauss(0, leg.sigma * 0.35), 0.03, 0.97)
        shock = self.rng.gauss(0, leg.sigma * 2.2) if self.rng.random() < 0.18 else 0.0
        drift = 0.18 * (leg.fair - leg.price)
        leg.price = clamp(leg.price + drift + self.rng.gauss(0, leg.sigma) + shock, 0.02, 0.98)
        leg.last_delta = leg.fair - leg.price
        leg.volume = max(5_000.0, leg.volume * self.rng.uniform(0.97, 1.05))

    def _step_token(self, leg: _Leg) -> None:
        leg.momentum = 0.75 * leg.momentum + self.rng.gauss(0, leg.sigma)
        leg.price = max(1e-7, leg.price * math.exp(leg.momentum * 0.5 + self.rng.gauss(0, leg.sigma * 0.4)))
        leg.fair = leg.price
        leg.last_delta = leg.momentum
        leg.meta["market_cap_usd"] = leg.price * 1e9
        leg.meta["age_minutes"] = leg.meta.get("age_minutes", 0.0) + 0.25
        leg.volume = max(1_000.0, leg.volume * self.rng.uniform(0.95, 1.08))

    # ---------------------------------------------------------------- accessors
    def markets(self, venue: Venue) -> list[Market]:
        out: list[Market] = []
        for leg in self.legs.values():
            if leg.venue is not venue:
                continue
            half = 0.01 if venue is Venue.POLYMARKET else 0.01
            out.append(
                Market(
                    venue=venue,
                    market_id=leg.market_id,
                    question=leg.question,
                    token_id=leg.token_id,
                    outcome=leg.meta.get("leg", "TOKEN"),
                    price=round(leg.price, 6),
                    bid=round(leg.price * (1 - half), 6),
                    ask=round(min(0.995, leg.price * (1 + half)), 6)
                    if venue is Venue.POLYMARKET
                    else round(leg.price * (1 + half), 10),
                    liquidity_usd=round(leg.liquidity, 2),
                    volume_24h_usd=round(leg.volume, 2),
                    end_date=leg.end_date,
                    tags=("mock",),
                    meta=dict(leg.meta, fair_hint=round(leg.fair, 4)),
                )
            )
        return out

    def trades(self, venue: Venue, limit: int = 60) -> list[dict[str, Any]]:
        """Synthetic tape whose aggression follows the last price move."""
        out: list[dict[str, Any]] = []
        for leg in self.legs.values():
            if leg.venue is not venue:
                continue
            if self.rng.random() < 0.5:
                continue
            bullish = leg.last_delta >= 0
            for _ in range(self.rng.randint(1, 3)):
                notional = self.rng.uniform(500, 9_000)
                out.append(
                    {
                        "key": f"{venue.value}:{leg.token_id}",
                        "side": "BUY" if bullish else "SELL",
                        "price": leg.price,
                        "qty": notional / max(leg.price, 1e-9),
                        "notional": notional,
                        "wallet": f"0xmock{self.rng.randrange(16**6):06x}",
                        "ts": utcnow(),
                        "market": leg.market_id,
                        "title": leg.question,
                    }
                )
        out.sort(key=lambda t: t["notional"], reverse=True)
        return out[:limit]


class MockVenue(VenueAdapter):
    """Reads from a shared `MockWorld`; advancing the world is the feed's job."""

    def __init__(self, world: MockWorld, venue: Venue):
        self.world = world
        self.venue = venue
        self.name = f"mock-{venue.value}"

    async def list_markets(self) -> list[Market]:
        return self.world.markets(self.venue)

    async def get_quote(self, market: Market) -> Quote:
        return market.quote()

    async def recent_trades(self) -> list[dict[str, Any]]:
        return self.world.trades(self.venue)

    async def flow_snapshot(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for market in self.world.markets(self.venue):
            if self.venue is not Venue.PUMPFUN:
                continue
            momentum = market.meta.get("curve_progress", 0.0)
            buys = 40 + int(momentum * 120)
            sells = 40 + int((1 - momentum) * 120)
            out[market.key] = {
                "buys": float(buys),
                "sells": float(sells),
                "volume_24h_usd": market.volume_24h_usd,
                "liquidity_usd": market.liquidity_usd,
                "price_usd": market.price,
            }
        return out
