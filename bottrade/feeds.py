"""Market feeds: where each tick's market list comes from.

A feed hides the difference between live venues, the synthetic world and a CSV
replay, so the orchestrator and every agent stay identical across all three.
"""

from __future__ import annotations

import asyncio
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import Config
from .models import Market, Venue
from .utils import parse_ts, utcnow
from .venues.base import VenueAdapter
from .venues.mock import MockVenue, MockWorld


class Feed(ABC):
    name = "feed"

    @abstractmethod
    async def step(self) -> list[Market] | None:
        """Return the next batch of markets, or None when the feed is exhausted."""

    async def close(self) -> None:
        return None


class LiveFeed(Feed):
    """Polls the real venue adapters every `run.tick_seconds`."""

    name = "live"

    def __init__(self, cfg: Config, venues: dict[Venue, VenueAdapter]):
        self.cfg = cfg
        self.venues = venues
        self.tick_seconds = cfg.f("run.tick_seconds", 15.0)
        self._first = True

    async def step(self) -> list[Market] | None:
        if not self._first:
            await asyncio.sleep(self.tick_seconds)
        self._first = False

        markets: list[Market] = []
        for venue, adapter in self.venues.items():
            try:
                markets.extend(await adapter.list_markets())
            except Exception:  # noqa: BLE001 - one bad venue must not kill the loop
                continue
        return markets

    async def close(self) -> None:
        for adapter in self.venues.values():
            await adapter.close()


class MockFeed(Feed):
    """Synthetic world; advances one step per call, no sleeping."""

    name = "mock"

    def __init__(self, cfg: Config, world: MockWorld | None = None,
                 delay: float | None = None):
        self.cfg = cfg
        self.world = world or MockWorld(cfg)
        # 0 = run as fast as possible (tests, backtests); >0 = watchable pacing
        self.delay = cfg.f("run.mock_tick_seconds", 0.0) if delay is None else delay
        self._first = True
        self.venues: dict[Venue, MockVenue] = {
            venue: MockVenue(self.world, venue) for venue in (Venue.POLYMARKET, Venue.PUMPFUN)
        }

    async def step(self) -> list[Market] | None:
        if self.delay and not self._first:
            await asyncio.sleep(self.delay)
        self._first = False
        self.world.advance()
        markets: list[Market] = []
        for adapter in self.venues.values():
            markets.extend(await adapter.list_markets())
        return markets

    @property
    def adapters(self) -> dict[Venue, VenueAdapter]:
        return dict(self.venues)


class ReplayFeed(Feed):
    """Replays a CSV of historical quotes for backtesting.

    Expected columns: `ts,key,price` with optional `bid,ask,question,venue`.
    `key` looks like `polymarket:<token_id>` or `pumpfun:<mint>`.
    """

    name = "replay"

    def __init__(self, cfg: Config, path: str | Path):
        self.cfg = cfg
        self.path = Path(path)
        self.frames: list[list[Market]] = []
        self._index = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"replay CSV not found: {self.path}")

        buckets: dict[str, list[Market]] = {}
        order: list[str] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("key") or "").strip()
                if not key or ":" not in key:
                    continue
                venue_text, token = key.split(":", 1)
                try:
                    venue = Venue(venue_text.strip().lower())
                except ValueError:
                    continue
                price = _f(row.get("price"))
                if price <= 0:
                    continue
                ts_text = (row.get("ts") or "").strip() or utcnow().isoformat()
                market = Market(
                    venue=venue,
                    market_id=(row.get("market_id") or token).strip(),
                    question=(row.get("question") or token).strip(),
                    token_id=token.strip(),
                    outcome=(row.get("outcome") or ("YES" if venue is Venue.POLYMARKET else "TOKEN")),
                    price=price,
                    bid=_f(row.get("bid")) or price,
                    ask=_f(row.get("ask")) or price,
                    liquidity_usd=_f(row.get("liquidity_usd"), 50_000.0),
                    volume_24h_usd=_f(row.get("volume_24h_usd"), 100_000.0),
                    end_date=parse_ts(row.get("end_date")),
                    tags=("replay",),
                    meta={"ts": ts_text},
                )
                if ts_text not in buckets:
                    buckets[ts_text] = []
                    order.append(ts_text)
                buckets[ts_text].append(market)
        self.frames = [buckets[ts] for ts in order]

    async def step(self) -> list[Market] | None:
        if self._index >= len(self.frames):
            return None
        frame = self.frames[self._index]
        self._index += 1
        return frame

    def __len__(self) -> int:
        return len(self.frames)


def build_feed(cfg: Config, source: str | None = None) -> Feed:
    """Construct the feed named by `run.source` (or the explicit override)."""
    source = (source or cfg.s("run.source", "mock")).lower()
    if source == "mock":
        return MockFeed(cfg)
    if source == "replay":
        return ReplayFeed(cfg, cfg.s("run.replay_csv", ""))
    if source == "live":
        from .venues.polymarket import PolymarketVenue
        from .venues.pumpfun import PumpFunVenue

        venues: dict[Venue, VenueAdapter] = {}
        enabled = {str(v).lower() for v in cfg.list("run.venues", ["polymarket", "pumpfun"])}
        if "polymarket" in enabled:
            venues[Venue.POLYMARKET] = PolymarketVenue(cfg)
        if "pumpfun" in enabled:
            venues[Venue.PUMPFUN] = PumpFunVenue(cfg)
        if not venues:
            raise ValueError("no venues enabled under run.venues")
        return LiveFeed(cfg, venues)
    raise ValueError(f"unknown feed source: {source!r} (use live | mock | replay)")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
