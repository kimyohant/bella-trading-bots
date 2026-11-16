"""How big the tradable universe is, and what silently shrank it.

Two separate limits decide how many markets the bot ever looks at:

  * the venue fetch — Gamma caps a page at 100 rows, so asking for 60 was
    already leaving markets on the table;
  * the scanner quota — a per-venue cap that was 12 while the venue supplied 21.

Both are invisible in the log: a smaller universe looks exactly like a quiet
market, which is why the symptom shows up as "the bot is not trading".
"""

from __future__ import annotations

import asyncio
import json
import logging
import unittest
from datetime import timedelta
from unittest.mock import patch

from bottrade.agents.base import Context
from bottrade.agents.scanner import ScannerAgent
from bottrade.config import Config
from bottrade.data.http import HttpError
from bottrade.models import Market, Venue
from bottrade.utils import utcnow
from bottrade.venues.paper import PaperBroker
from bottrade.venues.polymarket import PolymarketVenue


def row(i: int, cond: str | None = None) -> dict:
    """One Gamma market row, minimal but shaped like the real thing."""
    return {
        "conditionId": cond or f"cond{i}",
        "id": cond or f"cond{i}",
        "question": f"Question {i}?",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.5", "0.5"]),
        "clobTokenIds": json.dumps([f"tok{i}a", f"tok{i}b"]),
        "endDate": (utcnow() + timedelta(days=10)).isoformat(),
        "liquidityNum": 50_000,
        "volume24hr": 50_000,
    }


def pm_market(i: int, price: float = 0.5, liquidity: float = 50_000.0) -> Market:
    return Market(
        venue=Venue.POLYMARKET, market_id=f"m{i}", question=f"Question {i}?",
        token_id=f"tok{i}", outcome="Yes", price=price,
        bid=round(price - 0.01, 4), ask=round(price + 0.01, 4),
        liquidity_usd=liquidity, volume_24h_usd=50_000,
        end_date=utcnow() + timedelta(days=10),
    )


def broker() -> PaperBroker:
    return PaperBroker(
        cash=1000.0, starting_cash=1000.0,
        fee_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 100.0},
        slippage_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
        fixed_fee_usd={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
    )


def context(cfg: Config, markets: list[Market]) -> Context:
    ctx = Context(cfg=cfg, broker=broker(), log=logging.getLogger("test"))
    ctx.markets = markets
    return ctx


class TestPolymarketPaging(unittest.TestCase):
    """Gamma returns at most 100 rows per request, whatever `limit` says."""

    def _venue(self, **scanner) -> PolymarketVenue:
        return PolymarketVenue(Config({"scanner": {"polymarket_pages": 2, **scanner}}))

    def _run(self, venue, pages: list[list[dict] | Exception]):
        calls: list[dict] = []
        step = max(1, venue.limit)

        async def fake(url, params=None, **_kwargs):
            calls.append(dict(params or {}))
            # Index by offset, not by call number: a failed page is retried with
            # two more parameter shapes before the scan gives up on it.
            index = (params or {}).get("offset", 0) // step
            page = pages[index] if index < len(pages) else []
            if isinstance(page, Exception):
                raise page
            return page

        with patch("bottrade.venues.polymarket.fetch_json", new=fake):
            rows = asyncio.run(venue._fetch_markets())
        return rows, calls

    def test_second_page_is_fetched_when_the_first_is_full(self):
        rows, calls = self._run(
            self._venue(polymarket_limit=100),
            [[row(i) for i in range(100)], [row(i) for i in range(100, 130)]],
        )
        self.assertEqual(len(rows), 130)
        self.assertEqual([c["offset"] for c in calls], [0, 100])
        self.assertTrue(all(c["limit"] == 100 for c in calls))

    def test_short_page_ends_the_scan(self):
        rows, calls = self._run(
            self._venue(polymarket_limit=100), [[row(i) for i in range(40)]]
        )
        self.assertEqual(len(rows), 40)
        self.assertEqual(len(calls), 1, "a short page means there is no page 2")

    def test_one_page_when_configured(self):
        rows, calls = self._run(
            self._venue(polymarket_limit=100, polymarket_pages=1),
            [[row(i) for i in range(100)]],
        )
        self.assertEqual(len(rows), 100)
        self.assertEqual(len(calls), 1)

    def test_rows_are_deduplicated_across_pages(self):
        # Paging by offset is not a stable snapshot: a market can appear on both
        # pages if the ordering shifts between the two requests.
        rows, _ = self._run(
            self._venue(polymarket_limit=100),
            [[row(i) for i in range(100)], [row(99, cond="cond99"), row(200)]],
        )
        self.assertEqual(len(rows), 101)
        self.assertEqual(len({r["conditionId"] for r in rows}), 101)

    def test_limit_above_the_page_size_is_clamped(self):
        # Asking for 500 gets 100 back, but an unclamped offset would then skip
        # 400 rows that never existed.
        venue = self._venue(polymarket_limit=500)
        self.assertEqual(venue.limit, 100)
        _, calls = self._run(venue, [[row(i) for i in range(100)], [row(100)]])
        self.assertEqual([c["offset"] for c in calls], [0, 100])

    def test_a_failed_page_does_not_lose_the_ones_already_fetched(self):
        rows, calls = self._run(
            self._venue(polymarket_limit=100),
            [[row(i) for i in range(100)], HttpError("HTTP 500")],
        )
        self.assertEqual(len(rows), 100)
        # page 1 once, then the three parameter shapes it tries before giving up
        self.assertEqual([c["offset"] for c in calls], [0, 100, 100, 100])


class TestScannerQuota(unittest.TestCase):
    """A per-venue cap smaller than the supply throws markets away silently."""

    def _cfg(self, **scanner):
        base = {
            "min_hours_to_resolution": 1.0,
            "min_liquidity_usd": 3000,
            "min_price": 0.03,
            "max_price": 0.97,
            "max_spread_pct": 0.08,
            "max_candidates": 34,
            "max_per_venue": 20,
        }
        base.update(scanner)
        return Config({"scanner": base})

    def _universe(self, cfg, count=30):
        markets = [pm_market(i) for i in range(count)]
        ctx = context(cfg, markets)
        asyncio.run(ScannerAgent(cfg).run(ctx))
        return ctx

    def test_quota_admits_more_than_the_old_twelve(self):
        ctx = self._universe(self._cfg(), count=30)
        self.assertEqual(len(ctx.markets), 20)

    def test_quota_still_binds_below_the_supply(self):
        ctx = self._universe(self._cfg(max_per_venue=12), count=30)
        self.assertEqual(len(ctx.markets), 12)

    def test_global_ceiling_still_truncates(self):
        ctx = self._universe(self._cfg(max_per_venue=20, max_candidates=10), count=30)
        self.assertEqual(len(ctx.markets), 10)

    def test_all_thirty_survive_when_both_caps_allow_it(self):
        ctx = self._universe(self._cfg(max_per_venue=40, max_candidates=40), count=30)
        self.assertEqual(len(ctx.markets), 30)


class TestShippedUniverseConfig(unittest.TestCase):
    """The shipped numbers, read from the real file — this is the whole fix."""

    def setUp(self):
        self.cfg = Config.load()

    def test_page_size_is_exactly_the_gamma_cap(self):
        # Gamma returns at most 100 rows whatever you ask for, so anything below
        # 100 is capacity thrown away for free — and above 100 is a lie the
        # offset maths would then believe.
        limit = self.cfg.i("scanner.polymarket_limit", 60)
        self.assertEqual(limit, 100, "Gamma caps a page at 100 rows")

    def test_paging_is_on(self):
        self.assertGreaterEqual(self.cfg.i("scanner.polymarket_pages", 1), 2)

    def test_venue_quota_does_not_throw_away_the_wider_universe(self):
        per_venue = self.cfg.i("scanner.max_per_venue", 12)
        self.assertGreater(per_venue, 12, "the venue supplied 21 and the quota kept 12")

    def test_global_ceiling_fits_the_per_venue_quota(self):
        per_venue = self.cfg.i("scanner.max_per_venue", 12)
        ceiling = self.cfg.i("scanner.max_candidates", 24)
        self.assertGreaterEqual(
            ceiling, per_venue, "max_candidates below max_per_venue hides the quota"
        )

    def test_signal_quota_can_see_the_wider_universe(self):
        per_venue = self.cfg.i("scanner.max_per_venue", 12)
        signals = self.cfg.i("signal.max_signals_per_venue", 4)
        self.assertLessEqual(signals, per_venue)
        self.assertGreater(signals, 4, "four signals per tick from twenty markets")


if __name__ == "__main__":
    unittest.main()
