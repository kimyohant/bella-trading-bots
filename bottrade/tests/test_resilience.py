"""Surviving a hostile API: rate-limit backoff, a fallback host, re-marked marks.

Three separate failures motivated these tests:

  1. GeckoTerminal answered 429 once and the bot kept asking every two seconds,
     which turned a temporary throttle into a multi-hour lockout.
  2. When Gecko was refusing us the whole pump.fun venue looked *empty* rather
     than broken — no markets, no positions, no signal, nothing in the log.
  3. A position that dropped out of the scan quota stopped receiving marks, so
     its P&L froze and no exit rule could ever fire.
"""

from __future__ import annotations

import asyncio
import logging
import time
import unittest
from datetime import timedelta
from unittest.mock import patch

from bottrade.agents.base import Context
from bottrade.agents.scanner import ScannerAgent
from bottrade.config import Config
from bottrade.data import http
from bottrade.data.http import HttpError, HttpStatusError, fetch_json
from bottrade.models import Market, Position, Venue
from bottrade.utils import utcnow
from bottrade.venues.paper import PaperBroker
from bottrade.venues.pumpfun import PumpFunVenue

HOST = "https://api.geckoterminal.com"


def raise_status(status: int, retry_after: float | None = None):
    """An async `_request` stand-in that always fails with `status`."""

    async def _request(*_args, **_kwargs):
        raise HttpStatusError(f"HTTP {status}", status, retry_after)

    return _request


class TestRateLimitCooldown(unittest.TestCase):
    """A 429 is an instruction to stop asking, not an invitation to retry."""

    def setUp(self) -> None:
        http.cache_clear()

    def test_429_parks_the_whole_host(self):
        with patch("bottrade.data.http._request", new=raise_status(429, 30)):
            with self.assertRaises(HttpStatusError) as caught:
                asyncio.run(fetch_json(f"{HOST}/networks/solana", ttl=0))
        self.assertEqual(caught.exception.status, 429)
        # Any path on that host is now off-limits, not just the one that failed.
        self.assertGreater(http.cooling_down(f"{HOST}/some/other/path"), 25)

    def test_second_call_fails_fast_without_touching_the_network(self):
        calls = []

        async def _request(*_args, **_kwargs):
            calls.append(1)
            raise HttpStatusError("HTTP 429", 429, 30)

        with patch("bottrade.data.http._request", new=_request):
            with self.assertRaises(HttpStatusError):
                asyncio.run(fetch_json(f"{HOST}/a", ttl=0))
            with self.assertRaises(HttpError) as caught:
                asyncio.run(fetch_json(f"{HOST}/b", ttl=0))

        self.assertEqual(len(calls), 1, "the second call must not hit the network")
        self.assertIn("cooling down", str(caught.exception))

    def test_retry_after_is_honoured(self):
        with patch("bottrade.data.http._request", new=raise_status(429, 120)):
            with self.assertRaises(HttpStatusError):
                asyncio.run(fetch_json(f"{HOST}/slow", ttl=0))
        self.assertGreater(http.cooling_down(f"{HOST}/x"), 110)

    def test_missing_retry_after_falls_back_to_a_minimum_pause(self):
        with patch("bottrade.data.http._request", new=raise_status(429, None)):
            with self.assertRaises(HttpStatusError):
                asyncio.run(fetch_json(f"{HOST}/bare", ttl=0))
        wait = http.cooling_down(f"{HOST}/x")
        self.assertGreater(wait, 0.0)
        self.assertLessEqual(wait, http._MIN_COOLDOWN)

    def test_a_successful_call_clears_the_cooldown(self):
        with patch("bottrade.data.http._request", new=raise_status(429, 30)):
            with self.assertRaises(HttpStatusError):
                asyncio.run(fetch_json(f"{HOST}/park", ttl=0))
        self.assertGreater(http.cooling_down(f"{HOST}/park"), 0)

        # Fast-forward the parked deadline rather than sleeping out the pause.
        http._cooldown[http._host(f"{HOST}/park")] = time.monotonic() - 0.1

        async def ok(*_args, **_kwargs):
            return {"ok": True}

        with patch("bottrade.data.http._request", new=ok):
            self.assertEqual(asyncio.run(fetch_json(f"{HOST}/park", ttl=0)), {"ok": True})
        self.assertEqual(http.cooling_down(f"{HOST}/park"), 0.0)

    def test_a_server_error_is_retried_but_never_parked(self):
        with patch("bottrade.data.http._request", new=raise_status(500)):
            with self.assertRaises(HttpError) as caught:
                asyncio.run(fetch_json(f"{HOST}/broken", retries=0, ttl=0))
        # A 500 says "I am unwell", not "you are asking too much" — retrying is
        # legitimate, so the host must stay usable.
        self.assertNotIsInstance(caught.exception, HttpStatusError)
        self.assertEqual(http.cooling_down(f"{HOST}/broken"), 0.0)

    def test_http_date_retry_after_is_ignored(self):
        self.assertIsNone(http._retry_after("Wed, 21 Oct 2015 07:28:00 GMT"))
        self.assertEqual(http._retry_after("12"), 12.0)


def dexscreener_pair(mint="So1pump", price="0.00042", dex="pumpswap", age_min=90.0):
    """One `/tokens/v1/solana/{mints}` row, shaped like the real response."""
    return {
        "chainId": "solana",
        "dexId": dex,
        "pairAddress": "pair1",
        "baseToken": {"address": mint, "symbol": "TEST", "name": "Test"},
        "priceUsd": price,
        "priceChange": {"m5": 1.5, "h1": 8.0, "h24": 40.0},
        "volume": {"h24": 25_000.0},
        "liquidity": {"usd": 18_000.0},
        "marketCap": 60_000.0,
        "pairCreatedAt": int((utcnow() - timedelta(minutes=age_min)).timestamp() * 1000),
        "txns": {"h24": {"buys": 120, "sells": 80, "buyers": 70}},
    }


def routed(routes: dict[str, object]):
    """Fake `fetch_json`: first matching needle wins, exceptions are raised."""

    async def _fetch(url, **_kwargs):
        for needle, value in routes.items():
            if needle in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise HttpError(f"no route for {url}")

    return _fetch


class TestDexscreenerFallback(unittest.TestCase):
    """When GeckoTerminal is down, a different host keeps the venue alive."""

    def setUp(self) -> None:
        http.cache_clear()

    def _venue(self, **scanner) -> PumpFunVenue:
        return PumpFunVenue(Config({"scanner": scanner}))

    def test_venue_falls_back_to_a_different_host(self):
        venue = self._venue()
        profiles = [
            {"chainId": "solana", "tokenAddress": "So1pump"},
            {"chainId": "base", "tokenAddress": "0xdeadpump"},   # wrong chain
            {"chainId": "solana", "tokenAddress": "So1pump"},    # duplicate
        ]
        routes = {
            "geckoterminal": HttpError("HTTP 429"),
            "frontend-api": HttpError("HTTP 503"),
            "token-profiles": profiles,
            "token-boosts": [{"chainId": "solana", "tokenAddress": "So2pump"}],
            "tokens/v1": [dexscreener_pair("So1pump"), dexscreener_pair("So2pump")],
        }
        with patch("bottrade.venues.pumpfun.fetch_json", new=routed(routes)):
            markets = asyncio.run(venue.list_markets())

        self.assertEqual(venue.source, "dexscreener")
        self.assertEqual({m.market_id for m in markets}, {"So1pump", "So2pump"})
        self.assertTrue(all(m.meta["source"] == "dexscreener" for m in markets))

    def test_graduated_pairs_are_kept_for_the_fallback_only(self):
        pair = dexscreener_pair(dex="pumpswap")
        venue = self._venue()
        market = venue._parse_dexscreener(pair)
        self.assertIsNotNone(market)
        self.assertTrue(market.meta["complete"], "pumpswap means it graduated")
        # DexScreener only indexes tokens that already have a pair, so applying
        # the bonding-curve filter here would empty the fallback.
        self.assertTrue(venue._passes_filters(market))

        strict = self._venue(pumpfun_fallback_allow_graduated=False)
        self.assertFalse(strict._passes_filters(strict._parse_dexscreener(pair)))

    def test_unpriced_pair_is_dropped(self):
        venue = self._venue()
        self.assertIsNone(venue._parse_dexscreener(dexscreener_pair(price=None)))
        self.assertIsNone(venue._parse_dexscreener({"baseToken": {"address": "x"}}))

    def test_quote_missing_prices_strays_in_one_batch(self):
        venue = self._venue()
        seen = []

        async def _fetch(url, **_kwargs):
            seen.append(url)
            return [dexscreener_pair("So1pump"), dexscreener_pair("So2pump")]

        with patch("bottrade.venues.pumpfun.fetch_json", new=_fetch):
            prices = asyncio.run(
                venue.quote_missing(["pumpfun:So1pump", "pumpfun:So2pump"])
            )

        self.assertEqual(len(seen), 1, "one request, not one per position")
        self.assertAlmostEqual(prices["pumpfun:So1pump"], 0.00042)
        self.assertEqual(len(prices), 2)

    def test_quote_missing_degrades_to_nothing(self):
        venue = self._venue()

        async def _down(url, **_kwargs):
            raise HttpError("HTTP 500")

        with patch("bottrade.venues.pumpfun.fetch_json", new=_down):
            self.assertEqual(asyncio.run(venue.quote_missing(["pumpfun:So1pump"])), {})


class TestSourceIsVisible(unittest.TestCase):
    """A silent fallback is indistinguishable from a dead venue."""

    def setUp(self) -> None:
        http.cache_clear()

    def _venue(self) -> PumpFunVenue:
        return PumpFunVenue(Config({}))

    def _fallback_routes(self) -> dict[str, object]:
        return {
            "geckoterminal": HttpError("HTTP 429"),
            "frontend-api": HttpError("HTTP 503"),
            "token-profiles": [{"chainId": "solana", "tokenAddress": "So1pump"}],
            "token-boosts": [],
            "tokens/v1": [dexscreener_pair("So1pump")],
        }

    def test_fallback_is_announced_once_not_every_tick(self):
        venue = self._venue()
        with patch("bottrade.venues.pumpfun.fetch_json", new=routed(self._fallback_routes())):
            with self.assertLogs("bottrade.pumpfun", level="WARNING") as captured:
                asyncio.run(venue.list_markets())
                asyncio.run(venue.list_markets())
        self.assertEqual(len(captured.records), 1, "one warning, not one per tick")
        self.assertIn("dexscreener", captured.output[0])

    def test_a_total_outage_is_reported_as_such(self):
        venue = self._venue()
        routes = {
            "geckoterminal": HttpError("HTTP 429"),
            "frontend-api": HttpError("HTTP 503"),
            "token-profiles": HttpError("HTTP 500"),
            "token-boosts": HttpError("HTTP 500"),
        }
        with patch("bottrade.venues.pumpfun.fetch_json", new=routed(routes)):
            with self.assertLogs("bottrade.pumpfun", level="WARNING") as captured:
                markets = asyncio.run(venue.list_markets())
        self.assertEqual(markets, [])
        self.assertIn("every source failed", captured.output[0])

    def test_recovery_is_logged_at_info(self):
        venue = self._venue()
        with patch("bottrade.venues.pumpfun.fetch_json", new=routed(self._fallback_routes())):
            asyncio.run(venue.list_markets())
        self.assertEqual(venue.source, "dexscreener")

        healthy = {"geckoterminal": {"data": [{"attributes": {}, "relationships": {}}]}}
        with patch("bottrade.venues.pumpfun.fetch_json", new=routed(healthy)):
            with self.assertLogs("bottrade.pumpfun", level="INFO") as captured:
                asyncio.run(venue.list_markets())
        self.assertEqual(venue.source, "geckoterminal")
        self.assertIn("source recovered", captured.output[0])


def binary_market(price=0.50) -> Market:
    return Market(
        venue=Venue.POLYMARKET,
        market_id="cond1",
        question="Will it happen?",
        token_id="tok1",
        price=price,
        bid=price - 0.01,
        ask=price + 0.01,
        liquidity_usd=100_000,
        volume_24h_usd=100_000,
        end_date=utcnow() + timedelta(days=10),
    )


def stray(mint="So1pump", minutes_old=20.0) -> Position:
    return Position(
        key=f"pumpfun:{mint}",
        venue=Venue.PUMPFUN,
        token_id=mint,
        question="TEST",
        qty=1000.0,
        avg_price=1e-5,
        last_price=1e-5,
        high_water=1e-5,
        marked_at=utcnow() - timedelta(minutes=minutes_old),
    )


class FakeAdapter:
    """Stands in for a venue adapter; records what it was asked for."""

    def __init__(self, prices=None, error=None):
        self.prices = prices or {}
        self.error = error
        self.asked: list[list[str]] = []

    async def quote_missing(self, keys):
        self.asked.append(list(keys))
        if self.error:
            raise self.error
        return {k: v for k, v in self.prices.items() if k in keys}


class TestScannerRemarksStrays(unittest.TestCase):
    """Every open position must get a fresh mark each tick, in or out of quota."""

    def _run(self, broker, adapter) -> Context:
        cfg = Config({"scanner": {"min_liquidity_usd": 100.0}})
        ctx = Context(cfg=cfg, broker=broker, log=logging.getLogger("test"))
        ctx.markets = [binary_market()]
        ctx.adapters = {Venue.PUMPFUN: adapter}
        asyncio.run(ScannerAgent(cfg).run(ctx))
        return ctx

    def test_out_of_quota_position_is_requoted(self):
        broker = PaperBroker(cash=1000.0)
        broker.positions["pumpfun:So1pump"] = stray()
        adapter = FakeAdapter({"pumpfun:So1pump": 2e-5})

        self._run(broker, adapter)

        position = broker.positions["pumpfun:So1pump"]
        self.assertAlmostEqual(position.last_price, 2e-5)
        self.assertLess((utcnow() - position.marked_at).total_seconds(), 5)
        self.assertEqual(adapter.asked, [["pumpfun:So1pump"]])

    def test_in_universe_positions_are_not_requoted(self):
        broker = PaperBroker(cash=1000.0)
        broker.positions["polymarket:tok1"] = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Will it happen?", qty=10.0, avg_price=0.5, last_price=0.5,
        )
        adapter = FakeAdapter()

        self._run(broker, adapter)

        self.assertEqual(adapter.asked, [], "the scan already priced this one")
        self.assertAlmostEqual(broker.positions["polymarket:tok1"].last_price, 0.50)

    def test_a_failing_quote_does_not_kill_the_tick(self):
        broker = PaperBroker(cash=1000.0)
        broker.positions["pumpfun:So1pump"] = stray()
        adapter = FakeAdapter(error=RuntimeError("venue exploded"))

        ctx = self._run(broker, adapter)

        self.assertTrue(ctx.markets, "the universe still gets published")
        self.assertAlmostEqual(broker.positions["pumpfun:So1pump"].last_price, 1e-5)

    def test_a_venue_with_no_adapter_is_skipped(self):
        broker = PaperBroker(cash=1000.0)
        broker.positions["pumpfun:So1pump"] = stray()
        cfg = Config({"scanner": {"min_liquidity_usd": 100.0}})
        ctx = Context(cfg=cfg, broker=broker, log=logging.getLogger("test"))
        ctx.markets = [binary_market()]
        ctx.adapters = {}

        asyncio.run(ScannerAgent(cfg).run(ctx))

        self.assertAlmostEqual(broker.positions["pumpfun:So1pump"].last_price, 1e-5)


if __name__ == "__main__":
    unittest.main()
