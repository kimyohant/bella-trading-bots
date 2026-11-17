"""Sniper agent: the new-pools feed, the launch gates, and the separate book.

Three things can go wrong here and only one of them is a bad trade:

* the feed silently returns nothing, so the agent looks idle but is broken;
* the gates let through the two shapes that kill snipers — a launch with one
  wallet talking to itself, and one that is already vertical;
* a $15 ticket eats a pump.fun slot meant for a $43 position the model priced.

The tests below pin one behaviour per failure.
"""

from __future__ import annotations

import asyncio
import logging
import unittest
from datetime import timedelta
from unittest.mock import patch

from bottrade.agents.base import Context
from bottrade.agents.risk import RiskAgent
from bottrade.agents.sniper import SniperAgent
from bottrade.config import Config
from bottrade.data.http import HttpError
from bottrade.models import (
    SNIPER_PROFILE,
    Market,
    Order,
    Position,
    Side,
    Signal,
    Venue,
)
from bottrade.utils import utcnow
from bottrade.venues.paper import PaperBroker
from bottrade.venues.pumpfun import PumpFunVenue


# --------------------------------------------------------------- fixtures
def launch(
    symbol="DEMO",
    price=1.6e-05,
    liquidity=29_000.0,
    buyers=96.0,
    buys=96.0,
    sells=13.0,
    change=13.0,
    fdv=16_000.0,
    age=2.0,
    token=None,
) -> Market:
    """A pool as the new-pools feed hands it over, after parsing."""
    token = token or symbol.lower()
    return Market(
        venue=Venue.PUMPFUN,
        market_id=token,
        question=symbol,
        token_id=token,
        outcome="TOKEN",
        price=price,
        bid=price * 0.99,
        ask=price * 1.01,
        liquidity_usd=liquidity,
        volume_24h_usd=liquidity,
        end_date=None,
        tags=("memecoin", "pumpfun"),
        meta={
            "symbol": symbol,
            "age_minutes": age,
            "buyers_m5": buyers,
            "buys_m5": buys,
            "sells_m5": sells,
            "change_m5": change,
            "market_cap_usd": fdv,
            "profile": SNIPER_PROFILE,
        },
    )


def broker(cash=1_000.0) -> PaperBroker:
    return PaperBroker(
        cash=cash,
        starting_cash=cash,
        fee_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 100.0},
        slippage_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
        fixed_fee_usd={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
    )


class FakeFeed:
    """Stands in for the venue adapter: only `new_pools` matters here."""

    def __init__(self, pools, error=None):
        self.pools = pools
        self.error = error
        self.calls = 0

    async def new_pools(self, limit=20):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.pools)[:limit]


def context(cfg, brk, feed) -> Context:
    return Context(
        cfg=cfg,
        broker=brk,
        log=logging.getLogger("test"),
        adapters={Venue.PUMPFUN: feed},
    )


def sniper_cfg(**overrides) -> Config:
    base = {
        "enabled": True,
        "max_age_minutes": 8.0,
        "min_liquidity_usd": 3_000.0,
        "min_buyers_m5": 25,
        "min_buy_sell_ratio": 1.1,
        "min_change_m5": -15.0,
        "max_change_m5": 60.0,
        "min_fdv_usd": 8_000.0,
        "max_fdv_usd": 250_000.0,
        "max_candidates": 2,
        "feed_limit": 20,
        "cooldown_minutes": 60.0,
        "min_edge": 0.12,
        "signal_ttl_seconds": 120,
    }
    base.update(overrides)
    return Config({"sniper": base})


# ------------------------------------------------------------------- feed
def gecko_pool(symbol, dex="pumpfun", created_minutes_ago=2.0, price="0.0000162"):
    created = (utcnow() - timedelta(minutes=created_minutes_ago)).isoformat()
    return {
        "id": f"solana_{symbol.lower()}",
        "type": "pool",
        "attributes": {
            "name": f"{symbol} / SOL",
            "address": f"pool{symbol}",
            "base_token_price_usd": price,
            "pool_created_at": created,
            "reserve_in_usd": "29000.0",
            "market_cap_usd": "16000.0",
            "fdv_usd": "16000.0",
            "volume_usd": {"h24": "41000.0"},
            "transactions": {"h24": {"buys": 96, "sells": 13}, "m5": {"buys": 96, "sells": 13}},
            "price_change_percentage": {"m5": "13.1", "h24": "42.0"},
        },
        "relationships": {
            "dex": {"data": {"id": dex, "type": "dex"}},
            "base_token": {"data": {"id": f"solana_{symbol.lower()}", "type": "token"}},
        },
    }


class TestNewPoolsFeed(unittest.TestCase):
    """`new_pools` is the only place the agent can see a two-minute-old token."""

    def _feed(self, rows, error=None):
        async def fake_fetch(url, **kwargs):
            if error:
                raise error
            return {"data": rows}

        return fake_fetch

    def test_keeps_pumpfun_and_drops_the_rest_of_the_network(self):
        cfg = Config({"venues": {"pumpfun": {"new_pool_dexes": ["pumpfun", "pumpswap"]}}})
        rows = [
            gecko_pool("AAA", dex="pumpfun"),
            gecko_pool("BBB", dex="meteora"),
            gecko_pool("CCC", dex="pumpswap"),
            gecko_pool("DDD", dex="raydium"),
        ]
        with patch("bottrade.venues.pumpfun.fetch_json", new=self._feed(rows)):
            pools = asyncio.run(PumpFunVenue(cfg).new_pools(limit=20))

        self.assertEqual([p.question for p in pools], ["AAA", "CCC"])
        self.assertTrue(all(p.meta["profile"] == SNIPER_PROFILE for p in pools))

    def test_age_is_measured_from_pool_created_at(self):
        cfg = Config({})
        rows = [gecko_pool("AGE", created_minutes_ago=3.5)]
        with patch("bottrade.venues.pumpfun.fetch_json", new=self._feed(rows)):
            pools = asyncio.run(PumpFunVenue(cfg).new_pools(limit=20))

        self.assertAlmostEqual(pools[0].meta["age_minutes"], 3.5, delta=0.2)

    def test_a_dead_feed_is_empty_not_an_exception(self):
        cfg = Config({})
        with patch(
            "bottrade.venues.pumpfun.fetch_json",
            new=self._feed([], error=HttpError("429")),
        ):
            self.assertEqual(asyncio.run(PumpFunVenue(cfg).new_pools()), [])

    def test_rows_without_a_price_are_skipped(self):
        cfg = Config({})
        broken = gecko_pool("NOPRICE")
        broken["attributes"]["base_token_price_usd"] = None
        with patch(
            "bottrade.venues.pumpfun.fetch_json",
            new=self._feed([broken, gecko_pool("OK")]),
        ):
            pools = asyncio.run(PumpFunVenue(cfg).new_pools(limit=20))
        self.assertEqual([p.question for p in pools], ["OK"])


class TestLaunchGates(unittest.TestCase):
    """Each gate below is a shape that loses money if it is let through."""

    def _run(self, pools, cfg=None, brk=None, ctx=None):
        cfg = cfg or sniper_cfg()
        feed = FakeFeed(pools)
        ctx = ctx or context(cfg, brk or broker(), feed)
        asyncio.run(SniperAgent(cfg).run(ctx))
        return ctx

    def test_a_crowd_becomes_one_signal(self):
        ctx = self._run([launch()])
        self.assertEqual(len(ctx.signals), 1)
        signal = ctx.signals[0]
        self.assertIs(signal.side, Side.BUY)
        self.assertGreaterEqual(signal.edge, 0.12)
        # downstream pricing needs the market in the universe
        self.assertIn(signal.market.key, [m.key for m in ctx.markets])

    def test_one_wallet_talking_to_itself_is_refused(self):
        # 2 buyers, 18 buys: the "demand" is a single wallet clicking
        ctx = self._run([launch(buyers=2.0, buys=18.0, sells=1.0)])
        self.assertEqual(ctx.signals, [])
        self.assertTrue(any("buyers in 5m" in r["reason"] for r in ctx.rejections))

    def test_an_already_vertical_launch_is_not_chased(self):
        ctx = self._run([launch(change=180.0)])
        self.assertEqual(ctx.signals, [])
        self.assertTrue(any("already up" in r["reason"] for r in ctx.rejections))

    def test_an_already_dumping_launch_is_skipped(self):
        ctx = self._run([launch(change=-70.0)])
        self.assertEqual(ctx.signals, [])
        self.assertTrue(any("already dumped" in r["reason"] for r in ctx.rejections))

    def test_thin_liquidity_is_refused(self):
        ctx = self._run([launch(liquidity=900.0)])
        self.assertEqual(ctx.signals, [])
        self.assertTrue(any("liquidity" in r["reason"] for r in ctx.rejections))

    def test_a_stale_launch_is_not_even_considered(self):
        # older than max_age: not a decision, so not a rejection either
        ctx = self._run([launch(age=45.0)])
        self.assertEqual(ctx.signals, [])
        self.assertEqual(ctx.rejections, [])

    def test_only_the_best_candidates_are_taken(self):
        pools = [
            launch("AAA", buyers=30.0, buys=40.0, sells=10.0),
            launch("BBB", buyers=120.0, buys=400.0, sells=20.0),
            launch("CCC", buyers=60.0, buys=90.0, sells=20.0),
        ]
        ctx = self._run(pools, cfg=sniper_cfg(max_candidates=2))
        self.assertEqual(len(ctx.signals), 2)
        # widest crowd and heaviest pressure first
        self.assertEqual(ctx.signals[0].market.question, "BBB")

    def test_the_same_launch_is_not_bought_twice(self):
        cfg = sniper_cfg()
        brk = broker()
        ctx = context(cfg, brk, FakeFeed([launch()]))
        agent = SniperAgent(cfg)
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.signals), 1)

        again = Context(cfg=cfg, broker=brk, log=logging.getLogger("test"),
                        adapters=ctx.adapters, state=ctx.state)
        asyncio.run(agent.run(again))
        self.assertEqual(again.signals, [])

    def test_an_empty_feed_produces_nothing(self):
        ctx = self._run([])
        self.assertEqual(ctx.signals, [])
        self.assertEqual(ctx.rejections, [])

    def test_agent_is_silent_when_the_adapter_cannot_list_pools(self):
        cfg = sniper_cfg()
        ctx = Context(cfg=cfg, broker=broker(), log=logging.getLogger("test"),
                      adapters={Venue.PUMPFUN: object()})
        asyncio.run(SniperAgent(cfg).run(ctx))
        self.assertEqual(ctx.signals, [])


class TestSniperBook(unittest.TestCase):
    """The $15 lottery ticket must not spend the $43 book's slots."""

    def _risk(self, cfg, brk, signals):
        agent = RiskAgent(cfg)
        ctx = Context(cfg=cfg, broker=brk, log=logging.getLogger("test"))
        ctx.markets = [s.market for s in signals]
        ctx.signals = signals
        asyncio.run(agent.run(ctx))
        return ctx

    def _signal(self, m=None, fair=None):
        m = m or launch()
        return Signal(
            market=m,
            side=Side.BUY,
            fair_price=fair if fair is not None else m.price * 1.25,
            confidence=0.7,
            reasons=["test"],
        )

    def test_size_is_capped_at_the_sniper_ticket(self):
        cfg = Config({"risk": {
            "min_edge_after_costs": 0.0, "min_confidence": 0.0,
            "max_sniper_positions": 2, "max_sniper_trades_per_day": 15,
            "max_position_usd_sniper": 15.0,
            "max_position_usd": 500.0, "kelly_fraction": 1.0,
        }})
        ctx = self._risk(cfg, broker(), [self._signal()])
        self.assertEqual(len(ctx.orders), 1)
        self.assertLessEqual(ctx.orders[0].notional, 15.0 + 1e-6)

    def test_a_snipe_does_not_use_a_pumpfun_slot(self):
        cfg = Config({"risk": {
            "min_edge_after_costs": 0.0, "min_confidence": 0.0,
            "max_positions_per_venue_pumpfun": 0,
            "max_sniper_positions": 2, "max_position_usd_sniper": 15.0,
        }})
        ctx = self._risk(cfg, broker(), [self._signal()])
        self.assertEqual(len(ctx.orders), 1)
        self.assertFalse(any("venue cap" in r["reason"] for r in ctx.rejections))

    def test_the_book_is_two_tickets_wide(self):
        cfg = Config({"risk": {
            "min_edge_after_costs": 0.0, "min_confidence": 0.0,
            "max_sniper_positions": 2, "max_position_usd_sniper": 15.0,
            "max_position_usd": 500.0,
        }})
        brk = broker()
        for i in range(2):
            key = f"pumpfun:held{i}"
            brk.positions[key] = Position(
                key=key, venue=Venue.PUMPFUN, token_id=f"held{i}",
                question=f"HELD{i}", qty=1e6, avg_price=1e-5, last_price=1e-5,
                profile=SNIPER_PROFILE,
            )
        ctx = self._risk(cfg, brk, [self._signal()])
        self.assertEqual(ctx.orders, [])
        self.assertTrue(any("sniper cap" in r["reason"] for r in ctx.rejections))

    def test_the_daily_trade_count_stops_a_lottery_ticket_habit(self):
        cfg = Config({"risk": {
            "min_edge_after_costs": 0.0, "min_confidence": 0.0,
            "max_sniper_positions": 5, "max_sniper_trades_per_day": 3,
            "max_position_usd_sniper": 15.0,
        }})
        brk = broker()
        agent = RiskAgent(cfg)
        ctx = Context(cfg=cfg, broker=brk, log=logging.getLogger("test"))
        ctx.state["sniper_trades_today"] = 3
        ctx.markets = [launch()]
        ctx.signals = [self._signal()]
        asyncio.run(agent.run(ctx))
        self.assertEqual(ctx.orders, [])
        self.assertTrue(any("trades today" in r["reason"] for r in ctx.rejections))

    def test_the_exit_clock_is_the_thesis(self):
        # 25 minutes is a different position for a launch, and nothing for a
        # market the model actually priced
        cfg = Config({"risk": {"max_hold_minutes_sniper": 20.0, "max_hold_hours": 6.0}})
        brk = broker()
        pos = Position(
            key="pumpfun:demo", venue=Venue.PUMPFUN, token_id="demo",
            question="DEMO", qty=1e6, avg_price=1.6e-5, last_price=1.6e-5,
            high_water=1.6e-5, profile=SNIPER_PROFILE,
            opened_at=utcnow() - timedelta(minutes=25),
            marked_at=utcnow(),
        )
        brk.positions[pos.key] = pos
        agent = RiskAgent(cfg)
        ctx = Context(cfg=cfg, broker=brk, log=logging.getLogger("test"))
        ctx.markets = [launch()]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIs(ctx.orders[0].side, Side.SELL)
        self.assertIn("max-hold", ctx.orders[0].reason)


if __name__ == "__main__":
    unittest.main()
