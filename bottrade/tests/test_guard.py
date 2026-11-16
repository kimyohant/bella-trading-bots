"""Guard agent: staleness detection, no fabricated fills, dead-man's switch."""

from __future__ import annotations

import asyncio
import logging
import unittest
from datetime import timedelta

from bottrade.agents.base import Context
from bottrade.agents.guard import GuardAgent
from bottrade.agents.orchestrator import Orchestrator
from bottrade.agents.risk import RiskAgent
from bottrade.config import Config
from bottrade.data.store import Store
from bottrade.models import Market, Position, Side, Signal, Venue
from bottrade.utils import utcnow
from bottrade.venues.paper import PaperBroker


def market(price=0.50, venue=Venue.POLYMARKET, token="tok1", question="Q?") -> Market:
    return Market(
        venue=venue, market_id=token, question=question, token_id=token,
        price=price, bid=price - 0.01, ask=price + 0.01,
        liquidity_usd=100_000, volume_24h_usd=100_000,
        end_date=utcnow() + timedelta(days=10),
    )


def both() -> list[Market]:
    """One market per venue, so neither is accidentally 'quiet'."""
    return [market(), market(price=1e-5, venue=Venue.PUMPFUN, token="pf1")]


def broker(cash=1000.0) -> PaperBroker:
    return PaperBroker(
        cash=cash, starting_cash=cash,
        fee_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 100.0},
        slippage_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
        fixed_fee_usd={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
    )


def context(cfg: Config, brk: PaperBroker, markets=None, store=None) -> Context:
    ctx = Context(cfg=cfg, broker=brk, log=logging.getLogger("test"), store=store)
    ctx.markets = markets if markets is not None else []
    return ctx


def run(agent, ctx: Context) -> Context:
    asyncio.run(agent.run(ctx))
    return ctx


def ticks(guard: GuardAgent, ctx: Context, markets, n: int = 1) -> Context:
    for _ in range(n):
        ctx.tick += 1
        ctx.markets = list(markets)
        asyncio.run(guard.run(ctx))
    return ctx


class TestStaleness(unittest.TestCase):
    def test_venue_going_quiet_is_marked_stale(self):
        cfg = Config({"guard": {"max_quote_age_ticks": 2}})
        guard = GuardAgent(cfg)
        ctx = context(cfg, broker())

        ticks(guard, ctx, both(), 1)
        self.assertEqual(ctx.state["stale_venues"], set())

        ticks(guard, ctx, [], 1)   # miss 1 — still within tolerance
        self.assertEqual(ctx.state["stale_venues"], set())

        ticks(guard, ctx, [], 1)   # miss 2 — now stale
        self.assertIn(Venue.POLYMARKET, ctx.state["stale_venues"])
        self.assertTrue(any("stale" in note for note in ctx.notes))

    def test_recovery_clears_the_flag(self):
        cfg = Config({"guard": {"max_quote_age_ticks": 1}})
        guard = GuardAgent(cfg)
        ctx = context(cfg, broker())

        ticks(guard, ctx, both(), 1)
        ticks(guard, ctx, [], 1)
        self.assertIn(Venue.POLYMARKET, ctx.state["stale_venues"])

        ticks(guard, ctx, both(), 1)
        self.assertEqual(ctx.state["stale_venues"], set())
        self.assertTrue(ctx.state["health"]["polymarket"]["ok"])
        self.assertEqual(ctx.state["health"]["polymarket"]["miss"], 0)

    def test_health_rows_are_persisted(self):
        cfg = Config({"guard": {"max_quote_age_ticks": 1}})
        store = Store(":memory:")
        guard = GuardAgent(cfg)
        ctx = context(cfg, broker(), store=store)

        ticks(guard, ctx, both(), 1)
        rows = store.query("SELECT venue, ok FROM health WHERE tick = 1 ORDER BY venue")
        self.assertEqual([r["venue"] for r in rows], ["polymarket", "pumpfun"])
        self.assertEqual({r["ok"] for r in rows}, {1})


class TestNoFabricatedFills(unittest.TestCase):
    """A fill priced off a frozen mark is a fiction — hold instead."""

    def _setup(self, stale_action: str = "hold", age_hours: float = 0.0):
        cfg = Config({
            "guard": {"max_quote_age_ticks": 1, "max_stale_hours": 6.0,
                      "stale_action": stale_action},
            "risk": {"stop_loss_pct": 0.30},
        })
        brk = broker()
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.50, last_price=0.20,
            high_water=0.50, opened_at=utcnow() - timedelta(hours=age_hours),
        )
        brk.positions[pos.key] = pos
        return cfg, brk, pos

    def test_stale_mark_holds_instead_of_selling(self):
        cfg, brk, _ = self._setup()
        guard, risk = GuardAgent(cfg), RiskAgent(cfg)
        ctx = context(cfg, brk, [market()])

        ticks(guard, ctx, [], 1)          # polymarket stops answering
        self.assertIn(Venue.POLYMARKET, ctx.state["stale_venues"])

        run(risk, ctx)
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(any("stale-mark" in r["reason"] for r in ctx.rejections))

    def test_fresh_mark_still_triggers_the_stop(self):
        cfg, brk, _ = self._setup()
        guard, risk = GuardAgent(cfg), RiskAgent(cfg)
        ctx = context(cfg, brk, [market(price=0.20)])

        ticks(guard, ctx, [market(price=0.20)], 1)
        run(risk, ctx)
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("stop-loss", ctx.orders[0].reason)

    def test_forced_close_after_max_stale_hours(self):
        cfg, brk, _ = self._setup()
        guard, risk = GuardAgent(cfg), RiskAgent(cfg)
        ctx = context(cfg, brk, [market()])

        ticks(guard, ctx, [], 1)
        ctx.state["health"]["polymarket"]["stale_since"] = utcnow() - timedelta(hours=7)
        ticks(guard, ctx, [], 1)

        self.assertIn(Venue.POLYMARKET, ctx.state["stale_forced"])
        run(risk, ctx)
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("stale-forced", ctx.orders[0].reason)

    def test_liquidate_mode_sells_anyway(self):
        cfg, brk, _ = self._setup(stale_action="liquidate")
        guard, risk = GuardAgent(cfg), RiskAgent(cfg)
        ctx = context(cfg, brk, [market()])

        ticks(guard, ctx, [], 1)
        run(risk, ctx)
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("stop-loss", ctx.orders[0].reason)
        self.assertNotIn("stale-forced", ctx.orders[0].reason)


class TestDeadman(unittest.TestCase):
    def test_empty_feed_halts_entries(self):
        cfg = Config({
            "guard": {"max_empty_ticks": 2, "max_quote_age_ticks": 1},
            "risk": {"min_edge_after_costs": 0.0, "max_position_usd": 50},
        })
        guard, risk = GuardAgent(cfg), RiskAgent(cfg)
        ctx = context(cfg, broker())

        ticks(guard, ctx, [], 3)
        self.assertTrue(ctx.state.get("deadman"))

        ctx.markets = [market()]
        ctx.signals = [Signal(market=market(), side=Side.BUY, fair_price=0.90,
                              confidence=0.9)]
        run(risk, ctx)
        self.assertEqual(len(ctx.orders), 0)

    def test_data_returning_clears_the_switch(self):
        cfg = Config({"guard": {"max_empty_ticks": 2}})
        guard = GuardAgent(cfg)
        ctx = context(cfg, broker())

        ticks(guard, ctx, [], 3)
        self.assertTrue(ctx.state.get("deadman"))

        ticks(guard, ctx, [market()], 1)
        self.assertFalse(ctx.state.get("deadman"))
        self.assertEqual(ctx.state["empty_streak"], 0)


class ScriptedFeed:
    """Feed stub: returns each frame in turn, then stops."""

    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    async def step(self):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame

    async def close(self) -> None:
        return None


class TestPipelineWiring(unittest.TestCase):
    def test_guard_runs_first_and_sees_the_outage(self):
        cfg = Config({
            "agents": {"enabled": ["guard", "risk"]},
            "guard": {"max_quote_age_ticks": 1, "max_empty_ticks": 2},
            "risk": {"stop_loss_pct": 0.30},
        })
        brk = broker()
        brk.positions["polymarket:tok1"] = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.50, last_price=0.20,
        )
        frames = [[market(price=0.20)], [market(price=0.20)], [], [], []]
        orch = Orchestrator(cfg, feed=ScriptedFeed(frames),
                            agents=[GuardAgent(cfg), RiskAgent(cfg)], broker=brk)

        summary = asyncio.run(orch.run())

        self.assertEqual(summary["ticks"], 5)
        self.assertTrue(orch.ctx.state.get("deadman"))
        self.assertIn(Venue.POLYMARKET, orch.ctx.state["stale_venues"])
        # no exits at all: the venue went quiet and we refused to invent a fill
        self.assertEqual(summary["exits"], 0)


if __name__ == "__main__":
    unittest.main()
