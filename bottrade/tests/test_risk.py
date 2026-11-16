"""Risk agent: sizing maths, hard gates, exposure budgets, exits, circuit breakers."""

from __future__ import annotations

import asyncio
import logging
import unittest
from datetime import timedelta

from bottrade.agents.base import Context
from bottrade.agents.risk import RiskAgent
from bottrade.config import Config
from bottrade.models import Market, Order, Position, Side, Signal, Venue
from bottrade.utils import utcnow
from bottrade.venues.paper import PaperBroker


def market(price=0.50, bid=0.49, ask=0.51, venue=Venue.POLYMARKET, question="Q?") -> Market:
    return Market(
        venue=venue, market_id="m1", question=question, token_id="tok1",
        price=price, bid=bid, ask=ask, liquidity_usd=100_000,
        volume_24h_usd=100_000, end_date=utcnow() + timedelta(days=10),
    )


def broker(cash=1000.0) -> PaperBroker:
    return PaperBroker(
        cash=cash, starting_cash=cash,
        fee_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 100.0},
        slippage_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
        fixed_fee_usd={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
    )


def context(cfg: Config, brk: PaperBroker) -> Context:
    return Context(cfg=cfg, broker=brk, log=logging.getLogger("test"))


class TestSizing(unittest.TestCase):
    def test_kelly_fraction_for_binary_market(self):
        cfg = Config({"risk": {"kelly_fraction": 0.25, "max_position_pct": 0.5,
                               "max_position_usd": 10_000}})
        agent = RiskAgent(cfg)
        ctx = context(cfg, broker())

        # executable price is the ask (0.51), so kelly = (0.60-0.51)/(1-0.51) = 0.1837
        # quarter kelly = 4.59% of equity -> 45.92 on 1000
        signal = Signal(market=market(), side=Side.BUY, fair_price=0.60, confidence=0.8)
        size = agent._size(ctx, signal, equity=1_000.0, max_position_pct=0.5,
                           max_position_usd=10_000)
        self.assertAlmostEqual(size, 1_000 * 0.25 * (0.09 / 0.49), places=6)

    def test_position_size_is_capped(self):
        cfg = Config({"risk": {"kelly_fraction": 1.0, "max_position_pct": 0.08,
                               "max_position_usd": 30}})
        agent = RiskAgent(cfg)
        ctx = context(cfg, broker())
        signal = Signal(market=market(), side=Side.BUY, fair_price=0.95, confidence=0.9)
        size = agent._size(ctx, signal, equity=1_000.0, max_position_pct=0.08,
                           max_position_usd=30)
        self.assertAlmostEqual(size, 30.0, places=6)

    def test_token_sizing_uses_fixed_risk(self):
        cfg = Config({"risk": {"risk_per_trade_pct": 0.02, "stop_loss_pct": 0.4,
                               "max_position_pct": 1.0, "max_position_usd": 10_000}})
        agent = RiskAgent(cfg)
        ctx = context(cfg, broker())
        token = market(price=2e-5, bid=1.98e-5, ask=2.02e-5, venue=Venue.PUMPFUN)
        signal = Signal(market=token, side=Side.BUY, fair_price=2.4e-5, confidence=0.8)
        size = agent._size(ctx, signal, equity=1_000.0, max_position_pct=1.0,
                           max_position_usd=10_000)
        # 2% risk / 40% stop = 5% of equity
        self.assertAlmostEqual(size, 50.0, places=6)


class TestGates(unittest.TestCase):
    def _run(self, cfg, brk, ctx_signals, markets=None):
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = markets if markets is not None else [s.market for s in ctx_signals]
        ctx.signals = ctx_signals
        asyncio.run(agent.run(ctx))
        return ctx

    def test_rejects_edge_below_after_cost_threshold(self):
        cfg = Config({"risk": {"min_edge_after_costs": 0.10, "min_confidence": 0.0}})
        m = market()
        signal = Signal(market=m, side=Side.BUY, fair_price=0.52, confidence=0.9)
        ctx = self._run(cfg, broker(), [signal])
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(any("edge" in r["reason"] for r in ctx.rejections))

    def test_rejects_wide_spread(self):
        cfg = Config({"risk": {"max_spread_pct": 0.02, "min_edge_after_costs": 0.0}})
        m = market(price=0.5, bid=0.40, ask=0.60)  # 40% spread
        signal = Signal(market=m, side=Side.BUY, fair_price=0.80, confidence=0.9)
        ctx = self._run(cfg, broker(), [signal])
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(any("spread" in r["reason"] for r in ctx.rejections))

    def test_blocks_duplicate_market(self):
        cfg = Config({"risk": {"min_edge_after_costs": 0.0}})
        m = market()
        brk = broker()
        brk.positions["polymarket:other"] = Position(
            key="polymarket:other", venue=Venue.POLYMARKET, token_id="other",
            question="Q?", qty=10, avg_price=0.5, last_price=0.5,
        )
        signal = Signal(market=m, side=Side.BUY, fair_price=0.7, confidence=0.9)
        ctx = self._run(cfg, brk, [signal])
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(any("already positioned" in r["reason"] for r in ctx.rejections))

    def test_respects_max_open_positions(self):
        cfg = Config({"risk": {"max_open_positions": 2, "min_edge_after_costs": 0.0}})
        brk = broker()
        for i in range(2):
            key = f"polymarket:other{i}"
            brk.positions[key] = Position(
                key=key, venue=Venue.POLYMARKET, token_id=f"other{i}",
                question=f"Other {i}", qty=10, avg_price=0.5, last_price=0.5,
            )
        signal = Signal(market=market(), side=Side.BUY, fair_price=0.7, confidence=0.9)
        ctx = self._run(cfg, brk, [signal])
        self.assertEqual(len(ctx.orders), 0)

    def test_exposure_budget_clips_size(self):
        cfg = Config({
            "risk": {"min_edge_after_costs": 0.0, "max_total_exposure_pct": 0.05,
                     "max_position_usd": 1_000, "kelly_fraction": 1.0,
                     "max_position_pct": 1.0, "min_notional_usd": 0.01},
        })
        signal = Signal(market=market(), side=Side.BUY, fair_price=0.90, confidence=0.9)
        ctx = self._run(cfg, broker(cash=1_000), [signal])
        self.assertEqual(len(ctx.orders), 1)
        self.assertLessEqual(ctx.orders[0].notional, 1_000 * 0.05 + 1e-6)

    def test_daily_loss_limit_halts_entries(self):
        cfg = Config({"risk": {"daily_loss_limit_pct": 0.05, "min_edge_after_costs": 0.0}})
        brk = broker(cash=900)  # equity already down 10% from 1000
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.state["day_start_equity"] = 1_000.0
        ctx.state["day"] = ctx.now.date()
        ctx.markets = [market()]
        ctx.signals = [Signal(market=market(), side=Side.BUY, fair_price=0.9, confidence=0.9)]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(ctx.state.get("halted"))


class TestExits(unittest.TestCase):
    def _position(self, brk, avg=0.50, last=0.50, age_hours=0.0):
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=avg, last_price=last,
            opened_at=utcnow() - timedelta(hours=age_hours),
        )
        brk.positions[pos.key] = pos
        return pos

    def test_stop_loss_emits_sell(self):
        cfg = Config({"risk": {"stop_loss_pct": 0.3}})
        brk = broker()
        self._position(brk, avg=0.50, last=0.20)
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=0.20, bid=0.19, ask=0.21)]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIs(ctx.orders[0].side, Side.SELL)
        self.assertIn("stop-loss", ctx.orders[0].reason)

    def test_take_profit_emits_sell(self):
        cfg = Config({"risk": {"take_profit_pct": 0.5}})
        brk = broker()
        self._position(brk, avg=0.50, last=0.90)
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=0.90, bid=0.89, ask=0.91)]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("take-profit", ctx.orders[0].reason)

    def test_time_stop_emits_sell(self):
        cfg = Config({"risk": {"max_hold_hours": 1.0}})
        brk = broker()
        self._position(brk, avg=0.50, last=0.51, age_hours=3.0)
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=0.51, bid=0.50, ask=0.52)]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("max-hold", ctx.orders[0].reason)

    def test_healthy_position_is_left_alone(self):
        cfg = Config({"risk": {"stop_loss_pct": 0.3, "take_profit_pct": 1.0,
                               "max_hold_hours": 100}})
        brk = broker()
        self._position(brk, avg=0.50, last=0.52)
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=0.52, bid=0.51, ask=0.53)]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 0)


if __name__ == "__main__":
    unittest.main()
