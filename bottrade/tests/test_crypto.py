"""15-minute crypto up/down markets: the model, the gates, the risk profile."""

from __future__ import annotations

import asyncio
import logging
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from bottrade.agents.base import Context
from bottrade.agents.crypto import CryptoAgent
from bottrade.agents.executor import ExecutorAgent
from bottrade.agents.risk import RiskAgent
from bottrade.agents.scanner import ScannerAgent
from bottrade.agents.signal import SignalAgent
from bottrade.config import Config
from bottrade.models import CRYPTO_PROFILE, Market, Position, Side, Signal, Venue
from bottrade.utils import utcnow
from bottrade.venues.paper import PaperBroker

SYMBOL = "BTC"


def crypto_market(leg="Up", price=0.50, spread=0.01, seconds_left=600,
                  symbol=SYMBOL, liquidity=10_000.0) -> Market:
    end_ts = int(utcnow().timestamp() + seconds_left)
    return Market(
        venue=Venue.POLYMARKET,
        market_id="cond1",
        question=f"{symbol} Up or Down",
        token_id=f"{symbol}-{leg}",
        outcome=leg,
        price=price,
        bid=round(price - spread / 2, 4),
        ask=round(price + spread / 2, 4),
        liquidity_usd=liquidity,
        volume_24h_usd=500,
        end_date=datetime.fromtimestamp(end_ts, timezone.utc),
        meta={
            "profile": CRYPTO_PROFILE,
            "symbol": symbol,
            "window_start": end_ts - 900,
            "window_end": end_ts,
            "leg": leg,
        },
    )


def plain_market(price=0.5, hours=10.0, liquidity=50_000.0) -> Market:
    return Market(
        venue=Venue.POLYMARKET, market_id="m2", question="Will it rain?",
        token_id="plain", outcome="Yes", price=price, bid=price - 0.01,
        ask=price + 0.01, liquidity_usd=liquidity, volume_24h_usd=50_000,
        end_date=utcnow() + timedelta(hours=hours),
    )


def broker(cash=1000.0) -> PaperBroker:
    return PaperBroker(
        cash=cash, starting_cash=cash,
        fee_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 100.0},
        slippage_bps={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
        fixed_fee_usd={Venue.POLYMARKET: 0.0, Venue.PUMPFUN: 0.0},
    )


def context(cfg, brk, markets) -> Context:
    ctx = Context(cfg=cfg, broker=brk, log=logging.getLogger("test"))
    ctx.markets = markets
    return ctx


def run_crypto(agent: CryptoAgent, ctx: Context, spot: float, open_px: float) -> Context:
    """Drive the agent with a stubbed spot feed."""
    start_ts = ctx.markets[0].meta["window_start"]
    candles = [(start_ts, open_px, open_px), (start_ts + 60, open_px, spot)]
    with patch("bottrade.data.spot.spot_price", new=AsyncMock(return_value=spot)), \
         patch("bottrade.data.spot.candles", new=AsyncMock(return_value=candles)):
        asyncio.run(agent.run(ctx))
    return ctx


class TestFairValue(unittest.TestCase):
    def test_move_below_basis_noise_is_blocked(self):
        cfg = Config({"crypto": {"basis_bps": 4.0, "min_z": 0.0}})
        ctx = context(cfg, broker(), [crypto_market(seconds_left=600)])
        run_crypto(CryptoAgent(cfg), ctx, spot=100.02, open_px=100.00)  # +2bp

        info = ctx.state["crypto"]["polymarket:BTC-Up"]
        self.assertIsNotNone(info["blocked"])
        self.assertIn("basis", info["blocked"])

    def test_weak_z_is_blocked(self):
        cfg = Config({"crypto": {"basis_bps": 0.0, "min_z": 1.0}})
        ctx = context(cfg, broker(), [crypto_market(seconds_left=600)])
        # +10bp with a big sigma leaves z well under 1
        run_crypto(CryptoAgent(cfg), ctx, spot=100.10, open_px=100.00)

        info = ctx.state["crypto"]["polymarket:BTC-Up"]
        self.assertIsNotNone(info["blocked"])
        self.assertIn("z=", info["blocked"])

    def test_probability_rises_with_the_move(self):
        cfg = Config({"crypto": {"basis_bps": 0.0, "min_z": 0.0}})
        up = context(cfg, broker(), [crypto_market(leg="Up", seconds_left=600)])
        run_crypto(CryptoAgent(cfg), up, spot=100.50, open_px=100.00)

        down = context(cfg, broker(), [crypto_market(leg="Down", seconds_left=600)])
        run_crypto(CryptoAgent(cfg), down, spot=100.50, open_px=100.00)

        up_info = up.state["crypto"]["polymarket:BTC-Up"]
        down_info = down.state["crypto"]["polymarket:BTC-Down"]
        self.assertGreater(up_info["fair"], 0.5)
        self.assertAlmostEqual(up_info["fair"] + down_info["fair"], 1.0, places=6)

    def test_no_output_when_the_window_is_almost_over(self):
        cfg = Config({"crypto": {"min_seconds_left": 300}})
        ctx = context(cfg, broker(), [crypto_market(seconds_left=120)])
        run_crypto(CryptoAgent(cfg), ctx, spot=100.50, open_px=100.00)
        self.assertEqual(ctx.state["crypto"], {})


class TestCryptoSignal(unittest.TestCase):
    def _ctx(self, cfg, fair, price):
        market = crypto_market(price=price, seconds_left=600)
        ctx = context(cfg, broker(), [market])
        ctx.state["crypto"] = {
            market.key: {
                "symbol": SYMBOL, "fair": fair, "p_up": fair, "z": 1.4,
                "move_bps": 12.0, "tau_s": 600.0, "sigma_15m": 0.003,
                "blocked": None,
            }
        }
        return ctx, market

    def test_emits_when_the_leg_is_cheap(self):
        cfg = Config({"signal": {"min_edge": 0.08, "min_confidence": 0.5}})
        ctx, market = self._ctx(cfg, fair=0.62, price=0.50)
        asyncio.run(SignalAgent(cfg).run(ctx))
        self.assertEqual(len(ctx.signals), 1)
        self.assertEqual(ctx.signals[0].market.key, market.key)
        self.assertGreater(ctx.signals[0].edge, 0.08)

    def test_skips_when_the_price_already_reflects_the_model(self):
        cfg = Config({"signal": {"min_edge": 0.08, "min_confidence": 0.5}})
        ctx, _ = self._ctx(cfg, fair=0.62, price=0.61)
        asyncio.run(SignalAgent(cfg).run(ctx))
        self.assertEqual(ctx.signals, [])

    def test_skips_blocked_markets(self):
        cfg = Config({"signal": {"min_edge": 0.0, "min_confidence": 0.0}})
        ctx, market = self._ctx(cfg, fair=0.90, price=0.40)
        ctx.state["crypto"][market.key]["blocked"] = "inside basis noise"
        asyncio.run(SignalAgent(cfg).run(ctx))
        self.assertEqual(ctx.signals, [])


class TestCryptoRisk(unittest.TestCase):
    def _position(self, brk, seconds_left, age_min=2.0, avg=0.50, last=0.50):
        market = crypto_market(seconds_left=seconds_left, price=last)
        pos = Position(
            key=market.key, venue=Venue.POLYMARKET, token_id=market.token_id,
            question=market.question, qty=100, avg_price=avg, last_price=last,
            high_water=max(avg, last), profile=CRYPTO_PROFILE,
            opened_at=utcnow() - timedelta(minutes=age_min),
        )
        brk.positions[pos.key] = pos
        return market

    def test_hold_time_is_minutes_not_hours(self):
        cfg = Config({"risk": {"max_hold_minutes_crypto": 10}})
        brk = broker()
        market = self._position(brk, seconds_left=600, age_min=12.0)
        ctx = context(cfg, brk, [market])
        asyncio.run(RiskAgent(cfg).run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("max-hold", ctx.orders[0].reason)

    def test_flat_before_the_window_closes(self):
        cfg = Config({"risk": {"exit_seconds_before_window": 120}})
        brk = broker()
        market = self._position(brk, seconds_left=60, age_min=3.0)
        ctx = context(cfg, brk, [market])
        asyncio.run(RiskAgent(cfg).run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("window-end", ctx.orders[0].reason)

    def test_position_cap_is_separate_from_the_venue_cap(self):
        cfg = Config({
            "risk": {"max_crypto_positions": 1, "max_positions_per_venue": 8,
                     "min_edge_after_costs": 0.0, "max_position_usd_crypto": 40},
        })
        brk = broker()
        self._position(brk, seconds_left=600)  # one crypto position already open
        signal = Signal(market=crypto_market(leg="Down", price=0.50, seconds_left=600),
                        side=Side.BUY, fair_price=0.90, confidence=0.9)
        ctx = context(cfg, brk, [signal.market])
        ctx.signals = [signal]
        asyncio.run(RiskAgent(cfg).run(ctx))
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(any("crypto cap" in r["reason"] for r in ctx.rejections))

    def test_size_is_capped_for_coin_flips(self):
        cfg = Config({
            "risk": {"min_edge_after_costs": 0.0, "max_position_usd_crypto": 40,
                     "max_position_usd": 10_000, "kelly_fraction": 1.0,
                     "max_position_pct": 1.0},
        })
        brk = broker()
        signal = Signal(market=crypto_market(price=0.50, seconds_left=600),
                        side=Side.BUY, fair_price=0.95, confidence=0.9)
        ctx = context(cfg, brk, [signal.market])
        ctx.signals = [signal]
        asyncio.run(RiskAgent(cfg).run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertLessEqual(ctx.orders[0].notional, 40.0 + 1e-6)


class TestScannerKeepsCrypto(unittest.TestCase):
    def test_short_window_survives_the_resolution_filter(self):
        cfg = Config({
            "scanner": {"min_hours_to_resolution": 1.0, "min_liquidity_usd": 3000,
                        "crypto_min_liquidity_usd": 400, "crypto_max_spread_pct": 0.08},
        })
        crypto = crypto_market(price=0.50, spread=0.01, seconds_left=600, liquidity=1000)
        short = plain_market(price=0.5, hours=0.25)   # a regular market about to close
        ctx = context(cfg, broker(), [crypto, short])
        asyncio.run(ScannerAgent(cfg).run(ctx))

        keys = {m.key for m in ctx.markets}
        self.assertIn(crypto.key, keys)
        self.assertNotIn(short.key, keys)


class TestExecutorProfile(unittest.TestCase):
    def test_fill_records_the_profile_and_the_daily_counter(self):
        cfg = Config({"risk": {"min_edge_after_costs": 0.0}})
        brk = broker()
        market = crypto_market(price=0.50, seconds_left=600)
        order = __import__("bottrade.models", fromlist=["Order"]).Order(
            market=market, side=Side.BUY, qty=50, limit_price=0.50, notional=25.0,
        )
        ctx = context(cfg, brk, [market])
        ctx.orders = [order]
        asyncio.run(ExecutorAgent(cfg).run(ctx))

        position = brk.positions[market.key]
        self.assertEqual(position.profile, CRYPTO_PROFILE)
        self.assertEqual(ctx.state.get("crypto_trades_today"), 1)


class ScriptedFeed:
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


class TestEndToEnd(unittest.TestCase):
    """Buy the cheap leg, then get forced flat before the window resolves."""

    def test_entry_then_window_end_exit(self):
        from bottrade.agents.base import build_agents
        from bottrade.agents.orchestrator import Orchestrator

        cfg = Config({
            "agents": {"enabled": ["scanner", "crypto", "signal", "risk",
                                   "executor", "portfolio"]},
            "crypto": {"min_seconds_left": 300, "basis_bps": 0.0, "min_z": 0.0,
                       "min_edge": 0.02},
            "signal": {"min_edge": 0.02, "min_confidence": 0.0, "max_crypto_signals": 2},
            "risk": {"exit_seconds_before_window": 120, "min_edge_after_costs": 0.0,
                     "max_position_usd_crypto": 40, "max_crypto_positions": 2,
                     "stop_loss_pct_crypto": 0.30},
        })
        first = crypto_market(leg="Up", price=0.40, seconds_left=600)
        last = crypto_market(leg="Up", price=0.40, seconds_left=60)
        feed = ScriptedFeed([[first], [last]])

        brk = broker()
        orch = Orchestrator(cfg, feed=feed, agents=build_agents(cfg), broker=brk)
        start_ts = first.meta["window_start"]
        candles = [(start_ts, 100.00, 100.00)]

        with patch("bottrade.data.spot.spot_price", new=AsyncMock(return_value=100.60)), \
             patch("bottrade.data.spot.candles", new=AsyncMock(return_value=candles)):
            summary = asyncio.run(orch.run())

        self.assertEqual(summary["fills"], 2)          # one buy, one sell
        self.assertEqual(summary["open_positions"], 0)
        exits = [f for f in brk.fills if f.tag == "exit"]
        self.assertEqual(len(exits), 1)
        self.assertIn("window-end", exits[0].reason)


if __name__ == "__main__":
    unittest.main()
