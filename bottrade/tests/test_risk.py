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
    def _position(self, brk, avg=0.50, last=0.50, age_hours=0.0, high_water=None,
                  partial_taken=False):
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=avg, last_price=last,
            high_water=high_water if high_water is not None else max(avg, last),
            partial_taken=partial_taken,
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


class TestProfitTaking(unittest.TestCase):
    """The bug this covers: a +90% winner round-tripping back to the stop-loss."""

    def _run(self, cfg, brk, price):
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=price, bid=price - 0.01, ask=price + 0.01)]
        asyncio.run(agent.run(ctx))
        return ctx

    def test_trailing_stop_locks_in_profit(self):
        cfg = Config({"risk": {"trail_arm_pct": 0.35, "trail_giveback_pct": 0.30,
                               "take_profit_pct": 5.0}})
        brk = broker()
        # avg 0.40, ran to 0.76 (+90%), now back at 0.50 (+25%)
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.40, last_price=0.50,
            high_water=0.76, opened_at=utcnow(),
        )
        brk.positions[pos.key] = pos

        ctx = self._run(cfg, brk, 0.50)
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("trailing-stop", ctx.orders[0].reason)
        self.assertAlmostEqual(ctx.orders[0].qty, 100, places=6)

    def test_trailing_not_armed_below_threshold(self):
        cfg = Config({"risk": {"trail_arm_pct": 0.35, "trail_giveback_pct": 0.30,
                               "take_profit_pct": 5.0, "stop_loss_pct": 0.5}})
        brk = broker()
        # peak only +20%: never armed, and +5% now is not a loss either
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.40, last_price=0.42,
            high_water=0.48, opened_at=utcnow(),
        )
        brk.positions[pos.key] = pos
        ctx = self._run(cfg, brk, 0.42)
        self.assertEqual(len(ctx.orders), 0)

    def test_partial_take_profit_sells_half(self):
        cfg = Config({"risk": {"take_profit_partial_pct": 0.60,
                               "take_profit_partial_fraction": 0.5,
                               "take_profit_pct": 1.5, "trail_arm_pct": 5.0}})
        brk = broker()
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.40, last_price=0.66,
            high_water=0.66, opened_at=utcnow(),
        )
        brk.positions[pos.key] = pos

        ctx = self._run(cfg, brk, 0.66)
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("partial", ctx.orders[0].reason)
        self.assertAlmostEqual(ctx.orders[0].qty, 50, places=6)

    def test_partial_does_not_fire_twice(self):
        cfg = Config({"risk": {"take_profit_partial_pct": 0.60,
                               "take_profit_partial_fraction": 0.5,
                               "take_profit_pct": 1.5, "trail_arm_pct": 5.0}})
        brk = broker()
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=50, avg_price=0.40, last_price=0.66,
            high_water=0.66, partial_taken=True, opened_at=utcnow(),
        )
        brk.positions[pos.key] = pos
        ctx = self._run(cfg, brk, 0.66)
        self.assertEqual(len(ctx.orders), 0)

    def test_hard_target_still_wins_over_trailing(self):
        cfg = Config({"risk": {"take_profit_pct_token": 1.0, "trail_arm_pct": 0.35,
                               "trail_giveback_pct": 0.30}})
        brk = broker()
        pos = Position(
            key="pumpfun:tok1", venue=Venue.PUMPFUN, token_id="tok1",
            question="T", qty=1_000_000, avg_price=1e-5, last_price=2.1e-5,
            high_water=2.1e-5, opened_at=utcnow(),
        )
        brk.positions[pos.key] = pos
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=2.1e-5, bid=2.08e-5, ask=2.12e-5,
                              venue=Venue.PUMPFUN)]
        asyncio.run(agent.run(ctx))
        self.assertEqual(len(ctx.orders), 1)
        self.assertIn("take-profit", ctx.orders[0].reason)
        self.assertNotIn("trailing", ctx.orders[0].reason)


    def test_lower_arm_frees_a_slot_sooner(self):
        # The live symptom: six positions, three of them between +10% and +25%,
        # none of them reaching the +35% arm — so nothing exited, no slot ever
        # freed, and 121 entries died at "venue cap" while the money was idle.
        # Same position, same price: only the arm differs.
        brk = broker()
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.40, last_price=0.46,
            high_water=0.49, opened_at=utcnow(),  # peak +22.5%, now +15%
        )
        brk.positions[pos.key] = pos

        tight = self._run(Config({"risk": {
            "trail_arm_pct": 0.35, "trail_giveback_pct": 0.30,
            "take_profit_pct": 5.0, "max_hold_hours": 100,
        }}), broker(), 0.46)
        self.assertEqual(len(tight.orders), 0, "arm 0.35 must still hold")

        loose = self._run(Config({"risk": {
            "trail_arm_pct": 0.20, "trail_giveback_pct": 0.30,
            "take_profit_pct": 5.0, "max_hold_hours": 100,
        }}), brk, 0.46)
        self.assertEqual(len(loose.orders), 1, "arm 0.20 must exit")
        self.assertIn("trailing-stop", loose.orders[0].reason)


class TestPerVenueSlotCap(unittest.TestCase):
    """pump.fun ran out of *slots* while its *budget* was still free.

    Four slots at ~$43 each is $172 against a 30% venue budget of ~$300, so the
    count limit bound first and the extra capacity was unreachable.
    """

    def _cfg(self, **risk):
        base = {
            "min_edge_after_costs": 0.0,
            "max_spread_pct": 0.99,
            "min_liquidity_usd": 0.0,
            "max_open_positions": 12,
            "max_positions_per_venue": 4,
            "max_per_venue_pct": 0.99,
            "max_total_exposure_pct": 0.99,
            "cooldown_minutes": 0.0,
        }
        base.update(risk)
        return Config({"risk": base})

    def _open(self, brk, i, venue):
        key = f"{venue.value}:t{i}"
        brk.positions[key] = Position(
            key=key, venue=venue, token_id=f"t{i}", question=f"Q{i}?",
            qty=100, avg_price=0.5, last_price=0.5, high_water=0.5,
            opened_at=utcnow(),
        )

    def _run(self, cfg, brk, venue, question="NEW?"):
        m = market(price=0.50, bid=0.49, ask=0.51, venue=venue, question=question)
        ctx = context(cfg, brk)
        ctx.signals = [Signal(market=m, side=Side.BUY, fair_price=0.90, confidence=0.9)]
        asyncio.run(RiskAgent(cfg).run(ctx))
        return ctx

    def test_pumpfun_fifth_and_sixth_slot_are_usable(self):
        cfg = self._cfg(max_positions_per_venue_pumpfun=6)
        brk = broker()
        for i in range(5):
            self._open(brk, i, Venue.PUMPFUN)

        ctx = self._run(cfg, brk, Venue.PUMPFUN)
        self.assertEqual(len(ctx.orders), 1, [r["reason"] for r in ctx.rejections])

    def test_pumpfun_default_is_still_four(self):
        cfg = self._cfg()
        brk = broker()
        for i in range(4):
            self._open(brk, i, Venue.PUMPFUN)

        ctx = self._run(cfg, brk, Venue.PUMPFUN)
        self.assertEqual(len(ctx.orders), 0)
        self.assertIn("venue cap: 4 open on pumpfun", [r["reason"] for r in ctx.rejections])

    def test_polymarket_keeps_its_own_count(self):
        # The override is keyed by venue: giving pump.fun six must not give the
        # binary book six as well.
        cfg = self._cfg(max_positions_per_venue_pumpfun=6)
        brk = broker()
        for i in range(4):
            self._open(brk, i, Venue.POLYMARKET)

        ctx = self._run(cfg, brk, Venue.POLYMARKET)
        self.assertEqual(len(ctx.orders), 0)
        self.assertIn(
            "venue cap: 4 open on polymarket", [r["reason"] for r in ctx.rejections]
        )

    def test_raised_slot_count_still_respects_the_exposure_budget(self):
        # Slots are not a licence to over-concentrate: the venue budget is what
        # actually stops the sixth position when the money runs out.
        cfg = self._cfg(max_positions_per_venue_pumpfun=6, max_per_venue_pct=0.20)
        brk = broker()
        for i in range(5):
            self._open(brk, i, Venue.PUMPFUN)  # $50 each = $250 = 20%+ of equity

        ctx = self._run(cfg, brk, Venue.PUMPFUN)
        self.assertEqual(len(ctx.orders), 0)
        self.assertTrue(any("exposure" in r["reason"] for r in ctx.rejections),
                        [r["reason"] for r in ctx.rejections])


class TestShippedConfig(unittest.TestCase):
    """The three numbers that were the fix, asserted against the real file.

    They are config, not code, so nothing else would catch a silent revert —
    and all three failed together in a live run: six positions, none of them
    reaching the +35% trail arm, and 121 entries rejected at "4 open on pumpfun"
    while the venue's budget was one third unused.
    """

    def setUp(self):
        self.cfg = Config.load()

    def test_pumpfun_gets_more_slots_than_the_binary_book(self):
        base = self.cfg.i("risk.max_positions_per_venue", 4)
        pumpfun = self.cfg.i("risk.max_positions_per_venue_pumpfun", base)
        self.assertGreater(
            pumpfun, base, "pump.fun slots must exceed the shared default"
        )

    def test_global_counter_does_not_undercut_the_venue_slots(self):
        # If max_open_positions is smaller than the sum of the venue slots, the
        # global counter binds first and the extra slots are decorative.
        base = self.cfg.i("risk.max_positions_per_venue", 4)
        pumpfun = self.cfg.i("risk.max_positions_per_venue_pumpfun", base)
        crypto = self.cfg.i("risk.max_crypto_positions", 2)
        ceiling = self.cfg.i("risk.max_open_positions", 8)
        self.assertGreaterEqual(
            ceiling, pumpfun + base + crypto,
            "max_open_positions must fit pump.fun + Polymarket + crypto",
        )

    def test_trail_arm_is_reachable(self):
        arm = self.cfg.f("risk.trail_arm_pct", 0.35)
        self.assertLessEqual(
            arm, 0.25, "a +35% arm leaves +10..25% winners locked in a full book"
        )

    def test_a_stagnant_position_cannot_hold_a_slot_all_day(self):
        # live4: 13 hours of frozen equity, 4 positions inside ±3%, nothing new
        # could open because every slot was taken
        hours = self.cfg.f("risk.stagnant_exit_hours", 0.0)
        self.assertGreater(hours, 0.0, "without this rule a flat book never rotates")
        self.assertLess(hours, self.cfg.f("risk.max_hold_hours", 24.0))

    def test_the_discovery_cache_does_not_spend_the_monthly_quota(self):
        # 20s on a 35s tick is ~10k calls/month — GeckoTerminal's whole free tier
        self.assertGreaterEqual(self.cfg.i("scanner.pumpfun_ttl_seconds", 20), 60)


class TestStagnantExit(unittest.TestCase):
    """A slot held by a position that is not moving is a slot doing nothing."""

    def _run(self, cfg, avg, last, age_hours):
        brk = broker()
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=avg, last_price=last,
            high_water=max(avg, last),
            opened_at=utcnow() - timedelta(hours=age_hours),
        )
        brk.positions[pos.key] = pos
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=last, bid=last - 0.01, ask=last + 0.01)]
        asyncio.run(agent.run(ctx))
        return ctx

    def test_flat_position_is_cut_and_frees_the_slot(self):
        cfg = Config({"risk": {"stagnant_exit_hours": 3.0,
                               "stagnant_band_pct": 0.03, "max_hold_hours": 24.0}})
        ctx = self._run(cfg, avg=0.50, last=0.51, age_hours=5.0)
        self.assertEqual(len(ctx.orders), 1)
        self.assertIs(ctx.orders[0].side, Side.SELL)
        self.assertIn("stagnant", ctx.orders[0].reason)

    def test_a_position_that_is_working_is_left_alone(self):
        cfg = Config({"risk": {"stagnant_exit_hours": 3.0,
                               "stagnant_band_pct": 0.03, "max_hold_hours": 24.0}})
        ctx = self._run(cfg, avg=0.50, last=0.62, age_hours=5.0)
        self.assertEqual(ctx.orders, [])

    def test_the_clock_has_to_run_out_first(self):
        cfg = Config({"risk": {"stagnant_exit_hours": 3.0,
                               "stagnant_band_pct": 0.03, "max_hold_hours": 24.0}})
        ctx = self._run(cfg, avg=0.50, last=0.51, age_hours=1.0)
        self.assertEqual(ctx.orders, [])

    def test_a_position_that_once_ran_is_not_called_stagnant(self):
        # it had its move and gave it back — that is a trade, not a parked slot
        cfg = Config({"risk": {"stagnant_exit_hours": 3.0,
                               "stagnant_band_pct": 0.03, "max_hold_hours": 24.0,
                               "trail_arm_pct": 0.35}})
        brk = broker()
        pos = Position(
            key="polymarket:tok1", venue=Venue.POLYMARKET, token_id="tok1",
            question="Q?", qty=100, avg_price=0.50, last_price=0.51,
            high_water=0.60,
            opened_at=utcnow() - timedelta(hours=5),
        )
        brk.positions[pos.key] = pos
        agent = RiskAgent(cfg)
        ctx = context(cfg, brk)
        ctx.markets = [market(price=0.51, bid=0.50, ask=0.52)]
        asyncio.run(agent.run(ctx))
        self.assertFalse(any("stagnant" in o.reason for o in ctx.orders))


if __name__ == "__main__":
    unittest.main()
