"""Paper broker mechanics: fills, slippage, fees, partial fills, rejections."""

from __future__ import annotations

import unittest

from bottrade.config import Config
from bottrade.models import Market, Order, Side, Venue
from bottrade.venues.paper import OrderRejected, PaperBroker


def make_market(price=0.50, bid=0.49, ask=0.51, venue=Venue.POLYMARKET) -> Market:
    return Market(
        venue=venue,
        market_id="m1",
        question="Will the thing happen?",
        token_id="tok1",
        price=price,
        bid=bid,
        ask=ask,
        liquidity_usd=50_000,
    )


def make_broker(cash=100.0, fee_bps=0.0, slip_bps=0.0, fixed=0.0) -> PaperBroker:
    return PaperBroker(
        cash=cash,
        starting_cash=cash,
        fee_bps={Venue.POLYMARKET: fee_bps, Venue.PUMPFUN: fee_bps},
        slippage_bps={Venue.POLYMARKET: slip_bps, Venue.PUMPFUN: slip_bps},
        fixed_fee_usd={Venue.POLYMARKET: fixed, Venue.PUMPFUN: fixed},
    )


class TestBuySell(unittest.TestCase):
    def test_buy_pays_the_ask(self):
        broker = make_broker(cash=100)
        market = make_market()
        fill = broker.execute(
            Order(market=market, side=Side.BUY, qty=100, limit_price=0.51), market.quote()
        )
        self.assertAlmostEqual(fill.price, 0.51, places=6)
        self.assertAlmostEqual(broker.cash, 100 - 51, places=6)
        position = broker.positions["polymarket:tok1"]
        self.assertAlmostEqual(position.qty, 100, places=6)
        self.assertAlmostEqual(position.avg_price, 0.51, places=6)

    def test_round_trip_realizes_pnl(self):
        broker = make_broker(cash=100)
        market = make_market()
        broker.execute(Order(market=market, side=Side.BUY, qty=100, limit_price=0.51), market.quote())

        # price moves up: bid is now 0.60
        up = make_market(price=0.61, bid=0.60, ask=0.62)
        fill = broker.execute(
            Order(market=up, side=Side.SELL, qty=100, limit_price=0.60), up.quote()
        )
        self.assertAlmostEqual(fill.realized_pnl, (0.60 - 0.51) * 100, places=6)
        self.assertAlmostEqual(broker.realized_pnl, 9.0, places=6)
        self.assertNotIn("polymarket:tok1", broker.positions)
        self.assertAlmostEqual(broker.cash, 100 + 9.0, places=6)

    def test_fees_are_charged_both_ways(self):
        broker = make_broker(cash=100, fee_bps=100)  # 1%
        market = make_market()
        fill = broker.execute(
            Order(market=market, side=Side.BUY, qty=10, limit_price=0.51), market.quote()
        )
        self.assertAlmostEqual(fill.fee, 10 * 0.51 * 0.01, places=6)
        self.assertAlmostEqual(broker.fees_paid, fill.fee, places=6)

    def test_slippage_moves_price_against_you(self):
        broker = make_broker(cash=100, slip_bps=100)  # 1%
        market = make_market()
        fill = broker.execute(
            Order(market=market, side=Side.BUY, qty=10, limit_price=0.51), market.quote()
        )
        self.assertAlmostEqual(fill.price, 0.51 * 1.01, places=6)
        self.assertGreater(fill.slippage_bps, 0)

    def test_partial_fill_when_cash_is_short(self):
        broker = make_broker(cash=10)
        market = make_market()
        fill = broker.execute(
            Order(market=market, side=Side.BUY, qty=100, limit_price=0.51), market.quote()
        )
        self.assertLess(fill.qty, 100)
        self.assertGreaterEqual(broker.cash, 0)
        self.assertIn("partial", fill.reason)

    def test_sell_without_position_is_rejected(self):
        broker = make_broker()
        market = make_market()
        with self.assertRaises(OrderRejected):
            broker.execute(Order(market=market, side=Side.SELL, qty=1, limit_price=0.49), market.quote())

    def test_sell_cannot_exceed_position(self):
        broker = make_broker(cash=100)
        market = make_market()
        broker.execute(Order(market=market, side=Side.BUY, qty=10, limit_price=0.51), market.quote())
        fill = broker.execute(
            Order(market=market, side=Side.SELL, qty=999, limit_price=0.49), market.quote()
        )
        self.assertAlmostEqual(fill.qty, 10, places=6)

    def test_binary_price_is_bounded(self):
        broker = make_broker(cash=100, slip_bps=5_000)  # absurd slippage
        market = make_market(price=0.98, bid=0.97, ask=0.99)
        fill = broker.execute(
            Order(market=market, side=Side.BUY, qty=1, limit_price=0.99), market.quote()
        )
        self.assertLessEqual(fill.price, 0.995)


class TestEquity(unittest.TestCase):
    def test_equity_includes_marked_positions(self):
        broker = make_broker(cash=100)
        market = make_market()
        broker.execute(Order(market=market, side=Side.BUY, qty=100, limit_price=0.51), market.quote())
        self.assertAlmostEqual(broker.equity(), 100.0, places=6)

        broker.mark({"polymarket:tok1": 0.60})
        self.assertAlmostEqual(broker.equity(), 100 + 9.0, places=6)
        self.assertAlmostEqual(broker.unrealized_pnl(), 9.0, places=6)

    def test_exposure_by_venue(self):
        broker = make_broker(cash=100)
        market = make_market()
        broker.execute(Order(market=market, side=Side.BUY, qty=10, limit_price=0.51), market.quote())
        exposure = broker.exposure_by_venue()
        self.assertAlmostEqual(exposure[Venue.POLYMARKET], 10 * 0.51, places=6)


if __name__ == "__main__":
    unittest.main()
