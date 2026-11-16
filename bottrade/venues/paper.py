"""Paper broker: realistic-ish simulated fills.

Models the things that actually cost money in these two venues:
  * paying the ask / hitting the bid rather than the mid;
  * slippage in basis points (pump.fun curves move against you);
  * explicit fees (Polymarket currently 0 bps on most markets, pump.fun ~100 bps)
    plus the fixed Solana network fee;
  * partial fills when cash runs short.

No real orders are ever placed. `PaperBroker` is the only execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..models import Fill, Order, Position, Quote, Side, Venue
from ..utils import clamp, utcnow

BINARY_FLOOR = 0.005
BINARY_CEIL = 0.995


class OrderRejected(RuntimeError):
    """Raised when the broker refuses an order; the message is the reason."""


@dataclass
class PaperBroker:
    cash: float
    fee_bps: dict[Venue, float] = field(default_factory=dict)
    slippage_bps: dict[Venue, float] = field(default_factory=dict)
    fixed_fee_usd: dict[Venue, float] = field(default_factory=dict)
    max_slippage_bps: float = 500.0
    min_notional_usd: float = 1.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    starting_cash: float = 0.0

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, cfg: Config) -> "PaperBroker":
        cash = cfg.f("portfolio.starting_cash", 1_000.0)
        return cls(
            cash=cash,
            starting_cash=cash,
            fee_bps={
                Venue.POLYMARKET: cfg.f("fees.polymarket_fee_bps", 0.0),
                Venue.PUMPFUN: cfg.f("fees.pumpfun_fee_bps", 100.0),
            },
            slippage_bps={
                Venue.POLYMARKET: cfg.f("slippage.polymarket_bps", 30.0),
                Venue.PUMPFUN: cfg.f("slippage.pumpfun_bps", 250.0),
            },
            fixed_fee_usd={
                Venue.POLYMARKET: cfg.f("fees.polymarket_fixed_usd", 0.0),
                Venue.PUMPFUN: cfg.f("fees.solana_fee_usd", 0.02),
            },
            max_slippage_bps=cfg.f("slippage.max_bps", 600.0),
            min_notional_usd=cfg.f("risk.min_notional_usd", 1.0),
        )

    # -------------------------------------------------------------------- state
    def equity(self) -> float:
        return self.cash + self.gross_exposure()

    def gross_exposure(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())

    def exposure_by_venue(self) -> dict[Venue, float]:
        out: dict[Venue, float] = {venue: 0.0 for venue in Venue}
        for pos in self.positions.values():
            out[pos.venue] = out.get(pos.venue, 0.0) + pos.market_value
        return out

    def unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def mark(self, prices: dict[str, float]) -> None:
        for key, position in self.positions.items():
            price = prices.get(key)
            if price and price > 0:
                position.last_price = price
                position.high_water = max(position.high_water, price)
                position.marked_at = utcnow()

    # ---------------------------------------------------------------- execution
    def fill_price(self, order: Order, quote: Quote) -> tuple[float, float]:
        """Return (fill_price, realised_slippage_bps)."""
        slip = self.slippage_bps.get(order.market.venue, 0.0) / 10_000.0
        if order.side is Side.BUY:
            reference = quote.ask or quote.mid or order.limit_price
            price = reference * (1.0 + slip)
            if order.market.venue is Venue.POLYMARKET:
                price = clamp(price, BINARY_FLOOR, BINARY_CEIL)
            realised = (price / reference - 1.0) * 10_000 if reference else 0.0
        else:
            reference = quote.bid or quote.mid or order.limit_price
            price = reference * (1.0 - slip)
            if order.market.venue is Venue.POLYMARKET:
                price = clamp(price, BINARY_FLOOR, BINARY_CEIL)
            realised = (1.0 - price / reference) * 10_000 if reference else 0.0
        return price, max(0.0, realised)

    def execute(self, order: Order, quote: Quote) -> Fill:
        if order.qty <= 0:
            raise OrderRejected("qty <= 0")
        price, slippage = self.fill_price(order, quote)
        if price <= 0:
            raise OrderRejected("no valid price")
        if slippage > self.max_slippage_bps:
            raise OrderRejected(f"slippage {slippage:.0f}bps > limit")

        if order.side is Side.BUY:
            return self._buy(order, price, slippage)
        return self._sell(order, price, slippage)

    def _buy(self, order: Order, price: float, slippage: float) -> Fill:
        venue = order.market.venue
        fee_rate = self.fee_bps.get(venue, 0.0) / 10_000.0
        fixed = self.fixed_fee_usd.get(venue, 0.0)

        qty = order.qty
        cost = qty * price * (1.0 + fee_rate) + fixed
        if cost > self.cash:
            # scale down to what the cash actually supports (partial fill)
            usable = max(0.0, self.cash - fixed)
            qty = usable / (price * (1.0 + fee_rate)) if price > 0 else 0.0
            if qty * price < self.min_notional_usd:
                raise OrderRejected(
                    f"insufficient cash (need {cost:.2f}, have {self.cash:.2f})"
                )
            order.reason = f"{order.reason} [partial]".strip()

        notional = qty * price
        fee = notional * fee_rate + fixed
        self.cash -= notional + fee
        self.fees_paid += fee

        position = self.positions.get(order.key)
        if position is None:
            position = Position(
                key=order.key,
                venue=venue,
                token_id=order.market.token_id,
                question=order.market.question,
                qty=0.0,
                avg_price=0.0,
                last_price=price,
                high_water=price,
                entry_edge=order.edge,
            )
            self.positions[order.key] = position
        total_cost = position.avg_price * position.qty + notional
        position.qty += qty
        position.avg_price = total_cost / position.qty if position.qty else 0.0
        position.last_price = price
        position.fees_paid += fee

        fill = Fill(
            key=order.key,
            venue=venue,
            question=order.market.question,
            side=Side.BUY,
            qty=qty,
            price=price,
            fee=fee,
            slippage_bps=slippage,
            reason=order.reason,
            tag=order.tag,
        )
        self.fills.append(fill)
        return fill

    def _sell(self, order: Order, price: float, slippage: float) -> Fill:
        position = self.positions.get(order.key)
        if position is None or not position.is_open:
            raise OrderRejected("no open position to sell")

        venue = order.market.venue
        fee_rate = self.fee_bps.get(venue, 0.0) / 10_000.0
        fixed = self.fixed_fee_usd.get(venue, 0.0)

        qty = min(order.qty, position.qty)
        notional = qty * price
        fee = notional * fee_rate + fixed
        realized = (price - position.avg_price) * qty - fee

        self.cash += notional - fee
        self.fees_paid += fee
        self.realized_pnl += realized
        position.realized_pnl += realized
        position.fees_paid += fee
        position.qty -= qty
        position.last_price = price
        if position.qty <= 1e-9:
            self.positions.pop(order.key, None)

        fill = Fill(
            key=order.key,
            venue=venue,
            question=order.market.question,
            side=Side.SELL,
            qty=qty,
            price=price,
            fee=fee,
            slippage_bps=slippage,
            reason=order.reason,
            tag=order.tag,
            realized_pnl=realized,
        )
        self.fills.append(fill)
        return fill

    # ------------------------------------------------------------------ helpers
    def close_position(self, key: str, quote: Quote, reason: str = "manual") -> Fill:
        position = self.positions[key]
        order = Order(
            market=_position_market(position),
            side=Side.SELL,
            qty=position.qty,
            limit_price=quote.bid or position.last_price,
            reason=reason,
            tag="exit",
        )
        return self.execute(order, quote)

    def snapshot(self) -> dict[str, Any]:
        return {
            "ts": utcnow().isoformat(timespec="seconds"),
            "cash": round(self.cash, 4),
            "equity": round(self.equity(), 4),
            "exposure": round(self.gross_exposure(), 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl(), 4),
            "fees_paid": round(self.fees_paid, 4),
            "open_positions": len(self.positions),
            "n_fills": len(self.fills),
        }


def _position_market(position: Position):
    """Rebuild a minimal Market for closing a position (used by `close_position`)."""
    from ..models import Market

    return Market(
        venue=position.venue,
        market_id=position.token_id,
        question=position.question,
        token_id=position.token_id,
        price=position.last_price,
        bid=position.last_price,
        ask=position.last_price,
    )
