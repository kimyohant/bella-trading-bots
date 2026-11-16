"""Portfolio agent — marks to market and writes the audit trail.

Runs last in the pipeline: re-prices every open position with this tick's quotes,
then persists equity, positions and a one-line summary. This is what the report
generator reads, so nothing about a run is lost when the process exits.
"""

from __future__ import annotations

from ..utils import fmt_pct, fmt_usd
from .base import Agent, Context, register


@register("portfolio")
class PortfolioAgent(Agent):
    name = "portfolio"

    async def run(self, ctx: Context) -> None:
        prices = ctx.price_map()
        # positions outside the current universe keep their last known mark
        for key, position in ctx.broker.positions.items():
            if key not in prices and position.last_price > 0:
                prices[key] = position.last_price

        ctx.broker.mark(prices)

        equity = ctx.broker.equity()
        cash = ctx.broker.cash
        exposure = ctx.broker.gross_exposure()
        realized = ctx.broker.realized_pnl
        unrealized = ctx.broker.unrealized_pnl()

        curve: list[float] = ctx.state.setdefault("equity_curve", [])
        curve.append(round(equity, 4))
        starting = ctx.broker.starting_cash or 1.0
        total_return = equity / starting - 1.0

        if ctx.store:
            ctx.store.record_equity(equity, cash, exposure, realized, unrealized)
            ctx.store.record_positions(ctx.broker.positions.values(), ctx.tick)
            ctx.store.record_tick(
                tick_no=ctx.tick,
                equity=equity,
                cash=cash,
                exposure=exposure,
                realized=realized,
                unrealized=unrealized,
                open_positions=len(ctx.broker.positions),
                n_markets=len(ctx.markets),
                n_signals=len(ctx.signals),
                n_orders=len(ctx.orders),
                n_fills=len(ctx.fills),
            )

        self.log.info(
            "tick=%d equity=%s (%s) cash=%s exp=%s pos=%d fills=%d",
            ctx.tick,
            fmt_usd(equity),
            fmt_pct(total_return),
            fmt_usd(cash),
            fmt_usd(exposure),
            len(ctx.broker.positions),
            len(ctx.fills),
        )
