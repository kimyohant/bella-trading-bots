"""Executor agent — the only component that touches the broker.

Takes sized orders from risk, asks the broker for a fill, and records everything
that happened. In paper mode the broker simulates; the same code path can be
pointed at a live adapter later without any agent changing.
"""

from __future__ import annotations

from ..models import Side
from ..utils import fmt_price, utcnow
from ..venues.paper import OrderRejected
from .base import Agent, Context, register


@register("executor")
class ExecutorAgent(Agent):
    name = "executor"

    async def run(self, ctx: Context) -> None:
        if not ctx.orders:
            return

        for order in ctx.orders:
            quote = ctx.quote_for(order.market)
            try:
                fill = ctx.broker.execute(order, quote)
            except OrderRejected as exc:
                ctx.reject(order.market, str(exc), agent="executor")
                if ctx.store:
                    ctx.store.record_order(order, "rejected", str(exc))
                self.log.warning("rejected %s %s: %s", order.side, order.key, exc)
                continue

            ctx.fills.append(fill)
            if ctx.store:
                ctx.store.record_order(order, "filled")
                ctx.store.record_fill(fill)

            if order.side is Side.BUY:
                ctx.state["trades_today"] = ctx.state.get("trades_today", 0) + 1
                ctx.state.setdefault("last_trade_ts", {})[order.key] = utcnow()
                # remember the thesis at entry so the risk agent can detect decay
                position = ctx.broker.positions.get(order.key)
                if position is not None:
                    position.entry_thesis = order.thesis
            else:
                # a sell that leaves a remainder means we took partial profit;
                # the risk agent must not fire that same partial rule twice
                remaining = ctx.broker.positions.get(order.key)
                if remaining is not None:
                    remaining.partial_taken = True

            direction = "BUY " if order.side is Side.BUY else "SELL"
            pnl = f" pnl={fill.realized_pnl:+.2f}" if fill.realized_pnl else ""
            self.log.info(
                "%s %s qty=%s @ %s fee=%.2f%s | %s",
                direction, order.market.question[:44],
                f"{fill.qty:,.2f}", fmt_price(fill.price),
                fill.fee, pnl, order.reason[:70],
            )
