"""Orchestrator — drives the agent pipeline, one tick at a time.

    feed.step() -> scanner -> sentiment -> whale -> signal -> risk -> executor -> portfolio

The orchestrator knows nothing about markets, venues or strategies. It only moves
a `Context` through the ordered agent list, isolates agent failures, and stops on
tick count, wall-clock duration, or end of data.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import Config
from ..data.store import Store
from ..feeds import Feed, build_feed
from ..models import Venue
from ..utils import fmt_pct, fmt_usd, utcnow
from ..venues.base import VenueAdapter
from ..venues.paper import PaperBroker
from .base import Agent, Context, build_agents

log = logging.getLogger("bottrade.orchestrator")


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        feed: Feed | None = None,
        agents: list[Agent] | None = None,
        store: Store | None = None,
        broker: PaperBroker | None = None,
    ):
        self.cfg = cfg
        self.feed = feed or build_feed(cfg)
        self.agents = agents if agents is not None else build_agents(cfg)
        self.broker = broker or PaperBroker.from_config(cfg)
        self.store = store

        adapters: dict[Venue, VenueAdapter] = {}
        adapters.update(getattr(self.feed, "adapters", {}) or {})
        adapters.update(getattr(self.feed, "venues", {}) or {})

        self.ctx = Context(
            cfg=cfg,
            broker=self.broker,
            log=log,
            adapters=adapters,
            store=store,
        )

    # -------------------------------------------------------------------- run
    async def run(
        self,
        max_ticks: int | None = None,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        self.ctx.duration_s = duration_s or 0.0
        stopped = "completed"

        try:
            while True:
                if max_ticks is not None and self.ctx.tick >= max_ticks:
                    stopped = "max_ticks"
                    break
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    stopped = "duration"
                    break

                markets = await self.feed.step()
                if markets is None:
                    stopped = "end_of_data"
                    break

                await self.tick(markets)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            stopped = "interrupted"
            log.warning("interrupted by user")
        finally:
            await self.feed.close()

        summary = self.summary(stopped, time.monotonic() - started)
        log.info(
            "done: %d ticks, equity %s (%s), %d fills, stopped=%s",
            summary["ticks"], fmt_usd(summary["equity"]),
            fmt_pct(summary["return_pct"]), summary["fills"], stopped,
        )
        return summary

    async def tick(self, markets: list[Any]) -> Context:
        ctx = self.ctx
        ctx.tick += 1
        ctx.now = utcnow()

        # per-tick scratch
        ctx.markets = markets
        ctx.quotes = {}
        ctx.signals = []
        ctx.orders = []
        ctx.fills = []
        ctx.rejections = []
        ctx.notes = []

        for agent in self.agents:
            agent_started = time.monotonic()
            try:
                await agent.run(ctx)
            except Exception:  # noqa: BLE001 - one agent must not kill the loop
                log.exception("agent %s failed, continuing", agent.name)
                if ctx.store:
                    ctx.store.record_event("ERROR", agent.name, "agent raised")
            elapsed = (time.monotonic() - agent_started) * 1000
            if elapsed > 2_000:
                log.warning("agent %s took %.0fms", agent.name, elapsed)

        if ctx.store and ctx.rejections:
            ctx.store.record_rejections(ctx.rejections, ctx.tick)

        return ctx

    # ---------------------------------------------------------------- summary
    def summary(self, stopped: str = "completed", elapsed_s: float = 0.0) -> dict[str, Any]:
        broker = self.broker
        equity = broker.equity()
        starting = broker.starting_cash or 1.0
        exits = [f for f in broker.fills if f.tag == "exit"]
        wins = [f for f in exits if f.realized_pnl > 0]
        losses = [f for f in exits if f.realized_pnl < 0]

        return {
            "stopped": stopped,
            "ticks": self.ctx.tick,
            "elapsed_s": round(elapsed_s, 2),
            "starting_cash": round(starting, 2),
            "equity": round(equity, 2),
            "cash": round(broker.cash, 2),
            "exposure": round(broker.gross_exposure(), 2),
            "pnl": round(equity - starting, 2),
            "return_pct": round(equity / starting - 1.0, 6),
            "realized_pnl": round(broker.realized_pnl, 2),
            "unrealized_pnl": round(broker.unrealized_pnl(), 2),
            "fees_paid": round(broker.fees_paid, 2),
            "fills": len(broker.fills),
            "entries": len(broker.fills) - len(exits),
            "exits": len(exits),
            "wins": len(wins),
            "losses": len(losses),
            "open_positions": len(broker.positions),
            "universe": len(self.ctx.markets),
            "notes": list(self.ctx.notes),
        }
