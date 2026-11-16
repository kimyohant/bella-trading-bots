"""Scanner agent — the tradable universe for this tick.

Takes whatever the feed produced, drops anything untradable (thin book, silly
price, too wide a spread, resolving in five minutes), ranks what is left and
publishes the quotes every later agent works from. It also refreshes marks so
risk and equity always see the latest price.
"""

from __future__ import annotations

import math
from typing import Any

from ..models import Market, Venue
from .base import Agent, Context, register


@register("scanner")
class ScannerAgent(Agent):
    name = "scanner"

    async def run(self, ctx: Context) -> None:
        raw = ctx.markets
        if not raw:
            ctx.note("scanner: feed returned no markets")
            return

        min_price = self.f("min_price", 0.03)
        max_price = self.f("max_price", 0.97)
        max_spread = self.f("max_spread_pct", 0.08)
        min_liquidity = self.f("min_liquidity_usd", 3_000.0)
        min_hours_left = self.f("min_hours_to_resolution", 1.0)
        max_candidates = self.i("max_candidates", 24)

        kept: list[Market] = []
        dropped = {"price": 0, "spread": 0, "liquidity": 0, "resolution": 0}

        for market in raw:
            if market.venue is Venue.POLYMARKET:
                if not (min_price <= market.price <= max_price):
                    dropped["price"] += 1
                    continue
                hours = market.hours_to_resolution
                if hours is not None and hours < min_hours_left:
                    dropped["resolution"] += 1
                    continue
            if market.spread_pct > max_spread:
                dropped["spread"] += 1
                continue
            if market.liquidity_usd < min_liquidity:
                dropped["liquidity"] += 1
                continue
            kept.append(market)

        kept.sort(key=_rank, reverse=True)

        # Per-venue quota: Polymarket has ~40 tradable legs per scan and would
        # otherwise crowd every pump.fun token out of the candidate list.
        max_per_venue = self.i("max_per_venue", 12)
        by_venue: dict[Any, list[Market]] = {}
        for market in kept:
            by_venue.setdefault(market.venue, []).append(market)
        quota: list[Market] = []
        for items in by_venue.values():
            quota.extend(items[:max_per_venue])
        quota.sort(key=_rank, reverse=True)

        if len(quota) > max_candidates:
            quota = quota[:max_candidates]

        ctx.markets = quota
        ctx.quotes = {m.key: m.quote() for m in quota}
        ctx.state["universe_size"] = len(quota)
        ctx.state["dropped"] = dropped

        ctx.broker.mark(ctx.price_map())
        if ctx.store:
            ctx.store.record_markets(quota)

        self.log.debug(
            "universe=%d dropped=%s", len(quota),
            {k: v for k, v in dropped.items() if v},
        )


def _rank(market: Market) -> float:
    """Liquidity-weighted, spread-penalised score used only for ordering."""
    depth = math.log1p(max(market.volume_24h_usd, 1.0)) + 0.5 * math.log1p(
        max(market.liquidity_usd, 1.0)
    )
    penalty = market.spread_pct * 12.0
    if market.venue is Venue.PUMPFUN:
        depth += 0.4  # small universe, keep memecoins in view
    return depth - penalty
