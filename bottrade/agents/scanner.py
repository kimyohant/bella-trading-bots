"""Scanner agent — the tradable universe for this tick.

Takes whatever the feed produced, drops anything untradable (thin book, silly
price, too wide a spread, resolving in five minutes), ranks what is left and
publishes the quotes every later agent works from. It also refreshes marks so
risk and equity always see the latest price.
"""

from __future__ import annotations

import math
from typing import Any

from ..models import CRYPTO_PROFILE, Market, Venue
from ..utils import utcnow
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
        crypto: list[Market] = []
        dropped = {"price": 0, "spread": 0, "liquidity": 0, "resolution": 0}

        crypto_max_spread = self.f("crypto_max_spread_pct", 0.08)
        crypto_min_liq = self.f("crypto_min_liquidity_usd", 400.0)

        for market in raw:
            if market.meta.get("profile") == CRYPTO_PROFILE:
                # A 15-minute market resolves in minutes and trades on ~$1k of
                # liquidity by design. The political-market filters would drop
                # every single one of them, so judge them on their own terms.
                if market.spread_pct > crypto_max_spread:
                    dropped["spread"] += 1
                    continue
                if market.liquidity_usd < crypto_min_liq:
                    dropped["liquidity"] += 1
                    continue
                if not (min_price <= market.price <= max_price):
                    dropped["price"] += 1
                    continue
                crypto.append(market)
                continue

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

        # Crypto keeps its own book. Ranked by volume it would never survive a
        # combined cut against the political markets, and it must not crowd them
        # out either — so it gets a separate, small slot on top.
        crypto.sort(key=_crypto_rank, reverse=True)
        quota.extend(crypto[: self.i("crypto_max_candidates", 6)])

        ctx.markets = quota
        ctx.quotes = {m.key: m.quote() for m in quota}
        ctx.state["universe_size"] = len(quota)
        ctx.state["dropped"] = dropped

        await self._mark_positions(ctx)

        if ctx.store:
            ctx.store.record_markets(quota)

        self.log.debug(
            "universe=%d dropped=%s", len(quota),
            {k: v for k, v in dropped.items() if v},
        )

    async def _mark_positions(self, ctx: Context) -> None:
        """Mark every open position, including ones outside the scan quota.

        `PaperBroker.mark` only touches keys it is given, so a position that
        drops out of the universe used to keep its last price forever: P&L froze,
        every exit rule read that frozen number, and the dashboard sat on a mark
        nobody could trade. Ask the venue for the strays — one call per venue.
        """
        prices = ctx.price_map()
        missing: dict[Venue, list[str]] = {}
        for key in ctx.broker.positions:
            if key in prices:
                continue
            try:
                venue = Venue(key.split(":", 1)[0])
            except ValueError:
                continue
            missing.setdefault(venue, []).append(key)

        for venue, keys in missing.items():
            adapter = ctx.adapters.get(venue)
            if adapter is None:
                continue
            try:
                prices.update(await adapter.quote_missing(keys))
            except Exception:  # noqa: BLE001 - a stale mark must not kill the tick
                self.log.debug("quote_missing failed for %s", venue.value)

        ctx.broker.mark(prices)


def _rank(market: Market) -> float:
    """Liquidity-weighted, spread-penalised score used only for ordering."""
    depth = math.log1p(max(market.volume_24h_usd, 1.0)) + 0.5 * math.log1p(
        max(market.liquidity_usd, 1.0)
    )
    penalty = market.spread_pct * 12.0
    if market.venue is Venue.PUMPFUN:
        depth += 0.4  # small universe, keep memecoins in view
    return depth - penalty


def _crypto_rank(market: Market) -> float:
    """Live window first, then depth, then tightness.

    The next window is listed alongside the current one so the bot can see it
    coming, but it has no opening price yet and must not take a trading slot.
    """
    start = market.meta.get("window_start") or 0
    live = 1.0 if start <= utcnow().timestamp() else 0.0
    return live * 1e6 + market.liquidity_usd - market.spread_pct * 1_000.0
