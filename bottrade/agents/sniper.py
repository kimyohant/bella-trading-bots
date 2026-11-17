"""Sniper agent — the first minutes of a pump.fun launch.

Everything else in the pipeline assumes a market with a history: the scanner
ranks by 24-hour volume, the signal agent needs four momentum points before it
will say anything, and the risk profile is built for positions held for hours. A
token that is ninety seconds old has none of that, so it never reaches the book
— and by the time it does, the launch is over.

This agent reads the other end of the curve. It polls the network-wide
new-pools feed, keeps pump.fun launches younger than `max_age_minutes`, and asks
one question: is a crowd forming, or is one wallet talking to itself? In the
first minutes the only evidence available is breadth — distinct buyers, and the
balance of buys against sells — so that is what it scores.

It is not a promise of profit. Most launches go to zero, and the public feed
cannot see what would actually matter: dev wallet share, bundled buys, freeze
authority, whether the "buyers" are one cluster. What this buys is a small,
capped, time-boxed position in the rare launch that shows real demand, with an
exit clock measured in minutes rather than hours.
"""

from __future__ import annotations

from datetime import datetime

from ..models import SNIPER_PROFILE, Signal, Side, Venue
from ..utils import clamp, utcnow
from .base import Agent, Context, register


@register("sniper")
class SniperAgent(Agent):
    name = "sniper"

    async def run(self, ctx: Context) -> None:
        if not self.b("enabled", True):
            return

        adapter = ctx.adapters.get(Venue.PUMPFUN)
        new_pools = getattr(adapter, "new_pools", None)
        if new_pools is None:
            return

        max_age = self.f("max_age_minutes", 8.0)
        min_liquidity = self.f("min_liquidity_usd", 3_000.0)
        min_buyers = self.i("min_buyers_m5", 25)
        min_ratio = self.f("min_buy_sell_ratio", 1.1)
        min_change = self.f("min_change_m5", -15.0)
        max_change = self.f("max_change_m5", 60.0)
        min_fdv = self.f("min_fdv_usd", 8_000.0)
        max_fdv = self.f("max_fdv_usd", 250_000.0)
        max_candidates = self.i("max_candidates", 2)
        cooldown = self.f("cooldown_minutes", 60.0)
        min_edge = self.f("min_edge", 0.12)

        try:
            pools = await new_pools(limit=self.i("feed_limit", 20))
        except Exception:  # noqa: BLE001 - a dead feed must not stop the tick
            self.log.debug("sniper: new-pools feed unavailable", exc_info=True)
            return
        if not pools:
            return

        seen: dict[str, datetime] = ctx.state.setdefault("sniper_seen", {})
        if len(seen) > 500:  # keep the dedupe map from growing without bound
            cutoff = utcnow().timestamp() - 6 * 3600
            for key, when in list(seen.items()):
                if when.timestamp() < cutoff:
                    seen.pop(key, None)

        held = {(p.venue, p.question) for p in ctx.broker.positions.values()}
        existing = {m.key for m in ctx.markets}
        now = utcnow()
        picked: list[Signal] = []
        looked = 0

        for market in pools:
            age = float(market.meta.get("age_minutes") or 0.0)
            # Not a launch any more, already ours, or sniped recently: these are
            # not decisions, so they do not belong in the rejection log either.
            if age > max_age:
                continue
            if (market.venue, market.question) in held or market.key in existing:
                continue
            last = seen.get(market.key)
            if last and (now - last).total_seconds() / 60.0 < cooldown:
                continue

            # From here it is a genuine candidate, so a refusal is worth recording.
            looked += 1
            if market.liquidity_usd < min_liquidity:
                ctx.reject(
                    market,
                    f"sniper: liquidity ${market.liquidity_usd:,.0f} < "
                    f"${min_liquidity:,.0f}",
                    agent="sniper",
                )
                continue

            buys = float(market.meta.get("buys_m5") or 0.0)
            sells = float(market.meta.get("sells_m5") or 0.0)
            buyers = float(market.meta.get("buyers_m5") or 0.0)
            change = float(market.meta.get("change_m5") or 0.0)
            fdv = float(market.meta.get("market_cap_usd") or 0.0)
            ratio = buys / max(sells, 1.0)

            if buyers < min_buyers:
                ctx.reject(
                    market,
                    f"sniper: only {buyers:.0f} buyers in 5m < {min_buyers}",
                    agent="sniper",
                )
                continue
            if ratio < min_ratio:
                ctx.reject(
                    market, f"sniper: buys/sells {ratio:.2f} < {min_ratio:.2f}",
                    agent="sniper",
                )
                continue
            if change < min_change:
                ctx.reject(
                    market, f"sniper: already dumped {change:+.0f}% in 5m",
                    agent="sniper",
                )
                continue
            if change > max_change:
                # Chasing a vertical is how a sniper becomes the exit liquidity.
                ctx.reject(
                    market, f"sniper: already up {change:+.0f}% in 5m", agent="sniper"
                )
                continue
            if fdv and not (min_fdv <= fdv <= max_fdv):
                ctx.reject(
                    market, f"sniper: fdv ${fdv:,.0f} outside "
                    f"${min_fdv:,.0f}-${max_fdv:,.0f}",
                    agent="sniper",
                )
                continue

            # Breadth and pressure, nothing else: 25 buyers at parity is noise,
            # 100 buyers with four buys per sell is a crowd.
            breadth = clamp(buyers / (min_buyers * 4.0), 0.0, 1.0)
            pressure = clamp(ratio - 1.0, 0.0, 2.0)
            edge = clamp(0.04 + 0.08 * pressure + 0.08 * breadth, 0.0, 0.28)
            if edge < min_edge:
                ctx.reject(
                    market,
                    f"sniper: edge {edge * 100:.0f}% < {min_edge * 100:.0f}% "
                    f"({buyers:.0f} buyers, {ratio:.1f}x)",
                    agent="sniper",
                )
                continue

            fair = market.price * (1.0 + edge)
            confidence = clamp(0.55 + 0.25 * breadth + 0.10 * min(pressure, 1.0), 0.0, 0.95)
            picked.append(
                Signal(
                    market=market,
                    side=Side.BUY,
                    fair_price=round(fair, 14),
                    confidence=round(confidence, 4),
                    reasons=[
                        f"snipe {age:.1f}m old: {buyers:.0f} buyers, "
                        f"{ratio:.1f}x buys/sells, {change:+.0f}% in 5m",
                        f"fair {market.price * (1.0 + edge):.3e} vs ask "
                        f"{market.ask:.3e} (+{edge * 100:.0f}%)",
                        f"liquidity ${market.liquidity_usd:,.0f}, fdv ${fdv:,.0f}",
                    ],
                    ttl_seconds=self.i("signal_ttl_seconds", 120),
                )
            )

        picked.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        for signal in picked[:max_candidates]:
            signal.market.meta.setdefault("profile", SNIPER_PROFILE)
            # The market has to be in the universe or nothing downstream can
            # price it: risk reads the quote, portfolio marks it, and an
            # unmarked position can never be exited honestly.
            if signal.market.key not in existing:
                ctx.markets.append(signal.market)
                existing.add(signal.market.key)
            ctx.signals.append(signal)
            seen[signal.key] = now

        if looked or picked:
            self.log.debug(
                "sniper: %d pools, %d looked at, %d picked",
                len(pools), looked, min(len(picked), max_candidates),
            )
