"""Guard agent — data health, freshness, and the dead-man's switch.

Runs first, because every agent downstream assumes it is looking at live prices.
When a venue stops answering, `LiveFeed` swallows the exception and simply
contributes nothing, so the pipeline keeps evaluating positions against a frozen
`last_price`. That is how a bot sells at a price that has not existed for twenty
minutes and books a fill it could never have got.

This is the one agent allowed to say "I don't know". It:

* keeps a miss streak per venue and publishes `state["stale_venues"]`
* publishes `state["stale_forced"]` once a venue has been quiet too long
* trips `state["deadman"]` when the whole feed goes empty, which halts entries
* writes one `health` row per venue per tick so the monitor can show it

It never places an order. Risk owns the decision; guard owns the facts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..models import Venue
from ..utils import iso
from .base import Agent, Context, register


@register("guard")
class GuardAgent(Agent):
    name = "guard"

    async def run(self, ctx: Context) -> None:
        max_age = max(1, self.i("max_quote_age_ticks", 3))
        max_empty = max(1, self.i("max_empty_ticks", 5))
        max_stale_hours = self.f("max_stale_hours", 6.0)

        health: dict[str, dict[str, Any]] = ctx.state.setdefault("health", {})
        seen = {m.venue for m in ctx.markets}

        stale: set[Venue] = set()
        forced: set[Venue] = set()
        stale_hours: dict[str, float] = {}
        rows: list[dict[str, Any]] = []

        for venue in self._tracked(ctx):
            entry = health.setdefault(venue.value, _blank(ctx.tick))

            if venue in seen:
                self._recover(ctx, venue, entry)
            else:
                entry["miss"] += 1
                if entry["miss"] >= max_age:
                    self._go_stale(ctx, venue, entry)
                    stale.add(venue)
                    hours = _hours_since(entry.get("stale_since"), ctx.now)
                    stale_hours[venue.value] = hours
                    if hours >= max_stale_hours:
                        forced.add(venue)
                        self._force_once(ctx, venue, entry, hours)

            rows.append(
                {
                    "venue": venue.value,
                    "ok": 1 if entry["ok"] else 0,
                    "miss": entry["miss"],
                    "stale_hours": round(stale_hours.get(venue.value, 0.0), 3),
                    "note": "" if entry["ok"] else f"quiet {entry['miss']}t",
                }
            )

        ctx.state["stale_venues"] = stale
        ctx.state["stale_forced"] = forced
        ctx.state["stale_hours"] = stale_hours
        ctx.state["health_ok"] = bool(seen)

        self._deadman(ctx, seen, max_empty)

        if ctx.store:
            ctx.store.record_health(rows, ctx.tick)

    # ------------------------------------------------------------- transitions
    def _recover(self, ctx: Context, venue: Venue, entry: dict[str, Any]) -> None:
        if not entry["ok"]:
            self._event(ctx, "OK", f"{venue.value} data restored after {entry['miss']} quiet ticks")
            ctx.note(f"guard: {venue.value} data restored")
            self.log.info("%s data restored", venue.value)
        entry["ok"] = True
        entry["miss"] = 0
        entry["last_ok_tick"] = ctx.tick
        entry["last_ok_ts"] = iso(ctx.now)
        entry["stale_since"] = None
        entry["forced_logged"] = False

    def _go_stale(self, ctx: Context, venue: Venue, entry: dict[str, Any]) -> None:
        if not entry["ok"]:
            return
        entry["ok"] = False
        entry["stale_since"] = ctx.now
        message = f"{venue.value} stale: no quotes for {entry['miss']} ticks"
        self._event(ctx, "WARN", message)
        ctx.note(f"guard: {message}")
        # one rejection on the transition, so the dashboard shows it without
        # flooding the rejection table every tick the venue stays down
        ctx.reject(None, f"stale venue: {venue.value}", agent="guard")
        self.log.warning(message)

    def _force_once(self, ctx: Context, venue: Venue, entry: dict[str, Any],
                    hours: float) -> None:
        if entry.get("forced_logged"):
            return
        entry["forced_logged"] = True
        message = (
            f"{venue.value} stale for {hours:.1f}h — positions on it will be "
            f"closed on the last known mark"
        )
        self._event(ctx, "ERROR", message)
        ctx.note(f"guard: {message}")
        self.log.error(message)

    def _deadman(self, ctx: Context, seen: set[Venue], max_empty: int) -> None:
        if seen:
            ctx.state["empty_streak"] = 0
            if ctx.state.get("deadman"):
                ctx.state["deadman"] = False
                self._event(ctx, "OK", "dead-man switch cleared, data flowing again")
                ctx.note("guard: dead-man switch cleared")
                self.log.info("dead-man switch cleared")
            return

        streak = int(ctx.state.get("empty_streak", 0)) + 1
        ctx.state["empty_streak"] = streak
        if streak >= max_empty and not ctx.state.get("deadman"):
            ctx.state["deadman"] = True
            message = f"dead-man switch: feed empty for {streak} ticks, entries halted"
            self._event(ctx, "ERROR", message)
            ctx.note(f"guard: DEAD-MAN SWITCH, entries halted after {streak} empty ticks")
            self.log.error(message)

    # ------------------------------------------------------------------ helpers
    def _tracked(self, ctx: Context) -> list[Venue]:
        """Venues that are supposed to be reporting, in a stable order."""
        venues: list[Venue] = []

        def add(venue: Venue) -> None:
            if venue not in venues:
                venues.append(venue)

        for raw in self.cfg.list("run.venues", ["polymarket", "pumpfun"]):
            try:
                add(Venue(str(raw).strip().lower()))
            except ValueError:
                continue
        for name in ctx.state.get("health", {}):
            try:
                add(Venue(str(name)))
            except ValueError:
                continue
        for position in ctx.broker.positions.values():
            add(position.venue)
        return venues

    def _event(self, ctx: Context, level: str, message: str) -> None:
        if ctx.store:
            ctx.store.record_event(level, self.name, message)


def _blank(tick: int) -> dict[str, Any]:
    return {
        "ok": True,
        "miss": 0,
        "last_ok_tick": tick,
        "last_ok_ts": None,
        "stale_since": None,
        "forced_logged": False,
    }


def _hours_since(since: datetime | None, now: datetime) -> float:
    if since is None:
        return 0.0
    return max(0.0, (now - since).total_seconds() / 3600.0)
