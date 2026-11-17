"""Risk agent — the only thing standing between a signal and your capital.

Two jobs, in this order:

1. **Exits first.** Stops, targets, time stops, thesis flips and near-resolution
   de-risking are always evaluated, even when entries are halted.
2. **Entries second.** Every signal passes a hard gate list, then gets a position
   size from fractional Kelly (binary markets) or fixed-risk sizing (tokens),
   then is clipped by portfolio-level exposure budgets.

Anything rejected is recorded with a reason, which is what makes the bot auditable.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..models import (
    CRYPTO_PROFILE,
    SNIPER_PROFILE,
    Market,
    Order,
    Position,
    Side,
    Signal,
    Venue,
)
from ..utils import clamp, utcnow
from .base import Agent, Context, register


@register("risk")
class RiskAgent(Agent):
    name = "risk"

    async def run(self, ctx: Context) -> None:
        state = ctx.state
        self._roll_day(ctx, state)

        equity = ctx.broker.equity()
        state.setdefault("day_start_equity", equity)
        state.setdefault("last_trade_ts", {})

        day_pnl = equity - state["day_start_equity"]
        day_pnl_pct = day_pnl / state["day_start_equity"] if state["day_start_equity"] else 0.0

        limit = self.f("daily_loss_limit_pct", 0.08)
        if day_pnl_pct <= -limit and not state.get("halted"):
            state["halted"] = True
            ctx.note(f"risk: HALT new entries, daily loss {day_pnl_pct * 100:.1f}%")
            self.log.warning("daily loss limit hit (%.1f%%), halting new entries", day_pnl_pct * 100)

        exits = self._build_exits(ctx)
        entries = self._build_entries(ctx, equity, day_pnl_pct)
        ctx.orders = exits + entries

        if exits or entries:
            self.log.info("orders: %d exits, %d entries", len(exits), len(entries))

    # ------------------------------------------------------------------ day
    def _roll_day(self, ctx: Context, state: dict[str, Any]) -> None:
        today = ctx.now.date()
        if state.get("day") != today:
            state["day"] = today
            state["day_start_equity"] = ctx.broker.equity()
            state["trades_today"] = 0
            state["halted"] = False
            state["realized_today"] = 0.0

    # ---------------------------------------------------------------- exits
    def _build_exits(self, ctx: Context) -> list[Order]:
        near_resolution = self.f("exit_hours_before_resolution", 2.0)
        flip_delta = self.f("thesis_flip_delta", 0.25)
        min_hold = self.f("min_hold_hours", 0.25)

        # Guard publishes which venues have stopped reporting. A fill priced off
        # a frozen mark is a fiction, so by default we hold rather than invent it
        # — but a position stranded on a dead venue must not be held forever.
        stale: set[Any] = ctx.state.get("stale_venues") or set()
        forced: set[Any] = ctx.state.get("stale_forced") or set()
        stale_action = self.cfg.s("guard.stale_action", "hold")

        # Holding beats inventing a fill off a frozen mark — but only while the
        # venue is expected back. A token that is delisted (no price, no volume)
        # never comes back, and waiting for guard.max_stale_hours strands the
        # capital and freezes equity on a number nobody can trade. After this
        # long without a quote the position is written off at its last mark.
        dead_mark_minutes = self.f("dead_mark_minutes", 15.0)

        # Profit handling. A fixed target alone is not enough: a position that runs
        # to +90% and then reverses must not be allowed to round-trip back to the
        # stop. So we (a) take half off at a first target, (b) trail the rest.
        partial_target = self.f("take_profit_partial_pct", 0.60)
        partial_fraction = clamp(self.f("take_profit_partial_fraction", 0.5), 0.05, 1.0)
        trail_arm = self.f("trail_arm_pct", 0.35)
        trail_giveback = clamp(self.f("trail_giveback_pct", 0.30), 0.05, 0.95)

        # A binary market can resolve over weeks; a memecoin cannot. Separate
        # risk parameters per venue keep the two from fighting each other.
        binary = (
            self.f("stop_loss_pct", 0.35),
            self.f("take_profit_pct", 1.5),
            self.f("max_hold_hours", 24.0),
        )
        token = (
            self.f("stop_loss_pct_token", 0.25),
            self.f("take_profit_pct_token", 1.0),
            self.f("max_hold_hours_token", 2.0),
        )
        # A 15-minute window is over before any of the above can matter, and it
        # resolves to 0 or 1 — so the only acceptable end state is flat.
        crypto = (
            self.f("stop_loss_pct_crypto", 0.30),
            self.f("take_profit_pct_crypto", 1.0),
            self.f("max_hold_minutes_crypto", 10.0) / 60.0,
        )
        # A snipe is a different trade: the thesis is "this crowd is real for the
        # next twenty minutes", so the clock is the thesis. Holding a launch for
        # two hours is not patience, it is a different position with no reason.
        sniper = (
            self.f("stop_loss_pct_sniper", 0.35),
            self.f("take_profit_pct_sniper", 1.5),
            self.f("max_hold_minutes_sniper", 20.0) / 60.0,
        )
        exit_seconds = self.f("exit_seconds_before_window", 120.0)

        orders: list[Order] = []
        by_key = ctx.market_by_key()

        for key, position in list(ctx.broker.positions.items()):
            market = by_key.get(key) or _market_from_position(position)
            price = market.price or position.last_price
            if price <= 0:
                continue

            stop_loss, take_profit, max_hold = (
                token if market.venue is Venue.PUMPFUN else binary
            )
            is_crypto = (
                market.meta.get("profile") == CRYPTO_PROFILE
                or position.profile == CRYPTO_PROFILE
            )
            if is_crypto:
                stop_loss, take_profit, max_hold = crypto
            is_sniper = (
                market.meta.get("profile") == SNIPER_PROFILE
                or position.profile == SNIPER_PROFILE
            )
            if is_sniper:
                stop_loss, take_profit, max_hold = sniper
            reason = None
            qty_fraction = 1.0
            ret = position.return_pct

            # best return this position has ever shown (high_water is a price,
            # tracked by PaperBroker.mark on every tick)
            peak = 0.0
            if position.avg_price > 0 and position.high_water > 0:
                peak = (position.high_water - position.avg_price) / position.avg_price

            # A token the venue still refuses to quote while it is happily
            # quoting everything else has been delisted, not gone quiet. Every
            # rule below reads `ret`, computed from a mark we can no longer
            # trust, so the write-off is decided before them. Note the venue
            # must be *alive*: if the whole feed is down the token is fine and
            # the guard's hold (then max_stale_hours) is the right answer.
            mark_age_min = _minutes_since(position.marked_at or position.opened_at)
            write_off = (
                dead_mark_minutes > 0
                and position.venue is Venue.PUMPFUN
                and mark_age_min >= dead_mark_minutes
                and position.venue not in stale
            )

            if write_off:
                reason = f"unpriced {mark_age_min:.0f}m (no {position.venue.value} quote)"
            elif ret <= -stop_loss:
                reason = f"stop-loss {ret * 100:.1f}%"
            elif ret >= take_profit:
                reason = f"take-profit {ret * 100:+.1f}%"
            elif (
                not position.partial_taken
                and partial_target > 0
                and ret >= partial_target
                and position.market_value * partial_fraction >= self.f("min_notional_usd", 5.0) * 2
            ):
                reason = f"take-profit-partial {ret * 100:+.1f}% (half off)"
                qty_fraction = partial_fraction
            elif peak >= trail_arm and ret <= peak * (1.0 - trail_giveback):
                reason = (
                    f"trailing-stop peak {peak * 100:+.0f}% -> {ret * 100:+.0f}%"
                )
            elif is_crypto and (
                market.hours_to_resolution is not None
                and market.hours_to_resolution * 3600.0 <= exit_seconds
            ):
                reason = f"window-end {market.hours_to_resolution * 3600.0:.0f}s left"
            elif position.age_hours >= max_hold:
                reason = f"max-hold {position.age_hours:.1f}h"
            else:
                # only exit on thesis decay once the position has had time to work,
                # and only if the thesis actually got *worse* than it was at entry
                thesis = self._thesis_score(ctx, key)
                decayed = (
                    thesis is not None
                    and position.age_hours >= min_hold
                    and thesis - position.entry_thesis <= -flip_delta
                )
                if decayed:
                    reason = (
                        f"thesis-decay {thesis:+.2f} "
                        f"(entry {position.entry_thesis:+.2f})"
                    )
                elif (
                    market.venue is Venue.POLYMARKET
                    and not is_crypto
                    and market.hours_to_resolution is not None
                    and 0 <= market.hours_to_resolution <= near_resolution
                ):
                    reason = "near-resolution"

            if not reason:
                continue

            if position.venue in stale and stale_action == "hold":
                if position.venue not in forced:
                    ctx.reject(
                        market,
                        f"stale-mark: {position.venue.value} quiet, "
                        f"holding instead of filling {reason}",
                        agent="risk",
                    )
                    continue
                reason = f"{reason} [stale-forced]"

            orders.append(
                Order(
                    market=market,
                    side=Side.SELL,
                    qty=position.qty * qty_fraction,
                    limit_price=market.bid or price,
                    reason=reason,
                    tag="exit",
                )
            )
        return orders

    def _thesis_score(self, ctx: Context, key: str) -> float | None:
        """Sentiment + flow only (no price momentum) — 'is the story still valid?'"""
        parts: list[float] = []
        sentiment = ctx.sentiment.get(key)
        if sentiment:
            parts.append(sentiment.score * sentiment.confidence)
        flow = ctx.flow.get(key)
        if flow:
            parts.append(flow.bias)
        if not parts:
            return None
        return sum(parts) / len(parts)

    # --------------------------------------------------------------- entries
    def _build_entries(self, ctx: Context, equity: float, day_pnl_pct: float) -> list[Order]:
        state = ctx.state
        if state.get("halted") or state.get("deadman"):
            return []

        max_trades_day = self.i("max_trades_per_day", 40)
        if state.get("trades_today", 0) >= max_trades_day:
            state["halted"] = True
            ctx.note(f"risk: HALT, {max_trades_day} trades today")
            return []

        max_positions = self.i("max_open_positions", 8)
        max_per_venue_positions = self.i("max_positions_per_venue", 4)
        max_total = self.f("max_total_exposure_pct", 0.45)
        max_venue = self.f("max_per_venue_pct", 0.30)
        max_position_pct = self.f("max_position_pct", 0.08)
        max_position_usd = self.f("max_position_usd", 80.0)
        min_edge = self.f("min_edge_after_costs", 0.03)
        max_spread = self.f("max_spread_pct", 0.06)
        max_spread_crypto = self.f("crypto_max_spread_pct", 0.08)
        min_liquidity = self.f("min_liquidity_usd", 5_000.0)
        cooldown_min = self.f("cooldown_minutes", 30.0)
        min_notional = self.f("min_notional_usd", 5.0)

        orders: list[Order] = []
        open_groups = {(p.venue, p.question) for p in ctx.broker.positions.values()}

        # Crypto keeps its own book. It has separate caps below, so counting it
        # in the venue counters would let two 15-minute positions halve
        # Polymarket's capacity for the binary book — which is exactly what a
        # "separate cap" is supposed to prevent.
        venue_positions: dict[Venue, int] = {}
        venue_exposure: dict[Venue, float] = {venue: 0.0 for venue in Venue}
        crypto_positions = 0
        crypto_exposure = 0.0
        sniper_positions = 0
        sniper_exposure = 0.0
        for position in ctx.broker.positions.values():
            if position.profile == CRYPTO_PROFILE:
                crypto_positions += 1
                crypto_exposure += position.market_value
                continue
            # A snipe is a pump.fun position, but it must not spend the pump.fun
            # book's six slots: a $15 lottery ticket would then displace a $43
            # position that the model actually priced.
            if position.profile == SNIPER_PROFILE:
                sniper_positions += 1
                sniper_exposure += position.market_value
                continue
            venue_positions[position.venue] = venue_positions.get(position.venue, 0) + 1
            venue_exposure[position.venue] = (
                venue_exposure.get(position.venue, 0.0) + position.market_value
            )
        total_exposure = ctx.broker.gross_exposure()

        max_crypto_positions = self.i("max_crypto_positions", 2)
        max_crypto_trades = self.i("max_crypto_trades_per_day", 24)
        max_position_usd_crypto = self.f("max_position_usd_crypto", 40.0)
        crypto_trades = int(state.get("crypto_trades_today", 0))
        crypto_added = 0
        # Crypto is sized by its own caps, so it gets its own exposure budget
        # instead of borrowing Polymarket's 30%.
        crypto_room = max_crypto_positions * max_position_usd_crypto - crypto_exposure

        # The sniper book is deliberately tiny and separate. Two tickets of $15
        # against a $1000 book is 3% at risk on a coin flip with a twenty-minute
        # clock — sized to be survivable when most launches go to zero, which is
        # the only sizing that makes sense for this trade.
        max_sniper_positions = self.i("max_sniper_positions", 2)
        max_sniper_trades = self.i("max_sniper_trades_per_day", 15)
        max_position_usd_sniper = self.f("max_position_usd_sniper", 15.0)
        sniper_trades = int(state.get("sniper_trades_today", 0))
        sniper_added = 0
        sniper_room = max_sniper_positions * max_position_usd_sniper - sniper_exposure

        for signal in ctx.signals:
            market = signal.market
            group = (market.venue, market.question)
            is_crypto = market.meta.get("profile") == CRYPTO_PROFILE
            is_sniper = market.meta.get("profile") == SNIPER_PROFILE

            if is_crypto:
                if crypto_positions + crypto_added >= max_crypto_positions:
                    ctx.reject(market, f"crypto cap: {max_crypto_positions} open")
                    continue
                if crypto_trades + crypto_added >= max_crypto_trades:
                    ctx.reject(market, f"crypto cap: {max_crypto_trades} trades today")
                    continue

            if is_sniper:
                if sniper_positions + sniper_added >= max_sniper_positions:
                    ctx.reject(market, f"sniper cap: {max_sniper_positions} open")
                    continue
                if sniper_trades + sniper_added >= max_sniper_trades:
                    ctx.reject(market, f"sniper cap: {max_sniper_trades} trades today")
                    continue

            if len(ctx.broker.positions) + len(orders) >= max_positions:
                ctx.reject(market, f"max_open_positions ({max_positions})")
                continue
            # Slot counts are per venue, not one number for all of them. A
            # pump.fun token is ~$43 and the venue carries a 30% exposure budget
            # of its own, so four slots is a *count* limit that bites long before
            # the money runs out: a live run rejected 121 entries at "4 open on
            # pumpfun" while $137 of that budget sat unused. Polymarket positions
            # are ~$37 with a real book behind them, so its four stay as they are.
            venue_cap = self.i(
                f"max_positions_per_venue_{market.venue.value}", max_per_venue_positions
            )
            if not is_crypto and not is_sniper and venue_positions.get(market.venue, 0) >= venue_cap:
                ctx.reject(
                    market,
                    f"venue cap: {venue_cap} open on {market.venue.value}",
                )
                continue
            if group in open_groups:
                ctx.reject(market, "already positioned in this market")
                continue

            last_trade = state["last_trade_ts"].get(market.key)
            if last_trade:
                age_min = (utcnow() - last_trade).total_seconds() / 60.0
                if age_min < cooldown_min:
                    ctx.reject(market, f"cooldown {age_min:.0f}/{cooldown_min:.0f} min")
                    continue

            # A 15-minute leg trades near 15c, where the minimum 1c tick is
            # already ~7% of price. Judging it by the binary book's 6% cap
            # rejects every cheap leg — and the cheap legs are where the edge is.
            spread_cap = max_spread_crypto if is_crypto else max_spread
            if market.spread_pct > spread_cap:
                ctx.reject(
                    market,
                    f"spread {market.spread_pct * 100:.1f}% > {spread_cap * 100:.1f}%",
                )
                continue
            if market.liquidity_usd < min_liquidity:
                ctx.reject(market, f"liquidity ${market.liquidity_usd:,.0f} too thin")
                continue
            if signal.edge < min_edge:
                ctx.reject(market, f"edge {signal.edge * 100:.1f}% < {min_edge * 100:.1f}% after costs")
                continue

            size_usd = self._size(ctx, signal, equity, max_position_pct, max_position_usd)
            if is_crypto:
                # a coin flip with a 15-minute clock does not deserve a full
                # Kelly slice, however good the number looks
                size_usd = min(size_usd, max_position_usd_crypto)
            if is_sniper:
                size_usd = min(size_usd, max_position_usd_sniper)
            if size_usd <= 0:
                ctx.reject(market, "size resolved to zero")
                continue

            # portfolio budgets
            room_total = equity * max_total - total_exposure
            if is_crypto:
                room_venue = crypto_room
            elif is_sniper:
                room_venue = sniper_room
            else:
                room_venue = equity * max_venue - venue_exposure.get(market.venue, 0.0)
            room_cash = ctx.broker.cash * 0.95
            budget = min(size_usd, room_total, room_venue, room_cash)
            if budget < min_notional:
                ctx.reject(market, f"exposure budget exhausted (${budget:,.0f} left)")
                continue
            size_usd = budget

            price = signal.entry_price
            if price <= 0:
                ctx.reject(market, "no executable price")
                continue
            qty = size_usd / price
            notional = qty * price
            if notional < min_notional:
                ctx.reject(market, f"notional ${notional:,.2f} < min ${min_notional:,.2f}")
                continue

            orders.append(
                Order(
                    market=market,
                    side=Side.BUY,
                    qty=qty,
                    limit_price=price,
                    reason="; ".join(signal.reasons[:3]),
                    edge=signal.edge,
                    confidence=signal.confidence,
                    notional=notional,
                    tag="entry",
                    thesis=self._thesis_score(ctx, market.key) or 0.0,
                )
            )
            total_exposure += notional
            open_groups.add(group)
            if is_crypto:
                crypto_added += 1
                crypto_room -= notional
            elif is_sniper:
                sniper_added += 1
                sniper_room -= notional
            else:
                venue_exposure[market.venue] = venue_exposure.get(market.venue, 0.0) + notional
                venue_positions[market.venue] = venue_positions.get(market.venue, 0) + 1

        return orders

    def _size(self, ctx: Context, signal: Signal, equity: float,
              max_position_pct: float, max_position_usd: float) -> float:
        kelly_fraction = self.f("kelly_fraction", 0.25)
        risk_per_trade = self.f("risk_per_trade_pct", 0.015)
        stop_loss = max(self.f("stop_loss_pct", 0.35), 0.01)

        if signal.market.venue is Venue.POLYMARKET:
            price = clamp(signal.entry_price, 0.01, 0.99)
            fair = clamp(signal.fair_price, 0.01, 0.99)
            kelly = (fair - price) / max(1.0 - price, 1e-6)
            frac = clamp(kelly, 0.0, 1.0) * kelly_fraction
        else:
            frac = risk_per_trade / stop_loss

        frac = clamp(frac, 0.0, max_position_pct)
        return min(equity * frac, max_position_usd)


# ------------------------------------------------------------------- helpers

def _minutes_since(when) -> float:
    """Minutes elapsed since `when`, clamped at zero for clock skew."""
    if when is None:
        return 0.0
    return max(0.0, (utcnow() - when).total_seconds() / 60.0)


def _market_from_position(position: Position) -> Market:
    return Market(
        venue=position.venue,
        market_id=position.token_id,
        question=position.question,
        token_id=position.token_id,
        price=position.last_price,
        bid=position.last_price,
        ask=position.last_price,
        meta={"profile": position.profile} if position.profile else {},
    )
