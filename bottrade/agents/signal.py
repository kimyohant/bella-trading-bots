"""Signal agent — where edge comes from.

For every candidate it builds a *fair* price from three independent nudges applied
to the current market price, then keeps only the ideas whose fair price is far
enough from the executable price to survive fees and slippage.

    fair = f(price, momentum, sentiment, whale flow)

* Polymarket: nudges are applied in logit space, so a shift means the same thing
  at 5% as at 50% (probabilities are bounded, prices are not).
* pump.fun: nudges are applied in log space, since token prices are unbounded and
  multiplicative.
"""

from __future__ import annotations

import math
from collections import deque
from ..models import CRYPTO_PROFILE, SNIPER_PROFILE, Market, Side, Signal, Venue
from ..utils import clamp, fmt_price, mean, sigmoid
from .base import Agent, Context, register


@register("signal")
class SignalAgent(Agent):
    name = "signal"

    async def run(self, ctx: Context) -> None:
        min_edge = self.f("min_edge", 0.05)
        min_confidence = self.f("min_confidence", 0.55)
        max_signals = self.i("max_signals_per_tick", 6)
        w_mom = self.f("momentum_weight", 0.35)
        w_sent = self.f("sentiment_weight", 0.30)
        w_flow = self.f("whale_weight", 0.25)
        binary_scale = self.f("binary_logit_scale", 0.7)
        token_scale = self.f("token_log_scale", 0.25)
        history_len = self.i("momentum_window", 20)
        min_history = self.i("momentum_min_points", 4)

        history: dict[str, deque[float]] = ctx.state.setdefault("price_history", {})
        signals: list[Signal] = []

        for market in ctx.markets:
            prices = history.setdefault(market.key, deque(maxlen=history_len))
            prices.append(market.price)

            # 15-minute crypto markets are priced by the crypto agent from spot,
            # not by nudging the book price with sentiment and flow.
            if market.meta.get("profile") == CRYPTO_PROFILE:
                crypto_signal = self._crypto_signal(ctx, market)
                if crypto_signal is not None:
                    signals.append(crypto_signal)
                continue

            # A launch in its first minutes has no history to nudge: momentum
            # needs four points and the token has none. The sniper agent prices
            # these from buyer breadth instead, so leave them alone here.
            if market.meta.get("profile") == SNIPER_PROFILE:
                continue

            components: dict[str, float] = {}
            coverage = 0.0
            reasons: list[str] = []

            momentum = _venue_momentum(market)
            if momentum is None:
                momentum = _momentum(prices, market.venue, min_history)
            if momentum is not None:
                components["momentum"] = momentum * w_mom
                coverage += 1
                if abs(momentum) > 0.15:
                    reasons.append(f"momentum {momentum:+.2f}")

            sentiment = ctx.sentiment.get(market.key)
            if sentiment and sentiment.confidence > 0.05:
                components["sentiment"] = sentiment.score * sentiment.confidence * w_sent
                coverage += 1
                if abs(sentiment.score) > 0.2:
                    reasons.append(
                        f"sentiment {sentiment.score:+.2f} ({sentiment.source}, "
                        f"n={sentiment.samples})"
                    )

            flow = ctx.flow.get(market.key)
            if flow and abs(flow.bias) > 0.05:
                components["whale"] = flow.bias * w_flow
                coverage += 1
                if abs(flow.bias) > 0.25:
                    reasons.append(f"whale {flow.bias:+.2f} ({flow.detail})")

            if not components:
                continue

            delta = clamp(sum(components.values()), -1.5, 1.5)
            if market.venue is Venue.POLYMARKET:
                price = clamp(market.price, 0.01, 0.99)
                fair = clamp(sigmoid(_logit(price) + delta * binary_scale), 0.01, 0.99)
            else:
                fair = market.price * math.exp(delta * token_scale)

            ask = market.ask or market.price
            if ask <= 0 or fair <= ask:
                continue
            edge = (fair - ask) / ask
            if edge < min_edge:
                continue

            data_quality = clamp(coverage / 3.0, 0.0, 1.0)
            spread_penalty = clamp(1.0 - market.spread_pct / 0.08, 0.0, 1.0)
            confidence = clamp(
                0.30 + 0.40 * min(abs(delta), 1.0) + 0.20 * data_quality
                + 0.10 * spread_penalty,
                0.0, 1.0,
            )
            if confidence < min_confidence:
                continue

            reasons.insert(0, f"fair {fmt_price(fair)} vs ask {fmt_price(ask)} (+{edge * 100:.1f}%)")
            signals.append(
                Signal(
                    market=market,
                    side=Side.BUY,
                    fair_price=round(fair, 6),
                    confidence=round(confidence, 4),
                    reasons=reasons,
                    ttl_seconds=self.i("signal_ttl_seconds", 900),
                )
            )

        # A YES leg and its NO leg are the same bet twice; keep the better one.
        signals = _dedupe(signals)
        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)

        # Per-venue quota: a deep Polymarket book would otherwise fill every slot
        # and the bot would never look at a pump.fun token.
        max_per_venue = self.i("max_signals_per_venue", 4)
        max_crypto = self.i("max_crypto_signals", 2)
        picked: list[Signal] = []
        per_venue: dict[Venue, int] = {}
        crypto_picked = 0
        for signal in signals:
            if signal.market.meta.get("profile") == CRYPTO_PROFILE:
                if crypto_picked >= max_crypto:
                    continue
                crypto_picked += 1
            venue = signal.market.venue
            if per_venue.get(venue, 0) >= max_per_venue:
                continue
            per_venue[venue] = per_venue.get(venue, 0) + 1
            picked.append(signal)
            if len(picked) >= max_signals:
                break
        ctx.signals = picked

        for signal in ctx.signals:
            if ctx.store:
                ctx.store.record_signal(signal)

        self.log.debug(
            "signals=%d (of %d candidates), best edge %.1f%%",
            len(ctx.signals), len(ctx.markets),
            (ctx.signals[0].edge * 100) if ctx.signals else 0.0,
        )


    def _crypto_signal(self, ctx: Context, market: Market) -> Signal | None:
        """Turn the crypto agent's fair probability into a signal, or nothing."""
        info = (ctx.state.get("crypto") or {}).get(market.key)
        if not info or info.get("blocked"):
            return None

        fair = float(info["fair"])
        ask = market.ask or market.price
        if ask <= 0 or fair <= ask:
            return None
        edge = (fair - ask) / ask
        # crypto has its own threshold: the basis risk is bigger than the
        # mispricing on an ordinary opinion market, so it demands more edge
        if edge < self.cfg.f("crypto.min_edge", 0.08):
            return None

        z = abs(float(info["z"]))
        tau = float(info["tau_s"])
        spread_penalty = clamp(1.0 - market.spread_pct / 0.08, 0.0, 1.0)
        confidence = clamp(
            0.42
            + 0.28 * min(z / 2.5, 1.0)
            + 0.20 * clamp(tau / 300.0, 0.0, 1.0)
            + 0.10 * spread_penalty,
            0.0, 0.95,
        )
        if confidence < self.f("min_confidence", 0.55):
            return None

        reasons = [
            f"fair {fmt_price(fair)} vs ask {fmt_price(ask)} (+{edge * 100:.1f}%)",
            f"{info['symbol']} {info['move_bps']:+.0f}bp vs open, z={info['z']:+.2f}",
            f"{tau / 60:.1f}m left, sigma {info['sigma_15m'] * 100:.2f}%/15m",
        ]
        return Signal(
            market=market,
            side=Side.BUY,
            fair_price=round(fair, 6),
            confidence=round(confidence, 4),
            reasons=reasons,
            ttl_seconds=self.i("signal_ttl_seconds", 900),
        )


# ------------------------------------------------------------------- helpers

def _logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))


def _venue_momentum(market: Market) -> float | None:
    """Use the venue's own 1h price change when it publishes one.

    Waiting for our own price history to fill means the bot is blind for the first
    few ticks — and in live mode a tick is 15 seconds, so that is a real gap.
    """
    if market.venue is Venue.POLYMARKET:
        change = market.meta.get("change_1h")
        if change:
            return clamp(float(change) / 0.05, -1.0, 1.0)  # 5 cents/hour = full strength
    else:
        change = market.meta.get("change_h1")
        if change is not None:
            return clamp(float(change) / 25.0, -1.0, 1.0)  # percent, 25%/hour = full
    return None


def _momentum(prices: deque[float], venue: Venue, min_points: int) -> float | None:
    """Deviation of the latest price from its trailing mean, squashed to -1..1."""
    if len(prices) < min_points:
        return None
    latest = prices[-1]
    baseline = mean(list(prices)[:-1])
    if baseline <= 0:
        return None
    deviation = (latest - baseline) / baseline
    scale = 0.06 if venue is Venue.POLYMARKET else 0.18
    return clamp(deviation / scale, -1.0, 1.0)


def _dedupe(signals: list[Signal]) -> list[Signal]:
    best: dict[str, Signal] = {}
    for signal in signals:
        group = f"{signal.market.venue.value}:{signal.market.market_id}"
        current = best.get(group)
        if current is None or signal.edge * signal.confidence > current.edge * current.confidence:
            best[group] = signal
    return list(best.values())
