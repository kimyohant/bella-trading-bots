"""Crypto agent — fair value for the 15-minute BTC/ETH/XRP up-or-down markets.

Those markets are not opinion markets, they are a coin flip with a clock. The
resolution rule is: "Up" if the price at the end of the window is at or above the
price at the beginning. So the whole problem is one number —

    r  = ln(spot_now / spot_at_window_open)      how far it has moved
    tau = seconds left in the window             how much time is left to move
    sigma = per-second volatility                how much it *can* still move

and if log price is roughly Gaussian, the probability of finishing above the open
is `Phi(r / (sigma * sqrt(tau)))`. As tau -> 0 that goes to 1 or 0, which is
exactly right: the last tick decides.

This agent only produces numbers. `signal` turns them into orders, and only when
the move is big enough that it cannot be explained by the basis between Coinbase
spot and the Chainlink feed the market actually settles on.
"""

from __future__ import annotations

import asyncio
import math

from ..data import spot as spot_data
from ..models import CRYPTO_PROFILE
from ..utils import clamp, utcnow
from .base import Agent, Context, register

WINDOW_SECONDS = 900.0


@register("crypto")
class CryptoAgent(Agent):
    name = "crypto"

    async def run(self, ctx: Context) -> None:
        ctx.state["crypto"] = {}
        markets = [m for m in ctx.markets if m.meta.get("profile") == CRYPTO_PROFILE]
        if not markets:
            return

        min_seconds = self.i("min_seconds_left", 90)
        basis_bps = self.f("basis_bps", 4.0)
        min_z = self.f("min_z", 0.8)
        default_sigma = self.f("default_sigma_15m", 0.003)
        lookback = self.i("vol_lookback_minutes", 90)
        now_ts = utcnow().timestamp()

        snapshots: dict[str, tuple[float | None, list[tuple[int, float, float]], float | None]] = {}
        spots: dict[str, float] = {}

        # One round of requests per symbol, in parallel: at a 15-second tick,
        # six sequential round trips is most of the tick budget.
        symbols = sorted(
            {str(m.meta.get("symbol") or "").upper() for m in markets} - {""}
        )
        fetched = await asyncio.gather(
            *(self._snapshot(symbol, lookback) for symbol in symbols)
        )
        snapshots.update(zip(symbols, fetched))

        for market in markets:
            symbol = str(market.meta.get("symbol") or "").upper()
            start_ts = _int(market.meta.get("window_start"))
            end_ts = _int(market.meta.get("window_end"))
            if not symbol or not start_ts or not end_ts:
                continue

            price, rows, sigma = snapshots.get(symbol, (None, [], None))
            if price is None:
                continue
            spots[symbol] = price

            base = spot_data.window_open(rows, start_ts)
            if base is None or base <= 0:
                continue

            tau = end_ts - now_ts
            if tau <= min_seconds or tau > WINDOW_SECONDS:
                continue

            per_second = sigma if sigma and sigma > 0 else default_sigma / math.sqrt(WINDOW_SECONDS)
            denom = per_second * math.sqrt(tau)
            if denom <= 0:
                continue

            move = math.log(price / base)
            move_bps = (price / base - 1.0) * 10_000.0
            z = move / denom
            p_up = _phi(z)

            fair = p_up if str(market.meta.get("leg", "Up")).lower() == "up" else 1.0 - p_up
            blocked = None
            if abs(move_bps) < basis_bps:
                blocked = f"inside basis noise ({move_bps:+.1f}bp < {basis_bps:.0f}bp)"
            elif abs(z) < min_z:
                blocked = f"weak signal (z={z:+.2f} < {min_z:.2f})"

            ctx.state["crypto"][market.key] = {
                "symbol": symbol,
                "fair": round(clamp(fair, 0.01, 0.99), 4),
                "p_up": round(p_up, 4),
                "z": round(z, 3),
                "move_bps": round(move_bps, 2),
                "tau_s": round(tau, 1),
                "spot": price,
                "open": base,
                "sigma_15m": round(per_second * math.sqrt(WINDOW_SECONDS), 6),
                "blocked": blocked,
            }

            # Only surface a rejection when the price looked cheap and a gate
            # stopped us — otherwise every tick of every window becomes noise.
            ask = market.ask or market.price
            if blocked and ask > 0 and (fair - ask) / ask >= self.f("noise_edge", 0.08):
                ctx.reject(market, f"crypto: {blocked}", agent=self.name)

        if spots:
            ctx.state["spot"] = spots

    async def _snapshot(
        self, symbol: str, lookback: int
    ) -> tuple[float | None, list[tuple[int, float, float]], float | None]:
        price = await spot_data.spot_price(symbol)
        rows = await spot_data.candles(symbol, lookback)
        sigma = spot_data.sigma_per_second([close for _, _, close in rows])
        return price, rows, sigma


def _phi(z: float) -> float:
    """Standard normal CDF, without pulling in scipy for one line of maths."""
    return 0.5 * (1.0 + math.erf(clamp(z, -8.0, 8.0) / math.sqrt(2.0)))


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
