"""Whale agent — who is actually pushing the tape.

Two data paths, both public:
  * Polymarket: the Data API trade tape. Large taker prints are aggregated into a
    net buy/sell imbalance per token, weighted by notional and wallet breadth.
  * pump.fun: no public tape, so Dexscreener 24h taker counts are used as the
    aggressive-flow proxy (see `PumpFunVenue.flow_snapshot`).

Output is a `FlowBias` in -1..+1 per market: `+` means aggressive buying.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

from ..models import FlowBias
from ..utils import clamp, utcnow
from .base import Agent, Context, register


@register("whale")
class WhaleAgent(Agent):
    name = "whale"

    async def run(self, ctx: Context) -> None:
        if not self.cfg.b("whale.enabled", True):
            return

        min_notional = self.cfg.f("whale.min_notional_usd", 2_000.0)
        window = self.cfg.i("whale.window_minutes", 60)
        cutoff = utcnow() - timedelta(minutes=window)

        buckets: dict[str, dict[str, Any]] = {}

        # --- path 1: real trade tape --------------------------------------
        for adapter in ctx.adapters.values():
            try:
                trades = await adapter.recent_trades()
            except Exception:  # noqa: BLE001
                continue
            for trade in trades:
                key = trade.get("key")
                notional = float(trade.get("notional") or 0.0)
                if not key or notional < min_notional:
                    continue
                ts = trade.get("ts")
                if ts is not None and ts < cutoff:
                    continue
                bucket = buckets.setdefault(
                    key, {"buy": 0.0, "sell": 0.0, "wallets": set(), "n": 0, "tape": True}
                )
                side = str(trade.get("side", "")).upper()
                if side == "BUY":
                    bucket["buy"] += notional
                elif side == "SELL":
                    bucket["sell"] += notional
                else:
                    continue
                wallet = trade.get("wallet")
                if wallet:
                    bucket["wallets"].add(wallet)
                bucket["n"] += 1

        # --- path 2: aggregated taker counts (pump.fun) --------------------
        for adapter in ctx.adapters.values():
            snapshot = getattr(adapter, "flow_snapshot", None)
            if snapshot is None:
                continue
            try:
                data = await snapshot()
            except Exception:  # noqa: BLE001
                continue
            for key, stats in data.items():
                if key in buckets:
                    continue
                buys = float(stats.get("buys") or 0.0)
                sells = float(stats.get("sells") or 0.0)
                if buys + sells <= 0:
                    continue
                buckets[key] = {
                    "buy": buys,
                    "sell": sells,
                    "wallets": set(),
                    "n": int(buys + sells),
                    "tape": False,
                    "volume": float(stats.get("volume_24h_usd") or 0.0),
                }

        for key, bucket in buckets.items():
            buy, sell = bucket["buy"], bucket["sell"]
            total = buy + sell
            if total <= 0:
                continue
            imbalance = (buy - sell) / total
            if bucket.get("tape"):
                # conviction grows with notional size relative to the threshold
                conviction = clamp(total / (min_notional * 12.0), 0.15, 1.0)
                breadth = clamp(len(bucket["wallets"]) / 5.0, 0.2, 1.0)
                bias = math.tanh(imbalance * 1.6) * conviction * breadth
                detail = (
                    f"tape: {bucket['n']} prints, ${buy:,.0f} buy / ${sell:,.0f} sell, "
                    f"{len(bucket['wallets'])} wallets"
                )
            else:
                bias = math.tanh(imbalance * 1.4) * 0.8
                detail = f"takers 24h: {int(buy)} buy / {int(sell)} sell"

            ctx.flow[key] = FlowBias(
                key=key,
                bias=round(clamp(bias, -1.0, 1.0), 4),
                notional_usd=round(total, 2),
                wallets=len(bucket["wallets"]),
                samples=int(bucket["n"]),
                source="tape" if bucket.get("tape") else "taker-counts",
                detail=detail,
            )

        if ctx.flow:
            strongest = max(ctx.flow.values(), key=lambda f: abs(f.bias))
            self.log.debug(
                "flow on %d markets, strongest %+.2f (%s)",
                len(ctx.flow), strongest.bias, strongest.source,
            )
