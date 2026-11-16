"""pump.fun adapter — new-token discovery, read-only.

pump.fun publishes no documented trading API and its frontend JSON endpoint is
behind Cloudflare, so this adapter uses a layered strategy:

  1. **GeckoTerminal** `networks/solana/dexes/pump-fun/pools` — primary source.
     Returns live bonding-curve pools with USD price, FDV, reserve depth, 24h
     taker counts and multi-horizon price change. Free, no key, and it gives the
     whale agent real flow data for free.
  2. **pump.fun frontend API** — legacy fallback (works from some networks).
  3. **Dexscreener** — last resort for a single mint's price/liquidity.

Every layer degrades to an empty list instead of raising, because these are all
unofficial endpoints that can disappear.

For production trading you would replace this with your own indexer (Geyser /
Helius) plus direct bonding-curve reads; the adapter interface stays identical.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..data.http import HttpError, fetch_json
from ..models import Market, Quote, Venue
from ..utils import clamp, parse_ts, utcnow
from .base import VenueAdapter

GECKO_BASE = "https://api.geckoterminal.com/api/v2"
FRONTEND_BASES = (
    "https://frontend-api-v3.pump.fun",
    "https://frontend-api.pump.fun",
)
DEXSCREENER = "https://api.dexscreener.com/latest/dex"
SOL_MINT = "So11111111111111111111111111111111111111112"

VIRTUAL_SOL_START = 30.0
VIRTUAL_SOL_COMPLETE = 85.0
LAMPORTS = 1e9
TOKEN_DECIMALS = 1e6


class PumpFunVenue(VenueAdapter):
    venue = Venue.PUMPFUN
    name = "pumpfun"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.limit = cfg.i("scanner.pumpfun_limit", 25)
        self.min_mcap = cfg.f("scanner.pumpfun_min_market_cap_usd", 8_000)
        self.max_mcap = cfg.f("scanner.pumpfun_max_market_cap_usd", 500_000)
        self.min_age = cfg.f("scanner.pumpfun_min_age_minutes", 3)
        self.max_age = cfg.f("scanner.pumpfun_max_age_minutes", 720)
        self.include_complete = cfg.b("scanner.pumpfun_include_complete", False)
        self.spread_bps = cfg.f("scanner.pumpfun_spread_bps", 200)
        self.sol_fallback = cfg.f("scanner.pumpfun_sol_price_fallback", 150.0)
        self.pages = cfg.i("scanner.pumpfun_pages", 2)
        self.source = "none"
        self._sol_price: float | None = None
        self._flow: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ markets
    async def list_markets(self) -> list[Market]:
        pools = await self._fetch_pools()
        sol_usd = self.sol_fallback
        if self.source == "frontend-api":
            sol_usd = await self._sol_price_usd()

        markets: list[Market] = []
        for raw in pools:
            parsed = (
                self._parse_gecko(raw) if "attributes" in raw
                else self._parse_coin(raw, sol_usd)
            )
            if parsed and self._passes_filters(parsed):
                markets.append(parsed)

        # newest and most active first
        markets.sort(
            key=lambda m: (m.volume_24h_usd, -m.meta.get("age_minutes", 0.0)),
            reverse=True,
        )
        markets = markets[: self.limit]
        self._flow = {
            m.key: {
                "buys": m.meta.get("buys_24h", 0.0),
                "sells": m.meta.get("sells_24h", 0.0),
                "volume_24h_usd": m.volume_24h_usd,
                "liquidity_usd": m.liquidity_usd,
                "price_usd": m.price,
            }
            for m in markets
        }
        return markets

    async def _fetch_pools(self) -> list[dict[str, Any]]:
        pools = await self._gecko_pools()
        if pools:
            self.source = "geckoterminal"
            return pools

        coins = await self._frontend_coins()
        if coins:
            self.source = "frontend-api"
            return coins

        self.source = "none"
        return []

    async def _gecko_pools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, max(1, self.pages) + 1):
            try:
                payload = await fetch_json(
                    f"{GECKO_BASE}/networks/solana/dexes/pump-fun/pools",
                    params={"page": page},
                    ttl=20,
                    retries=1,
                )
            except HttpError:
                break
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                break
            out.extend(item for item in data if isinstance(item, dict))
        return out

    async def _frontend_coins(self) -> list[dict[str, Any]]:
        params = {
            "offset": 0,
            "limit": max(self.limit * 2, 40),
            "sort": "created_timestamp",
            "order": "DESC",
            "includeNsfw": "false",
        }
        for base in FRONTEND_BASES:
            try:
                data = await fetch_json(f"{base}/coins", params=params, ttl=20, retries=1)
            except HttpError:
                continue
            if isinstance(data, list) and data:
                return [c for c in data if isinstance(c, dict)]
        return []

    # ------------------------------------------------------------------ parsing
    def _parse_gecko(self, pool: dict[str, Any]) -> Market | None:
        """GeckoTerminal pool shape."""
        attrs = pool.get("attributes") or {}
        if not attrs.get("base_token_price_usd"):
            return None

        relationships = pool.get("relationships") or {}
        token_id = (
            (relationships.get("base_token") or {}).get("data", {}).get("id") or ""
        )
        mint = token_id.split("_", 1)[1] if "_" in token_id else attrs.get("address", "")
        if not mint:
            return None

        price = _num(attrs.get("base_token_price_usd"))
        if price <= 0:
            return None

        created = parse_ts(attrs.get("pool_created_at"))
        age_minutes = (utcnow() - created).total_seconds() / 60.0 if created else 0.0

        mcap = _num(attrs.get("market_cap_usd")) or _num(attrs.get("fdv_usd"))
        volume = _num((attrs.get("volume_usd") or {}).get("h24"))
        liquidity = _num(attrs.get("reserve_in_usd"))
        txns = (attrs.get("transactions") or {}).get("h24") or {}
        changes = attrs.get("price_change_percentage") or {}

        raw_name = str(attrs.get("name") or "")
        symbol = raw_name.split("/")[0].strip() if "/" in raw_name else raw_name.strip()

        half = self.spread_bps / 10_000.0 / 2.0
        return Market(
            venue=self.venue,
            market_id=mint,
            question=symbol or mint[:8],
            token_id=mint,
            outcome="TOKEN",
            price=price,
            bid=price * (1.0 - half),
            ask=price * (1.0 + half),
            liquidity_usd=liquidity,
            volume_24h_usd=volume,
            end_date=None,
            tags=("memecoin", "pumpfun"),
            meta={
                "symbol": symbol,
                "age_minutes": round(age_minutes, 1),
                "market_cap_usd": mcap,
                "liquidity_usd": liquidity,
                "buys_24h": _num(txns.get("buys")),
                "sells_24h": _num(txns.get("sells")),
                "buyers_24h": _num(txns.get("buyers")),
                "change_m5": _num(changes.get("m5")),
                "change_h1": _num(changes.get("h1")),
                "change_h24": _num(changes.get("h24")),
                "complete": False,
                "pool": attrs.get("address"),
                "source": "geckoterminal",
            },
        )

    def _parse_coin(self, coin: dict[str, Any], sol_usd: float) -> Market | None:
        """Legacy pump.fun frontend shape."""
        mint = str(coin.get("mint") or "")
        if not mint:
            return None

        created = parse_ts(coin.get("created_timestamp"))
        age_minutes = (utcnow() - created).total_seconds() / 60.0 if created else 0.0

        vsr = _num(coin.get("virtual_sol_reserves"))
        vtr = _num(coin.get("virtual_token_reserves"))
        price_usd = 0.0
        if vsr > 0 and vtr > 0:
            price_usd = (vsr / (vtr * 1000.0)) * sol_usd
        price_usd = price_usd or _num(coin.get("price_usd"))
        if price_usd <= 0:
            return None

        supply = _num(coin.get("total_supply")) / TOKEN_DECIMALS
        mcap = _num(coin.get("usd_market_cap")) or price_usd * supply
        half = self.spread_bps / 10_000.0 / 2.0
        progress = (
            clamp((vsr / LAMPORTS - VIRTUAL_SOL_START)
                  / (VIRTUAL_SOL_COMPLETE - VIRTUAL_SOL_START), 0.0, 1.0)
            if vsr > 0 else 0.0
        )

        name = str(coin.get("name") or "")
        symbol = str(coin.get("symbol") or "")
        return Market(
            venue=self.venue,
            market_id=mint,
            question=f"{name} ({symbol})" if name else symbol or mint[:8],
            token_id=mint,
            outcome="TOKEN",
            price=price_usd,
            bid=price_usd * (1.0 - half),
            ask=price_usd * (1.0 + half),
            liquidity_usd=_num(coin.get("liquidity_usd")) or mcap * 0.15,
            volume_24h_usd=_num(coin.get("volume_24h_usd")),
            end_date=None,
            tags=("memecoin", "pumpfun"),
            meta={
                "symbol": symbol,
                "age_minutes": round(age_minutes, 1),
                "market_cap_usd": mcap,
                "curve_progress": round(progress, 4),
                "complete": bool(coin.get("complete") or coin.get("raydium_pool")),
                "replies": _num(coin.get("reply_count")),
                "source": "frontend-api",
            },
        )

    def _passes_filters(self, market: Market) -> bool:
        meta = market.meta
        if meta.get("complete") and not self.include_complete:
            return False
        if not (self.min_mcap <= meta.get("market_cap_usd", 0.0) <= self.max_mcap):
            return False
        age = meta.get("age_minutes", 0.0)
        if not (self.min_age <= age <= self.max_age):
            return False
        return True

    # ------------------------------------------------------------------- quotes
    async def get_quote(self, market: Market) -> Quote:
        """pump.fun trades against a bonding curve, so the "spread" is a model."""
        half = self.spread_bps / 10_000.0 / 2.0
        return Quote(
            bid=market.price * (1.0 - half),
            ask=market.price * (1.0 + half),
            last=market.price,
            liquidity_usd=market.liquidity_usd,
            ts=utcnow(),
        )

    # --------------------------------------------------------------- flow data
    async def flow_snapshot(self) -> dict[str, dict[str, float]]:
        """24h taker buy/sell counts, captured during `list_markets` (no extra calls)."""
        return dict(self._flow)

    async def recent_trades(self) -> list[dict[str, Any]]:
        """No public per-trade tape; flow_snapshot is used instead."""
        return []

    # -------------------------------------------------------------------- util
    async def _sol_price_usd(self) -> float:
        if self._sol_price:
            return self._sol_price
        try:
            data = await fetch_json(f"{DEXSCREENER}/tokens/{SOL_MINT}", ttl=180, retries=1)
            pairs = [p for p in (data.get("pairs") or []) if isinstance(p, dict)]
            if pairs:
                best = max(pairs, key=lambda p: _num((p.get("liquidity") or {}).get("usd")))
                price = _num(best.get("priceUsd"))
                if price > 0:
                    self._sol_price = price
                    return price
        except HttpError:
            pass
        self._sol_price = self.sol_fallback
        return self._sol_price


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
