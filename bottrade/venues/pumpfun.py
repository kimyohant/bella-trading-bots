"""pump.fun adapter — new-token discovery, read-only.

pump.fun publishes no documented trading API and its frontend JSON endpoint is
behind Cloudflare, so this adapter uses a layered strategy:

  1. **GeckoTerminal** `networks/solana/dexes/pump-fun/pools` — primary source.
     Returns live bonding-curve pools with USD price, FDV, reserve depth, 24h
     taker counts and multi-horizon price change. Free, no key, and it gives the
     whale agent real flow data for free.
  2. **pump.fun frontend API** — legacy fallback (works from some networks).
  3. **DexScreener** — independent host, so it survives GeckoTerminal rate-limiting
     us by IP: collect pump.fun mints from `token-profiles`/`token-boosts`, then
     price them all in one batched `tokens/v1` request.

Every layer degrades to an empty list instead of raising, because these are all
unofficial endpoints that can disappear.

For production trading you would replace this with your own indexer (Geyser /
Helius) plus direct bonding-curve reads; the adapter interface stays identical.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from ..data.http import HttpError, fetch_json
from ..models import SNIPER_PROFILE, Market, Quote, Venue
from ..utils import clamp, parse_ts, utcnow
from .base import VenueAdapter

GECKO_BASE = "https://api.geckoterminal.com/api/v2"
FRONTEND_BASES = (
    "https://frontend-api-v3.pump.fun",
    "https://frontend-api.pump.fun",
)
DEXSCREENER = "https://api.dexscreener.com/latest/dex"
DEXSCREENER_API = "https://api.dexscreener.com"
SOL_MINT = "So11111111111111111111111111111111111111112"

VIRTUAL_SOL_START = 30.0
VIRTUAL_SOL_COMPLETE = 85.0
LAMPORTS = 1e9
TOKEN_DECIMALS = 1e6

log = logging.getLogger("bottrade.pumpfun")


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
        # Which dex ids count as "a pump.fun launch" in the network-wide
        # new-pools feed. In a typical poll the feed is 13 pump-fun + 3 pumpswap
        # + 4 Meteora, so this filter is what makes it a pump.fun feed at all.
        self.new_pool_dexes = [
            str(d).strip()
            for d in cfg.list("scanner.pumpfun_new_pool_dexes",
                              ["pumpfun", "pump-fun", "pumpswap"])
            if str(d).strip()
        ]
        # DexScreener only indexes tokens that already have a DEX pair, so
        # essentially everything it returns has graduated off the bonding curve.
        # Excluding those would leave the fallback with ~1 candidate and the
        # venue looking dead again, so it is allowed by default — for that
        # source only.
        self.allow_graduated_fallback = cfg.b(
            "scanner.pumpfun_fallback_allow_graduated", True
        )
        self.spread_bps = cfg.f("scanner.pumpfun_spread_bps", 200)
        self.sol_fallback = cfg.f("scanner.pumpfun_sol_price_fallback", 150.0)
        self.pages = cfg.i("scanner.pumpfun_pages", 2)
        # GeckoTerminal rate-limits by IP on a monthly quota, and the bot was
        # spending it: a 20s cache on a ~35s tick means every tick is a fresh
        # call, which is 3 calls/minute for a pool list that barely changes and
        # ~10k calls/month — the whole free tier. The list is a 24-hour ranking;
        # 90s is still fresher than the number it is ranking.
        self.discovery_ttl = cfg.i("scanner.pumpfun_ttl_seconds", 90)
        # The launch feed is the one place freshness matters, but a token stays
        # snipeable for 8 minutes, so seeing the same snapshot for 45s still
        # catches every launch with room to spare.
        self.new_pool_ttl = cfg.i("sniper.feed_ttl_seconds", 45)
        # Prices come from a different host than discovery, so they can be fresher
        # than the list they belong to without costing any GeckoTerminal quota.
        self.price_ttl = cfg.i("scanner.pumpfun_price_ttl_seconds", 15)
        self.source = "none"
        self._last_source: str | None = None
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
            if "attributes" in raw:
                parsed = self._parse_gecko(raw)
            elif "baseToken" in raw:
                parsed = self._parse_dexscreener(raw)
            else:
                parsed = self._parse_coin(raw, sol_usd)
            if parsed and self._passes_filters(parsed):
                markets.append(parsed)

        # newest and most active first
        markets.sort(
            key=lambda m: (m.volume_24h_usd, -m.meta.get("age_minutes", 0.0)),
            reverse=True,
        )
        markets = markets[: self.limit]
        # The pool list is cached for 90s to stop it eating the monthly quota, but
        # a memecoin's price is not a 90-second quantity: the bot would size and
        # fill against a price nobody can trade. Refresh the prices of the names
        # it actually picked from a different host — one batched call.
        await self._refresh_prices(markets)
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
            self._note_source("geckoterminal")
            return pools

        coins = await self._frontend_coins()
        if coins:
            self._note_source("frontend-api")
            return coins

        pairs = await self._dexscreener_pools()
        if pairs:
            self._note_source("dexscreener")
            return pairs

        self._note_source("none")
        return []

    def _note_source(self, source: str) -> None:
        """Record which host served this tick, and say so when it changes.

        A venue that has quietly fallen back to a thin source looks exactly like
        a venue with nothing worth trading, and that ambiguity is what lets an
        outage run for an hour before anyone notices. `_last_source` starts as
        None so the very first healthy tick stays quiet, while a venue that is
        dead from the start still says so.
        """
        previous, self._last_source = self._last_source, source
        self.source = source
        if source == previous:
            return
        if source == "none":
            log.warning("every source failed — venue empty this tick")
        elif source == "dexscreener":
            log.warning("primary source down — served by dexscreener")
        elif previous is not None:
            log.info("source recovered: %s", source)

    async def _gecko_pools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, max(1, self.pages) + 1):
            try:
                payload = await fetch_json(
                    f"{GECKO_BASE}/networks/solana/dexes/pump-fun/pools",
                    params={"page": page},
                    ttl=self.discovery_ttl,
                    retries=1,
                )
            except HttpError:
                break
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                break
            out.extend(item for item in data if isinstance(item, dict))
        return out

    async def new_pools(self, limit: int = 20) -> list[Market]:
        """Pools created in roughly the last minute, newest first.

        `/new_pools` is a network-wide feed, so the pump.fun filter has to be
        applied on the relationship rather than in the URL: in a typical poll 16
        of 20 rows are pump.fun (13 pump-fun + 3 pumpswap) and the rest are
        Meteora. Ordering is by creation time, which is the only ordering that
        matters here — by the time a launch ranks by volume it is not a launch.
        """
        try:
            payload = await fetch_json(
                f"{GECKO_BASE}/networks/solana/new_pools",
                params={"page": 1},
                ttl=self.new_pool_ttl,
                retries=1,
            )
        except HttpError:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []

        wanted = {str(d).strip() for d in self.new_pool_dexes if str(d).strip()}
        out: list[Market] = []
        for pool in rows[: max(1, limit)]:
            if not isinstance(pool, dict):
                continue
            dex = (
                ((pool.get("relationships") or {}).get("dex") or {}).get("data") or {}
            ).get("id")
            if wanted and dex not in wanted:
                continue
            market = self._parse_gecko(pool)
            if market is None:
                continue
            market.meta["dex"] = dex
            market.meta["profile"] = SNIPER_PROFILE
            out.append(market)
        return out

    async def _refresh_prices(self, markets: list[Market]) -> None:
        """Overwrite `price/bid/ask` with a fresh batch quote, if one arrives.

        DexScreener prices a list of mints in one request on a host with its own
        budget, so the expensive discovery call can stay cached without the marks
        going stale. Any failure leaves the cached prices in place: a stale mark
        is worse than a fresh one and better than none.
        """
        mints = [m.token_id for m in markets if m.token_id][:30]
        if not mints:
            return
        try:
            payload = await fetch_json(
                f"{DEXSCREENER_API}/tokens/v1/solana/{','.join(mints)}",
                ttl=self.price_ttl,
                retries=0,
            )
        except HttpError:
            return
        rows = payload if isinstance(payload, list) else []
        fresh: dict[str, float] = {}
        for pair in rows:
            if not isinstance(pair, dict):
                continue
            mint = str((pair.get("baseToken") or {}).get("address") or "")
            price = _num(pair.get("priceUsd"))
            if mint and price > 0:
                fresh[mint] = price
        if not fresh:
            return

        half = self.spread_bps / 10_000.0 / 2.0
        for market in markets:
            price = fresh.get(market.token_id)
            if not price:
                continue
            market.price = price
            market.bid = price * (1.0 - half)
            market.ask = price * (1.0 + half)

    async def _dexscreener_pools(self) -> list[dict[str, Any]]:
        """Last resort, and a different host — which is the whole point.

        GeckoTerminal rate-limits by IP, so when it is refusing us the bot used
        to lose the entire venue and look "empty" rather than broken. DexScreener
        is an independent API: collect pump.fun mints from the endpoints that
        actually name them (recently profiled, recently boosted), then price all
        of them in one batch request.
        """
        mints: list[str] = []
        for path in ("token-profiles/latest/v1", "token-boosts/latest/v1"):
            try:
                rows = await fetch_json(f"{DEXSCREENER_API}/{path}", ttl=60, retries=0)
            except HttpError:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or row.get("chainId") != "solana":
                    continue
                mint = str(row.get("tokenAddress") or "")
                if mint.endswith("pump") and mint not in mints:
                    mints.append(mint)
        if not mints:
            return []

        try:
            pairs = await fetch_json(
                f"{DEXSCREENER_API}/tokens/v1/solana/{','.join(mints[:30])}",
                ttl=20,
                retries=0,
            )
        except HttpError:
            return []
        return [p for p in pairs if isinstance(p, dict)] if isinstance(pairs, list) else []

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
        # A pool that is ninety seconds old reports the same numbers in every
        # window, so h24 is not a 24-hour figure — it is "since launch". Keep
        # the five-minute slice separately: for a launch that is the only
        # window that describes the crowd rather than the token's whole life.
        m5 = (attrs.get("transactions") or {}).get("m5") or {}
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
                "buys_m5": _num(m5.get("buys")),
                "sells_m5": _num(m5.get("sells")),
                "buyers_m5": _num(m5.get("buyers")),
                "sellers_m5": _num(m5.get("sellers")),
                "change_m5": _num(changes.get("m5")),
                "change_h1": _num(changes.get("h1")),
                "change_h24": _num(changes.get("h24")),
                "complete": False,
                "pool": attrs.get("address"),
                "source": "geckoterminal",
            },
        )

    def _parse_dexscreener(self, pair: dict[str, Any]) -> Market | None:
        """DexScreener pair shape (`/tokens/v1/solana/{mints}`)."""
        base = pair.get("baseToken") or {}
        mint = str(base.get("address") or "")
        price = _num(pair.get("priceUsd"))
        if not mint or price <= 0:
            return None

        created = pair.get("pairCreatedAt")
        age_minutes = 0.0
        if isinstance(created, (int, float)) and created > 0:
            age_minutes = max(0.0, (utcnow().timestamp() - created / 1000.0) / 60.0)

        symbol = str(base.get("symbol") or "").strip() or mint[:8]
        liquidity = _num((pair.get("liquidity") or {}).get("usd"))
        volume = _num((pair.get("volume") or {}).get("h24"))
        changes = pair.get("priceChange") or {}
        txns = (pair.get("txns") or {}).get("h24") or {}
        mcap = _num(pair.get("marketCap")) or _num(pair.get("fdv"))

        half = self.spread_bps / 10_000.0 / 2.0
        return Market(
            venue=self.venue,
            market_id=mint,
            question=symbol,
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
                # graduated tokens move to pumpswap; keep them out by default
                "complete": str(pair.get("dexId") or "").lower()
                not in ("pumpfun", "pump-fun"),
                "pool": pair.get("pairAddress"),
                "source": "dexscreener",
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
            from_fallback = meta.get("source") == "dexscreener"
            if not (from_fallback and self.allow_graduated_fallback):
                return False
        if not (self.min_mcap <= meta.get("market_cap_usd", 0.0) <= self.max_mcap):
            return False
        age = meta.get("age_minutes", 0.0)
        if not (self.min_age <= age <= self.max_age):
            return False
        return True

    # ------------------------------------------------------------------- quotes
    async def quote_missing(self, keys: list[str]) -> dict[str, float]:
        """Price specific mints via DexScreener, one batched request."""
        mints = [k.split(":", 1)[1] for k in keys if ":" in k][:30]
        if not mints:
            return {}
        try:
            pairs = await fetch_json(
                f"{DEXSCREENER_API}/tokens/v1/solana/{','.join(mints)}",
                ttl=15,
                retries=0,
            )
        except HttpError:
            return {}
        if not isinstance(pairs, list):
            return {}

        out: dict[str, float] = {}
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            mint = str((pair.get("baseToken") or {}).get("address") or "")
            price = _num(pair.get("priceUsd"))
            if mint and price > 0:
                out[f"{self.venue.value}:{mint}"] = price
        return out

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
