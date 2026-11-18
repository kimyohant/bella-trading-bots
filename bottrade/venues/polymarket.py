"""Polymarket adapter — read-only market data.

Sources (all public, no auth):
  * Gamma API  https://gamma-api.polymarket.com  — market metadata & prices
  * CLOB API   https://clob.polymarket.com       — order book (optional, per-token)
  * Data API   https://data-api.polymarket.com   — public trade tape (whale signal)

A binary market produces two legs. The YES leg carries the real book; the NO leg
is synthesised from it, because buying NO at p is the same as selling YES at 1-p.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import Config
from ..data.http import HttpError, fetch_json
from ..models import CRYPTO_PROFILE, Market, Quote, Venue
from ..utils import clamp, parse_ts, utcnow
from .base import VenueAdapter

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

PRICE_FLOOR = 0.005
PRICE_CEIL = 0.995


class PolymarketVenue(VenueAdapter):
    venue = Venue.POLYMARKET
    name = "polymarket"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Gamma caps a page at 100 rows whatever you ask for, so 60 was already
        # leaving markets on the table: at limit=100 thirty-five pass the filters
        # against twenty-one at limit=60, for the same single request.
        self.limit = min(cfg.i("scanner.polymarket_limit", 100), 100)
        self.pages = max(1, cfg.i("scanner.polymarket_pages", 2))
        self.min_liquidity = cfg.f("scanner.polymarket_min_liquidity_usd", 5_000)
        self.min_volume = cfg.f("scanner.polymarket_min_volume_24h_usd", 10_000)
        self.max_days = cfg.f("scanner.polymarket_max_days_to_resolution", 45)
        self.default_spread = cfg.f("scanner.polymarket_default_spread", 0.02)
        self.deep_quotes = cfg.b("scanner.polymarket_deep_quotes", False)
        self._errors = 0

        # 15-minute crypto up/down series (btc-up-or-down-15m, ...)
        self.crypto_enabled = cfg.b("crypto.enabled", True)
        self.crypto_series = [
            str(s).strip() for s in cfg.list("crypto.series", []) if str(s).strip()
        ]
        self.crypto_windows = max(1, cfg.i("crypto.windows_per_series", 2))
        self.crypto_min_liquidity = cfg.f("scanner.crypto_min_liquidity_usd", 400.0)
        # Volume is a useless gate here: a window that opened two minutes ago has
        # a 24h volume of roughly zero by construction.
        self.crypto_min_volume = cfg.f("scanner.crypto_min_volume_24h_usd", 0.0)
        self.crypto_max_spread = cfg.f("scanner.crypto_max_spread_pct", 0.08)

    # ------------------------------------------------------------------ markets
    async def list_markets(self) -> list[Market]:
        raw = await self._fetch_markets()
        markets: list[Market] = []
        for item in raw:
            markets.extend(self._parse_market(item))
        markets = [m for m in markets if self._passes_filters(m)]
        markets.sort(key=lambda m: m.volume_24h_usd, reverse=True)
        markets.extend(await self._fetch_crypto())
        return markets

    async def _fetch_markets(self) -> list[dict[str, Any]]:
        """Fetch the top markets by 24h volume, one page at a time.

        A single page only reaches 100 rows, and the tradable slice of that is
        the *third* of it that survives the filters below. Paging is what turns
        a 21-market universe into a 60-market one; it stops as soon as a page
        comes back short, so a quiet market costs one request, not two.
        """
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for page in range(self.pages):
            base = {
                "active": "true",
                "closed": "false",
                "limit": self.limit,
                "offset": page * self.limit,
            }
            data = None
            for params in (
                {**base, "order": "volume24hr", "ascending": "false"},
                {**base, "order": "volumeNum", "ascending": "false"},
                base,
            ):
                try:
                    data = await fetch_json(f"{GAMMA_BASE}/markets", params=params, ttl=30)
                    break
                except HttpError:
                    self._errors += 1
                    data = None
            if not data:
                break
            chunk = _as_objects(data)
            for item in chunk:
                key = str(item.get("conditionId") or item.get("id") or "")
                if key and key in seen:
                    continue
                seen.add(key)
                rows.append(item)
            if len(chunk) < self.limit:
                break
        return rows

    def _parse_market(self, item: dict[str, Any]) -> list[Market]:
        outcomes = _json_list(item.get("outcomes"))
        prices = _json_list(item.get("outcomePrices"))
        tokens = _json_list(item.get("clobTokenIds"))
        if not tokens or len(tokens) < 2:
            return []

        condition_id = str(item.get("conditionId") or item.get("id") or "")
        question = str(item.get("question") or item.get("title") or "").strip()
        if not question:
            return []

        end_date = parse_ts(item.get("endDate") or item.get("end_date_iso"))
        liquidity = _num(item.get("liquidityNum"), _num(item.get("liquidity"), 0.0))
        volume = _num(item.get("volume24hr"), _num(item.get("volumeNum"), 0.0))

        quoted = _num(prices[0], 0.0) if prices else _num(item.get("lastTradePrice"), 0.0)
        if not 0 < quoted < 1:
            return []

        half = max(self.default_spread, _num(item.get("spread"), self.default_spread)) / 2.0
        yes_bid = _num(item.get("bestBid"), quoted - half)
        yes_ask = _num(item.get("bestAsk"), quoted + half)
        yes_bid = clamp(yes_bid, PRICE_FLOOR, PRICE_CEIL)
        yes_ask = clamp(max(yes_ask, yes_bid + 0.001), PRICE_FLOOR, PRICE_CEIL)

        # `outcomePrices` is often stale versus the live book, so trust the book
        # when it exists: an edge measured against a stale price is not an edge.
        has_book = item.get("bestBid") is not None and item.get("bestAsk") is not None
        yes_price = (yes_bid + yes_ask) / 2.0 if has_book else quoted
        yes_price = clamp(yes_price, PRICE_FLOOR, PRICE_CEIL)

        change_1h = _num(item.get("oneHourPriceChange"))
        change_24h = _num(item.get("oneDayPriceChange"))

        tags: tuple[str, ...] = ()
        category = item.get("category")
        if isinstance(category, str) and category:
            tags = (category.lower(),)

        meta = {
            "condition_id": condition_id,
            "slug": item.get("slug"),
            "neg_risk": bool(item.get("negRisk")),
            "spread": round(yes_ask - yes_bid, 4),
            "group": _event_title(item),
            "change_1h": round(change_1h, 4),
            "change_24h": round(change_24h, 4),
        }

        legs: list[Market] = []
        for index in range(min(2, len(tokens))):
            outcome = str(outcomes[index]) if index < len(outcomes) else ("Yes" if index == 0 else "No")
            if index == 0:
                price, bid, ask = yes_price, yes_bid, yes_ask
            else:
                price = clamp(1.0 - yes_price, PRICE_FLOOR, PRICE_CEIL)
                bid = clamp(1.0 - yes_ask, PRICE_FLOOR, PRICE_CEIL)
                ask = clamp(1.0 - yes_bid, PRICE_FLOOR, PRICE_CEIL)
            legs.append(
                Market(
                    venue=self.venue,
                    market_id=condition_id,
                    question=question,
                    token_id=str(tokens[index]),
                    outcome=outcome,
                    price=round(price, 4),
                    bid=round(bid, 4),
                    ask=round(ask, 4),
                    liquidity_usd=liquidity,
                    volume_24h_usd=volume,
                    end_date=end_date,
                    tags=tags,
                    meta=dict(
                        meta,
                        leg=outcome,
                        yes_price=yes_price,
                        # a YES move up is a NO move down
                        change_1h=meta["change_1h"] if index == 0 else -meta["change_1h"],
                        change_24h=meta["change_24h"] if index == 0 else -meta["change_24h"],
                    ),
                )
            )
        return legs

    def _passes_filters(self, market: Market) -> bool:
        if market.liquidity_usd < self.min_liquidity:
            return False
        if market.volume_24h_usd < self.min_volume:
            return False
        hours = market.hours_to_resolution
        if hours is not None and (hours <= 0 or hours > self.max_days * 24):
            return False
        return True

    # ---------------------------------------------------------- crypto 15m
    async def _fetch_crypto(self) -> list[Market]:
        """The 15-minute up/down series.

        They never appear in the volume-ordered top-60 listing and their
        liquidity is an order of magnitude below the political markets, so they
        need their own query and their own (much lower) filters.
        """
        if not self.crypto_enabled or not self.crypto_series:
            return []

        out: list[Market] = []
        for series in self.crypto_series:
            try:
                data = await fetch_json(
                    f"{GAMMA_BASE}/events",
                    params={
                        "series_slug": series,
                        "closed": "false",
                        "order": "endDate",
                        "ascending": "true",
                        "limit": self.crypto_windows,
                    },
                    ttl=20,
                )
            except HttpError:
                self._errors += 1
                continue
            for event in _as_objects(data):
                for item in event.get("markets") or []:
                    if isinstance(item, dict):
                        out.extend(await self._parse_crypto(item, event))

        return [m for m in out if self._passes_crypto(m)]

    async def _parse_crypto(
        self, item: dict[str, Any], event: dict[str, Any]
    ) -> list[Market]:
        tokens = _json_list(item.get("clobTokenIds"))
        if len(tokens) < 2:
            return []

        slug = str(event.get("slug") or item.get("slug") or "")
        symbol = slug.split("-")[0].upper() if slug else ""
        if not symbol:
            return []

        end_date = parse_ts(event.get("endDate") or item.get("endDate"))
        if end_date is None:
            return []
        end_ts = int(end_date.timestamp())
        start_ts = _slug_ts(slug) or (end_ts - 900)

        question = str(item.get("question") or event.get("title") or "").strip()
        condition_id = str(item.get("conditionId") or item.get("id") or "")
        liquidity = _num(item.get("liquidityNum"), _num(item.get("liquidity"), 0.0))
        volume = _num(item.get("volume24hr"), _num(item.get("volumeNum"), 0.0))
        outcomes = _json_list(item.get("outcomes")) or ["Up", "Down"]

        # Gamma's bestBid/bestAsk on these markets lag the real book by minutes —
        # on one BTC window it read 0.56/0.57 while the CLOB was 0.49/0.50. On a
        # fifteen-minute bet that gap is the whole trade, so read the book itself.
        books = {index: await self._book(str(tokens[index])) for index in range(2)}

        legs: list[Market] = []
        for index in range(2):
            bid, ask = books.get(index, (None, None))
            if bid is None or ask is None:
                bid = _num(item.get("bestBid"), 0.0) or None
                ask = _num(item.get("bestAsk"), 0.0) or None
            if not bid or not ask:
                continue
            bid = clamp(bid, PRICE_FLOOR, PRICE_CEIL)
            ask = clamp(max(ask, bid + 0.001), PRICE_FLOOR, PRICE_CEIL)
            outcome = (
                str(outcomes[index]) if index < len(outcomes)
                else ("Up" if index == 0 else "Down")
            )
            legs.append(
                Market(
                    venue=self.venue,
                    market_id=condition_id,
                    question=question,
                    token_id=str(tokens[index]),
                    outcome=outcome,
                    price=round((bid + ask) / 2.0, 4),
                    bid=round(bid, 4),
                    ask=round(ask, 4),
                    liquidity_usd=liquidity,
                    volume_24h_usd=volume,
                    end_date=end_date,
                    tags=("crypto", "updown"),
                    meta={
                        "condition_id": condition_id,
                        "slug": slug,
                        "profile": CRYPTO_PROFILE,
                        "symbol": symbol,
                        "window_start": start_ts,
                        "window_end": end_ts,
                        "leg": outcome,
                        "spread": round(ask - bid, 4),
                        "group": str(event.get("title") or ""),
                        "change_1h": 0.0,
                        "change_24h": 0.0,
                    },
                )
            )
        return legs

    async def _book(self, token_id: str) -> tuple[float | None, float | None]:
        try:
            book = await fetch_json(
                f"{CLOB_BASE}/book", params={"token_id": token_id}, ttl=8
            )
        except HttpError:
            return None, None
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = max((_num(b.get("price"), 0.0) for b in bids), default=0.0) or None
        best_ask = min(
            (_num(a.get("price"), 1.0) for a in asks if _num(a.get("price"), 0.0) > 0),
            default=0.0,
        ) or None
        return best_bid, best_ask

    def _passes_crypto(self, market: Market) -> bool:
        if market.liquidity_usd < self.crypto_min_liquidity:
            return False
        if market.volume_24h_usd < self.crypto_min_volume:
            return False
        if market.spread_pct > self.crypto_max_spread:
            return False
        hours = market.hours_to_resolution
        if hours is not None and hours <= 0:
            return False
        return True

    # ------------------------------------------------------------------- quotes
    async def quote_missing(self, keys: list[str]) -> dict[str, float]:
        """Price specific legs from the CLOB book, for positions out of quota."""
        out: dict[str, float] = {}
        for key in keys[:8]:
            token_id = key.split(":", 1)[1] if ":" in key else key
            bid, ask = await self._book(token_id)
            if bid and ask and bid > 0 and ask > 0:
                out[key] = round((bid + ask) / 2.0, 4)
            elif ask and ask > 0:
                out[key] = round(ask, 4)
            elif bid and bid > 0:
                out[key] = round(bid, 4)
        return out

    async def get_quote(self, market: Market) -> Quote:
        if not self.deep_quotes:
            return market.quote()
        try:
            book = await fetch_json(
                f"{CLOB_BASE}/book", params={"token_id": market.token_id}, ttl=10
            )
        except HttpError:
            return market.quote()

        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = max((_num(b.get("price"), 0.0) for b in bids), default=0.0) or None
        best_ask = min((_num(a.get("price"), 1.0) for a in asks), default=0.0) or None
        if not best_bid and not best_ask:
            return market.quote()
        return Quote(
            bid=clamp(best_bid, PRICE_FLOOR, PRICE_CEIL) if best_bid else market.bid,
            ask=clamp(best_ask, PRICE_FLOOR, PRICE_CEIL) if best_ask else market.ask,
            last=market.price,
            liquidity_usd=market.liquidity_usd,
            ts=utcnow(),
        )

    # ------------------------------------------------------------------- tape
    async def recent_trades(self) -> list[dict[str, Any]]:
        """The public trade tape — v2, because v1 was retired (HTTP 410).

        The dead endpoint did not fail loudly: `recent_trades` degrades to an
        empty list by design, so the whale agent simply had no Polymarket flow
        for every tick, and the retry-on-4xx spent ~2.3s per tick finding out.
        v2 wraps the rows in `{"data": [...], "pagination": {...}}` and uses
        snake_case field names.
        """
        limit = self.cfg.i("whale.lookback_trades", 150)
        try:
            raw = await fetch_json(
                f"{DATA_BASE}/v2/trades",
                params={"limit": min(limit, 500), "takerOnly": "true"},
                ttl=15,
            )
        except HttpError:
            return []
        out: list[dict[str, Any]] = []
        for trade in _as_objects(raw.get("data") if isinstance(raw, dict) else raw):
            asset = str(trade.get("token_id") or trade.get("asset") or "")
            if not asset:
                continue
            size = _num(trade.get("size"), 0.0)
            price = _num(trade.get("price"), 0.0)
            out.append(
                {
                    "key": f"{self.venue.value}:{asset}",
                    "side": str(trade.get("side", "")).upper(),
                    "price": price,
                    "qty": size,
                    "notional": size * price,
                    "wallet": str(
                        trade.get("proxy_wallet") or trade.get("proxyWallet") or ""
                    ),
                    "ts": parse_ts(trade.get("timestamp")),
                    "market": trade.get("condition_id") or trade.get("conditionId") or "",
                    "title": trade.get("title") or "",
                }
            )
        return out


# ------------------------------------------------------------------ helpers

def _as_objects(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "markets", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _json_list(value: Any) -> list[Any]:
    """Gamma returns some array fields as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_title(item: dict[str, Any]) -> str:
    events = item.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return str(events[0].get("title") or "")
    return ""


def _slug_ts(slug: str) -> int:
    """Trailing unix timestamp of an updown slug (`btc-updown-15m-1794798000`)."""
    tail = slug.rsplit("-", 1)[-1] if slug else ""
    try:
        value = int(tail)
    except ValueError:
        return 0
    return value if 1_500_000_000 < value < 2_500_000_000 else 0
