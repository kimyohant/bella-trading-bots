"""Venue adapter interface.

Every venue — Polymarket, pump.fun, the mock world, a CSV replay — implements
this surface. Adding a new venue means implementing three methods and
registering it in `bottrade.feeds`; nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import Market, Quote, Venue


class VenueAdapter(ABC):
    venue: Venue
    name: str = "venue"

    @abstractmethod
    async def list_markets(self) -> list[Market]:
        """Return the currently tradable markets, best candidates first."""

    async def get_quote(self, market: Market) -> Quote:
        """Top of book for one market. Defaults to the fields on `Market`."""
        return market.quote()

    async def recent_trades(self) -> list[dict[str, Any]]:
        """Normalised public trade tape, newest first.

        Shape: {"key", "side", "notional", "wallet", "ts", "price", "qty"}.
        Venues without a public tape simply return an empty list and the whale
        agent degrades to a neutral bias.
        """
        return []

    async def quote_missing(self, keys: list[str]) -> dict[str, float]:
        """Best-effort prices for positions outside the scanned universe.

        A position that drops out of the scan quota stops being marked, and an
        unmarked position freezes its P&L and can never exit honestly. Asking the
        venue for those specific keys costs one call and keeps the mark real.
        Return {market_key: price}; an empty dict means "cannot price".
        """
        return {}

    async def close(self) -> None:
        """Release any resources. Default: nothing to do."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} venue={self.venue.value}>"


class VenueError(RuntimeError):
    """Raised when a venue returns data we cannot interpret."""
