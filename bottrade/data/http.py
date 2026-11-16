"""Async JSON fetching with caching, retries and a graceful offline mode.

Uses httpx when it is installed (connection pooling, HTTP/2) and falls back to
`urllib` on a worker thread so the bot still runs on a bare Python install.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:  # optional
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

USER_AGENT = (
    "bottrade/0.1 (+paper-trading; contact: local) "
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
)

_cache: dict[str, tuple[float, Any]] = {}
_DEFAULT_TTL = 20.0


class HttpError(RuntimeError):
    """Raised when a request fails after all retries."""


def cache_clear() -> None:
    _cache.clear()


async def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    retries: int = 2,
    ttl: float = _DEFAULT_TTL,
) -> Any:
    """GET a URL and decode JSON, memoised for `ttl` seconds."""
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}{'&' if '?' in url else '?'}{query}"

    cached = _cache.get(url)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = await _request(url, headers=headers, timeout=timeout)
            _cache[url] = (time.monotonic() + ttl, data)
            return data
        except Exception as exc:  # noqa: BLE001 - we surface it after retries
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
    raise HttpError(f"GET {url} failed: {last_error}") from last_error


async def fetch_text(url: str, *, timeout: float = 15.0, retries: int = 1) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await _request(url, headers=None, timeout=timeout, raw=True)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
    raise HttpError(f"GET {url} failed: {last_error}") from last_error


async def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    """POST JSON and decode the JSON response. Used for optional LLM scoring."""
    merged = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    if headers:
        merged.update(headers)
    body = json.dumps(payload).encode("utf-8")

    if httpx is not None:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.post(url, headers=merged, content=body)
            response.raise_for_status()
            return response.json()

    return await asyncio.to_thread(_urllib_post, url, merged, body, timeout)


def _urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> Any:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise HttpError(f"POST {exc.code} for {url}: {detail}") from exc


async def _request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    raw: bool = False,
) -> Any:
    merged = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    if headers:
        merged.update(headers)

    if httpx is not None:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=merged)
            response.raise_for_status()
            return response.text if raw else response.json()

    return await asyncio.to_thread(_urllib_get, url, merged, timeout, raw)


def _urllib_get(url: str, headers: dict[str, str], timeout: float, raw: bool) -> Any:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        raise HttpError(f"HTTP {exc.code} for {url}: {body}") from exc
    return payload if raw else json.loads(payload)
