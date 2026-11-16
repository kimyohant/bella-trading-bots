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

# host -> monotonic deadline. A 429 is a request to stop asking, so retrying it
# on the next tick is the one thing guaranteed to keep us locked out.
_cooldown: dict[str, float] = {}
_MIN_COOLDOWN = 5.0


class HttpError(RuntimeError):
    """Raised when a request fails after all retries."""


class HttpStatusError(HttpError):
    """A response we can reason about, rather than a generic failure."""

    def __init__(self, message: str, status: int, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def cache_clear() -> None:
    _cache.clear()
    _cooldown.clear()


def cooling_down(url: str) -> float:
    """Seconds left before `url`'s host may be asked again (0 = go ahead)."""
    until = _cooldown.get(_host(url))
    if not until:
        return 0.0
    return max(0.0, until - time.monotonic())


def _host(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).hostname or url
    except ValueError:
        return url


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

    host = _host(url)
    wait = cooling_down(url)
    if wait > 0:
        # Fail fast instead of hammering a host that just told us to back off.
        raise HttpError(f"GET {url} skipped: {host} cooling down {wait:.0f}s")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = await _request(url, headers=headers, timeout=timeout)
            _cache[url] = (time.monotonic() + ttl, data)
            _cooldown.pop(host, None)
            return data
        except HttpStatusError as exc:
            last_error = exc
            if exc.status in (429, 503):
                # Retrying a rate limit immediately is what turns one 429 into a
                # lockout. Park the whole host and let the next tick decide.
                pause = max(exc.retry_after or 0.0, _MIN_COOLDOWN)
                _cooldown[host] = time.monotonic() + pause
                raise
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
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
            if response.status_code >= 400:
                raise HttpStatusError(
                    f"HTTP {response.status_code} for {url}",
                    response.status_code,
                    _retry_after(response.headers.get("retry-after")),
                )
            return response.text if raw else response.json()

    return await asyncio.to_thread(_urllib_get, url, merged, timeout, raw)


def _retry_after(value: Any) -> float | None:
    """Parse Retry-After, which may be seconds or an HTTP date we ignore."""
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except ValueError:
        return None


def _urllib_get(url: str, headers: dict[str, str], timeout: float, raw: bool) -> Any:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        raise HttpStatusError(
            f"HTTP {exc.code} for {url}: {body}",
            exc.code,
            _retry_after(exc.headers.get("Retry-After") if exc.headers else None),
        ) from exc
    return payload if raw else json.loads(payload)
