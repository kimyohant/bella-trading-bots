"""Sentiment agent — turns text into a probability nudge.

Three tiers, cheapest first, all optional:
  1. lexicon      — offline keyword scoring of the market question (default);
  2. news         — Google News RSS headlines for the question's keywords;
  3. llm          — any OpenAI-compatible chat endpoint, if you configure one.

Output is `SentimentScore` per *leg*: a bullish headline is `+` for the YES leg
and `-` for the NO leg, so downstream sizing never double-counts.
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from ..data.http import HttpError, fetch_text, post_json
from ..models import Market, SentimentScore, Venue
from ..utils import clamp, keywords, shorten
from .base import Agent, Context, register

BULLISH = {
    "win": 0.6, "wins": 0.6, "won": 0.6, "victory": 0.6, "lead": 0.45, "leads": 0.45,
    "pass": 0.55, "passes": 0.55, "passed": 0.55, "approve": 0.6, "approved": 0.6,
    "agreement": 0.5, "deal": 0.4, "surge": 0.65, "rally": 0.6, "soar": 0.65,
    "beat": 0.5, "beats": 0.5, "record": 0.4, "growth": 0.5, "strong": 0.5,
    "boost": 0.5, "confirm": 0.45, "confirmed": 0.5, "ahead": 0.4, "success": 0.6,
    "successful": 0.6, "upgrade": 0.5, "bullish": 0.7, "higher": 0.45, "rise": 0.5,
    "rises": 0.5, "gain": 0.5, "gains": 0.5, "breakout": 0.6, "support": 0.35,
    "frontrunner": 0.5, "favored": 0.5, "favorite": 0.5, "clears": 0.5, "settles": 0.35,
}

BEARISH = {
    "lose": 0.6, "loses": 0.6, "lost": 0.6, "defeat": 0.6, "fail": 0.6, "fails": 0.6,
    "failed": 0.6, "reject": 0.6, "rejected": 0.6, "delay": 0.45, "delayed": 0.45,
    "crash": 0.7, "drop": 0.5, "drops": 0.5, "plunge": 0.65, "plunges": 0.65,
    "miss": 0.5, "misses": 0.5, "cancel": 0.55, "cancelled": 0.55, "lawsuit": 0.5,
    "ban": 0.6, "banned": 0.6, "weak": 0.5, "slump": 0.6, "downgrade": 0.5,
    "halt": 0.5, "halted": 0.5, "scandal": 0.6, "bearish": 0.7, "lower": 0.45,
    "fall": 0.5, "falls": 0.5, "decline": 0.5, "risk": 0.3, "uncertain": 0.3,
    "doubt": 0.45, "stall": 0.5, "stalls": 0.5, "sued": 0.5, "probe": 0.4,
}

NEGATORS = {"not", "no", "never", "without", "denies", "deny", "isn't", "won't",
            "cannot", "can't", "unlikely", "hardly", "fails", "refuses"}

HEDGES = {"likely": 0.3, "probably": 0.25, "expected": 0.2, "may": 0.12, "might": 0.12,
          "could": 0.12, "unlikely": -0.3, "unclear": -0.15, "possible": 0.1}

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_news_cache: dict[str, tuple[float, list[str]]] = {}


@register("sentiment")
class SentimentAgent(Agent):
    name = "sentiment"

    async def run(self, ctx: Context) -> None:
        if not self.cfg.b("sentiment.enabled", True):
            return

        model = self.cfg.s("sentiment.model", "lexicon").lower()
        weight_news = self.cfg.f("sentiment.news_weight", 0.6)
        news_enabled = self.cfg.b("sentiment.news_enabled", False)
        max_queries = self.cfg.i("sentiment.max_news_queries_per_tick", 3)
        queries_used = 0

        for market in ctx.markets:
            text_score, hits = score_text(market.question)
            components: list[tuple[float, float]] = [(text_score, 0.35)]
            headlines: list[str] = []

            should_fetch = (
                news_enabled
                and queries_used < max_queries
                and market.venue is Venue.POLYMARKET
                and market.volume_24h_usd >= self.cfg.f("sentiment.news_min_volume_usd", 20_000)
            )
            if should_fetch:
                queries_used += 1
                headlines = await self._headlines(market.question)
                if headlines:
                    news_score = sum(score_text(h)[0] for h in headlines) / len(headlines)
                    components.append((news_score, weight_news))

            if model == "llm":
                llm_score = await self._llm_score(market, headlines)
                if llm_score is not None:
                    components.append((llm_score, 1.0))

            total_w = sum(w for _, w in components)
            raw = sum(s * w for s, w in components) / total_w if total_w else 0.0
            raw = _orient(raw, market)

            coverage = clamp(len(components) / 2.0, 0.35, 1.0)
            confidence = clamp(abs(raw) * 1.4 * coverage + min(hits, 4) * 0.05, 0.0, 1.0)

            ctx.sentiment[market.key] = SentimentScore(
                key=market.key,
                score=round(clamp(raw, -1.0, 1.0), 4),
                confidence=round(confidence, 4),
                samples=hits + len(headlines),
                source="llm" if model == "llm" else ("news" if headlines else "lexicon"),
                headlines=headlines[:3],
            )

        if ctx.sentiment:
            avg = sum(s.score for s in ctx.sentiment.values()) / len(ctx.sentiment)
            self.log.debug("scored %d markets, avg bias %+.3f", len(ctx.sentiment), avg)

    # ------------------------------------------------------------------- news
    async def _headlines(self, question: str) -> list[str]:
        query = " ".join(keywords(question, 5))
        if not query:
            return []
        cached = _news_cache.get(query)
        if cached and cached[0] > time.monotonic():
            return cached[1]

        url = (
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        )
        try:
            xml_text = await fetch_text(url, timeout=10.0, retries=0)
        except HttpError:
            _news_cache[query] = (time.monotonic() + 300, [])
            return []

        titles: list[str] = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.iter("item"):
                title = item.findtext("title") or ""
                if title:
                    titles.append(shorten(title, 120))
                if len(titles) >= 5:
                    break
        except ET.ParseError:
            titles = []

        _news_cache[query] = (time.monotonic() + 900, titles)
        return titles

    # -------------------------------------------------------------------- llm
    async def _llm_score(self, market: Market, headlines: list[str]) -> float | None:
        base_url = self.cfg.s("sentiment.llm_base_url", "").rstrip("/")
        model = self.cfg.s("sentiment.llm_model", "")
        api_key = os.environ.get("BOTTRADE_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not base_url or not model:
            return None

        prompt = (
            "You are a prediction-market analyst. Estimate how likely the following "
            "event is to resolve YES, using only the headline evidence given.\n\n"
            f"Question: {market.question}\n"
            f"Current market price (implied probability): {market.price:.3f}\n"
            f"Headlines:\n" + ("\n".join(f"- {h}" for h in headlines) or "- (none)")
            + "\n\nReply with JSON only: {\"probability\": <0..1>, \"confidence\": <0..1>}"
        )
        try:
            response = await post_json(
                f"{base_url}/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 60,
                },
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            )
        except Exception:  # noqa: BLE001 - LLM is best-effort
            return None

        content = _extract_content(response)
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return None
        try:
            import json

            payload: dict[str, Any] = json.loads(match.group(0))
            probability = float(payload.get("probability", 0.5))
        except (ValueError, TypeError):
            return None

        # convert probability-vs-price into a -1..1 score
        return clamp((probability - market.price) / 0.25, -1.0, 1.0)


# ------------------------------------------------------------------ pure logic

def score_text(text: str) -> tuple[float, int]:
    """Lexicon sentiment of a short string. Returns (score in -1..1, hit count)."""
    if not text:
        return 0.0, 0
    tokens = _WORD.findall(text.lower())
    total = 0.0
    hits = 0
    negate = False

    for token in tokens:
        if token in NEGATORS:
            negate = True
            continue
        weight = BULLISH.get(token)
        if weight is None:
            weight = -BEARISH[token] if token in BEARISH else None
        if weight is None:
            hedge = HEDGES.get(token)
            if hedge is not None:
                total += hedge * (-1 if negate else 1)
                hits += 1
            negate = False
            continue
        total += weight * (-1 if negate else 1)
        hits += 1
        negate = False

    if hits == 0:
        return 0.0, 0
    # squash: 3 strong hits ≈ ±0.8
    score = total / (1.0 + abs(total) * 0.55)
    return clamp(score, -1.0, 1.0), hits


def _orient(score: float, market: Market) -> float:
    """Flip the score for the NO leg so `+` always means 'this leg wins'."""
    outcome = (market.outcome or "").strip().lower()
    if market.venue is Venue.POLYMARKET and outcome in {"no", "n"}:
        return -score
    return score


def _extract_content(response: Any) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""
