"""Report generator — turns the sqlite run log into a readable markdown report.

Reads only from the database, so it works after the bot has exited (or on a run
from last week). Outputs `<out_dir>/report_<timestamp>.md` plus a fills CSV.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .data.store import Store
from .utils import fmt_pct, fmt_price, fmt_usd, max_drawdown, parse_ts, stdev

SPARK = "▁▂▃▄▅▆▇█"


def generate_report(
    db_path: str | Path,
    out_dir: str | Path = "reports",
    title: str | None = None,
) -> dict[str, Any]:
    store = Store(db_path)
    try:
        equity_rows = store.query("SELECT ts, equity, cash, exposure FROM equity ORDER BY id")
        fill_rows = store.query("SELECT * FROM fills ORDER BY id")
        order_rows = store.query("SELECT status, COUNT(*) AS n FROM orders GROUP BY status")
        tick_rows = store.query("SELECT * FROM ticks ORDER BY id")
        signal_count = int(store.scalar("SELECT COUNT(*) FROM signals") or 0)
        market_count = int(store.scalar("SELECT COUNT(DISTINCT key) FROM markets") or 0)
        events = store.query("SELECT * FROM events ORDER BY id DESC LIMIT 20")
        rejected = _group_reasons(store.query("SELECT reason FROM rejections LIMIT 20000"))
        if not rejected:
            rejected = store.query(
                "SELECT reason, COUNT(*) AS n FROM orders WHERE status='rejected' "
                "GROUP BY reason ORDER BY n DESC LIMIT 8"
            )
    finally:
        store.close()

    metrics = _metrics(equity_rows, fill_rows)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_path / f"report_{stamp}.md"
    csv_path = out_path / f"fills_{stamp}.csv"

    _write_fills_csv(csv_path, fill_rows)
    md_path.write_text(
        _render(
            title or "bottrade paper run",
            metrics,
            equity_rows,
            fill_rows,
            tick_rows,
            signal_count,
            market_count,
            order_rows,
            rejected,
            events,
        ),
        encoding="utf-8",
    )
    return {"markdown": str(md_path), "csv": str(csv_path), "metrics": metrics}


# ------------------------------------------------------------------- metrics

def _metrics(equity_rows: list[dict[str, Any]], fills: list[dict[str, Any]]) -> dict[str, Any]:
    curve = [float(r["equity"]) for r in equity_rows if r["equity"] is not None]
    start = curve[0] if curve else 0.0
    end = curve[-1] if curve else 0.0

    returns = [
        (curve[i] / curve[i - 1] - 1.0)
        for i in range(1, len(curve))
        if curve[i - 1] > 0
    ]
    mean_ret = sum(returns) / len(returns) if returns else 0.0
    sd = stdev(returns)
    # per-tick information ratio; deliberately NOT annualised, because tick
    # spacing differs between live (15s) and replay runs
    sharpe = (mean_ret / sd) if sd > 0 else 0.0

    exits = [f for f in fills if f.get("tag") == "exit"]
    pnls = [float(f["realized_pnl"] or 0.0) for f in exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    by_venue: dict[str, dict[str, float]] = {}
    for fill in fills:
        venue = fill.get("venue") or "?"
        entry = by_venue.setdefault(venue, {"fills": 0.0, "pnl": 0.0, "fees": 0.0, "notional": 0.0})
        entry["fills"] += 1
        entry["pnl"] += float(fill["realized_pnl"] or 0.0)
        entry["fees"] += float(fill["fee"] or 0.0)
        entry["notional"] += float(fill["notional"] or 0.0)

    per_market: dict[str, dict[str, float]] = {}
    for fill in fills:
        key = fill.get("question") or fill.get("key") or "?"
        entry = per_market.setdefault(key, {"pnl": 0.0, "fills": 0.0})
        entry["pnl"] += float(fill["realized_pnl"] or 0.0)
        entry["fills"] += 1

    return {
        "start_equity": start,
        "end_equity": end,
        "pnl": end - start,
        "return_pct": (end / start - 1.0) if start else 0.0,
        "max_drawdown": max_drawdown(curve) if curve else 0.0,
        "sharpe": sharpe,
        "ticks": len(curve),
        "fills": len(fills),
        "entries": len(fills) - len(exits),
        "exits": len(exits),
        "win_rate": (len(wins) / len(exits)) if exits else 0.0,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (gross_loss / len(losses)) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win else 0.0,
        "expectancy": (sum(pnls) / len(pnls)) if pnls else 0.0,
        "fees": sum(float(f["fee"] or 0.0) for f in fills),
        "by_venue": by_venue,
        "per_market": per_market,
    }


# -------------------------------------------------------------------- render

def _render(title, m, equity_rows, fills, ticks, signal_count, market_count,
            order_rows, rejected, events) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# {title}")
    add("")
    add(f"*generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    add("")
    add("## Headline")
    add("")
    add("| metric | value |")
    add("|---|---|")
    add(f"| starting equity | {fmt_usd(m['start_equity'])} |")
    add(f"| ending equity | {fmt_usd(m['end_equity'])} |")
    add(f"| net PnL | {fmt_usd(m['pnl'])} ({fmt_pct(m['return_pct'], 2)}) |")
    add(f"| max drawdown | {fmt_pct(-m['max_drawdown'], 2)} |")
    add(f"| sharpe per tick (not annualised) | {m['sharpe']:.3f} |")
    add(f"| ticks / markets seen | {m['ticks']} / {market_count} |")
    add(f"| signals emitted | {signal_count} |")
    add(f"| fills (entry / exit) | {m['fills']} ({m['entries']} / {m['exits']}) |")
    add(f"| win rate | {fmt_pct(m['win_rate'], 1).replace('+', '')} |")
    add(f"| avg win / avg loss | {fmt_usd(m['avg_win'])} / {fmt_usd(m['avg_loss'])} |")
    pf = m["profit_factor"]
    add(f"| profit factor | {'∞' if pf == float('inf') else f'{pf:.2f}'} |")
    add(f"| expectancy per exit | {fmt_usd(m['expectancy'])} |")
    add(f"| fees paid | {fmt_usd(m['fees'])} |")
    add("")

    if equity_rows:
        curve = [float(r["equity"]) for r in equity_rows]
        add("## Equity curve")
        add("")
        add("```")
        add(_sparkline(curve, 68))
        add(f"min {fmt_usd(min(curve))}   max {fmt_usd(max(curve))}   last {fmt_usd(curve[-1])}")
        add("```")
        add("")

    if m["by_venue"]:
        add("## By venue")
        add("")
        add("| venue | fills | notional | realized PnL | fees |")
        add("|---|---|---|---|---|")
        for venue, stats in sorted(m["by_venue"].items()):
            add(
                f"| {venue} | {int(stats['fills'])} | {fmt_usd(stats['notional'])} | "
                f"{fmt_usd(stats['pnl'])} | {fmt_usd(stats['fees'])} |"
            )
        add("")

    if fills:
        add("## Trades")
        add("")
        add("| time (UTC) | venue | market | side | qty | price | fee | realized |")
        add("|---|---|---|---|---|---|---|---|")
        for fill in fills[-25:]:
            ts = parse_ts(fill.get("ts"))
            add(
                f"| {ts.strftime('%m-%d %H:%M') if ts else '—'} | {fill.get('venue')} | "
                f"{_short(fill.get('question'))} | {fill.get('side')} | "
                f"{float(fill['qty'] or 0):,.4f} | {fmt_price(float(fill['price'] or 0))} | "
                f"{fmt_usd(float(fill['fee'] or 0))} | {fmt_usd(float(fill['realized_pnl'] or 0))} |"
            )
        if len(fills) > 25:
            add(f"| … | | *{len(fills) - 25} earlier fills in the CSV* | | | | | |")
        add("")

    if m["per_market"]:
        add("## Best and worst markets")
        add("")
        ranked = sorted(m["per_market"].items(), key=lambda kv: kv[1]["pnl"], reverse=True)
        add("| market | fills | realized PnL |")
        add("|---|---|---|")
        for name, stats in ranked[:5]:
            add(f"| {_short(name, 52)} | {int(stats['fills'])} | {fmt_usd(stats['pnl'])} |")
        if len(ranked) > 5:
            add("| … | | |")
            for name, stats in ranked[-3:]:
                add(f"| {_short(name, 52)} | {int(stats['fills'])} | {fmt_usd(stats['pnl'])} |")
        add("")

    if order_rows:
        add("## Order outcomes")
        add("")
        add("| status | count |")
        add("|---|---|")
        for row in order_rows:
            add(f"| {row['status']} | {row['n']} |")
        add("")

    if rejected:
        add("### Why orders were rejected")
        add("")
        add("| reason | count |")
        add("|---|---|")
        for row in rejected:
            add(f"| {_short(row['reason'], 68)} | {row['n']} |")
        add("")

    if events:
        add("## Recent events")
        add("")
        for event in events[:10]:
            add(f"- `{event['ts']}` **{event['level']}** {event['agent']}: {event['message']}")
        add("")

    add("---")
    add("")
    add("*Paper trading only — no real orders were placed.*")
    return "\n".join(lines)


def _group_reasons(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Collapse near-identical rejection reasons into buckets."""
    buckets: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "").split(" (")[0]
        key = re.sub(r"\d+(?:\.\d+)?", "#", reason).strip()
        if key:
            buckets[key] = buckets.get(key, 0) + 1
    ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"reason": k, "n": n} for k, n in ranked]


def _sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return SPARK[3] * len(values)
    return "".join(SPARK[min(len(SPARK) - 1, int((v - low) / span * (len(SPARK) - 1)))] for v in values)


def _short(text: Any, width: int = 40) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _write_fills_csv(path: Path, fills: list[dict[str, Any]]) -> None:
    columns = ["ts", "key", "venue", "question", "side", "qty", "price", "notional",
               "fee", "slippage_bps", "realized_pnl", "tag", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for fill in fills:
            writer.writerow({c: fill.get(c) for c in columns})
