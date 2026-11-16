"""Backtester — replays a CSV of historical quotes through the real pipeline.

Because `ReplayFeed` implements the same interface as the live and mock feeds, the
backtest runs the *same* signal, risk and execution code as production. No
separate "backtest strategy" that silently drifts from the live one.

CSV format (one row per quote, any number of tokens):
    ts,key,price[,bid,ask,question,venue,liquidity_usd,volume_24h_usd]
    key = "polymarket:<token_id>" or "pumpfun:<mint>"
"""

from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path
from typing import Any

from .agents.base import build_agents
from .agents.orchestrator import Orchestrator
from .config import Config
from .feeds import ReplayFeed
from .utils import fmt_usd, max_drawdown, stdev, utcnow
from .venues.paper import PaperBroker

BACKTEST_AGENTS = ["scanner", "signal", "risk", "executor", "portfolio"]


def write_demo_csv(path: str | Path, ticks: int = 240, seed: int = 11) -> Path:
    """Generate a synthetic price history so `backtest --demo` works offline."""
    from .venues.mock import MockWorld

    from .models import Venue

    cfg = Config({"run": {"seed": seed}})
    world = MockWorld(cfg, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start = utcnow() - timedelta(minutes=ticks)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "key", "price", "bid", "ask", "question",
                         "liquidity_usd", "volume_24h_usd"])
        for step in range(ticks):
            world.advance()
            ts = (start + timedelta(minutes=step)).isoformat(timespec="seconds")
            for venue in (Venue.POLYMARKET, Venue.PUMPFUN):
                for market in world.markets(venue):
                    writer.writerow([
                        ts, market.key, f"{market.price:.8f}",
                        f"{market.bid:.8f}", f"{market.ask:.8f}",
                        market.question,
                        f"{market.liquidity_usd:.2f}", f"{market.volume_24h_usd:.2f}",
                    ])
    return path


async def run_backtest(
    cfg: Config,
    csv_path: str | Path,
    starting_cash: float | None = None,
) -> dict[str, Any]:
    """Replay a CSV and return performance metrics."""
    cfg = Config(cfg.as_dict())  # local copy; never mutate the caller's config
    cfg.set("agents.enabled", BACKTEST_AGENTS)

    feed = ReplayFeed(cfg, csv_path)
    if len(feed) == 0:
        raise ValueError(f"no usable rows in {csv_path}")

    broker = PaperBroker.from_config(cfg)
    if starting_cash is not None:
        broker.cash = starting_cash
        broker.starting_cash = starting_cash

    orch = Orchestrator(cfg, feed=feed, agents=build_agents(cfg), store=None, broker=broker)
    summary = await orch.run()

    curve: list[float] = orch.ctx.state.get("equity_curve", [])
    fills = broker.fills
    exits = [f for f in fills if f.tag == "exit"]
    pnls = [f.realized_pnl for f in exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))

    returns = [
        curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve)) if curve[i - 1] > 0
    ]
    sd = stdev(returns)
    mean_ret = sum(returns) / len(returns) if returns else 0.0

    metrics = {
        "rows": len(feed),
        "ticks": summary["ticks"],
        "starting_cash": summary["starting_cash"],
        "equity": summary["equity"],
        "pnl": summary["pnl"],
        "return_pct": summary["return_pct"],
        "max_drawdown": max_drawdown(curve) if curve else 0.0,
        "sharpe": (mean_ret / sd) if sd > 0 else 0.0,
        "fills": len(fills),
        "exits": len(exits),
        "win_rate": (len(wins) / len(exits)) if exits else 0.0,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (gross_loss / len(losses)) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss else (float("inf") if gross_win else 0.0),
        "fees": broker.fees_paid,
        "open_positions": len(broker.positions),
    }
    return {"summary": summary, "metrics": metrics, "curve": curve}


def format_backtest(result: dict[str, Any]) -> str:
    m = result["metrics"]
    pf = m["profit_factor"]
    lines = [
        "",
        "  backtest result",
        "  " + "─" * 46,
        f"  quotes replayed     {m['rows']:,} rows / {m['ticks']} ticks",
        f"  starting capital    {fmt_usd(m['starting_cash'])}",
        f"  ending equity       {fmt_usd(m['equity'])}",
        f"  net PnL             {fmt_usd(m['pnl'])}  ({m['return_pct'] * 100:+.2f}%)",
        f"  max drawdown        {m['max_drawdown'] * 100:.2f}%",
        f"  sharpe per tick     {m['sharpe']:.3f}  (not annualised)",
        f"  fills               {m['fills']}  ({m['exits']} closed)",
        f"  win rate            {m['win_rate'] * 100:.1f}%",
        f"  avg win / loss      {fmt_usd(m['avg_win'])} / {fmt_usd(m['avg_loss'])}",
        f"  profit factor       {'inf' if pf == float('inf') else f'{pf:.2f}'}",
        f"  fees paid           {fmt_usd(m['fees'])}",
        f"  open positions      {m['open_positions']}",
        "  " + "─" * 46,
    ]
    return "\n".join(lines)
