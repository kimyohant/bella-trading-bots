"""Command line interface.

    python -m bottrade run        # paper-trade against live or synthetic markets
    python -m bottrade scan       # inspect the tradable universe right now
    python -m bottrade backtest   # replay a CSV of historical quotes
    python -m bottrade serve      # web dashboard for a live or finished run
    python -m bottrade report     # turn a run database into a markdown report
    python -m bottrade agents     # list every registered agent
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .agents.base import build_agents, registry
from .agents.orchestrator import Orchestrator
from .backtest import format_backtest, run_backtest, write_demo_csv
from .config import Config
from .data.store import Store
from .feeds import build_feed
from .log import setup_logging
from .report import generate_report
from .utils import fmt_pct, fmt_price, fmt_usd, shorten
from .venues.paper import PaperBroker

BANNER = r"""
  ┌───────────────────────────────────────────────┐
  │  bottrade · paper trading only                │
  │  polymarket + pump.fun · multi-agent pipeline │
  └───────────────────────────────────────────────┘
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bottrade",
        description="Paper-trading bot for Polymarket and pump.fun.",
    )
    parser.add_argument("--version", action="version", version=f"bottrade {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- run
    run = sub.add_parser("run", help="run the agent pipeline")
    run.add_argument("-c", "--config", default=None, help="path to config.toml")
    run.add_argument("--source", choices=["live", "mock", "replay"], default=None,
                     help="override run.source")
    run.add_argument("--ticks", type=int, default=None, help="stop after N ticks")
    run.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    run.add_argument("--cash", type=float, default=None, help="override starting cash")
    run.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING")
    run.add_argument("--no-store", action="store_true", help="do not write the sqlite log")
    run.add_argument("--report", action="store_true", help="write a markdown report at the end")
    run.add_argument("--quiet", action="store_true", help="suppress the banner")

    # --------------------------------------------------------------- scan
    scan = sub.add_parser("scan", help="show the current tradable universe")
    scan.add_argument("-c", "--config", default=None)
    scan.add_argument("--source", choices=["live", "mock", "replay"], default=None)
    scan.add_argument("--venue", choices=["polymarket", "pumpfun", "all"], default="all")
    scan.add_argument("--limit", type=int, default=15)

    # ----------------------------------------------------------- backtest
    bt = sub.add_parser("backtest", help="replay historical quotes")
    bt.add_argument("-c", "--config", default=None)
    bt.add_argument("--csv", default=None, help="CSV of quotes (ts,key,price[,bid,ask])")
    bt.add_argument("--demo", action="store_true", help="generate a synthetic CSV first")
    bt.add_argument("--ticks", type=int, default=240, help="rows per token for --demo")
    bt.add_argument("--cash", type=float, default=None)

    # -------------------------------------------------------------- serve
    srv = sub.add_parser("serve", help="start the web monitor dashboard")
    srv.add_argument("-c", "--config", default=None)
    srv.add_argument("--db", default=None, help="sqlite run log to display")
    srv.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    srv.add_argument("--port", type=int, default=8787)
    srv.add_argument("--source", choices=["live", "mock", "replay"], default=None)
    srv.add_argument("--ticks", type=int, default=None, help="stop the bot after N ticks")
    srv.add_argument("--no-run", action="store_true",
                     help="read-only: monitor an existing run instead of starting a new one")
    srv.add_argument("--interval", type=float, default=2.0, help="push interval, seconds")
    srv.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    srv.add_argument("--log-level", default=None)

    # ------------------------------------------------------------- report
    rep = sub.add_parser("report", help="build a markdown report from a run database")
    rep.add_argument("-c", "--config", default=None)
    rep.add_argument("--db", default=None)
    rep.add_argument("--out", default=None)

    # ------------------------------------------------------------- agents
    sub.add_parser("agents", help="list registered agents")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(getattr(args, "config", None))

    if args.command == "run":
        return _cmd_run(args, cfg)
    if args.command == "scan":
        return _cmd_scan(args, cfg)
    if args.command == "backtest":
        return _cmd_backtest(args, cfg)
    if args.command == "serve":
        return _cmd_serve(args, cfg)
    if args.command == "report":
        return _cmd_report(args, cfg)
    if args.command == "agents":
        return _cmd_agents()
    return 1


# ------------------------------------------------------------------- commands

def _cmd_run(args, cfg: Config) -> int:
    level = args.log_level or cfg.s("run.log_level", "INFO")
    setup_logging(level, cfg.s("run.log_file", "") or None)
    if not args.quiet:
        print(BANNER)
        print(f"  source      : {args.source or cfg.s('run.source', 'mock')}")
        print(f"  agents      : {', '.join(cfg.list('agents.enabled'))}")
        print(f"  cash        : {fmt_usd(args.cash or cfg.f('portfolio.starting_cash', 1000))}")
        print()

    if args.source:
        cfg.set("run.source", args.source)
    if args.cash is not None:
        cfg.set("portfolio.starting_cash", args.cash)
    if args.log_level:
        setup_logging(args.log_level, cfg.s("run.log_file", "") or None)

    feed = build_feed(cfg)
    broker = PaperBroker.from_config(cfg)
    store = None if args.no_store else Store(cfg.s("run.state_db", "state/bottrade.sqlite3"))

    orch = Orchestrator(cfg, feed=feed, agents=build_agents(cfg), store=store, broker=broker)

    try:
        summary = asyncio.run(orch.run(max_ticks=args.ticks, duration_s=args.duration))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        if store:
            store.close()

    _print_summary(summary)

    if args.report and not args.no_store:
        result = generate_report(
            cfg.s("run.state_db", "state/bottrade.sqlite3"),
            cfg.s("report.out_dir", "reports"),
        )
        print(f"\n  report -> {result['markdown']}")
        print(f"  fills  -> {result['csv']}")
    return 0


def _cmd_scan(args, cfg: Config) -> int:
    setup_logging("WARNING")
    if args.source:
        cfg.set("run.source", args.source)

    feed = build_feed(cfg)
    agents = build_agents(cfg)

    async def _once():
        markets = await feed.step()
        try:
            from .agents.base import Context
            from .models import Venue

            ctx = Context(cfg=cfg, broker=PaperBroker.from_config(cfg),
                          log=__import__("logging").getLogger("bottrade.scan"))
            ctx.markets = markets or []
            adapters = getattr(feed, "adapters", None) or getattr(feed, "venues", None) or {}
            ctx.adapters = adapters
            scanner = next((a for a in agents if a.name == "scanner"), None)
            if scanner:
                await scanner.run(ctx)
            return ctx.markets
        finally:
            await feed.close()

    markets = asyncio.run(_once())
    if not markets:
        print("no tradable markets returned (network down, or filters too tight)")
        return 1

    from .models import CRYPTO_PROFILE

    if args.venue != "all":
        markets = [m for m in markets if m.venue.value == args.venue][: args.limit]
    else:
        # crypto trades on ~$1k of liquidity, so ranked by volume it never makes
        # the cut — show it after the main list instead of hiding it
        crypto = [m for m in markets if m.meta.get("profile") == CRYPTO_PROFILE]
        regular = [m for m in markets if m.meta.get("profile") != CRYPTO_PROFILE]
        markets = regular[: args.limit] + crypto[:6]

    print(f"\n  {len(markets)} markets\n")
    header = f"  {'venue':<11} {'price':>10} {'spread':>7} {'liq':>10} {'vol24h':>10}  market"
    print(header)
    print("  " + "─" * (len(header) + 20))
    for m in markets:
        print(
            f"  {m.venue.value:<11} {fmt_price(m.price):>10} "
            f"{m.spread_pct * 100:>6.2f}% {fmt_usd(m.liquidity_usd):>10} "
            f"{fmt_usd(m.volume_24h_usd):>10}  {shorten(m.question, 56)}"
        )
    print()
    return 0


def _cmd_backtest(args, cfg: Config) -> int:
    setup_logging("WARNING")

    csv_path = args.csv
    if args.demo or not csv_path:
        csv_path = str(Path(cfg.s("run.state_db", "state/bottrade.sqlite3")).parent
                       / "demo_prices.csv")
        print(f"  generating synthetic price history -> {csv_path}")
        write_demo_csv(csv_path, ticks=args.ticks)

    try:
        result = asyncio.run(run_backtest(cfg, csv_path, starting_cash=args.cash))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    print(format_backtest(result))
    return 0


def _cmd_serve(args, cfg: Config) -> int:
    setup_logging(args.log_level or cfg.s("run.log_level", "INFO"),
                  cfg.s("run.log_file", "") or None)
    if args.source:
        cfg.set("run.source", args.source)

    from .web.server import run_server

    try:
        run_server(
            cfg,
            db_path=args.db or cfg.s("run.state_db", "state/bottrade.sqlite3"),
            host=args.host,
            port=args.port,
            run_bot=not args.no_run,
            max_ticks=args.ticks,
            interval=args.interval,
            open_browser=args.open,
        )
    except OSError as exc:
        print(f"error: cannot start server on {args.host}:{args.port} — {exc}")
        return 1
    return 0


def _cmd_report(args, cfg: Config) -> int:
    db = args.db or cfg.s("run.state_db", "state/bottrade.sqlite3")
    out = args.out or cfg.s("report.out_dir", "reports")
    if not Path(db).exists():
        print(f"error: database not found: {db}")
        return 1

    result = generate_report(db, out)
    m = result["metrics"]
    print("\n  report written")
    print("  " + "─" * 46)
    print(f"  equity      {fmt_usd(m['start_equity'])} -> {fmt_usd(m['end_equity'])}")
    print(f"  return      {fmt_pct(m['return_pct'], 2)}")
    print(f"  fills       {m['fills']} ({m['exits']} closed)")
    print(f"  win rate    {m['win_rate'] * 100:.1f}%")
    print("  " + "─" * 46)
    print(f"  markdown    {result['markdown']}")
    print(f"  fills csv   {result['csv']}")
    return 0


def _cmd_agents() -> int:
    reg = registry()
    print(f"\n  {len(reg)} registered agents\n")
    for name in ("guard", "scanner", "sentiment", "whale", "crypto", "signal",
                 "risk", "executor", "portfolio"):
        cls = reg.pop(name, None)
        if cls:
            print(f"  {name:<11} {_summary(cls)}")
    for name, cls in sorted(reg.items()):
        print(f"  {name:<11} {_summary(cls)}")
    print()
    return 0


def _summary(cls) -> str:
    doc = (cls.__doc__ or "").strip().splitlines()
    return doc[0] if doc else ""


def _print_summary(summary: dict) -> None:
    print("\n  run summary")
    print("  " + "─" * 52)
    print(f"  ticks              {summary['ticks']}  ({summary['elapsed_s']}s)")
    print(f"  starting cash      {fmt_usd(summary['starting_cash'])}")
    print(f"  ending equity      {fmt_usd(summary['equity'])}")
    print(f"  net PnL            {fmt_usd(summary['pnl'])}  ({summary['return_pct'] * 100:+.2f}%)")
    print(f"  realized/unreal    {fmt_usd(summary['realized_pnl'])} / {fmt_usd(summary['unrealized_pnl'])}")
    print(f"  fills              {summary['fills']} ({summary['entries']} in, {summary['exits']} out)")
    print(f"  wins / losses      {summary['wins']} / {summary['losses']}")
    print(f"  fees paid          {fmt_usd(summary['fees_paid'])}")
    print(f"  open positions     {summary['open_positions']}")
    if summary["notes"]:
        print("  notes")
        for note in summary["notes"][:6]:
            print(f"    - {note}")
    print("  " + "─" * 52)
    print("  paper trading only — no real orders were placed")


if __name__ == "__main__":
    sys.exit(main())
