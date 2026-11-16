"""HTTP monitor server — stdlib only.

Serves a single-page dashboard plus a small JSON API:

    GET /               dashboard (ui.html)
    GET /api/snapshot   full state as JSON (polling fallback)
    GET /api/stream     server-sent events, one snapshot per `interval` seconds
    GET /api/health     liveness + db info

Two modes:
  * `run_bot=True`  — starts the pipeline in a background thread and streams live
  * `run_bot=False` — read-only view over an existing sqlite run log

Reads happen on short-lived read-only connections so the bot's writer never blocks.
Binds to 127.0.0.1 by default: this is a local monitor, not a public service.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..config import Config
from ..utils import parse_ts

UI_PATH = Path(__file__).with_name("ui.html")
MAX_CURVE_POINTS = 320


# --------------------------------------------------------------------- snapshot

def build_snapshot(db_path: str | Path, cfg: Config | None = None,
                   meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the run log and shape it for the dashboard. Never raises."""
    meta = dict(meta or {})
    path = Path(db_path)
    empty = {
        "ok": False,
        "ts": time.time(),
        "meta": meta,
        "equity": {},
        "positions": [],
        "fills": [],
        "signals": [],
        "rejections": [],
        "venues": [],
        "events": [],
        "health": [],
        "deadman": False,
        "stats": {},
    }
    if not path.exists():
        meta["error"] = f"database not found: {path}"
        empty["meta"] = meta
        return empty

    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=3.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:  # pragma: no cover - locked/corrupt file
        meta["error"] = f"cannot open database: {exc}"
        empty["meta"] = meta
        return empty

    try:
        equity_rows = _rows(conn, "SELECT ts, equity, cash, exposure, realized_pnl, "
                                  "unrealized_pnl FROM equity ORDER BY id")
        tick_row = _one(conn, "SELECT * FROM ticks ORDER BY id DESC LIMIT 1")
        # exactly one tick's snapshot: several ticks can share a wall-clock second
        position_rows = _rows(
            conn,
            "SELECT * FROM positions WHERE tick = (SELECT MAX(tick) FROM positions) "
            "ORDER BY market_value DESC",
        )
        fill_rows = _rows(conn, "SELECT * FROM fills ORDER BY id DESC LIMIT 40")
        signal_rows = _rows(conn, "SELECT * FROM signals ORDER BY id DESC LIMIT 24")
        rejection_rows = _rows(conn, "SELECT reason FROM rejections LIMIT 20000")
        venue_rows = _rows(
            conn,
            "SELECT venue, COUNT(*) AS fills, "
            "SUM(COALESCE(notional,0)) AS notional, "
            "SUM(COALESCE(realized_pnl,0)) AS pnl, "
            "SUM(COALESCE(fee,0)) AS fees "
            "FROM fills GROUP BY venue ORDER BY venue",
        )
        event_rows = _rows(conn, "SELECT * FROM events ORDER BY id DESC LIMIT 8")
        health_rows = _rows(
            conn,
            "SELECT venue, ok, miss, stale_hours, note FROM health "
            "WHERE tick = (SELECT MAX(tick) FROM health) ORDER BY venue",
        )
        totals = _one(
            conn,
            "SELECT COUNT(*) AS fills, "
            "SUM(CASE WHEN tag='exit' THEN 1 ELSE 0 END) AS exits, "
            "SUM(CASE WHEN tag='exit' AND realized_pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            "SUM(CASE WHEN tag='exit' AND realized_pnl < 0 THEN 1 ELSE 0 END) AS losses, "
            "SUM(COALESCE(fee,0)) AS fees FROM fills",
        ) or {}
        signal_count = _scalar(conn, "SELECT COUNT(*) FROM signals")
        market_count = _scalar(conn, "SELECT COUNT(DISTINCT key) FROM markets")
    finally:
        conn.close()

    curve = [float(r["equity"]) for r in equity_rows if r["equity"] is not None]
    start_equity = curve[0] if curve else 0.0
    last = equity_rows[-1] if equity_rows else {}
    current = float(last.get("equity") or 0.0)

    wins = int(totals.get("wins") or 0)
    losses = int(totals.get("losses") or 0)
    exits = int(totals.get("exits") or 0)

    meta.update(
        {
            "db": str(path),
            "mode": cfg.s("run.source", "?") if cfg else meta.get("mode", "?"),
            "tick": int(tick_row["tick_no"]) if tick_row else 0,
            "markets": int(market_count or 0),
            "signals_total": int(signal_count or 0),
        }
    )

    return {
        "ok": True,
        "ts": time.time(),
        "meta": meta,
        "equity": {
            "now": current,
            "start": start_equity,
            "pnl": current - start_equity,
            "return_pct": (current / start_equity - 1.0) if start_equity else 0.0,
            "cash": float(last.get("cash") or 0.0),
            "exposure": float(last.get("exposure") or 0.0),
            "realized": float(last.get("realized_pnl") or 0.0),
            "unrealized": float(last.get("unrealized_pnl") or 0.0),
            "fees": float(totals.get("fees") or 0.0),
            "max_dd": _max_drawdown(curve),
            "curve": _downsample(equity_rows, MAX_CURVE_POINTS),
        },
        "positions": [_clean(r) for r in position_rows],
        "fills": [_clean(r) for r in fill_rows],
        "signals": [_signal(r) for r in signal_rows],
        "rejections": _group_rejections(rejection_rows),
        "venues": [
            {
                "venue": r["venue"],
                "fills": r["fills"],
                "notional": round(float(r["notional"] or 0), 2),
                "pnl": round(float(r["pnl"] or 0), 2),
                "fees": round(float(r["fees"] or 0), 2),
            }
            for r in venue_rows
        ],
        "events": [_clean(r) for r in event_rows],
        "health": [
            {
                "venue": r["venue"],
                "ok": bool(r["ok"]),
                "miss": int(r["miss"] or 0),
                "stale_hours": round(float(r["stale_hours"] or 0.0), 2),
                "note": r["note"] or "",
            }
            for r in health_rows
        ],
        "deadman": bool(health_rows) and not any(r["ok"] for r in health_rows),
        "stats": {
            "fills": int(totals.get("fills") or 0),
            "exits": exits,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / exits) if exits else 0.0,
            "open_positions": len(position_rows),
        },
    }


def _group_rejections(rows: list[dict[str, Any]], limit: int = 9) -> list[dict[str, Any]]:
    """Collapse near-identical reasons ("spread 8.3% > 6.0%") into one bucket."""
    buckets: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "").split(" (")[0]
        key = re.sub(r"\d+(?:\.\d+)?", "#", reason).strip()
        if not key:
            continue
        buckets[key] = buckets.get(key, 0) + 1
    ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"reason": key, "n": n} for key, n in ranked]


def _rows(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    try:
        return [dict(r) for r in conn.execute(sql).fetchall()]
    except sqlite3.Error:
        return []


def _one(conn: sqlite3.Connection, sql: str) -> dict[str, Any] | None:
    rows = _rows(conn, sql)
    return rows[0] if rows else None


def _scalar(conn: sqlite3.Connection, sql: str) -> Any:
    try:
        row = conn.execute(sql).fetchone()
        return row[0] if row else 0
    except sqlite3.Error:
        return 0


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "id":
            continue
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def _signal(row: dict[str, Any]) -> dict[str, Any]:
    out = _clean(row)
    try:
        out["reasons"] = json.loads(row.get("reasons") or "[]")
    except (ValueError, TypeError):
        out["reasons"] = []
    return out


def _downsample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) <= limit:
        picked = rows
    else:
        step = len(rows) / limit
        picked = [rows[min(int(i * step), len(rows) - 1)] for i in range(limit)]
        picked[-1] = rows[-1]
    out = []
    for row in picked:
        ts = parse_ts(row.get("ts"))
        out.append(
            {
                "t": ts.timestamp() if ts else 0.0,
                "v": round(float(row.get("equity") or 0.0), 2),
            }
        )
    return out


def _max_drawdown(curve: list[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


# ----------------------------------------------------------------------- server

class MonitorState:
    """Shared between the bot thread and the request handlers."""

    def __init__(self, cfg: Config, db_path: str | Path):
        self.cfg = cfg
        self.db_path = str(db_path)
        self.running = False
        self.started = time.time()
        self.error: str | None = None
        self.lock = threading.Lock()

    def meta(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "uptime_s": round(time.time() - self.started, 1),
            "error": self.error,
            "mode": self.cfg.s("run.source", "?"),
            "agents": self.cfg.list("agents.enabled"),
        }


def make_handler(state: MonitorState, interval: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "bottrade-monitor/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence per-request noise
            return

        # ------------------------------------------------------------- routing
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_file()
            elif path == "/api/snapshot":
                self._send_json(build_snapshot(state.db_path, state.cfg, state.meta()))
            elif path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "running": state.running,
                        "db": state.db_path,
                        "uptime_s": round(time.time() - state.started, 1),
                    }
                )
            elif path == "/api/stream":
                self._stream()
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404, "not found")

        # ------------------------------------------------------------ responses
        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self) -> None:
            try:
                body = UI_PATH.read_bytes()
            except OSError:
                self.send_error(500, "ui.html missing")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    payload = build_snapshot(state.db_path, state.cfg, state.meta())
                    chunk = f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def _start_bot_thread(state: MonitorState, max_ticks: int | None) -> threading.Thread:
    def target() -> None:
        from ..agents.base import build_agents
        from ..agents.orchestrator import Orchestrator
        from ..data.store import Store
        from ..feeds import build_feed

        try:
            # The store must live on this thread: sqlite connections are thread-bound.
            store = Store(state.db_path)
            orch = Orchestrator(
                state.cfg,
                feed=build_feed(state.cfg),
                agents=build_agents(state.cfg),
                store=store,
            )
            asyncio.run(orch.run(max_ticks=max_ticks))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            state.error = f"{type(exc).__name__}: {exc}"
        finally:
            state.running = False

    thread = threading.Thread(target=target, name="bottrade-bot", daemon=True)
    thread.start()
    return thread


def run_server(
    cfg: Config,
    db_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    run_bot: bool = True,
    max_ticks: int | None = None,
    interval: float = 2.0,
    open_browser: bool = False,
) -> None:
    """Start the monitor (and optionally the bot) and block until Ctrl+C."""
    db_path = db_path or cfg.s("run.state_db", "state/bottrade.sqlite3")
    state = MonitorState(cfg, db_path)

    if run_bot:
        # A mock run has no natural pacing; without this the bot burns thousands of
        # ticks before the browser has finished its first paint.
        if cfg.s("run.source", "") == "mock" and cfg.get("run.mock_tick_seconds") is None:
            cfg.set("run.mock_tick_seconds", max(0.4, interval * 0.5))
        state.running = True
        _start_bot_thread(state, max_ticks)

    handler = make_handler(state, interval)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True

    url = f"http://{host}:{port}/"
    print(f"\n  bottrade monitor  ->  {url}")
    print(f"  mode              :  {'running bot' if run_bot else 'read-only'} "
          f"({cfg.s('run.source', '?')})")
    print(f"  database          :  {db_path}")
    print("  press Ctrl+C to stop\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping monitor…")
    finally:
        httpd.shutdown()
        httpd.server_close()
        state.running = False
