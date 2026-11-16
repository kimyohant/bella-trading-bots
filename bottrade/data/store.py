"""SQLite persistence: every tick, signal, order, fill and equity point.

The report generator reads from this database, so a run is fully replayable.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..utils import iso, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tick_no INTEGER,
    equity REAL,
    cash REAL,
    exposure REAL,
    realized_pnl REAL,
    unrealized_pnl REAL,
    open_positions INTEGER,
    n_markets INTEGER,
    n_signals INTEGER,
    n_orders INTEGER,
    n_fills INTEGER
);

CREATE TABLE IF NOT EXISTS markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    key TEXT NOT NULL,
    venue TEXT,
    question TEXT,
    price REAL,
    bid REAL,
    ask REAL,
    liquidity_usd REAL,
    volume_24h_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_markets_key ON markets(key);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    key TEXT NOT NULL,
    venue TEXT,
    question TEXT,
    side TEXT,
    price REAL,
    fair_price REAL,
    edge REAL,
    confidence REAL,
    reasons TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    key TEXT NOT NULL,
    venue TEXT,
    question TEXT,
    side TEXT,
    qty REAL,
    price REAL,
    notional REAL,
    status TEXT,
    tag TEXT,
    reason TEXT,
    edge REAL,
    confidence REAL
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    key TEXT NOT NULL,
    venue TEXT,
    question TEXT,
    side TEXT,
    qty REAL,
    price REAL,
    notional REAL,
    fee REAL,
    slippage_bps REAL,
    realized_pnl REAL,
    tag TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    equity REAL,
    cash REAL,
    exposure REAL,
    realized_pnl REAL,
    unrealized_pnl REAL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT,
    agent TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER,
    ts TEXT NOT NULL,
    key TEXT NOT NULL,
    venue TEXT,
    question TEXT,
    qty REAL,
    avg_price REAL,
    last_price REAL,
    market_value REAL,
    unrealized_pnl REAL,
    return_pct REAL,
    age_hours REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_tick ON positions(tick);

CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tick INTEGER,
    agent TEXT,
    key TEXT,
    question TEXT,
    reason TEXT
);
"""

# Insert column order per table (the autoincrement `id` is never supplied).
COLUMNS: dict[str, tuple[str, ...]] = {
    "ticks": ("ts", "tick_no", "equity", "cash", "exposure", "realized_pnl",
              "unrealized_pnl", "open_positions", "n_markets", "n_signals",
              "n_orders", "n_fills"),
    "markets": ("ts", "key", "venue", "question", "price", "bid", "ask",
                "liquidity_usd", "volume_24h_usd"),
    "signals": ("ts", "key", "venue", "question", "side", "price", "fair_price",
                "edge", "confidence", "reasons"),
    "orders": ("ts", "key", "venue", "question", "side", "qty", "price", "notional",
               "status", "tag", "reason", "edge", "confidence"),
    "fills": ("ts", "key", "venue", "question", "side", "qty", "price", "notional",
              "fee", "slippage_bps", "realized_pnl", "tag", "reason"),
    "equity": ("ts", "equity", "cash", "exposure", "realized_pnl", "unrealized_pnl"),
    "events": ("ts", "level", "agent", "message"),
    "positions": ("tick", "ts", "key", "venue", "question", "qty", "avg_price",
                  "last_price", "market_value", "unrealized_pnl", "return_pct",
                  "age_hours"),
    "rejections": ("ts", "tick", "agent", "key", "question", "reason"),
}


class Store:
    """Write-through sqlite log. Safe to run with `path=':memory:'` in tests."""

    def __init__(self, path: str | Path = "state/bottrade.sqlite3"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=15.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        if self.path != ":memory:":
            # WAL lets the monitor read while the bot is writing.
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error:  # pragma: no cover - exotic filesystems
                pass

    # ------------------------------------------------------------------ writes
    def record_tick(self, tick_no: int, equity: float, cash: float, exposure: float,
                    realized: float, unrealized: float, open_positions: int,
                    n_markets: int, n_signals: int, n_orders: int, n_fills: int) -> None:
        self._insert(
            "ticks",
            (iso(utcnow()), tick_no, equity, cash, exposure, realized, unrealized,
             open_positions, n_markets, n_signals, n_orders, n_fills),
        )

    def record_markets(self, markets: Iterable[Any]) -> None:
        ts = iso(utcnow())
        rows = [
            (ts, m.key, m.venue.value, m.question, m.price, m.bid, m.ask,
             m.liquidity_usd, m.volume_24h_usd)
            for m in markets
        ]
        self._insert_many("markets", rows)

    def record_signal(self, signal: Any) -> None:
        self._insert(
            "signals",
            (iso(utcnow()), signal.key, signal.market.venue.value, signal.market.question,
             signal.side.value, signal.entry_price, signal.fair_price, signal.edge,
             signal.confidence, json.dumps(signal.reasons, ensure_ascii=False)),
        )

    def record_order(self, order: Any, status: str, reason: str = "") -> None:
        self._insert(
            "orders",
            (iso(utcnow()), order.key, order.market.venue.value, order.market.question,
             order.side.value, order.qty, order.limit_price, order.notional,
             status, order.tag, reason or order.reason, order.edge, order.confidence),
        )

    def record_fill(self, fill: Any) -> None:
        self._insert(
            "fills",
            (iso(fill.ts), fill.key, fill.venue.value, fill.question, fill.side.value,
             fill.qty, fill.price, fill.notional, fill.fee, fill.slippage_bps,
             fill.realized_pnl, fill.tag, fill.reason),
        )

    def record_equity(self, equity: float, cash: float, exposure: float,
                      realized: float, unrealized: float) -> None:
        self._insert("equity", (iso(utcnow()), equity, cash, exposure, realized, unrealized))

    def record_event(self, level: str, agent: str, message: str) -> None:
        self._insert("events", (iso(utcnow()), level, agent, message))

    def record_positions(self, positions: Iterable[Any], tick: int = 0) -> None:
        """Snapshot open positions once per tick so the monitor can show them live."""
        ts = iso(utcnow())
        rows = [
            (tick, ts, p.key, p.venue.value, p.question, round(p.qty, 8),
             round(p.avg_price, 10), round(p.last_price, 10),
             round(p.market_value, 4), round(p.unrealized_pnl, 4),
             round(p.return_pct, 6), round(p.age_hours, 4))
            for p in positions
        ]
        self._insert_many("positions", rows)

    def record_rejections(self, rejections: Iterable[dict[str, Any]], tick: int = 0) -> None:
        """Persist why orders were refused — the most useful audit trail there is."""
        ts = iso(utcnow())
        rows = [
            (ts, tick, r.get("agent", ""), r.get("key", ""), r.get("question", ""),
             str(r.get("reason", ""))[:300])
            for r in rejections
        ]
        self._insert_many("rejections", rows)

    # ------------------------------------------------------------------- reads
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cursor = self._conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def scalar(self, sql: str, params: tuple = (), default: Any = 0) -> Any:
        row = self._conn.execute(sql, params).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]

    def table_names(self) -> list[str]:
        return [r["name"] for r in self.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in self.table_names():
            out[name] = int(self.scalar(f"SELECT COUNT(*) FROM {name}") or 0)
        return out

    # ------------------------------------------------------------------ house
    def _insert(self, table: str, values: tuple) -> None:
        cols = COLUMNS[table]
        placeholders = ",".join("?" * len(cols))
        self._conn.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", values
        )
        self._conn.commit()

    def _insert_many(self, table: str, rows: list[tuple]) -> None:
        if not rows:
            return
        cols = COLUMNS[table]
        placeholders = ",".join("?" * len(cols))
        self._conn.executemany(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", rows
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
