"""Monitor API: snapshot shaping, aggregation and failure modes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bottrade.config import Config
from bottrade.data.store import Store
from bottrade.models import Position, Venue
from bottrade.web.server import _group_rejections, build_snapshot


def seed_run(db_path: Path, ticks: int = 12) -> None:
    """Write a small but realistic run log by hand."""
    import asyncio

    from bottrade.agents.orchestrator import Orchestrator
    from bottrade.feeds import MockFeed

    cfg = Config.load()
    cfg.set("run.source", "mock")
    cfg.set("run.seed", 4)
    cfg.set("sentiment.news_enabled", False)

    store = Store(db_path)
    orch = Orchestrator(cfg, feed=MockFeed(cfg), store=store)
    asyncio.run(orch.run(max_ticks=ticks))
    store.close()


class TestSnapshot(unittest.TestCase):
    def test_missing_database_is_reported_not_raised(self):
        snap = build_snapshot(Path("does/not/exist.sqlite3"))
        self.assertFalse(snap["ok"])
        self.assertIn("database not found", snap["meta"]["error"])
        self.assertEqual(snap["positions"], [])

    def test_snapshot_shape_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "run.sqlite3"
            seed_run(db, ticks=12)

            snap = build_snapshot(db)
            self.assertTrue(snap["ok"])
            for key in ("meta", "equity", "positions", "fills", "signals",
                        "rejections", "venues", "events", "stats"):
                self.assertIn(key, snap)

            eq = snap["equity"]
            self.assertGreater(eq["now"], 0)
            self.assertAlmostEqual(eq["pnl"], eq["now"] - eq["start"], places=2)
            self.assertGreaterEqual(eq["max_dd"], 0.0)
            self.assertLessEqual(len(eq["curve"]), 320)
            self.assertGreater(len(eq["curve"]), 0)

            # positions are a single tick's snapshot, never a pile of duplicates
            keys = [p["key"] for p in snap["positions"]]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(snap["positions"]), snap["stats"]["open_positions"])

            # rejections must actually be recorded now
            self.assertGreater(len(snap["rejections"]), 0)
            for bucket in snap["rejections"]:
                self.assertIn("reason", bucket)
                self.assertGreater(bucket["n"], 0)

            self.assertGreater(snap["stats"]["fills"], 0)
            self.assertGreaterEqual(snap["stats"]["win_rate"], 0.0)
            self.assertLessEqual(snap["stats"]["win_rate"], 1.0)

    def test_snapshot_is_json_serialisable(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "run.sqlite3"
            seed_run(db, ticks=6)
            payload = build_snapshot(db)
            text = json.dumps(payload, default=str)
            self.assertGreater(len(text), 200)
            self.assertNotIn("NaN", text)


class TestRejectionGrouping(unittest.TestCase):
    def test_near_identical_reasons_collapse(self):
        rows = [
            {"reason": "spread 8.3% > 6.0%"},
            {"reason": "spread 9.1% > 6.0%"},
            {"reason": "cooldown 12/30 min"},
            {"reason": "venue cap: 4 open on polymarket"},
            {"reason": "max_open_positions (8)"},
        ]
        grouped = {g["reason"]: g["n"] for g in _group_rejections(rows)}
        self.assertEqual(grouped.get("spread #% > #%"), 2)
        self.assertEqual(grouped.get("cooldown #/# min"), 1)
        self.assertEqual(grouped.get("venue cap: # open on polymarket"), 1)
        self.assertEqual(grouped.get("max_open_positions"), 1)

    def test_empty_input(self):
        self.assertEqual(_group_rejections([]), [])


class TestRestartBehindTheCounter(unittest.TestCase):
    """A restarted bot counts from 1 again; the dashboard must follow the clock.

    Seen live: the old run reached tick 39, the restart was at tick 7, and the
    dashboard kept rendering tick 39's positions — a frozen screen over a bot
    that was actively trading.
    """

    def _store(self, db: Path) -> Store:
        store = Store(db)
        return store

    def _position(self, token: str, question: str, value: float) -> Position:
        return Position(
            key=f"pumpfun:{token}", venue=Venue.PUMPFUN, token_id=token,
            question=question, qty=10.0, avg_price=value / 10.0,
            last_price=value / 10.0, high_water=value / 10.0,
        )

    def test_newest_run_wins_even_with_a_lower_tick_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "run.sqlite3"
            store = self._store(db)

            # The old run got as far as tick 39, so it also passed through 12 —
            # the number the restarted bot will reuse on its twelfth tick.
            store.record_tick(12, 1010.0, 950.0, 60.0, 10.0, 0.0, 2, 20, 0, 0, 0)
            store.record_positions(
                [self._position("OLD_A", "OLD A", 30.0),
                 self._position("OLD_B", "OLD B", 30.0)],
                tick=12,
            )
            store.record_health(
                [{"venue": "pumpfun", "ok": False, "miss": 4, "note": "old run"}], tick=12
            )
            store.record_tick(39, 1048.0, 900.0, 148.0, 0.0, 48.0, 3, 20, 0, 0, 0)
            store.record_positions(
                [self._position("OLD_C", "OLD C", 40.0)], tick=39
            )

            # ... the process is restarted and starts counting from 1 again
            store.record_tick(7, 1002.0, 817.53, 150.0, -5.07, 7.0, 1, 20, 0, 0, 0)
            store.record_positions([self._position("MID", "MID", 150.0)], tick=7)
            store.record_tick(12, 1006.87, 817.53, 189.34, -5.07, 13.78, 1, 20, 0, 0, 0)
            store.record_positions([self._position("NEW", "NEW TOKEN", 189.34)], tick=12)
            store.record_health(
                [{"venue": "pumpfun", "ok": True, "miss": 0, "note": "new run"}], tick=12
            )
            store.close()

            snap = build_snapshot(db)

            # Only the current run's batch, even though tick 12 exists twice.
            self.assertEqual([p["key"] for p in snap["positions"]], ["pumpfun:NEW"])
            self.assertEqual(snap["meta"]["tick"], 12)
            self.assertEqual(snap["stats"]["open_positions"], 1)
            health = {h["venue"]: h for h in snap["health"]}
            self.assertTrue(health["pumpfun"]["ok"], "old run's stale health leaked in")
            self.assertFalse(snap["deadman"], "a healthy new run must not look dead")

    def test_health_never_shows_a_venue_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "run.sqlite3"
            store = self._store(db)
            for tick, ok in ((12, False), (39, False), (12, True)):
                store.record_health(
                    [{"venue": "pumpfun", "ok": ok, "miss": 0, "note": f"t{tick}"}],
                    tick=tick,
                )
            store.close()

            snap = build_snapshot(db)
            venues = [h["venue"] for h in snap["health"]]
            self.assertEqual(venues, sorted(set(venues)))
            self.assertTrue(snap["health"][0]["ok"], "newest health row must win")


if __name__ == "__main__":
    unittest.main()
