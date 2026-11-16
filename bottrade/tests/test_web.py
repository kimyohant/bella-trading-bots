"""Monitor API: snapshot shaping, aggregation and failure modes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bottrade.config import Config
from bottrade.data.store import Store
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


if __name__ == "__main__":
    unittest.main()
