"""End-to-end: mock pipeline, sqlite logging, report generation, backtester."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bottrade.agents.base import build_agents, registry
from bottrade.agents.orchestrator import Orchestrator
from bottrade.agents.sentiment import score_text
from bottrade.backtest import run_backtest, write_demo_csv
from bottrade.config import Config
from bottrade.data.store import Store
from bottrade.feeds import MockFeed
from bottrade.report import generate_report


def test_config() -> Config:
    cfg = Config.load()
    cfg.set("run.source", "mock")
    cfg.set("run.seed", 3)
    cfg.set("sentiment.news_enabled", False)
    return cfg


class TestRegistry(unittest.TestCase):
    def test_all_builtin_agents_registered(self):
        names = set(registry())
        for expected in ("scanner", "sentiment", "whale", "signal", "risk",
                         "executor", "portfolio"):
            self.assertIn(expected, names)

    def test_build_agents_follows_config_order(self):
        cfg = test_config()
        cfg.set("agents.enabled", ["scanner", "risk", "portfolio"])
        self.assertEqual([a.name for a in build_agents(cfg)],
                         ["scanner", "risk", "portfolio"])

    def test_unknown_agent_raises(self):
        cfg = test_config()
        cfg.set("agents.enabled", ["nope"])
        with self.assertRaises(KeyError):
            build_agents(cfg)


class TestSentiment(unittest.TestCase):
    def test_bullish_and_bearish_words(self):
        up, hits_up = score_text("Company wins approval, shares surge")
        down, hits_down = score_text("Company fails vote, shares plunge")
        self.assertGreater(up, 0.2)
        self.assertLess(down, -0.2)
        self.assertGreater(hits_up, 0)
        self.assertGreater(hits_down, 0)

    def test_negation_flips_polarity(self):
        positive, _ = score_text("deal will pass")
        negated, _ = score_text("deal will not pass")
        self.assertGreater(positive, 0)
        self.assertLess(negated, positive)

    def test_neutral_text_scores_zero(self):
        score, hits = score_text("will the event happen in june")
        self.assertEqual(hits, 0)
        self.assertEqual(score, 0.0)


class TestMockPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_run_trades_and_records(self):
        cfg = test_config()
        store = Store(":memory:")
        orch = Orchestrator(cfg, feed=MockFeed(cfg), store=store)
        summary = await orch.run(max_ticks=40)

        self.assertEqual(summary["ticks"], 40)
        self.assertGreater(summary["fills"], 0, "mock pipeline should trade")
        self.assertGreater(summary["universe"], 0)

        counts = store.counts()
        self.assertEqual(counts["fills"], summary["fills"])
        self.assertEqual(counts["equity"], 40)
        self.assertGreater(counts["markets"], 0)
        store.close()

    async def test_equity_is_conserved_on_entry(self):
        cfg = test_config()
        orch = Orchestrator(cfg, feed=MockFeed(cfg))
        await orch.run(max_ticks=6)
        broker = orch.broker
        # cash + marked positions must equal reported equity, and nothing may go negative
        self.assertGreaterEqual(broker.cash, -1e-9)
        self.assertAlmostEqual(broker.equity(), broker.cash + broker.gross_exposure(), places=6)

    async def test_max_ticks_respected(self):
        cfg = test_config()
        orch = Orchestrator(cfg, feed=MockFeed(cfg))
        summary = await orch.run(max_ticks=3)
        self.assertEqual(summary["ticks"], 3)
        self.assertEqual(summary["stopped"], "max_ticks")


class TestReport(unittest.IsolatedAsyncioTestCase):
    async def test_report_from_run(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "run.sqlite3"
            store = Store(db)
            orch = Orchestrator(cfg, feed=MockFeed(cfg), store=store)
            await orch.run(max_ticks=25)
            store.close()

            result = generate_report(db, Path(tmp) / "out", title="unit test run")
            markdown = Path(result["markdown"]).read_text(encoding="utf-8")
            self.assertTrue(Path(result["csv"]).exists())
            self.assertIn("# unit test run", markdown)
            self.assertIn("Equity curve", markdown)
            self.assertIn("Paper trading only", markdown)
            self.assertGreater(result["metrics"]["fills"], 0)


class TestBacktest(unittest.IsolatedAsyncioTestCase):
    async def test_backtest_demo_csv(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_demo_csv(Path(tmp) / "prices.csv", ticks=80, seed=5)
            self.assertTrue(csv_path.exists())
            result = await run_backtest(cfg, csv_path)
            metrics = result["metrics"]
            self.assertGreater(metrics["ticks"], 0)
            self.assertGreater(metrics["starting_cash"], 0)
            self.assertGreaterEqual(metrics["equity"], 0)
            self.assertIsInstance(metrics["win_rate"], float)


if __name__ == "__main__":
    unittest.main()
