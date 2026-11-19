"""Tests for the synthetic demo data and main.py --demo.

Run from the repository root, after building the `scheduler` extension:

    python3 -m unittest discover -s tests -v

None of these tests needs data/: they write the demo data into a temporary
directory.  They check that the demo goes through the real pipeline (splitter,
data factory, C++ evaluator), not that its numbers mean anything.
"""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.getcwd())

from data_processing import demo_data  # noqa: E402
from data_processing.csv_preprocessing import NORMALIZED_ARRIVAL_COLUMN  # noqa: E402


class DemoTrace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_same_seed_same_file(self):
        a, b, c = self.dir / "a.csv", self.dir / "b.csv", self.dir / "c.csv"
        demo_data.write_demo_trace(str(a), seed=7)
        demo_data.write_demo_trace(str(b), seed=7)
        demo_data.write_demo_trace(str(c), seed=8)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertNotEqual(a.read_bytes(), c.read_bytes())

    def test_columns_and_arrivals(self):
        path = self.dir / "trace.csv"
        n = demo_data.write_demo_trace(str(path), minutes=60.0, tasks_per_minute=10.0)
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), n)
        self.assertEqual(list(rows[0].keys()), demo_data.COLUMNS)
        # Poisson process of rate 10/min over 60 min: mean 600, sd ~24.5.
        self.assertTrue(450 < n < 750, n)
        arrivals = [float(r["Arrival Time (System)"]) for r in rows]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(0.0 <= arrivals[0] and arrivals[-1] < 60.0)

    def test_faster_servers_execute_faster(self):
        # The only structure the GP can exploit: execution time depends on the
        # server.  Check that it is there (means differ between servers).
        path = self.dir / "trace.csv"
        demo_data.write_demo_trace(str(path))
        per_server = {}
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                per_server.setdefault(r["Server Name"], []).append(float(r["Execution time"]))
        means = [sum(v) / len(v) for v in per_server.values()]
        self.assertEqual(len(per_server), 12)
        self.assertGreater(max(means) / min(means), 1.5)


class DemoWorlds(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = demo_data.prepare_demo_worlds(self._tmp.name)

    def test_six_ten_minute_worlds(self):
        self.assertEqual(len(self.paths), 6)
        manifest = json.loads((self.paths[0].parent / "manifest.json").read_text())
        self.assertEqual(manifest["num_worlds"], 6)
        for world in manifest["worlds"]:
            self.assertLess(world["span"], 10.0)
            self.assertTrue(60 <= world["tasks"] <= 140, world)

    def test_data_factory_reads_them(self):
        from data_processing.data_factory import create_jobs_from_csv, create_satellites_from_csv
        for path in self.paths:
            with path.open(newline="", encoding="utf-8") as f:
                first = next(csv.DictReader(f))
            self.assertEqual(float(first[NORMALIZED_ARRIVAL_COLUMN]), 0.0)
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = create_jobs_from_csv(str(path))
                sats = create_satellites_from_csv(str(path), seed=1)
            self.assertGreater(len(jobs), 50)
            self.assertGreaterEqual(len(sats), 10)
            self.assertEqual(min(j.arrivalTime for j in jobs), 0)
            # Every column the jobs read must vary inside a world.
            for field in ("executionTime", "transferTime", "imageSize", "numberOfHops", "ram"):
                self.assertGreater(len({getattr(j, field) for j in jobs}), 1, field)


class MainDemoMode(unittest.TestCase):
    def setUp(self):
        import main
        self.main = main
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_dir = main.DEMO_DIR
        main.DEMO_DIR = self._tmp.name
        self.addCleanup(setattr, main, "DEMO_DIR", self._old_dir)

    def run_main(self, argv):
        config, _ = self.main.resolve_config(argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.main.run_genetic_algorithm(config)
        return out.getvalue()

    def test_demo_configuration_file(self):
        config, _ = self.main.resolve_config(["-c", "demo"])
        self.assertTrue(config["demo"])
        self.assertGreaterEqual(config["seed"], 0)

    def test_runs_end_to_end_and_is_reproducible(self):
        argv = ["--demo", "--generations", "8", "--population", "20", "--max-depth", "4",
                "--seed", "3", "--print-every", "0", "--disable-metrics"]
        first = self.run_main(argv)
        self.assertIn("DEMO MODE", first)
        self.assertIn("Jobs scheduled:", first)
        self.assertTrue((Path(self._tmp.name) / "worlds" / "world_000.csv").is_file())
        # Same seed, same demo data: same result line.
        second = self.run_main(argv)
        line = [l for l in first.splitlines() if l.startswith("Jobs scheduled:")]
        self.assertEqual(line, [l for l in second.splitlines() if l.startswith("Jobs scheduled:")])


if __name__ == "__main__":
    unittest.main()
