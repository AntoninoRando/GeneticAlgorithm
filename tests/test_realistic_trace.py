"""Tests for the realistic trace generator (data_processing/realistic_trace).

Run from the repository root, after building the `scheduler` extension:

    python3 -m unittest discover -s tests -v

None of these tests needs data/ or a network connection.  The orbit tests use
published reference values and one public element set; the simulator tests
use the idealised Walker constellation, which is built in code.  They check
the geometry, the model equations and the file format, not that any number
"looks realistic": that comparison is in the validation report the generator
prints (see docs/onboarding/realistic-data.html).

The whole file is skipped when the optional `sgp4` package is missing.
"""
import contextlib
import csv
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.getcwd())

try:
    import numpy as np
    import sgp4  # noqa: F401
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

if HAVE_DEPS:
    from data_processing.realistic_trace import generate, orbits
    from data_processing.realistic_trace import simulator as sim
    from data_processing.realistic_trace.scenario import (CPU_DATA, CPU_INTENSIVE, GENERIC,
                                                          Scenario)

# The classic ISS element set used in the sgp4 package documentation.
ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482
"""

# One record in the OMM CSV layout served by CelesTrak (public data).
OMM_CSV = (
    "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,"
    "ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,"
    "ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT\n"
    "STARLINK-1008,2019-074B,2026-11-20T00:00:02.000160,15.70814436,.00027603,53.1440,"
    "29.5978,11.0714,251.2644,0,U,44714,999,587,-.18873323E-4,-.2995E-4,0\n"
)


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class Geometry(unittest.TestCase):
    def test_gmst_reference_value(self):
        # Textbook check value for the IAU 1982 formula (the GMST worked
        # example in Vallado's Fundamentals of Astrodynamics and Applications):
        # 1992-08-20 12:14:00 UT1 -> GMST = 152.578787810 deg.
        jd = 2448854.5 + (12 + 14 / 60) / 24
        self.assertAlmostEqual(math.degrees(float(orbits.gmst_rad(jd))), 152.578787810, places=5)

    def test_observer_on_the_ellipsoid(self):
        equator = orbits.Observer("eq", 0.0, 0.0).ecef_km()
        pole = orbits.Observer("np", 90.0, 0.0).ecef_km()
        np.testing.assert_allclose(equator, [6378.137, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(pole, [0.0, 0.0, 6356.7523], atol=1e-3)

    def test_elevation_of_simple_cases(self):
        rome = orbits.Observer("Roma", 41.8967, 12.4822)
        overhead = rome.ecef_km() + 550.0 * rome.up()
        self.assertAlmostEqual(float(orbits.elevation_deg(overhead, rome)), 90.0, places=4)
        # A point 550 km up and 550 km to the side is 45 degrees above the horizon.
        east = np.cross([0.0, 0.0, 1.0], rome.up())
        east /= np.linalg.norm(east)
        self.assertAlmostEqual(
            float(orbits.elevation_deg(rome.ecef_km() + 550.0 * (rome.up() + east), rome)),
            45.0, places=4)
        below = orbits.Observer("antipode", -41.8967, 12.4822 - 180.0).ecef_km()
        self.assertLess(float(orbits.elevation_deg(below, rome)), 0.0)

    def test_rotation_keeps_lengths_and_z(self):
        r = np.array([[[7000.0, 100.0, -50.0]]])
        out = orbits.teme_to_ecef(r, np.array([2460000.25]))
        self.assertAlmostEqual(float(np.linalg.norm(out)), float(np.linalg.norm(r)), places=6)
        self.assertEqual(out[0, 0, 2], -50.0)


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class OrbitFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_tle_with_name_lines(self):
        path = self.dir / "iss.tle"
        path.write_text(ISS_TLE)
        c = orbits.load_orbits(str(path))
        self.assertEqual(c.names, ["ISS (ZARYA)"])
        self.assertEqual(c.satrecs[0].satnum, 25544)
        # Epoch: day 343.69339541 of 2019.
        self.assertTrue(orbits.jd_to_iso(c.median_epoch_jd()).startswith("2019-12-09T16:38"))

    def test_iss_altitude_is_plausible(self):
        path = self.dir / "iss.tle"
        path.write_text(ISS_TLE)
        c = orbits.load_orbits(str(path))
        err, pos = orbits._propagate_ecef(c.satrecs, c.median_epoch_jd(), np.arange(0.0, 5600.0, 60.0))
        self.assertTrue((err == 0).all())
        radius = np.linalg.norm(pos[0], axis=-1)
        self.assertTrue(6770.0 < radius.min() and radius.max() < 6810.0, (radius.min(), radius.max()))

    def test_omm_csv(self):
        path = self.dir / "starlink.csv"
        path.write_text(OMM_CSV)
        c = orbits.load_orbits(str(path))
        self.assertEqual(c.names, ["STARLINK-1008"])
        self.assertEqual(c.satrecs[0].satnum, 44714)
        self.assertAlmostEqual(math.degrees(c.satrecs[0].inclo), 53.1440, places=4)

    def test_truncated_file_keeps_the_complete_records(self):
        path = self.dir / "cut.csv"
        path.write_text(OMM_CSV + "STARLINK-1017,2019-074L,2026-11-19T15:04:03.123552,15.45286023,.00007")
        self.assertEqual(len(orbits.load_orbits(str(path))), 1)

    def test_empty_file_is_an_error(self):
        path = self.dir / "empty.csv"
        path.write_text("\n")
        with self.assertRaises(ValueError):
            orbits.load_orbits(str(path))


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class WalkerEphemeris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rome = orbits.Observer("Roma", 41.8967, 12.4822)
        cls.shell = orbits.walker_constellation()
        cls.eph = orbits.build_ephemeris(cls.shell, cls.rome, duration_s=120.0)

    def test_shell_size_and_radius(self):
        self.assertEqual(len(self.shell), 120 * 40)
        radius = np.linalg.norm(self.eph.pos, axis=-1)
        # 550 km above the equatorial radius, within SGP4's short-period terms.
        self.assertTrue(6905.0 < radius.min() and radius.max() < 6945.0, (radius.min(), radius.max()))

    def test_only_nearby_satellites_are_kept(self):
        self.assertTrue(0 < len(self.eph.names) < len(self.shell))
        dist = np.linalg.norm(self.eph.pos - self.rome.ecef_km(), axis=-1)
        self.assertLessEqual(dist.min(axis=1).max(), 4000.0 + 120.0)

    def test_speed_and_interpolation(self):
        # Circular-orbit speed at 550 km is about 7.6 km/s; in the Earth-fixed
        # frame the ground's rotation changes it by at most 0.5 km/s.
        step = np.linalg.norm(self.eph.pos[:, 1] - self.eph.pos[:, 0], axis=-1)
        self.assertTrue((step > 7.0).all() and (step < 8.1).all(), (step.min(), step.max()))
        mid = self.eph.at(0.5)
        np.testing.assert_allclose(mid, 0.5 * (self.eph.pos[:, 0] + self.eph.pos[:, 1]))
        np.testing.assert_allclose(self.eph.at(1e9), self.eph.pos[:, -1])

    def test_some_satellites_are_above_40_degrees(self):
        visible = (self.eph.elevation_series() >= 40.0).sum(axis=0)
        self.assertGreaterEqual(visible.min(), 1)


def _walker_ephemeris(scenario):
    observer = orbits.Observer(scenario.observer_name, scenario.observer_lat_deg,
                               scenario.observer_lon_deg)
    return orbits.build_ephemeris(orbits.walker_constellation(), observer,
                                  duration_s=scenario.duration_s + scenario.visibility_horizon_s)


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class ModelEquations(unittest.TestCase):
    def test_deadline_and_energy_formulas(self):
        sc = Scenario()
        # c_net = s / b_isl (Eq. 6.1); D = 1.1 (c + c_net) (Eq. 6.2).
        self.assertAlmostEqual(sc.network_demand_s(215.0), 0.1)
        self.assertAlmostEqual(sc.deadline_s(0.5, 215.0), 1.1 * 0.6)
        # eps_cpu = e c C^3 (Eq. 6.3): 5e-26 * 0.5 s * (1e9 Hz)^3 = 25 J.
        self.assertAlmostEqual(sc.cpu_energy_j(0.5, 1e9), 25.0)
        # eps_net = P c_net with P = 1 W.
        self.assertAlmostEqual(sc.net_energy_j(0.1), 0.1)
        self.assertEqual(sc.buffer_elevation_deg, 25.0)

    def test_invalid_scenarios_are_rejected(self):
        with self.assertRaises(ValueError):
            Scenario(policy="Random")
        with self.assertRaises(ValueError):
            Scenario(arrival_rate_per_s=0.0)
        with self.assertRaises(ValueError):
            Scenario(type_mix=((GENERIC, 0.5), (CPU_INTENSIVE, 0.4)))

    def test_waiting_time_estimate(self):
        # W = r_min + max(0, A - m r_min) / m, with r_min = 0 while a slot is free.
        stage = sim._Stage(capacity=2)
        self.assertEqual(stage.wait_estimate(0.0), 0.0)
        stage.running[1] = 2.0                       # one slot busy for 2 s
        self.assertAlmostEqual(stage.wait_estimate(0.0), 2.0 / 2)
        stage.running[2] = 3.0                       # both busy: r_min = 2, A = 5
        self.assertAlmostEqual(stage.wait_estimate(0.0), 2.0 + (5.0 - 4.0) / 2)
        stage.queue.append((None, 1.0, None))        # one queued request of 1 s: A = 6
        self.assertAlmostEqual(stage.wait_estimate(0.0), 2.0 + (6.0 - 4.0) / 2)
        self.assertAlmostEqual(stage.wait_estimate(1.0), 1.0 + (4.0 - 2.0) / 2)
        self.assertEqual(stage.occupancy(), 3)


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class TopologySnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = Scenario()
        cls.eph = _walker_ephemeris(cls.sc)
        cls.topology = sim.Topology(cls.eph, cls.sc, seed=1)

    def test_dome_buffer_and_neighbours(self):
        for k in (0, 40, 149):
            snap = self.topology.snapshot(k)
            self.assertEqual(snap.time, 2.0 * k)
            self.assertTrue(set(snap.sunset_s) <= set(snap.active))
            for i in snap.active:
                self.assertGreater(snap.elevation[i], 25.0)
                self.assertEqual(i in snap.sunset_s, snap.elevation[i] >= 40.0)
                for j in snap.neighbours[i]:
                    self.assertIn(i, snap.neighbours[j])                 # symmetric
                    self.assertLess(snap.distance_km[(i, j)], 659.0)
                    self.assertAlmostEqual(snap.distance_km[(i, j)], snap.distance_km[(j, i)])

    def test_access_points_are_the_nearest_approaching_dome_satellites(self):
        snap = self.topology.snapshot(10)
        g = 20                                                           # 20 s on a 1 s grid
        slant = np.linalg.norm(self.eph.pos - self.eph.observer.ecef_km(), axis=-1)
        approaching = [i for i in snap.sunset_s if slant[i, g + 1] < slant[i, g]]
        expected = sorted(approaching, key=lambda i: slant[i, g])[:5]
        self.assertEqual(snap.access_points, expected)
        self.assertTrue(1 <= len(snap.access_points) <= 5)

    def test_sunset_is_the_time_left_in_the_dome(self):
        elev = self.eph.elevation_series()
        snap = self.topology.snapshot(5)                                 # t = 10 s
        self.assertTrue(snap.sunset_s)
        for i, left in snap.sunset_s.items():
            self.assertGreaterEqual(left, 0)
            if left < 800:                                               # sets inside the horizon
                self.assertTrue((elev[i, 10:10 + left] >= 40.0).all())
                self.assertLess(elev[i, 10 + left], 40.0)

    def test_link_bandwidth(self):
        snap = self.topology.snapshot(0)
        i = next(i for i in snap.active if snap.neighbours[i])
        j = snap.neighbours[i][0]
        b = self.topology.link_mb_per_s(0, i, j)
        self.assertTrue(2150.0 <= b <= 3125.0)
        self.assertEqual(b, self.topology.link_mb_per_s(0, j, i))        # same both ways
        self.assertNotEqual(b, self.topology.link_mb_per_s(1, i, j))     # redrawn each update


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class SimulationRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A CPU frequency at which batteries matter but are not the only limit.
        cls.sc = Scenario(energy_cpu_hz=4.5e9)
        cls.eph = _walker_ephemeris(cls.sc)
        cls.simulation = sim.Simulation(cls.eph, cls.sc, seed=7)
        cls.result = cls.simulation.run()
        cls.rows = cls.result.rows
        cls.done = [r for r in cls.rows if r["Status"] == "Completed"]

    def test_same_seed_same_log(self):
        again = sim.simulate(self.eph, self.sc, seed=7)
        self.assertEqual(again.rows, self.rows)
        self.assertEqual(again.stats, self.result.stats)
        self.assertNotEqual(sim.simulate(self.eph, self.sc, seed=8).rows, self.rows)

    def test_arrivals_are_poisson_with_the_published_mix(self):
        stats = self.result.stats
        # Rate 10/s for 300 s: mean 3000, standard deviation about 55.
        self.assertTrue(2780 < stats["generated"] < 3220, stats["generated"])
        mix = stats["requests_by_type"]
        for name, share in ((GENERIC, 0.40), (CPU_INTENSIVE, 0.35), (CPU_DATA, 0.25)):
            self.assertAlmostEqual(mix[name] / stats["generated"], share, delta=0.04)
        logged = stats["completed"] + stats["rejected"]
        self.assertEqual(len(self.rows), logged + stats["waiting_at_end"])
        self.assertLessEqual(logged + stats["dropped_unlogged"], stats["generated"])

    def test_rows_have_the_simulator_columns(self):
        names = set(self.eph.names)
        for row in self.rows:
            self.assertEqual(list(row), sim.COLUMNS)
            self.assertIn(row["Status"], ("Completed", "Rejected", "In Queue"))
            self.assertIn(row["Server Name"], names)
        self.assertGreater(len(self.done), 500)
        self.assertGreater(len(self.rows) - len(self.done), 100)

    def test_completed_rows_are_consistent(self):
        for row in self.done:
            arrival, start, end = row["Arrival Time (System)"], row["Start Time"], row["End Time"]
            self.assertTrue(0.0 <= arrival <= start <= end <= 300.0, row)
            self.assertAlmostEqual(start - arrival, row["transfer_time"])
            self.assertAlmostEqual(row["Execution time"], end - start)
            self.assertAlmostEqual(row["Time in system"], end - arrival)
            self.assertEqual(row["Num Hops"], 1)
            self.assertGreaterEqual(row["Queue length"], 1)               # counts itself
            self.assertGreaterEqual(row["Remaining_energy [J]"], 0.0)
            self.assertEqual(row["Rejection Reason"], "N/A")
            if row["Routing End Time"] != "N/A":
                self.assertGreaterEqual(row["Routing End Time"], end)
                self.assertTrue(1 <= row["Num Hops Routing"] <= 21)
            # eps_cpu = e c C^3 with the run-wide frequency: c is shorter than the stay.
            demand = row["Energy_CPU [J]"] / (5e-26 * 4.5e9 ** 3)
            self.assertLessEqual(demand, row["Execution time"] + 1e-9)

    def test_rejected_rows(self):
        reasons = {sim.R_NO_SERVER, sim.R_SUNSET, sim.R_DEADLINE, sim.R_ENERGY_CPU,
                   sim.R_ENERGY_CPU_NET, sim.R_ENERGY_NET, sim.R_ENERGY_ROUTING}
        rejected = [r for r in self.rows if r["Status"] == "Rejected"]
        for row in rejected:
            self.assertIn(row["Rejection Reason"], reasons)
            self.assertEqual(row["End Time"], "N/A")
            self.assertEqual(row["Energy_TOTAL [J]"], 0.0)
        seen = {r["Rejection Reason"] for r in rejected}
        self.assertIn(sim.R_NO_SERVER, seen)
        self.assertTrue(seen & {sim.R_ENERGY_CPU, sim.R_ENERGY_CPU_NET})

    def test_energy_is_conserved_and_never_negative(self):
        spent = {}
        for row in self.done:
            spent[row["Server Name"]] = spent.get(row["Server Name"], 0.0) + row["Energy_TOTAL [J]"]
        for server in self.simulation.servers.values():
            self.assertGreaterEqual(server.energy, 0.0)
            used = self.sc.initial_energy_j - server.energy
            executed = spent.get(server.name, 0.0)
            # What a node spends beyond the executions in the log is the
            # transmission of payloads and results (1 W for a few seconds in
            # total) and the requests still in service when the run ends.
            self.assertGreaterEqual(used, executed - 1e-6)
            self.assertLess(used - executed, 60.0, server.name)

    def test_batteries_limit_the_work(self):
        cool = sim.simulate(self.eph, self.sc.with_changes(energy_cpu_hz=1e9), seed=7).stats
        hot = sim.simulate(self.eph, self.sc.with_changes(energy_cpu_hz=9e9), seed=7).stats
        self.assertEqual(cool["energy_share_of_rejections"], 0.0)
        self.assertGreater(hot["energy_share_of_rejections"], 0.5)
        self.assertGreater(cool["success_rate"], self.result.stats["success_rate"])
        self.assertGreater(self.result.stats["success_rate"], hot["success_rate"])
        # The same seed gives the same workload whatever happens to it.
        self.assertEqual(cool["requests_by_type"], hot["requests_by_type"])

    def test_one_cpu_slot_serves_one_request_at_a_time(self):
        sc = self.sc.with_changes(cpu_slots=1, energy_cpu_hz=1e9)
        rows = sim.simulate(self.eph, sc, seed=7).rows
        busy = {}
        for row in rows:
            if row["Status"] == "Completed" and row["Task Type"] != CPU_DATA:
                begin = row["Start Time"] + row["Time in queue"]
                demand = row["Energy_CPU [J]"] / (5e-26 * 1e9 ** 3)
                busy.setdefault(row["Server Name"], []).append((begin, begin + demand))
        self.assertTrue(busy)
        for intervals in busy.values():
            intervals.sort()
            for (_, end), (begin, _) in zip(intervals, intervals[1:]):
                self.assertLessEqual(end, begin + 1e-9)

    def test_visibility_table(self):
        vis = self.result.visibility
        self.assertEqual({v["Time"] for v in vis}, {2.0 * k for k in range(151)})
        for v in vis[:500]:
            self.assertGreater(v["Elevation [deg]"], 25.0)
            self.assertEqual(v["In dome"], v["Elevation [deg]"] >= 40.0)
            if v["Access point"]:
                self.assertTrue(v["In dome"])

    def test_ephemeris_must_cover_the_run(self):
        observer = orbits.Observer("Roma", 41.8967, 12.4822)
        short = orbits.build_ephemeris(orbits.walker_constellation(), observer, duration_s=60.0)
        with self.assertRaises(ValueError):
            sim.simulate(short, self.sc)


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class GeneratedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.dir = Path(cls._tmp.name) / "realistic"
        cls.shell = orbits.walker_constellation()
        cls.sc = Scenario(duration_s=60.0, energy_cpu_hz=4.5e9)
        cls.report = generate.generate_trace(cls.shell, str(cls.dir), scenario=cls.sc,
                                             runs=2, seed=5)
        with (cls.dir / "trace.csv").open(newline="", encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))

    def test_trace_file(self):
        self.assertEqual(list(self.rows[0]), sim.COLUMNS)
        ids = [r["Task ID"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))                        # unique across runs
        arrivals = [float(r["Arrival Time (System)"]) for r in self.rows
                    if r["Arrival Time (System)"] != "N/A"]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(0.0 <= arrivals[0] and arrivals[-1] < 120.0)
        self.assertTrue(any(a >= 60.0 for a in arrivals))                # the second run is shifted
        # Two runs of 60 s at 10 requests per second.
        self.assertTrue(1000 < len(self.rows) < 1400, len(self.rows))

    def test_same_arguments_same_bytes(self):
        other = Path(self._tmp.name) / "again"
        generate.generate_trace(self.shell, str(other), scenario=self.sc, runs=2, seed=5)
        self.assertEqual((other / "trace.csv").read_bytes(), (self.dir / "trace.csv").read_bytes())
        generate.generate_trace(self.shell, str(other), scenario=self.sc, runs=2, seed=6)
        self.assertNotEqual((other / "trace.csv").read_bytes(), (self.dir / "trace.csv").read_bytes())

    def test_worlds_and_report(self):
        manifest = json.loads((self.dir / "worlds" / "manifest.json").read_text())
        self.assertEqual(manifest["num_worlds"], 12)                     # 2 runs x 6 windows of 10 s
        for world in manifest["worlds"]:
            self.assertLess(world["span"], 10.0)
        report = json.loads((self.dir / "report.json").read_text())
        self.assertEqual(report["worlds"], 12)
        self.assertEqual(report["runs"], 2)
        self.assertEqual(len(report["per_run"]), 2)
        self.assertEqual(report["scenario"]["duration_s"], 60.0)
        self.assertNotEqual(report["per_run"][0]["start_utc"], report["per_run"][1]["start_utc"])
        self.assertIn("NOT the original log", report["what"])
        text = generate.format_report(report)
        self.assertIn("IDEALISED CONSTELLATION", text)
        self.assertIn("No published figure", text)                       # 60 s is not the published setting
        with (self.dir / "visibility.csv").open(newline="", encoding="utf-8") as f:
            self.assertEqual(next(csv.reader(f))[:3], ["Run", "Time", "Server Name"])

    def test_data_factory_reads_the_worlds(self):
        from data_processing.data_factory import create_jobs_from_csv, create_satellites_from_csv
        paths = generate.load_world_paths(str(self.dir / "worlds"))
        self.assertEqual(len(paths), 12)
        for path in paths[:4]:
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = create_jobs_from_csv(str(path))
                sats = create_satellites_from_csv(str(path), seed=1)
            self.assertGreater(len(jobs), 50)
            self.assertGreaterEqual(len(sats), 5)
            self.assertEqual(min(j.arrivalTime for j in jobs), 0)
            for name in ("executionTime", "transferTime", "imageSize", "numberOfHops", "ram"):
                self.assertGreater(len({getattr(j, name) for j in jobs}), 1, name)
            self.assertGreater(len({s.cpuCapacity for s in sats}), 1)
            self.assertGreater(len({s.remainingEnergy for s in sats}), 1)

    def test_energy_frequency_is_fitted_drawn_or_given(self):
        out = str(Path(self._tmp.name) / "energy")
        free = Scenario(duration_s=60.0)                      # energy_cpu_hz left open
        fitted = generate.generate_trace(self.shell, out, scenario=free, runs=1, seed=5)
        fit = fitted["calibration"]
        self.assertTrue(1e9 <= fit["energy_cpu_hz"] <= 1e10)
        self.assertEqual(fitted["energy_cpu_hz_used"], [fit["energy_cpu_hz"]])
        self.assertEqual(fit["evaluations"], 16 + 7)
        self.assertEqual(fit["published"], generate.PUBLISHED)
        # The fit is at least as close to the published pair as both ends of the range.
        eph = _walker_ephemeris(free)
        for hz in generate.CALIBRATION_RANGE_HZ:
            s = sim.simulate(eph, free.with_changes(energy_cpu_hz=hz), seed=5).stats
            error = ((s["success_rate"] - 0.757) ** 2
                     + (s["energy_share_of_rejections"] - 0.61) ** 2)
            self.assertLessEqual(fit["squared_error"], error + 1e-12)
        self.assertIn("fitted to the two published figures", generate.format_report(fitted))

        drawn = generate.generate_trace(self.shell, out, scenario=free, runs=2, seed=5,
                                        energy_cpu_hz="draw")
        self.assertIsNone(drawn["calibration"])
        self.assertEqual(len(drawn["energy_cpu_hz_used"]), 2)  # one draw per run
        given = generate.generate_trace(self.shell, out, scenario=free, runs=1, seed=5,
                                        energy_cpu_hz=3e9)
        self.assertEqual(given["energy_cpu_hz_used"], [3e9])
        # A scenario that fixes the value is used as is.
        self.assertIsNone(self.report["calibration"])
        self.assertEqual(self.report["energy_cpu_hz_used"], [4.5e9])

    def test_command_line(self):
        out_dir = Path(self._tmp.name) / "cli"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            generate.main(["--walker", "--output-dir", str(out_dir), "--runs", "1",
                           "--duration", "40", "--rate", "4", "--policy", "DTS"])
        self.assertIn("4 worlds", out.getvalue())
        self.assertIn("policy DTS", out.getvalue())
        self.assertTrue((out_dir / "worlds" / "world_003.csv").is_file())
        with self.assertRaises(SystemExit) as raised:
            generate.main(["--orbits", str(out_dir / "no-such-file.csv")])
        self.assertIn("--download", str(raised.exception))


@unittest.skipUnless(HAVE_DEPS, "needs the sgp4 package (pip install sgp4)")
class MainOnGeneratedWorlds(unittest.TestCase):
    def setUp(self):
        import main
        self.main = main
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def run_main(self, argv):
        config, _ = self.main.resolve_config(argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.main.run_genetic_algorithm(config)
        return out.getvalue()

    def test_configuration_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            config, _ = self.main.resolve_config(["-c", "realistic"])
        self.assertEqual(config["worlds_dir"], "data/realistic/worlds")
        self.assertFalse(config["demo"])
        self.assertEqual(self.main.default_config()["worlds_dir"], "data/worlds")

    def test_trains_on_a_worlds_directory(self):
        out_dir = Path(self._tmp.name) / "realistic"
        generate.generate_trace(orbits.walker_constellation(), str(out_dir),
                                scenario=Scenario(duration_s=40.0, energy_cpu_hz=4.5e9), runs=1)
        argv = ["--worlds-dir", str(out_dir / "worlds"), "--generations", "6", "--population", "20",
                "--max-depth", "4", "--seed", "3", "--print-every", "0", "--disable-metrics",
                "--csv-row-limit", "0"]
        first = self.run_main(argv)
        self.assertIn("Loaded 4 pre-processed world(s)", first)
        self.assertIn("Jobs scheduled:", first)
        line = [l for l in first.splitlines() if l.startswith("Jobs scheduled:")]
        self.assertEqual(line, [l for l in self.run_main(argv).splitlines()
                                if l.startswith("Jobs scheduled:")])

    def test_missing_worlds_directory_is_an_error(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main(["--worlds-dir", str(Path(self._tmp.name) / "nothing"),
                           "--generations", "1", "--disable-metrics"])
        self.assertIn("data_processing.realistic_trace", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
