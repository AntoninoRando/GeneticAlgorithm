"""Write a realistic task log and split it into worlds.

Command line (from the repository root)::

    python3 -m data_processing.realistic_trace --download      # first time: fetch the orbits
    python3 -m data_processing.realistic_trace                 # later: reuse data/orbits/starlink.csv
    ./run realistic --no-start-prompt                          # train on the result

What is written, under ``data/realistic/`` (never under ``data/worlds/``, so
generated and original data cannot be mixed up):

``trace.csv``       the task log, with the 27 columns of the simulator's
                    result file: a drop-in replacement for ``data/example.csv``.
``worlds/``         the log split into 10-second windows by the usual splitter
                    (``world_000.csv``, ..., ``manifest.json``).
``visibility.csv``  for every topology update and satellite: elevation, time
                    to sunset, whether it is an access point.  Not read by the
                    current pipeline; kept because the real orbits are the
                    most valuable part of the simulation.
``report.json``     the scenario, the orbit file (name, SHA-256, epoch), the
                    statistics of every run and the comparison with the
                    figures published in [magliarisi2026thesis].

Several runs
------------
One run is 300 simulated seconds, i.e. 30 worlds, like the original log.  With
``--runs N`` the generator simulates N independent runs and writes them one
after the other: run k starts ``k * run_spacing`` seconds later in real time
(another part of the constellation is overhead), uses seed ``seed + k``, and
its times are shifted by ``k * duration`` so that the log stays ordered.
Batteries start full in every run.  More, independent worlds are the one thing
a generator offers that a single original log does not.

The one fitted parameter
------------------------
The original simulator charges CPU energy with a single frequency C that it
draws at random once per run, from 10 MHz to 10 GHz, and that the thesis does
not report.  Energy grows with C cubed, so this one number decides whether
batteries never matter or run out within seconds.  By default the generator
therefore *calibrates* it: it picks the C for which the simulated system comes
closest to the two figures published for this setting (success rate 75.7 %,
61 % of the rejections due to energy, see ``PUBLISHED``), using the same
orbits and seeds as the runs it is about to write.  Formally, with S(C) and
G(C) the mean success rate and energy share over the calibration runs::

    C* = argmin over C in [1 GHz, 10 GHz] of (S(C) - 0.757)^2 + (G(C) - 0.61)^2

searched on a logarithmic grid and then on a finer grid around its best point.
The two residuals are printed and stored in ``report.json``: one parameter
cannot match two numbers exactly, and the gap that is left says how far the
re-implementation is from the original.  ``--energy-cpu-hz draw`` restores
the random draw, ``--energy-cpu-hz 4.5e9`` fixes the value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_processing.csv_preprocessing import load_world_paths, split_into_worlds  # noqa: E402
from data_processing.realistic_trace import orbits  # noqa: E402
from data_processing.realistic_trace.scenario import Scenario  # noqa: E402
from data_processing.realistic_trace.simulator import COLUMNS, NA, simulate  # noqa: E402

REALISTIC_DIR = "data/realistic"
ORBITS_PATH = "data/orbits/starlink.csv"
DEFAULT_SEED = 2026
DEFAULT_RUNS = 5
DEFAULT_RUN_SPACING_S = 600.0
WINDOW_S = 10.0
# Only satellites that come this close to the ground user are propagated on the
# fine grid.  A satellite 25 degrees above the horizon at 600 km altitude is
# about 1250 km away, so nothing the simulation can use is lost.
KEEP_RADIUS_KM = 2000.0
CALIBRATION_RUNS = 3
CALIBRATION_RANGE_HZ = (1e9, 1e10)

# Published figures the generated data is compared with: OrbitAware at 10
# requests per second and a 100 kJ budget, [magliarisi2026thesis] Sec. 6.5.1.
# Every other result of that section is given only as a plot.
PUBLISHED = {
    "success_rate": 0.757,
    "energy_share_of_rejections": 0.61,
}

_TIME_COLUMNS = ("Arrival Time (System)", "Arrival Time (Queue)", "Start Time", "End Time",
                 "Routing Init Time", "Routing End Time")
_VISIBILITY_COLUMNS = ["Run", "Time", "Server Name", "Elevation [deg]", "In dome",
                       "Sunset [s]", "Access point", "Slant range [km]"]


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


def _fmt(value) -> str:
    """Numbers with enough digits to round-trip, everything else as text."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, 9))
    return str(value)


def _iso_to_jd(text: str) -> float:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    reference = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    return 2451545.0 + (moment - reference).total_seconds() / 86400.0


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def calibrate_energy_cpu_hz(ephemerides: Sequence[orbits.Ephemeris], seeds: Sequence[int],
                            scenario: Scenario) -> Tuple[float, dict]:
    """Fit the energy CPU frequency to the published figures (module docstring).

    ``scenario`` is used at the published setting (10 requests per second,
    100 kJ, OrbitAware) whatever the caller asked for, because that is the
    only setting with published numbers.  Returns the frequency and a record
    of the fit for the report.
    """
    reference = scenario.with_changes(arrival_rate_per_s=10.0, initial_energy_j=100000.0,
                                      policy="OrbitAware")

    def evaluate(hz: float) -> Tuple[float, float, float]:
        stats = [simulate(eph, reference.with_changes(energy_cpu_hz=hz), s).stats
                 for eph, s in zip(ephemerides, seeds)]
        success = _mean([s["success_rate"] for s in stats])
        share = _mean([s["energy_share_of_rejections"] for s in stats])
        loss = ((success - PUBLISHED["success_rate"]) ** 2
                + (share - PUBLISHED["energy_share_of_rejections"]) ** 2)
        return loss, success, share

    def log_grid(low: float, high: float, points: int) -> List[float]:
        return [low * (high / low) ** (i / (points - 1)) for i in range(points)]

    coarse = log_grid(*CALIBRATION_RANGE_HZ, 16)
    tried = {hz: evaluate(hz) for hz in coarse}
    best = min(coarse, key=lambda hz: tried[hz])
    i = coarse.index(best)
    for hz in log_grid(coarse[max(i - 1, 0)], coarse[min(i + 1, len(coarse) - 1)], 9)[1:-1]:
        tried[hz] = evaluate(hz)
    best = min(tried, key=lambda hz: (tried[hz][0], hz))
    loss, success, share = tried[best]
    return best, {
        "method": "least squares on the two published figures, log grid 16 + 7 points",
        "runs_used": len(ephemerides),
        "energy_cpu_hz": best,
        "success_rate": success,
        "energy_share_of_rejections": share,
        "published": dict(PUBLISHED),
        "squared_error": loss,
        "evaluations": len(tried),
    }


def generate_trace(constellation: orbits.Constellation, output_dir: str = REALISTIC_DIR, *,
                   scenario: Optional[Scenario] = None, runs: int = DEFAULT_RUNS,
                   seed: int = DEFAULT_SEED, start_jd: Optional[float] = None,
                   run_spacing_s: float = DEFAULT_RUN_SPACING_S,
                   window_s: float = WINDOW_S, orbit_file: Optional[str] = None,
                   energy_cpu_hz: Union[str, float] = "calibrate") -> dict:
    """Simulate ``runs`` runs, write the files listed in the module docstring
    and return the report (also written to ``report.json``).

    ``energy_cpu_hz`` is "calibrate" (fit it to the published figures),
    "draw" (one random draw per run, like the original simulator) or a number
    in hertz.  A scenario that already fixes ``energy_cpu_hz`` is used as is.

    The output is a pure function of the constellation and the arguments: the
    same orbit file, scenario and seed give the same ``trace.csv`` byte for byte.
    """
    scenario = scenario or Scenario()
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if start_jd is None:
        start_jd = constellation.median_epoch_jd()
    observer = orbits.Observer(scenario.observer_name, scenario.observer_lat_deg,
                               scenario.observer_lon_deg)

    cache: Dict[int, orbits.Ephemeris] = {}

    def ephemeris_of(k: int) -> orbits.Ephemeris:
        if k not in cache:
            eph = orbits.build_ephemeris(
                constellation, observer, start_jd=start_jd + k * run_spacing_s / 86400.0,
                duration_s=scenario.duration_s + scenario.visibility_horizon_s,
                keep_radius_km=KEEP_RADIUS_KM)
            if k >= CALIBRATION_RUNS:
                return eph                      # only the calibration runs are kept in memory
            cache[k] = eph
        return cache[k]

    calibration = None
    if scenario.energy_cpu_hz is None and energy_cpu_hz != "draw":
        if energy_cpu_hz == "calibrate":
            used = range(min(runs, CALIBRATION_RUNS))
            value, calibration = calibrate_energy_cpu_hz(
                [ephemeris_of(k) for k in used], [seed + k for k in used], scenario)
        else:
            value = float(energy_cpu_hz)
        scenario = scenario.with_changes(energy_cpu_hz=value)

    out = _resolve(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path, visibility_path = out / "trace.csv", out / "visibility.csv"

    run_stats: List[dict] = []
    next_id = 1
    with trace_path.open("w", newline="", encoding="utf-8") as trace_file, \
            visibility_path.open("w", newline="", encoding="utf-8") as vis_file:
        trace = csv.writer(trace_file)
        trace.writerow(COLUMNS)
        vis = csv.writer(vis_file)
        vis.writerow(_VISIBILITY_COLUMNS)

        for k in range(runs):
            run_start = start_jd + k * run_spacing_s / 86400.0
            ephemeris = ephemeris_of(k)
            result = simulate(ephemeris, scenario, seed + k)

            offset = k * scenario.duration_s
            new_ids: Dict[int, int] = {}
            for row in result.rows:
                # Task ids restart at 1 in every run: renumber so they stay unique.
                original = row["Task ID"]
                if original not in new_ids:
                    new_ids[original] = next_id
                    next_id += 1
                row = dict(row)
                row["Task ID"] = new_ids[original]
                for column in _TIME_COLUMNS:
                    if row[column] != NA:
                        row[column] = row[column] + offset
                trace.writerow([_fmt(row[column]) for column in COLUMNS])
            for entry in result.visibility:
                vis.writerow([k, _fmt(entry["Time"] + offset)]
                             + [_fmt(entry[c]) for c in _VISIBILITY_COLUMNS[2:]])

            stats = dict(result.stats)
            stats.update(run=k, start_utc=orbits.jd_to_iso(run_start),
                         satellites_near_observer=len(ephemeris.names))
            run_stats.append(stats)

    manifest = split_into_worlds(csv_path=str(trace_path), output_dir=str(out / "worlds"),
                                 window_minutes=window_s)

    measured = {key: _mean([s[key] for s in run_stats])
                for key in ("success_rate", "energy_share_of_rejections", "executing_servers",
                            "servers_in_log", "share_of_top4_servers", "mean_time_in_system_s",
                            "mean_routing_hops", "dome_size_mean", "network_size_mean",
                            "access_points_mean", "generated", "completed", "rejected")}
    sha = None
    if orbit_file and Path(orbit_file).is_file():
        sha = hashlib.sha256(Path(orbit_file).read_bytes()).hexdigest()
    report = {
        "what": "Task log generated by data_processing/realistic_trace: a re-implementation "
                "of the simulator of Magliarisi et al. on real orbits. NOT the original log.",
        "orbits": {
            "source": constellation.source,
            "file_sha256": sha,
            "satellites": len(constellation),
            "median_epoch_utc": orbits.jd_to_iso(constellation.median_epoch_jd()),
        },
        "seed": seed,
        "runs": runs,
        "run_spacing_s": run_spacing_s,
        "first_start_utc": orbits.jd_to_iso(start_jd),
        "window_s": window_s,
        "scenario": scenario.to_dict(),
        "energy_cpu_hz_used": sorted({s["energy_cpu_hz"] for s in run_stats}),
        "calibration": calibration,
        "worlds": manifest.num_worlds,
        "rows": manifest.total_tasks + manifest.skipped_rows,
        "rows_without_arrival_time": manifest.skipped_rows,
        "mean_over_runs": measured,
        "published": {**PUBLISHED, "source": "magliarisi2026thesis, Sec. 6.5.1 (OrbitAware, "
                                             "10 req/s, 100 kJ)"},
        "per_run": run_stats,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prepare_realistic_worlds(constellation: orbits.Constellation,
                             output_dir: str = REALISTIC_DIR, **options) -> List[Path]:
    """Generate the trace and return the world paths, as ``load_world_paths`` would."""
    generate_trace(constellation, output_dir, **options)
    return load_world_paths(str(_resolve(output_dir) / "worlds"))


def format_report(report: dict) -> str:
    """The summary printed after a generation: what was written and how it
    compares with the published figures."""
    m, scenario = report["mean_over_runs"], report["scenario"]
    default = Scenario()
    comparable = all(scenario[key] == getattr(default, key) for key in
                     ("arrival_rate_per_s", "initial_energy_j", "policy", "duration_s"))
    lines = [
        f"Orbits: {report['orbits']['satellites']} satellites from {report['orbits']['source']}",
        f"        median epoch {report['orbits']['median_epoch_utc']}, "
        f"first run starts {report['first_start_utc']}",
        f"Runs:   {report['runs']} x {scenario['duration_s']:g} s at "
        f"{scenario['arrival_rate_per_s']:g} requests/s, policy {scenario['policy']}",
        f"Wrote   {report['rows']} rows, {report['worlds']} worlds "
        f"({report['rows_without_arrival_time']} rows without arrival time are not in any world)",
        "",
        f"{'mean over runs':34s} {'generated':>10s} {'published':>10s}",
    ]

    def line(label, key, pattern="{:.3f}", published=None):
        ref = pattern.format(published) if published is not None else "-"
        lines.append(f"{label:34s} {pattern.format(m[key]):>10s} {ref:>10s}")

    line("success rate", "success_rate",
         published=PUBLISHED["success_rate"] if comparable else None)
    line("share of rejections due to energy", "energy_share_of_rejections",
         published=PUBLISHED["energy_share_of_rejections"] if comparable else None)
    line("satellites in the dome (>= 40 deg)", "dome_size_mean", "{:.1f}")
    line("satellites in the network (> 25 deg)", "network_size_mean", "{:.1f}")
    line("distinct servers in the log", "servers_in_log", "{:.1f}")
    line("share of work on the top 4 servers", "share_of_top4_servers")
    line("mean time in system [s]", "mean_time_in_system_s")
    line("mean hops of the result's way back", "mean_routing_hops", "{:.2f}")
    lines.append("")
    if comparable:
        lines.append("Published: OrbitAware at 10 req/s and 100 kJ, magliarisi2026thesis Sec. 6.5.1.")
    else:
        lines.append("No published figure for these settings (only 10 req/s, 100 kJ, OrbitAware).")
    fit = report.get("calibration")
    if fit:
        lines.append(f"Energy CPU frequency fitted to the two published figures: "
                     f"{fit['energy_cpu_hz'] / 1e9:.2f} GHz. At the published setting, over "
                     f"{fit['runs_used']} run(s), it gives success {fit['success_rate']:.3f} and "
                     f"energy share {fit['energy_share_of_rejections']:.3f}.")
    else:
        used = ", ".join(f"{hz / 1e9:.2f}" for hz in report["energy_cpu_hz_used"])
        lines.append(f"Energy CPU frequency not fitted; used {used} GHz.")
    if "walker" in report["orbits"]["source"]:
        lines.append("IDEALISED CONSTELLATION, not Starlink: for trying the program only. "
                     "Never report these numbers.")
    lines.append("This is generated data, not the original log: say so wherever you use it.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    default = Scenario()
    parser = argparse.ArgumentParser(
        prog="python3 -m data_processing.realistic_trace",
        description="Generate a realistic task log from real Starlink orbits and split it "
                    "into worlds (see docs/onboarding/realistic-data.html).")
    source = parser.add_argument_group("orbits")
    source.add_argument("--orbits", default=ORBITS_PATH,
                        help=f"Orbital elements, OMM CSV or TLE (default {ORBITS_PATH}).")
    source.add_argument("--download", action="store_true",
                        help="Download the current Starlink elements from CelesTrak to --orbits "
                             "first. CelesTrak allows one download every 2 hours.")
    source.add_argument("--walker", action="store_true",
                        help="Use an idealised Walker constellation instead of a file. No "
                             "network needed, NOT Starlink: for trying the program only.")
    source.add_argument("--start", default=None,
                        help="UTC start of the first run, e.g. 2026-11-19T10:00:00. Default: "
                             "the median epoch of the orbit file, where SGP4 is most accurate.")
    parser.add_argument("--output-dir", default=REALISTIC_DIR)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="Independent 300 s simulations, 30 worlds each (default "
                             f"{DEFAULT_RUNS}).")
    parser.add_argument("--run-spacing", type=float, default=DEFAULT_RUN_SPACING_S,
                        help="Real seconds between the starts of two runs (default "
                             f"{DEFAULT_RUN_SPACING_S:g}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    model = parser.add_argument_group("scenario (defaults: the published settings)")
    model.add_argument("--rate", type=float, default=default.arrival_rate_per_s,
                       help="Requests per second; the thesis uses 2, 4, 6, 8 and 10.")
    model.add_argument("--duration", type=float, default=default.duration_s,
                       help="Simulated seconds per run.")
    model.add_argument("--policy", choices=("OrbitAware", "DTS"), default=default.policy,
                       help="Scheduler of the simulated system.")
    model.add_argument("--energy-budget", type=float, default=default.initial_energy_j,
                       help="Initial battery of every satellite in joules; the thesis uses "
                            "80000, 100000 and 120000.")
    model.add_argument("--energy-cpu-hz", default="calibrate",
                       help="CPU frequency used to charge energy: 'calibrate' (default) fits "
                            "it to the published figures, 'draw' draws it per run like the "
                            "original simulator, or give a number in hertz.")
    model.add_argument("--log-batch", action="store_true",
                       help="Also write the background batch requests as rows.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    scenario = Scenario().with_changes(
        arrival_rate_per_s=args.rate, duration_s=args.duration, policy=args.policy,
        initial_energy_j=args.energy_budget, log_batch_rows=args.log_batch)
    energy = args.energy_cpu_hz
    if energy not in ("calibrate", "draw"):
        try:
            energy = float(energy)
        except ValueError:
            raise SystemExit("--energy-cpu-hz takes 'calibrate', 'draw' or a number in hertz")

    orbit_file = None
    if args.walker:
        constellation = orbits.walker_constellation()
    else:
        path = _resolve(args.orbits)
        if args.download:
            print(f"Downloading Starlink orbital elements from CelesTrak to {path} ...")
            orbits.download_orbits(str(path))
        if not path.is_file():
            raise SystemExit(
                f"No orbit file at {path}.\n"
                "  Download the current Starlink elements (about 1.7 MB, from celestrak.org):\n"
                "      python3 -m data_processing.realistic_trace --download\n"
                "  or pass a TLE / OMM CSV file with --orbits FILE,\n"
                "  or try the program offline with --walker (idealised, not Starlink).")
        constellation = orbits.load_orbits(str(path))
        orbit_file = str(path)

    start_jd = _iso_to_jd(args.start) if args.start else None
    if energy == "calibrate":
        print("Fitting the energy CPU frequency to the published figures "
              "(about 70 short simulations) ...")
    report = generate_trace(constellation, args.output_dir, scenario=scenario, runs=args.runs,
                            seed=args.seed, start_jd=start_jd, run_spacing_s=args.run_spacing,
                            orbit_file=orbit_file, energy_cpu_hz=energy)
    print(format_report(report))
    print(f"\nFiles are in {_resolve(args.output_dir)}. Train on them with:\n"
          f"    ./run realistic --no-start-prompt\n"
          f"or  python3 main.py --worlds-dir {args.output_dir}/worlds --no-start-prompt")


if __name__ == "__main__":
    main()
