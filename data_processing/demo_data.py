"""Synthetic demo data: run the whole pipeline without the real simulation log.

Why this module exists
----------------------
The real input, ``data/example.csv``, is the task log of the satellite
simulator of Magliarisi et al.  It is not redistributed with the code (see
"Data" in readme.md), so on a fresh clone ``main.py`` has nothing to train on.
That makes the project impossible to demonstrate, or to try out, before the
data arrives.

This module writes a *synthetic* task log with the same columns as the real
one, then splits it into worlds with the real splitter
(:func:`data_processing.csv_preprocessing.split_into_worlds`).  From there on
everything is the normal code path: ``data_factory`` turns rows into jobs and
satellites, the C++ core evolves and scores formulas.  ``main.py --demo`` (or
``./run demo``) uses it.

What the numbers mean: NOTHING
------------------------------
The generator is a toy.  It is not fitted to the real trace, it does not model
orbits, and its distributions were chosen only so that every column varies and
the GP has something to learn (faster servers really do finish jobs sooner).
Results obtained on demo data must never be reported, compared with the
thesis, or used to choose a parameter.  The files go to ``data/demo/``,
separate from ``data/worlds/``, so the two can never be mixed up.

How a row is generated
----------------------
* ``servers`` satellites, each with a fixed speed factor, link factor,
  starting energy and typical queue length, drawn once.
* Arrivals form a Poisson process of rate ``tasks_per_minute`` over
  ``minutes`` minutes (exponential inter-arrival gaps), so 10-minute windows
  hold about ``10 * tasks_per_minute`` tasks, as in the real log (~100).
* Each task gets a type, an amount of work and an image size; its execution
  time is work / server speed and its transfer time image size / link factor.
  The server is drawn uniformly, so the log does not leak a "good" schedule.
* The server's remaining energy drains linearly over the run, with noise.

Run it by hand (from the repository root)::

    python3 -m data_processing.demo_data            # writes data/demo/
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_processing.csv_preprocessing import (  # noqa: E402
    SYSTEM_ARRIVAL_COLUMN,
    load_world_paths,
    split_into_worlds,
)

# Fixed seed of the demo data.  The data is the same on every machine and every
# run; the run's own --seed only changes the evolution.
DEMO_SEED = 2026
DEMO_DIR = "data/demo"

# Column names read by data_factory.py (plus the ones it ignores, so the file
# looks like the real log).  The real header has extra spaces around the
# arrival column; utils._load_csv_rows strips them, so plain names are fine.
COLUMNS = [
    "Task ID",
    "Task Type",
    "Status",
    SYSTEM_ARRIVAL_COLUMN,
    "Execution time",
    "transfer_time",
    "Image_Size_MB",
    "Num Hops Routing",
    "Queue length",
    "Server Name",
    "Remaining_energy [J]",
]

# (type, probability, mean work, image size range).  The first two names are
# the ones data_factory gives extra RAM and disk; any other name is generic.
_TASK_TYPES = [
    ("CPU_Intensive",          0.40, 12.0, (5.0, 20.0)),
    ("CPU_and_Data_Intensive", 0.25, 16.0, (20.0, 60.0)),
    ("Generic_Service",        0.35,  6.0, (2.0, 15.0)),
]


def write_demo_trace(
    csv_path: str,
    *,
    minutes: float = 60.0,
    tasks_per_minute: float = 10.0,
    servers: int = 12,
    seed: int = DEMO_SEED,
) -> int:
    """Write a synthetic task log to ``csv_path``; return the number of rows.

    The output is a pure function of the arguments: the same seed gives the
    same file, byte for byte.
    """
    if minutes <= 0 or tasks_per_minute <= 0 or servers <= 0:
        raise ValueError("minutes, tasks_per_minute and servers must be > 0")

    rng = random.Random(seed)
    profiles = []
    for k in range(servers):
        profiles.append({
            "name": f"DEMO-SAT-{k + 1:02d}",
            "speed": rng.uniform(0.6, 1.6),        # work units per second
            "link": rng.uniform(0.6, 1.6),         # MB per second
            "energy": rng.uniform(2000.0, 6000.0),  # joules at minute 0
            "drain": rng.uniform(5.0, 25.0),       # joules per minute
            "queue": rng.uniform(0.0, 8.0),        # typical queue length
        })

    names = [t[0] for t in _TASK_TYPES]
    weights = [t[1] for t in _TASK_TYPES]
    by_name = {t[0]: t for t in _TASK_TYPES}

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    arrival = 0.0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        while True:
            arrival += rng.expovariate(tasks_per_minute)   # Poisson arrivals
            if arrival >= minutes:
                break
            task_type = rng.choices(names, weights)[0]
            _, _, mean_work, (img_lo, img_hi) = by_name[task_type]
            server = profiles[rng.randrange(servers)]

            work = rng.lognormvariate(0.0, 0.4) * mean_work
            image = rng.uniform(img_lo, img_hi)
            execution = work / server["speed"]
            transfer = image / (8.0 * server["link"])
            queue = max(0, round(rng.gauss(server["queue"], 1.5)))
            energy = max(0.0, server["energy"] - server["drain"] * arrival
                         + rng.gauss(0.0, 50.0))

            writer.writerow([
                rows,
                task_type,
                "Completed",
                f"{arrival:.4f}",
                f"{execution:.4f}",
                f"{transfer:.4f}",
                f"{image:.3f}",
                rng.randint(1, 5),
                queue,
                server["name"],
                f"{energy:.2f}",
            ])
            rows += 1
    return rows


def prepare_demo_worlds(
    output_dir: str = DEMO_DIR,
    *,
    window_minutes: float = 10.0,
    seed: int = DEMO_SEED,
    **trace_options,
) -> List[Path]:
    """Write ``<output_dir>/trace.csv`` and split it into ``<output_dir>/worlds/``.

    Returns the world paths in order, exactly as ``load_world_paths`` would.
    The files are rewritten on every call; that takes a few milliseconds and
    keeps them in step with this module.
    """
    out = Path(output_dir)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    trace = out / "trace.csv"
    worlds = out / "worlds"
    write_demo_trace(str(trace), seed=seed, **trace_options)
    split_into_worlds(csv_path=str(trace), output_dir=str(worlds),
                      window_minutes=window_minutes)
    return load_world_paths(str(worlds))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write synthetic demo data (NOT real results) to data/demo/.")
    parser.add_argument("--output-dir", default=DEMO_DIR)
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--tasks-per-minute", type=float, default=10.0)
    parser.add_argument("--servers", type=int, default=12)
    args = parser.parse_args(argv)
    paths = prepare_demo_worlds(args.output_dir, seed=args.seed, minutes=args.minutes,
                                tasks_per_minute=args.tasks_per_minute,
                                servers=args.servers)
    print(f"Wrote synthetic demo data: {len(paths)} world(s) in "
          f"{paths[0].parent if paths else args.output_dir}")


if __name__ == "__main__":
    main()
