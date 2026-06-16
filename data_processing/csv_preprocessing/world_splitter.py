"""Split a raw simulation CSV into pre-processed per-window "world" files.

Why this module exists
----------------------
The raw dataset (e.g. ``data/example.csv``) is the task log of a *single*
300-minute LEO simulation.  Two properties of that data matter for the GA:

1. **Arrival times are only meaningful relatively.**  The absolute minute at
   which a task arrived carries no information for the scheduler; what matters
   is how arrivals *evolve* relative to one another (their ordering and the
   gaps between them).  We therefore normalize each world by shifting its
   earliest arrival to ``t = 0`` while preserving the inter-arrival gaps.

2. **The simulation world changes over time.**  As satellites orbit, the set
   of reachable servers and the load profile change.  Rather than re-deriving
   a "world" from the whole CSV on every GA reset, we pre-split the log into
   fixed-length time windows so each world is an already-prepared CSV.

Output
------
For an input CSV this writes, into ``output_dir`` (default ``data/worlds``):

* ``world_000.csv``, ``world_001.csv``, ... — one file per non-empty window,
  in chronological order.  Every original column is preserved and a new
  ``Arrival Time (Normalized)`` column is appended (full precision).
* ``manifest.json`` — ordered index of the worlds with per-world metadata
  (window index, task count, arrival min/max/span).  Downstream code reads
  this to iterate worlds deterministically.

Run it from the repository root::

    python -m data_processing.csv_preprocessing.world_splitter \
        --csv data/example.csv --output-dir data/worlds --window-minutes 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Make the repository root importable so that ``utils`` resolves whether this
#    file is run as a script or imported as part of the package. ──────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import _load_csv_rows, _parse_csv_number  # noqa: E402


# ── Column names ─────────────────────────────────────────────────────────────
# Header keys are stripped of surrounding whitespace by ``_load_csv_rows``.
SYSTEM_ARRIVAL_COLUMN = "Arrival Time (System)"
NORMALIZED_ARRIVAL_COLUMN = "Arrival Time (Normalized)"

# Files this tool is allowed to remove when cleaning an output directory.
_MANIFEST_NAME = "manifest.json"
_WORLD_GLOB = "world_*.csv"


@dataclass
class WorldEntry:
    """Metadata for a single generated world file."""

    file: str
    window_index: int
    tasks: int
    arrival_min: float
    arrival_max: float
    span: float


@dataclass
class WorldManifest:
    """Ordered index of every world produced from one source CSV."""

    source_csv: str
    window_minutes: float
    arrival_column: str
    normalized_column: str
    generated_at: str
    num_worlds: int
    total_tasks: int
    skipped_rows: int
    worlds: List[WorldEntry]

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2)


def _resolve(path_like: str) -> Path:
    """Resolve a path relative to the repository root when not absolute."""
    path = Path(path_like)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _format_number(value: float) -> str:
    """Round-trippable textual form of a float (re-parses via _parse_csv_number)."""
    return repr(value)


def _clean_output_dir(output_dir: Path) -> None:
    """Remove previously generated world files and manifest (nothing else)."""
    if not output_dir.exists():
        return
    for stale in output_dir.glob(_WORLD_GLOB):
        stale.unlink()
    manifest = output_dir / _MANIFEST_NAME
    if manifest.exists():
        manifest.unlink()


def split_into_worlds(
    csv_path: str = "data/example.csv",
    output_dir: str = "data/worlds",
    window_minutes: float = 10.0,
    arrival_column: str = SYSTEM_ARRIVAL_COLUMN,
    normalized_column: str = NORMALIZED_ARRIVAL_COLUMN,
    min_tasks_per_world: int = 1,
    clean: bool = True,
) -> WorldManifest:
    """Split ``csv_path`` into per-window world CSVs under ``output_dir``.

    Args:
        csv_path: Source CSV (relative paths resolve against the repo root).
        output_dir: Destination directory for ``world_XXX.csv`` + manifest.
        window_minutes: Length of each time window, in the same units as the
            arrival column (minutes for the LEO dataset).
        arrival_column: Column used to bucket rows into windows.
        normalized_column: Name of the appended relative-arrival column.
        min_tasks_per_world: Windows with fewer tasks than this are dropped.
        clean: Remove pre-existing ``world_*.csv``/manifest before writing.

    Returns:
        The :class:`WorldManifest` describing every world that was written.
    """
    if window_minutes <= 0:
        raise ValueError(f"window_minutes must be > 0, got {window_minutes!r}")

    source = _resolve(csv_path)
    rows = _load_csv_rows(str(source), limit=None)
    if not rows:
        raise ValueError(f"No data rows found in {source}")

    # Preserve original column order and append the normalized column.
    fieldnames = list(rows[0].keys())
    if normalized_column not in fieldnames:
        fieldnames.append(normalized_column)

    # Bucket every parseable row into a time window.
    buckets: Dict[int, list] = defaultdict(list)
    skipped = 0
    for row in rows:
        arrival = _parse_csv_number(row.get(arrival_column))
        if arrival is None:
            skipped += 1
            continue
        window_index = int(arrival // window_minutes)
        buckets[window_index].append((arrival, row))

    out_dir = _resolve(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_output_dir(out_dir)

    import csv  # local import keeps module import cost low

    entries: List[WorldEntry] = []
    total_tasks = 0
    for window_index in sorted(buckets):
        items = buckets[window_index]
        if len(items) < min_tasks_per_world:
            continue

        # Chronological order within the world, then shift earliest to zero.
        items.sort(key=lambda pair: pair[0])
        window_min = items[0][0]
        window_max = items[-1][0]

        file_name = f"world_{len(entries):03d}.csv"
        with (out_dir / file_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for arrival, row in items:
                out_row = dict(row)
                out_row[normalized_column] = _format_number(arrival - window_min)
                writer.writerow(out_row)

        entries.append(
            WorldEntry(
                file=file_name,
                window_index=window_index,
                tasks=len(items),
                arrival_min=window_min,
                arrival_max=window_max,
                span=window_max - window_min,
            )
        )
        total_tasks += len(items)

    manifest = WorldManifest(
        source_csv=str(Path(csv_path)),
        window_minutes=window_minutes,
        arrival_column=arrival_column,
        normalized_column=normalized_column,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        num_worlds=len(entries),
        total_tasks=total_tasks,
        skipped_rows=skipped,
        worlds=entries,
    )
    (out_dir / _MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def load_world_paths(worlds_dir: str = "data/worlds") -> List[Path]:
    """Return the ordered list of world CSV paths for ``worlds_dir``.

    Prefers ``manifest.json`` (authoritative order); falls back to a sorted
    glob of ``world_*.csv`` so the function still works if the manifest is
    missing.  Returns an empty list when nothing has been generated yet.
    """
    directory = _resolve(worlds_dir)
    if not directory.is_dir():
        return []

    manifest_path = directory / _MANIFEST_NAME
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = [directory / world["file"] for world in data.get("worlds", [])]
            paths = [p for p in paths if p.is_file()]
            if paths:
                return paths
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # fall back to globbing

    return sorted(directory.glob(_WORLD_GLOB))


# ── CLI ──────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a raw simulation CSV into per-window 'world' CSVs "
        "with relatively-normalized arrival times."
    )
    parser.add_argument("--csv", default="data/example.csv",
                        help="Source CSV (default: data/example.csv).")
    parser.add_argument("--output-dir", default="data/worlds",
                        help="Output directory for world files (default: data/worlds).")
    parser.add_argument("--window-minutes", type=float, default=10.0,
                        help="Time-window length used to split worlds (default: 10).")
    parser.add_argument("--arrival-column", default=SYSTEM_ARRIVAL_COLUMN,
                        help=f"Arrival column to bucket on (default: {SYSTEM_ARRIVAL_COLUMN!r}).")
    parser.add_argument("--normalized-column", default=NORMALIZED_ARRIVAL_COLUMN,
                        help="Name of the appended relative-arrival column.")
    parser.add_argument("--min-tasks-per-world", type=int, default=1,
                        help="Drop windows with fewer tasks than this (default: 1).")
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep any pre-existing world files instead of clearing them.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    manifest = split_into_worlds(
        csv_path=args.csv,
        output_dir=args.output_dir,
        window_minutes=args.window_minutes,
        arrival_column=args.arrival_column,
        normalized_column=args.normalized_column,
        min_tasks_per_world=args.min_tasks_per_world,
        clean=not args.no_clean,
    )
    print(
        f"Wrote {manifest.num_worlds} world(s) "
        f"({manifest.total_tasks} tasks, {manifest.skipped_rows} skipped) "
        f"to {_resolve(args.output_dir)}"
    )
    if manifest.worlds:
        first, last = manifest.worlds[0], manifest.worlds[-1]
        print(
            f"  first: {first.file} — {first.tasks} tasks, "
            f"span {first.span:.2f} min (window #{first.window_index})"
        )
        print(
            f"  last:  {last.file} — {last.tasks} tasks, "
            f"span {last.span:.2f} min (window #{last.window_index})"
        )


if __name__ == "__main__":
    main()
