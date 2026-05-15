import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_snapshot(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _build_points(snapshot: Dict) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    satellites = snapshot.get("satellites", [])
    jobs = snapshot.get("jobs", [])
    current_minute = _to_float(snapshot.get("current_minute"), 0.0)

    satellite_names = [str(sat.get("name", f"SAT-{idx}")) for idx, sat in enumerate(satellites)]
    if not satellite_names:
        satellite_names = ["UNASSIGNED"]

    lane_by_satellite = {name: idx for idx, name in enumerate(satellite_names)}
    next_free_time = {name: current_minute for name in satellite_names}

    satellite_x: List[float] = []
    satellite_y: List[float] = []
    task_x: List[float] = []
    task_y: List[float] = []

    # One red dot per satellite at the snapshot timestamp.
    for lane, sat_name in enumerate(satellite_names):
        satellite_x.append(current_minute)
        satellite_y.append(float(lane))
        next_free_time.setdefault(sat_name, current_minute)

    sorted_jobs = sorted(
        jobs,
        key=lambda job: (
            _to_float(job.get("arrivalTime"), current_minute),
            int(job.get("id", 0)),
        ),
    )

    for job in sorted_jobs:
        sat_name = str(job.get("executionServerName") or "UNASSIGNED")
        if sat_name not in lane_by_satellite:
            lane_by_satellite[sat_name] = len(satellite_names)
            satellite_names.append(sat_name)
            next_free_time[sat_name] = current_minute
            satellite_x.append(current_minute)
            satellite_y.append(float(lane_by_satellite[sat_name]))

        lane = float(lane_by_satellite[sat_name])
        arrival_time = _to_float(job.get("arrivalTime"), current_minute)
        transfer_time = _to_float(job.get("transferTime"), 0.0)
        execution_time = _to_float(job.get("executionTime"), 0.0)

        start_time = max(arrival_time, next_free_time[sat_name], current_minute)
        end_time = start_time + transfer_time + execution_time
        next_free_time[sat_name] = end_time

        # Blue dot: task entering execution timeline.
        task_x.append(start_time)
        task_y.append(lane)
        # Red dot: satellite state update after processing the task.
        satellite_x.append(end_time)
        satellite_y.append(lane)

    return satellite_names, satellite_x, satellite_y, task_x, task_y


def build_timeline_plot(
    snapshot_file: Path,
    output_file: Path,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "This script requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    snapshot = _load_snapshot(snapshot_file)
    satellite_names, satellite_x, satellite_y, task_x, task_y = _build_points(snapshot)

    fig_height = max(4.0, 0.45 * len(satellite_names) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    ax.scatter(
        satellite_x,
        satellite_y,
        facecolors="none",
        edgecolors="red",
        linewidths=1.1,
        s=58,
        alpha=0.95,
        label="Satellites",
        zorder=2,
    )
    ax.scatter(
        task_x,
        task_y,
        color="royalblue",
        s=32,
        alpha=0.95,
        label="Tasks",
        zorder=3,
    )

    ax.set_yticks(range(len(satellite_names)))
    ax.set_yticklabels(satellite_names)
    ax.set_xlabel("Time [minutes]")
    ax.set_ylabel("Satellite")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(title or f"Snapshot timeline: {snapshot_file.name}")

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a timeline graph from a simulation snapshot JSON file."
    )
    parser.add_argument("snapshot_file", type=Path, help="Path to a snapshot JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image file path (defaults to <snapshot_stem>_timeline.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional custom chart title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image DPI (default: 150).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or args.snapshot_file.with_name(f"{args.snapshot_file.stem}_timeline.png")
    build_timeline_plot(
        snapshot_file=args.snapshot_file,
        output_file=output,
        title=(args.title.strip() or None),
        dpi=max(72, args.dpi),
    )
    print(f"Timeline graph saved to: {output}")
