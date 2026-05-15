import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _numeric_stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def _collect_numeric(items: List[Any], attr_name: str) -> List[float]:
    values: List[float] = []
    for item in items:
        raw_value = getattr(item, attr_name, None)
        if raw_value is None:
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            continue
    return values


def snapshot_to_dict(
    satellites: List[Any],
    jobs: List[Any],
    current_minute: Optional[float] = None,
    energy_per_minute: Optional[List[float]] = None,
) -> Dict:
    """Convert satellites and jobs to a serializable dictionary."""
    sats_data = []
    for sat in satellites:
        sats_data.append({
            "id": sat.id,
            "name": sat.name,
            "cpuCapacity": sat.cpuCapacity,
            "networkCapacity": sat.networkCapacity,
            "latency": sat.latency,
            "bandwidth": sat.bandwidth,
            "remainingEnergy": sat.remainingEnergy,
            "completedTasks": sat.completedTasks,
            "cpuBusyUntil": sat.cpuBusyUntil,
            "networkBusyUntil": sat.networkBusyUntil,
            "elevationAngle": sat.elevationAngle,
            "isAccessPoint": getattr(sat, "isAccessPoint", False),
            "neighborsCount": len(getattr(sat, "neighbors", [])),
        })

    jobs_data = []
    for job in jobs:
        jobs_data.append({
            "id": job.id,
            "arrivalTime": job.arrivalTime,
            "type": job.type,
            "ram": job.ram,
            "disk": job.disk,
            "imageSize": job.imageSize,
            "executionTime": job.executionTime,
            "transferTime": job.transferTime,
            "numberOfHops": job.numberOfHops,
            "executionServerName": job.executionServerName,
        })

    execution_times = _collect_numeric(jobs, "executionTime")
    transfer_times = _collect_numeric(jobs, "transferTime")
    hops = _collect_numeric(jobs, "numberOfHops")
    arrivals = _collect_numeric(jobs, "arrivalTime")
    energies = _collect_numeric(satellites, "remainingEnergy")

    return {
        "timestamp": datetime.now().isoformat(),
        "current_minute": current_minute,
        "satellites_count": len(satellites),
        "jobs_count": len(jobs),
        "energy_per_minute": list(energy_per_minute or []),
        "stats": {
            "satellite_energy": _numeric_stats(energies),
            "job_execution_time": _numeric_stats(execution_times),
            "job_transfer_time": _numeric_stats(transfer_times),
            "job_hops": _numeric_stats(hops),
            "job_arrival_time": _numeric_stats(arrivals),
            "jobs_by_type": dict(Counter((job.type or "UNKNOWN") for job in jobs)),
            "jobs_by_execution_server": dict(
                Counter((job.executionServerName or "UNKNOWN") for job in jobs)
            ),
        },
        "satellites": sats_data,
        "jobs": jobs_data,
    }


def save_snapshot(
    satellites: List[Any],
    jobs: List[Any],
    output_dir: str = "snapshots",
    prefix: str = "snapshot",
    generation: Optional[int] = None,
    reset_count: Optional[int] = None,
    current_minute: Optional[float] = None,
    energy_per_minute: Optional[List[float]] = None,
) -> str:
    """Save a snapshot to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix_parts = []
    if reset_count is not None:
        suffix_parts.append(f"reset{reset_count}")
    if generation is not None:
        suffix_parts.append(f"gen{generation:04d}")

    suffix = "_".join(suffix_parts) if suffix_parts else "initial"
    filename = f"{prefix}_{suffix}.json"
    filepath = output_path / filename

    data = snapshot_to_dict(
        satellites,
        jobs,
        current_minute=current_minute,
        energy_per_minute=energy_per_minute,
    )
    with filepath.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)

    return str(filepath)


def save_simulation_snapshot(
    snapshot: Any,
    output_dir: str = "snapshots",
    prefix: str = "snapshot",
    generation: Optional[int] = None,
    reset_count: Optional[int] = None,
) -> str:
    """Persist a scheduler.SimulationSnapshot to JSON."""
    return save_snapshot(
        satellites=snapshot.satellites,
        jobs=snapshot.jobs,
        output_dir=output_dir,
        prefix=prefix,
        generation=generation,
        reset_count=reset_count,
        current_minute=getattr(snapshot, "currentMinute", None),
        energy_per_minute=getattr(snapshot, "energyPerMinute", None),
    )


def print_snapshot_summary(
    satellites: List[Any],
    jobs: List[Any],
    current_minute: Optional[float] = None,
) -> None:
    """Print a human-readable summary of a simulation snapshot."""
    print("\n" + "=" * 72)
    print("SIMULATION SNAPSHOT")
    print("=" * 72)
    if current_minute is not None:
        print(f"Current minute: {current_minute:.1f}")
    print(f"Satellites: {len(satellites)} | Jobs: {len(jobs)}")

    print(f"\nSatellites ({len(satellites)}):")
    if satellites:
        for sat in sorted(satellites, key=lambda item: str(getattr(item, "name", ""))):
            print(
                f"  {sat.name:18s} | CPU: {sat.cpuCapacity:6.1f} | Net: {sat.networkCapacity:6.1f} | "
                f"BW: {sat.bandwidth:7.1f} | Lat: {sat.latency:5.1f} | Energy: {sat.remainingEnergy:6.1f}"
            )
    else:
        print("  (no satellites)")

    if jobs:
        jobs_by_server = Counter((job.executionServerName or "UNKNOWN") for job in jobs)
        jobs_by_type = Counter((job.type or "UNKNOWN") for job in jobs)

        print(f"\nJobs by execution server ({len(jobs_by_server)}):")
        for server, count in sorted(jobs_by_server.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {server:30s}: {count:4d}")

        print(f"\nJobs by type ({len(jobs_by_type)}):")
        for job_type, count in sorted(jobs_by_type.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {job_type:30s}: {count:4d}")

        execution_stats = _numeric_stats(_collect_numeric(jobs, "executionTime"))
        transfer_stats = _numeric_stats(_collect_numeric(jobs, "transferTime"))
        hop_stats = _numeric_stats(_collect_numeric(jobs, "numberOfHops"))
        arrival_stats = _numeric_stats(_collect_numeric(jobs, "arrivalTime"))

        if execution_stats is not None:
            print(
                "\nExecution time (min/avg/max): "
                f"{execution_stats['min']:.1f} / {execution_stats['avg']:.1f} / {execution_stats['max']:.1f}"
            )
        if transfer_stats is not None:
            print(
                "Transfer time  (min/avg/max): "
                f"{transfer_stats['min']:.1f} / {transfer_stats['avg']:.1f} / {transfer_stats['max']:.1f}"
            )
        if hop_stats is not None:
            print(
                "Job hops       (min/avg/max): "
                f"{hop_stats['min']:.1f} / {hop_stats['avg']:.1f} / {hop_stats['max']:.1f}"
            )
        if arrival_stats is not None:
            print(
                "Arrival minute (min/avg/max): "
                f"{arrival_stats['min']:.1f} / {arrival_stats['avg']:.1f} / {arrival_stats['max']:.1f}"
            )
    else:
        print("\nJobs: (no jobs in snapshot)")

    print("=" * 72 + "\n")


def print_simulation_snapshot_summary(snapshot: Any) -> None:
    """Print summary directly from scheduler.SimulationSnapshot."""
    print_snapshot_summary(
        satellites=snapshot.satellites,
        jobs=snapshot.jobs,
        current_minute=getattr(snapshot, "currentMinute", None),
    )
