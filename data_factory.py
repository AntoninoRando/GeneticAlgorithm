import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple
from scheduler import Satellite, Job


"""
// NON CONSIDERARE (“Feature” “Colonna”)
// taskId A 
// status C
// arrivalTime D
// arrivalTime E
// startTime F
// endTime G
// timeInTheSystem J
// serverName L
// execAfterSet R
// remainingEnergy% W
// rejectionReason X
// routingInitTime Y
// routingEndTime Z

"""

# ── Satellite name pool ───────────────────────────────────────────────────────
_SAT_NAMES = [
    "SEO-A", "SEO-B", "SEO-C", "SEO-D", "SEO-E",
    "LEO-1", "LEO-2", "LEO-3", "MEO-X", "MEO-Y",
]

# ── Job name pool ─────────────────────────────────────────────────────────────
_JOB_PREFIXES = [
    "Target", "Survey", "Relay", "Scan", "Monitor",
    "Capture", "Track", "Probe", "Sense", "Map",
]

_IGNORED_CSV_COLUMNS = {
    "Task ID",
    "Status",
    "Arrival Time (System)",
    "Arrival Time (Queue)",
    "Start Time",
    "End Time",
    "Time in system",
    "Server Name",
    "Exec_after_set",
    "Remaining_energy [%]",
    "Rejection Reason",
    "Routing Init Time",
    "Routing End Time",
}


def _parse_csv_number(raw_value: Optional[str]) -> Optional[float]:
    if raw_value is None:
        return None

    value = raw_value.strip().replace('"', "")
    if not value:
        return None

    upper = value.upper()
    if upper in {"N/A", "NA", "NONE", "NULL"}:
        return None

    value = value.replace(" ", "")

    # The dataset uses locale formatting (e.g. 12.345.678 and "2,17E+10").
    if "," in value:
        if "E" in upper:
            value = value.replace(".", "").replace(",", ".")
        elif "." in value and value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        elif "." not in value:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
    elif value.count(".") > 1:
        value = value.replace(".", "")

    try:
        return float(value)
    except ValueError:
        return None


def _rescale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    ratio = max(0.0, min(1.0, ratio))
    return dst_min + ratio * (dst_max - dst_min)


def _load_csv_rows(
    csv_path: str,
    limit: Optional[int] = None,
    random_sample: bool = False,
    sample_seed: Optional[int] = None,
) -> list:
    base_dir = Path(__file__).resolve().parent
    path = Path(csv_path)
    if not path.is_absolute():
        path = base_dir / path

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if limit is not None:
        limit = max(0, limit)
        if limit < len(rows):
            if random_sample:
                rng = random.Random(sample_seed)
                rows = rng.sample(rows, limit)
            else:
                rows = rows[:limit]

    return rows


def create_jobs_from_csv(
    csv_path: str = "data/example.csv",
    limit: Optional[int] = None,
    snapshot_minute: float = 0.0,
    random_sample: bool = False,
    sample_seed: Optional[int] = None,
) -> list:
    rows = _load_csv_rows(
        csv_path,
        limit,
        random_sample=random_sample,
        sample_seed=sample_seed,
    )

    if not rows:
        return []

    task_rows = []
    for row in rows:
        # Keep only non-ignored fields from the CSV comments and build our
        # internal data model from these.
        filtered = {k: v for k, v in row.items() if k not in _IGNORED_CSV_COLUMNS}

        task_rows.append({
            "task_type": (filtered.get("Task Type") or "Generic_Service").strip() or "Generic_Service",
            "execution_raw": _parse_csv_number(filtered.get("Execution time")),
            "transfer_raw": _parse_csv_number(filtered.get("transfer_time")),
            "image_raw": _parse_csv_number(filtered.get("Image_Size_MB")),
            "hops_raw": _parse_csv_number(filtered.get("Num Hops Routing")) or _parse_csv_number(filtered.get("Num Hops")),
            "queue_raw": _parse_csv_number(filtered.get("Queue length")),
            "server_name": (row.get("Server Name") or "").strip(),
        })

    def collect(key: str, fallback: float) -> Tuple[float, float]:
        values = [row[key] for row in task_rows if row[key] is not None]
        if not values:
            return fallback, fallback
        return min(values), max(values)

    execution_min, execution_max = collect("execution_raw", 1.0)
    transfer_min, transfer_max = collect("transfer_raw", 1.0)
    image_min, image_max = collect("image_raw", 1.0)
    queue_min, queue_max = collect("queue_raw", 1.0)

    jobs = []
    for idx, row in enumerate(task_rows):
        execution_raw = row["execution_raw"] if row["execution_raw"] is not None else (execution_min + execution_max) / 2.0
        transfer_raw = row["transfer_raw"] if row["transfer_raw"] is not None else (transfer_min + transfer_max) / 2.0
        image_raw = row["image_raw"] if row["image_raw"] is not None else (image_min + image_max) / 2.0
        queue_raw = row["queue_raw"] if row["queue_raw"] is not None else (queue_min + queue_max) / 2.0

        execution_time = int(round(_rescale(execution_raw, execution_min, execution_max, 5.0, 30.0)))
        transfer_time = int(round(_rescale(transfer_raw, transfer_min, transfer_max, 1.0, 8.0)))
        image_size = _rescale(image_raw, image_min, image_max, 8.0, 24.0)
        queue_scale = _rescale(queue_raw, queue_min, queue_max, 0.0, 1.0)

        hops_value = int(round(row["hops_raw"])) if row["hops_raw"] is not None else 1
        hops_value = max(1, min(5, hops_value))

        job = Job()
        job.id = idx
        # Keep task arrivals aligned with the snapshot minute used by the evaluator.
        job.arrivalTime = int(snapshot_minute)
        job.type = row["task_type"]
        job.imageSize = image_size
        job.executionTime = max(1, execution_time)
        job.transferTime = max(0, transfer_time)
        job.numberOfHops = hops_value
        job.executionServerName = row["server_name"]

        if job.type == "CPU_and_Data_Intensive":
            job.ram = 8.0 + (8.0 * queue_scale)
            job.disk = image_size * 3.0
        elif job.type == "CPU_Intensive":
            job.ram = 4.0 + (8.0 * queue_scale)
            job.disk = image_size * 2.0
        else:
            job.ram = 1.0 + (5.0 * queue_scale)
            job.disk = image_size * 1.5

        jobs.append(job)

    return jobs


def create_satellites_from_csv(
    csv_path: str = "data/example.csv",
    limit: Optional[int] = None,
    max_satellites: Optional[int] = None,
    seed: int = None,
    random_sample: bool = False,
    sample_seed: Optional[int] = None,
) -> list:
    rows = _load_csv_rows(
        csv_path,
        limit,
        random_sample=random_sample,
        sample_seed=sample_seed,
    )
    if not rows:
        return create_satellites(n=max_satellites or 3, seed=seed)

    per_server = defaultdict(lambda: {
        "count": 0,
        "execution_sum": 0.0,
        "execution_count": 0,
        "transfer_sum": 0.0,
        "transfer_count": 0,
        "queue_sum": 0.0,
        "queue_count": 0,
        "remaining_energy_sum": 0.0,
        "remaining_energy_count": 0,
    })

    for row in rows:
        server_name = (row.get("Server Name") or "").strip()
        if not server_name:
            continue

        stats = per_server[server_name]
        stats["count"] += 1

        execution = _parse_csv_number(row.get("Execution time"))
        transfer = _parse_csv_number(row.get("transfer_time"))
        queue = _parse_csv_number(row.get("Queue length"))
        remaining_energy = _parse_csv_number(row.get("Remaining_energy [J]"))
        if remaining_energy is None:
            remaining_energy = _parse_csv_number(row.get("Remaining_energy [%]"))

        if execution is not None:
            stats["execution_sum"] += execution
            stats["execution_count"] += 1
        if transfer is not None:
            stats["transfer_sum"] += transfer
            stats["transfer_count"] += 1
        if queue is not None:
            stats["queue_sum"] += queue
            stats["queue_count"] += 1
        if remaining_energy is not None:
            stats["remaining_energy_sum"] += remaining_energy
            stats["remaining_energy_count"] += 1

    if not per_server:
        return create_satellites(n=max_satellites or 3, seed=seed)

    profiles = []
    for server_name, stats in per_server.items():
        execution_mean = (
            stats["execution_sum"] / stats["execution_count"]
            if stats["execution_count"] > 0 else None
        )
        transfer_mean = (
            stats["transfer_sum"] / stats["transfer_count"]
            if stats["transfer_count"] > 0 else None
        )
        queue_mean = (
            stats["queue_sum"] / stats["queue_count"]
            if stats["queue_count"] > 0 else None
        )
        remaining_energy_mean = (
            stats["remaining_energy_sum"] / stats["remaining_energy_count"]
            if stats["remaining_energy_count"] > 0 else None
        )

        profiles.append({
            "name": server_name,
            "count": stats["count"],
            "execution_mean": execution_mean,
            "transfer_mean": transfer_mean,
            "queue_mean": queue_mean,
            "remaining_energy_mean": remaining_energy_mean,
        })

    profiles.sort(key=lambda profile: (-profile["count"], profile["name"]))
    if max_satellites is not None:
        profiles = profiles[: max(1, max_satellites)]

    def collect(key: str, fallback: float) -> Tuple[float, float]:
        values = [profile[key] for profile in profiles if profile[key] is not None]
        if not values:
            return fallback, fallback
        return min(values), max(values)

    execution_min, execution_max = collect("execution_mean", 1.0)
    transfer_min, transfer_max = collect("transfer_mean", 1.0)
    queue_min, queue_max = collect("queue_mean", 1.0)
    energy_min, energy_max = collect("remaining_energy_mean", 1.0)

    count_values = [profile["count"] for profile in profiles]
    count_min = min(count_values)
    count_max = max(count_values)

    rng = random.Random(seed)
    satellites = []
    for idx, profile in enumerate(profiles):
        sat = Satellite()
        sat.id = idx
        sat.name = profile["name"]
        sat.completedTasks = 0

        if profile["remaining_energy_mean"] is None:
            sat.remainingEnergy = _rescale(float(profile["count"]), float(count_min), float(count_max), 80.0, 120.0)
        else:
            sat.remainingEnergy = _rescale(profile["remaining_energy_mean"], energy_min, energy_max, 70.0, 130.0)

        if profile["execution_mean"] is None:
            sat.cpuCapacity = _rescale(float(profile["count"]), float(count_min), float(count_max), 22.0, 12.0)
        else:
            sat.cpuCapacity = _rescale(profile["execution_mean"], execution_min, execution_max, 26.0, 10.0)

        if profile["transfer_mean"] is None:
            sat.networkCapacity = 120.0
            sat.latency = 25.0
            sat.bandwidth = 120.0
        else:
            sat.networkCapacity = _rescale(profile["transfer_mean"], transfer_min, transfer_max, 180.0, 70.0)
            sat.latency = _rescale(profile["transfer_mean"], transfer_min, transfer_max, 8.0, 70.0)
            sat.bandwidth = _rescale(profile["transfer_mean"], transfer_min, transfer_max, 220.0, 50.0)

        if profile["queue_mean"] is None:
            sat.elevationAngle = 45.0
        else:
            sat.elevationAngle = _rescale(profile["queue_mean"], queue_min, queue_max, 65.0, 30.0)

        sat.cpuBusyUntil = 0
        sat.networkBusyUntil = 0
        sat.isAccessPoint = False
        sat.orbitalSunset = ""
        sat.energyReserved = 0.0
        sat.rejectedTasks = []
        sat.tasks = []
        sat.deadTasks = []

        # Tiny deterministic perturbation keeps satellites distinct even when
        # all source stats collapse to a single point.
        sat.cpuCapacity *= (1.0 + rng.uniform(-0.02, 0.02))
        sat.networkCapacity *= (1.0 + rng.uniform(-0.02, 0.02))
        sat.latency *= (1.0 + rng.uniform(-0.02, 0.02))
        sat.bandwidth *= (1.0 + rng.uniform(-0.02, 0.02))

        satellites.append(sat)

    for sat in satellites:
        sat.neighbors = [other.id for other in satellites if other.id != sat.id]

    return satellites


def create_satellites(
    n:                  int   = 3,
    energy_range:       tuple = (80.0, 120.0),
    load_range:         tuple = (0.10, 0.40),
    capability_range:   tuple = (12.0, 24.0),
    seed:               int   = None,
) -> list:
    rng = random.Random(seed)

    # Pick n distinct names, falling back to "SAT-<id>" if the pool runs out.
    name_pool = _SAT_NAMES[:n] if n <= len(_SAT_NAMES) else \
                _SAT_NAMES + [f"SAT-{i}" for i in range(len(_SAT_NAMES), n)]

    satellites = []
    for i in range(n):
        sat                    = Satellite()
        sat.id                 = i
        sat.name               = name_pool[i]
        sat.completedTasks     = 0
        sat.remainingEnergy    = rng.uniform(*energy_range)
        sat.cpuCapacity        = rng.uniform(*capability_range)
        sat.cpuBusyUntil       = 0
        sat.networkCapacity    = 100.0
        sat.networkBusyUntil   = 0
        sat.latency            = rng.uniform(10.0, 50.0)
        sat.bandwidth          = rng.uniform(50.0, 200.0)
        sat.elevationAngle     = 45.0
        sat.isAccessPoint      = False
        sat.orbitalSunset      = ""
        sat.energyReserved     = 0.0
        sat.rejectedTasks      = []
        sat.tasks              = []
        sat.deadTasks          = []
        satellites.append(sat)

    # Fully-connected listening dome (every satellite hears all others).
    for sat in satellites:
        sat.neighbors = [other.id for other in satellites if other.id != sat.id]

    return satellites


def create_jobs(
    n:               int   = 10,
    duration_range:  tuple = (5, 15),
    window_range:    tuple = (60, 120),
    task_size_range: tuple = (8.0, 20.0),
    priority_range:  tuple = (0.5, 2.0),
    start_minute:    float = 0.0,
    arrival_spread:  int   = 200,
    seed:            int   = None,
) -> list:
    rng = random.Random(seed)

    # Draw and sort arrival times so jobs appear in chronological order.
    arrivals = sorted(
        int(rng.uniform(start_minute, start_minute + arrival_spread))
        for _ in range(n)
    )

    jobs = []
    for i in range(n):
        duration  = rng.randint(*duration_range)
        window    = rng.randint(*window_range)
        due       = arrivals[i] + window

        prefix    = _JOB_PREFIXES[i % len(_JOB_PREFIXES)]
        name      = f"{prefix}-{i+1:02d}"

        job                   = Job()
        job.id                = i
        job.arrivalTime       = arrivals[i]
        job.type              = prefix
        job.ram               = rng.uniform(1.0, 16.0)
        job.disk              = rng.uniform(10.0, 100.0)
        job.imageSize         = rng.uniform(*task_size_range)
        job.executionTime     = duration
        job.transferTime      = rng.randint(1, 5)
        job.numberOfHops      = rng.randint(1, 3)
        job.executionServerName = ""
        jobs.append(job)

    return jobs
