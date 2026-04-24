import random
from scheduler import Satellite, Job


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


def create_satellites(
    n:                  int   = 3,
    energy_range:       tuple = (80.0, 120.0),
    load_range:         tuple = (0.10, 0.40),
    capability_range:   tuple = (12.0, 24.0),
    seed:               int   = None,
) -> list:
    """
    Generate a list of randomised satellites.

    Every satellite is connected to all others via its ``listeningDome``
    (fully-connected topology).  All numeric fields are drawn uniformly
    from the supplied ranges.

    Parameters
    ----------
    n : int
        Number of satellites to generate.  Must be ≥ 1.
    energy_range : tuple[float, float]
        ``(min, max)`` remaining energy (same unit as your energy model).
    load_range : tuple[float, float]
        ``(min, max)`` current computing load as a fraction of capability.
    capability_range : tuple[float, float]
        ``(min, max)`` maximum computing capability.
    seed : int, optional
        Random seed for reproducibility.  Pass ``None`` for a different
        result every call.

    Returns
    -------
    list[Satellite]
        Fully initialised satellite objects ready to be passed to
        :py:class:`SimulationSnapshot`.

    Example
    -------
    .. code-block:: python

        from data_factory import create_satellites
        sats = create_satellites(n=4, seed=42)
    """
    rng = random.Random(seed)

    # Pick n distinct names, falling back to "SAT-<id>" if the pool runs out.
    name_pool = _SAT_NAMES[:n] if n <= len(_SAT_NAMES) else \
                _SAT_NAMES + [f"SAT-{i}" for i in range(len(_SAT_NAMES), n)]

    satellites = []
    for i in range(n):
        sat                    = Satellite(i, name_pool[i])
        sat.activeTasks        = 0
        sat.RemainingEnergy    = rng.uniform(*energy_range)
        sat.ComputingCapability= rng.uniform(*capability_range)
        # Load is expressed as an absolute value ≤ capability.
        sat.ComputingLoad      = rng.uniform(*load_range) * sat.ComputingCapability
        satellites.append(sat)

    # Fully-connected listening dome (every satellite hears all others).
    for sat in satellites:
        sat.listeningDome = [other for other in satellites if other.id != sat.id]

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
    """
    Generate a list of randomised jobs.

    Arrival times are drawn uniformly from
    ``[start_minute, start_minute + arrival_spread]`` and sorted so that
    jobs arrive in chronological order.  Each job's ``dueMinute`` is set to
    ``arrivalTime + window``, where ``window`` is drawn from ``window_range``,
    guaranteeing that the deadline is always reachable from the moment of
    arrival.

    Parameters
    ----------
    n : int
        Number of jobs to generate.  Must be ≥ 1.
    duration_range : tuple[int, int]
        ``(min, max)`` execution duration in minutes.
    window_range : tuple[int, int]
        ``(min, max)`` scheduling window in minutes, measured from arrival.
        ``dueMinute = arrivalTime + window``.  Must be > ``duration_range[1]``
        to guarantee at least one feasible assignment exists.
    task_size_range : tuple[float, float]
        ``(min, max)`` abstract task-size / workload units.
    priority_range : tuple[float, float]
        ``(min, max)`` job priority weight.
    start_minute : float
        Earliest possible arrival time (minutes from simulation epoch).
    arrival_spread : int
        Range of minutes over which arrivals are spread.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[Job]
        Fully initialised job objects ready to be passed to
        :py:class:`SimulationSnapshot`.

    Example
    -------
    .. code-block:: python

        from data_factory import create_jobs
        jobs = create_jobs(n=15, seed=42)
    """
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

        job                   = Job(i, name, duration, due)
        job.arrivalTime       = arrivals[i]
        job.taskSize          = rng.uniform(*task_size_range)
        job.priority          = rng.uniform(*priority_range)
        job.initialDeadline   = float(due)
        job.remainingDeadline = float(due)
        jobs.append(job)

    return jobs