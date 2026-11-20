"""Parameters of the realistic trace generator, each with its source.

One :class:`Scenario` object holds every number the simulator in
``simulator.py`` uses.  Nothing here is tuned to make the GP look good: every
default is taken from one of three sources, named in the comment next to it.

Sources
-------
``[T]``   Magliarisi's PhD thesis [magliarisi2026thesis], Chapter 6 and its
          Table 6.3 ("Main simulation parameters").  This chapter is the
          heterogeneous, energy-aware version of the simulator of
          [magliarisi2025orbit], the one that produced ``data/example.csv``.
``[C]``   ``config.json5`` in the root of this repository: the settings file of
          the simulation run that produced ``data/example.csv``.
``[S]``   The public source code of the simulator [cosmos2025], where the
          thesis gives no number (payload sizes) or where the code that
          produced the log behaves differently from the text.  The differences
          are listed in ``docs/onboarding/realistic-data.html``.

Units: seconds, megabytes (1 MB = 1024**2 bytes, as in [S]), joules, hertz,
kilometres, degrees.  The simulator's clock is in **seconds** ([T] Sec. 4.9 and
Table 6.3: simulated time 300 s).  The rest of this project calls the same
numbers "minutes"; the pipeline only uses them relative to each other, so the
label makes no difference to the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

MB = 1024 ** 2                     # bytes per megabyte [S]
SPEED_OF_LIGHT_KM_S = 299792.458   # inter-satellite latency = distance / c [S]

GENERIC = "Generic_Service"
CPU_INTENSIVE = "CPU_Intensive"
CPU_DATA = "CPU_and_Data_Intensive"
BATCH = "Batch"
REAL_TIME_TYPES = (GENERIC, CPU_INTENSIVE, CPU_DATA)

# (probability, (min MB, max MB)): one range is picked, then a size is drawn
# uniformly inside it.
PayloadMix = Tuple[Tuple[float, Tuple[float, float]], ...]


@dataclass(frozen=True)
class Scenario:
    # ── Ground user and geometry ─────────────────────────────────────────────
    observer_name: str = "Roma"                 # [C] simulation_location
    observer_lat_deg: float = 41.8967           # [C] locations.Roma
    observer_lon_deg: float = 12.4822           # [C]
    min_elevation_deg: float = 40.0             # [T] Table 6.3, [C] Phi_max: the "listening dome"
    buffer_deg: float = 15.0                    # [C] Phi_buffer: satellites at 25-40 deg relay only
    isl_range_km: float = 659.0                 # [T] Table 6.3, [C] Laser_Communication_Range
    access_points: int = 5                      # [T] Table 6.3, [C] access_point
    topology_step_s: float = 2.0                # [T] Table 6.3, [C] Interval_between_Configurations
    visibility_horizon_s: float = 900.0         # [S] sunset is searched up to 15 min ahead

    # ── Workload ─────────────────────────────────────────────────────────────
    duration_s: float = 300.0                   # [T] Table 6.3, [C] simulation_duration
    # [T] Table 6.3 lists 2, 4, 6, 8 and 10 requests per second.  10 is the
    # headline case of [T] Sec. 6.5 and the rate of data/example.csv (2960
    # rows in 300 s, about 100 per 10 s window).
    arrival_rate_per_s: float = 10.0
    type_mix: Tuple[Tuple[str, float], ...] = (  # [T] Table 6.3 (beta), [S]
        (GENERIC, 0.40), (CPU_INTENSIVE, 0.35), (CPU_DATA, 0.25))
    # Mean CPU demand in seconds, exponentially distributed.  [C] CPU_timeout
    # gives the means; [S] draws an exponential without bounds.  ([T] Table 6.3
    # writes U(0.01, 0.1) and U(0.1, 1) instead: see the guide.)
    generic_demand_mean_s: float = 0.05
    cpu_demand_mean_s: float = 0.5
    small_payload_mb: Tuple[float, float] = (0.01, 0.088)            # [S] Generic, CPU-intensive
    cpu_data_payload_mb: PayloadMix = (                               # [T] alpha weights, [S] ranges
        (0.3, (0.022, 2.2)), (0.5, (2.2, 24.2)), (0.2, (132.5, 500.0)))
    batch_payload_mb: PayloadMix = (                                  # [T] gamma weights, [S] ranges
        (0.7, (2.2, 24.2)), (0.3, (132.5, 500.0)))
    batch_per_sen_max: int = 3                  # [T] Table 6.3: U[0, 3] background requests per SEN
    deadline_factor: float = 0.1                # [T] Eq. 6.2 delta_D, [C] deadline: 10 (percent)
    batch_deadline_factor: float = 0.2          # [S] default used for batch requests

    # ── Satellite edge nodes ─────────────────────────────────────────────────
    cpu_slots: int = 5                          # [C] cpu_capacity_queue: tasks served in parallel
    net_slots: int = 5                          # [C] net_capacity_queue
    cpu_hz_range: Tuple[float, float] = (1e7, 1e10)   # [T] Table 6.3: U(10, 10000) MHz, [C] C_sen
    # CPU frequency used to CHARGE energy.  [S] draws one value for the whole
    # run and charges every node with it; each node's own frequency only
    # enters the scheduler's estimate.  None = draw it from cpu_hz_range with
    # the run's seed, as [S] does.  The generator (generate.py) normally
    # replaces None by a value fitted to the published figures.
    energy_cpu_hz: Optional[float] = None
    energy_coefficient: float = 5e-26           # [T] Table 6.3 e, [C] energy_coefficient
    tx_power_w: float = 1.0                     # [T] Table 6.3 P_net, [C] Ptrasm
    initial_energy_j: float = 100000.0          # [T] Table 6.3 {80, 100, 120} kJ, [C] initial_energy

    # ── Network ──────────────────────────────────────────────────────────────
    isl_mb_per_s: Tuple[float, float] = (2150.0, 3125.0)   # [C] available_bandwidth, per link
    nominal_isl_mb_per_s: float = 2150.0        # [T] Table 6.3 b_isl; used for deadlines and c_net
    ground_bytes_per_s: float = 35000000.0      # [C] Bandwidth_to_GU_Bps ([T]: 33.4 MB/s)
    routing_interval_s: float = 0.5             # [C] Routing_Interval: one hop per tick
    routing_ttl: int = 20                       # [S] a result is dropped after 20 hops

    # ── Scheduler of the simulated system ────────────────────────────────────
    # "OrbitAware" also requires the estimated completion to come before the
    # node leaves the dome; "DTS" does not ([T] Table 6.2, [C] request_distribution).
    policy: str = "OrbitAware"
    weight_energy: float = 0.7                  # [C] weight_energy
    weight_response: float = 0.3                # [C] weight_response

    # ── What is written ──────────────────────────────────────────────────────
    # Batch requests are background traffic, not requests to schedule
    # ([T] Sec. 6.2: the arrival rate "refers to real-time requests only").
    # They load the network stage either way; this only decides whether they
    # also get a row in the log.  [S] writes them; data/example.csv shows no
    # block of rows at t = 0 (its first world has 100 tasks), so the default
    # leaves them out.
    log_batch_rows: bool = False

    def __post_init__(self):
        if self.policy not in ("OrbitAware", "DTS"):
            raise ValueError(f"policy must be 'OrbitAware' or 'DTS', got {self.policy!r}")
        if self.duration_s <= 0 or self.arrival_rate_per_s <= 0:
            raise ValueError("duration_s and arrival_rate_per_s must be > 0")
        total = sum(p for _, p in self.type_mix)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"type_mix probabilities sum to {total}, not 1")

    @property
    def buffer_elevation_deg(self) -> float:
        """Lowest elevation at which a satellite takes part at all (25 deg)."""
        return self.min_elevation_deg - self.buffer_deg

    def with_changes(self, **changes) -> "Scenario":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    # ── Model equations ([T] Chapter 6) ──────────────────────────────────────
    def payload_bytes(self, size_mb: float) -> float:
        return size_mb * MB

    def network_demand_s(self, size_mb: float) -> float:
        """c_net = s / b_isl, [T] Eq. 6.1, at the nominal ISL rate."""
        return size_mb / self.nominal_isl_mb_per_s

    def deadline_s(self, cpu_demand_s: float, size_mb: float) -> float:
        """D = (1 + delta_D) (c + c_net), [T] Eq. 6.2 (relative to arrival)."""
        return (1.0 + self.deadline_factor) * (cpu_demand_s + self.network_demand_s(size_mb))

    def cpu_energy_j(self, cpu_demand_s: float, cpu_hz: float) -> float:
        """eps_cpu = e c C^3, [T] Eq. 6.3 (dynamic voltage scaling)."""
        return self.energy_coefficient * cpu_demand_s * cpu_hz ** 3

    def net_energy_j(self, transmit_s: float) -> float:
        """eps_net = P_net c_net, [T] Eq. 6.3."""
        return self.tx_power_w * transmit_s
