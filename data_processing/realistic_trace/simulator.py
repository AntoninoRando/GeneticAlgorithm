"""Discrete-event simulation of satellite edge computing over real orbits.

This is an independent re-implementation of the model behind the simulator of
Magliarisi et al. ([magliarisi2026thesis] Chapter 6, the energy-aware extension
of [magliarisi2025orbit]), written from the thesis, from ``config.json5`` and
from the observable behaviour of the public code [cosmos2025].  It exists so
that a task log with the structure of ``data/example.csv`` can be produced
without access to the original file.  No code is copied from [cosmos2025].

A *discrete-event simulation* keeps a list of future events sorted by time
(here a heap) and jumps from one event to the next; nothing happens in
between.  The events are: a request arrives, a request reaches the node that
will run it, a CPU or network stage finishes, and a routing tick.

The model in one page
---------------------
**Geometry** (``Topology``).  Every ``topology_step_s`` (2 s) the satellites
with elevation above 25 degrees over the ground user form the network.  Those
above 40 degrees are in the *listening dome* and can execute requests; the
others (the *buffer zone*) only relay results.  Two satellites are neighbours
when they are less than 659 km apart.  The *access points* (APs) are the 5
dome satellites nearest to the user among those still approaching it.  A dome
satellite's *sunset* is the time left before its elevation drops below 40
degrees.

**Workload.**  Requests arrive as a Poisson process of rate lambda.  Each has
a type (generic, CPU-intensive, CPU-and-data-intensive), a CPU demand ``c`` in
seconds, a payload ``s`` in MB, the network demand ``c_net = s / b_isl`` and
the deadline ``D = (1 + delta)(c + c_net)`` ([T] Eqs. 6.1-6.2).

**Scheduling** (what the simulated system does with a request):

1. The request enters at the AP with the highest *score*
   ``w_e B / B_max - w_R W / D``, where ``B`` is the node's battery, ``B_max``
   the largest battery among the APs and ``W`` the node's estimated waiting
   time.  The score is minus infinity when the node cannot meet the deadline
   or lacks the energy.
2. The AP considers its neighbours that are in the dome, best score first,
   and keeps those whose *estimated completion time* is below the deadline
   and, for the OrbitAware policy, below the node's sunset.  The first one
   gets the request; if none is left the request is rejected.
3. The node checks energy and deadline again when the request arrives, then
   runs it: a CPU stage (5 requests in parallel, first come first served),
   then either a network stage (data-intensive requests) or the downlink
   delay ``s / b_ground``.
4. Energy ([T] Eq. 6.3): ``eps_cpu = e c C^3`` for the CPU stage and
   ``eps_net = P c_net`` for the network stage.  Batteries never recharge.
5. The result travels back to an AP by greedy geographic routing, one hop per
   ``routing_interval_s``: to a neighbouring AP if there is one, otherwise to
   the unvisited neighbour nearest to the user.

Waiting-time estimate of a stage with ``m`` slots: with ``r_min`` the shortest
remaining time among the running requests when all slots are busy (0
otherwise) and ``A`` the total remaining work (running plus queued)::

    W = r_min + max(0, A - m r_min) / m

Where this code follows the public simulator rather than the thesis text, the
comment says so with ``[S]``; the list is in
``docs/onboarding/realistic-data.html``.

Determinism: the result is a pure function of the ephemeris, the scenario and
the seed.  Three independent random streams are derived from the seed
(workload, per-satellite CPU frequency, link bandwidth), so changing one part
of the model never shifts the random numbers of another.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .orbits import Ephemeris
from .scenario import (BATCH, CPU_DATA, CPU_INTENSIVE, GENERIC, SPEED_OF_LIGHT_KM_S,
                       Scenario)

# Column order of the simulator's result file [S].  data_factory.py reads
# "Task Type", "Execution time", "transfer_time", "Image_Size_MB",
# "Num Hops Routing" (else "Num Hops"), "Queue length", "Server Name",
# "Remaining_energy [J]" and the arrival column; the rest is kept so that the
# file is a drop-in replacement for data/example.csv.
COLUMNS = [
    "Task ID", "Task Type", "Status", "Arrival Time (System)",
    "Arrival Time (Queue)", "Start Time", "End Time", "Execution time",
    "Service Time", "Time in system", "Time in queue", "Server Name", "Num Hops",
    "Num Hops Routing", "Queue length", "transfer_time", "Image_Size_MB", "Exec_after_set",
    "Energy_CPU [J]", "Energy_NET [J]", "Energy_TOTAL [J]", "Remaining_energy [J]",
    "Remaining_energy [%]", "Rejection Reason",
    "Routing Init Time", "Routing End Time", "Routing Duration",
]
NA = "N/A"

# Rejection reasons, spelled as in the simulator's log [S].
R_NO_SERVER = "No suitable server found after deadline/sunset filters"
R_SUNSET = "Invalid orbitalSunset"
R_DEADLINE = "Deadline Exceeded"
R_ENERGY_CPU = "Insufficient Energy for CPU"
R_ENERGY_CPU_NET = "Insufficient Energy for CPU+NET"
R_ENERGY_NET = "Insufficient Energy for NET"
R_ENERGY_ROUTING = "Routing Energy Exhausted"
R_IN_QUEUE = "In Queue at End"

_ARRIVE, _ENQUEUE, _CPU_END, _DOWNLINK_END, _NET_END, _ROUTE = range(6)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: who is visible, who is a neighbour, who is an access point
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Snapshot:
    """The network at one topology update."""

    index: int
    time: float
    active: List[int]                       # satellites above 25 deg, by index
    elevation: np.ndarray                   # degrees, all satellites
    slant_km: np.ndarray                    # distance to the ground user, all satellites
    neighbours: Dict[int, List[int]]        # active satellite -> active satellites < 659 km
    distance_km: Dict[Tuple[int, int], float]
    sunset_s: Dict[int, int]                # dome satellite -> whole seconds of visibility left
    access_points: List[int]


class Topology:
    """Topology snapshots derived from an :class:`Ephemeris`."""

    def __init__(self, ephemeris: Ephemeris, scenario: Scenario, seed: int):
        self.eph = ephemeris
        self.sc = scenario
        self.seed = seed
        self.n = len(ephemeris.names)
        self._elev = ephemeris.elevation_series()                      # (n_sat, n_steps)
        self._slant = np.linalg.norm(ephemeris.pos - ephemeris.observer.ecef_km(), axis=-1)
        # For every grid step, the first later step at which the satellite is
        # below the dome (its loss of signal).  Scanned from the end backwards.
        steps = self._elev.shape[1]
        below = self._elev < scenario.min_elevation_deg
        index = np.where(below, np.arange(steps), steps)
        self._los_step = np.minimum.accumulate(index[:, ::-1], axis=1)[:, ::-1]
        self._cache: Dict[int, Snapshot] = {}
        self._bandwidth: Dict[Tuple[int, int, int], float] = {}

    def _grid(self, t: float) -> int:
        return min(int(round(t / self.eph.step_s)), self._elev.shape[1] - 1)

    def at(self, t: float) -> Snapshot:
        """The snapshot in force at second ``t`` (updated every topology step)."""
        return self.snapshot(int(t // self.sc.topology_step_s))

    def snapshot(self, k: int) -> Snapshot:
        snap = self._cache.get(k)
        if snap is not None:
            return snap
        sc = self.sc
        t = k * sc.topology_step_s
        g = self._grid(t)
        elev, slant = self._elev[:, g], self._slant[:, g]
        pos = self.eph.pos[:, g]

        active = np.flatnonzero(elev > sc.buffer_elevation_deg)
        neighbours: Dict[int, List[int]] = {int(i): [] for i in active}
        distance: Dict[Tuple[int, int], float] = {}
        if active.size > 1:
            d = np.linalg.norm(pos[active, None, :] - pos[None, active, :], axis=-1)
            close = (d < sc.isl_range_km) & ~np.eye(active.size, dtype=bool)   # [S] strict "<"
            for a, b in zip(*np.nonzero(close)):
                i, j = int(active[a]), int(active[b])
                neighbours[i].append(j)
                distance[(i, j)] = float(d[a, b])

        dome = [int(i) for i in active if elev[i] >= sc.min_elevation_deg]
        horizon = (self._elev.shape[1] - 1) * self.eph.step_s
        sunset = {}
        for i in dome:
            los = self._los_step[i, g] * self.eph.step_s
            sunset[i] = int(min(los, horizon) - t)          # whole seconds [S]

        # AP-Optimal ([T] Eq. 5.2): the nearest dome satellites among those
        # approaching the user (range decreasing).  [S]: with fewer dome
        # satellites than requested APs, use half of them.
        g_next = min(g + 1, self._slant.shape[1] - 1)
        approaching = [i for i in dome if self._slant[i, g_next] < slant[i]]
        wanted = sc.access_points if len(dome) >= sc.access_points else len(dome) // 2
        aps = sorted(approaching, key=lambda i: (slant[i], i))[:wanted]

        snap = Snapshot(index=k, time=t, active=[int(i) for i in active], elevation=elev,
                        slant_km=slant, neighbours=neighbours, distance_km=distance,
                        sunset_s=sunset, access_points=aps)
        self._cache[k] = snap
        return snap

    def link_mb_per_s(self, k: int, i: int, j: int) -> float:
        """Bandwidth of link i-j during snapshot k: U(min, max), redrawn at
        every topology update [S], the same in both directions."""
        key = (k, min(i, j), max(i, j))
        value = self._bandwidth.get(key)
        if value is None:
            rng = random.Random(f"{self.seed}:link:{key[0]}:{self.eph.names[key[1]]}:"
                                f"{self.eph.names[key[2]]}")
            value = rng.uniform(*self.sc.isl_mb_per_s)
            self._bandwidth[key] = value
        return value


# ─────────────────────────────────────────────────────────────────────────────
# Nodes and requests
# ─────────────────────────────────────────────────────────────────────────────
class _Stage:
    """A service stage with ``capacity`` parallel slots and a FIFO queue."""

    __slots__ = ("capacity", "running", "queue")

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.running: Dict[int, float] = {}        # task id -> time its service ends
        self.queue: deque = deque()                # (task, demand in seconds, grant callback)

    def occupancy(self) -> int:
        return len(self.running) + len(self.queue)

    def wait_estimate(self, now: float) -> float:
        """Estimated wait of a request arriving now (see the module docstring)."""
        remaining = [max(0.0, end - now) for end in self.running.values()]
        r_min = min(remaining) if len(remaining) >= self.capacity else 0.0
        work = sum(remaining) + sum(entry[1] for entry in self.queue)
        return r_min + max(0.0, work - self.capacity * r_min) / self.capacity


@dataclass
class _Task:
    id: int
    type: str
    arrival: float
    cpu_demand: float
    size_mb: float
    deadline: float                   # relative to arrival
    net_demand: float                 # c_net at the nominal ISL rate
    server: int = -1                  # executing node (satellite index)
    hops: int = 0
    transfer: float = 0.0
    queue_arrival: float = 0.0
    cpu_wait: float = 0.0
    net_wait: float = 0.0
    cpu_done: float = 0.0
    eps_cpu: float = 0.0
    eps_net: float = 0.0
    reserved: float = 0.0
    qlen_cpu: Optional[int] = None
    qlen_net: Optional[int] = None
    # result routing
    location: int = -1
    routing_hops: int = 0
    routing_end: Optional[float] = None
    ttl: int = 0
    visited: set = field(default_factory=set)


class _Server:
    __slots__ = ("index", "name", "energy", "reserved", "cpu_hz", "cpu", "net", "outbox")

    def __init__(self, index: int, name: str, scenario: Scenario, cpu_hz: float):
        self.index = index
        self.name = name
        self.energy = scenario.initial_energy_j
        self.reserved = 0.0
        self.cpu_hz = cpu_hz
        self.cpu = _Stage(scenario.cpu_slots)
        self.net = _Stage(scenario.net_slots)
        self.outbox: List[_Task] = []        # results waiting to be routed


@dataclass
class SimulationResult:
    rows: List[dict]                 # one per logged request, keys = COLUMNS
    visibility: List[dict]           # per topology update and satellite
    stats: dict


# ─────────────────────────────────────────────────────────────────────────────
# The simulation
# ─────────────────────────────────────────────────────────────────────────────
class Simulation:
    def __init__(self, ephemeris: Ephemeris, scenario: Scenario, seed: int = 0):
        if ephemeris.duration_s < scenario.duration_s:
            raise ValueError("the ephemeris is shorter than the simulated time")
        self.sc = scenario
        self.seed = seed
        self.eph = ephemeris
        self.topology = Topology(ephemeris, scenario, seed)
        self.rng = random.Random(f"{seed}:workload")
        if scenario.energy_cpu_hz is not None:
            self.energy_cpu_hz = float(scenario.energy_cpu_hz)
        else:                                    # one draw for the whole run [S]
            self.energy_cpu_hz = random.Random(f"{seed}:energy-cpu").uniform(*scenario.cpu_hz_range)
        self.servers: Dict[int, _Server] = {}
        self._events: list = []
        self._seq = 0
        self.now = 0.0
        self._completed: List[Tuple[_Task, dict]] = []
        self._rejected: List[Tuple[_Task, int, str]] = []
        self._tasks: Dict[int, _Task] = {}
        self._generated = 0
        self._dropped = 0
        self._batch_rows: List[Tuple[_Task, dict]] = []
        self._next_batch_id = 1_000_001

    # ── small helpers ────────────────────────────────────────────────────────
    def _push(self, time: float, kind: int, payload) -> None:
        self._seq += 1
        heapq.heappush(self._events, (time, self._seq, kind, payload))

    def _server(self, index: int) -> _Server:
        """Node state, created the first time the satellite is used.  A node's
        own CPU frequency depends only on the seed and the satellite's name."""
        server = self.servers.get(index)
        if server is None:
            name = self.eph.names[index]
            cpu_hz = random.Random(f"{self.seed}:cpu:{name}").uniform(*self.sc.cpu_hz_range)
            server = _Server(index, name, self.sc, cpu_hz)
            self.servers[index] = server
        return server

    def _draw_from_mix(self, mix) -> float:
        r, acc = self.rng.random(), 0.0
        for probability, bounds in mix:
            acc += probability
            if r < acc:
                return self.rng.uniform(*bounds)
        return self.rng.uniform(*mix[-1][1])

    def _new_request(self, task_id: int, arrival: float) -> _Task:
        sc = self.sc
        r, acc, kind = self.rng.random(), 0.0, sc.type_mix[-1][0]
        for name, probability in sc.type_mix:
            acc += probability
            if r < acc:
                kind = name
                break
        if kind == CPU_DATA:
            size = self._draw_from_mix(sc.cpu_data_payload_mb)
        else:
            size = self.rng.uniform(*sc.small_payload_mb)
        mean = sc.generic_demand_mean_s if kind == GENERIC else sc.cpu_demand_mean_s
        demand = self.rng.expovariate(1.0 / mean)
        return _Task(id=task_id, type=kind, arrival=arrival, cpu_demand=demand, size_mb=size,
                     deadline=sc.deadline_s(demand, size), net_demand=sc.network_demand_s(size))

    def _score(self, server: _Server, task: _Task, b_max: float) -> float:
        """Selection score of a node for a request (step 1 of the docstring).
        The energy estimate uses the node's own CPU frequency [S]."""
        sc = self.sc
        wc = server.cpu.wait_estimate(self.now)
        wn = server.net.wait_estimate(self.now)
        need = sc.cpu_energy_j(task.cpu_demand, server.cpu_hz) + sc.net_energy_j(task.net_demand)
        if wc + task.cpu_demand + wn + task.net_demand > task.deadline or need > server.energy:
            return -math.inf
        battery = server.energy / b_max if b_max > 0 else 0.0
        if task.type == CPU_DATA:
            wait = wc + wn
        elif task.type == BATCH:
            wait = wn
        else:
            wait = wc
        return sc.weight_energy * battery - sc.weight_response * wait / (task.deadline or 1.0)

    def _reject(self, task: _Task, server: _Server, reason: str) -> None:
        self._rejected.append((task, server.index, reason))

    # ── stage bookkeeping ────────────────────────────────────────────────────
    def _request(self, stage: _Stage, task: _Task, demand: float, grant) -> None:
        if len(stage.running) < stage.capacity and not stage.queue:
            stage.running[task.id] = self.now          # placeholder until grant sets the end
            grant(task)
        else:
            stage.queue.append((task, demand, grant))

    def _release(self, stage: _Stage, task: _Task) -> None:
        stage.running.pop(task.id, None)
        while stage.queue and len(stage.running) < stage.capacity:
            nxt, _, grant = stage.queue.popleft()
            stage.running[nxt.id] = self.now
            grant(nxt)

    # ── event handlers ───────────────────────────────────────────────────────
    def _on_arrive(self, task: _Task) -> None:
        sc, snap = self.sc, self.topology.at(self.now)
        aps = [self._server(i) for i in snap.access_points]
        if not aps:
            self._dropped += 1
            return
        b_max = max(ap.energy for ap in aps)
        entry, best = None, -math.inf
        for ap in aps:
            score = self._score(ap, task, b_max)
            if score > best:
                entry, best = ap, score
        if entry is None:
            # Every AP scores minus infinity: the simulator drops the request
            # without writing it to the log [S].  Counted in the statistics.
            self._dropped += 1
            return

        # Candidates: the AP's neighbours.  The AP itself is in the list of
        # [S] too, but its own link has no bandwidth entry there, its
        # completion estimate is infinite and it is never chosen; it is left
        # out here with the same effect.
        candidates = []
        for j in snap.neighbours.get(entry.index, []):
            sunset = snap.sunset_s.get(j)
            if not sunset:                      # buffer zone, or leaving within a second
                continue
            node = self._server(j)
            on_link = task.size_mb / self.topology.link_mb_per_s(snap.index, entry.index, j)
            transfer = on_link + snap.distance_km[(entry.index, j)] / SPEED_OF_LIGHT_KM_S
            # [S] adds the payload's time on the link twice: once as transfer
            # and once as network demand on that link.
            completion = (node.cpu.wait_estimate(self.now) + node.net.wait_estimate(self.now)
                          + task.cpu_demand + on_link + transfer)
            candidates.append((self._score(node, task, b_max), node, transfer, completion, sunset))
        if not candidates:
            self._reject(task, entry, R_SUNSET)
            return
        candidates.sort(key=lambda c: -c[0])          # stable: ties keep neighbour order
        chosen = None
        for _, node, transfer, completion, sunset in candidates:
            if completion < task.deadline and (sc.policy != "OrbitAware" or completion < sunset):
                chosen = (node, transfer)
                break
        if chosen is None:
            self._reject(task, entry, R_NO_SERVER)
            return

        node, transfer = chosen
        # The AP pays for sending the payload over the link.
        link = self.topology.link_mb_per_s(snap.index, entry.index, node.index)
        cost = sc.net_energy_j(task.size_mb / link)
        if entry.energy < cost:
            self._reject(task, entry, R_ENERGY_ROUTING)
            return
        entry.energy -= cost
        task.server, task.hops, task.transfer = node.index, 1, transfer
        self._push(self.now + transfer, _ENQUEUE, task)

    def _on_enqueue(self, task: _Task) -> None:
        """The request reaches its node: second admission check, then CPU."""
        sc, node = self.sc, self._server(task.server)
        task.queue_arrival = self.now
        task.eps_net = sc.net_energy_j(task.net_demand)
        wc, wn = node.cpu.wait_estimate(self.now), node.net.wait_estimate(self.now)

        if task.type == BATCH:
            if node.energy - node.reserved < task.eps_net:
                return self._reject(task, node, R_ENERGY_NET)
            if wn + task.net_demand > task.deadline:
                return self._reject(task, node, R_DEADLINE)
            task.reserved = task.eps_net
            node.reserved += task.reserved
            self._request(node.net, task, task.net_demand, self._grant_net)
            task.qlen_net = node.net.occupancy()
            return

        # Energy is charged with the run-wide CPU frequency [S].
        task.eps_cpu = sc.cpu_energy_j(task.cpu_demand, self.energy_cpu_hz)
        if task.type == CPU_DATA:
            need, response = task.eps_cpu + task.eps_net, wc + task.cpu_demand + wn + task.net_demand
            reason = R_ENERGY_CPU_NET
        else:
            task.eps_net = 0.0                 # downlink only: no network stage
            need, response = task.eps_cpu, wc + task.cpu_demand + task.net_demand
            reason = R_ENERGY_CPU
        if node.energy - node.reserved < need:
            return self._reject(task, node, reason)
        if response > task.deadline:
            return self._reject(task, node, R_DEADLINE)
        task.reserved = need
        node.reserved += need
        self._request(node.cpu, task, task.cpu_demand, self._grant_cpu)
        task.qlen_cpu = node.cpu.occupancy()

    def _grant_cpu(self, task: _Task) -> None:
        node = self._server(task.server)
        if node.energy < task.eps_cpu:          # drained while the request waited
            node.reserved -= task.reserved
            self._reject(task, node, R_ENERGY_CPU)
            self._release(node.cpu, task)
            return
        task.cpu_wait = self.now - task.queue_arrival
        node.cpu.running[task.id] = self.now + task.cpu_demand
        self._push(self.now + task.cpu_demand, _CPU_END, task)

    def _on_cpu_end(self, task: _Task) -> None:
        node = self._server(task.server)
        node.energy -= task.eps_cpu
        task.cpu_done = self.now
        self._release(node.cpu, task)
        if task.type == CPU_DATA:
            self._request(node.net, task, task.net_demand, self._grant_net)
            task.qlen_net = node.net.occupancy()
        else:
            downlink = self.sc.payload_bytes(task.size_mb) / self.sc.ground_bytes_per_s
            self._push(self.now + downlink, _DOWNLINK_END, task)

    def _grant_net(self, task: _Task) -> None:
        node = self._server(task.server)
        if node.energy < task.eps_net:
            node.reserved -= task.reserved
            self._reject(task, node, R_ENERGY_NET)
            self._release(node.net, task)
            return
        task.net_wait = self.now - (task.cpu_done if task.type == CPU_DATA else task.queue_arrival)
        node.net.running[task.id] = self.now + task.net_demand
        self._push(self.now + task.net_demand, _NET_END, task)

    def _on_net_end(self, task: _Task) -> None:
        node = self._server(task.server)
        node.energy -= task.eps_net
        self._release(node.net, task)
        self._finish(task)

    def _finish(self, task: _Task) -> None:
        sc, node = self.sc, self._server(task.server)
        node.reserved -= task.reserved
        snap = self.topology.at(self.now)
        cpu_q = task.qlen_cpu if task.qlen_cpu is not None else node.cpu.occupancy()
        net_q = task.qlen_net if task.qlen_net is not None else node.net.occupancy()
        waited = task.cpu_wait + task.net_wait
        execution = self.now - task.queue_arrival          # queue wait + CPU + network [S]
        record = {
            "end": self.now, "execution": execution, "waited": waited,
            "service": execution + task.transfer + waited,
            "queue": cpu_q + net_q,
            "after_set": bool(snap.elevation[node.index] < sc.min_elevation_deg),
            "remaining": node.energy,
        }
        if task.type == BATCH:
            self._batch_rows.append((task, record))
            return
        self._completed.append((task, record))
        # The result starts its way back to the ground user.
        task.location, task.ttl, task.visited = node.index, sc.routing_ttl, {node.index}
        node.outbox.append(task)

    def _on_route(self) -> None:
        """One routing tick: every waiting result moves at most one hop."""
        sc, snap = self.sc, self.topology.at(self.now)
        aps = set(snap.access_points)
        moves = []
        for index in sorted(self.servers):
            node = self.servers[index]
            if not node.outbox:
                continue
            keep = []
            for task in node.outbox:
                if task.ttl <= 0:
                    continue                                   # dropped: TTL expired
                if index in aps:                               # an AP hands it to the user
                    task.routing_hops += 1
                    task.routing_end = self.now
                    continue
                options = [j for j in snap.neighbours.get(index, []) if j not in task.visited]
                if not options:
                    keep.append(task)                          # stuck: try again next tick
                    continue
                target = min(options, key=lambda j: (j not in aps, snap.slant_km[j], j))
                link = self.topology.link_mb_per_s(snap.index, index, target)
                node.energy = max(0.0, node.energy - sc.net_energy_j(task.size_mb / link))
                task.routing_hops += 1
                task.ttl -= 1
                task.visited.add(target)
                moves.append((task, target))
            node.outbox = keep
        for task, target in moves:
            task.location = target
            self._server(target).outbox.append(task)
        if self.now + sc.routing_interval_s <= sc.duration_s:
            self._push(self.now + sc.routing_interval_s, _ROUTE, None)

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self) -> SimulationResult:
        sc = self.sc
        # Background batch requests: U[0, 3] per node of the initial topology,
        # all injected at t = 0 ([T] Table 6.3, [S]).
        first = self.topology.snapshot(0)
        for index in first.active:
            for _ in range(self.rng.randint(0, sc.batch_per_sen_max)):
                size = self._draw_from_mix(sc.batch_payload_mb)
                net = sc.network_demand_s(size)
                task = _Task(id=self._next_batch_id, type=BATCH, arrival=0.0, cpu_demand=0.0,
                             size_mb=size, net_demand=net,
                             deadline=(1.0 + sc.batch_deadline_factor) * net, server=index)
                self._next_batch_id += 1
                self._push(0.0, _ENQUEUE, task)

        # Real-time requests: a Poisson process (exponential gaps).
        t, task_id = 0.0, 1
        while True:
            t += self.rng.expovariate(sc.arrival_rate_per_s)
            if t >= sc.duration_s:
                break
            task = self._new_request(task_id, t)
            self._tasks[task_id] = task
            self._push(t, _ARRIVE, task)
            task_id += 1
        self._generated = task_id - 1
        self._push(sc.routing_interval_s, _ROUTE, None)

        handlers = {_ARRIVE: self._on_arrive, _ENQUEUE: self._on_enqueue,
                    _CPU_END: self._on_cpu_end, _DOWNLINK_END: self._finish,
                    _NET_END: self._on_net_end}
        while self._events:
            time, _, kind, payload = heapq.heappop(self._events)
            if time > sc.duration_s:
                break
            self.now = time
            if kind == _ROUTE:
                self._on_route()
            else:
                handlers[kind](payload)
        self.now = sc.duration_s
        return self._collect()

    # ── output ───────────────────────────────────────────────────────────────
    def _collect(self) -> SimulationResult:
        sc = self.sc
        budget = sc.initial_energy_j
        rows: List[dict] = []

        def base(task: _Task, status: str, server: _Server) -> dict:
            row = dict.fromkeys(COLUMNS, NA)
            row.update({"Task ID": task.id, "Task Type": task.type, "Status": status,
                        "Arrival Time (System)": task.arrival, "Server Name": server.name,
                        "Image_Size_MB": task.size_mb})
            return row

        for task, rec in self._completed + (self._batch_rows if sc.log_batch_rows else []):
            server = self.servers[task.server]
            row = base(task, "Completed", server)
            row.update({
                "Arrival Time (Queue)": task.queue_arrival, "Start Time": task.queue_arrival,
                "End Time": rec["end"], "Execution time": rec["execution"],
                "Service Time": rec["service"], "Time in system": rec["end"] - task.arrival,
                "Time in queue": rec["waited"], "Num Hops": task.hops,
                "Queue length": rec["queue"], "transfer_time": task.transfer,
                "Exec_after_set": rec["after_set"],
                "Energy_CPU [J]": task.eps_cpu, "Energy_NET [J]": task.eps_net,
                "Energy_TOTAL [J]": task.eps_cpu + task.eps_net,
                "Remaining_energy [J]": rec["remaining"],
                "Remaining_energy [%]": 100.0 * rec["remaining"] / budget,
            })
            if task.type != BATCH:
                # [S]: the routing clock starts when the request is assigned.
                row["Num Hops Routing"] = task.routing_hops
                row["Routing Init Time"] = task.arrival
                if task.routing_end is not None:
                    row["Routing End Time"] = task.routing_end
                    row["Routing Duration"] = task.routing_end - task.arrival
            rows.append(row)

        for task, index, reason in self._rejected:
            if task.type == BATCH and not sc.log_batch_rows:
                continue
            server = self.servers[index]
            row = base(task, "Rejected", server)
            row.update({
                "Execution time": task.cpu_demand, "Service Time": 0.0, "Time in system": 0.0,
                "Time in queue": 0.0, "Num Hops": 0, "transfer_time": 0.0,
                "Exec_after_set": False, "Energy_CPU [J]": 0.0, "Energy_NET [J]": 0.0,
                "Energy_TOTAL [J]": 0.0, "Remaining_energy [J]": server.energy,
                "Remaining_energy [%]": 100.0 * server.energy / budget,
                "Rejection Reason": reason,
            })
            rows.append(row)

        # Requests still waiting in a queue when the run ends [S].  Requests
        # in service at that moment are not logged at all [S].
        waiting = in_service = 0
        for server in self.servers.values():
            in_service += len(server.cpu.running) + len(server.net.running)
            for stage, label in ((server.cpu, "CPU_Waiting"), (server.net, "NET_Waiting")):
                for task, demand, _ in stage.queue:
                    waiting += 1
                    row = dict.fromkeys(COLUMNS, NA)
                    row.update({
                        "Task ID": task.id, "Task Type": f"Residual ({label})",
                        "Status": "In Queue", "Server Name": server.name,
                        "Execution time": demand if label == "CPU_Waiting" else 0.0,
                        "Service Time": demand if label == "NET_Waiting" else 0.0,
                        "Queue length": len(server.cpu.queue) + len(server.net.queue),
                        "Exec_after_set": False, "Energy_CPU [J]": 0.0, "Energy_NET [J]": 0.0,
                        "Energy_TOTAL [J]": 0.0, "Remaining_energy [J]": server.energy,
                        "Remaining_energy [%]": 100.0 * server.energy / budget,
                        "Rejection Reason": R_IN_QUEUE,
                    })
                    rows.append(row)

        rows.sort(key=lambda r: (r["Arrival Time (System)"] == NA,
                                 r["Arrival Time (System)"] if r["Arrival Time (System)"] != NA else 0.0,
                                 r["Task ID"]))

        visibility = []
        steps = int(sc.duration_s // sc.topology_step_s) + 1
        dome_sizes, ap_counts, active_sizes = [], [], []
        for k in range(steps):
            snap = self.topology.snapshot(k)
            aps = set(snap.access_points)
            dome_sizes.append(len(snap.sunset_s))
            ap_counts.append(len(aps))
            active_sizes.append(len(snap.active))
            for i in snap.active:
                visibility.append({
                    "Time": snap.time, "Server Name": self.eph.names[i],
                    "Elevation [deg]": round(float(snap.elevation[i]), 3),
                    "In dome": i in snap.sunset_s,
                    "Sunset [s]": snap.sunset_s.get(i, 0),
                    "Access point": i in aps,
                    "Slant range [km]": round(float(snap.slant_km[i]), 2),
                })

        real_time_rejected = [(t, i, r) for t, i, r in self._rejected if t.type != BATCH]
        completed = [t for t, _ in self._completed]
        reasons = Counter(r for _, _, r in real_time_rejected)
        by_type = Counter(t.type for t in self._tasks.values())
        executing = Counter(self.eph.names[t.server] for t in completed)
        top = sum(n for _, n in executing.most_common(4))
        times = [rec["end"] - t.arrival for t, rec in self._completed]
        delivered = [t for t in completed if t.routing_end is not None]
        energy_reasons = sum(n for r, n in reasons.items() if "Energy" in r)
        used = [budget - s.energy for s in self.servers.values()]
        stats = {
            "seed": self.seed,
            "generated": self._generated,
            "completed": len(completed),
            "rejected": len(real_time_rejected),
            "dropped_unlogged": self._dropped,
            "in_service_at_end_unlogged": in_service,
            "waiting_at_end": waiting,
            "success_rate": len(completed) / self._generated if self._generated else 0.0,
            "rejection_reasons": dict(reasons.most_common()),
            "energy_share_of_rejections": (energy_reasons / len(real_time_rejected)
                                           if real_time_rejected else 0.0),
            "requests_by_type": dict(by_type),
            "executing_servers": len(executing),
            "servers_in_log": len({r["Server Name"] for r in rows}),
            "share_of_top4_servers": top / len(completed) if completed else 0.0,
            "mean_time_in_system_s": float(np.mean(times)) if times else 0.0,
            "exec_after_set": sum(1 for _, rec in self._completed if rec["after_set"]),
            "results_delivered": len(delivered),
            "mean_routing_hops": (float(np.mean([t.routing_hops for t in delivered]))
                                  if delivered else 0.0),
            "energy_cpu_hz": self.energy_cpu_hz,
            "drained_servers": sum(1 for s in self.servers.values() if s.energy < 0.01 * budget),
            "mean_energy_used_j": float(np.mean(used)) if used else 0.0,
            "dome_size_mean": float(np.mean(dome_sizes)),
            "dome_size_min": int(min(dome_sizes)),
            "dome_size_max": int(max(dome_sizes)),
            "network_size_mean": float(np.mean(active_sizes)),
            "access_points_mean": float(np.mean(ap_counts)),
            "batch_requests": self._next_batch_id - 1_000_001,
        }
        return SimulationResult(rows=rows, visibility=visibility, stats=stats)


def simulate(ephemeris: Ephemeris, scenario: Optional[Scenario] = None,
             seed: int = 0) -> SimulationResult:
    """Run one simulation and return its log rows, visibility table and statistics."""
    return Simulation(ephemeris, scenario or Scenario(), seed).run()
