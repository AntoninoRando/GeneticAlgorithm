"""Tests for the literature-based options of the GP scheduler.

Run from the repository root, after building the `scheduler` extension:

    python3 -m unittest discover -s tests -v

Every test is deterministic (fixed seeds) and uses either a hand-built world
whose answer can be computed by hand, or one of the shipped worlds in
data/worlds.  The comments say which paper claim each test pins down.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.getcwd())

from scheduler import (ExprNode, FitnessEvaluator, Job, NodeType, SchedulerGP,  # noqa: E402
                       Satellite, SimulationSnapshot, buildTerminalRegistry)

REG = buildTerminalRegistry()
TERM = {t.name: i for i, t in enumerate(REG)}
WORLD0 = "data/worlds/world_000.csv"


# ── helpers ─────────────────────────────────────────────────────────────────
def terminal(name):
    n = ExprNode()
    n.nodeType = NodeType.TERMINAL
    n.terminalIndex = TERM[name]
    return n


def unary(op, child):
    n = ExprNode()
    n.nodeType = op
    n.children = [child]
    return n


def make_job(i, arrival=0, execution=10, transfer=0):
    j = Job()
    j.id, j.arrivalTime, j.type = i, arrival, "gen"
    j.ram, j.disk, j.imageSize = 1.0, 1.0, 1.0
    j.executionTime, j.transferTime, j.numberOfHops = execution, transfer, 1
    j.executionServerName = ""
    return j


def make_sat(i, energy=1000.0, until=1e9):
    s = Satellite()
    s.id, s.name = i, f"sat{i}"
    s.operationalUntil = until
    s.isAccessPoint = False
    s.elevationAngle, s.neighbors = 45.0, []
    s.latency, s.bandwidth = 1.0, 1.0
    s.completedTasks, s.cpuBusyUntil, s.networkBusyUntil = 0, 0, 0
    s.cpuCapacity, s.networkCapacity = 1.0, 1.0
    s.energyReserved, s.remainingEnergy = 0.0, energy
    s.rejectedTasks, s.tasks, s.deadTasks = [], [], []
    return s


def real_snapshot(world_index=0, limit=60, seed=1):
    from data_processing.csv_preprocessing import load_world_paths
    from data_processing.data_factory import create_jobs_from_csv, create_satellites_from_csv
    path = str(load_world_paths("data/worlds")[world_index])
    jobs = create_jobs_from_csv(path, limit=limit, random_sample=False, sample_seed=seed)
    minute = float(max(j.arrivalTime for j in jobs))
    sats = create_satellites_from_csv(path, limit=limit, seed=seed, random_sample=False,
                                      sample_seed=seed, snapshot_minute=minute)
    return SimulationSnapshot(minute, sats, jobs)


def depth(n):
    return 0 if not n.children else 1 + max(depth(c) for c in n.children)


def terminals_in(n, acc=None):
    acc = [] if acc is None else acc
    if n.nodeType == NodeType.TERMINAL:
        acc.append(n.terminalIndex)
    for c in n.children:
        terminals_in(c, acc)
    return acc


def evolve(snapshot, generations, seed, **options):
    gp = SchedulerGP([], [], seed=seed)
    gp.initialize(populationSize=40, crossoverRate=0.8, mutationRate=0.08,
                  maxDepth=5, **options)
    ev = FitnessEvaluator(REG)
    ev.trackBusyUntil = options.get("trackBusyUntil", False)
    for _ in range(generations):
        ev.evaluate(gp.getPopulation(), snapshot)
        gp.solveNextGeneration()
    ev.evaluate(gp.getPopulation(), snapshot)
    return gp


def best_tree_terminals(gp):
    best = gp.topIndividuals(1)[0]
    return sorted(terminals_in(best.tree))


# ── build information ───────────────────────────────────────────────────────
class BuildInfo(unittest.TestCase):
    def test_module_reports_how_it_was_built(self):
        # CMakeLists.txt passes the configuration name; bindings.cpp derives
        # `optimized` from the compiler.  The two must agree, because an
        # unoptimised module is many times slower and nothing else shows it.
        import scheduler
        self.assertIsInstance(scheduler.build_type, str)
        self.assertIsInstance(scheduler.optimized, bool)
        if scheduler.build_type == "Debug":
            self.assertFalse(scheduler.optimized)
        elif scheduler.build_type in ("Release", "RelWithDebInfo", "MinSizeRel"):
            self.assertTrue(scheduler.optimized)


# ── terminal weighting (Yang et al. 2025, Sec. IV-D; thesis Sec. 3.2.4) ──────
class TerminalWeightSolver(unittest.TestCase):
    def test_thesis_worked_example(self):
        # yang2025thesis Eq. 3.22: counts [3, 5, 8], r = 1/2 -> [1/24, 7/24, 2/3]
        p = SchedulerGP.solveTerminalWeights([3, 5, 8], 0.5)
        for got, want in zip(p, [1 / 24, 7 / 24, 2 / 3]):
            self.assertAlmostEqual(got, want, places=12)

    def test_full_sampling_ratio_is_the_empirical_frequency(self):
        # With r = 1 there is no uniform floor: the MLE is just x_i / sum(x).
        p = SchedulerGP.solveTerminalWeights([1, 2, 3, 0], 1.0)
        for got, want in zip(p, [1 / 6, 2 / 6, 3 / 6, 0.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_is_a_probability_vector_and_a_maximiser(self):
        # Compare against random feasible points: no one may beat the solution.
        import random
        rng = random.Random(0)
        counts, r = [7, 0, 2, 11, 1, 0, 4], 0.6
        n = len(counts)
        p = SchedulerGP.solveTerminalWeights(counts, r)
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertTrue(all(v >= 0 for v in p))

        def loglik(q):
            return sum(x * math.log(r * qi + (1 - r) / n) for x, qi in zip(counts, q))

        best = loglik(p)
        for _ in range(2000):
            w = [rng.random() ** 3 for _ in range(n)]
            q = [v / sum(w) for v in w]
            self.assertLessEqual(loglik(q), best + 1e-12)

    def test_no_observations_gives_uniform(self):
        p = SchedulerGP.solveTerminalWeights([0, 0, 0, 0], 0.7)
        self.assertEqual(p, [0.25] * 4)

    def test_sampling_ratio_schedule(self):
        # "log rev" (thesis Table 3.9): starts at 0, increases, capped at 0.9.
        rs = [SchedulerGP.terminalSamplingRatio(g) for g in range(0, 3000)]
        self.assertEqual(rs[0], 0.0)
        self.assertTrue(all(b >= a for a, b in zip(rs, rs[1:])))
        self.assertAlmostEqual(rs[50], 1 - 1 / (1 + 2 * math.log(51)), places=12)
        self.assertEqual(rs[-1], 0.9)


# ── evaluation on several worlds (Yang Sec. V-A, Zhang & Yang Sec. 4.1) ──────
@unittest.skipUnless(os.path.exists(WORLD0), "needs data/worlds")
class MultiWorldEvaluation(unittest.TestCase):
    def setUp(self):
        self.snaps = [real_snapshot(0, seed=1), real_snapshot(3, seed=2)]
        self.gp = SchedulerGP([], [], seed=5)
        self.gp.initialize(populationSize=30, crossoverRate=0.8, mutationRate=0.08, maxDepth=4)
        self.ev = FitnessEvaluator(REG)

    def test_one_world_equals_evaluate(self):
        pop = self.gp.getPopulation()
        self.ev.evaluate(pop, self.snaps[0])
        single = [i.fitness for i in pop]
        self.ev.evaluate_many(pop, [self.snaps[0]])
        self.assertEqual(single, [i.fitness for i in pop])

    def test_pooling_matches_an_independent_computation(self):
        # Re-derive the pooled objectives from evaluate_single (raw values per
        # world), normalise by hand, and compare with evaluate_many.
        pop = self.gp.getPopulation()
        self.ev.evaluate_many(pop, self.snaps)
        got = [i.fitness for i in pop]

        n_jobs = [len(s.jobs) for s in self.snaps]
        ratios, resps, energies = [], [], []
        for ind in pop:
            sched = resp_total = energy = 0.0
            for sn, nj in zip(self.snaps, n_jobs):
                ratio, mean_resp, en = self.ev.evaluate_single(ind, sn)
                k = round(ratio * nj)
                sched += k
                resp_total += mean_resp * k
                energy += en
            ratios.append(sched / sum(n_jobs))
            resps.append(resp_total / sched if sched else 0.0)
            energies.append(energy)

        def minmax(v, skip=()):
            vals = [x for i, x in enumerate(v) if i not in skip]
            lo, hi = min(vals), max(vals)
            return [0.0 if hi - lo < 1e-12 else (x - lo) / (hi - lo) for x in v]

        empty = {i for i, r in enumerate(ratios) if r <= 0}
        rn = minmax(resps, skip=empty)
        for i in empty:
            rn[i] = 1.0
        en = minmax(energies)
        want = [self.ev.weightJobs * c - self.ev.weightTime * r - self.ev.weightEnergy * e
                for c, r, e in zip(ratios, rn, en)]
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=9)

    def test_empty_snapshot_list_zeroes_fitness(self):
        pop = self.gp.getPopulation()
        self.ev.evaluate_many(pop, [])
        self.assertTrue(all(i.fitness == 0.0 for i in pop))


# ── live availability terminal ───────────────────────────────────────────────
class BusyUntilTracking(unittest.TestCase):
    """Hand-computed: 4 identical 10-minute jobs, 2 identical satellites.

    Tree = -sat.cpuBusyUntil ("prefer the satellite that is free sooner").
    Untracked the terminal is constant 0, every pair ties, and the decoder keeps
    choosing the first satellite: completions 10, 20, 30, 40 (mean 25).
    Tracked, the satellites alternate: 10, 10, 20, 20 (mean 15).
    """

    def snapshot(self):
        return SimulationSnapshot(0.0, [make_sat(0), make_sat(1)],
                                  [make_job(i) for i in range(4)])

    def test_tracking_balances_the_satellites(self):
        ind = SchedulerGP([], [], seed=1)
        ind.initialize(populationSize=2, crossoverRate=0.8, mutationRate=0.08, maxDepth=3)
        individual = ind.getPopulation()[0]
        individual.tree = unary(NodeType.NEG, terminal("sat.cpuBusyUntil"))

        ev = FitnessEvaluator(REG)
        _, mean_off, _ = ev.evaluate_single(individual, self.snapshot())
        ev.trackBusyUntil = True
        ratio_on, mean_on, _ = ev.evaluate_single(individual, self.snapshot())
        self.assertAlmostEqual(mean_off, 25.0)
        self.assertAlmostEqual(mean_on, 15.0)
        self.assertEqual(ratio_on, 1.0)

    def test_constant_terminals_report(self):
        if not os.path.exists(WORLD0):
            self.skipTest("needs data/worlds")
        snap = real_snapshot(0)
        ev = FitnessEvaluator(REG)
        names = {REG[i].name for i in ev.constant_terminals(snap)}
        # These four are 0 for every satellite in the shipped data.
        self.assertEqual(names, {"sat.cpuBusyUntil", "sat.networkBusyUntil",
                                 "sat.energyReserved", "sat.loadRatio"})
        # The decoder updates completedTasks, so it is never "dead".
        self.assertNotIn("sat.completedTasks", names)
        # Once the decoder writes cpuBusyUntil it is no longer dead either.
        ev.trackBusyUntil = True
        self.assertNotIn("sat.cpuBusyUntil", {REG[i].name for i in ev.constant_terminals(snap)})


# ── fixed-bound fitness (Marler & Arora 2004; Shabtay et al. 2013) ───────────
class FixedBoundFitness(unittest.TestCase):
    """Hand-computed world.  Snapshot minute 4; jobs (arrival, duration) =
    (0, 10), (2, 20), (4, 30); satellite 0 has 30 energy units and no sunset,
    satellite 1 has 25 units and sets at minute 19; energy rate 1 per minute.

    Bounds (detail::computeWorldBounds).  D = 60, the work available.
    T_0 = min(60, inf, 30) = 30 and T_1 = min(60, 19 - 4, 25) = 15, so
        Rmax = (4 - 0) + 30 = 34,    Emax = min(30 + 15, 1 * 60) = 45.

    Shortest job first (-job.executionTime): job 0 goes to satellite 0 (done at
    14, response 14), job 1 to satellite 0 (done at 34, response 32), job 2 fits
    nowhere.  C = 2/3, mean response 23, energy 30, hence
        R~ = (14/34 + 32/34 + 1) / 3 = 80/102,    E~ = 30/45,
        fitness = 0.60 * 2/3 - 0.25 * 80/102 - 0.15 * 30/45 = 0.10392...
    """

    def world(self):
        return SimulationSnapshot(
            4.0,
            [make_sat(0, energy=30.0), make_sat(1, energy=25.0, until=19.0)],
            [make_job(0, arrival=0, execution=10), make_job(1, arrival=2, execution=20),
             make_job(2, arrival=4, execution=30)])

    @staticmethod
    def synthetic_world():
        # 40 jobs and 8 satellites with different budgets and sunsets; built in
        # code so the test needs no data file.
        jobs = [make_job(i, arrival=i % 9, execution=5 + (i * 7) % 23, transfer=(i * 3) % 6)
                for i in range(40)]
        sats = [make_sat(s, energy=20.0 + 7.0 * s, until=48.0 + 15.0 * s) for s in range(8)]
        return SimulationSnapshot(8.0, sats, jobs)

    @staticmethod
    def population(trees):
        gp = SchedulerGP([], [], seed=1)
        gp.initialize(populationSize=len(trees), crossoverRate=0.8, mutationRate=0.08, maxDepth=3)
        pop = gp.getPopulation()
        for i, tree in enumerate(trees):
            pop[i].tree = tree
        return gp, pop

    @staticmethod
    def trees():
        return [unary(NodeType.NEG, terminal("job.executionTime")),   # shortest job first
                terminal("job.executionTime"),                        # longest job first
                terminal("sat.remainingEnergy")]

    @staticmethod
    def formula(ev, pooled):
        """Fixed-bound fitness from raw per-world results.

        `pooled` is a list of (C, mean response, energy, jobs, Rmax, Emax)."""
        jobs = sum(p[3] for p in pooled)
        scheduled = [p[0] * p[3] for p in pooled]
        comp = sum(scheduled) / jobs
        resp = (sum(k * p[1] / p[4] for k, p in zip(scheduled, pooled)) + jobs - sum(scheduled)) / jobs
        energy = sum(p[2] for p in pooled) / sum(p[5] for p in pooled)
        return ev.weightJobs * comp - ev.weightTime * resp - ev.weightEnergy * energy

    def test_off_by_default(self):
        self.assertFalse(FitnessEvaluator(REG).fixedBounds)

    def test_world_bounds_by_hand(self):
        r_max, e_max = FitnessEvaluator.world_bounds(self.world())
        self.assertAlmostEqual(r_max, 34.0, places=12)
        self.assertAlmostEqual(e_max, 45.0, places=12)

    def test_fitness_by_hand(self):
        gp, pop = self.population(self.trees())
        ev = FitnessEvaluator(REG)
        for got, want in zip(ev.evaluate_single(pop[0], self.world()), (2 / 3, 23.0, 30.0)):
            self.assertAlmostEqual(got, want, places=12)
        ev.fixedBounds = True
        ev.evaluate(pop, self.world())
        self.assertAlmostEqual(pop[0].fitness, 0.60 * 2 / 3 - 0.25 * 80 / 102 - 0.15 * 30 / 45,
                               places=12)

    def test_fitness_does_not_depend_on_the_population(self):
        # The defining property: the same tree scores the same whatever the
        # other individuals are (population-relative fitness cannot promise it).
        snap = self.synthetic_world()
        ev = FitnessEvaluator(REG)
        ev.fixedBounds = True
        gp, pop = self.population(self.trees())
        ev.evaluate(pop, snap)
        alone = pop[0].fitness
        gp2, pop2 = self.population([self.trees()[0], terminal("job.arrivalTime"),
                                     unary(NodeType.NEG, terminal("sat.completedTasks")),
                                     terminal("job.imageSize"), terminal("sat.bandwidth")])
        ev.evaluate(pop2, snap)
        self.assertEqual(pop2[0].fitness, alone)

    def test_every_schedule_respects_the_bounds(self):
        # Rmax and Emax are upper bounds for ANY tree, so both terms are in
        # [0, 1] and the fitness is in [-(wT + wE), wJ].
        snap = self.synthetic_world()
        r_max, e_max = FitnessEvaluator.world_bounds(snap)
        gp = SchedulerGP([], [], seed=3)
        gp.initialize(populationSize=60, crossoverRate=0.8, mutationRate=0.08, maxDepth=5)
        pop = gp.getPopulation()
        ev = FitnessEvaluator(REG)
        ev.fixedBounds = True
        ev.evaluate(pop, snap)
        for ind in pop:
            comp, resp, energy = ev.evaluate_single(ind, snap)
            self.assertLessEqual(resp, r_max + 1e-9)
            self.assertLessEqual(energy, e_max + 1e-9)
            self.assertLessEqual(ind.fitness, ev.weightJobs + 1e-12)
            self.assertGreaterEqual(ind.fitness, -(ev.weightTime + ev.weightEnergy) - 1e-12)
            self.assertAlmostEqual(
                ind.fitness,
                self.formula(ev, [(comp, resp, energy, len(snap.jobs), r_max, e_max)]), places=12)

    def test_empty_schedule_gets_the_worst_response(self):
        # No energy anywhere: nothing can run.  C = 0, R~ = 1, E~ = 0, so the
        # fitness is exactly -weightTime.  (Population-relative fitness needed a
        # special case for this; here it follows from the rejection penalty.)
        jobs = [make_job(0, arrival=0, execution=10), make_job(1, arrival=4, execution=20)]
        snap = SimulationSnapshot(4.0, [make_sat(0, energy=0.0), make_sat(1, energy=0.0)], jobs)
        gp, pop = self.population(self.trees())
        ev = FitnessEvaluator(REG)
        ev.fixedBounds = True
        ev.evaluate(pop, snap)
        for ind in pop:
            self.assertAlmostEqual(ind.fitness, -ev.weightTime, places=12)

    def test_several_worlds_are_pooled_per_job(self):
        # World 2: minute 3, jobs (0,5) (1,7) (3,9) and one that arrives at 9
        # (not available: it counts as unscheduled).  Budgets 12 and 9, no
        # sunset.  D = 21, T = (12, 9): Rmax = 3 + 12 = 15, Emax = min(21, 21).
        snap1 = self.world()
        snap2 = SimulationSnapshot(
            3.0, [make_sat(0, energy=12.0), make_sat(1, energy=9.0)],
            [make_job(0, arrival=0, execution=5), make_job(1, arrival=1, execution=7),
             make_job(2, arrival=3, execution=9), make_job(3, arrival=9, execution=4)])
        b1, b2 = FitnessEvaluator.world_bounds(snap1), FitnessEvaluator.world_bounds(snap2)
        self.assertAlmostEqual(b2[0], 15.0, places=12)
        self.assertAlmostEqual(b2[1], 21.0, places=12)

        gp, pop = self.population(self.trees())
        ev = FitnessEvaluator(REG)
        ev.fixedBounds = True
        ev.evaluate_many(pop, [snap1, snap2])
        for ind in pop:
            c1, r1, e1 = ev.evaluate_single(ind, snap1)
            c2, r2, e2 = ev.evaluate_single(ind, snap2)
            want = self.formula(ev, [(c1, r1, e1, 3, b1[0], b1[1]), (c2, r2, e2, 4, b2[0], b2[1])])
            self.assertAlmostEqual(ind.fitness, want, places=12)

    def test_best_fitness_never_decreases_under_elitism(self):
        # Absolute fitness + elitism on a fixed world => the best fitness is
        # monotone.  With the population-relative fitness it can go down
        # (docs/onboarding/quirks.html), so this is the property that makes
        # "N generations without improvement" a meaningful stagnation test.
        snap = self.synthetic_world()
        gp = SchedulerGP([], [], seed=7)
        gp.initialize(populationSize=40, crossoverRate=0.8, mutationRate=0.08,
                      elitismCount=2, maxDepth=5)
        ev = FitnessEvaluator(REG)
        ev.fixedBounds = True
        best = -math.inf
        for _ in range(40):
            ev.evaluate(gp.getPopulation(), snap)
            current = max(ind.fitness for ind in gp.getPopulation())
            self.assertGreaterEqual(current, best)
            best = current
            gp.solveNextGeneration()

    def test_switching_back_restores_the_population_relative_fitness(self):
        snap = self.synthetic_world()
        gp, pop = self.population(self.trees())
        ev = FitnessEvaluator(REG)
        ev.evaluate(pop, snap)
        legacy = [ind.fitness for ind in pop]
        ev.fixedBounds = True
        ev.evaluate(pop, snap)
        ev.fixedBounds = False
        ev.evaluate(pop, snap)
        self.assertEqual(legacy, [ind.fitness for ind in pop])


# ── decoder tie-break (Graham 1966; Topcuoglu et al. 2002; Braun et al. 2001) ─
class EarliestFreeTieBreak(unittest.TestCase):
    """Hand-computed.  Snapshot minute 0, every job has arrived, no sunset, ample
    energy, so only the selection rule matters.

    The decoder picks the feasible pair with the highest score.  By default a tie
    goes to the first pair of the scan (lowest job, then lowest satellite).  With
    earliestFreeTieBreak it goes to the pair that can start first, i.e. the
    lexicographic maximum of (score, -start, -job index, -satellite index).
    """

    @staticmethod
    def individual(tree):
        gp = SchedulerGP([], [], seed=1)
        gp.initialize(populationSize=2, crossoverRate=0.8, mutationRate=0.08, maxDepth=3)
        ind = gp.getPopulation()[0]
        ind.tree = tree
        return gp, ind

    @staticmethod
    def evaluator(tie_break):
        ev = FitnessEvaluator(REG)
        ev.earliestFreeTieBreak = tie_break
        return ev

    def test_off_by_default(self):
        self.assertFalse(FitnessEvaluator(REG).earliestFreeTieBreak)

    def test_job_only_rule_spreads_over_idle_satellites(self):
        # Four identical 10-minute jobs, two identical satellites, rule =
        # job.executionTime: every pair scores 10.
        #   default      : all four on satellite 0 -> done at 10, 20, 30, 40 (mean 25)
        #   earliest free: 0,1,0,1                 -> done at 10, 10, 20, 20 (mean 15)
        snap = SimulationSnapshot(0.0, [make_sat(0), make_sat(1)], [make_job(i) for i in range(4)])
        gp, ind = self.individual(terminal("job.executionTime"))
        ratio, mean_default, _ = self.evaluator(False).evaluate_single(ind, snap)
        self.assertEqual(ratio, 1.0)
        self.assertAlmostEqual(mean_default, 25.0)
        ratio, mean_spread, _ = self.evaluator(True).evaluate_single(ind, snap)
        self.assertEqual(ratio, 1.0)
        self.assertAlmostEqual(mean_spread, 15.0)

    def test_job_order_still_follows_the_rule(self):
        # Shortest job first on two satellites.  Durations 30, 10, 20, 40.
        # Order of selection: 10, 20, 30, 40.
        #   10 -> sat 0 (both free at 0: lowest index)   done 10
        #   20 -> sat 1 (free at 0 < 10)                 done 20
        #   30 -> sat 0 (free at 10 < 20)                done 40
        #   40 -> sat 1 (free at 20 < 40)                done 60
        # mean response = (10 + 20 + 40 + 60) / 4 = 32.5.
        # Default: all on satellite 0: 10, 30, 60, 100 -> mean 50.
        jobs = [make_job(i, execution=d) for i, d in enumerate([30, 10, 20, 40])]
        snap = SimulationSnapshot(0.0, [make_sat(0), make_sat(1)], jobs)
        gp, ind = self.individual(unary(NodeType.NEG, terminal("job.executionTime")))
        self.assertAlmostEqual(self.evaluator(False).evaluate_single(ind, snap)[1], 50.0)
        self.assertAlmostEqual(self.evaluator(True).evaluate_single(ind, snap)[1], 32.5)

    def test_equal_start_keeps_the_lowest_satellite_index(self):
        # One job, two idle satellites: score and start both tie, so the scan
        # order decides and satellite 0 gets it.  The satellites differ only in
        # energy rate (1 and 3 per minute), which makes the choice observable:
        # energy 10 on satellite 0, 30 on satellite 1.
        snap = SimulationSnapshot(0.0, [make_sat(0), make_sat(1)], [make_job(0)])
        snap.energyPerMinute = [1.0, 3.0]
        gp, ind = self.individual(terminal("job.executionTime"))
        for tie_break in (False, True):
            self.assertAlmostEqual(self.evaluator(tie_break).evaluate_single(ind, snap)[2], 10.0)

    def test_a_strictly_higher_score_beats_an_earlier_start(self):
        # The tree prefers satellite 1 outright (higher bandwidth).  The option
        # must not override that: all four jobs queue on satellite 1 (mean 25)
        # although satellite 0 is idle the whole time.
        sats = [make_sat(0), make_sat(1)]
        sats[1].bandwidth = 5.0
        snap = SimulationSnapshot(0.0, sats, [make_job(i) for i in range(4)])
        snap.energyPerMinute = [1.0, 3.0]
        gp, ind = self.individual(terminal("sat.bandwidth"))
        for tie_break in (False, True):
            ratio, mean_resp, energy = self.evaluator(tie_break).evaluate_single(ind, snap)
            self.assertEqual(ratio, 1.0)
            self.assertAlmostEqual(mean_resp, 25.0)
            self.assertAlmostEqual(energy, 4 * 10 * 3.0)        # everything on satellite 1

    def test_infinite_and_nan_scores_are_still_scheduled(self):
        # job.ram = 1e200, so ram * ram overflows to +inf.
        #   -(ram * ram)              = -inf for every pair
        #   (ram * ram) - (ram * ram) =  NaN for every pair
        # Default: a pair must beat the initial -inf with a strict '>', which
        # neither -inf nor NaN can do, so nothing is scheduled.  With the
        # option the order is total and the first feasible pair is accepted.
        def big(i):
            job = make_job(i)
            job.ram = 1e200
            return job

        def product():
            n = ExprNode()
            n.nodeType = NodeType.MUL
            n.children = [terminal("job.ram"), terminal("job.ram")]
            return n

        minus_inf = unary(NodeType.NEG, product())
        nan = ExprNode()
        nan.nodeType = NodeType.SUB
        nan.children = [product(), product()]

        snap = SimulationSnapshot(0.0, [make_sat(0), make_sat(1)], [big(i) for i in range(4)])
        for tree in (minus_inf, nan):
            gp, ind = self.individual(tree)
            self.assertEqual(self.evaluator(False).evaluate_single(ind, snap)[0], 0.0)
            ratio, mean_resp, _ = self.evaluator(True).evaluate_single(ind, snap)
            self.assertEqual(ratio, 1.0)
            self.assertAlmostEqual(mean_resp, 15.0)

    def test_nan_ranks_below_every_number(self):
        # One satellite.  Job 0 (30 minutes) scores NaN, job 1 (10 minutes)
        # scores a finite number: job 1 must go first -> done at 10 and 40,
        # mean 25 (the other order would give 30 and 40, mean 35).
        # Tree: (ram * ram) - (ram * ram) is NaN for ram = 1e200 and 0 for ram = 1.
        def product():
            n = ExprNode()
            n.nodeType = NodeType.MUL
            n.children = [terminal("job.ram"), terminal("job.ram")]
            return n

        tree = ExprNode()
        tree.nodeType = NodeType.SUB
        tree.children = [product(), product()]
        slow = make_job(0, execution=30)
        slow.ram = 1e200
        snap = SimulationSnapshot(0.0, [make_sat(0)], [slow, make_job(1, execution=10)])
        gp, ind = self.individual(tree)
        ratio, mean_resp, _ = self.evaluator(True).evaluate_single(ind, snap)
        self.assertEqual(ratio, 1.0)
        self.assertAlmostEqual(mean_resp, 25.0)

    def test_no_job_finishes_later_for_a_job_only_rule(self):
        # Proposition 2 (thesis, method chapter): for a rule that looks at the
        # job only, without sunset or energy limits, every job completes no
        # later with the earliest-free tie-break than with the default.  The
        # mean response is therefore never larger.  Checked on 20 random
        # job-only trees and a world with unequal jobs.
        jobs = [make_job(i, arrival=0, execution=5 + (i * 7) % 23, transfer=(i * 3) % 6)
                for i in range(30)]
        for j in jobs:                      # make the job terminals differ
            j.ram, j.disk, j.imageSize = 1.0 + j.id % 4, 2.0 + j.id % 7, 3.0 + j.id % 5
        snap = SimulationSnapshot(0.0, [make_sat(s) for s in range(5)], jobs)
        job_terminals = [i for i, t in enumerate(REG) if t.name.startswith("job.")]
        satellite_terminals = [i for i, t in enumerate(REG) if t.name.startswith("sat.")]

        gp = SchedulerGP([], [], seed=4)
        gp.initialize(populationSize=20, crossoverRate=0.8, mutationRate=0.08, maxDepth=4,
                      excludedTerminals=satellite_terminals)
        for ind in gp.getPopulation():
            self.assertTrue(set(terminals_in(ind.tree)) <= set(job_terminals))
            ratio_a, mean_default, _ = self.evaluator(False).evaluate_single(ind, snap)
            ratio_b, mean_spread, _ = self.evaluator(True).evaluate_single(ind, snap)
            # -inf / NaN scores can make the default schedule fewer jobs; the
            # proposition compares schedules of the same jobs.
            if ratio_a == ratio_b == 1.0:
                self.assertLessEqual(mean_spread, mean_default + 1e-9)

    def test_tight_energy_can_make_spreading_schedule_fewer_jobs(self):
        # The caveat of Proposition 2: with the energy filter the option is NOT
        # always better.  Two satellites with 30 energy units each, jobs of 10,
        # 10 and 30 minutes, shortest job first.
        #   default      : 10 -> sat 0 (20 left), 10 -> sat 0 (10 left),
        #                  30 -> sat 1 (fits exactly)           3 of 3 scheduled
        #   earliest free: 10 -> sat 0 (20 left), 10 -> sat 1 (20 left),
        #                  30 fits nowhere                      2 of 3 scheduled
        # Filling one satellite first keeps the other one's budget whole, as
        # first-fit does in bin packing.  This is why the option is off by
        # default and has to be measured on the real worlds.
        jobs = [make_job(i, execution=d) for i, d in enumerate([10, 10, 30])]
        snap = SimulationSnapshot(0.0, [make_sat(0, energy=30.0), make_sat(1, energy=30.0)], jobs)
        gp, ind = self.individual(unary(NodeType.NEG, terminal("job.executionTime")))
        self.assertAlmostEqual(self.evaluator(False).evaluate_single(ind, snap)[0], 1.0)
        self.assertAlmostEqual(self.evaluator(True).evaluate_single(ind, snap)[0], 2 / 3)

    def test_bounds_of_the_fixed_bound_fitness_still_hold(self):
        # The tie-break changes which pair is picked, not the decoder's
        # constraints, so Rmax and Emax remain upper bounds.
        snap = FixedBoundFitness.synthetic_world()
        r_max, e_max = FitnessEvaluator.world_bounds(snap)
        gp = SchedulerGP([], [], seed=3)
        gp.initialize(populationSize=60, crossoverRate=0.8, mutationRate=0.08, maxDepth=5)
        ev = self.evaluator(True)
        ev.fixedBounds = True
        ev.evaluate(gp.getPopulation(), snap)
        for ind in gp.getPopulation():
            _, resp, energy = ev.evaluate_single(ind, snap)
            self.assertLessEqual(resp, r_max + 1e-9)
            self.assertLessEqual(energy, e_max + 1e-9)
            self.assertGreaterEqual(ind.fitness, -(ev.weightTime + ev.weightEnergy) - 1e-12)


# ── GP options ───────────────────────────────────────────────────────────────
@unittest.skipUnless(os.path.exists(WORLD0), "needs data/worlds")
class GPOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = real_snapshot(0, limit=60, seed=1)

    def test_same_seed_same_evolution(self):
        opts = dict(tournamentSize=7, initialMaxDepth=3, internalCrossoverBias=0.9,
                    cloneRetries=2, terminalWeighting=True, excludedTerminals=[TERM["sat.loadRatio"]])
        a = evolve(self.snap, 8, seed=11, **opts)
        b = evolve(self.snap, 8, seed=11, **opts)
        c = evolve(self.snap, 8, seed=12, **opts)
        self.assertEqual(best_tree_terminals(a), best_tree_terminals(b))
        sig = lambda gp: [i.fitness for i in gp.getPopulation()]  # noqa: E731
        self.assertEqual(sig(a), sig(b))
        self.assertNotEqual(sig(a), sig(c))

    def test_initial_depth_is_below_the_cap(self):
        gp = SchedulerGP([], [], seed=3)
        gp.initialize(populationSize=60, crossoverRate=0.8, mutationRate=0.08,
                      maxDepth=7, initialMaxDepth=2)
        self.assertTrue(all(depth(i.tree) <= 2 for i in gp.getPopulation()))

    def test_depth_one_cap_no_longer_divides_by_zero(self):
        gp = SchedulerGP([], [], seed=3)
        gp.initialize(populationSize=10, crossoverRate=0.8, mutationRate=0.08, maxDepth=1)
        self.assertTrue(all(depth(i.tree) <= 1 for i in gp.getPopulation()))

    def test_excluded_terminals_are_never_drawn(self):
        dead = [TERM[n] for n in ("sat.networkBusyUntil", "sat.energyReserved", "sat.loadRatio")]
        gp = evolve(self.snap, 12, seed=4, excludedTerminals=dead, terminalWeighting=True)
        for ind in gp.getPopulation():
            self.assertFalse(set(terminals_in(ind.tree)) & set(dead))

    def test_bad_excluded_terminals_are_rejected(self):
        gp = SchedulerGP([], [], seed=1)
        with self.assertRaises(ValueError):
            gp.initialize(populationSize=10, crossoverRate=0.8, mutationRate=0.08,
                          excludedTerminals=[len(REG)])
        with self.assertRaises(ValueError):
            gp.initialize(populationSize=10, crossoverRate=0.8, mutationRate=0.08,
                          excludedTerminals=list(range(len(REG))))

    def test_size_and_depth_caps_hold_for_every_option(self):
        cap = 4
        gp = SchedulerGP([], [], seed=9)
        gp.initialize(populationSize=50, crossoverRate=0.9, mutationRate=0.2, maxDepth=cap,
                      initialMaxDepth=3, internalCrossoverBias=1.0, cloneRetries=3,
                      terminalWeighting=True, hoistRate=0.1)
        ev = FitnessEvaluator(REG)
        for _ in range(15):
            ev.evaluate(gp.getPopulation(), self.snap)
            gp.solveNextGeneration()
            for ind in gp.getPopulation():
                self.assertLessEqual(depth(ind.tree), cap)
                self.assertLessEqual(ind.tree.size(), 2 ** (cap + 1))

    def test_clone_retries_keep_the_population_diverse(self):
        # With crossover and mutation off every offspring is a copy of a
        # parent; the clone check is what injects variation.
        def diversity(retries):
            gp = SchedulerGP([], [], seed=2)
            gp.initialize(populationSize=40, crossoverRate=0.0, mutationRate=0.0, maxDepth=4,
                          cloneRetries=retries)
            ev = FitnessEvaluator(REG)
            for _ in range(6):
                ev.evaluate(gp.getPopulation(), self.snap)
                gp.solveNextGeneration()
            return gp.structuralDiversity()

        self.assertGreater(diversity(3), diversity(0))

    def test_tournament_size_changes_selection_pressure(self):
        # Fitness = number of copies of one terminal, assigned by hand.  A
        # larger tournament must raise the mean count faster than size 1
        # (pure random selection), whatever the seed.
        target = TERM["job.executionTime"]

        def mean_count(tournament):
            gp = SchedulerGP([], [], seed=21)
            gp.initialize(populationSize=80, crossoverRate=0.8, mutationRate=0.05,
                          maxDepth=5, tournamentSize=tournament)
            for _ in range(10):
                for ind in gp.getPopulation():
                    ind.fitness = float(terminals_in(ind.tree).count(target))
                gp.solveNextGeneration()
            pop = gp.getPopulation()
            return sum(terminals_in(i.tree).count(target) for i in pop) / len(pop)

        self.assertGreater(mean_count(7), mean_count(1))

    def test_terminal_weighting_biases_mutation_towards_the_elite_terminals(self):
        # Fitness = copies of one terminal.  Mutation-only evolution isolates the
        # operator that the weighting changes.
        target = TERM["job.executionTime"]

        def mean_count(weighting):
            total = 0.0
            for seed in range(6):
                gp = SchedulerGP([], [], seed=100 + seed)
                gp.initialize(populationSize=60, crossoverRate=0.0, mutationRate=0.25,
                              maxDepth=5, terminalWeighting=weighting)
                for _ in range(40):
                    for ind in gp.getPopulation():
                        ind.fitness = float(terminals_in(ind.tree).count(target))
                    gp.solveNextGeneration()
                pop = gp.getPopulation()
                total += sum(terminals_in(i.tree).count(target) for i in pop) / len(pop)
            return total / 6

        self.assertGreater(mean_count(True), mean_count(False))


if __name__ == "__main__":
    unittest.main()
