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
