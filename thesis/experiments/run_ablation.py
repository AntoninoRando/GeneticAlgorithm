"""Ablation of the literature-based GP options on held-out worlds.

Every configuration is trained on worlds 0..19 and scored on the held-out worlds
(20..), exactly like thesis/experiments/run_experiments.py (same world seeds,
same population/depth/generations, same "best of the top-K by training
completion" selection), changing only the options under test.  Seeds are paired:
seed s uses the same random stream start for every configuration.

Run from the project root, with the compiled `scheduler` module importable:

    python3 thesis/experiments/run_ablation.py --seeds 10 --out thesis/experiments/results_ablation

The metrics are the thesis metrics on test worlds: completion ratio C (higher is
better), mean response time R in minutes and energy E (lower is better).  The
population-relative fitness is NOT comparable across runs, so it is not used.

Where each option comes from (see include/scheduler/*.h for section numbers):
  tournament7    Sun 2024 / Yang 2025 use tournaments of 7
  init3          initial trees below the depth cap (Sun, Yang, Zhang & Yang, Koza)
  xbias          90% internal crossover points (Koza 1994, Sec. 4)
  clones         repetition test of Zhang & Yang 2025, Sec. 3.2.1 (structural proxy)
  termweight     terminal weighting of Yang et al. 2025, Sec. IV-D
  prune          drop constant terminals (registry.h notes; cf. small hand-picked
                 terminal sets in the papers)
  busy           live availability terminal (Zhang & Yang NT/CL, Sun AAT/IAT)
  worlds3        score each individual on 3 worlds (Yang 5 instances, Zhang & Yang
                 100 simulations)
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiments as rx  # noqa: E402  (thesis harness: worlds, metrics, helpers)
from scheduler import FitnessEvaluator, SchedulerGP  # noqa: E402

POP, DEPTH, GENS, HOIST, SWITCH = 100, 4, 300, 0.05, 10

CONFIGS = {
    "baseline":    dict(),
    "tournament7": dict(tournamentSize=7),
    "init3":       dict(initialMaxDepth=3),
    "xbias":       dict(internalCrossoverBias=0.9),
    "clones":      dict(cloneRetries=3),
    "termweight":  dict(terminalWeighting=True),
    "prune":       dict(prune=True),
    "busy":        dict(trackBusyUntil=True),
    "worlds3":     dict(worldsPerEval=3),
    "prune+busy":  dict(prune=True, trackBusyUntil=True),
    "clones+worlds3": dict(cloneRetries=3, worldsPerEval=3),
    "all":         dict(tournamentSize=7, initialMaxDepth=3, internalCrossoverBias=0.9,
                        cloneRetries=3, terminalWeighting=True, prune=True,
                        trackBusyUntil=True, worldsPerEval=3),
}

_TRAIN = _TEST = None


def _init():
    global _TRAIN, _TEST
    _TRAIN, _TEST = rx.load_worlds()


def train_one(args):
    name, opts, seed = args
    opts = dict(opts)
    prune = opts.pop("prune", False)
    track = opts.pop("trackBusyUntil", False)
    k = opts.pop("worldsPerEval", 1)

    fe = FitnessEvaluator(rx.REG)
    fe.trackBusyUntil = track

    excluded = []
    if prune:
        # Constant in every training world (see main.py, --prune-constant-terminals).
        excluded = sorted(set.intersection(*[set(fe.constant_terminals(s)) for s in _TRAIN]))

    t0 = time.time()
    ga = SchedulerGP([], [], seed=seed)
    ga.initialize(populationSize=POP, maxDepth=DEPTH, hoistRate=HOIST,
                  excludedTerminals=excluded, **opts)
    P = ga.getPopulation()
    for g in range(GENS):
        idx = (g // SWITCH) % len(_TRAIN)
        if k == 1:
            fe.evaluate(P, _TRAIN[idx])
        else:
            fe.evaluate_many(P, [_TRAIN[(idx + j) % len(_TRAIN)] for j in range(k)])
        if g < GENS - 1:
            ga.solveNextGeneration()

    # Same selection as run_experiments.train: best of the top-K by fitness,
    # ranked by mean completion (then response) on the TRAINING worlds.
    top = sorted(range(len(P)), key=lambda i: -P[i].fitness)[:rx.TOP_K]
    ranked = []
    for i in top:
        m = rx.mean_metrics(fe, P[i], _TRAIN)
        ranked.append((-m[0], m[1], i))
    i = min(ranked)[2]
    C, R, E = rx.mean_metrics(fe, P[i], _TEST)
    return dict(config=name, seed=seed, C=C, R=R, E=E, size=P[i].tree.size(),
                excluded=len(excluded), diversity=ga.structuralDiversity(),
                tree=rx.tostr(P[i].tree), seconds=time.time() - t0)


def mann_whitney_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation, tie-corrected)."""
    n1, n2 = len(a), len(b)
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    counts = {}
    for v, _ in allv:
        counts[v] = counts.get(v, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    sd = math.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie / ((n1 + n2) * (n1 + n2 - 1))))
    if sd == 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / sd
    return math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="First seed. Use a fresh range to confirm a result found with seeds 0..n-1.")
    ap.add_argument("--out", default="thesis/experiments/results_ablation")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in a.configs.split(",") if n]

    jobs = [(n, CONFIGS[n], s) for n in names
            for s in range(a.seed_start, a.seed_start + a.seeds)]
    t0 = time.time()
    with mp.Pool(a.procs, initializer=_init) as pool:
        runs = []
        for r in pool.imap_unordered(train_one, jobs):
            runs.append(r)
            print(f"  {r['config']:12s} seed {r['seed']:2d}  C={r['C']:.3f} R={r['R']:.2f} "
                  f"E={r['E']:.1f}  ({r['seconds']:.0f}s)", flush=True)
    json.dump(runs, open(os.path.join(a.out, "ablation.json"), "w"), indent=1)

    by = {n: sorted([r for r in runs if r["config"] == n], key=lambda r: r["seed"]) for n in names}
    base = by.get("baseline")
    print(f"\n{a.seeds} seeds, {len(rx.load_worlds()[1])} held-out worlds, total {time.time() - t0:.0f}s")
    print(f"{'config':12s} {'C mean':>8s} {'C sd':>7s} {'R mean':>8s} {'E mean':>8s} "
          f"{'size':>6s} {'div':>5s} {'p(C)':>7s} {'wins':>5s}")
    for n in names:
        C = [r["C"] for r in by[n]]
        R = [r["R"] for r in by[n]]
        E = [r["E"] for r in by[n]]
        p = wins = ""
        if base and n != "baseline":
            p = f"{mann_whitney_p(C, [r['C'] for r in base]):.3f}"
            wins = f"{sum(c > b['C'] for c, b in zip(C, base))}/{len(C)}"
        print(f"{n:12s} {st.fmean(C):8.4f} {st.pstdev(C):7.4f} {st.fmean(R):8.2f} {st.fmean(E):8.1f} "
              f"{st.fmean(r['size'] for r in by[n]):6.1f} "
              f"{st.fmean(r['diversity'] for r in by[n]):5.2f} {p:>7s} {wins:>5s}")


if __name__ == "__main__":
    main()
