"""Experiments for the thesis: train GP schedulers on training worlds and
evaluate them on held-out worlds against hand-written baselines.

Run from the project root (where the compiled `scheduler` module lives):
    python3 thesis/experiments/run_experiments.py --out thesis/experiments/results
"""
import argparse, json, os, sys, time, statistics as st

sys.path.insert(0, os.getcwd())
from scheduler import (SchedulerGP, FitnessEvaluator, SimulationSnapshot,
                       buildTerminalRegistry, ExprNode, NodeType)
from data_processing.data_factory import create_jobs_from_csv, create_satellites_from_csv
from data_processing.csv_preprocessing import load_world_paths

N_TRAIN, ROW_LIMIT, TOP_K = 20, 100, 20          # worlds 0..19 train, 20.. test
# Worlds to train and test on.  The default is the real simulator log.  Set the
# environment variable GA_WORLDS_DIR (or pass --worlds-dir, which sets it) to
# use another set, e.g. data/realistic/worlds from data_processing/realistic_trace.
# An environment variable, so that the worker processes of run_ablation.py see it.
WORLDS_DIR_ENV, DEFAULT_WORLDS_DIR = "GA_WORLDS_DIR", "data/worlds"
REG = buildTerminalRegistry()
TERM = {t.name: i for i, t in enumerate(REG)}
OPS = {NodeType.ADD: "+", NodeType.SUB: "-", NodeType.MUL: "*", NodeType.DIV: "/?",
       NodeType.MAX_OP: "MAX", NodeType.MIN_OP: "MIN"}
UN = {NodeType.NEG: "-", NodeType.ABS: "ABS", NodeType.INV: "INV"}


def tostr(n):
    if n.nodeType == NodeType.TERMINAL: return REG[n.terminalIndex].name
    if n.nodeType == NodeType.CONST: return "c"
    ch = [tostr(c) for c in n.children]
    if n.nodeType in OPS:
        o = OPS[n.nodeType]
        return f"({ch[0]} {o} {ch[1]})" if o in "+-*" or o == "/?" else f"{o}({ch[0]}, {ch[1]})"
    return f"{UN[n.nodeType]}({ch[0]})"


def term(name):
    n = ExprNode(); n.nodeType = NodeType.TERMINAL; n.terminalIndex = TERM[name]; return n


def op(t, *kids):
    n = ExprNode(); n.nodeType = t; n.children = list(kids); return n


BASELINES = {
    "FCFS (first-fit)":      op(NodeType.NEG, term("job.arrivalTime")),
    "Shortest job first":    op(NodeType.NEG, op(NodeType.ADD, term("job.executionTime"), term("job.transferTime"))),
    "Least-loaded satellite": op(NodeType.NEG, term("sat.completedTasks")),
    "Most energy left":      term("sat.remainingEnergy"),
}


def load_worlds():
    worlds_dir = os.environ.get(WORLDS_DIR_ENV, DEFAULT_WORLDS_DIR)
    paths = [str(p) for p in load_world_paths(worlds_dir)]
    if len(paths) <= N_TRAIN:
        raise SystemExit(f"{worlds_dir}: {len(paths)} world(s) found, but more than {N_TRAIN} are "
                         "needed (the first 20 train, the rest test).")
    snaps = []
    for i, p in enumerate(paths):
        seed = 1000 + i                      # same world for every method
        jobs = create_jobs_from_csv(p, limit=ROW_LIMIT, random_sample=False, sample_seed=seed)
        minute = float(max((j.arrivalTime for j in jobs), default=0))
        sats = create_satellites_from_csv(p, limit=ROW_LIMIT, seed=seed, random_sample=False,
                                          sample_seed=seed, snapshot_minute=minute)
        snaps.append(SimulationSnapshot(minute, sats, jobs))
    return snaps[:N_TRAIN], snaps[N_TRAIN:]


def mean_metrics(fe, ind, snaps):
    r = [fe.evaluate_single(ind, s) for s in snaps]
    return [st.fmean(x[k] for x in r) for k in range(3)]       # C, R, E


def train(seed, pop, depth, gens, hoist, w, switch, train_s, test_s, ckpt=50):
    ga = SchedulerGP([], [], seed=seed)
    ga.initialize(populationSize=pop, maxDepth=depth, hoistRate=hoist)
    fe = FitnessEvaluator(REG)
    fe.weightJobs, fe.weightTime, fe.weightEnergy = w
    P = ga.getPopulation()
    rnd = [mean_metrics(fe, P[i], test_s) for i in range(len(P))]       # untrained trees
    rand_mean = [st.fmean(x[k] for x in rnd) for k in range(3)]
    curve = []
    for g in range(gens):
        snap = train_s[(g // switch) % len(train_s)]
        fe.evaluate(P, snap)
        if g % ckpt == 0 or g == gens - 1:
            b = max(range(len(P)), key=lambda i: P[i].fitness)
            curve.append([g + 1] + mean_metrics(fe, P[b], test_s))
        if g < gens - 1: ga.solveNextGeneration()
    top = sorted(range(len(P)), key=lambda i: -P[i].fitness)[:TOP_K]
    sc = []
    for i in top:
        m = mean_metrics(fe, P[i], train_s)
        sc.append((-m[0], m[1], i))              # max completion, then min response
    i = min(sc)[2]
    return dict(test=mean_metrics(fe, P[i], test_s), train=mean_metrics(fe, P[i], train_s),
                random=rand_mean, curve=curve, tree=tostr(P[i].tree), size=P[i].tree.size())


DEF = dict(pop=100, depth=4, gens=300, hoist=0.05, w=(0.60, 0.25, 0.15), switch=10)
SWEEPS = {
    "depth":  ("depth",  [2, 3, 4, 5, 6, 7]),
    "pop":    ("pop",    [20, 50, 100, 200, 400]),
    "gens":   ("gens",   [50, 100, 300, 1000]),
    "hoist":  ("hoist",  [0.0, 0.05, 0.2]),
    "switch": ("switch", [1, 10, 50, 10**9]),
    "weights": ("w", [(0.60, 0.25, 0.15), (1.0, 0.0, 0.0), (0.34, 0.33, 0.33), (0.2, 0.4, 0.4)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="thesis/experiments/results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--worlds-dir", default=None,
                    help="Directory of pre-split worlds (default data/worlds, or $GA_WORLDS_DIR).")
    ap.add_argument("--baselines-only", action="store_true",
                    help="Print the hand-written baselines on the test worlds and stop. They are "
                         "deterministic, take a second and write no file: a quick way to compare "
                         "two sets of worlds.")
    a = ap.parse_args()
    if a.worlds_dir:
        os.environ[WORLDS_DIR_ENV] = a.worlds_dir
    if not a.baselines_only:
        os.makedirs(a.out, exist_ok=True)
    train_s, test_s = load_worlds()
    print(f"{len(train_s)} train worlds, {len(test_s)} test worlds", flush=True)

    # baselines (deterministic) on test worlds
    ga = SchedulerGP([], [], seed=0); ga.initialize(populationSize=2, maxDepth=3, hoistRate=0.0)
    fe = FitnessEvaluator(REG); P = ga.getPopulation()
    base = {}
    for name, tree in BASELINES.items():
        P[0].tree = tree
        base[name] = mean_metrics(fe, P[0], test_s)
    if a.baselines_only:
        print(f"{'baseline':26s} {'C':>7s} {'R':>8s} {'E':>8s}")
        for name, (c, r, e) in base.items():
            print(f"{name:26s} {c:7.3f} {r:8.2f} {e:8.1f}")
        return
    res = dict(baselines=base, defaults={k: (list(v) if isinstance(v, tuple) else v) for k, v in DEF.items()},
               n_train=len(train_s), n_test=len(test_s),
               worlds_dir=os.environ.get(WORLDS_DIR_ENV, DEFAULT_WORLDS_DIR), runs=[])

    cache = {}
    def run(cfg, seed):
        key = (json.dumps(cfg, sort_keys=True), seed)
        if key not in cache:
            t = time.time()
            c = dict(cfg)
            cache[key] = train(seed, c["pop"], c["depth"], c["gens"], c["hoist"], tuple(c["w"]),
                               c["switch"], train_s, test_s, ckpt=50 if c["gens"] >= 300 else 10)
            cache[key]["seconds"] = time.time() - t
        return cache[key]

    jobs = [("main", dict(DEF, gens=1000), None)]
    jobs += [("default300", dict(DEF), None)]
    for sw, (k, vals) in SWEEPS.items():
        for v in vals:
            jobs.append((f"sweep:{sw}", dict(DEF, **{k: v}), v))
    for tag, cfg, v in jobs:
        for s in range(a.seeds):
            r = run(cfg, s)
            res["runs"].append(dict(tag=tag, value=(list(v) if isinstance(v, tuple) else v),
                                    cfg={k: (list(x) if isinstance(x, tuple) else x) for k, x in cfg.items()},
                                    seed=s, **r))
        print(tag, v, "done", flush=True)
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"))


if __name__ == "__main__":
    main()
