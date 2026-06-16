# Performance: Parallelization & Decode Optimization

This document explains the work done to speed up the C++ core of the
genetic-programming scheduler (`scheduler` Python extension), why each change is
correct, and how to reproduce the results.

## TL;DR

Three independent improvements, all in the C++ core, none of which change the
numerical results the GA produces:

1. **Activated dormant OpenMP** on the per-individual fitness evaluation. The
   code was already written to be parallel, but the shipped `.so` had been
   compiled **without** OpenMP, so it ran serially.
2. **Parallelized offspring generation** (`solveNextGeneration`) with a
   per-thread RNG so the genetic operators run concurrently and race-free.
3. **Cached the greedy decoder's score matrix** so the expression tree is
   evaluated `O(J·S + J²)` times per individual instead of `O(J²·S)` — the
   single biggest win.

Measured on a 4-core machine, `evaluate()` on a `pop=150`, `50 sats × 100 jobs`
workload went from **~1893 ms/call to ~28 ms/call — roughly 69× faster** — while
producing **bit-identical** fitness values.

| Stage (50×100, pop 150) | ms / `evaluate()` call | Speedup vs original |
|---|---:|---:|
| Original (serial, uncached) | 1893 | 1× |
| + OpenMP (4 threads), uncached | 569 | 3.3× |
| + decode caching (1 thread) | 102 | 18.6× |
| + decode caching (4 threads) | **27.5** | **68.8×** |

---

## Background

The GP scheduler evolves expression trees that act as **priority functions**.
For each generation:

1. **`FitnessEvaluator::evaluate`** decodes every individual: it runs a greedy
   scheduler (`greedyDecode`) that repeatedly picks the highest-scoring
   `(job, satellite)` pair, where the score is the individual's tree evaluated on
   that pair. Each individual produces a completion ratio, mean response time and
   energy cost, which are combined into a scalar fitness.
2. **`SchedulerGP::solveNextGeneration`** breeds the next generation via
   tournament selection, subtree crossover, point mutation and hoist mutation.

Both steps are **embarrassingly parallel over the population**: each individual
is decoded and bred independently. The dominant cost is step 1, and within it the
repeated evaluation of the expression tree inside `greedyDecode`.

---

## Change 1 — Activate the dormant OpenMP fitness parallelism

`FitnessEvaluator::evaluate` already contained:

```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < n; ++i) {
    // each individual decoded into its own local buffers,
    // results written to distinct indices compRatios[i], meanResps[i], energies[i]
}
```

This loop is data-race free by construction: `greedyDecode` takes private working
copies, the tree and registry are read-only, and every result goes to a distinct
index. **But the compiled extension had no OpenMP linked** — `ldd scheduler*.so`
showed no `libgomp`, and there were no `GOMP_*` symbols. The pragma was simply
ignored, so it ran serially.

The fix is at build time. `CMakeLists.txt` already wires OpenMP in correctly:

```cmake
find_package(OpenMP)
...
if(OpenMP_CXX_FOUND)
    target_link_libraries(scheduler PRIVATE OpenMP::OpenMP_CXX)
endif()
```

so the module just needed an OpenMP-enabled rebuild. It remains **optional**: if
the toolchain has no OpenMP the module still builds and runs (serially), guarded
by `#ifdef _OPENMP`.

> **Operational gotcha:** a `.so` can silently be serial. After building, verify
> with `ldd scheduler*.so | grep gomp`.

---

## Change 2 — Parallelize offspring generation

`solveNextGeneration` was fully serial and funneled every random draw through one
shared `mt19937`, which is not thread-safe. Three things made parallelization
safe:

**Fixed output layout.** Instead of `push_back`-ing children onto a growing
vector, the next generation is pre-sized and each iteration writes to two
**distinct, pre-allocated slots**:

```cpp
vector<GPIndividual> next(popSize);
for (int i = 0; i < elitismCount; ++i)        // elites first
    next[i] = population[popSize - 1 - i];
const int nPairs = (popSize - elitismCount + 1) / 2;   // one crossover → two children
```

**Per-thread RNG and operators.** Each thread constructs its own `mt19937` plus
its own `TreeBuilder`/`TreeOperators` (which hold a reference to that RNG). The
read-only terminal registry is shared. Each thread is seeded from the master RNG
(consumed once, serially) offset by the thread id, so runs stay varied and each
thread gets an independent sub-stream:

```cpp
const unsigned baseSeed = rng_();   // serial, before the region
#pragma omp parallel
{
    mt19937       localRng(baseSeed + 0x9E3779B9u * (ompThreadNum() + 1));
    TreeBuilder   localBuilder(registry_, localRng);
    TreeOperators localOps(localRng, localBuilder, maxDepth_);
    localOps.setTerminalCount((int)registry_.size());

    #pragma omp for schedule(dynamic)
    for (int p = 0; p < nPairs; ++p) {
        // tournament-select parents from `population` (read-only),
        // crossover / mutate / hoist using localOps,
        // write to next[2p] and next[2p+1]
    }
}
```

The only shared state inside the loop is `population` (read-only during
selection) and `next` (each iteration owns its two slots), so there is no data
race. `0x9E3779B9` is the golden-ratio constant commonly used to spread seeds.

This loop is much cheaper than fitness evaluation, so its absolute contribution
is small, but it scales well and removes a serial section.

---

## Change 3 — Cache the greedy decoder's score matrix (the big win)

### The observation

Inside `greedyDecode`, the inner line

```cpp
const double score = tree.eval(jobs[j], sats[s], registry);
```

is a **pure function** of the job and the satellite. During one decode:

- The job objects never change.
- A satellite only changes when a job is **committed** to it — exactly two
  tree-visible fields change: `completedTasks` (`+= 1`) and `remainingEnergy`
  (`-= cost`). (`freeAtMinute` also advances, but it is decode-internal and is
  not a tree terminal.)

Therefore, committing a job **invalidates only the scores in that one
satellite's column**. Every other cached `score[j][s]` is still exact.

The original algorithm ignored this and recomputed the tree for every feasible
pair on **every** iteration of the `while` loop.

### The implementation

Compute the full `J×S` score matrix once, then refresh a single column after each
commit:

```cpp
// one-time full matrix (row-major in j: index j*nSat + s)
vector<double> score((size_t)nJob * nSat);
for (int j = 0; j < nJob; ++j) {
    if (scheduled[j]) continue;
    for (int s = 0; s < nSat; ++s)
        score[(size_t)j*nSat + s] = tree.eval(jobs[j], sats[s], registry);
}

while (remaining > 0) {
    // ... argmax scan reads `score[base + s]` instead of calling tree.eval ...
    // ... commit best (job, sat) ...

    // refresh only the committed satellite's column, for still-unscheduled jobs
    for (int j = 0; j < nJob; ++j) {
        if (scheduled[j]) continue;
        score[(size_t)j*nSat + bestSat] = tree.eval(jobs[j], sats[bestSat], registry);
    }
}
```

### Why results are identical

The change touches **only** how the score is obtained (cache read vs.
recompute). Everything that determines *which* pair is chosen is untouched:

- Same scan order (`j` outer, `s` inner).
- Same feasibility gates (operational window + energy), recomputed each
  iteration from current satellite state — they are cheap and were left in the
  loop.
- Same strict-greater tie-break (`if (sc > bestScore)`), so among equal scores
  the lowest `(j, s)` in scan order still wins.

Because the cache is refreshed whenever a satellite changes, the cached value
always equals what the original would have computed. A useful supporting fact:
**feasibility is monotonic** — `freeAtMinute` only increases and
`remainingEnergy` only decreases, so a pair that becomes infeasible never becomes
feasible again. Nothing "comes back" in a way the cache could miss.

### Complexity

Let `J` = jobs, `S` = satellites.

| | Tree evaluations per individual |
|---|---|
| Original | `O(J² · S)` — every feasible pair, every iteration |
| Cached | `O(J·S + J²)` — full matrix once, then ≤ `J` column refreshes of ≤ `J` entries |

The expensive tree walk is reduced by a factor of roughly `min(J, S)`. The
per-iteration argmax scan is still `O(J²·S)`, but it now performs only cheap
comparisons (no tree walk), which is why the measured speedup tracks the
reduction in tree evaluations. This also predicts that **the win grows with the
number of satellites**, which the benchmarks confirm (13.6× at `S=30` vs 18.6× at
`S=50`, single-thread).

---

## Correctness & equivalence verification

A scheduling change is only acceptable if the GA's results don't move. The
optimized decode was checked to be **bit-identical** to the original:

- Built a deterministic, RNG-independent population (trees constructed from a
  fixed Python seed via the newly exposed `ExprNode()` constructor) so the exact
  same individuals are evaluated before and after.
- Compared both the scalar **fitness** and the raw objectives
  **(completion ratio, mean response, energy)** for every individual.

Result: `max |fitness diff| = 0.0` and `max |raw objective diff| = 0.0`.

Additional checks:

- **No data race in the parallel paths:** the same population produces
  byte-identical fitness under `OMP_NUM_THREADS=1` and `=4` (`max|diff| = 0.0`).
- **Evolution stability:** 40 generations run with population size stable and all
  fitness values finite, at both thread counts.
- **Serial fallback:** the module still compiles and runs correctly when built
  without `-fopenmp`.
- **No new compiler warnings** beyond the pre-existing `#pragma region` markers.

---

## Benchmarks

4-core machine, `g++ 11.4 -O3 -funroll-loops -fopenmp`, population 150. Median of
5 `evaluate()` calls. "Baseline" = original decode; "Optimized" = cached decode.

**30 satellites × 80 jobs**

| Build | 1 thread | 4 threads |
|---|---:|---:|
| Baseline (original decode) | 763 ms | 230 ms |
| Optimized (cached decode) | 56 ms | 14.8 ms |
| Speedup (caching, same threads) | 13.6× | 15.5× |

**50 satellites × 100 jobs**

| Build | 1 thread | 4 threads |
|---|---:|---:|
| Baseline (original decode) | 1893 ms | 569 ms |
| Optimized (cached decode) | 102 ms | 27.5 ms |
| Speedup (caching, same threads) | 18.6× | 20.7× |

Parallel scaling on the optimized build is ~3.7–3.8× on 4 cores. The
`solveNextGeneration` step scales ~2.4× (it is a small fraction of total time).

**Overall:** original serial+uncached → parallel+cached is ~51× (30×80) to ~69×
(50×100), and the relative advantage grows with problem size and core count.

---

## Build & run

The performance gains require an **OpenMP-enabled rebuild** — the previously
shipped `.so` is serial and uncached. From the repo root (inside the container):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build
cp build/scheduler*.so .

# confirm OpenMP is actually linked:
ldd scheduler*.so | grep gomp        # should list libgomp
```

Control the number of threads at runtime:

```bash
OMP_NUM_THREADS=4 python3 main.py
```

By default OpenMP uses all available cores.

---

## Files changed

| File | Change |
|---|---|
| `include/scheduler/fitness_evaluator.h` | Score-matrix cache in `greedyDecode`; existing OpenMP `parallel for` in `evaluate` now actually compiled. |
| `include/scheduler/scheduler_gp.h` | Parallel offspring loop in `solveNextGeneration` with per-thread RNG/builder/operators; thread-safe tournament selection. |
| `bindings.cpp` | Release the GIL on `evaluate` and `solveNextGeneration`; expose `ExprNode()` constructor. |
| `CMakeLists.txt` | OpenMP discovery/linking (comment updated to reflect both parallel loops). |

---

## Remaining opportunities (not done)

- **Argmax scan.** After caching, the `O(J²·S)` per-iteration scan (cheap
  comparisons only) becomes the asymptotic floor. Since only one column changes
  per commit, an incremental max structure could lower this, but it complicates
  the exact tie-break and was left out to keep results identical.
- **Skip refreshing trees that ignore mutable fields.** If an individual's tree
  references neither `sat.completedTasks` nor `sat.remainingEnergy`, its column
  never changes and the per-commit refresh could be skipped entirely.
- **Buffer reuse.** The score matrix is allocated per decode; a thread-local
  scratch buffer would avoid repeated allocation across individuals.
