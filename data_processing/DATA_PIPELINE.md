# Data Pipeline: Relative Arrival Normalization & Pre-Split Worlds

This document explains, in full detail, the data-loading changes that introduce
(1) **relative arrival-time normalization** and (2) **pre-processed per-window
"world" CSV files** for the genetic-programming scheduler.

It covers the motivation, the algorithm, every changed function, the C++
evaluator semantics that constrained the design, how to run it, how it was
verified, and the known trade-offs.

---

## 1. Motivation

The raw dataset (`data/example.csv`) is the task log of a **single 300-minute
LEO simulation** — 2 960 rows, one per completed task, ~92 distinct Starlink
servers, with an `Arrival Time (System)` column spanning `0.0018 → 299.75`.

Two properties of that data were being handled incorrectly:

### Problem 1 — Arrival times were thrown away

The scheduler does not care about the *absolute* minute a task arrived; it only
cares about how arrivals **evolve relative to each other** (their ordering and
the gaps between them). The old loader did the opposite of preserving that:

```python
# OLD data_factory.create_jobs_from_csv
_IGNORED_CSV_COLUMNS = { ..., "Arrival Time (System)", "Arrival Time (Queue)", ... }
...
job.arrivalTime = int(snapshot_minute)   # every job gets the SAME arrival time
```

The arrival columns were explicitly ignored, and **every job in a world was
assigned one identical arrival time** (the snapshot minute). The relative
arrival pattern was completely discarded, and `job.arrivalTime` — which the GP
can use as a feature and which drives the response-time objective — was a
constant.

### Problem 2 — "World resets" re-read the same data

When the GA stops improving it regenerates the world. The old code re-read the
*same* `example.csv` with the same row limit and `random_sample=False`:

```python
# OLD main.build_world — same file, same first N rows, every reset
satellites = create_satellites_from_csv(csv_path, limit=job_limit, ...)
jobs       = create_jobs_from_csv(csv_path, limit=job_limit, ...)
```

So every "new world" was the same task mix; only the snapshot minute and a
satellite RNG seed changed. There was no genuine variety across resets.

### The fix, in one sentence

Pre-split the raw log into fixed-length **time-window CSVs**, normalize each
world's arrivals **relative to its own earliest task**, and have the GA **cycle
through a different world** on every reset.

---

## 2. Architecture & data flow

```
            data/example.csv  (raw 300-min simulation log)
                       │
                       ▼
   data_processing/csv_preprocessing/world_splitter.py     ── PRE-PROCESS (run once)
   • bucket rows into [0,10), [10,20), … minute windows
   • per window: shift earliest arrival → 0, keep gaps
   • append "Arrival Time (Normalized)" column
                       │
                       ▼
            data/worlds/                                    ── GENERATED ARTIFACTS (gitignored)
              ├── world_000.csv   (window 0,  100 tasks)
              ├── world_001.csv   …
              ├── world_029.csv   (window 29, 117 tasks)
              └── manifest.json   (ordered index + metadata)
                       │
                       ▼
   main.run_genetic_algorithm                              ── RUN
   • load_world_paths("data/worlds")  → ordered list
   • build_world(world_path):
       – create_jobs_from_csv  → relative (0-based) arrivals
       – currentMinute = max(arrival)  (so all jobs are "arrived")
       – create_satellites_from_csv
   • on each convergence reset → select the NEXT world
                       │
                       ▼
   scheduler (C++)  greedyDecode / fitness evaluation
```

Three code locations change: the **new preprocessing module**, the
**`data_factory`** loader, and the **`main`** GA driver. The C++ core is
**unchanged**.

---

## 3. Component 1 — The world splitter

**File:** `data_processing/csv_preprocessing/world_splitter.py` (new)
**Package:** `data_processing/csv_preprocessing/__init__.py` re-exports the public
names so callers can `from data_processing.csv_preprocessing import …`.

### 3.1 Public constants

| Constant | Value | Purpose |
|---|---|---|
| `SYSTEM_ARRIVAL_COLUMN` | `"Arrival Time (System)"` | Raw absolute arrival column used to bucket rows into windows. |
| `NORMALIZED_ARRIVAL_COLUMN` | `"Arrival Time (Normalized)"` | New column written into each world file (per-world relative arrival). |

Header keys are matched **after stripping surrounding whitespace**, because
`utils._load_csv_rows` strips them (the raw header is `" Arrival Time (System)  "`).

### 3.2 Data model

Two dataclasses describe the output:

```python
@dataclass
class WorldEntry:        # one generated world file
    file: str            # "world_000.csv"
    window_index: int    # which absolute time window (0 = [0,10), 1 = [10,20), …)
    tasks: int           # rows in this world
    arrival_min: float   # earliest absolute arrival in the window
    arrival_max: float   # latest absolute arrival in the window
    span: float          # arrival_max - arrival_min

@dataclass
class WorldManifest:     # the whole run
    source_csv, window_minutes, arrival_column, normalized_column,
    generated_at, num_worlds, total_tasks, skipped_rows,
    worlds: List[WorldEntry]
```

### 3.3 `split_into_worlds(...)` — the core algorithm

```python
split_into_worlds(
    csv_path="data/example.csv",
    output_dir="data/worlds",
    window_minutes=10.0,
    arrival_column=SYSTEM_ARRIVAL_COLUMN,
    normalized_column=NORMALIZED_ARRIVAL_COLUMN,
    min_tasks_per_world=1,
    clean=True,
) -> WorldManifest
```

Step by step:

1. **Validate** `window_minutes > 0` (raises `ValueError` otherwise).
2. **Load all rows** via `utils._load_csv_rows(source, limit=None)` — this strips
   header whitespace and reuses the project's existing CSV reader, so locale
   quirks are handled identically to the rest of the codebase.
3. **Capture column order** from the first row's keys and append the normalized
   column, so the output preserves every original column plus the new one.
4. **Bucket** each row by window:
   ```python
   arrival = _parse_csv_number(row[arrival_column])
   window_index = int(arrival // window_minutes)     # floor division
   ```
   Rows whose arrival cannot be parsed are **skipped** and counted
   (`skipped_rows`). Note this groups **across the whole file** — the source rows
   are *not* pre-sorted by arrival, so a window collects every task whose arrival
   falls in its `[k·w, (k+1)·w)` interval regardless of original position.
5. **Clean** the output directory if `clean=True` — only `world_*.csv` and
   `manifest.json` are removed (nothing else is ever deleted).
6. **For each non-empty window** (in ascending `window_index` order, dropping any
   with fewer than `min_tasks_per_world` rows):
   - sort its rows by arrival (chronological within the world),
   - compute `window_min = earliest arrival`,
   - write `world_NNN.csv` (sequential index by output order) with
     `csv.DictWriter`, appending
     `normalized = repr(arrival - window_min)` to every row,
   - record a `WorldEntry`.
7. **Write `manifest.json`** and return the `WorldManifest`.

Files are named `world_000.csv`, `world_001.csv`, … by **chronological output
order**; the original (possibly non-contiguous) `window_index` is preserved in
the manifest. Full float precision is kept on disk via `repr()`, which
round-trips exactly through `_parse_csv_number`.

### 3.4 `load_world_paths(worlds_dir="data/worlds") -> List[Path]`

Used by `main.py` to discover worlds:

- If `manifest.json` exists, it returns the world files **in manifest order**
  (the authoritative order).
- If the manifest is missing or malformed, it **falls back** to a sorted glob of
  `world_*.csv`.
- If the directory does not exist, it returns `[]` (the GA then falls back to the
  raw CSV — see §5).

### 3.5 CLI

The module is runnable as a script from the repo root:

```bash
python -m data_processing.csv_preprocessing.world_splitter \
    --csv data/example.csv \
    --output-dir data/worlds \
    --window-minutes 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `data/example.csv` | Source CSV. |
| `--output-dir` | `data/worlds` | Destination for world files + manifest. |
| `--window-minutes` | `10.0` | Window length (same unit as the arrival column). |
| `--arrival-column` | `Arrival Time (System)` | Column to bucket on. |
| `--normalized-column` | `Arrival Time (Normalized)` | Name of the appended column. |
| `--min-tasks-per-world` | `1` | Drop windows with fewer tasks. |
| `--no-clean` | off | Keep pre-existing world files instead of clearing. |

Relative paths resolve against the **repository root** (computed as
`Path(__file__).parents[2]`), which is also added to `sys.path` so `import utils`
works whether the file is run as a module or directly.

---

## 4. The normalization scheme — "shift to zero, keep gaps"

Within each world, the earliest task is moved to `t = 0` and **every
inter-arrival gap is preserved exactly**:

```
normalized[i] = arrival[i] − min(arrival in this world)
```

Worked example (`world_000.csv`, first five tasks):

| `Arrival Time (System)` | `Arrival Time (Normalized)` | `job.arrivalTime` (int) |
|---:|---:|---:|
| 0.00180 | 0.00000 | 0 |
| 0.07494 | 0.07314 | 0 |
| 0.45261 | 0.45081 | 0 |
| 0.49470 | 0.49290 | 0 |
| 0.90665 | 0.90485 | 1 |

Properties:

- **Absolute offset is removed** (the world no longer "knows" it started at
  minute 92 vs minute 0) — this is exactly the information the user identified as
  irrelevant.
- **Relative structure is exact:** `normalized[i] − normalized[i−1] ==
  system[i] − system[i−1]`. The verification measured a maximum gap
  reconstruction error of `~9 × 10⁻¹⁶` (floating-point epsilon).
- This was chosen over min-max `[0,1]` scaling or scale-to-fixed-span precisely
  because those distort the magnitude of the gaps; "shift to zero" keeps them.

The on-disk value keeps full float precision. The rounding to whole minutes in
the last column happens only when populating the C++ `Job` (see §7).

---

## 5. Component 2 — `data_factory.py`

**File:** `data_processing/data_factory.py` (modified)

### 5.1 New constants

```python
_SYSTEM_ARRIVAL_COLUMN = "Arrival Time (System)"
_NORMALIZED_ARRIVAL_COLUMN = "Arrival Time (Normalized)"
```

They mirror the splitter's names locally so `data_factory` has **no import
dependency** on the preprocessing package.

### 5.2 Reading the arrival columns

The arrival columns are listed in `_IGNORED_CSV_COLUMNS`, so they are removed
from the per-row `filtered` dict. They are therefore read from the **full row**
instead:

```python
task_rows.append({
    ...,
    "arrival_norm_raw": _parse_csv_number(row.get(_NORMALIZED_ARRIVAL_COLUMN)),
    "arrival_sys_raw":  _parse_csv_number(row.get(_SYSTEM_ARRIVAL_COLUMN)),
})
```

### 5.3 Computing the relative arrival

```python
_has_normalized = any(r["arrival_norm_raw"] is not None for r in task_rows)
_system_values  = [r["arrival_sys_raw"] for r in task_rows if r["arrival_sys_raw"] is not None]
_system_min     = min(_system_values) if _system_values else 0.0

def relative_arrival(task_row) -> float:
    if task_row["arrival_norm_raw"] is not None:     # pre-split world file
        return task_row["arrival_norm_raw"]
    if _has_normalized:                              # world file, row missing value
        return 0.0
    if task_row["arrival_sys_raw"] is not None:      # raw CSV → normalize on the fly
        return task_row["arrival_sys_raw"] - _system_min
    return 0.0                                       # no arrival info → flat
```

This makes the loader **robust to both inputs**:

- a **pre-split world** (has the `Normalized` column) → use it directly;
- a **raw CSV** (e.g. `example.csv`, no `Normalized` column) → shift the
  `System` arrivals to zero on the fly, so relative arrivals still work even
  without preprocessing.

### 5.4 Assigning `job.arrivalTime`

```python
# OLD: job.arrivalTime = int(snapshot_minute)        # constant for all jobs
# NEW:
job.arrivalTime = int(round(snapshot_minute + relative_arrival(row)))
```

`snapshot_minute` is now only an **optional base offset** (default `0.0`); with
the default, the earliest task of a world arrives at minute 0 and the rest keep
their real gaps. The value is rounded to an `int` because that is the type of
the C++ field (§7).

---

## 6. Component 3 — `main.py`

**File:** `main.py` (modified)

### 6.1 World discovery & selection

```python
from data_processing.csv_preprocessing import load_world_paths
...
world_paths = [str(p) for p in load_world_paths("data/worlds")]
if world_paths:
    print(f"Loaded {len(world_paths)} pre-processed world(s) from data/worlds/.")
else:
    world_paths = ["data/example.csv"]      # graceful fallback
    print("No pre-processed worlds … falling back to data/example.csv. "
          "Run `python -m data_processing.csv_preprocessing.world_splitter` …")

def select_world_path(world_index: int) -> str:
    return world_paths[world_index % len(world_paths)]   # cycles, wraps around
```

If no worlds have been generated yet, the GA still runs against the raw CSV (and
still benefits from on-the-fly arrival normalization in §5.3).

### 6.2 `build_world` rewrite

```python
def build_world(world_path: str, job_limit: int):
    world_seed = world_rng.randrange(0, 2_147_483_647)
    # 0/negative limit ⇒ load the whole world (honours --csv-row-limit help text)
    effective_limit = job_limit if (job_limit and job_limit > 0) else None

    # Jobs first: arrivals are relative (earliest = 0), so "now" is derived from them.
    jobs_local = create_jobs_from_csv(world_path, limit=effective_limit,
                                      random_sample=False, sample_seed=world_seed)

    # Evaluate at the END of the arrival window so every task has already arrived.
    minute = float(max((job.arrivalTime for job in jobs_local), default=0))

    satellites_local = create_satellites_from_csv(world_path, limit=effective_limit,
                                                  seed=world_seed, random_sample=False,
                                                  sample_seed=world_seed,
                                                  snapshot_minute=minute)
    return satellites_local, jobs_local, minute, world_seed
```

Key changes vs. the old version:

- takes a **`world_path`** instead of always using `example.csv`;
- builds **jobs first**, then derives the snapshot minute from them;
- returns a **4-tuple** `(satellites, jobs, minute, seed)` — the minute is now an
  output, not an input;
- a non-positive `--csv-row-limit` now means "load the whole world" (previously
  it silently produced an empty world due to a quirk in `_load_csv_rows`).

### 6.3 Initial world and resets

```python
# initial world (index 0)
satellites, jobs, current_minute, _ = build_world(select_world_path(0), initial_job_limit)
...
# on convergence reset: advance to the NEXT world
next_world_path = select_world_path(total_resets + 1)
satellites, jobs, current_minute, world_seed = build_world(next_world_path, reset_job_limit)
snapshot = SimulationSnapshot(current_minute, satellites, jobs)
```

The old `current_minute = generation * 5.0` line is gone; the snapshot minute now
comes from the world's data. World selection cycles `0, 1, 2, …` and wraps, so
across a long run the population is exposed to every world in turn.

---

## 7. Why `currentMinute = max(arrival)` — the evaluator semantics

This is the subtle part, and it is dictated by the C++ evaluator
(`include/scheduler/fitness_evaluator.h`, function `greedyDecode`):

1. **Availability gate.** Jobs that have not arrived yet are excluded from
   scheduling but **still counted** in the completion-ratio denominator:
   ```cpp
   if (static_cast<double>(jobs[j].arrivalTime) > snap.currentMinute) {
       scheduled[j] = true; --remaining;   // treated as unavailable
   }
   ```
2. **Satellites start free at "now".** `satStates[s].freeAtMinute = snap.currentMinute;`
3. **Response time uses the individual arrival:**
   ```cpp
   arrival = static_cast<double>(jobs[a.jobIndex].arrivalTime);
   totalResponse += a.completionMinute - arrival;     // (computeRawObjectives)
   ```

Consequences for our design:

- If we spread arrivals but left `currentMinute` at the window **start**, most
  jobs would satisfy `arrivalTime > currentMinute` and be **silently dropped**,
  tanking the completion ratio. That is why `build_world` sets
  **`currentMinute = max(arrival)`** — every task in the world has "already
  arrived", so all of them are scheduled and counted.
- Because response time is `completion − arrival`, an **early-arriving** task
  (small `arrivalTime`) that completes late is penalised more than a late
  arriver completing at the same time. The relative arrival pattern therefore
  **directly shapes the fitness landscape** — which is exactly the behaviour the
  normalization is meant to enable. Previously, with all arrivals equal, this
  term was a constant offset and carried no signal.

The invariant `snapshot.currentMinute == max(job.arrivalTime)` is asserted in the
verification (§9).

---

## 8. The `int arrivalTime` trade-off

`Job::arrivalTime` is declared `int` in `include/scheduler/types.h`. Every C++
consumer already reads it through `static_cast<double>` (the gate, the registry
terminal `job.arrivalTime`, and the response-time sum), so the field could be
widened to `double` with no other code change — but that is a C++ API/ABI change
that requires **rebuilding the extension**, and `AGENT.md` asks for the C++
surface to stay stable.

Decision: **keep `int`.** Implications:

- Sub-minute spacing is rounded to whole minutes when building the `Job`
  (`int(round(...))`). With ~10 tasks/minute in this dataset, roughly ten tasks
  share each integer-minute bucket, so fine-grained ordering within a minute is
  lost. Broad relative evolution across the window is preserved.
- **No information is lost on disk** — the world CSVs store full-precision
  normalized arrivals, so switching to `double` later requires no re-split.

**Optional upgrade to full resolution:** change `int arrivalTime;` to
`double arrivalTime;` in `types.h`, drop the `int(round(...))` to keep the float
in `data_factory`, and rebuild:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build && cp build/scheduler*.so .
```

---

## 9. Verification

The C++ `scheduler` extension is built for Python 3.13 and could not be imported
in the 3.10 test sandbox, so the C++ core was **not** exercised here — please run
`main.py` inside the Docker container to confirm end-to-end. The Python logic was
verified as follows:

| Check | Result |
|---|---|
| `py_compile` of all four changed/added files | pass |
| Splitter on `example.csv` (10-min windows) | 30 worlds, 2 959 tasks, 0 skipped |
| Per-world earliest normalized arrival == 0 | pass (max deviation `0`) |
| Gap reconstruction error vs. source | `~9 × 10⁻¹⁶` (exact) |
| All 27 original columns preserved + 1 normalized | 28 columns |
| `create_jobs_from_csv` on a world file → `min=0`, `== round(normalized)` | pass |
| `create_jobs_from_csv` on raw `example.csv` → on-the-fly shift to zero | pass |
| `load_world_paths` → 30 paths in manifest order | pass |
| Full `run_genetic_algorithm` loop (mocked C++ scheduler) | pass |
| `snapshot.currentMinute == max(arrivalTime)` every snapshot | pass |
| Worlds advance to distinct files on each reset | pass (`world_001`, `002`, `003`, …) |
| Worlds load fully with `--csv-row-limit 0` | 100 / 109 / 91 / 87 jobs |

The full GA loop was driven with a stub `scheduler` module (simple `Job` /
`Satellite` attribute holders and no-op `SchedulerGP` / `FitnessEvaluator`),
forcing a reset every generation to confirm world cycling and the snapshot-minute
invariant.

---

## 10. How to use

```bash
# 1. Generate the worlds (run once, or whenever the source data changes)
python -m data_processing.csv_preprocessing.world_splitter --window-minutes 10

# 2. Run the GA — it auto-discovers data/worlds/ and cycles worlds on reset
python3 main.py --no-start-prompt
```

Tuning the window length changes the number and size of worlds (the data is
~10 tasks/minute over 300 minutes):

| `--window-minutes` | Worlds | ~Tasks/world |
|---:|---:|---:|
| 30 | ~10 | ~296 |
| 10 *(default)* | ~30 | ~100 |
| 6.67 | ~45 | ~66 |

---

## 11. File-by-file summary

| File | Status | Change |
|---|---|---|
| `data_processing/csv_preprocessing/world_splitter.py` | **new** | Splitter + normalizer + manifest + `load_world_paths` + CLI. |
| `data_processing/csv_preprocessing/__init__.py` | **new** | Re-exports the public API. |
| `data_processing/data_factory.py` | modified | Reads normalized/system arrival; relative-arrival logic; `job.arrivalTime` now relative, not flat. |
| `main.py` | modified | World discovery + cycling; `build_world` derives the snapshot minute from arrivals; full-world load on non-positive limit. |
| `data/worlds/` | generated | 30 world CSVs + `manifest.json` (git-ignored under `/data`). |
| `include/scheduler/*` | **unchanged** | No C++ changes. |

---

## 12. Manifest schema (reference)

`data/worlds/manifest.json`:

```json
{
  "source_csv": "data/example.csv",
  "window_minutes": 10.0,
  "arrival_column": "Arrival Time (System)",
  "normalized_column": "Arrival Time (Normalized)",
  "generated_at": "2026-06-16T15:41:28",
  "num_worlds": 30,
  "total_tasks": 2959,
  "skipped_rows": 0,
  "worlds": [
    {
      "file": "world_000.csv",
      "window_index": 0,
      "tasks": 100,
      "arrival_min": 0.0018007839974683562,
      "arrival_max": 9.739209058762718,
      "span": 9.737408274765249
    }
    /* … 29 more … */
  ]
}
```

---

## 13. Limitations & possible follow-ups

- **Integer arrival resolution** (§8) — whole-minute rounding; switch to `double`
  for sub-minute precision (needs a rebuild).
- **Sequential world cycling** — resets walk worlds `0,1,2,…`. Random selection
  is a one-line change if more stochastic exposure is preferred.
- **Fixed-length windows** — windows are uniform in time, so task counts vary
  (80–117 here). If equal-size worlds are needed, a count-based splitter could be
  added alongside the time-based one.
- **C++ not exercised in-sandbox** — the Python logic is covered by tests, but
  the integrated GA should be run in the Docker container to confirm fitness
  behaviour against the real scheduler.
