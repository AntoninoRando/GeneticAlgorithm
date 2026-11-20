# Project

This repository implements a genetic-programming-based scheduler: a C++ core library (`scheduler`) built as a Python extension, plus Python tooling that runs simulations, evaluates scheduling formulas, and iterates the genetic algorithm.

Purpose of this `AGENT.md`:
- Give an AI assistant clear, safe instructions for working on the project.
- Describe build/run commands, key files, project goals, and constraints.

Quick summary
- The C++ code builds into a Python extension named `scheduler` used by the Python scripts.
- Goal: evolve scheduling formulas that minimize average response time and energy while maximizing completed jobs.

What the scheduler does
- A scheduling formula scores (0..1) how suitable a satellite is for a job.
- For each simulation snapshot, formulas are used to select satellites for jobs; formulas are scored by a fitness combining completed jobs, average completion time, and energy.

How to work on this repo (recommended workflow for an AI)
- Inspect key files: [bindings.cpp](bindings.cpp#L1), [include/scheduler](include/scheduler), [main.py](main.py#L1), and [data_processing/data_factory.py](data_processing/data_factory.py#L1).
- Run the CMake build to compile the extension before running Python scripts.
- Prefer small, focused changes and run a local simulation to validate behavior.

Build & run commands
- Build the Python extension (from repo root). CMake writes `scheduler*.so` straight into the repo root, so there is nothing to copy:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build --target scheduler
```
  `RelWithDebInfo` (optimised, with symbols) is the build for every experiment and the default when no type is given. `Debug` is a real unoptimised build for gdb, many times slower: never benchmark or run experiments with it. `import scheduler` raises a `RuntimeWarning` for an unoptimised module, and `scheduler.build_type` / `scheduler.optimized` report which one is loaded. Always pass the build type explicitly: `build/` remembers the last one.
- The compiled module is **not** in the repository (`.gitignore`): it must be built on each machine and rebuilt after every change to `bindings.cpp` or `include/scheduler`.
- There is no CLI target; the only CMake target is `scheduler`.
- Short helper: the VS Code tasks `Build scheduler module` and `Run tests`, or `scripts/build.sh`. For a debugger build: the task `Build scheduler module (Debug, unoptimised)` or `scripts/build-debug.sh`, then rebuild the optimised module afterwards.
- Run a Python simulation example:
```
python3 main.py --no-start-prompt --generations 5
```
or `scripts/run.sh` (build, then run), or `./run <config>` for a named file in `runConfigurations/`.
- The input data (`data/`) is not in the repository either; without it `main.py` stops with a message and the data-dependent tests are skipped. See "Data" in `readme.md`.
- Without the original log, generate realistic data and train on it: `python3 -m data_processing.realistic_trace --download`, then `./run realistic --no-start-prompt` (or `python3 main.py --worlds-dir data/realistic/worlds`). It is generated data: label it as such and never mix its results with results on `data/worlds/`. See "Realistic data" in `readme.md`.

Key files & directories
- `bindings.cpp`: Python bindings / glue.
- `include/scheduler`: public headers and core types.
- `data_processing/data_factory.py`: dataset / snapshot helpers.
- `data_processing/realistic_trace/`: generator of a realistic task log (real Starlink orbits + a re-implementation of the original simulator) for working without `data/example.csv`. Writes `data/realistic/`; needs the `sgp4` package. Not imported by `main.py`.
- `main.py`, `run` (launcher script), and `scripts/`: runners and helper scripts.
- `tests/`: unit tests.
- `metrics/` and `snapshots/`: run outputs (not tracked).

Coding conventions & constraints for AI changes
- Keep C++ ABI/API stable: avoid renaming public functions in `include/scheduler` without updating bindings.
- Limit changes in `bindings.cpp` to small, well-tested additions.
- Follow existing code style and minimal impact edits.
- Run the build after C++ changes and verify Python scripts still import `scheduler` successfully.

Testing & validation
- Unit tests live in `tests/`. Run them from the repo root after building: `python3 -m unittest discover -s tests -v`. Tests that need `data/worlds` are skipped when the data is absent.
- CI (`.github/workflows/ci.yml`) builds the extension and runs the same command on every push and pull request.
- Also validate behaviour changes by running small simulations and checking output in `metrics/` or console logs.
- When adding tests, prefer lightweight Python tests that import `scheduler` and run deterministic small snapshots built in code, so they also run without `data/`.

Reporting changes
- For every PR or patch, include:
	- a concise description of the change and rationale,
	- how it was tested locally (build commands + example run),
	- any backward-compatibility impacts.

If you are unsure
- Ask before making large API or algorithmic changes.
- When editing configuration files (CMakeLists, scripts), prefer minimal changes and explain the reason.

Contact points in the code (good start points)
- `data_processing/data_factory.py` — dataset generation and snapshots.
- `metrics_tracker.py` — metrics aggregation and saving.
- `bindings.cpp` — Python/C++ interface.

Safety and limits
- Do not upload data or metrics to external services from automated changes.
- Avoid long-running experiments without confirmation (they may be expensive).

Done: if you want, I can now run a build and a quick sample run to verify the environment.