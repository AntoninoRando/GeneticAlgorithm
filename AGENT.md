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
- Inspect key files: [bindings.cpp](bindings.cpp#L1), [include/scheduler](include/scheduler), [main.py](main.py#L1), and [data_factory.py](data_factory.py#L1).
- Run the CMake build to compile the extension before running Python scripts.
- Prefer small, focused changes and run a local simulation to validate behavior.

Build & run commands
- Build the Python extension (from repo root):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build
cp build/scheduler*.so .
```
- Build the CLI target:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build --target scheduler_cli
```
- Short helper (existing tasks are available in the workspace tasks): use the VS Code task `Build scheduler module (Debug)` or `Build scheduler cli (Debug)`.
- Run a Python simulation example:
```
python3 main.py
```
or use the provided `scripts/run.sh` for example runs.

Key files & directories
- `bindings.cpp`: Python bindings / glue.
- `include/scheduler`: public headers and core types.
- `data_factory.py`: dataset / snapshot helpers.
- `main.py`, `run/`, and `scripts/`: runners and example scripts.
- `metrics/` and `snapshots/`: sample outputs and inputs.

Coding conventions & constraints for AI changes
- Keep C++ ABI/API stable: avoid renaming public functions in `include/scheduler` without updating bindings.
- Limit changes in `bindings.cpp` to small, well-tested additions.
- Follow existing code style and minimal impact edits.
- Run the build after C++ changes and verify Python scripts still import `scheduler` successfully.

Testing & validation
- There are no formal unit tests in the repo; validate changes by running small simulations and checking output in `metrics/` or console logs.
- When adding tests, prefer lightweight Python tests that import `scheduler` and run deterministic small snapshots.

Reporting changes
- For every PR or patch, include:
	- a concise description of the change and rationale,
	- how it was tested locally (build commands + example run),
	- any backward-compatibility impacts.

If you are unsure
- Ask before making large API or algorithmic changes.
- When editing configuration files (CMakeLists, scripts), prefer minimal changes and explain the reason.

Contact points in the code (good start points)
- `data_factory.py` — dataset generation and snapshots.
- `metrics_tracker.py` — metrics aggregation and saving.
- `bindings.cpp` — Python/C++ interface.

Safety and limits
- Do not upload data or metrics to external services from automated changes.
- Avoid long-running experiments without confirmation (they may be expensive).

Done: if you want, I can now run a build and a quick sample run to verify the environment.