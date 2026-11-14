#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs automatically the first time the dev container is created
# (wired up via "postCreateCommand" in devcontainer.json).
#
# It makes the project runnable end-to-end with zero manual steps:
#   1. system packages   -> debug tooling + Python/pybind11 build headers
#   2. Python libraries  -> everything in requirements.txt (numpy, matplotlib)
#   3. native module     -> builds the C++ "scheduler" pybind11 module (.so)
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> [1/3] Installing system packages (debug tooling + Python/pybind11 build deps)…"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gdb gdbserver python3-dbg python3-dev python3-pip pybind11-dev
sudo rm -rf /var/lib/apt/lists/*

echo "==> [2/3] Installing Python libraries from requirements.txt…"
# Debian's system Python is "externally managed" (PEP 668), so install globally
# with --break-system-packages instead of a virtualenv.
pip3 install --no-cache-dir --break-system-packages -r requirements.txt

echo "==> [3/3] Building the C++ 'scheduler' module…"
# CMake writes the .so straight into the repository root
# (LIBRARY_OUTPUT_DIRECTORY in CMakeLists.txt): there is nothing to copy.  The
# old `cp build/scheduler*.so .` matched no file and, under `set -e`, made this
# script exit with an error after a successful build.
# RelWithDebInfo = optimised, with debug symbols for gdb.  It is the same code
# the container built before, when it passed Debug and CMakeLists.txt forced
# -O3 on top; Debug now really means unoptimised (scripts/build-debug.sh).
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE="$(which python3)"
cmake --build build

echo ""
echo "==> Setup complete. Run the genetic algorithm with:"
echo "      python3 main.py --no-start-prompt --generations 5"
