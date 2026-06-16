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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE="$(which python3)"
cmake --build build
cp build/scheduler*.so .

echo ""
echo "==> Setup complete. Run the genetic algorithm with:"
echo "      python3 main.py --no-start-prompt --generations 5"
