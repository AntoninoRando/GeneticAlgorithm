#!/usr/bin/env bash
# Clean rebuild: delete the CMake cache, reconfigure and build the extension.
# Use it after changing compilers, Python versions or CMake options.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf build
cmake -S . -B build -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build --target scheduler
