#!/usr/bin/env bash
# Clean rebuild: delete the CMake cache, reconfigure and build the extension
# (optimised, RelWithDebInfo).  Use it after changing compilers, Python
# versions or CMake options.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build --target scheduler
