#!/usr/bin/env bash
# Build the extension UNOPTIMISED (CMAKE_BUILD_TYPE=Debug: -O0 -g, assertions
# on), for stepping through the C++ in a debugger.
#
# Do not run experiments with this build: it is many times slower.  Python
# prints a RuntimeWarning when it imports it, and `scheduler.optimized` is
# False.  The module in the repository root is REPLACED by the debug one; run
# scripts/build.sh afterwards to get the optimised module back.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build --target scheduler
echo
echo "Debug build installed. Restore the optimised module with: scripts/build.sh"
