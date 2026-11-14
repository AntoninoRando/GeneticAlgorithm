#!/usr/bin/env bash
# Configure (if needed) and build the `scheduler` Python extension.
# Works from any directory.  CMake writes the .so straight into the repository
# root (LIBRARY_OUTPUT_DIRECTORY in CMakeLists.txt), so there is nothing to copy.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmake -S . -B build -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build --target scheduler
