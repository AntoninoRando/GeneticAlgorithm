#!/usr/bin/env bash
# Configure (if needed) and build the `scheduler` Python extension, optimised
# and with debug symbols (RelWithDebInfo).  This is the build to use for
# experiments.  Works from any directory.  CMake writes the .so straight into
# the repository root (LIBRARY_OUTPUT_DIRECTORY in CMakeLists.txt), so there is
# nothing to copy.
#
# The build type is passed explicitly on purpose: build/ remembers the type of
# its last configuration, and after scripts/build-debug.sh that would be Debug.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build --target scheduler
