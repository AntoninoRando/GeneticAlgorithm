#!/usr/bin/env bash
# Configure and build with CMAKE_BUILD_TYPE=Debug, out of source (in build/).
#
# This is the configuration the dev container uses (.devcontainer/postCreate.sh).
# NOTE: CMakeLists.txt adds -O3 to the extension whatever the build type, so
# the result is optimised code WITH debug symbols (-g).  You can set
# breakpoints, but some local variables are optimised away and stepping can
# jump around.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build --target scheduler
