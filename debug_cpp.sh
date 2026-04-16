#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE="$(which python3)"
cmake --build build --target scheduler_cli
exec gdb --args ./build/scheduler_cli
