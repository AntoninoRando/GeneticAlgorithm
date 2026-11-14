#!/usr/bin/env bash
# Rebuild the `scheduler` extension in an already configured build/ directory
# (fast: no reconfigure).  Run scripts/build.sh once first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmake --build build --target scheduler
