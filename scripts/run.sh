#!/usr/bin/env bash
# Build the extension, then run main.py.  Extra arguments go to main.py:
#
#     scripts/run.sh --no-start-prompt --generations 50
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scripts/build.sh
exec python3 main.py "$@"
