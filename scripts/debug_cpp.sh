#!/usr/bin/env bash
# Build the extension and run main.py under gdb, so breakpoints can be set in
# the C++ headers (include/scheduler/*.h).  Extra arguments go to main.py:
#
#     scripts/debug_cpp.sh --generations 5 --population 20
#
# In gdb:  (gdb) break SchedulerGP::solveNextGeneration   then   (gdb) run
# ("set breakpoint pending on" is passed for you: the extension is loaded only
# when Python imports it.)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scripts/build-debug.sh
exec gdb -ex "set breakpoint pending on" --args python3 main.py --no-start-prompt "$@"
