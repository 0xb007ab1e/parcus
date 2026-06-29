#!/usr/bin/env bash
# Coverage-guided parser fuzzing with Atheris (libFuzzer).
#
# Atheris is NOT a project dependency — install it ad hoc (needs clang):
#   .venv/bin/python -m pip install atheris      # or: pip install atheris
# then run this. Any libFuzzer crash file written here means a parser raised on some input
# (a fail-open gap to fix). Crashes are gitignored.
#
# Usage:  qa/fuzz/run_atheris.sh                 # 30s campaign
#         MAX_TOTAL_TIME=300 qa/fuzz/run_atheris.sh
#         qa/fuzz/run_atheris.sh -runs=1000000   # pass libFuzzer flags through
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARCUS_PY="${PARCUS_PY:-$REPO_ROOT/.venv/bin/python}"
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-30}"

"$PARCUS_PY" -c "import atheris" 2>/dev/null || {
  echo "atheris not installed — run: $PARCUS_PY -m pip install atheris (needs clang)" >&2
  exit 2
}

exec "$PARCUS_PY" "$SCRIPT_DIR/fuzz_parsers.py" \
  -max_total_time="$MAX_TOTAL_TIME" \
  -artifact_prefix="$SCRIPT_DIR/crash-" \
  "$@"
