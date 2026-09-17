#!/usr/bin/env bash
# Train the phase-5 v3 RUL head for every engine, ONE AT A TIME.
#
# One at a time is not a style choice: this machine has 8 GB, and two TensorFlow
# jobs swap until neither finishes. The loop stops at the first engine whose
# notebook fails rather than training three more against a broken one.
#
#   scripts/train_rul_v3.sh                 # all four engines
#   scripts/train_rul_v3.sh 914 916         # just these
#   RUL_PREFIX=rul_v4 scripts/train_rul_v3.sh 914      # pin an older probe dataset
#
# Logs land in .logs/phase5_rul_v3_<key>.log. Stop the app stack first
# (scripts/start_stack.sh holds ~1.5 GB and the AI service loads four model sets).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/.logs}"
mkdir -p "$LOG_DIR"
KEYS=("$@")
[ ${#KEYS[@]} -eq 0 ] && KEYS=(912 914 915 916)

[ -x "$ROOT/validation/venv/bin/python3" ] || { echo "missing validation/venv" >&2; exit 1; }

for key in "${KEYS[@]}"; do
  nb="phase5_rul_v3_${key}.ipynb"
  [ -f "$ROOT/validation/$nb" ] || { echo "no $nb - run validation/make_phase5_v3_notebooks.py" >&2; exit 1; }
  log="$LOG_DIR/phase5_rul_v3_${key}.log"
  echo "=== $key -> $log"
  ( cd "$ROOT/validation" && exec venv/bin/python3 run_notebook.py "$nb" ) >"$log" 2>&1 || {
    echo "  FAILED - see $log" >&2
    grep -E "CELL [0-9]+ FAILED|Error" "$log" | tail -5 >&2 || true
    exit 1
  }
  grep -E "hybrid \(test\)|\[PASS\]|\[FAIL\]|ALL GATES|SOME GATES|promoted|NOT promoted" "$log" || true
done
echo "done."
