#!/usr/bin/env bash
# Every test suite in one command, each module in its own process with the environment
# it needs. main.py reads AERO_PHYSICS_VERSION once, at import, so modules that test
# different physics versions cannot share an interpreter.
#
#   scripts/test_all.sh               # backend + v5 AI service + cache-free validation_v5
#   scripts/test_all.sh --with-cache  # also the tests that read validation_v5/cache/914b
#
# BACKEND_PY / AI_PY override the interpreters (default: backend/.venv, validation/venv).
# AI_PY=none skips the TensorFlow suites (a machine without the AI environment);
# BACKEND_PY=none skips the physics-API suites (CI runs the two halves as separate jobs).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_PY="${BACKEND_PY:-$ROOT/backend/.venv/bin/python}"
AI_PY="${AI_PY:-$ROOT/validation/venv/bin/python}"
WITH_CACHE=0
[ "${1:-}" = "--with-cache" ] && WITH_CACHE=1

failed=()
run() {  # label, then the command (run from backend/)
  local label=$1; shift
  printf '%-48s' "$label"
  local out
  if out=$(cd "$ROOT/backend" && "$@" 2>&1); then
    echo "ok   $(grep -E '^Ran ' <<<"$out" | tail -1)"
  else
    echo "FAIL"
    tail -25 <<<"$out" | sed 's/^/    /'
    failed+=("$label")
  fi
}

if [ "$BACKEND_PY" = "none" ]; then
  echo "BACKEND_PY=none - skipping the physics-API suites"
elif [ ! -x "$BACKEND_PY" ]; then
  echo "missing $BACKEND_PY - set BACKEND_PY, or BACKEND_PY=none to skip the physics-API suites" >&2
  failed+=("backend environment")
else
  for f in "$ROOT"/backend/tests/test_*.py; do
    m=$(basename "$f" .py)
    case "$m" in
      test_ai_v5_service) continue ;;                 # TensorFlow: below
      test_simulation_safety)
        for v in v3 v4 v5; do
          run "backend $m ($v)" env AERO_PHYSICS_VERSION=$v "$BACKEND_PY" -m unittest "tests.$m"
        done ;;
      *) run "backend $m" "$BACKEND_PY" -m unittest "tests.$m" ;;
    esac
  done
fi

if [ "$AI_PY" = "none" ]; then
  echo "AI_PY=none - skipping the TensorFlow suites"
elif [ -x "$AI_PY" ]; then
  run "ai test_ai_v5_service" "$AI_PY" -m unittest tests.test_ai_v5_service
  v5=(test_graph_parity_v5 test_relabel_v5b test_rul_smoothing_v5)
  [ $WITH_CACHE = 1 ] && v5+=(test_pipeline_v5 test_specialists_v5 test_export_parity_v5)
  for m in "${v5[@]}"; do
    run "validation_v5 $m" "$AI_PY" "$ROOT/validation_v5/tests/$m.py"
  done
else
  echo "missing $AI_PY - set AI_PY, or AI_PY=none to skip the TensorFlow suites" >&2
  failed+=("ai environment")
fi

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "ALL PASS"
else
  echo "${#failed[@]} FAILED: ${failed[*]}"
  exit 1
fi
