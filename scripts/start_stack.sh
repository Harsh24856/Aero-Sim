#!/usr/bin/env bash
# Start the whole AERO-SIM stack with one command:
#   AI service :8100 -> physics backend :8000 -> frontend :3000
# Each service is started only after the previous one passes its health check.
# Logs go to .logs/. Ctrl-C (or any service exiting) stops everything.
#
#   scripts/start_stack.sh                          # physics v4 + aiv4.py (default)
#   AERO_PHYSICS_VERSION=v3 scripts/start_stack.sh  # physics v3 + aiv3.py
#   AERO_PHYSICS_VERSION=v2 scripts/start_stack.sh  # legacy v2 + ai.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/.logs}"
PHYSICS_VERSION="${AERO_PHYSICS_VERSION:-v4}"
case "$PHYSICS_VERSION" in v2) AI_MODULE=ai ;; v3) AI_MODULE=aiv3 ;; *) AI_MODULE=aiv4 ;; esac

for port in 8100 8000 3000; do
  if lsof -ti:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is already in use - stop that process first" >&2
    exit 1
  fi
done
[ -x "$ROOT/validation/venv/bin/uvicorn" ] || { echo "missing validation/venv - see README (AI environment)" >&2; exit 1; }
[ -x "$ROOT/backend/.venv/bin/python" ]    || { echo "missing backend/.venv - see README (physics environment)" >&2; exit 1; }
[ -d "$ROOT/frontend/node_modules" ]       || { echo "run 'npm install' in frontend/ first" >&2; exit 1; }
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "stopping AERO-SIM..."
  if [ ${#pids[@]} -gt 0 ]; then kill "${pids[@]}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait_for() {  # name url timeout_seconds
  local name=$1 url=$2 timeout=$3 i
  for ((i = 0; i < timeout; i++)); do
    if curl -fsS -m 2 -o /dev/null "$url" 2>/dev/null; then
      echo "  $name ready ($url)"
      return 0
    fi
    sleep 1
  done
  echo "  $name did not come up within ${timeout}s - see $LOG_DIR" >&2
  exit 1
}

echo "AERO-SIM: physics $PHYSICS_VERSION with $AI_MODULE.py - logs in $LOG_DIR"

(cd "$ROOT/backend" && exec ../validation/venv/bin/uvicorn "$AI_MODULE:app" --host 127.0.0.1 --port 8100) \
  >"$LOG_DIR/ai.log" 2>&1 &
pids+=($!)
wait_for "AI service" http://127.0.0.1:8100/health 180
if ! curl -fsS -m 5 http://127.0.0.1:8100/health | grep -q '"backend_validated":true'; then
  echo "  WARNING: AI service reports backend_validated=false - diagnostics are not trustworthy" >&2
fi

(cd "$ROOT/backend" && AERO_PHYSICS_VERSION="$PHYSICS_VERSION" exec ./.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000) \
  >"$LOG_DIR/backend.log" 2>&1 &
pids+=($!)
wait_for "physics backend" http://127.0.0.1:8000/health 60

(cd "$ROOT/frontend" && exec npm run dev -- --port 3000) >"$LOG_DIR/frontend.log" 2>&1 &
pids+=($!)
wait_for "frontend" http://localhost:3000/ 120

echo
echo "Open http://localhost:3000 (use localhost, not 127.0.0.1). Ctrl-C stops everything."
while :; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "a service exited unexpectedly - see $LOG_DIR" >&2
      exit 1
    fi
  done
  sleep 2
done
