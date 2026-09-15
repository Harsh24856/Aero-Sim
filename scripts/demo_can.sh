#!/usr/bin/env bash
# Fly the simulated aircraft over the CAN bus into a running AERO-SIM stack.
#   1. scripts/start_stack.sh (in another terminal)
#   2. open http://localhost:3000/simulate?engine=<ENGINE> and press Start
#   3. scripts/demo_can.sh [engine] [mismatch] [profile]
# Starts the CAN bridge and the aircraft; Ctrl-C stops both. The Diagnostics panel shows
# CAN LIVE while sensor frames arrive.
#
#   scripts/demo_can.sh                                   # 914, no mismatch, mission profile
#   scripts/demo_can.sh Rotax_914_ULF sensor_drift cruise # a drifting CHT sensor on the aircraft
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="${1:-Rotax_914_ULF}"
MISMATCH="${2:-none}"
PROFILE="${3:-mission}"
BUS="${CAN_BUS:-udp}"
PY="$ROOT/backend/.venv/bin/python"

[ -x "$PY" ] || { echo "missing backend/.venv - see README" >&2; exit 1; }
if ! curl -fsS -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "physics backend is not running on :8000 - start scripts/start_stack.sh first" >&2
  exit 1
fi
selected=$(curl -fsS -m 2 http://127.0.0.1:8000/health | "$PY" -c "import json,sys; print(json.load(sys.stdin)['engine_model'])")
if [ "$selected" != "$ENGINE" ]; then
  echo "the twin is on $selected but the aircraft would fly $ENGINE - open /simulate?engine=$ENGINE first" >&2
  exit 1
fi

pids=()
cleanup() {
  trap - INT TERM EXIT
  if [ ${#pids[@]} -gt 0 ]; then kill "${pids[@]}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  echo "CAN demo stopped"
}
trap cleanup INT TERM EXIT

cd "$ROOT/backend"
"$PY" -u can_ingest.py --bus "$BUS" &
pids+=($!)
sleep 1
echo "aircraft: $ENGINE, mismatch=$MISMATCH, profile=$PROFILE, bus=$BUS"
"$PY" -u aircraft_sim.py --engine "$ENGINE" --mismatch "$MISMATCH" --profile "$PROFILE" --bus "$BUS" &
pids+=($!)
wait "${pids[1]}"
