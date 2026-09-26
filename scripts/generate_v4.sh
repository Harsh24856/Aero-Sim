#!/bin/bash
# Generate the v4 dataset.
#
# SEQUENTIAL BY DEFAULT: one engine at a time, output straight to the terminal.
# On 8 GB this keeps memory pressure low, gives each engine the whole CPU, and
# makes the live output readable - four interleaved progress streams are not.
#
# The trade is wall-clock. One engine alone runs at roughly 2,000 rows/s; four at
# once run at about 1,100 each. So:
#     sequential (default)   ~85 min per engine, ~5.5 h for all four
#     --jobs 4               ~150 min per engine, ~2.5 h for all four
# Parallel finishes sooner overall; sequential is gentler and easier to follow.
#
#   scripts/generate_v4.sh                  all four, one at a time
#   scripts/generate_v4.sh --jobs 4         all four at once
#   scripts/generate_v4.sh --engines 914    just one
#   scripts/generate_v4.sh --rows 500000    quick pilot (~4 min each)
#   scripts/generate_v4.sh --background     detach instead of holding the terminal
#   scripts/generate_v4.sh --stop           stop everything
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Run from the main checkout even when invoked from a worktree: the venv, the
# data and the logs all live there.
case "$(pwd)" in
  */.claude/worktrees/*) cd "$(pwd | sed 's|/.claude/worktrees/.*||')" || exit 1 ;;
esac

PROC_PAT="backend/generate_dataset_v4.py"

find_python() {
  local c="$(pwd)/validation/venv/bin/python3"
  [ -x "$c" ] && { echo "$c"; return; }
  command -v python3
}

ROWS=10000000
ENGINES="Rotax_912_ULS Rotax_914_ULF Rotax_915_iS Rotax_916_iS"
FRESH=1
JOBS=1
BACKGROUND=0

while [ $# -gt 0 ]; do
  case "$1" in
    --rows)       ROWS="$2"; shift 2 ;;
    --jobs|-j)    JOBS="$2"; shift 2 ;;
    --keep)       FRESH=0; shift ;;
    --background) BACKGROUND=1; shift ;;
    --engines)    ENGINES=""
                  for k in ${2//,/ }; do
                    case $k in
                      912) ENGINES="$ENGINES Rotax_912_ULS" ;;
                      914) ENGINES="$ENGINES Rotax_914_ULF" ;;
                      915) ENGINES="$ENGINES Rotax_915_iS"  ;;
                      916) ENGINES="$ENGINES Rotax_916_iS"  ;;
                      *) echo "unknown engine '$k' (expected 912/914/915/916)"; exit 1 ;;
                    esac
                  done; shift 2 ;;
    --stop)       pkill -f "$PROC_PAT" 2>/dev/null; sleep 1
                  echo "stopped. still running: $(pgrep -fc "$PROC_PAT" 2>/dev/null || echo 0)"
                  exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

PY="$(find_python)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "no python3 found (looked for validation/venv/bin/python3, then PATH)"; exit 1
fi

# Never let two runs write the same directory: chunk files are numbered from
# zero each time, so they would interleave and corrupt the set.
if pgrep -f "$PROC_PAT" >/dev/null 2>&1; then
  echo "stopping generation already in progress..."
  pkill -f "$PROC_PAT"; sleep 2
fi

# Marker for this run. The progress view needs to tell a manifest written by
# THIS run from one left by a previous run - without it, engines that have not
# started yet report the old run's totals as though they were finished.
touch /tmp/v4_runstart

N_ENG=$(echo $ENGINES | wc -w | tr -d ' ')
echo "=== v4 dataset generation ==="
echo "interpreter : $PY"
echo "engines     :$ENGINES"
echo "rows each   : $(printf "%'d" "$ROWS")"
if [ "$JOBS" -le 1 ]; then
  echo "mode        : sequential, one engine at a time"
else
  echo "mode        : $JOBS engines at a time"
fi
echo

run_sequential() {
  local i=0 E K LOG START
  for E in $ENGINES; do
    i=$((i + 1))
    K=$(echo "$E" | cut -d_ -f2)
    LOG="/tmp/v4_$K.log"
    [ "$FRESH" = "1" ] && rm -rf "data/rotax_v4/$K"
    : > "$LOG"
    START=$(date +%s)
    echo "========================================================"
    echo " [$i/$N_ENG] $K   started $(date '+%H:%M:%S')   log $LOG"
    echo "========================================================"
    "$PY" -u "$(pwd)/backend/generate_dataset_v4.py" \
          --engine "$E" --rows "$ROWS" 2>&1 | tee -a "$LOG"
    echo " [$i/$N_ENG] $K finished in $((($(date +%s) - START) / 60)) min"
    echo
  done
}

run_parallel() {
  local i=0 running=0 E K
  for E in $ENGINES; do
    i=$((i + 1))
    K=$(echo "$E" | cut -d_ -f2)
    [ "$FRESH" = "1" ] && rm -rf "data/rotax_v4/$K"
    : > "/tmp/v4_$K.log"
    echo " [$i/$N_ENG] $K started (log /tmp/v4_$K.log)"
    "$PY" -u "$(pwd)/backend/generate_dataset_v4.py" \
          --engine "$E" --rows "$ROWS" > "/tmp/v4_$K.log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
      wait -n 2>/dev/null || wait
      running=$((running - 1))
    fi
  done
  wait
}

main_loop() {
  if [ "$JOBS" -le 1 ]; then run_sequential; else run_parallel; fi
  echo "=== generation complete ==="
  du -sh data/rotax_v4/* 2>/dev/null | sed 's/^/  /'
}

if [ "$BACKGROUND" = "1" ]; then
  nohup bash -c "cd '$(pwd)'; PY='$PY'; ROWS='$ROWS'; ENGINES='$ENGINES'; \
    FRESH='$FRESH'; JOBS='$JOBS'; N_ENG='$N_ENG'; \
    $(declare -f run_sequential run_parallel main_loop); main_loop" \
    > /tmp/v4_all.log 2>&1 &
  echo "running in background (pid $!)"
  echo "  combined log : /tmp/v4_all.log"
  echo "  progress     : scripts/v4_progress.sh --watch"
  echo "  stop         : scripts/generate_v4.sh --stop"
  exit 0
fi

echo "live output below. ctrl-c stops generation (foreground run)."
echo
main_loop
