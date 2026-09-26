#!/bin/bash
# Progress of the v4 dataset generation.
#
# The generator prints one line per 50 scenarios, which at ~1,200 rows/s is
# roughly every four minutes - so an empty log early on means "not yet", not
# "broken". This reads the logs, the parquet output and the process table
# together, so a stalled or dead run is distinguishable from a quiet one.
#
#   scripts/v4_progress.sh            once
#   scripts/v4_progress.sh --watch    refresh every 30s until generation ends
#
# --watch is built in because macOS has no watch(1) and installing coreutils
# just to see a progress bar is not a reasonable ask.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Run from the main checkout even when invoked from a git worktree: the data and
# the logs live there.
case "$(pwd)" in
  */.claude/worktrees/*) cd "$(pwd | sed 's|/.claude/worktrees/.*||')" || exit 1 ;;
esac

INTERVAL=10
WATCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --watch|-w) WATCH=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

if [ "$WATCH" = "1" ]; then
  while true; do
    clear
    "$0" --interval "$INTERVAL" 2>/dev/null || true
    # Stop on its own once every process has exited, so this does not sit there
    # redrawing a finished run forever.
    if ! pgrep -f "backend/generate_dataset_v4.py" >/dev/null 2>&1; then
      echo
      echo "all generation processes have exited."
      break
    fi
    echo
      echo "(refreshing every ${INTERVAL}s -- ctrl-c stops watching, not generating)"
    sleep "$INTERVAL"
  done
  exit 0
fi

TARGET=10000000
running=$(pgrep -f "backend/generate_dataset_v4.py" 2>/dev/null | wc -l | tr -d ' ')

echo "=== v4 dataset generation ==="
echo "processes alive: ${running}/4    target: 10,000,000 rows per engine"
echo

total=0
for K in 912 914 915 916; do
  LOG=/tmp/v4_$K.log
  line=$(grep -E "rows/s" "$LOG" 2>/dev/null | tail -1)

  # Three states, decided by the manifest and the log's freshness rather than
  # by the log alone. The manifest is written only on completion, so it is the
  # one unambiguous marker; a log is "live" only if it was written in the last
  # 90 s, which at a 15 s report cadence cannot produce a false negative.
  # Comparing against v4_all.log instead reported FINISHED engines as queued,
  # because that file is written continuously and every other log looks older.
  MAN="data/rotax_v4/$K/manifest.json"
  fresh=0
  if [ -f "$LOG" ]; then
    age=$(( $(date +%s) - $(stat -f %m "$LOG" 2>/dev/null || echo 0) ))
    [ "$age" -lt 90 ] && fresh=1
  fi
  # Only a manifest newer than this run's marker means "finished in this run".
  done_now=0
  if [ -f "$MAN" ]; then
    if [ ! -f /tmp/v4_runstart ] || [ "$MAN" -nt /tmp/v4_runstart ]; then done_now=1; fi
  fi
  if [ "$done_now" = "1" ] && [ "$fresh" = "0" ]; then
    rows=$(sed -n 's/.*"rows": \([0-9]*\).*/\1/p' "$MAN" | head -1)
    scen=$(sed -n 's/.*"scenarios": \([0-9]*\).*/\1/p' "$MAN" | head -1)
    printf "  %s  DONE   %'12d rows  %5s scenarios\n" "$K" "${rows:-0}" "${scen:-?}"
    total=$((total + ${rows:-0}))
    continue
  fi
  if [ "$fresh" = "0" ]; then
    printf "  %s  queued\n" "$K"; continue
  fi

  if [ -z "$line" ]; then
    if grep -qiE "traceback|error" "$LOG" 2>/dev/null; then
      printf "  %s  ERROR -- tail %s\n" "$K" "$LOG"
    elif [ "$running" -gt 0 ]; then
      printf "  %s  starting up (first report lands at 50 scenarios, ~4 min)\n" "$K"
    else
      printf "  %s  NOT RUNNING\n" "$K"
    fi
    continue
  fi

  rows=$(echo "$line" | sed -E 's/.* ([0-9,]+) rows .*/\1/' | tr -d ',')
  scen=$(echo "$line" | sed -E 's/.*\] +([0-9]+) scenarios.*/\1/')
  rate=$(echo "$line" | sed -E 's/.* ([0-9,]+) rows\/s.*/\1/' | tr -d ',')
  pct=$(echo "$line"  | sed -E 's/.* ([0-9.]+)%.*/\1/')
  total=$((total + rows))

  eta="?"
  if [ "${rate:-0}" -gt 0 ] 2>/dev/null; then
    eta="$(( (TARGET - rows) / rate / 60 ))m left"
  fi
  n=$(( ${pct%.*} * 40 / 100 ))
  bar=$(printf "%${n}s" | tr ' ' '#')
  printf "  %s [%-40s] %5s%%  %'12d rows  %4s scen  %'6d r/s  %s\n" \
         "$K" "$bar" "$pct" "$rows" "$scen" "$rate" "$eta"
done

echo
printf "  combined %'d / 40,000,000 rows\n" "$total"
echo
echo "on disk (chunks flush every 400k rows, so 0B early is normal):"
du -sh data/rotax_v4/* 2>/dev/null | sed 's/^/  /'
