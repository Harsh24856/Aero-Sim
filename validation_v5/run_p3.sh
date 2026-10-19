#!/usr/bin/env bash
# P3 (docs/superpowers/plans/2026-10-18-v5-real-world-fixes.md): every engine on the new
# physics (iS mixture schedule, constant-speed governor, coolant thermostat), with the
# P2 throttle-step flights. New names throughout, so the live engines' data, caches and
# artifacts stay as they are:
#   data/rotax_v5_p3/<e> (+ data/rotax_v5_p3_steps/<e>), linked as data/rotax_v5/<e>p3
#   cache/<e>p3 -> relabel -> cache/<e>p3b -> artifacts/<e>p3b_specialists
# Generation runs for all four engines at once (CPU); training one engine at a time,
# the 914 first (user rule). Every step is skipped when its output exists, so a rerun
# resumes.
#
#   nohup caffeinate -is bash validation_v5/run_p3.sh > validation_v5/logs/p3.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=validation/venv/bin/python
GEN=backend/.venv/bin/python
L=validation_v5/logs
mkdir -p "$L" data/rotax_v5_p3 data/rotax_v5_p3_steps
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
model() { case "$1" in 912) echo Rotax_912_ULS ;; 914) echo Rotax_914_ULF ;; 915) echo Rotax_915_iS ;; 916) echo Rotax_916_iS ;; esac; }
KEYS="${*:-914 912 915 916}"

# 1. generate (all engines in parallel; each process is single-threaded)
for k in $KEYS; do
  m=$(model "$k")
  [ -f "data/rotax_v5_p3/$k/manifest.json" ] || \
    $GEN backend/generate_dataset_v5.py --engine "$m" --flights 6000 --out-dir data/rotax_v5_p3 > "$L/${k}p3_gen.log" 2>&1 &
  [ -f "data/rotax_v5_p3_steps/$k/manifest.json" ] || \
    $GEN backend/generate_dataset_v5.py --engine "$m" --flights 900 --start-id 100000 \
      --missions throttle_steps,idle_descent --out-dir data/rotax_v5_p3_steps > "$L/${k}p3_gen_steps.log" 2>&1 &
done
wait
say "generation finished"

run_one() (
  set -e
  k=$1
  ln -sfn "../rotax_v5_p3/$k" "data/rotax_v5/${k}p3"
  say "$k: checks ..."
  $PY validation_v5/run_v5.py "${k}p3" --steps checks > "$L/${k}p3_checks.log" 2>&1
  if [ ! -f "validation_v5/cache/${k}p3/contract_v5.json" ]; then
    say "$k: cache ..."
    $PY validation_v5/pipeline_v5.py "data/rotax_v5_p3/$k" "validation_v5/cache/${k}p3" \
      --extra "data/rotax_v5_p3_steps/$k" > "$L/${k}p3_cache.log" 2>&1
  fi
  if [ ! -f "validation_v5/cache/${k}p3b/contract_v5.json" ]; then
    say "$k: relabel ..."
    $PY validation_v5/relabel_v5b.py "${k}p3" > "$L/${k}p3_relabel.log" 2>&1
  fi
  say "$k: training all five specialists (~10 h GPU) ..."
  $PY validation_v5/specialists_v5.py "${k}p3b" --verbose 2 > "$L/${k}p3_train.log" 2>&1
  say "$k: DONE - validation_v5/artifacts/${k}p3b_specialists/model_card.md"
)

done_ok=() failed=()
for k in $KEYS; do
  if run_one "$k"; then done_ok+=("$k"); else failed+=("$k"); say "$k: FAILED - see $L/${k}p3_*.log"; fi
done
say "finished. done: ${done_ok[*]:-none}; failed: ${failed[*]:-none}"
