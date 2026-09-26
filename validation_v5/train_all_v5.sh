#!/bin/zsh
# Unattended v5 chain: wait for data generation, then checks + caches for every
# engine, the HP search on the 914 pilot, then train 914 -> 912 -> 915 -> 916.
# Resumable: every step skips work already done, so rerunning continues.
#
#   nohup caffeinate -i zsh validation_v5/train_all_v5.sh > validation_v5/logs/train_all_v5.log 2>&1 &
#   tail -f validation_v5/logs/train_all_v5.log
set -u
cd "$(dirname "$0")/.."
PY=validation/venv/bin/python
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
say() { print "[$(date '+%m-%d %H:%M')] $*"; }

# 1. Wait for generate_dataset_v5 to finish (it writes manifest.json per engine at the end).
while pgrep -f generate_dataset_v5 >/dev/null; do sleep 60; done
say "generation finished"

# 2. Data checks + window cache, every engine (CPU). A failing engine is not trained.
ok=()
for e in 914 912 915 916; do
  if $PY validation_v5/run_v5.py $e --steps checks,cache; then ok+=$e; else say "ENGINE $e FAILED checks/cache - not trained"; fi
done
say "engines ready: $ok"

# 3. HP search on the 914 pilot (GPU), unless a finished one exists. If one is
#    already running (a relaunched chain), wait for it rather than start a second.
while pgrep -f hp_search_v5 >/dev/null; do sleep 60; done
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('validation_v5/hp_search_v5.json')).get('best') else 1)" 2>/dev/null; then
  say "HP search start"
  $PY validation_v5/hp_search_v5.py --cache validation_v5/cache/pilot_914 --trials 24 --rungs 3,8,30 || say "HP search FAILED - training with DEFAULT_CFG"
fi

# 4. Train, calibrate, RUL, test, model card - one engine at a time (GPU).
for e in $ok; do
  say "engine $e start"
  $PY validation_v5/run_v5.py $e --steps train,calibrate,rul,test,card && say "engine $e DONE" || say "ENGINE $e FAILED"
done
say "chain finished"
