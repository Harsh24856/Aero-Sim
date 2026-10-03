#!/usr/bin/env bash
# v5 engines on the fleet-wear twin, end to end, one engine after another:
# regenerate -> checks + cache -> relabel (v5b) -> gate -> retrain all five specialists.
# Default: the three engines still on the 914's placeholder models.
#
#   cd ~/Documents/UAV_Engine
#   nohup caffeinate -is bash validation_v5/run_fleet.sh > validation_v5/logs/fleet.log 2>&1 &
#   tail -f validation_v5/logs/fleet.log
#
#   bash validation_v5/run_fleet.sh 915          # one engine
#
# An engine that fails a step is reported and skipped; the next engine still runs.
# Per engine: validation_v5/logs/<key>_fleet_{regen,cache,relabel,train}.log and the
# card at validation_v5/artifacts/<key>b_specialists/model_card.md.
# Afterwards, per engine: review the card, then `export_v5.py <key>b` (that replaces the
# placeholder in the live stack), the parity test and `e2e_v5.py --engines <key>`.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=validation/venv/bin/python
GEN=backend/.venv/bin/python
L=validation_v5/logs
say() { echo "[$(date '+%H:%M:%S')] $*"; }
engine_model() { case "$1" in 912) echo Rotax_912_ULS ;; 914) echo Rotax_914_ULF ;; 915) echo Rotax_915_iS ;; 916) echo Rotax_916_iS ;; *) return 1 ;; esac; }

run_one() (
  set -e
  key=$1
  model=$(engine_model "$key") || { say "$key: unknown engine"; exit 1; }
  art=validation_v5/artifacts/${key}b_specialists

  # 1. keep any earlier models for rollback (once)
  if [ -d "$art" ] && [ ! -d "${art}_prev" ]; then cp -R "$art" "${art}_prev"; say "$key: backed up $art"; fi

  # 2. regenerate (~2 h CPU)
  say "$key: regenerating ($model, fleet-wear twin) ..."
  $GEN backend/generate_dataset_v5.py --engine "$model" --flights 6000 > "$L/${key}_fleet_regen.log" 2>&1
  tail -1 "$L/${key}_fleet_regen.log"

  # 3. the data must carry the fleet-wear twin
  twin=$($PY -c "import json; print(json.load(open('data/rotax_v5/$key/manifest.json'))['contract'].get('twin','new_engine'))")
  [ "$twin" = "fleet_wear" ] || { say "$key: STOP - data twin is '$twin'"; exit 1; }

  # 4. checks + cache (run_v5 exits non-zero when a check fails)
  say "$key: checks + cache ..."
  $PY validation_v5/run_v5.py "$key" --steps checks,cache --force checks > "$L/${key}_fleet_cache.log" 2>&1
  echo "  $(grep -c PASS "$L/${key}_fleet_cache.log") checks passed"

  # 5. relabel: standard v5b labels
  say "$key: relabel ..."
  $PY validation_v5/relabel_v5b.py "$key" > "$L/${key}_fleet_relabel.log" 2>&1

  # 6. gate before the long train
  say "$key: gate ..."
  (cd validation_v5 && KEY=$key ../$PY - <<'EOF'
import os, sys, numpy as np
sys.path.insert(0, '.')
from pipeline_v5 import Cache, RES_SLICE, END_COLS
import features_v5 as F
import relabel_v5b as R
from sensors_v5 import FAULTABLE_CHANNELS
c = Cache(f"cache/{os.environ['KEY']}b")
assert c.contract.get('twin') == 'fleet_wear', f"cache twin {c.contract.get('twin')}"
ids = c.end_ids(['train'], jitter=False)
ids = np.sort(np.random.default_rng(0).choice(ids, min(20000, len(ids)), replace=False))
sc = c.contract['scaler']; mean, std = np.array(sc['mean'])[RES_SLICE], np.array(sc['std'])[RES_SLICE]
wm, ok = [], []
for i in range(0, len(ids), 2048):
    seq, _, E = c.batch(ids[i:i + 2048]); wm.append((seq[:, :, RES_SLICE] * std + mean).mean(1))
    fm = np.stack([c.labels(E, f'fm_{n}') for n in R.FAULT_NAMES], 1)
    sf = np.stack([c.labels(E, f'sf_{ch}_flag') for ch in FAULTABLE_CHANNELS], 1)
    ok.append((fm.max(1) < 0.01) & (sf == 0).all(1))
wm = np.concatenate(wm)[np.concatenate(ok)]
fail = []
for ch, bar in (('engine_rpm', 9.0), ('oil_pressure', 14.0), ('fuel_flow', 6.0)):   # 914: 7.8 / 12.0 / 4.3
    q = float(np.quantile(np.abs(wm[:, F.RESIDUAL_COLS.index(f'res_{ch}')]), 0.99))
    print(f'  healthy q99 {ch:13s} {q:5.2f} sigma (bar {bar})')
    if q > bar: fail.append(ch)
split = np.empty(len(c.ends), dtype=object)
for _, r in c.flights.iterrows(): split[int(r.end0):int(r.end0 + r.n_ends)] = r.split
for n in ('valve_leakage', 'compression_loss', 'oil_pump_degradation', 'bearing_wear'):
    k = END_COLS.index(f'fmv_{n}')
    tr, te = int(c.ends[split == 'train', k].sum()), int(c.ends[split == 'test', k].sum())
    print(f'  {n:22s} train {tr:6d}  test {te:5d}')
    if tr < 3000 or te < 300: fail.append(n)
if fail:
    sys.exit(f'GATE FAILED: {fail}')
print('  gate passed')
EOF
  )

  # 7. retrain every specialist, then assemble, RUL, test, card (~11 h GPU)
  say "$key: training all five specialists ..."
  $PY validation_v5/specialists_v5.py "${key}b" --force detection,diagnosis,severity,sensor,health > "$L/${key}_fleet_train.log" 2>&1
  grep -E '^\| (Detection|Diagnosis|Severity|Sensor|Health|RUL)' "$art/model_card.md" | cut -d'|' -f2,4 || true
  say "$key: DONE - card: $art/model_card.md"
)

done_ok=() failed=()
for key in "${@:-912 915 916}"; do
  for k in $key; do
    if run_one "$k"; then done_ok+=("$k"); else failed+=("$k"); say "$k: FAILED - see $L/${k}_fleet_*.log; moving on"; fi
  done
done
say "finished. done: ${done_ok[*]:-none}; failed: ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
