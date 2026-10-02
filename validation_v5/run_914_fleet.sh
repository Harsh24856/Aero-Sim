#!/usr/bin/env bash
# The 914 on the fleet-wear twin, end to end: regenerate -> checks + cache ->
# relabel (v5b, no per-channel) -> gate -> retrain all five specialists.
# Stops at the first failure; every step logs to validation_v5/logs/914_fleet_*.log.
#
#   cd ~/Documents/UAV_Engine
#   nohup caffeinate -i bash validation_v5/run_914_fleet.sh > validation_v5/logs/914_fleet.log 2>&1 &
#   tail -f validation_v5/logs/914_fleet.log
#
# Afterwards: read validation_v5/artifacts/914b_specialists/model_card.md, then
# export_v5.py 914b (that is what switches the live 914 to the fleet-wear twin).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=validation/venv/bin/python
GEN=backend/.venv/bin/python
L=validation_v5/logs
say() { echo "[$(date '+%H:%M:%S')] $*"; }

# 1. keep the new-engine-twin models for rollback (once)
if [ ! -d validation_v5/artifacts/914b_specialists_newengine ]; then
  cp -R validation_v5/artifacts/914b_specialists validation_v5/artifacts/914b_specialists_newengine
  say "backed up 914b_specialists -> 914b_specialists_newengine"
fi

# 2. regenerate (~2 h CPU)
say "regenerating 914 (fleet-wear twin) ..."
$GEN backend/generate_dataset_v5.py --engine Rotax_914_ULF --flights 6000 > $L/914_fleet_regen.log 2>&1
tail -1 $L/914_fleet_regen.log

# 3. the data must say fleet_wear, or the generator was the old one
twin=$($PY -c "import json; print(json.load(open('data/rotax_v5/914/manifest.json'))['contract'].get('twin','new_engine'))")
[ "$twin" = "fleet_wear" ] || { say "STOP: data twin is '$twin', not fleet_wear - old generator?"; exit 1; }
say "data twin: $twin"

# 4. checks + cache (run_v5 exits non-zero if a check fails)
say "checks + cache ..."
$PY validation_v5/run_v5.py 914 --steps checks,cache --force checks > $L/914_fleet_cache.log 2>&1
grep -E 'PASS|FAIL' $L/914_fleet_cache.log | grep -c PASS | xargs -I{} echo "  {} checks passed"

# 5. relabel: standard v5b labels (per-channel still hides oil degradation)
say "relabel ..."
$PY validation_v5/relabel_v5b.py 914 > $L/914_fleet_relabel.log 2>&1
cat $L/914_fleet_relabel.log

# 6. gate: healthy residuals must be tight now, and the faults the old twin hid must have positives
say "gate ..."
(cd validation_v5 && ../$PY - <<'EOF'
import sys, numpy as np
sys.path.insert(0, '.')
from pipeline_v5 import Cache, RES_SLICE, END_COLS
import features_v5 as F
import relabel_v5b as R
from sensors_v5 import FAULTABLE_CHANNELS
c = Cache('cache/914b')
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
for ch, bar in (('engine_rpm', 9.0), ('oil_pressure', 12.0), ('fuel_flow', 6.0)):
    q = float(np.quantile(np.abs(wm[:, F.RESIDUAL_COLS.index(f'res_{ch}')]), 0.99))
    print(f'  healthy q99 {ch:13s} {q:5.2f} sigma (bar {bar}; new-engine twin was 16.2 / 31 / 9.4)')
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

# 7. retrain every specialist, then assemble, RUL, test, card (~8 h GPU)
say "training all five specialists ..."
$PY validation_v5/specialists_v5.py 914b --force detection,diagnosis,severity,sensor,health > $L/914_fleet_train.log 2>&1
tail -3 $L/914_fleet_train.log
say "DONE - card: validation_v5/artifacts/914b_specialists/model_card.md"
