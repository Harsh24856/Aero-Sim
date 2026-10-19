#!/usr/bin/env python3
"""Hybrid export (plan 2026-10-18-v5-real-world-fixes, P2): the LIVE export with only the
candidate's throttle-step fix - its detection and sensor nets, detection GBT and sensor
bias. Diagnosis, its calibration, the severity gate, health and RUL stay live, so RUL
sees the inputs it was fitted on. On the 914 the full candidate lowered fault naming
(93.7 -> 90.5%), on the 915 it raised RUL error (52 -> 63 h); the hybrid kept both.

    validation/venv/bin/python scripts/hybrid_export_v5.py 914 <candidate export dir> backend/models_v5/914 [git ref]

The git ref holding the live export defaults to origin/main.
"""
import json, os, pickle, shutil, subprocess, sys
k, cand, out = sys.argv[1:4]
ref = sys.argv[4] if len(sys.argv) > 4 else "origin/main"
live = out + "_live_tmp"
shutil.rmtree(live, ignore_errors=True)
tmp = f"/tmp/hyb_{k}"
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
subprocess.run(f"git archive {ref} backend/models_v5/{k} | tar -x -C {tmp} --strip-components=2",
               shell=True, check=True, executable="/bin/bash")
shutil.copytree(f"/tmp/hyb_{k}/{k}", live)
for h in ("detection", "sensor"):
    shutil.rmtree(os.path.join(live, h)); shutil.copytree(os.path.join(cand, h), os.path.join(live, h))
a_live = pickle.load(open(os.path.join(live, "assembly.pkl"), "rb"))
a_new = pickle.load(open(os.path.join(cand, "assembly.pkl"), "rb"))
a_live["detection_gbt"], a_live["sensor_bias"] = a_new["detection_gbt"], a_new["sensor_bias"]
pickle.dump(a_live, open(os.path.join(live, "assembly.pkl"), "wb"))
m = json.load(open(os.path.join(live, "manifest.json")))
mc = json.load(open(os.path.join(cand, "manifest.json")))
m["hybrid"] = {"detection_sensor_from": mc["run"], "detection_sensor_git_sha": mc["git_sha"],
               "rest_from": m["run"], "why": "the candidate's diagnosis lowered fault naming in A/B; "
               "only the throttle-step fix (detection, sensor) is taken"}
json.dump(m, open(os.path.join(live, "manifest.json"), "w"), indent=1)
shutil.rmtree(out); shutil.move(live, out)
print("hybrid", k, "detection+sensor from", mc["run"], "rest from", m["run"])
