# Deployment roadmap

Status of the physics-v3 programme against PS 26054, and what remains before v3 becomes the default.

## Done

| Area | Deliverable |
|---|---|
| Physics | `physics.py` v3 behind `AERO_PHYSICS_VERSION`, v2 bit-identical; visible wear; engine failure modes (`failure_modes.py`); electrical and injection channels |
| Data | Dataset v3, 10 M rows per engine, life-stage sampled, real-hour RUL |
| Models | All four engines trained and exported to `backend/models_v3/` with manifests (all four retrained 2026-09-15 on the audited, regenerated data; 20/20 gates passed) |
| Serving | `aiv3.py` (uvicorn, same API as `ai.py`), contract checked at startup, parity-tested against the pipeline |
| Twin method | `residual.py` physics residuals: degradation index, sensor-drift and saturation detection |
| Advisory | failure-mode, engine-hour RUL, residual and saturation items; v2 output unchanged |
| Persistence | Supabase migration `add_physics_v3_model_versioning` (all earlier runs marked `v2`); `dbv3.py` |
| Frontend | version-aware RUL and flight time, legacy badge, live failure-mode and residual panels |
| Vehicle interface | Plant and twin split over CAN: `aircraft_sim.py` (simulated aircraft) -> `can_bus.py` frames (UDP multicast or SocketCAN/python-can) -> `can_ingest.py` -> `/params` + `/measured`; verified live, 0 frames lost |

## Before flipping the default to v3

1. ~~**Rotax 912 on v3.**~~ Done 2026-09-14: retrained, exported, parity passed.
2. **Residual scoring on the test split.** Run `validation/residual_eval.py` and record detection rate and
   signature accuracy per fault type in `docs/model_cards.md`.
3. **End-to-end check with persistence.** One logged-in v3 flight: confirm `simulations.model_version = 'v3'`,
   engine-hour RUL on the report, history and replay pages, and residual columns filled.
4. ~~**Confirm TBO values.**~~ Done 2026-09-15: 912 ULS 2000 h (Rotax datasheet), 914 2000 h (with
   SB-914-039UL), 915 iS 1200 h, 916 iS 2000 h.
5. ~~**Flip the default.**~~ Done 2026-09-15: `DEFAULT_PHYSICS_VERSION = "v3"`, `main.py` defaults to v3, README
   starts `aiv3.py`; legacy generators and `validate.py` pin `physics_version="v2"`.

## Physics fixes and regeneration

**Done (2026-09-14/15), all v3-only with v2 verified bit-identical:**

- **Propeller sizing.** `V3_PROP_OVERRIDES`: 912 1.97 m, 915 2.30 m, 916 2.28 m, so vibration/oil-pressure stress
  starts at the same throttle as the 914.
- **Early-life coverage.** Every 5th scenario starts on a new engine (wear 0-2%).
- **Cooling degradation** scales with power (`cooling_heat_delta`); phase-4 cooling AUC 0.63-0.77 -> 0.84-0.93.
- **Audit fixes:** 916 iS TBO 2000 h, airspeed floor, visible RPM glitches (~2% of seconds), random fault type per
  channel, v3 EGT/CHT stress thresholds, oil-temperature sensor 180 C and recalibrated oil temperature.
- **All four regenerated, retrained (phases 0-5, 19 h, 20/20 gates) and exported**; parity and per-channel
  reliability written. RUL MAE: 914 1.15%, 912 0.77%, 915 0.62%, 916 0.68% of TBO. The 916 RPM diagnosis channel is
  marked unreliable (precision 0.05) and cannot cap health. Previous exports: `backend/models_v3_backup_pre_final_20260915_0907/`.

- **Oil-pressure limit** in `advisory.py` is now 12 psi (Rotax 0.8 bar); the old 2.0 could never fire.
- **Takeoff alert hold** (`main.py` `hold_alerts_outside_envelope`): fault and failure-mode alerts are held
  while the AI's 128 s window still contains ground-roll samples below the 32 m/s dataset floor. Removes the
  ~60 s false misfire/combustion alarm the 916 raised after every takeoff; health and RUL still display.

**Open, found by the plant-mismatch evaluation (see model cards):**

- **Residual wear voting.** Temperature calibration offsets read as wear and push healthy oil-pressure and
  vibration channels into deviation. Three mechanical-anchored voting designs fixed that but raised false
  alarms on real flight data to 24-45% (model cards); the fix needs per-installation sensor calibration at
  engine fit (zero the residuals on a known-healthy ground run), not a voting rule.
- **Residual tolerances for real sensors.** Vibration (0.0005 g) and RPM (5 rpm) scales are sized to simulator
  noise; calibrate them on real bench data or widen them for hardware.
- ~~**Sensor vs engine attribution.**~~ Done: `RES_<CH>_SENSOR_SUSPECT` advisory item.
- ~~**RUL uncertainty band.**~~ Done: test-split MAE served as `rul_mae_hours` and shown in the cockpit.

**Closed (kept for history):**

- **Oil-pressure limit check.** `advisory.py` uses a 2.0 minimum while the physics reports ~30-90 (psi);
  align the limit with the unit.

## Remaining PS coverage

| Item | State |
|---|---|
| Edge deployment (TFLite + latency) | `validation/export_edge.py` written; run when the GPU is free |
| Federated learning demo | `validation/federated_demo.py` written; run when the GPU is free |
| Uncertainty on RUL | not possible with MC-dropout on this architecture; a deep-ensemble or quantile head is the option |
| Mission-preset durations | Fixed: `/mission` presets show flight time, not v2 "h real" equivalents |
| Offline post-flight summary | `summary.py` `local_summary`: deterministic report from the run digest when Groq is not configured or unreachable (no venue internet needed for summaries; sign-in and persistence still use Supabase) |
| One-command startup | `scripts/start_stack.sh`: AI service, physics backend and frontend in order with health checks, logs in `.logs/` |
| Real-time latency | Measured: sample-to-diagnosis p50 260 ms, p95 378 ms on the M2 laptop (`/health` `sim_status.ai_latency_ms`) |

## Operational rules

- Serve the models from `validation/venv` (Metal). CPU-only TensorFlow gives different answers.
- Run exactly one of `ai.py` / `aiv3.py`; `GET /state` must show matching `physics_version` and `ai_model_version`.
- Never run two TensorFlow training jobs at once on the 8 GB machine.
