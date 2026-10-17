# AERO-SIM Backend

Two Python services plus the offline data generators. `AERO_PHYSICS_VERSION` picks the
physics in `main.py`, and the AI service on 8100 must be the matching one (run one of them):

| Process | File | Port | Environment |
|---|---|---|---|
| Physics API | `main.py` | 8000 | `backend/.venv` (`requirements.txt`) |
| AI inference, physics **v4 (default)** | `aiv4.py` | 8100 | `validation/venv` (`requirements_ai.txt`), Metal GPU |
| AI inference, physics v5 (the 914's own specialists; 912/915/916 on the 914's as stand-ins) | `aiv5.py` | 8100 | `validation/venv`, CPU by default |
| AI inference, physics v3 / legacy v2 | `aiv3.py` / `ai.py` | 8100 | `validation/venv`, Metal GPU |

`scripts/start_stack.sh` starts the matching pair (and the frontend):
`AERO_PHYSICS_VERSION=v5 scripts/start_stack.sh`. The physics API runs standalone. If the AI service is down, telemetry keeps streaming and the AI fields stop updating.

---

## Module map

| File | Role |
|---|---|
| `main.py` | FastAPI app: simulation loop, REST endpoints, `/ws` telemetry stream, AI + DB wiring |
| `twin_v4.py`, `physics_v4.py`, `sensors_v4.py`, `degradation_v4.py`, `scenarios_v4.py`, `timescale_v4.py` | Physics v4 (default): flown engine + on-board twin, instruments, wear and faults, demo presets, the flight/life clocks |
| `twin_v5.py`, `physics_v5.py`, `sensors_v5.py`, `degradation_v5.py`, `features_v5.py`, `observability_v5.py` | Physics v5: as v4, plus the fleet-wear on-board twin and the long-horizon context the v5 models read |
| `physics.py` | `UAVEngineTwin` — v3 and legacy v2 physics |
| `failure_modes.py` | Engine failure modes: misfire, injector fouling, cooling, combustion |
| `residual.py` | Model-free residuals: measured − expected, with sensor zeroing (`/residuals/zero`) |
| `advisory.py` | Deterministic maintenance advisories over AI + residual outputs |
| `safety.py` | Simulation safety limits / crash handling |
| `summary.py` | Post-flight natural-language summary via Groq (`/summarize/{sim_id}`) |
| `aiv4.py`, `aiv5.py`, `aiv3.py`, `ai.py` | AI services — 128 s rolling window → detection, diagnosis, severity, sensor faults, health, RUL |
| `db.py`, `dbv3.py`, `dbv4.py` | Optional Supabase persistence (v2 / v3 / v4 and v5 runs) |
| `can_bus.py`, `aircraft_sim.py`, `can_ingest.py` | CAN plant → bridge → `/params` + `/measured` |
| `bench.py` | Bench-test helpers used by `main.py` and tests |
| `generate_dataset_v3.py` | v3 scenario dataset generator |
| `generate_rul_dataset.py` | v3 RUL probe dataset (one engine, one frozen life stage, one window) |
| `generate_training_data.py`, `generate_unified_database.py`, `generate_multi_engine_data_v2.py` | Legacy v2 dataset generators |
| `models_v4/<key>/` | Deployed v4 heads (912, 914, 915; the 916 is served by the 914's) |
| `models_v5/<key>/` | Deployed v5 export (`validation_v5/export_v5.py`); 914 only so far |
| `models_v3/<key>/` | Deployed v3 heads, scaler, `manifest.json`; `legacy_rul/` keeps the superseded RUL head |
| `models/<key>/` | Deployed v2 heads + scaler |
| `tests/` | unittest suites; run them all with `scripts/test_all.sh` (section 5) |

Engine keys: `912`, `914`, `915`, `916` (Rotax 912 ULS, 914 ULF, 915 iS, 916 iS).

---

## 1. Physics API

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: <http://localhost:8000/docs>

### Environment variables (`backend/.env`, never committed)

| Variable | Default | Purpose |
|---|---|---|
| `SUPABASE_URL` | — | Run persistence (optional) |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Server-side Supabase key — never expose to the browser |
| `GROQ_API_KEY` | — | Post-flight summaries (optional) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Summary model |
| `AERO_PHYSICS_VERSION` | `v4` | Physics and AI pairing: `v4` + `aiv4.py` (default), `v5` + `aiv5.py`, `v3` + `aiv3.py`, `v2` + `ai.py` |
| `AERO_AI_URL` | `http://127.0.0.1:8100` | Where the AI service lives |
| `AERO_HEALTH_FAILURE_HOLD_S` | `20` | Seconds health must stay at the floor before the flight ends as an engine failure; `0` disables the shutdown |
| `AERO_CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API. `*` is for local use only |
| `AERO_AI_DEVICE` | `cpu` | `aiv5.py` only: `gpu` runs the v5 models on Metal (slower at batch 1) |

Without Supabase/Groq the simulator runs fully; only persistence and summaries are off.

### Endpoints (port 8000)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/select_engine` | Choose engine; clears the session |
| `POST` | `/start` | Start / resume the loop |
| `POST` | `/stop` | `{"final": true}` stop, `{"final": false}` pause |
| `POST` | `/reset` | Fresh twin + clear AI window |
| `POST` | `/params` | Any subset of `altitude`, `throttle`, `airspeed`, `aoa` |
| `POST` | `/resume` | Restore a saved run for its owner |
| `POST` | `/measured` | Feed real sensor values (CAN bridge) |
| `POST` | `/residuals/zero` | Re-zero residual offsets |
| `POST` | `/summarize/{sim_id}` | Generate the post-flight summary |
| `GET` | `/state` | Params, telemetry, latest AI result, `physics_version`, `ai_model_version` |
| `GET` | `/engines` | Engine list + AI readiness |
| `GET` | `/health` | Liveness |
| `WS` | `/ws` | Telemetry at ~20 Hz |

```bash
curl -X POST localhost:8000/params -H 'Content-Type: application/json' \
  -d '{"altitude": 2000, "throttle": 0.65, "airspeed": 40, "aoa": 5}'
```

Faults emerge from operating conditions (sustained high power/RPM, poor cooling) — there is no manual fault-injection endpoint.

---

## 2. AI inference service

**Must run from `validation/venv`** (Python 3.11, TensorFlow 2.16.2 + `tensorflow-metal`, Apple Silicon). The v2/v3/v4 heads were trained and validated on the Metal GPU backend; the same `.keras` files under CPU-only TensorFlow give materially wrong RUL (e.g. 915: 1.07 h MAE vs 513 h). See the root README for the full table. v5 is exempt: it builds ReLU as `max(z, 0)`, which computes the same on any device, and serves on the CPU by default.

```bash
cd validation
python3.11 -m venv venv
./venv/bin/python -m pip install -r ../backend/requirements_ai.txt
./venv/bin/python -c "import tensorflow as tf; print([d.device_type for d in tf.config.list_physical_devices()])"
# expect ['CPU', 'GPU']
```

Start it from `backend/` (module import path matters), before `main.py`:

```bash
cd backend
../validation/venv/bin/uvicorn aiv4:app --host 127.0.0.1 --port 8100   # with main.py's default (v4)
../validation/venv/bin/uvicorn aiv5:app --host 127.0.0.1 --port 8100   # with AERO_PHYSICS_VERSION=v5
```

`curl localhost:8100/health` must show `"backend_validated": true`. The models need 128 simulated seconds of history; until then `/step` returns `{"status": "warming_up"}` (`main.py` fast-forwards this).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/step` | One telemetry timestep in, latest prediction out |
| `POST` | `/reset` | Clear the rolling window |
| `POST` | `/select_engine` | Switch engine (also resets) |
| `GET` | `/health` | Active engine, `buffer_fill`, `tf_devices`, `backend_validated` |

Stale processes on a port answer with orphaned state — check first:

```bash
lsof -ti:8100 -sTCP:LISTEN
```

Always use `-sTCP:LISTEN`; a bare `lsof -ti:8100` also lists `main.py` as a client.

---

## 3. CAN bus flight

With the stack running and a flight started on `/simulate?engine=Rotax_914_ULF`:

```bash
./.venv/bin/python can_ingest.py --bus udp
./.venv/bin/python aircraft_sim.py --engine Rotax_914_ULF --profile mission --mismatch none
```

`--mismatch calibration | engine_spread | noisy_sensors | sensor_drift | v2_plant` flies an engine that differs from the twin. Real hardware: `--bus socketcan:can0` (needs `python-can`).

---

## 4. Datasets (write into `../validation/`, which is git-ignored)

```bash
# v3 scenario data (detection / diagnosis / severity / failure modes)
../validation/venv/bin/python3 generate_dataset_v3.py --help

# v3 RUL probes, ~36 min for all four engines
../validation/venv/bin/python3 generate_rul_dataset.py --all --parallel --probes 32000
```

Training, head export and scoring live in `validation/` — see the root README, *Validation and Training*.

---

## 5. Tests

From the repo root, every suite in its own process with the right environment:

```bash
scripts/test_all.sh                # backend, the v5 AI service, cache-free validation_v5
scripts/test_all.sh --with-cache   # plus the tests that read validation_v5/cache/914b
```

`main.py` reads `AERO_PHYSICS_VERSION` once, at import, so test modules for different
physics versions cannot share one interpreter: a module imported after `main` loaded
under another version fails with an `ImportError` that says so. One module alone:
`.venv/bin/python -m unittest tests.test_twin_v5` (TensorFlow suites:
`../validation/venv/bin/python -m unittest tests.test_ai_v5_service`).
