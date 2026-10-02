# AERO-SIM — UAV Engine Digital Twin

Full-stack UAV propulsion simulation and diagnostics platform. Combines a real-time physics engine, a Next.js cockpit dashboard, live telemetry streaming, automatic fault modelling, and a neural AI inference service for fault detection, per-channel diagnosis, severity scoring, and remaining useful life (RUL) estimation.

---

## Architecture Overview

```
Browser  ─── Next.js (port 3000)
                │  REST + WebSocket
         FastAPI Physics API (port 8000)  ──── Supabase (optional)
                │  one feature window per simulated second
         FastAPI AI Service (port 8100)
                │  TensorFlow/Keras heads (detection, diagnosis, severity,
                │  failure modes, RUL)
              backend/models_v3/   (legacy v2: backend/models/)
```

> **Key design principle**: the Physics API (`:8000`) runs standalone. The AI service (`:8100`) is optional and separated because its Keras checkpoints require **Python 3.11 + TensorFlow 2.16.2 with the Metal GPU backend**. If the AI service is offline, simulation and telemetry continue uninterrupted.

> ⚠️ **The AI service must run from `validation/venv`.** Not a conda env, not `backend/.venv`. The checkpoints were trained and validated against that environment's Metal GPU TensorFlow, and running them CPU-only returns the same weights with different — and badly wrong — answers. See [§2](#2--ai-inference-service-python-311--validationvenv).

---

## Repository Layout

| Path | Purpose |
|---|---|
| `frontend/` | Next.js 16 App Router cockpit dashboard — see [frontend/README.md](frontend/README.md) |
| `backend/main.py` | FastAPI physics API — simulation loop, REST endpoints, WebSocket stream |
| `backend/physics.py` | Engine, propeller, aerodynamics, thermal, oil, vibration, wear, and fault models |
| `backend/aiv3.py` | FastAPI AI inference service for physics v3 (port 8100) — five heads |
| `backend/ai.py` | The legacy v2 inference service. Same port; run one or the other |
| `backend/failure_modes.py` | Engine failure modes (misfire, injector fouling, cooling, combustion) |
| `backend/residual.py` | Model-free physics residuals: measured − expected, with sensor zeroing |
| `backend/advisory.py` | Deterministic maintenance advisory rules over the AI + residual outputs |
| `backend/can_bus.py`, `aircraft_sim.py`, `can_ingest.py` | SocketCAN plant → bridge → `/measured` |
| `backend/models_v3/<key>/` | Deployed per-engine heads + scaler + `manifest.json` (physics v3) |
| `backend/models_v3/<key>/legacy_rul/` | The superseded RUL head, kept for comparison and rollback |
| `backend/models/` | Deployed per-engine heads for the legacy v2 stack |
| `backend/generate_dataset_v3.py` | Scenario dataset generator (detection/diagnosis/severity/failure modes) |
| `backend/generate_rul_dataset.py` | RUL probe dataset — one engine, one frozen life stage, one window |
| `backend/db.py` | Optional Supabase persistence layer |
| `backend/requirements.txt` | Physics API runtime dependencies |
| `backend/requirements_ai.txt` | AI inference runtime — **must match validation environment** |
| `scripts/` | `start_stack.sh`, `demo_can.sh`, `train_rul_v3.sh`, `rul_live_ab.py` |
| `validation/` | Training/eval code and notebooks are committed; datasets, logs and trained artifacts stay local (gitignored) |
| `docs/` | Architecture, model cards, demo script, deployment roadmap, Q&A prep |

---

## Environment Summary

| Environment | Python | Manager | Location | Purpose |
|---|---|---|---|---|
| `backend/.venv` | 3.x | `python -m venv` | `backend/` | Physics API (`main.py`, `db.py`) |
| `validation/venv` | **3.11 + Metal** | `python3.11 -m venv` | `validation/` | AI inference service (`aiv3.py` / `ai.py`) **and** all validation/training work |
| Node.js | 18+ | `npm` | `frontend/` | Next.js dashboard |

The AI service and the validation notebooks deliberately share **one** environment.
That is what keeps live inference numerically identical to the numbers the models
were validated against — a second, separate AI environment is exactly how they
drifted apart before (see §2).

---

## 1 — Physics API (Backend)

The physics API uses its own venv, separate from the AI service's. **Do not run `main.py` from `validation/venv`, or `ai.py` from `backend/.venv`** — each has exactly one correct environment.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start the physics API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: <http://localhost:8000/docs>

### Backend environment variables

Create `backend/.env` (never commit this file):

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-key        # optional: post-flight summaries
```

Full variable list (incl. `AERO_PHYSICS_VERSION`, `AERO_AI_URL`): [backend/README.md](backend/README.md).

Without these variables the simulator still runs in full — only run history persistence is disabled.

---

## 2 — AI Inference Service (Python 3.11 — `validation/venv`)

Run this service from **`validation/venv`**. Two separate reasons, both confirmed
by real incidents in this project:

1. **Python version.** The checkpoints in `backend/models/` were saved under
   Python 3.11 + TensorFlow 2.16.2. Loading them from `backend/.venv` (Python
   3.13, no working TensorFlow) segfaults during deserialization.
2. **TensorFlow backend.** `validation/venv` uses the **Metal GPU** backend via
   `tensorflow-metal`. The models were trained *and* validated there. Loading the
   identical `.keras` files under a CPU-only TensorFlow gives you the same weights
   and a materially different answer — not float noise.

Scored against real `rul_true` labels on byte-identical validation windows:

| engine | RUL MAE (`validation/venv`) | RUL MAE (CPU-only env) | correlation (venv / CPU) |
|---|---|---|---|
| 914 | 0.89 h | **235.99 h** | 0.42 / 0.21 |
| 912 | 0.35 h | 12.28 h | 0.89 / 0.05 |
| 915 | 1.07 h | **513.21 h** | 0.94 / 0.81 |
| 916 | 0.57 h | 0.75 h | 0.58 / 0.74 |

True RUL never exceeds ~5.3 h in that data; the CPU-only path predicts up to
**572 h**. Running this service from a CPU-only conda environment was the
confirmed cause of a live incident in which 915 reported ~515 h remaining, 914
flipped between 0 h and 21 h in consecutive seconds, and health jittered roughly
8× more than it should. Note that 916 barely differs between the two backends —
which is precisely why spot-checking one engine did not catch it.

### Create the environment (once)

Requires **Python 3.11 on Apple Silicon (arm64) macOS** — `tensorflow-metal` has
no wheel for Intel Macs, Linux or Windows.

```bash
cd validation
python3.11 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r ../backend/requirements_ai.txt
```

`requirements_ai.txt` pins:

```
numpy==1.26.4
pandas==3.0.5
joblib==1.5.3
scikit-learn==1.9.0
tensorflow==2.16.2
tensorflow-metal==1.2.0      # load-bearing — see the table above
keras==3.15.1
ml-dtypes==0.3.2
h5py==3.14.0
fastapi==0.141.1
pydantic==2.13.5
uvicorn[standard]==0.52.4
```

Verify the GPU backend is actually present before going any further:

```bash
./venv/bin/python -c "import tensorflow as tf; print([d.device_type for d in tf.config.list_physical_devices()])"
```

Expected `['CPU', 'GPU']`. If you get `['CPU']`, `tensorflow-metal` is missing or
failed to load — fix that first, because nothing downstream will be correct.

### Start the AI service

From `backend/` (the module is imported as `aiv3`, so the working directory matters):

```bash
cd backend
../validation/venv/bin/uvicorn aiv3:app --host 0.0.0.0 --port 8100
```

No environment activation needed — calling the venv's `uvicorn` directly is
enough, and it removes any chance of a stray `conda activate` selecting the wrong
interpreter.

Health check: <http://localhost:8100/health>

```json
{
  "status": "alive",
  "active_engine": "Rotax_914_ULF",
  "available_engines": ["Rotax_914_ULF", "Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"],
  "buffer_fill": 0,
  "tf_devices": ["CPU", "GPU"],
  "backend_validated": true
}
```

**`backend_validated` must be `true`.** If it is `false` you are in the wrong
environment and every RUL/health number you are looking at is meaningless. The
service also prints a boxed warning at startup in that case; it still answers
requests rather than refusing, so the warning is the only signal.

> **Note**: the models need a full 128 **simulated seconds** of history before
> they predict at all — until then `/step` returns `{"status": "warming_up"}` and
> `buffer_fill` counts up to 128. `main.py` fast-forwards that window instead of
> waiting 128 real seconds.

### Stopping and restarting

Stale processes holding a port have caused real dead-ends here — an old process
answers new requests with orphaned state. Check before starting:

```bash
lsof -ti:8100 -sTCP:LISTEN          # expect nothing
kill $(lsof -ti:8100 -sTCP:LISTEN)  # if something is there
```

Use `-sTCP:LISTEN`. A bare `lsof -ti:8100` also lists *clients* connected to that
port — which includes `main.py`, so killing that set takes the physics API down
with it.

---

## 3 — Frontend (Next.js)

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:3000>.

### Frontend environment variables

Create `frontend/.env.local` (never commit this file). Copy from `frontend/.env.example`:

```dotenv
# Supabase — required for auth, login, and run history
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-publishable-or-anon-key

# EmailJS — required for the contact form on /about_us
NEXT_PUBLIC_EMAILJS_SERVICE_ID=service_xxxxxxx
NEXT_PUBLIC_EMAILJS_TEMPLATE_ID=template_xxxxxxx
NEXT_PUBLIC_EMAILJS_PUBLIC_KEY=your_public_key
```

| Variable | Where to get it |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project → Settings → API |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase project → Settings → API → anon/public key |
| `NEXT_PUBLIC_EMAILJS_SERVICE_ID` | [emailjs.com](https://emailjs.com) → Email Services |
| `NEXT_PUBLIC_EMAILJS_TEMPLATE_ID` | EmailJS → Email Templates |
| `NEXT_PUBLIC_EMAILJS_PUBLIC_KEY` | EmailJS → Account → Public Key |

> EmailJS variables are optional — the contact form will show a configuration error if they are missing but all other pages work normally.

---

## Full Start-Up Order

**One command** (after the three environments are set up):

```bash
scripts/start_stack.sh
```

It starts the AI service, physics backend and frontend in order, waits for each health check,
writes logs to `.logs/`, and stops everything on Ctrl-C. Open <http://localhost:3000>.
`AERO_PHYSICS_VERSION=v2 scripts/start_stack.sh` runs the legacy stack.

Or by hand, in **three terminals**:

**Terminal 1 — Physics API**
```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — AI Service** (start this before Terminal 1, so the first simulated second has somewhere to go)
```bash
cd backend
../validation/venv/bin/uvicorn aiv3:app --host 0.0.0.0 --port 8100
```
Confirm `"backend_validated": true` at <http://localhost:8100/health> before trusting any diagnostics.

**Terminal 3 — Frontend**
```bash
cd frontend
npm run dev
```

Then open <http://localhost:3000>.

---

## Dashboard Workflow

1. Sign in (if Supabase is configured) or use the simulator directly.
2. Navigate to **Simulator** → select a Rotax engine (912 ULS / 914 F/UL / 915 iS / 916 iS).
3. Set altitude, throttle, airspeed, and angle of attack.
4. Start the simulation — live telemetry streams at ~20 Hz.
5. Monitor engine, propeller, aerodynamic, thermal, oil, vibration, fault, and AI diagnostic panels.
6. Stop a run to persist its final state to Supabase, or resume a saved run from history.

> Faults emerge from operating conditions. Sustained high power, excessive RPM, or poor thermal management increases fault stress organically — no manual fault injection required.

---

## API Quick Reference

### Physics API — port 8000

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/select_engine` | Select engine model, clears current session |
| `POST` | `/start` | Start or resume the simulation loop |
| `POST` | `/stop` | Stop (`{"final": true}`) or pause (`{"final": false}`) |
| `POST` | `/reset` | Reset all physics and AI rolling state |
| `POST` | `/params` | Update `altitude`, `throttle`, `airspeed`, `aoa` |
| `GET` | `/state` | Current parameters, telemetry, and latest AI result |
| `GET` | `/engines` | Engine list and AI-ready status |
| `POST` | `/resume` | Restore a saved simulation for its owner |
| `POST` | `/measured` | Real sensor values from the CAN bridge |
| `POST` | `/residuals/zero` | Re-zero residual offsets |
| `POST` | `/summarize/{sim_id}` | Post-flight summary (needs `GROQ_API_KEY`) |
| `GET` | `/health` | Liveness |
| `WS` | `/ws` | Live telemetry stream at ~20 Hz |

```bash
curl -X POST http://localhost:8000/params \
  -H 'Content-Type: application/json' \
  -d '{"altitude": 2000, "throttle": 0.65, "airspeed": 40, "aoa": 5}'
```

### AI Service — port 8100

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Active engine, available sets, `buffer_fill`, and `tf_devices` / `backend_validated` |
| `POST` | `/step` | Push a telemetry frame, returns latest prediction |
| `POST` | `/reset` | Clear the rolling input window |
| `POST` | `/select_engine` | Switch active engine model (also clears the window) |

```bash
curl http://localhost:8100/health   # backend_validated must be true
```

#### Fly the aircraft over CAN

The twin can monitor an engine it does not simulate itself. With the stack running, open the
simulate page for the engine (e.g. `/simulate?engine=Rotax_914_ULF`) and press Start, then:

```bash
cd backend
./.venv/bin/python can_ingest.py --bus udp            # bridge: CAN frames -> /params + /measured
./.venv/bin/python aircraft_sim.py --engine Rotax_914_ULF --profile mission --mismatch none
```

The Diagnostics panel shows **CAN LIVE** while sensor frames arrive. `--mismatch calibration |
engine_spread | noisy_sensors | sensor_drift | v2_plant` flies an engine that differs from the twin;
`validation/mismatch_eval.py` scores those profiles (run it with no flight live). On Linux or real
hardware use `--bus socketcan:can0` (needs `pip install python-can`) on both sides.

#### Physics v3 (default) and legacy v2

Physics v3 is the default: `main.py` runs v3 physics and expects `backend/aiv3.py`
(models in `backend/models_v3/`, RUL in real engine hours, engine failure modes).
The legacy v2 stack is still available. Run **either** `ai.py` **or** `aiv3.py` - both bind 8100 -
with the matching physics version:

```bash
# Legacy v2: AI service and physics backend (from backend/)
cd backend && ../validation/venv/bin/uvicorn ai:app --host 127.0.0.1 --port 8100
AERO_PHYSICS_VERSION=v2 uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`GET /state` on port 8000 reports `physics_version` and `ai_model_version`; they must match.
v3 responses carry `model_version`, `rul_hours` / `tbo_hours` (engine hours) and `failure_modes`,
and have no `rul_hours_internal`. v3 runs persist through `backend/dbv3.py` with
`simulations.model_version = 'v3'`.

#### Physics v4 (opt-in until its end-to-end check passes)

Physics v4 (`backend/physics_v4.py`) runs a flown engine plus a healthy on-board twin, 12 instruments with
their own faults, component-level degradation, and two clocks: flight time in real seconds and engine
hours at x180 (`backend/timescale_v4.py`). Its AI service is `backend/aiv4.py`, serving `backend/models_v4/`.
Engines that have no export yet run on the 914's models and say so (`placeholder_models`). Plan and status:
[docs/v4_integration_plan.md](docs/v4_integration_plan.md); results: [docs/model_cards_v4.md](docs/model_cards_v4.md).

```bash
AERO_PHYSICS_VERSION=v4 scripts/start_stack.sh          # aiv4 :8100, physics :8000, frontend :3000
```

The v4 weights must run on the Metal GPU they were trained on (`validation/venv`); the CPU gives different
answers. Extra endpoints on port 8000: `GET /scenarios` (demo presets), `POST /scenario {name}`,
`POST /inject {kind: fault|sensor, ...}`; `/start` accepts `engine_id` to continue a saved engine from its
hour meter. v4 runs persist through `backend/dbv4.py` (tables `engines`, `maintenance_events`, and v4
columns on `simulations` / `telemetry_logs`).

Finalising an engine after its training (from `validation_v4/`, GPU):

```bash
../validation/venv/bin/python3 export_deployable_v4.py --engine 912 --gpu   # cut-offs, live-input RUL score, export
../validation/venv/bin/python3 retrain_rul_live_v4.py --engine 912          # RUL head on live inputs
../validation/venv/bin/python3 export_deployable_v4.py --engine 912 --gpu   # export again with it
../validation/venv/bin/python3 parity_ai_v4.py --engine 912                 # live window == training
cd ../backend && ../validation/venv/bin/python3 ../scripts/e2e_v4.py --engines 912
```

Backend tests for v4 (from `backend/`): `.venv/bin/python -m unittest tests.test_twin_v4 tests.test_timescale_v4 tests.test_dbv4 tests.test_ai_v4_contract`.

#### Physics v5 (opt-in; the 914 has its own models, other engines run on the 914's)

Physics v5 (`backend/physics_v5.py`, `backend/twin_v5.py`) has 14 instruments, a residual on every one,
and the flight's long-horizon context (60-minute trends). Its AI service is `backend/aiv5.py`, serving
`backend/models_v5/<engine>/`: five specialist networks, one per question (detection, diagnosis and fault
family, severity, sensor fault, health), assembled with calibrated cut-offs, stacking models and a separate
RUL model. It answers in `aiv4`'s format plus fault families. v5 flights reuse v4's engine records,
scenarios, clocks and storage.

```bash
AERO_PHYSICS_VERSION=v5 scripts/start_stack.sh          # aiv5 :8100, physics :8000, frontend :3000
```

The 914's v5 scorecard (900 held-out test flights) is `validation_v5/artifacts/914b_specialists/model_card.md`.
v5 models do **not** need the Metal GPU: the CPU gives the same answers
(`validation_v5/tests/test_export_parity_v5.py`), and `AERO_AI_DEVICE=cpu` is a valid choice.

Training and shipping an engine (from the repo root, `validation/venv`):

```bash
validation/venv/bin/python validation_v5/relabel_v5b.py 914                    # v5b labels -> cache/914b
validation/venv/bin/python validation_v5/specialists_v5.py 914b                # ~8 h on the GPU
validation/venv/bin/python validation_v5/export_v5.py 914b                     # -> backend/models_v5/914
validation/venv/bin/python validation_v5/tests/test_export_parity_v5.py 914b   # export == training
cd backend && ../validation/venv/bin/python3 ../scripts/e2e_v5.py --engines 914
```

Backend tests for v5 (from `backend/`): `.venv/bin/python -m unittest tests.test_twin_v5` and, with the
models, `../validation/venv/bin/python -m unittest tests.test_ai_v5_service`.

**Why v3/v4 are "GPU only".** tensorflow-metal computes ReLU after a dense layer *wrongly* inside a compiled
TensorFlow graph (eager mode and the CPU are exact). v3/v4 were trained and are served through the compiled
path, so they learned that miscomputed function, and the CPU, computing the true one, disagrees. v5 writes
ReLU as `max(z, 0)` (`validation_v5/model_architectures_v5.py`), which is exact everywhere;
`validation_v5/tests/test_graph_parity_v5.py` guards it.

---

## Validation and Training

Training code and notebooks live in `validation/`; datasets, logs and trained artifacts there are **excluded from the repository** (see `.gitignore`). They run in the *same* `validation/venv` that serves `ai.py` — that shared environment is what keeps live inference numerically identical to the validated numbers.

```bash
cd validation
./venv/bin/python -m pip install -r requirements.txt  # notebook/plot deps on top of the AI runtime
```

Anything that imports `ai.py` / `aiv3.py` — including one-off scripts — must be run with
`validation/venv/bin/python3`, never `backend/.venv/bin/python3`, which will
segfault or fail on a missing TensorFlow in a way that looks unrelated to your
actual change.

Some scripts contain machine-specific paths — update those before running.

### Training the RUL head (physics v3)

The RUL head trains on its own dataset, for a measured reason: in the scenario data the
label moves by a median of 15.8 h *inside a single 128-sample window*, which is the same
size as the head's own error, and only 1.6% of rows sit above 95% of TBO while every live
flight starts at ~99%. `backend/generate_rul_dataset.py` builds probes instead — one engine
held at one frozen life stage for one window, life stages drawn uniformly with a near-new
over-sample, and session age drawn independently of wear so the head cannot read the clock
instead of the sensors.

```bash
# ~36 min for all four engines, 40k probes each (5.1M rows of telemetry per engine)
validation/venv/bin/python3 backend/generate_rul_dataset.py --all --parallel --probes 32000

# train one engine at a time - 8 GB of RAM, and two TensorFlow jobs swap
scripts/train_rul_v3.sh 914

# ship it: --rul-phase swaps only the RUL head, the other four are untouched
validation/venv/bin/python3 validation/split_deployable_heads.py --engines 914 --rul-phase phase5_rul_v3
```

The notebooks (`validation/phase5_rul_v3_<key>.ipynb`) are generated from one template by
`validation/make_phase5_v3_notebooks.py`, so all four engines are scored by identical code.
Each starts its head at a ridge fit on the window statistics and keeps that fit if training
cannot beat it; the gates are MAE ≤ 2% of TBO, near-new bias within ±2%, correlation ≥ 0.95,
no decile inversions, beating the ridge, and still working with `bsfc_ratio` zeroed.

`scripts/rul_live_ab.py` scores two heads on live flights against the twin's own
`wear × tbo_hours`, which is how the deployed heads were compared with the ones they
replaced (7-29x lower error, and the near-new under-prediction of 3.7-6.2% of TBO gone).

---

## Production Build

```bash
cd frontend
npm run build
npm start
```

Before deploying:
- Replace the permissive backend CORS policy with your frontend origin.
- Configure HTTPS / WSS termination.
- Never expose `SUPABASE_SERVICE_ROLE_KEY` to the browser or a public repo.
- Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` as environment variables in your hosting provider (Vercel, etc).

---

## Further Reading

- [backend/README.md](backend/README.md) — module map, env vars, endpoints, datasets, tests
- [frontend/README.md](frontend/README.md) — routes, components, env vars
- [docs/architecture.md](docs/architecture.md) · [docs/model_cards.md](docs/model_cards.md) · [docs/deployment_roadmap.md](docs/deployment_roadmap.md) · [docs/demo_script.md](docs/demo_script.md)
- [docs/v4_integration_plan.md](docs/v4_integration_plan.md) — physics v4: plan, time model, database, status
- [docs/model_cards_v4.md](docs/model_cards_v4.md) — physics v4 model results
- <http://localhost:8000/docs> — live FastAPI docs (physics API)
- <http://localhost:8100/health> — AI service health check
