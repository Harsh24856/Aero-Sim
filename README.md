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
                │  TensorFlow/Keras 4-head multi-task model
              backend/models/
```

> **Key design principle**: the Physics API (`:8000`) runs standalone. The AI service (`:8100`) is optional and separated because its Keras checkpoints require **Python 3.11 + TensorFlow 2.16.2 with the Metal GPU backend**. If the AI service is offline, simulation and telemetry continue uninterrupted.

> ⚠️ **The AI service must run from `validation/venv`.** Not a conda env, not `backend/.venv`. The checkpoints were trained and validated against that environment's Metal GPU TensorFlow, and running them CPU-only returns the same weights with different — and badly wrong — answers. See [§2](#2--ai-inference-service-python-311--validationvenv).

---

## Repository Layout

| Path | Purpose |
|---|---|
| `frontend/` | Next.js 14 App Router cockpit dashboard |
| `backend/main.py` | FastAPI physics API — simulation loop, REST endpoints, WebSocket stream |
| `backend/physics.py` | Engine, propeller, aerodynamics, thermal, oil, vibration, wear, and fault models |
| `backend/ai.py` | Standalone FastAPI AI inference service (port 8100) |
| `backend/models/` | Per-engine Keras 4-head checkpoints and scikit-learn scalers |
| `backend/db.py` | Optional Supabase persistence layer |
| `backend/requirements.txt` | Physics API runtime dependencies |
| `backend/requirements_ai.txt` | AI inference runtime — **must match validation environment** |
| `validation/` | Training datasets, notebooks, and model validation tools (local-only) |

---

## Environment Summary

| Environment | Python | Manager | Location | Purpose |
|---|---|---|---|---|
| `backend/.venv` | 3.x | `python -m venv` | `backend/` | Physics API (`main.py`, `db.py`) |
| `validation/venv` | **3.11 + Metal** | `python3.11 -m venv` | `validation/` | AI inference service (`ai.py`) **and** all validation/training work |
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
```

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

From `backend/` (the module is imported as `ai`, so the working directory matters):

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
and have no `rul_hours_internal`. v3 runs persist through `backend/dbv3.py`; the database has no
`model_version` column yet, so do not run v3 against production data.

---

## Validation and Training

Training assets (datasets, notebooks, scripts) live in `validation/` and are **excluded from the repository** (see `.gitignore`). They run in the *same* `validation/venv` that serves `ai.py` — that shared environment is what keeps live inference numerically identical to the validated numbers.

```bash
cd validation
./venv/bin/python -m pip install -r requirements.txt  # notebook/plot deps on top of the AI runtime
```

Anything that imports `ai.py` — including one-off scripts — must be run with
`validation/venv/bin/python3`, never `backend/.venv/bin/python3`, which will
segfault or fail on a missing TensorFlow in a way that looks unrelated to your
actual change.

Some scripts contain machine-specific paths — update those before running.

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

- [backend/README.md](backend/README.md) — data generation commands and backend notes
- [frontend/README.md](frontend/README.md) — frontend component notes
- <http://localhost:8000/docs> — live FastAPI docs (physics API)
- <http://localhost:8100/health> — AI service health check
