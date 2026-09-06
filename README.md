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

> **Key design principle**: the Physics API (`:8000`) runs standalone. The AI service (`:8100`) is optional and separated because its Keras checkpoints require **Python 3.11 + TensorFlow 2.16.2**. If the AI service is offline, simulation and telemetry continue uninterrupted.

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
| `aero-sim-ai` | **3.11** | **conda** | conda base | AI inference service (`ai.py`) |
| Node.js | 18+ | `npm` | `frontend/` | Next.js dashboard |

---

## 1 — Physics API (Backend)

The backend uses its own venv. **Do not activate the AI conda env for this step.**

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

## 2 — AI Inference Service (Python 3.11 — conda)

The Keras model checkpoints in `backend/models/` were saved under Python 3.11 + TensorFlow 2.16.2. Loading them under a different Python version (e.g. 3.13 from the backend venv) causes a segmentation fault during deserialization.

### Create the conda environment (once)

```bash
conda create -n aero-sim-ai python=3.11 -y
conda activate aero-sim-ai
```

### Install AI runtime dependencies

```bash
cd backend
pip install -r requirements_ai.txt
```

`requirements_ai.txt` pins:

```
numpy==1.26.4
pandas==3.0.5
joblib==1.6.0
scikit-learn==1.9.0
tensorflow==2.16.2
keras==3.15.1
fastapi==0.141.1
pydantic==2.13.5
uvicorn[standard]==0.52.4
```

### Start the AI service

```bash
# from backend/, with aero-sim-ai conda env active
conda activate aero-sim-ai
cd backend
python ai.py
```

Health check: <http://localhost:8100/health>

A healthy response contains `"models_loaded": true` and lists the available engine model sets (912 ULS, 914 F/UL, 915 iS, 916 iS).

> **Note**: After starting, the AI service requires a 128-step warm-up window before predictions appear. The backend fast-forwards this window automatically on startup.

### Reactivate later

```bash
conda activate aero-sim-ai
cd backend
python ai.py
```

### Other useful conda commands

```bash
conda env list                     # list all environments
conda activate aero-sim-ai         # activate
conda deactivate                   # deactivate
conda remove -n aero-sim-ai --all  # delete environment
conda info --envs                  # same as conda env list, shows paths
```

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

Open **three terminals**:

**Terminal 1 — Physics API**
```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — AI Service**
```bash
conda activate aero-sim-ai
cd backend
python ai.py
```

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
| `GET` | `/health` | Model load status and available engine sets |
| `POST` | `/step` | Push a telemetry frame, returns latest prediction |
| `POST` | `/reset` | Clear the rolling input window |
| `POST` | `/select_engine` | Switch active engine model |

---

## Validation and Training

Training assets (datasets, notebooks, scripts) live in `validation/` and are **excluded from the repository** (see `.gitignore`). To run validation locally:

```bash
conda activate aero-sim-ai   # reuse the same environment
cd validation
pip install -r requirements.txt  # installs additional notebook/plot deps
```

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
