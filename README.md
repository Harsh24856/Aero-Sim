# UAV Digital Twin

A full-stack UAV engine digital twin. The Next.js dashboard controls a real-time FastAPI physics simulation and displays live telemetry, fault risk, and—in the optional AI mode—model-driven detection, diagnosis, severity, and remaining-useful-life estimates.

## Project structure

- `frontend/` — Next.js dashboard (port `3000`)
- `backend/` — FastAPI simulation API and WebSocket server (port `8000`)
- `backend/models/` — trained model files used by the optional AI service
- `validation/` — local-only data preparation, experiments, and validation assets; intentionally excluded from Git

## Prerequisites

- Node.js `20.9.0` or newer with npm
- Python `3.10` or newer with pip

## Quick start

Open two terminals at the repository root.

### 1. Start the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The backend is available at <http://localhost:8000>, with API documentation at <http://localhost:8000/docs>.

### 2. Start the frontend

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:3000>. The dashboard connects to the backend REST API at `http://localhost:8000` and telemetry WebSocket at `ws://localhost:8000/ws`.

## Using the dashboard

1. Start both services and open the dashboard.
2. Select **Start** to run the simulation.
3. Adjust altitude, throttle, airspeed, and angle of attack.
4. Observe live engine, propeller, aerodynamic, fuel, thermal, oil, vibration, and fault telemetry.

The simulation automatically derives fault stress from operating conditions. Sustained high throttle/RPM or poor cooling conditions can raise fault risk; faults are not manually injected from the UI.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/start` | Start the simulation loop |
| `POST` | `/stop` | Stop the simulation loop |
| `POST` | `/reset` | Reset the physics twin |
| `POST` | `/params` | Update altitude, throttle, airspeed, and/or angle of attack |
| `GET` | `/state` | Read the current simulation state |
| `GET` | `/ws` | Receive live telemetry over WebSocket |

Example parameter update:

```bash
curl -X POST http://localhost:8000/params \
  -H 'Content-Type: application/json' \
  -d '{"altitude": 2000, "throttle": 0.65, "airspeed": 40, "aoa": 5}'
```

## Optional AI inference service

`backend/ai.py` runs separately on port `8100` because the trained TensorFlow/Keras models require the Python environment used for validation. The main backend continues serving physics telemetry if this service is unavailable.

When the compatible validation environment and model dependencies are available, run the service from `backend/` using that environment's Python executable:

```bash
/path/to/validation/venv/bin/python3 ai.py
```

Check its status at <http://localhost:8100/health>.

## Production frontend build

```bash
cd frontend
npm run build
npm start
```

## Notes

- `validation/` is intentionally ignored by Git, including generated datasets and its virtual environment.
- The current backend CORS policy allows all origins for local MVP development. Restrict it before deploying.
- For details specific to a component, see `frontend/README.md` and `backend/README.md`.
