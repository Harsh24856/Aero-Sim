# UAV Digital Twin Backend

FastAPI service for the real-time UAV physics twin. It runs the simulation loop in the background and publishes telemetry to connected frontend clients over WebSocket.

## Requirements

- Python 3.10 or newer recommended
- `pip`
- macOS, Linux, or Windows with a shell that can activate a Python virtual environment

## Install

From this directory:

```bash
cd /Users/harsh/Documents/UAV_Engine/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the API

Start the development server from `backend/`:

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

The frontend expects this service on port `8000`, so start it before opening the frontend.

## API

### Simulation controls

```bash
curl -X POST http://localhost:8000/start
curl -X POST http://localhost:8000/stop
curl -X POST http://localhost:8000/reset
curl http://localhost:8000/state
```

`/start` begins the real-time loop. `/stop` stops it. `/reset` creates a fresh twin and preserves the running state if the simulation was running.

### Update parameters

Send any subset of the following JSON fields:

- `altitude` in metres
- `throttle` from `0` to `1` (the API clamps this value)
- `airspeed` in metres per second
- `aoa` in degrees

Example:

```bash
curl -X POST http://localhost:8000/params \
  -H 'Content-Type: application/json' \
  -d '{"altitude": 2000, "throttle": 0.65, "airspeed": 40, "aoa": 5}'
```

### Live telemetry

Connect to `ws://localhost:8000/ws` to receive JSON telemetry at approximately 20 Hz while the simulation is running. The first message contains the latest telemetry when one is already available.

Telemetry includes engine and propeller RPM, power, torque, fuel flow, thrust, lift, drag, margins, temperatures, oil readings, vibration channels, fault flags, and fault stress values.

Faults are derived automatically from operating conditions. There is no manual fault-trigger endpoint. Sustained high power/RPM or poor cooling can increase fault stress.

## Generate datasets

These jobs are optional and write output into `../validation/`.

Generate the approximately one-million-row training dataset:

```bash
python generate_training_data.py
```

Generate the unified Detection, Diagnosis, and RUL database:

```bash
python generate_unified_database.py
```

The unified database can be very large and may take a substantial amount of time. Both scripts use fixed random seeds for repeatable generation.

## Validation and preprocessing

Validation dependencies are separate from the API dependencies:

```bash
cd ../validation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the physics comparison against `validation/simulink_ground_truth.csv`:

```bash
python validate.py
```

After generating `unified_database.csv`, convert and split it with:

```bash
python validate_and_convert.py
python split_scenarios.py
python check_splits.py
```

Some validation scripts currently contain absolute paths beginning with `/Users/harsh/Documents/UAV_Engine`; update those paths before running the workflow on another machine. `convert_to_parquet.py` also reads `unified_database.csv` but writes `training_data.parquet`, so check the intended output before using it.

## Project files

- `main.py`: FastAPI routes, simulation loop, and WebSocket server
- `physics.py`: UAV physics and automatic fault model
- `generate_training_data.py`: randomized labeled training data
- `generate_unified_database.py`: failure, censoring, and RUL data
- `stability_check.py`, `stability_check2.py`: physics checks
- `bench.py`: backend benchmarking utility
