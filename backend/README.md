# UAV Digital Twin Backend

FastAPI service for the real-time UAV physics twin. It runs the simulation loop in the background and publishes telemetry to connected frontend clients over WebSocket.

## Requirements


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


The frontend expects this service on port `8000`, so start it before opening the frontend.

## Run the AI inference service (`ai.py`)

`ai.py` is a **separate process** from `main.py`. It listens on port `8100`,
holds the four trained model sets in memory, and answers one inference request
per simulated second. `main.py` calls it over HTTP and degrades gracefully if it
is down - telemetry keeps streaming, the AI fields just stop updating.

It runs under its **own** Python environment, not `backend/.venv`.

### Which environment (read this before changing anything)

Use `validation/venv`. Not `backend/.venv` (no working TensorFlow - it will
segfault or fail on import), and not a separate conda environment.

This is not a style preference. All four model sets were trained, saved and
validated under `validation/venv`, whose TensorFlow uses the **Metal GPU**
backend (`tensorflow-metal`). Loading the identical `.keras` files under a
CPU-only TensorFlow gives you the same weights and a different answer. Scored
against real `rul_true` labels on byte-identical validation windows:

| engine | RUL MAE (`validation/venv`) | RUL MAE (CPU-only env) | correlation (venv / CPU) |
| ------ | --------------------------- | ---------------------- | ------------------------ |
| 914    | 0.89 h                      | 235.99 h               | 0.42 / 0.21              |
| 912    | 0.35 h                      | 12.28 h                | 0.89 / 0.05              |
| 915    | 1.07 h                      | 513.21 h               | 0.94 / 0.81              |
| 916    | 0.57 h                      | 0.75 h                 | 0.58 / 0.74              |

True RUL never exceeds ~5.3 h in that data; the CPU-only path predicts up to
572 h. Launching this service from a CPU-only conda environment was the
confirmed cause of a real incident in which every "RUL and health look wrong"
symptom - 915 reporting ~515 h remaining, 914 flipping between 0 h and 21 h in
consecutive seconds, health jittering roughly 8x more than it should - traced
back to that and nothing else. Note 916 barely differs between the two, which is
exactly why its numbers looked plausible while 915's did not: a spot-check of one
engine will not catch this.

### Set the environment up

If `validation/venv` already exists, skip to *Start it*.

```bash
cd /Users/harsh/Documents/UAV_Engine/validation
python3.11 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r ../backend/requirements_ai.txt
```

Python **3.11** on Apple Silicon (arm64) macOS. `tensorflow-metal` has no wheel
for Intel Macs, Linux or Windows; on those platforms this service cannot
currently reproduce the validated numbers at all (see *Not on a Mac?* below).

Confirm the GPU backend is actually present before going further:

```bash
/Users/harsh/Documents/UAV_Engine/validation/venv/bin/python -c \
  "import tensorflow as tf; print([d.device_type for d in tf.config.list_physical_devices()])"
```

Expected: `['CPU', 'GPU']`. If you get `['CPU']`, `tensorflow-metal` is missing
or failed to load - fix that first, because nothing downstream will be correct.

### Start it

From the `backend/` directory (the module is imported as `ai`, so the working
directory matters):

```bash
cd /Users/harsh/Documents/UAV_Engine/backend
/Users/harsh/Documents/UAV_Engine/validation/venv/bin/uvicorn ai:app --host 127.0.0.1 --port 8100
```

Add `--reload` while developing. Start it **before** `main.py`, so the first
simulated second already has somewhere to send telemetry.

Startup takes a few seconds - all four engines' models are loaded eagerly, on
purpose, so no request ever pays a cold-load cost mid-flight. You should see:

```
Loading models for all registered engines...
  Rotax_914_ULF: loaded
  ...
All 4 engines loaded.
```

If instead you see a boxed `WARNING: TensorFlow sees no GPU backend`, stop: the
service will answer requests, but its RUL and severity numbers are not
trustworthy. Fix the environment.

### Check it

```bash
curl http://localhost:8100/health
```

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

`backend_validated` must be `true`. If it is `false`, you are in the wrong
environment and the diagnostics you are looking at are meaningless.

`buffer_fill` counts how much of the 128-step rolling window is filled. The
models need a full **128 simulated seconds** of history before they can predict
at all; until then `/step` returns `{"status": "warming_up"}`. `main.py`
fast-forwards through that during warmup rather than waiting 128 real seconds.

### Endpoints

| method | path             | purpose                                                     |
| ------ | ---------------- | ----------------------------------------------------------- |
| POST   | `/step`          | One raw telemetry timestep in, one inference result out.      |
| POST   | `/reset`         | Clear the rolling window. Call when a new flight starts.      |
| POST   | `/select_engine` | Switch active engine. Resets the window too.                  |
| GET    | `/health`        | Liveness, active engine, buffer fill, backend validity.       |

```bash
curl -X POST http://localhost:8100/select_engine \
  -H 'Content-Type: application/json' \
  -d '{"engine_model": "Rotax_915_iS"}'

curl -X POST http://localhost:8100/reset
```

`/step` expects every column in `ai.py`'s `FEATURE_COLS` plus `time`; that is
exactly what `main.py` sends, so it is rarely called by hand.

### Stopping and restarting

Stale processes holding port `8100` have caused real debugging dead-ends in this
project - an old process answers new requests with orphaned state. Always check:

```bash
lsof -ti:8100          # expect nothing before starting
kill $(lsof -ti:8100)  # if something is there
```

The same applies to `main.py` on port `8000`.

### Not on a Mac?

`tensorflow-metal` is Apple-Silicon-only, so the validated environment cannot be
reproduced elsewhere as-is. The models would need to be re-validated (and quite
possibly retrained) against whatever backend you deploy on, using real
`rul_true` labels - do not assume the weights transfer. `ai.py`'s startup guard
and `/health` will tell you honestly that the backend is unvalidated; treat that
as a blocker for trusting RUL and health, not a cosmetic warning.

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

After generating the database, use the validation scripts that are present in `validation/` as needed:

```bash
cd ../validation
python validate.py
python name_check.py
```

Training and evaluation notebooks in `validation/` cover preprocessing, model training, and Phase 8 evaluation. Some scripts and notebooks contain absolute paths beginning with `/Users/harsh/Documents/UAV_Engine`; update those paths before running the workflow on another machine.

## Project files

