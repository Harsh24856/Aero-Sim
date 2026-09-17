# Briefing: Fix RUL/Health diagnostics, Continue Simulation, and UI polish

## Project context

UAV digital twin: a physics simulator (`backend/physics.py`) feeding a rolling-window
AI inference service (`backend/ai.py`) via `backend/main.py`, with a Next.js frontend
and Supabase for auth/persistence. Four engines: Rotax 912/914/915/916. 914 is the
primary/reference engine; all four now share the identical `ai.py` inference pipeline.

Read `PLAN_TOMORROW.md` in the project root first if it still exists — it documents
the AI rebuild plan and known contract details. Also read `backend/ai.py`'s own
docstring; it cites the exact notebooks and files that define ground truth.

## MCP servers to connect before starting

- **Filesystem** — read/write across `backend/`, `frontend/`, `validation/`
- **Supabase MCP** (project ref: check `frontend/lib/supabase.ts` or `.env.local`) —
  needed to query real `simulations`/`telemetry_logs` rows for verification, not just
  trust what the UI shows
- **Playwright/browser MCP** — required for task 3 (UI) and for verifying task 1
  end-to-end; do not consider a frontend fix done from code-reading alone
- A way to run Python (validation venv has TensorFlow; backend/.venv does not —
  **always run `ai.py` and any script that imports it via
  `validation/venv/bin/python3`**, never `backend/.venv/bin/python3`, or you will
  get a segfault or missing-TF import error that looks unrelated to your actual
  change)

## Task 1 (highest priority): Continue Simulation does not resume exact state

### Symptom
Clicking "Continue Simulation" on `/telemetry/[id]` calls `/resume` (backend restores
the twin from the saved `final_telemetry` snapshot — altitude, throttle, wear, RUL,
etc.), then navigates to `/simulate?engine=X`. But the resumed session does **not**
actually continue from that exact altitude/speed/time/RUL — it behaves like a fresh
takeoff.

### Root cause I found, NOT yet fixed
`frontend/app/simulate/page.tsx` and the `Simulator*.tsx` components have an
**auto-climb takeoff sequence**: whenever `started` transitions to `true`, the
frontend animates altitude from 0 up to a cruise target via repeated `POST /params`
calls, and separately ramps throttle/speed toward a target. This exists so a
*genuinely fresh* takeoff looks visually right.

The bug: this auto-climb sequence has **no awareness that a resume happened**. After
`/resume` correctly restores the backend twin's exact altitude/throttle/wear/RUL, the
frontend's own auto-climb logic immediately starts overwriting those values via
`/params`, dragging altitude back toward 0 and throttle/speed toward the default
ramp targets — destroying the just-restored state within the first few seconds.

This is a real, previously-identified-but-deferred issue (I flagged it in an earlier
session as a "known limitation," but it was never actually fixed). It is very likely
the single largest reason the resumed session "doesn't work."

### What needs to happen
1. Thread a "this is a resume" signal through: `/telemetry/[id]`'s Continue button →
   the navigation (`router.push('/simulate?engine=X&resumed=true')` or similar) →
   `/simulate/page.tsx` → the active `Simulator*.tsx` component.
2. When resumed, **skip the auto-climb/ramp sequence entirely**. The twin already has
   the correct altitude/throttle/speed from `/resume`'s response
   (`restored_altitude`, `restored_throttle`, `restored_airspeed`, `restored_wear` —
   check `main.py`'s `/resume` endpoint for the exact response shape). The frontend
   should initialize its local altitude/speed/throttle state from **those returned
   values**, not from 0 or a default, and should not send any `/params` calls until
   the user actually moves a control.
3. Also verify `session_active` handling: `/resume` sets `state["session_active"] =
   True` (confirmed present already). If the user pauses a resumed session and
   un-pauses via `/start`, `/start`'s freshness check
   (`if not state["session_active"]`) must correctly see it as already active and
   NOT recreate the twin. This was tested and confirmed working earlier in isolation
   — re-verify it still holds after your changes, since it's easy to accidentally
   regress by touching `/start` or `/resume`.
4. Also verify the AI side: does `/resume` reset `ai.py`'s rolling window? It calls
   `ai.py`'s `/select_engine`, which internally calls `engine_state.reset()` — this
   is correct and already relied upon elsewhere, but a stale AI buffer mixing
   pre-resume and post-resume data would produce exactly the kind of "weird
   RUL/health for the first ~128 seconds" symptom described in task 2. Confirm this
   reset actually clears the buffer for a resumed session (not just a fresh one).

### Verification required (do not consider this done without it)
- Start a real session, let it accumulate non-trivial wear/altitude/time, stop it
  (genuine stop, not pause).
- Query the actual Supabase row (`final_telemetry`) directly — note the exact
  altitude, throttle, airspeed, time, wear values.
- Click Continue Simulation from `/telemetry/[id]`.
- Immediately (within the first 1-2 seconds) check the live telemetry via
  `curl http://127.0.0.1:8000/state` or the frontend's own displayed values —
  altitude/throttle/airspeed should match the stored snapshot almost exactly, not
  be climbing from 0 or ramping from a default.
- Let it run ~10 more seconds and confirm RUL/health continue evolving sensibly from
  where they left off (not resetting to 100%/fresh).

## Task 2: RUL and health diagnostics still have issues

This project went through a long debugging cycle already (see `PLAN_TOMORROW.md` and
`backend/ai.py`'s docstring for the full history). Confirmed-fixed so far:
- RPM-fault health cap now requires ≥70% confidence (was firing on ~50% coin-flip
  classifications)
- `ai.py` rebuilt from scratch with a parity-tested `RollingWindow` (verified against
  `tf_data_pipeline.py`'s batch computation to ~1e-6 for all 4 engines)
- Severity divisors are fixed constants (`/5800`, `/85`), not per-engine — confirmed
  via reading all 4 phase5 training notebooks that they use the unmodified pipeline

**Before writing any new code**, reproduce the current symptom precisely:
1. Run a real session per engine (all 4), log `health_percent`, `rul_hours_internal`,
   `rul_percent_remaining`, and the full `diagnosis` dict from `ai.py`'s response at
   multiple points (nominal cruise, high throttle, right after a fresh start, right
   after a resume).
2. Compare against the offline parity/validation baseline already established:
   914 RUL MAE ≈0.31h with correlation ≈0.90 against real validation data (see
   `PLAN_TOMORROW.md` §2 for the exact numbers and how they were measured).
3. If live numbers diverge from what the model produces on real validation data fed
   through the identical code path, the bug is in the **live feature pipeline**
   (`RollingWindow` in `ai.py`, or what `main.py` sends it), not the model. Do not
   default to "retrain the model" — that was tried once already this project and was
   the wrong diagnosis (see the "I was wrong, models are fine" correction in project
   history — the actual bug was a `.predict()` vs direct-call discrepancy in a
   validation script).
4. Specifically check whether Task 1's auto-climb-overwriting-resume bug is itself
   the cause of some of the "weird diagnostics" symptoms — a resumed session with a
   corrupted altitude/throttle trajectory would plausibly produce exactly this kind
   of "RUL/health looks wrong" complaint. Fix Task 1 first, then re-evaluate whether
   Task 2 symptoms persist independently.

## Task 3: UI polish — `/telemetry`, `/telemetry/[id]`, login page, `/engine`

- `/telemetry` and `/telemetry/[id]`: already has real charts (recharts), real
  aggregated stats, and the (currently broken per Task 1) Continue button. Needs a
  visual pass — consistent spacing/typography/card treatment with the rest of the
  site (compare against `/dashboard` and `/engine_info`, which were already redone
  and can serve as the style reference).
- Login page ("lofi" — likely means the login page's current bare-bones/wireframe
  look): bring it to the same visual standard as the rest of the site.
- `/engine`: retouch to match site-wide design language. `/engine_info` was already
  substantially redesigned in this project and is a good reference for tone
  (dark theme, tertiary-orange accents, `bg-surface/80` cards,
  `border-outline-variant/30`, uppercase tracked labels). Don't just copy
  `/engine_info` wholesale — `/engine` has a different job (engine *selection*, with
  working 3D viewers and "Configure in Simulator" / "Launch" buttons that must keep
  working) — but the visual language should feel like the same site.

## Hard-won lessons from this project (read before touching `ai.py` or `physics.py`)

1. **Always call `.predict()`, never a direct `model(x)` call.** On this
   architecture these produce different results. This has caused two real
   incidents: broke live inference once when tried in `ai.py` directly, and produced
   a false "the model is broken" conclusion once in an offline validation script
   (apparent MAE of 87h vs the true 0.31h, purely from this).
2. **`tf_data_pipeline.py` is the specification, not a suggestion.** Its formulas
   (e.g., the hardcoded `power_kw / 85.0` divisor used for every engine regardless of
   real max power) look like they could be "improved," but doing so changes the
   model's input distrib and causes real regressions. Verified by reading all
   4 engines' training notebooks — none override it.
3. **Run `ai.py` and anything importing it via `validation/venv/bin/python3`.**
   `backend/.venv` lacks a working TensorFlow for this and will segfault or fail
   confusingly.
4. **Parity-test before trusting any change to feature computation.** The pattern:
   feed a real validation scenario through the live code path incrementally, compare
   against `tf_data_pipeline.py`'s own batch computation for the identical window.
   Anything under ~1e-5 difference is float32 noise; anything more is a real bug.
5. **A short/fresh session's `elapsed_hours` and `running_max_severity` sit at the
   extreme low edge of the training distribution** (training data has
   `elapsed_hours` ranging 0.036–5.54h with mean 2.49h; a fresh session starts at
   ~0). This is a genuine, currently-unresolved distributional mismatch, separate
   from any code bug — don't mistake it for one, but don't ignore it either. Options
   considered: warm-starting the twin's history, clamping aux features, or
   generating training data with more short/early-flight scenarios. No final
   decision was made — flag it if it's still visibly affecting results after fixing
   Task 1.
6. **Kill and verify lingering backend processes before testing.** Multiple bugs in
   this project were actually stale `main.py`/`ai.py` processes left running from a
   previous test session, silently answering new requests with old/orphaned state.
   `lsof -ti:8000` / `lsof -ti:8100` before every real test run.
7. **Verify claims with real data, not visual inspection.** Several apparent fixes
   in this project turned out to be wrong on closer inspection (e.g., a "fixed"
   dashboard chart that rendered axes correctly but showed zero data due to a
   separate bug). Query Supabase directly, curl the backend directly, don't trust
   that a screenshot looking plausible means the underlying data is correct.

## Suggested order of work

1. Fix Task 1 (auto-climb overwrite on resume) — verify end-to-end per the checklist
   above.
2. Re-test Task 2 symptoms now that Task 1 is fixed — many "diagnostics look wrong"
   reports may have been downstream of Task 1.
3. Only pursue further `ai.py` changes if Task 2 symptoms genuinely persist
   independent of Task 1, and only after reproducing precisely (not from user report
   alone).
4. Task 3 (UI) last, since it's independent and lower-risk.
