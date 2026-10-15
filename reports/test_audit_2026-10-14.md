# Full test audit and plan — 2026-10-14

Commit under test: `780dbfa` (main, merged PR #11), run from a clean worktree. The
environments came from the main checkout: `backend/.venv` (Py 3.13), `validation/venv`
(Py 3.11, TF 2.16.2, `['CPU','GPU']`) and `frontend/node_modules`. The main checkout
holds another session's uncommitted work (a `calibrated_wear` twin and a 914b retrain),
and none of that was tested or touched.

## 1. Results

### Unit tests

| Suite | Env | Result |
|---|---|---|
| backend, each module in its own process (10 modules) | backend/.venv | **90/90 pass** (1 skip) |
| backend, all modules in one process | backend/.venv | **3 fail**: the known order leak, root cause below |
| `tests.test_ai_v5_service` | validation/venv | 8/8 pass |
| `validation_v5/tests/test_graph_parity_v5.py` (Metal ReLU guard) | validation/venv | pass |
| `test_relabel_v5b.py`, `test_rul_smoothing_v5.py` | validation/venv | pass (smoothing: raw 63.9 h to 50.9 h) |
| `test_export_parity_v5`, `test_pipeline_v5`, `test_specialists_v5` | — | **not run**: they need `validation_v5/cache/914b`, which the main checkout's WIP rebuilt as `twin: calibrated_wear`. The committed export is `fleet_wear`, so running them now would compare mismatched data |
| frontend `tsc --noEmit` | node | clean |
| frontend `next build` | node | passes with `.env.local`, **crashes without it** (see F4) |
| ruff F/E9/B over backend, validation_v5, scripts, simulation | — | no undefined names or syntax errors; 9 unused imports, 4 unused variables, 2 B008 |

**Order-leak root cause (confirmed).** `test_ai_v4_contract` imports `main` with no
`AERO_PHYSICS_VERSION`, so it loads as v4. `main` reads the version once, at import.
Later, `test_new_features` and `test_simulation_safety` do `setdefault("v3")`, but by then
they get the cached v4 `main`. Reproduced with
`AERO_PHYSICS_VERSION=v4 python -m unittest tests.test_new_features tests.test_simulation_safety`,
which gives the same 3 failures. Reordering the suite so the v3 tests run first passes 39/39.
The three failures:
- `MeasuredSensorsTest.test_bounds_overlay_and_staleness`: `KeyError: 'rpm_fault'`
- `TakeoffHoldTest.test_alerts_held_while_window_has_ground_roll_then_released`: `KeyError: 'settling'`
- `test_persistent_faults_halt_safely_and_start_resumes`: the loop never reaches `halted` within 5 s

These are v3-only behaviours, and nothing checks whether v4 (the default!) needs them.

### Integration tests (in-process flight: twin, AI window, models, advisory)

**`scripts/e2e_v5.py`, all engines.** Exit 0, "ALL PASS", but that verdict counts only engines with their own models.
- 914 (own models): **8/8**. About 15 ms per inference.
- 912 placeholder: 7/7.
- 915 and 916 placeholders: healthy engine flagged on 100% of samples, and both sensor presets fail (engine-fault calls 100%).
- Weak spots reproduced:
  - **W2**: with a sensor fault, RUL reads 826–896 h against 1,382 h true on 912, 914 and 916.
  - **W3**: EGT-stuck flagged on only 32/57 samples (56%) on the 914.

**`scripts/e2e_v4.py` (the current DEFAULT stack): 5 failures on engines with their own models.**
- EGT-stuck is **never detected** (0/57) on 912, 914, 915 and 916.
- 914 `turbo_high`: turbo degradation not named within 5 min.
- 915 `oil_pump`: not named within 5 min.
- RUL equals the calendar countdown in every fault preset, so no flight is judged wear-limited.

### System tests (live HTTP + WebSocket, real time)

Stack: `aiv5` :8100 (CPU), `main.py` v5 :8000, `next dev` :3000.

Scripted flight (`scratchpad/live_flight.py`): **14/14 checks pass.**
- Health, `/engines`, `/scenarios`, `/start`, `/params` and `/stop` all work.
- WebSocket stream works.
- The healthy flight calls no faults (health 92%, RUL 1,694 h).
- Injected `oil_pump_degradation` is named after 183 s, family `oil`. Advisory codes: `FAULT_OIL_PUMP_DEGRADATION`, `FAULT_OIL_DEGRADATION`, `RUL_WEAR_LIMITED`. Note that oil degradation is also advised.
- Clear works.
- Injected EGT-stuck is named after 133 s.
- 0 AI samples dropped. AI latency p50 36 ms, p95 121 ms. Max 1,089 ms was measured while e2e_v4 shared the machine.

Default v4 stack boot plus a short flight: works, `backend_validated: true`, AI latency
**p50 332 ms** (GPU), against v5's 36 ms.

Playwright sweep (playwright-skill): 12 routes × desktop 1440 and mobile 390.
- 0 console errors and 0 failed requests, apart from the expected `/nonexistent` 404.
- 0 horizontal overflow.
- `/mission`, `/dashboard` and `/telemetry` redirect guests to `/login`.
- The `/home` internal links all resolve.

Playwright MCP, live cockpit during a `bearing_wear` flight:
- Sensor tiles update from the WebSocket (RPM 5120 to 5110, oil 2.57 to 2.50 bar over 6 s).
- Diagnostics show FAULT 100% and health 26%.
- Console: only a dev StrictMode WebSocket warning and a `THREE.Clock` deprecation.

**Not tested: flying from the cockpit UI.** Starting a flight needs a Supabase sign-in
("Sign in to simulate"). That is the user's real account on a remote host, so it was not
used. A test account on a local Supabase would unblock it.

## 2. Findings (new, not already in plan.md)

| # | Area | Finding | Evidence |
|---|---|---|---|
| F1 | Models | The default live stack (v4) never detects EGT-stuck and misses 2 engine faults on own-model engines | e2e_v4 above |
| F2 | Tests | Backend suite is order-dependent; v3-only tests silently run against v4 | Root cause above |
| F3 | Tests | No single command runs every suite in the right venv, and there is no CI | `backend/README.md` says `pytest tests/`, but pytest is in neither venv |
| F4 | Frontend | `lib/supabase.ts` calls `createClient(undefined)` at import, so `next build` dies without env (`supabaseUrl is required`) and no CI can build | Build log |
| F5 | Frontend | Opening `/simulate` mid-flight shows "WAITING – CLICK START SIMULATION" over a live flight (`started` is page-local state), and STOP SIMULATION shows even when idle | `components/Simulator.tsx:734`, `:774` |
| F6 | Frontend, mobile | At 390 px the flight display collapses to a thin bar, the throttle/speed gauge text overlaps ("68 KT" over "(M)"), and the volume slider overlaps the hint text | `pw-mobile_simulate.png` |
| F7 | Frontend | `/engine` says "4 AI TWINS READY" and "AI READY" on every card, hard-coded. Under v5, 3 engines are placeholders that flag healthy engines | `app/engine/page.tsx:180` |
| F8 | Frontend | Login's "Continue without an account" leads to a cockpit that requires sign-in | `/simulate` guest gate |
| F9 | Frontend | `http://localhost:8000` is hard-coded in 5 pages, with no `NEXT_PUBLIC_API_URL` | `app/simulate/page.tsx:32` and others |
| F10 | Frontend | `Simulator_912/915/916.tsx` are copies of `Simulator.tsx` (~780 lines each, 38–80 lines differ). `BlackoutPlane`, `TurbofanIllustration` and `CombustionIllustration` are never imported | diff counts |
| F11 | Backend | AI warm-up fast-forward also broadcasts every step: **2,615 WebSocket frames in 5 s at ~10 KB, about 5 MB/s per client**. Steady state is a sane 20 Hz | live_flight + WebSocket probe |
| F12 | Docs | README still says "Physics v3 is the default" and points at `aiv3`/`models_v3`; `main.py` defaults to v4 | `README.md:366`, `:17`, `:173` |
| F13 | Security | CORS `*` on main, ai, aiv3, aiv4 and aiv5 (already flagged MVP-only in code) | `backend/main.py:111` |
| F14 | Setup | `backend/requirements.txt` could not be installed (numpy 1.26.4 pinned against scipy 1.18.1, which needs numpy 2), and pandas and pyarrow were missing | fresh venv, `ResolutionImpossible` |
| F15 | Setup | `requirements_ai.txt` lacked pyarrow, scipy and threadpoolctl, which `aiv5` and `validation_v5` import, so a fresh AI install could not run v5 | import list |
| F16 | Frontend | drei `<Environment preset="city">` downloads its HDR from raw.githack.com. If that is unreachable (offline demo, blocked network), `/engine`, `/simulate` and `/engine_info` crash with a runtime error | Playwright with the CDN blocked |
| F17 | Tests | `test_twin_v4` restore test compared two noisy CHT readings against a 5 C bound and failed about 1 run in 25 | 25-run loop |
| F18 | Frontend | `/engine` footer "Neural surrogates: 4 checkpoints loaded" was hard-coded, like F7 | `app/engine/page.tsx` |

## 3. Plan: eight phases, run one after another

Each phase starts only when the previous one's "Done when" holds. Phases 1-4 touch no
models. Phases 5-7 are the training work in plan.md, and they belong to the session that
owns the main checkout's uncommitted retrain, so agree on it before starting Phase 5.

### Phase 1: one green test command (about half a day)
1. Each test module that imports `main` sets its own `AERO_PHYSICS_VERSION` before the
   import, and fails fast if `main` is already loaded under another version (F2).
2. Add `scripts/test_all.sh`. It runs every backend test module in its own process with
   the right venv, then the cache-free `validation_v5` tests.
3. For each v3-only test, decide whether v4/v5 needs that behaviour: the `rpm_fault`
   overlay, the `settling` hold, halting on persistent faults. If v4/v5 needs it, log a
   bug for Phase 4. If not, label the test v3.
4. Run `ruff --fix` for the 9 unused imports and 4 unused variables.

**Done when:** `scripts/test_all.sh` runs green, the same in any order.

### Phase 2: CI on GitHub Actions (about 1 day)
1. Make the frontend build without secrets: create the Supabase client lazily (F4), and
   read the backend address from `NEXT_PUBLIC_API_URL` (F9).
2. Add a CPU requirements file for the v5 AI service without `tensorflow-metal`. v5 gives
   the same answers on any device.
3. Add the workflow:
   - backend tests and ruff;
   - v5 AI tests on a CPU runner;
   - frontend `tsc` and `next build` with dummy env;
   - the Playwright route sweep against `next start`, with the backend mocked.

   v3/v4 model tests stay local (they need the Metal GPU).

Skills: `github-actions-templates`, `playwright-best-practices`.

**Done when:** a pull request runs all of these and shows green.

### Phase 3: frontend bugs, each with a Playwright test (about 2 days)
1. **F5:** the cockpit reads `running` from `/state` when it loads, so it no longer shows
   "WAITING" over a live flight. STOP is hidden when nothing is flying.
2. **F6:** fix the mobile cockpit: the flight display height, the gauge text overlap and
   the volume slider.
3. **F7:** the engine page shows real AI readiness, including placeholder engines, from
   `/engines` and `/health`. The `engine_info` "V4 TRAINED" text gets the same treatment.
4. **F8:** let guests fly without saving, or rename "Continue without an account".
5. Every fix gets a kept Playwright test under `frontend/e2e/`, and the mobile tests use
   screenshot baselines.
6. Fix only, no redesign. HomeLanding and the other pages keep their look.

**Done when:** all Phase 3 tests pass in CI, and a phone-width check of `/simulate` looks right.

### Phase 4: backend and docs (about 1 day)
1. **F11:** cap WebSocket updates at 20 Hz while the AI window fills (today about 500 Hz
   and 5 MB/s per browser).
2. Fix any v4/v5 behaviour gaps logged in Phase 1, step 3.
3. **F12:** rewrite the README run instructions for v4 as the default and v5 as the
   option, and remove the stale v3 text.
4. **F13:** take the CORS origins from env.
5. Add `TestClient` tests for the main endpoints: `/start`, `/inject`, `/inject/clear`,
   `/stop`, `/state`. Skill: `fastapi`.

**Done when:** the live check script passes, and a WebSocket probe counts no more than
about 100 messages in 5 s from the start of a flight.

### Phase 5: 914 weak spots W1-W3 (plan.md; GPU, several hours each)
Baselines measured today:
- valve leakage recall 0.56;
- EGT-stuck flagged 56%;
- remaining life about 500 h low with a broken sensor.

Fix in training code only. Re-run the 3 cache-dependent model tests once the training
data and the export agree again.

**Done when:** the W1-W3 targets in plan.md are met, `e2e_v5 --engines 914` stays 8/8,
and the model card is updated.

### Phase 6: own models for the 912, 915 and 916 (plan.md Phase 7; about 13 h each)
Run `run_fleet.sh`, then export, the parity test and the 22-flight comparison against
the placeholder.

**Done when:** each engine beats its placeholder. Healthy engines are no longer flagged
(today 100% on the 915 and 916), and `e2e_v5` passes on all four engines.

### Phase 7: make v5 the default (plan.md Phase 6)
1. Run `e2e_v4` and `e2e_v5` on the same flights.
   - Today: v4 fails 5 presets (it never detects EGT-stuck).
   - v5's 914 passes 8/8 and answers about 9× faster.
2. If v5 wins on every engine, switch the default in `main.py` and `start_stack.sh`.
   v4 stays selectable.
3. Update the README, plan.md and a four-engine model-card summary.

**Done when:** a default `start_stack.sh` runs v5, and the live check and the Playwright
tests pass.

### Phase 8: cleanup
1. **F10:** merge `Simulator_912/915/916.tsx` into `Simulator.tsx` with a per-engine
   config (about 2,300 duplicate lines), then delete the 3 unused components. This waits
   until Phase 3 so its fixes land in one place.
2. Decide on the unmerged branch `claude/male-uav-digital-twin-validation-e0bbe4`.
3. Add `pytest-coverage`, and fill the largest gaps it shows.
4. Test cockpit flights from the UI: a local Supabase test account plus saved login state
   lets Playwright press Start and fly a full flight.

**Done when:** the Playwright tests still pass after the merge and the coverage report is in CI.

## Progress (2026-10-14, PR Harsh24856/Aero-Sim#15)

| Phase | State | Evidence |
|---|---|---|
| 1 One green test command | done | `scripts/test_all.sh`: 17 suites (the safety suite under v3, v4 and v5) pass; a mixed-version run now fails with a clear `ImportError` instead of silently testing the wrong physics |
| 2 CI | done | `.github/workflows/ci.yml`: backend + lint, v5 AI on Linux CPU TensorFlow, frontend types + production build without secrets + Playwright; all three jobs green. Also fixed F14, F15, F17 |
| 3 Frontend bugs | done | F5, F6, F7, F8, F16, F18 fixed; Playwright 40 tests (desktop + mobile), including an offline-CDN test and a mobile layout size check |
| 4 Backend and docs | done | F11: stream capped at 20 Hz in wall time (test: 1,389 frames/s without the cap, at most 25 with it); F12 READMEs; F13 `AERO_CORS_ORIGINS`; `tests/test_api_v5.py` (TestClient: engines, scenarios, start, inject, clear, stop) |

Still open from this audit: F1 (Phase 5-7, model work), F10 (Phase 8). F16 is contained, not removed:
self-hosting the HDR file would keep the reflections offline too.

## Phase 5 status (2026-10-14)

The weak-spot work this plan calls Phase 5 was already started by another session:
- branch `claude/v5-weak-spots` (`c5969a3`, 2026-10-03, not merged): the investigation found no training defect;
- `docs/superpowers/plans/2026-10-03-v6-weak-spots.md`: the v6 follow-up plan;
- the main checkout's uncommitted v6 work: the `calibrated_wear` twin, engine history and a 914 retrain. It was last touched 2026-10-04 and no session is running it.

Re-checked here on the committed v5 914 models (controlled flights, same seed):

| | Claim (2026-10-03) | Measured today | Holds? |
|---|---|---|---|
| W2 | Low RUL with a broken sensor is not caused by the sensor | 600 h engine (true 1,382 h): 814 h with no sensor fault; 826 h with EGT stuck, 875 h with CHT dropout | yes - the RUL model treats this healthy engine as wear-limited (about 570 h low) |
| W3 | EGT-stuck "53%" is window latency | first stuck call 130 s after onset (window 128 s); CHT dropout 5 s | yes |
| W1 | Shallow valve leaks are a physics ceiling | not re-checked (needs a test cache that matches the export) | - |

The v6 914 retrain (main checkout, not exported) against the live v5 914 on the test card:
- detection 0.965 -> 0.978;
- diagnosis F1 0.818 -> 0.842;
- valve-leak recall 0.56 -> 0.60;
- oil-pump recall 0.66 -> 0.72;
- RUL 9.5% -> 9.6%: W2 is not fixed yet, and the history inputs (v6 Task 4) are what targets it.

### Phase 5 outcome (2026-10-15)

- **v6 914 not accepted** (the v6 plan's own rule). The test card passes, as do export parity on GPU and CPU and `e2e_v5` (8/8). The 22-flight A/B fails:
  - exact fault naming 96.0% -> 93.7%;
  - RUL error 145 -> 225 h. v6 over-predicts life on every mid-fault engine and is better on sensor and old-engine cases.

  Parked with its evidence in Harsh24856/Aero-Sim#16; the live 914 stays on v5.
- **Bug fixed on the way:** the offline harness (`e2e_v5`, `ab_v5`) never sent `engine_history`, so every v6 A/B ran on neutral history. `twin_v5.AI_PAYLOAD` is now the one list (#16).
- **W3 shipped** (Harsh24856/Aero-Sim#17): a stuck sensor is called from its flat line 20 s after onset (was 130 s); EGT-stuck right kind 32 -> 54/57; 0 false calls. Confirmed live: 20 s, `source=flatline`.
- **Flaky runs explained:** every slow or flaky run tonight (a test file taking 245-603 s, the v5 safety suite failing once) coincided with the Mac's battery Maintenance Sleep (`pmset -g log`: ~603 s asleep, 21 s awake). It is not a code fault. Run long checks on power, or under `caffeinate -is`.

### Phase 5 closed (2026-10-15)

Done: W3 shipped (#17). Deferred with evidence: W1 (a physics ceiling) and W2 (needs a model change). Details are in Harsh24856/Aero-Sim#16.

- **Naming:** v6's drop is oil degradation only. It is called oil pump instead (same family); cut-off 0.39 vs 0.25. On the test set v6 is better, so this is a precision/recall trade, not a defect.
- **W2 RUL:** both models over-predict mid-fault engines (v5 +169 h, v6 +198 h).
  - About +20 to +80 h of that is expected: remaining life is convex in fault growth rate.
  - v6's extra error is the RUL regressor mis-reading faults absent from the logbook: +155 h on held-out windows whose labels average 170 h.
  - Three fixes (a two-point extrapolation as an input, as an override, and as a severity-gated override) were each rejected on validation and reverted.
  - What is needed: several logbook points, or a model of each fault's own removal time. Either changes the data contract, so the data must be regenerated and the models retrained.
- **Phase 8 cleanup:** one Simulator with per-engine scenery, pixel-identical (#18).

## 4. Skills (from `/find-skills`)

**Already installed and useful here:**
- `superpowers:test-driven-development` and `superpowers:systematic-debugging` (T1, T2)
- `playwright-skill` and the Playwright MCP connector (used today; reuse the sweep for T4)
- `snagly:e2e-codegen` and `snagly:visual-regression` (turn the sweep into kept tests and mobile baselines)
- `code-review` (each PR)

**Worth installing (verified install counts on skills.sh):**

| Skill | Installs | Use |
|---|---|---|
| `currents-dev/playwright-best-practices-skill@playwright-best-practices` | 93.2K | T2/T4 kept Playwright tests |
| `wshobson/agents@python-testing-patterns` | 35.5K | T1 test isolation and runner |
| `wshobson/agents@github-actions-templates` | 16.9K | T4 workflows |
| `github/awesome-copilot@pytest-coverage` | 12.8K | coverage gaps once tests run in one place |
| `fastapi/fastapi@fastapi` (official) | 12.2K | `TestClient` tests for main/aiv5 endpoints |

No well-established skill covers ML model regression or validation. The project's own
gates (`gates_v5.py`, `e2e_v5.py`, parity tests) are already stronger than anything listed.
