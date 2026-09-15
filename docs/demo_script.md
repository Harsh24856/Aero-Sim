# AERO-SIM demo script (7 minutes)

For the judges of PS 26054. Rehearse it end to end; record a full run as the backup video.

## Before the room (10 minutes before)

1. `scripts/start_stack.sh` - wait for "Open http://localhost:3000". Use `localhost`, not `127.0.0.1`.
2. Sign in, open `/simulate?engine=Rotax_914_ULF`, check the Diagnostics panel says "Waiting" (not an error).
3. Second terminal ready in `backend/` for the CAN part; do not start it yet.
4. Close everything else on the laptop (8 GB). Sound on, volume medium.
5. Backup: the recorded video on the desktop, and `docs/model_cards.md` open in a tab.

## 0:00 - The problem (30 s)

"A MALE UAV engine failure ends the mission and often the aircraft. DRDO asked for a digital twin that watches
a Rotax-class engine in real time, predicts faults and remaining life, and tells a maintainer what to do.
AERO-SIM does that for four real Rotax engines - 912, 914, 915 and 916."

## 0:30 - The twin flying (60 s)

- Press **Start** on the 914. Point at the sensor panel: RPM, EGT, CHT, oil pressure and temperature, vibration,
  fuel flow - all from a physics model of the engine running at 100 Hz.
- "The AI needs 128 seconds of history - that bar is it filling, by design, not a delay."
- When it comes online: **health ~99%, RUL ~1,890 engine hours ±23 h of a 2,000 h TBO**, maintenance advisory
  nominal. "±23 hours is the held-out test error."

## 1:30 - Break the engine (90 s)

- Drag throttle to **100%**. EGT and CHT climb to their sensor limits.
- Within seconds: **fault detected**, health falls, the advisory turns to **warning** with specific actions,
  the screen edge turns red.
- Point at **Physics Residuals**: "This layer has no AI in it. It compares every sensor with what physics says
  it should read, so it keeps working if the AI service is down - and it tells us whether the engine or a
  sensor is lying."
- Throttle back to 35%: health recovers, the red fades. "It recovers because the stress was operational, not
  permanent damage - but wear went up, and the engine-hour meter shows it."

## 3:00 - It is not just watching its own simulation (90 s)

"The fair question is: isn't the AI just predicting your own simulator? So we split them."

- Stop the flight, press Start again, then in the second terminal: `scripts/demo_can.sh`.
- A **separate process** now flies its own engine and speaks only **CAN frames** - the same SocketCAN frames a
  real engine bus carries. The twin's panel shows **CAN LIVE**.
- "The twin never sees that engine's internals. It gets set-points and sensor readings over the bus, runs its
  own physics, and judges the real engine against it."
- Numbers to say: 112,000 frames, 0 lost; on a full-power leg the aircraft's own EGT sensor failed, and the
  twin caught it from the bus alone.

## 4:30 - Evidence, not claims (60 s)

Show the model cards tab:

- Fault detection AUC **0.98-0.99** on all four engines; RUL error **0.6-1.2% of TBO**.
- Sample-to-diagnosis latency **p50 260 ms, p95 378 ms** on this laptop.
- Mismatch test: a plant engine deliberately different from the twin - calibration offsets, unit spread,
  noise - gives **0% AI false alarms**; a drifting CHT sensor is **named by the residual layer at +4 C**.
- "Where it breaks is documented too: a different thermal model breaks the twin - which is what should happen
  before a new engine type is calibrated."

## 5:30 - Maintainer view (45 s)

- Stop the flight -> **mission report**: post-flight summary (works offline with no LLM), final health, RUL,
  findings and recommendations.
- **Replay** page: scrub the mission, health and RUL at every instant.

## 6:15 - Close (45 s)

"Physics twin, five AI heads per engine, a model-free safety layer, CAN-bus integration and a plan to validate
on real engine logs. What we need from DRDO next is bench data from a Rotax-class engine - the pipeline is
built to recalibrate on it."

## If something goes wrong

| Symptom | Say / do |
|---|---|
| AI panel shows "AI service unavailable" | "The safety layer keeps physics and residuals running without the AI" - carry on; restart later. |
| Page will not load | Play the backup video from the same point. |
| CAN LIVE does not appear | Check the engine on the page matches `demo_can.sh`; otherwise skip to the model cards evidence. |
