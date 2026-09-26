# Q&A preparation

One-sentence answer first, then the evidence. Never claim more than the model cards show.

## Physics v4

Numbers are for the Rotax 914 on test flights the models never saw (`docs/model_cards_v4.md`). The 912, 915 and 916 are still training.

**What does the digital twin actually do?**
It flies a perfectly healthy copy of the engine on the same throttle, altitude and weather, every second. The AI reads the difference between what the sensors say and what the healthy engine would read. That difference is what separates "the engine is hot" from "it's a hot day at full power". Where the engine has clearly drifted from its twin, detection AUC is 0.952.

**How good is it at naming the failing part?**
Macro F1 is 0.612 over 13 component faults on the 914. It's strongest on bearing wear (F1 0.89), cooling (0.87), injector or fuel metering (0.96) and prop erosion (0.82). Each fault type has its own decision cut-off, chosen on validation flights.

**Why does it miss wastegate faults?**
Because a real 914 hides them too. Below the turbo's critical altitude (15,000 ft), the wastegate simply closes further and holds boost, so the fault leaves nothing to measure. The cockpit marks it "observable above 15,000 ft", and the demo flies above that to show it.

**Can it see turbo faults?**
Partly, and we show that. Flying above the critical altitude with a 43%-severity turbo fault, the model gives the turbo a median probability of 0.74. Its cut-off, chosen on validation flights, is 0.95, so it's never called, and it names valve leakage or air-filter fouling instead: faults that also starve the engine of charge air. We didn't lower the cut-off to make the demo pass. On the test flights, turbo degradation still scores F1 0.77 overall.

**Can it tell a broken sensor from a failing engine?**
It has a separate model for each of the 12 instruments. That model spots a faulty sensor 47% of the time and flags a healthy one only 0.9% of the time. When a sensor is flagged, the advisory stops trusting that channel, so a failed thermocouple can't ground a healthy engine. Dropouts are caught 81% of the time; a live CHT dropout was recognised on 56 of 57 samples with no false engine fault. A stuck sensor is the weak case: 37% on test flights, and a stuck EGT in the live check was missed entirely. Drift is caught 1% of the time: over 128 seconds it looks like a steady offset, and catching it needs a longer view (future work).

**How accurate is remaining useful life?**
Overall error is 4.6% of TBO. The hard case is engines that wear out before their overhaul date. There it's off by 387 h, where counting down to the overhaul date is off by 462 h: better than the calendar, but not yet the 15% of TBO we set as the bar. We report the version that uses only what an aircraft can measure. Fed the simulator's own wear labels, the same model scored 13.1%, but no aircraft has those labels.

**Why 128 seconds, and why once a second?**
The models were trained on one reading per second, and 128 readings cover the slow thermal response of the engine. So the AI always sees real seconds, even though the physics runs 100 times a second for the display. We measured the difference that makes: no input moves by more than 0.002 standard deviations.

**Engine hours move faster than the clock. Isn't that cheating?**
It's one declared constant: ×180, or 3 engine hours per real minute. It's derived from the training data: the fastest fault can grow at most 0.05 of its severity inside one 128-second window, below the point where a fault could appear from nothing inside a window the model is judging. Only engine life is sped up; the AI's inputs are never rescaled. Every run stores the scale it used.

**Why did you change the pass/fail gates?**
We kept them, and added honest ones. On the new data, 97.6% of severity values are zero and 97.8% of sensors are fine, so the old gates were passed by a model that predicts "nothing wrong" everywhere. Training now picks its best version on the severity error where a fault exists and on the per-condition sensor score. Those are the numbers we quote.

**Does the live system see the same thing the models were trained on?**
Yes, and it's proven: real test flights fed through the live service one row at a time give zero difference in the model inputs and outputs (`parity_ai_v4.py`). At start-up the service refuses to run if the twin, the training pipeline and the exported model disagree on the inputs.

**Why must it run on the GPU?**
We measured it: the same windows give very different answers on the CPU than on the Apple Metal GPU the models were trained on. So every reported number, the export and the live service run on the GPU, and the service says whether its device matches.

**Is the physics believable?**
It's calibrated to published Rotax figures (rated power, torque, fuel flow, idle, critical altitude, operating limits). An independent MATLAB/Simulink model of the 914, built from primitive blocks, matches it to about 2e-15 on every channel.

## Physics v3 (previous generation)

**Isn't the AI just learning your own simulator?**
Partly, and we tested exactly that: a separate aircraft process with a deliberately different engine talks to
the twin only over CAN, and the AI gave 0% false alarms under calibration offsets, unit-to-unit spread and
sensor noise, while a drifting sensor was caught. It breaks when the engine's thermal behaviour is different,
which is why the roadmap starts with recalibrating on real bench logs. (`docs/model_cards.md`, plant mismatch)

**You have no real engine data. Why should we trust any number?**
No labelled Rotax fault data is public, so the physics is calibrated to Rotax specifications (TBO, operating
bands, sensor ranges) and every metric is reported on held-out scenarios. The physics residual layer needs no
training data at all, and the pipeline (`validation/run.py`) retrains on real logs without code changes.

**Your diagnosis macro-F1 is only 0.62-0.67. Is that good enough?**
Detection (is something wrong) is 0.98-0.99 AUC; naming the exact fault type among five is harder, and every
channel's precision is measured and gated - a channel no better than chance (916 RPM) cannot move health or
raise an advisory.

**How accurate is remaining useful life?**
Mean absolute error 0.6-1.2% of TBO on held-out data (±8-23 engine hours), correlation above 0.99, monotonic;
on live flights 2-6% of TBO, and 15% when the plant engine's sensor gains differ.

**Is it real time?**
Sample to diagnosis p50 260 ms, p95 378 ms on an 8 GB laptop running physics, AI and the UI together; the AI
runs once per second, so every diagnosis is ready before the next sample.

**What happens when the AI fails?**
A circuit breaker isolates the AI service; physics, the residual layer, limit checks and the advisory keep
running and the UI says so. 15 automated tests cover these failure paths.

**How does it connect to a real UAV?**
Over CAN: set-point frames and sensor frames in the SocketCAN format, bridged to the twin. We run it over a
virtual bus on the laptop; on hardware the same code uses SocketCAN or any python-can adapter.

**Can this run on the aircraft (edge)?**
Yes: all five heads export to TensorFlow Lite. The classification heads are about 200 KB each in float16 at
0.37 ms per sample on one CPU thread, the RUL head about 780 KB at 0.53 ms, and float32 outputs match the full
model to within 0.0002 h. The physics residual layer is plain arithmetic and runs anywhere.

**Is it indigenous? You use Rotax, TensorFlow and a cloud LLM.**
Rotax engines are the ones flying Indian MALE-class UAVs today; the framework is engine-agnostic - a new engine
is a parameter set plus retraining. The live path uses no cloud service: the LLM summary is optional and
falls back to a local report generator.

**Why four separate models instead of one?**
Each engine has a different power band, propeller and TBO; per-engine models are smaller, individually
validated, and a new engine never degrades an existing one.

**What would you do with six more months?**
Bench logs from a Rotax-class engine to recalibrate physics and fine-tune the models; hardware-in-the-loop
on a real CAN bus; a RUL uncertainty model beyond the error band; fleet-level maintenance scheduling.
