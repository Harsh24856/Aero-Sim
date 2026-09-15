# Q&A preparation

One-sentence answer first, then the evidence. Never claim more than the model cards show.

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
The heads are small (about 1.7 MB per engine); see the edge export report for TensorFlow Lite size and
latency. The physics residual layer is pure Python arithmetic and runs anywhere.

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
