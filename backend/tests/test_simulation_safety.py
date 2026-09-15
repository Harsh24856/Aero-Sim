"""Safety-net tests for the live simulation loop (backend/main.py + backend/safety.py).

Standard library only (unittest) plus FastAPI's TestClient. The AI service is replaced
by an httpx MockTransport, so every failure mode is reproducible: down, slow, garbage,
and recovering. No test passes a user_id, so nothing is written to Supabase.

Run from backend/:
    .venv/bin/python -m unittest tests.test_simulation_safety -v
"""
import asyncio
import json
import math
import os
import sys
import threading
import time
import unittest

os.environ.setdefault("AERO_PHYSICS_VERSION", "v3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx                                     # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import main                                      # noqa: E402
import safety                                    # noqa: E402
from physics import UAVEngineTwin                # noqa: E402


def strict_json(text):
    """json.loads that fails on NaN/Infinity, exactly like the browser's JSON.parse."""
    def bad(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=bad)


class FakeAI:
    """Programmable stand-in for aiv3.py."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "ok"            # ok | down | slow | garbage | unsupported
        self.timeout_next_ok = False   # raise one ReadTimeout on the next would-be "ok" reply
        self.warmup_calls = 3
        self.slow_s = 2.0
        self.step_times = []
        self.resets = 0

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    async def handler(self, request: httpx.Request):
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "alive", "model_version": "v3"})
        if path == "/reset":
            with self.lock:
                self.resets += 1
                self.step_times.clear()
            return httpx.Response(200, json={"status": "reset"})
        if path == "/select_engine":
            return httpx.Response(200, json={"status": "ok"})
        with self.lock:
            mode = self.mode
        if mode == "down":
            raise httpx.ConnectError("connection refused", request=request)
        if mode == "unsupported":
            return httpx.Response(200, json={"status": "ai_unsupported_engine", "engine_model": "Rotax_912_ULS"})
        if mode == "garbage":
            return httpx.Response(200, text="<html>not json</html>")
        if mode == "slow":
            await asyncio.sleep(self.slow_s)
        body = json.loads(request.content)
        with self.lock:
            self.step_times.append(body["time"])
            n = len(self.step_times)
        if n <= self.warmup_calls:
            return httpx.Response(200, json={"status": "warming_up", "steps_collected": n, "steps_needed": 128})
        with self.lock:
            if self.timeout_next_ok:
                self.timeout_next_ok = False
                raise httpx.ReadTimeout("slow first inference", request=request)
        return httpx.Response(200, json={
            "status": "ok", "model_version": "v3", "fault_detected": False, "detection_confidence": 0.01,
            "health_percent": 99.0, "rul_hours": 1800.0, "tbo_hours": 2000.0, "rul_percent_remaining": 90.0,
            "diagnosis": {}, "severity_percent": {}, "failure_modes": {}, "faulty_channels": [],
        })


class SimulationSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ai = FakeAI()
        main.ai_client = httpx.AsyncClient(transport=httpx.MockTransport(cls.ai.handler))
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.post("/stop", json={"final": True})
        cls.client.__exit__(None, None, None)

    def setUp(self):
        self.ai.set(mode="ok", warmup_calls=3, slow_s=2.0)
        with self.ai.lock:
            self.ai.step_times.clear()
            self.ai.resets = 0
        self.client.post("/stop", json={"final": True})
        r = self.client.post("/select_engine", json={"engine_model": "Rotax_914_ULF"})
        self.assertEqual(r.json()["status"], "ok")

    # ---- helpers ----
    def state(self):
        return self.client.get("/state").json()

    def sim_time(self):
        return (self.state().get("telemetry") or {}).get("time") or 0.0

    def wait_for(self, predicate, timeout=10.0, every=0.1):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if predicate():
                return True
            time.sleep(every)
        return False

    def start_cruise(self):
        self.assertEqual(self.client.post("/start", json={}).json()["status"], "started")
        self.client.post("/params", json={"altitude": 1500, "throttle": 0.5, "airspeed": 45, "aoa": 2})

    def realtime_rate(self, seconds=2.0):
        a, w0 = self.sim_time(), time.time()
        time.sleep(seconds)
        return (self.sim_time() - a) / (time.time() - w0)

    # ---- pure units ----
    def test_json_safe_makes_browser_parsable_json(self):
        payload = {"egt": float("nan"), "vib": [0.1, float("inf")], "nested": {"x": float("-inf"), "ok": 1.5}}
        text = json.dumps(safety.json_safe(payload), allow_nan=False)
        self.assertEqual(strict_json(text), {"egt": None, "vib": [0.1, None], "nested": {"x": None, "ok": 1.5}})

    def test_telemetry_validation_catches_bad_steps(self):
        good = UAVEngineTwin(dt=0.01, physics_version="v3").step()
        self.assertIsNone(safety.telemetry_problem(good))
        self.assertIn("cht", safety.telemetry_problem({**good, "cht": float("nan")}))
        self.assertIsNotNone(safety.telemetry_problem(None))

    def test_circuit_breaker_opens_and_probes(self):
        now = [0.0]
        cb = safety.CircuitBreaker(failure_threshold=3, cooldown_s=5.0, clock=lambda: now[0])
        for _ in range(2):
            cb.record_failure("x")
        self.assertTrue(cb.allow())
        cb.record_failure("x")
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.allow())
        now[0] = 5.5
        self.assertEqual(cb.state, "half_open")
        cb.record_failure("still down")            # failed probe re-opens
        self.assertFalse(cb.allow())
        now[0] = 11.0
        self.assertTrue(cb.record_success())       # reports the recovery
        self.assertEqual(cb.state, "closed")

    # ---- endpoint input safety ----
    def test_params_are_clamped_and_non_finite_rejected(self):
        r = self.client.post("/params", content='{"altitude": 1e9, "throttle": -3, "aoa": NaN}',
                             headers={"Content-Type": "application/json"}).json()
        self.assertEqual(r["altitude"], 8000.0)
        self.assertEqual(r["throttle"], 0.0)
        self.assertIn("aoa", r["rejected"])
        self.assertTrue(math.isfinite(main.twin.aoa))

    # ---- live loop behaviour ----
    def test_ai_down_simulation_keeps_real_time_and_breaker_opens(self):
        self.ai.set(mode="down")
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.sim_time() > 2.0, timeout=8))
        rate = self.realtime_rate(2.0)
        self.assertLess(rate, 1.5, "physics must not fast-forward with no AI to feed")
        self.assertGreater(rate, 0.6, "physics must keep running while the AI is down")
        st = self.state()
        self.assertEqual(st["sim_status"]["ai"], "unavailable")
        self.assertIn(st["sim_status"]["ai_breaker"], ("open", "half_open"))
        self.assertTrue(self.client.get("/health").json()["loop_alive"])

    def test_ai_warms_up_with_samples_in_order(self):
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: (self.state().get("ai") or {}).get("status") == "ok", timeout=10))
        with self.ai.lock:
            times = list(self.ai.step_times)
        self.assertGreater(len(times), 3)
        self.assertEqual(times, sorted(times), "the AI must receive samples in simulated-time order")
        self.assertEqual(len(times), len(set(times)), "no sample may be sent twice")

    def test_slow_ai_never_blocks_physics(self):
        self.ai.set(mode="slow", slow_s=2.0, warmup_calls=0)
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.sim_time() > 1.0, timeout=8))
        rate = self.realtime_rate(3.0)
        self.assertGreater(rate, 0.6, f"physics stalled behind a slow AI (rate {rate:.2f}x)")

    def test_garbage_ai_response_is_contained(self):
        self.ai.set(mode="garbage")
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.state()["sim_status"]["ai"] == "unavailable", timeout=8))
        self.assertTrue(self.client.get("/health").json()["loop_alive"])

    def test_one_slow_ai_reply_does_not_reset_the_ai_window(self):
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: (self.state().get("ai") or {}).get("status") == "ok", timeout=10))
        resets_before = self.ai.resets
        self.ai.set(timeout_next_ok=True)
        self.assertTrue(self.wait_for(lambda: not self.ai.timeout_next_ok, timeout=5), "timeout was never triggered")
        time.sleep(2.0)
        self.assertEqual(self.ai.resets, resets_before, "a single slow reply must not reset the AI's window")
        self.assertEqual((self.state().get("ai") or {}).get("status"), "ok")
        self.assertEqual(self.state()["sim_status"]["ai_breaker"], "closed")

    def test_engine_without_ai_model_is_reported_not_faked(self):
        self.ai.set(mode="unsupported")
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.state()["sim_status"]["ai"] == "unsupported", timeout=8))
        st = self.state()
        self.assertEqual(st["sim_status"]["ai_breaker"], "closed", "a missing model is not an outage")
        rate = self.realtime_rate(2.0)
        self.assertLess(rate, 1.5, "no fast-forward when the AI can never warm up")
        self.assertTrue(self.client.get("/health").json()["loop_alive"])

    def test_ai_outage_then_recovery_resets_the_ai_window(self):
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: (self.state().get("ai") or {}).get("status") == "ok", timeout=10))
        self.ai.set(mode="down")
        # A real outage, not one failed sample: wait for the breaker to open. (A single
        # failure deliberately does NOT reset the window - see the slow-reply test.)
        self.assertTrue(self.wait_for(lambda: self.state()["sim_status"]["ai_breaker"] != "closed", timeout=10))
        resets_before = self.ai.resets
        self.ai.set(mode="ok")
        self.assertTrue(self.wait_for(lambda: self.ai.resets > resets_before, timeout=15),
                        "after an outage the AI buffer must be reset before new samples")
        self.assertTrue(self.wait_for(lambda: (self.state().get("ai") or {}).get("status") == "ok", timeout=15))

    def test_single_physics_fault_recovers_and_flight_continues(self):
        self.ai.set(mode="down")
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.sim_time() > 1.0, timeout=8))
        twin = main.twin
        original = twin.step
        fired = {"n": 0}

        def poisoned():
            out = original()
            if fired["n"] == 0:
                fired["n"] = 1
                out = {**out, "egt": float("nan")}
            return out
        twin.step = poisoned
        self.assertTrue(self.wait_for(lambda: self.state()["sim_status"]["recoveries"] >= 1, timeout=5))
        t_after = self.sim_time()
        time.sleep(1.0)
        st = self.state()
        self.assertTrue(st["running"])
        self.assertGreater(self.sim_time(), t_after, "the flight must continue after recovery")
        self.assertIsNot(main.twin, twin, "the twin is rebuilt from the last good step")

    def test_persistent_faults_halt_safely_and_start_resumes(self):
        self.ai.set(mode="down")
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.sim_time() > 1.0, timeout=8))
        original = UAVEngineTwin.step

        def broken(self_twin):
            raise RuntimeError("simulated solver blow-up")
        UAVEngineTwin.step = broken
        main.twin.step = broken.__get__(main.twin)
        try:
            self.assertTrue(self.wait_for(lambda: self.state()["sim_status"]["health"] == "halted", timeout=5))
            st = self.state()
            self.assertFalse(st["running"])
            self.assertFalse(self.client.get("/health").json()["loop_alive"])
            self.assertIn("simulated solver blow-up", st["sim_status"]["last_error"])
        finally:
            UAVEngineTwin.step = original
            if "step" in main.twin.__dict__:
                del main.twin.__dict__["step"]
        r = self.client.post("/start", json={}).json()
        self.assertEqual(r["status"], "started", "a halted session must be startable again")
        t0 = self.sim_time()
        self.assertTrue(self.wait_for(lambda: self.sim_time() > t0 + 0.5, timeout=5))
        self.assertNotEqual(self.state()["sim_status"]["health"], "halted")

    def test_rapid_pause_resume_never_runs_two_loops(self):
        self.ai.set(mode="down")
        self.start_cruise()
        for _ in range(15):
            self.client.post("/stop", json={"final": False})
            self.client.post("/start", json={})
        rate = self.realtime_rate(2.5)
        self.assertLess(rate, 1.5, f"duplicate loops would run physics ~2x real time (got {rate:.2f}x)")
        self.assertTrue(self.client.get("/health").json()["loop_alive"])

    def test_websocket_frames_are_valid_json_with_status(self):
        self.ai.set(mode="down")
        self.start_cruise()
        self.assertTrue(self.wait_for(lambda: self.sim_time() > 0.5, timeout=8))
        main.twin.egt_state = float("nan")        # force a non-finite value into the stream path
        frames = 0
        with self.client.websocket_connect("/ws") as ws:
            for _ in range(6):
                msg = strict_json(ws.receive_text())
                frames += 1
                if "sim_status" in msg:
                    self.assertIn(msg["sim_status"]["health"], ("running", "recovered", "halted"))
        self.assertEqual(frames, 6)
        self.assertTrue(self.wait_for(lambda: self.client.get("/health").json()["loop_alive"], timeout=3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
