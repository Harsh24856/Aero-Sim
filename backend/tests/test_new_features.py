"""Unit tests for features added after the safety suite: the RUL display filter, the
takeoff alert hold, measured sensors from the CAN bus, the sensor-suspect advisory and
residual sensor zeroing. No AI service, no Supabase, no network.

Run from backend/:
    .venv/bin/python -m unittest tests.test_new_features -v
"""
import asyncio
import os
import random
import sys
import time
import unittest

os.environ.setdefault("AERO_PHYSICS_VERSION", "v3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                              # noqa: E402

import advisory                                 # noqa: E402
import main                                     # noqa: E402
import residual                                 # noqa: E402
from aircraft_sim import MISMATCH_PROFILES, SENSOR_KEYS, apply_mismatch   # noqa: E402
from physics import UAVEngineTwin               # noqa: E402

TBO = 2000.0
OK = {"status": "ok", "tbo_hours": TBO}


class RulFilterTest(unittest.TestCase):
    def setUp(self):
        main.reset_rul_filter()

    def test_shown_rul_never_rises_and_tracks_engine_hours(self):
        random.seed(3)
        prev = None
        for t in range(300):
            wear = 0.0085 + 5e-5 * t
            raw = TBO * (1 - wear) + random.gauss(0, 6) + 0.03 * t          # jitter plus upward wander
            shown = main.smooth_rul({"time": 300 + t, "wear": wear}, {**OK, "rul_hours": raw})["rul_hours"]
            if prev is not None:
                self.assertLessEqual(shown, prev + 1e-9)
            prev = shown
        self.assertAlmostEqual(prev, TBO * (1 - 0.0085 - 5e-5 * 299), delta=5.0)

    def test_real_drop_passes_through_and_raw_is_kept(self):
        for t in range(460):                      # 60 s EMA: ~6.7 time constants after the drop
            raw = TBO - (150.0 if t >= 60 else 0.0)
            r = main.smooth_rul({"time": t, "wear": 0.0}, {**OK, "rul_hours": raw})
        self.assertLess(r["rul_hours"], TBO - 140.0)
        self.assertEqual(r["rul_hours_raw"], TBO - 150.0)
        self.assertAlmostEqual(r["rul_percent_remaining"], 100.0 * r["rul_hours"] / TBO, places=3)

    def test_held_takeoff_window_does_not_set_the_floor(self):
        """The shown RUL never rises within a flight, so the value that seeds the
        filter is a floor. A held (settling) window contains ground roll, which the
        RUL head was not trained on - seeding from it cost 10 h of permanent error on
        a measured 916 flight."""
        for t in range(60):                       # takeoff hold: a pessimistic estimate
            r = main.smooth_rul({"time": t, "wear": 0.0},
                                {**OK, "rul_hours": 1900.0, "settling": True})
            self.assertIs(r["rul_hours"], None)               # nothing shown at all
        first = main.smooth_rul({"time": 61, "wear": 0.0}, {**OK, "rul_hours": 1990.0})
        self.assertAlmostEqual(first["rul_hours"], 1990.0, places=3)   # floor is the
        # first IN-ENVELOPE prediction, not the held one
        later = main.smooth_rul({"time": 62, "wear": 0.0}, {**OK, "rul_hours": 1995.0})
        self.assertLessEqual(later["rul_hours"], first["rul_hours"] + 1e-9)  # still never rises

    def test_no_rul_is_shown_until_the_window_is_clean(self):
        """A 914 flight showed 99.1%, then 99.0%, then 100% during the takeoff hold:
        the head extrapolates while its window still holds ground roll, and a rising
        RUL is the one thing the gauge must never do. Show nothing instead."""
        for t, raw in ((5, 1980.0), (6, 1978.0), (7, 2000.0)):
            r = main.smooth_rul({"time": t, "wear": 0.0},
                                {**OK, "rul_hours": raw, "settling": True})
            self.assertIs(r["rul_hours"], None)
            self.assertIs(r["rul_percent_remaining"], None)
            self.assertFalse(r["rul_ready"])
            self.assertEqual(r["rul_hours_raw"], raw)      # kept for the logs
        live = main.smooth_rul({"time": 8, "wear": 0.0}, {**OK, "rul_hours": 1975.0})
        self.assertTrue(live["rul_ready"])
        self.assertAlmostEqual(live["rul_hours"], 1975.0, places=3)

    def test_shown_rul_never_exceeds_tbo(self):
        """TBO * relu(z) is unbounded above: a live 916 read 2101 h against a 2000 h
        TBO during the takeoff window, which the gauge would show as 105% remaining."""
        flying = main.smooth_rul({"time": 6, "wear": 0.0}, {**OK, "rul_hours": 2101.1})
        self.assertEqual(flying["rul_hours"], TBO)
        self.assertLessEqual(flying["rul_percent_remaining"], 100.0)

    def test_new_flight_resets_and_non_ok_passes_through(self):
        main.smooth_rul({"time": 500, "wear": 0.0}, {**OK, "rul_hours": 1500.0})
        self.assertEqual(main.smooth_rul({"time": 3, "wear": 0.0}, {**OK, "rul_hours": 1900.0})["rul_hours"], 1900.0)
        self.assertEqual(main.smooth_rul({"time": 4}, {"status": "warming_up"}), {"status": "warming_up"})


class AiWindowFreshnessTest(unittest.TestCase):
    """A restarted simulated clock means a new flight, and the AI's 128-sample window
    must not carry the old one into it. /start only clears that buffer when it treats
    the session as fresh - reopening the page without pressing Stop leaves
    session_active True, and the first predictions of the new flight then come from a
    window still holding the previous flight's telemetry."""

    def setUp(self):
        main.state["ai_last_seen_time"] = None
        main.state["ai_resync"] = False

    def feed(self, t):
        """The clock check the worker runs before sending a sample."""
        prev = main.state["ai_last_seen_time"]
        if isinstance(t, (int, float)) and isinstance(prev, (int, float)) and t < prev:
            main.state["ai_resync"] = True
        main.state["ai_last_seen_time"] = t
        return main.state["ai_resync"]

    def test_clock_going_backwards_forces_a_resync(self):
        for t in (100, 101, 102):
            self.assertFalse(self.feed(t))
        self.assertTrue(self.feed(1))            # new flight: t restarted

    def test_a_continuous_flight_never_resyncs(self):
        for t in range(0, 400, 1):
            self.assertFalse(self.feed(float(t)))


class TakeoffHoldTest(unittest.TestCase):
    ALARM = {"status": "ok", "fault_detected": True, "faulty_channels": ["cht"], "health_percent": 92.0,
             "detection_confidence": 0.97,
             "failure_modes": {"misfire": {"present": True, "severity_percent": 6.0}},
             "severity_percent": {"cht": 88.0, "egt": 0.0},
             "diagnosis": {"cht": {"fault_type": "bias", "confidence": 0.9, "reliable": True}}}

    def setUp(self):
        main.state["ai_last_sample_time"] = None
        main.state["ai_settle_until"] = 0.0

    def test_alerts_held_while_window_has_ground_roll_then_released(self):
        for t, airspeed in ((10, 12.0), (100, 31.0)):
            main.hold_alerts_outside_envelope({"time": t, "airspeed": airspeed}, self.ALARM)
        held = main.hold_alerts_outside_envelope({"time": 150, "airspeed": 40.0}, self.ALARM)
        self.assertTrue(held["settling"])
        self.assertFalse(held["fault_detected"])
        self.assertEqual(held["faulty_channels"], [])
        self.assertFalse(held["failure_modes"]["misfire"]["present"])
        # Everything the cockpit shows reads nominal while alerts are held - health, severities
        # and fault types together - and the model's own values stay available as *_raw.
        self.assertEqual(held["health_percent"], 100.0)
        self.assertEqual(held["health_percent_raw"], 92.0)
        self.assertEqual(held["severity_percent"], {"cht": 0.0, "egt": 0.0})
        self.assertEqual(held["severity_percent_raw"]["cht"], 88.0)
        self.assertEqual(held["diagnosis"]["cht"]["fault_type"], "none")
        self.assertEqual(held["diagnosis_raw"]["cht"]["fault_type"], "bias")
        # The cockpit needs a countdown, not just a flag.
        self.assertEqual(held["settle_window_s"], main.AI_WINDOW_S)
        self.assertAlmostEqual(held["settle_seconds_left"], 100 + main.AI_WINDOW_S - 150, places=1)
        self.assertEqual(held["detection_confidence"], 0.0)
        self.assertEqual(held["detection_confidence_raw"], 0.97)
        released = main.hold_alerts_outside_envelope({"time": 229, "airspeed": 40.0}, self.ALARM)
        self.assertNotIn("settling", released)
        self.assertTrue(released["fault_detected"])
        self.assertEqual(released["health_percent"], 92.0)
        self.assertEqual(released["severity_percent"]["cht"], 88.0)
        self.assertEqual(released["diagnosis"]["cht"]["fault_type"], "bias")
        self.assertEqual(released["detection_confidence"], 0.97)


class EngineFailureShutdownTest(unittest.TestCase):
    """Health at the floor for HEALTH_FAILURE_HOLD_S straight ends the flight."""
    DEAD = {"status": "ok", "health_percent": 0.0}

    def setUp(self):
        main.state["health_zero_since"] = None

    def elapsed(self, t, result):
        return main.health_failure_seconds(t, result)

    def test_sustained_zero_health_trips_at_the_hold_time(self):
        for t in range(int(main.HEALTH_FAILURE_HOLD_S)):
            self.assertLess(self.elapsed(t, self.DEAD), main.HEALTH_FAILURE_HOLD_S)
        self.assertGreaterEqual(self.elapsed(main.HEALTH_FAILURE_HOLD_S, self.DEAD), main.HEALTH_FAILURE_HOLD_S)

    def test_a_dip_that_recovers_never_trips(self):
        for t in range(200):
            health = 0.0 if t % 15 else 80.0        # zero for 14 s, then one good second
            self.assertLess(self.elapsed(t, {"status": "ok", "health_percent": health}),
                            main.HEALTH_FAILURE_HOLD_S)

    def test_held_takeoff_alerts_and_missing_ai_never_trip(self):
        for t in range(120):
            self.assertEqual(self.elapsed(t, {**self.DEAD, "settling": True}), 0.0)
        for t in range(120):
            self.assertEqual(self.elapsed(t, {"status": "warming_up"}), 0.0)
        for t in range(120):
            self.assertEqual(self.elapsed(t, None), 0.0)

    def test_the_loops_decision_trips_only_after_the_hold(self):
        for t in range(int(main.HEALTH_FAILURE_HOLD_S)):
            self.assertFalse(main.should_end_flight(t, self.DEAD))
        self.assertTrue(main.should_end_flight(main.HEALTH_FAILURE_HOLD_S, self.DEAD))

    def test_zero_hold_disables_the_shutdown(self):
        """AERO_HEALTH_FAILURE_HOLD_S=0 means off, not "fire on the first second".
        The loop used to test `elapsed >= hold` inline, and 0 >= 0 ended every flight
        at t=1 s - measured on a bench run, with no AI result at all."""
        hold = main.HEALTH_FAILURE_HOLD_S
        main.HEALTH_FAILURE_HOLD_S = 0.0
        try:
            for t in range(200):
                self.assertFalse(main.should_end_flight(t, self.DEAD),
                                 f"a zero hold tripped at t={t}")
        finally:
            main.HEALTH_FAILURE_HOLD_S = hold

    def test_a_new_flights_clock_restarts_the_count(self):
        for t in range(19):
            self.elapsed(100 + t, self.DEAD)
        self.assertEqual(self.elapsed(0, self.DEAD), 0.0)       # clock restarted: fresh flight
        self.assertEqual(self.elapsed(5, self.DEAD), 5.0)


class MeasuredSensorsTest(unittest.TestCase):
    def setUp(self):
        main.state["measured"] = {}
        main.state["measured_at"] = None

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_bounds_overlay_and_staleness(self):
        r = self.run_async(main.update_measured(main.MeasuredUpdate(egt=712.0, cht=300.0, rpm=4000.0, vibx=float("nan"))))
        self.assertEqual(sorted(r["accepted"]), ["egt", "rpm"])
        self.assertEqual(sorted(r["rejected"]), ["cht", "vibx"])
        twin_out = main.twin.step()
        merged = main.apply_measured(twin_out)
        self.assertEqual(merged["data_source"], "can")
        self.assertEqual(merged["egt"], 712.0)
        self.assertEqual(merged["rpm_fault"], 4000.0)
        self.assertEqual(merged["engine_rpm"], twin_out["engine_rpm"])      # twin physics untouched
        self.assertEqual(merged["cht"], twin_out["cht"])                    # out-of-range value not overlaid
        self.assertAlmostEqual(merged["twin_sensors"]["egt"], twin_out["egt"])
        main.state["measured_at"] = time.monotonic() - 2.0
        self.assertEqual(main.apply_measured(twin_out)["data_source"], "sim")

    def test_aircraft_owns_setpoints_while_live(self):
        run = self.run_async
        self.assertEqual(run(main.update_params(main.ParamUpdate(throttle=0.42)))["status"], "ok")
        run(main.update_measured(main.MeasuredUpdate(egt=650.0)))
        self.assertEqual(run(main.update_params(main.ParamUpdate(throttle=0.99)))["status"], "ignored")
        self.assertAlmostEqual(main.twin.throttle, 0.42)
        self.assertEqual(run(main.update_params(main.ParamUpdate(throttle=0.66, source="can")))["status"], "ok")
        main.state["measured_at"] = time.monotonic() - 2.0
        self.assertEqual(run(main.update_params(main.ParamUpdate(throttle=0.30)))["status"], "ok")


class SensorSuspectAdvisoryTest(unittest.TestCase):
    AI = {"status": "ok", "model_version": "v3", "fault_detected": True, "health_percent": 60.0,
          "rul_percent_remaining": 94.0, "rul_hours": 1880, "tbo_hours": 2000, "diagnosis": {},
          "severity_percent": {}, "failure_modes": {"cooling_degradation": {"present": True, "severity_percent": 35.0}}}
    CHT = {"label": "CHT", "unit": "C", "residual": 6.2, "z": 6.2, "expected": 98.0, "signature": "drift"}

    def codes(self, deviations, channels):
        res = {"enabled": True, "deviations": deviations, "saturated": [], "degradation_index": 0.01, "channels": channels}
        return [i["code"] for i in advisory.build_advisory(self.AI, {"oil_pressure": 80, "oil_temp": 95}, res)["items"]]

    def test_lone_drifting_channel_is_suspected_as_sensor(self):
        self.assertIn("RES_CHT_SENSOR_SUSPECT", self.codes(["cht"], {"cht": self.CHT}))

    def test_real_mode_moving_its_channels_together_is_not(self):
        oil = {"label": "Oil temp", "unit": "C", "residual": 4.0, "z": 8.0, "expected": 95.0, "signature": "drift"}
        self.assertNotIn("RES_CHT_SENSOR_SUSPECT", self.codes(["cht", "oil_temp"], {"cht": self.CHT, "oil_temp": oil}))


class ResidualZeroingTest(unittest.TestCase):
    CRUISE = dict(throttle=0.5, altitude=1500.0, airspeed=45.0, aoa=2.0, isa_dev_c=0.0)

    def fly(self, mismatch, zero_at, seconds, drift_from=None):
        np.random.seed(11)
        rng = np.random.default_rng(11)
        plant = UAVEngineTwin(dt=1.0, engine_model="Rotax_914_ULF")
        twin = UAVEngineTwin(dt=1.0, engine_model="Rotax_914_ULF")
        for e in (plant, twin):
            for k, v in self.CRUISE.items():
                setattr(e, k, v)
        mon = residual.ResidualMonitor(lag="euler")
        after, r = [], {}
        for s in range(seconds):
            p, t = plant.step(), twin.step()
            prof = {**mismatch, "drift_per_min": {"cht": 1.0}} if drift_from is not None and s >= drift_from else mismatch
            meas = apply_mismatch({k: float(p[src]) for k, src in SENSOR_KEYS.items()}, prof,
                                  p["time"] - (drift_from or 0), rng)
            merged = dict(t)
            for ch, v in meas.items():
                merged[{"rpm": "rpm_fault"}.get(ch, ch)] = v
            if s == zero_at:
                mon.begin_zeroing(known_wear=t["wear"], samples=60)
            r = mon.update(merged, dt=1.0)
            if s >= zero_at + 60 + 128 and (drift_from is None or s < drift_from):
                after.append(r)
        return mon, after, r

    def test_zeroing_recovers_calibration_offsets_and_removes_false_deviations(self):
        truth = MISMATCH_PROFILES["calibration"]["offset"]
        mon, after, _ = self.fly(MISMATCH_PROFILES["calibration"], zero_at=300, seconds=700)
        for ch, off in truth.items():
            self.assertAlmostEqual(mon.offsets[ch], off, delta=0.6, msg=ch)
        channels = {c for r in after for c in (r.get("deviations") or [])} - {"rpm"}
        self.assertEqual(channels, set())
        self.assertLess(np.mean([r["degradation_index"] for r in after]), 0.08)

    def test_zeroing_a_healthy_engine_is_harmless_and_real_drift_is_still_caught(self):
        mon, _, _ = self.fly({}, zero_at=300, seconds=500)
        self.assertTrue(all(abs(v) < 0.6 for v in mon.offsets.values()), mon.offsets)
        _, _, last = self.fly(MISMATCH_PROFILES["calibration"], zero_at=300, seconds=1300, drift_from=700)
        self.assertIn("cht", last["deviations"])


if __name__ == "__main__":
    unittest.main()
