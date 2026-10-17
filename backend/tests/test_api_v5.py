"""The physics API's HTTP surface under physics v5, through FastAPI's TestClient.

The AI service is an httpx MockTransport (it answers /health like aiv5 and keeps
warming up), so nothing needs TensorFlow, a network or Supabase.

Run alone (main reads AERO_PHYSICS_VERSION once, at import), from backend/:
    .venv/bin/python -m unittest tests.test_api_v5
"""
import os
import sys
import unittest

os.environ["AERO_PHYSICS_VERSION"] = "v5"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx                                     # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import main                                      # noqa: E402

if main.PHYSICS_VERSION != os.environ["AERO_PHYSICS_VERSION"]:   # main reads it once, at import
    raise ImportError(f"main is already loaded as physics {main.PHYSICS_VERSION}, this module needs "
                      f"{os.environ['AERO_PHYSICS_VERSION']} - run it in its own process (scripts/test_all.sh)")

ENGINES = ["Rotax_914_ULF", "Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"]


def fake_ai(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "alive", "model_version": "v5", "available_engines": ENGINES,
                                         "placeholder_engines": ENGINES[1:]})
    if request.url.path == "/step":
        return httpx.Response(200, json={"status": "warming_up", "steps_collected": 1, "steps_needed": 128})
    return httpx.Response(200, json={"status": "ok"})


class ApiV5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.ai_client = httpx.AsyncClient(transport=httpx.MockTransport(fake_ai))
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.post("/stop", json={"final": True})
        cls.client.__exit__(None, None, None)

    def setUp(self):
        self.client.post("/stop", json={"final": True})

    def post(self, path, body=None):
        r = self.client.post(path, json=body or {})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_engines_report_ai_readiness_and_placeholders(self):
        e = self.client.get("/engines").json()
        self.assertEqual(sorted(e["engines"]), sorted(ENGINES))
        self.assertTrue(e["ai_online"])
        self.assertEqual(sorted(e["ai_placeholder_engines"]), sorted(ENGINES[1:]))

    def test_scenarios_and_preset_selection(self):
        sc = self.client.get("/scenarios").json()
        self.assertEqual(sc["status"], "ok")
        names = [s["name"] if isinstance(s, dict) else s for s in sc["scenarios"]]
        self.assertIn("healthy", names)
        self.assertEqual(self.post("/scenario", {"name": "healthy"})["status"], "ok")
        self.assertEqual(self.post("/scenario", {"name": "no_such_preset"})["status"], "error")

    def test_flight_lifecycle_with_fault_and_sensor_injection(self):
        self.assertEqual(self.post("/start")["status"], "started")
        st = self.client.get("/state").json()
        self.assertTrue(st["running"])
        self.assertEqual(st["physics_version"], "v5")
        self.assertEqual(self.post("/scenario", {"name": "healthy"})["status"], "error",
                         "a preset is a different engine: refused while flying")

        self.assertEqual(self.post("/inject", {"kind": "fault", "name": "oil_pump_degradation", "severity": 0.5})["status"], "ok")
        self.assertEqual(self.post("/inject", {"kind": "fault", "name": "no_such_fault"})["status"], "error")
        self.assertEqual(self.post("/inject", {"kind": "sensor", "channel": "egt", "type": "stuck"})["status"], "ok")
        self.assertEqual(self.post("/inject", {"kind": "neither"})["status"], "error")

        self.assertEqual(self.post("/inject/clear", {"kind": "fault", "name": "oil_pump_degradation"})["status"], "ok")
        self.assertEqual(self.post("/inject/clear", {"kind": "fault", "name": "oil_pump_degradation"})["status"], "error",
                         "nothing left to remove")
        self.assertEqual(self.post("/inject/clear", {"kind": "sensor", "channel": "egt"})["status"], "ok")

        self.assertEqual(self.post("/stop", {"final": True})["status"], "stopped")
        self.assertFalse(self.client.get("/state").json()["running"])

    def test_params_reach_the_twin(self):
        # Clamping and non-finite rejection: test_simulation_safety (all physics versions).
        r = self.post("/params", {"altitude": 1200.0, "throttle": 0.7, "airspeed": 45.0})
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(self.client.get("/state").json()["params"]["throttle"], 0.7)


if __name__ == "__main__":
    unittest.main()
