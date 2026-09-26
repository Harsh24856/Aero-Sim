"""dbv4.py against an in-memory stand-in for the Supabase client: the rows a v4
flight writes, and the engine hour meter carrying over from flight to flight.
Nothing touches the real database. (The schema itself - constraint, access rules,
old rows untouched - was checked in a rolled-back transaction on the live project.)

Run from backend/:
    .venv/bin/python -m unittest tests.test_dbv4 -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db                                         # noqa: E402
import dbv4                                       # noqa: E402
import scenarios_v4                               # noqa: E402
from twin_v4 import UAVEngineTwinV4               # noqa: E402


class FakeTable:
    def __init__(self, store, name):
        self.store, self.name, self.op, self.payload, self.filters = store, name, None, None, []

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def update(self, row):
        self.op, self.payload = "update", row
        return self

    def select(self, *_):
        self.op = "select"
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def execute(self):
        rows = self.store.setdefault(self.name, [])
        match = lambda r: all(r.get(k) == v for k, v in self.filters)   # noqa: E731
        if self.op == "insert":
            out = []
            for row in (self.payload if isinstance(self.payload, list) else [self.payload]):
                row = {**row, "id": len(rows) + 1}
                rows.append(row)
                out.append(row)
            return type("R", (), {"data": out})
        if self.op == "update":
            for r in rows:
                if match(r):
                    r.update(self.payload)
            return type("R", (), {"data": [r for r in rows if match(r)]})
        return type("R", (), {"data": [r for r in rows if match(r)]})


class FakeClient:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return FakeTable(self.store, name)


class DbV4(unittest.TestCase):
    def setUp(self):
        self._saved = (db._enabled, db._client)
        db._enabled, db._client = True, FakeClient()

    def tearDown(self):
        db._enabled, db._client = self._saved

    def fly(self, record, seconds=30):
        tw = UAVEngineTwinV4(engine_model="Rotax_914_ULF", engine=record, seed=1)
        sim = dbv4.start_simulation("user-1", "Rotax_914_ULF", tw.record, {"placeholder": False}, "hot_high")
        for i in range(seconds * 100):
            out = tw.step()
            if out["sample_new"] and int(out["time"]) % 10 == 0:
                dbv4.log_telemetry(sim, out["time"], out, {
                    "status": "ok", "model_version": "v4", "fault_detected": True, "detection_confidence": 0.9,
                    "health_percent": 71.0, "rul_hours": 400.0, "rul_calendar_hours": 550.0, "rul_mae_hours": 80.0,
                    "wear_limited": True, "fault_modes": {"bearing_wear": {"probability": 0.9, "present": True,
                                                                           "severity": 0.4}},
                    "sensors": {"cht": {"condition": "none", "confidence": 0.99}}})
        dbv4.end_simulation(sim, "stopped", 71.0, 400.0, out)
        return sim, out

    def test_two_flights_of_one_engine(self):
        rec = scenarios_v4.make_engine("Rotax_914_ULF", "bearing_wear", seed=3)
        sim1, out1 = self.fly(rec)
        s = db._client.store
        run = s["simulations"][0]
        self.assertEqual(run["mission"], "hot_high")
        self.assertEqual((run["model_version"], run["life_scale"], run["start_engine_hours"]),
                         ("v4", 180.0, 1450.0))
        self.assertEqual(run["end_engine_hours"], round(out1["engine_hours"], 4))
        eng = s["engines"][0]
        self.assertEqual(eng["name"], "914 #1")
        self.assertAlmostEqual(eng["engine_hours"], out1["engine_hours"], places=3)   # meter advanced
        log = s["telemetry_logs"][0]
        self.assertEqual(log["time_offset_s"], 10.0)                              # flight clock, unscaled
        self.assertGreater(log["engine_hours"], 1450.0)                           # life clock
        self.assertIn("oil_pressure", log["residuals"])
        self.assertEqual(log["truth"]["faults_present"], ["bearing_wear"])
        self.assertTrue(log["fault_modes"]["bearing_wear"]["present"])
        self.assertEqual(s["channel_diagnostics"][0]["fault_type"], "none")
        self.assertIsNone(s["channel_diagnostics"][0]["severity_percent"])

        # The next flight of the same engine starts where the last one ended.
        again = dbv4.get_engine(eng["id"])
        self.assertAlmostEqual(again["engine_hours"], out1["engine_hours"], places=3)
        sim2, out2 = self.fly(again)
        self.assertEqual(len(s["engines"]), 1)                                     # same engine row
        self.assertAlmostEqual(s["simulations"][1]["start_engine_hours"], out1["engine_hours"], places=3)
        self.assertGreater(s["engines"][0]["engine_hours"], out1["engine_hours"])

    def test_anonymous_flights_write_nothing(self):
        rec = scenarios_v4.make_engine("Rotax_914_ULF", "healthy", seed=1)
        self.assertIsNone(dbv4.start_simulation(None, "Rotax_914_ULF", rec))
        self.assertEqual(db._client.store, {})


if __name__ == "__main__":
    unittest.main()
