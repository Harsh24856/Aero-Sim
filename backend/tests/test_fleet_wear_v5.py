"""The on-board twin runs fleet-average wear for the engine's hours.

A twin of a brand-new engine turns normal, in-limit ageing into a residual of
-10 to -16 sigma on rpm and oil pressure by mid-life, which drowns the faults that
show on those channels. A twin that knows only the hour meter and the fleet wear
curve leaves the residual to what is particular to this engine.

    cd backend && .venv/bin/python -m unittest tests.test_fleet_wear_v5
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features_v5 as F  # noqa: E402
import physics_v5 as P  # noqa: E402
import twin_v5  # noqa: E402
from degradation_v5 import DegradationStateV5, baseline_health, fleet_wear_health  # noqa: E402
from sensors_v5 import SENSOR_SPEC  # noqa: E402

CHANNELS = ["engine_rpm", "oil_pressure", "fuel_flow"]


def healthy_residuals(n: int, twin_health) -> np.ndarray:
    """|engine - twin| in noise sigma for n healthy (wear-only) engines at random
    hours and operating points; twin_health(deg, hours) picks the twin's health."""
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        d = DegradationStateV5(rng, start_hours=0.0, tbo_hours=2000.0, n_faults=0)
        hours = float(rng.uniform(0, 2000))
        hw, ht = d.health_wear_only(hours), twin_health(d, hours)
        u = P.Inputs(altitude_m=float(rng.uniform(0, 4500)), airspeed_ms=45, throttle=float(rng.uniform(0.35, 1.0)),
                     isa_dev_c=float(rng.uniform(-10, 15)), humidity_frac=0.5)
        e, t = P.PistonEngineV5("Rotax_914_ULF", dt=1.0), P.PistonEngineV5("Rotax_914_ULF", dt=1.0)
        e.warm_start(u, hw)
        t.warm_start(u, ht)
        for _ in range(600):
            o, r = e.step(u, hw), t.step(u, ht)
        out.append([abs(o[c] - r[c]) / SENSOR_SPEC[c]["noise_sd"] for c in CHANNELS])
    return np.array(out)


class TestFleetWear(unittest.TestCase):
    def test_new_engine_is_fleet_wear_at_zero_hours(self):
        self.assertEqual(fleet_wear_health(0.0, 2000.0).as_dict(), P.Health().as_dict())

    def test_wear_only_is_the_shared_baseline(self):
        d = DegradationStateV5(np.random.default_rng(3), start_hours=0.0, tbo_hours=2000.0, n_faults=0)
        for h in (0.0, 400.0, 1500.0):
            self.assertEqual(d.health_wear_only(h).as_dict(),
                             baseline_health(h, 2000.0, d.base_a, d.base_b, d.base_scale).as_dict())

    def test_fleet_wear_twin_keeps_healthy_residual_small(self):
        fleet = np.quantile(healthy_residuals(30, lambda d, h: fleet_wear_health(h, d.tbo)), 0.95, axis=0)
        new = np.quantile(healthy_residuals(30, lambda d, h: P.Health()), 0.95, axis=0)
        for c, f, n in zip(CHANNELS, fleet, new):
            self.assertLess(f, 8.0, f"{c}: fleet-wear twin p95 {f:.1f} sigma")
            self.assertLess(f, n / 2.5, f"{c}: fleet {f:.1f} vs new-engine {n:.1f} sigma")

    def test_contract_names_the_twin(self):
        self.assertEqual(F.contract()["twin"], "fleet_wear")


class TestTwinModeFollowsDeployedModel(unittest.TestCase):
    def _root(self, contracts: dict) -> str:
        root = tempfile.mkdtemp()
        for key, c in contracts.items():
            os.makedirs(os.path.join(root, key))
            with open(os.path.join(root, key, "contract_v5.json"), "w") as fh:
                json.dump(c, fh)
        return root

    def test_export_without_the_key_is_a_new_engine_twin(self):
        self.assertEqual(twin_v5.twin_mode("Rotax_914_ULF", self._root({"914": {}})), "new_engine")

    def test_export_with_the_key(self):
        root = self._root({"914": {"twin": "fleet_wear"}})
        self.assertEqual(twin_v5.twin_mode("Rotax_914_ULF", root), "fleet_wear")

    def test_engine_without_its_own_export_follows_the_placeholder(self):
        root = self._root({"914": {"twin": "fleet_wear"}})
        self.assertEqual(twin_v5.twin_mode("Rotax_916_iS", root), "fleet_wear")

    def test_live_twin_uses_fleet_wear_when_the_model_does(self):
        root = self._root({"914": {"twin": "fleet_wear"}})
        rec = {"engine_model": "Rotax_914_ULF", "engine_hours": 1500.0, "tbo_hours": 2000.0, "degradation_seed": 5}
        import scenarios_v4
        rec = {**scenarios_v4.make_engine("Rotax_914_ULF", seed=5), **rec}
        fleet = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=1, models_root=root)
        legacy = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=1,
                                         models_root=self._root({"914": {}}))
        res = {}
        for name, tw in (("fleet", fleet), ("legacy", legacy)):
            for _ in range(int(30 / tw.dt)):
                out = tw.step()
            res[name] = abs(out["res_engine_rpm"]) / SENSOR_SPEC["engine_rpm"]["noise_sd"]
        self.assertLess(res["fleet"], res["legacy"])


if __name__ == "__main__":
    unittest.main()
