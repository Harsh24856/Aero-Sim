"""The on-board twin runs fleet-average wear for the engine's hours.

A twin of a brand-new engine turns normal, in-limit ageing into a residual of
-10 to -16 sigma on rpm and oil pressure by mid-life, which drowns the faults that
show on those channels. A twin that knows only the hour meter and the fleet wear
curve leaves the residual to what is particular to this engine.

v6 goes one step further: the twin is calibrated to THIS engine from a
logbook-style history taken some hours earlier (engine_history).

    cd backend && .venv/bin/python -m unittest tests.test_engine_history_v6
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
from degradation_v4 import FaultEvent  # noqa: E402
from degradation_v5 import (  # noqa: E402
    FLEET_WEAR, DegradationStateV5, baseline_health, calibrated_wear_health, engine_history, fleet_wear_health,
)
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
        self.assertEqual(F.contract()["twin"], "calibrated_wear")


def deg_at(scale: float) -> DegradationStateV5:
    """A fault-free 914 (TBO 2,000 h) on the fleet's wear curve shape, wearing at `scale`."""
    d = DegradationStateV5(np.random.default_rng(0), start_hours=0.0, tbo_hours=2000.0, n_faults=0)
    d.base_a, d.base_b, d.base_scale = FLEET_WEAR[0], FLEET_WEAR[1], scale
    return d


def deg_with_fault_before(onset_h: float, sev: float) -> DegradationStateV5:
    """deg_at(1.0) plus one fault that is already at full severity well before 1,000 h."""
    d = deg_at(1.0)
    d.faults = [FaultEvent(name="bearing_wear", onset_h=onset_h, a=1.0, b=1.0, depth=sev, sense=1)]
    return d


class TestEngineHistory(unittest.TestCase):
    def test_calibrated_ratio_one_is_fleet_wear(self):
        self.assertEqual(calibrated_wear_health(900, 2000, 1.0).as_dict(), fleet_wear_health(900, 2000).as_dict())

    def test_engine_history_young_engine_ratio_one(self):
        h = engine_history(deg_at(1.2), 40.0, np.random.default_rng(0))
        self.assertEqual(h["wear_ratio"], 1.0)

    def test_engine_history_hours_below_lag(self):
        h = engine_history(deg_at(1.0), 30.0, np.random.default_rng(0), lag_h=100.0)
        self.assertEqual((h["lag_h"], h["wear_ratio"]), (30.0, 1.0))

    def test_engine_history_fault_contamination_clamped(self):
        h = engine_history(deg_with_fault_before(800, sev=0.9), 1000.0, np.random.default_rng(0))
        self.assertGreater(h["sev_max"], 0.5)
        self.assertLessEqual(h["wear_ratio"], 2.0)

    def test_history_severity_is_effective_and_gated(self):
        """The logbook holds what a past flight could see: severity in the same
        (effective) units the severity head reports, nothing for a sub-visible fault."""
        d = deg_at(1.0)
        spec = 0.60                                        # FAULT_MODES["bearing_wear"]["depth"]
        d.faults = [FaultEvent(name="bearing_wear", onset_h=100.0, a=1.0, b=1.0, depth=0.6 * spec, sense=-1)]
        h = engine_history(d, 1000.0, np.random.default_rng(0), lag_h=100.0)   # raw progress ~1.0 at 900 h
        self.assertTrue(0.45 < h["sev_max"] < 0.75, h["sev_max"])               # effective ~0.6, 10% noise
        d.faults = [FaultEvent(name="bearing_wear", onset_h=899.9, a=0.001, b=1.0, depth=spec, sense=-1)]
        self.assertEqual(engine_history(d, 1000.0, np.random.default_rng(0), lag_h=100.0)["sev_max"], 0.0)

    def test_calibrated_twin_beats_fleet_twin_on_healthy_residual(self):
        cal = np.quantile(healthy_residuals(30, lambda d, h: calibrated_wear_health(
            h, d.tbo, engine_history(d, h, np.random.default_rng(1))["wear_ratio"])), 0.95, axis=0)
        fleet = np.quantile(healthy_residuals(30, lambda d, h: fleet_wear_health(h, d.tbo)), 0.95, axis=0)
        for c, a, b in zip(CHANNELS, cal, fleet):
            self.assertLess(a, 0.7 * b, f"{c}: calibrated {a:.1f} vs fleet {b:.1f} sigma")


class TestGeneratorV6(unittest.TestCase):
    def test_generator_rows_carry_history_and_calibrated_twin(self):
        import glob
        import pandas as pd
        import generate_dataset_v5 as G
        out = tempfile.mkdtemp()
        man = G.generate_engine("Rotax_914_ULF", seed=3, n_flights=6, out_dir=out)
        self.assertEqual(man["contract"]["twin"], "calibrated_wear")
        d = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(out, "914", "flights_*.parquet"))])
        for c in F.HIST_COLS:
            self.assertIn(c, d.columns)
            self.assertTrue((d.groupby("scenario_id")[c].nunique() == 1).all(), c)
        self.assertTrue(d.hist_wear_ratio.between(0.5, 2.0).all())


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

    def _fast_wearing_engine(self) -> dict:
        import scenarios_v4
        rec = scenarios_v4.make_engine("Rotax_914_ULF", seed=5)
        rec = {**rec, "engine_hours": 1500.0, "fault_plan": {**rec["fault_plan"], "faults": [],
               "baseline": {"a": FLEET_WEAR[0], "b": FLEET_WEAR[1], "scale": 1.3}}}
        return rec

    def _rpm_residual(self, tw) -> float:
        for _ in range(int(30 / tw.dt)):
            out = tw.step()
        return abs(out["res_engine_rpm"]) / SENSOR_SPEC["engine_rpm"]["noise_sd"]

    def test_live_twin_calibrated_mode(self):
        rec = self._fast_wearing_engine()
        cal = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=1,
                                      models_root=self._root({"914": {"twin": "calibrated_wear"}}))
        fleet = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=1,
                                        models_root=self._root({"914": {"twin": "fleet_wear"}}))
        self.assertEqual(set(cal.history), {"lag_h", "wear_ratio", "sev_max"})
        self.assertGreater(cal.history["wear_ratio"], 1.0)        # it really does wear faster than the fleet
        self.assertLess(self._rpm_residual(cal), self._rpm_residual(fleet))

    def test_history_survives_restore(self):
        """A recovery or resume restores the same logbook: calibration never jumps mid-run."""
        root = self._root({"914": {"twin": "calibrated_wear"}})
        rec = self._fast_wearing_engine()
        # its own wear-curve shape (not the fleet's), so the ratio depends on when it is taken
        rec["fault_plan"]["baseline"] = {"a": 0.0035, "b": 1.30, "scale": 1.3}
        tw = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=1, models_root=root)
        for _ in range(int(30 / tw.dt)):
            snap = tw.step()
        self.assertGreater(snap["engine_hours"], tw.start_engine_hours)          # the life clock moved
        back = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", seed=1, models_root=root)
        back.restore_state(snap)
        self.assertEqual(back.history, tw.history)

    def test_engine_record_history_wins(self):
        rec = {**self._fast_wearing_engine(), "history": {"lag_h": 50.0, "wear_ratio": 0.8, "sev_max": 0.0}}
        tw = twin_v5.UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=1,
                                     models_root=self._root({"914": {"twin": "calibrated_wear"}}))
        self.assertEqual(tw.history["wear_ratio"], 0.8)
        out = tw.step()
        self.assertEqual(out["engine_history"]["wear_ratio"], 0.8)

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
