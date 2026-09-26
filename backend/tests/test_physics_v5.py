"""physics_v5: published Rotax figures still hold, a healthy engine is exactly v4,
and each Phase 1 fault signature exists and stays realistic.

    backend/.venv/bin/python tests/test_physics_v5.py
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_v4 as V4  # noqa: E402
import physics_v5 as P  # noqa: E402

ENGINES = list(P.ENGINE_SPECS)
TURBO = [e for e in ENGINES if P.ENGINE_SPECS[e].turbocharged]


def thr_for(eng, frac, alt=0.0, spd=45.0):
    """Throttle giving `frac` of rated power at this flight condition."""
    lo, hi = 0.05, 1.0
    for _ in range(14):
        m = (lo + hi) / 2
        o = settle(eng, P.Inputs(altitude_m=alt, airspeed_ms=spd, throttle=m), n=60)
        lo, hi = (m, hi) if o["power_kw"] * 1000 < frac * P.ENGINE_SPECS[eng].rated_power_w else (lo, m)
    return (lo + hi) / 2


def settle(eng, u, h=None, n=400, cls=P.PistonEngineV5):
    h = h or P.Health()
    e = cls(eng, dt=1.0)
    e.warm_start(u, h)
    for _ in range(n):
        o = e.step(u, h)
    return o


class RotaxFigures(unittest.TestCase):
    def test_rated_power_at_rated_rpm(self):
        for eng in ENGINES:
            e = P.PistonEngineV5(eng, dt=1.0)
            s = e.spec
            cyc = e._cycle(P.Inputs(altitude_m=0.0, throttle=1.0), P.Health(),
                           s.rated_rpm * 2 * math.pi / 60.0, warm=True)
            self.assertAlmostEqual(cyc["brake_power_w"] / s.rated_power_w, 1.0, delta=0.01, msg=eng)

    def test_ground_idle_near_1400_rpm(self):
        for eng in ENGINES:
            o = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=0.0, throttle=0.0))
            self.assertAlmostEqual(o["engine_rpm"], 1400.0, delta=150.0, msg=eng)

    def test_warm_cruise_75pct_standard_day(self):
        # 75% power, sea level, ISA: oil 90-110 C (Rotax normal), head well inside its
        # limit, EGT near the engine's nominal cruise EGT, oil 2-5 bar above 3,500 rpm.
        # (The cooling is sized for the hot-day climb below, so cruise runs cooler -
        # a 912 has no coolant thermostat.)
        for eng in ENGINES:
            s = P.ENGINE_SPECS[eng]
            o = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=45.0, throttle=thr_for(eng, 0.75)), n=2500)
            self.assertTrue(88.0 <= o["oil_temp"] <= 110.0, (eng, o["oil_temp"]))
            self.assertTrue(75.0 <= o["cht"] <= 115.0, (eng, o["cht"]))
            self.assertAlmostEqual(o["egt"], s.egt_nominal_c, delta=60.0, msg=eng)
            self.assertTrue(2.0 <= o["oil_pressure"] <= 5.0, (eng, o["oil_pressure"]))

    def test_certified_hot_day_climb_inside_limits_without_retard(self):
        # FAR 23 cooling case: full power, 38 m/s, sea level, ISA+23, design fuel.
        for eng in ENGINES:
            s = P.ENGINE_SPECS[eng]
            o = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=38.0, throttle=1.0, isa_dev_c=23.0), n=2500)
            self.assertEqual(o["knock_retard"], 0.0, eng)
            self.assertLess(o["egt"], s.egt_limit_c, eng)
            self.assertLess(o["cht"], s.cht_limit_c, eng)
            self.assertLess(o["oil_temp"], s.oil_temp_limit_c, eng)
            self.assertGreater(o["cht"], s.cht_limit_c - 25.0, eng)   # sized for it, not oversized

    def test_egt_rises_with_power(self):
        for eng in ENGINES:
            s = P.ENGINE_SPECS[eng]
            idle = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=0.0, throttle=0.0), n=1500)
            half = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=45.0, throttle=thr_for(eng, 0.55)), n=1500)
            full = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=50.0, throttle=1.0), n=1500)
            self.assertTrue(300.0 <= idle["egt"] <= 500.0, (eng, idle["egt"]))
            self.assertLess(idle["egt"], half["egt"] - 150.0, eng)
            self.assertLess(half["egt"], full["egt"] - 40.0, eng)
            self.assertLess(full["egt"], s.egt_limit_c, eng)

    def test_low_octane_on_hot_day_retards(self):
        for eng in ("Rotax_912_ULS", "Rotax_915_iS"):
            o = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=38.0, throttle=1.0,
                                     isa_dev_c=23.0, fuel_octane_mon=91.0))
            self.assertGreater(o["knock_retard"], 0.0, eng)

    def test_fuel_flow_matches_rotax_75pct_figures(self):
        # Published 75% cruise consumption: 912 ULS ~20 L/h, 914 ~25 L/h.
        for eng, ref in (("Rotax_912_ULS", 20.0), ("Rotax_914_ULF", 25.0)):
            o = settle(eng, P.Inputs(altitude_m=0.0, airspeed_ms=45.0, throttle=thr_for(eng, 0.75)), n=600)
            self.assertAlmostEqual(o["fuel_flow"] / ref, 1.0, delta=0.10, msg=(eng, o["fuel_flow"]))

    def test_turbo_holds_boost_to_critical_altitude(self):
        for eng in TURBO:
            s = P.ENGINE_SPECS[eng]
            below = settle(eng, P.Inputs(altitude_m=s.crit_alt_m - 500.0, airspeed_ms=50, throttle=1.0))
            above = settle(eng, P.Inputs(altitude_m=s.crit_alt_m + 1500.0, airspeed_ms=50, throttle=1.0))
            self.assertAlmostEqual(below["manifold_pressure_kpa"], s.max_map_pa / 1000.0, delta=1.0, msg=eng)
            self.assertLess(above["manifold_pressure_kpa"], s.max_map_pa / 1000.0 - 5.0, eng)


class HealthyIsV4(unittest.TestCase):
    def test_healthy_engine_identical_to_v4_except_egt(self):
        # Standard days only: v5 changed the knock reference (hot days) and the EGT
        # probe model; everything else about a healthy engine is v4.
        keys = ["engine_rpm", "power_kw", "cht", "coolant_temp", "oil_temp",
                "oil_pressure", "fuel_flow", "manifold_pressure_kpa", "wastegate_position"]
        for eng in ENGINES:
            for alt, thr in [(0, 1.0), (1500, 0.7), (3000, 0.5), (5500, 0.9)]:
                u = P.Inputs(altitude_m=alt, airspeed_ms=45, throttle=thr)
                a = settle(eng, u, cls=V4.PistonEngineV4)
                b = settle(eng, u)
                for k in keys:
                    self.assertEqual(a[k], b[k], (eng, alt, thr, k))


class IgnitionVsCombustion(unittest.TestCase):
    def _pair(self, eng, thr):
        u = P.Inputs(altitude_m=300.0, airspeed_ms=40.0, throttle=thr)
        return (settle(eng, u), settle(eng, u, P.Health(ignition_mod=0.75)),
                settle(eng, u, P.Health(combustion_eff_mod=0.78)))

    def test_ignition_rpm_drop_matches_ignition_check_envelope(self):
        # Single-circuit check at ~4,000 rpm: 50-100 rpm healthy, 300 rpm limit.
        for eng, thr in [("Rotax_912_ULS", 0.45), ("Rotax_914_ULF", 0.40),
                         ("Rotax_915_iS", 0.40), ("Rotax_916_iS", 0.38)]:
            h0, ign, _ = self._pair(eng, thr)
            self.assertTrue(3600 < h0["engine_rpm"] < 4300, (eng, h0["engine_rpm"]))
            drop = h0["engine_rpm"] - ign["engine_rpm"]
            self.assertTrue(120.0 <= drop <= 300.0, (eng, drop))

    def test_ignition_and_combustion_move_egt_in_opposite_directions(self):
        h0, ign, comb = self._pair("Rotax_914_ULF", 0.40)
        self.assertGreater(ign["egt"] - h0["egt"], 20.0)      # late burn: hotter exhaust
        self.assertLess(ign["cht"] - h0["cht"], 0.0)           # ...cooler head
        self.assertLess(comb["egt"] - h0["egt"], -20.0)        # incomplete burn: cooler exhaust


class Mixture(unittest.TestCase):
    def test_egt_peaks_near_stoichiometric(self):
        # EGT-mixture curve: rich of peak it is lower, lean of peak it falls again.
        for eng in ("Rotax_915_iS", "Rotax_916_iS"):          # injected: lambda = target
            egt = {lam: settle(eng, P.Inputs(altitude_m=1500.0, airspeed_ms=45.0, throttle=0.7,
                                             target_lambda=lam), n=900)["egt"]
                   for lam in (0.88, 1.00, 1.15)}
            self.assertGreater(egt[1.00] - egt[0.88], 30.0, eng)
            self.assertLess(egt[1.15], egt[1.00] - 30.0, eng)

    def test_fouled_injector_leans_a_rich_engine_egt_up_power_kept(self):
        u = P.Inputs(altitude_m=3000.0, airspeed_ms=45.0, throttle=0.7)
        for eng in ("Rotax_912_ULS", "Rotax_915_iS"):
            b = settle(eng, u, n=900)
            o = settle(eng, u, P.Health(injector_flow_mod=0.89), n=900)
            self.assertLess(o["fuel_flow"], b["fuel_flow"] - 0.5, eng)
            self.assertGreater(o["egt"], b["egt"], eng)
            self.assertLess(abs(o["engine_rpm"] - b["engine_rpm"]), 30.0, eng)


class OilFamily(unittest.TestCase):
    PTS = [(0.10, 30.0), (0.85, 50.0)]      # low-rpm and high-rpm operating points

    def _deficit(self, h, thr, spd, **env):
        u = P.Inputs(altitude_m=500.0, airspeed_ms=spd, throttle=thr, **env)
        a, b = settle("Rotax_914_ULF", u, n=500), settle("Rotax_914_ULF", u, h, n=500)
        return 1.0 - b["oil_pressure"] / a["oil_pressure"], b

    def test_pump_wear_worst_at_low_rpm(self):
        lo, _ = self._deficit(P.Health(oil_pump_mod=0.6), *self.PTS[0])
        hi, _ = self._deficit(P.Health(oil_pump_mod=0.6), *self.PTS[1])
        self.assertGreater(lo, hi + 0.08)

    def test_bearing_wear_worst_at_high_rpm(self):
        lo, _ = self._deficit(P.Health(friction_mod=1.6), *self.PTS[0])
        hi, _ = self._deficit(P.Health(friction_mod=1.6), *self.PTS[1])
        self.assertGreater(hi, lo + 0.04)

    def test_oil_degradation_grows_with_oil_temperature(self):
        cool, _ = self._deficit(P.Health(oil_quality_mod=0.55), 0.10, 30.0)
        hot, _ = self._deficit(P.Health(oil_quality_mod=0.55), 0.8, 45.0,
                               isa_dev_c=25.0, cooling_airflow_factor=0.8)
        self.assertGreater(hot, cool + 0.15)

    def test_pressures_stay_in_rotax_band(self):
        for h in (P.Health(oil_pump_mod=0.6), P.Health(friction_mod=1.6), P.Health(oil_quality_mod=0.55)):
            for thr, spd in self.PTS:
                _, o = self._deficit(h, thr, spd)
                self.assertTrue(0.8 <= o["oil_pressure"] <= 7.0, o["oil_pressure"])


class TurboSeparability(unittest.TestCase):
    def test_turbo_wear_heats_the_charge_a_leaking_wastegate_does_not(self):
        # Below critical altitude, where v4 hid both faults completely.
        for eng in TURBO:
            u = P.Inputs(altitude_m=1000.0, airspeed_ms=50.0, throttle=0.95)
            h0 = settle(eng, u)
            turbo = settle(eng, u, P.Health(turbo_eff_mod=0.7, turbo_flow_mod=0.7))
            wg = settle(eng, u, P.Health(wastegate_mod=0.45))
            self.assertAlmostEqual(turbo["manifold_pressure_kpa"], h0["manifold_pressure_kpa"], delta=0.5)
            self.assertLess(turbo["wastegate_position"], h0["wastegate_position"] - 0.05, eng)
            self.assertLess(wg["wastegate_position"], h0["wastegate_position"] - 0.05, eng)
            self.assertGreater(turbo["airbox_temp_c"] - h0["airbox_temp_c"], 5.0, eng)
            self.assertLess(abs(wg["airbox_temp_c"] - h0["airbox_temp_c"]), 0.5, eng)

    def test_912_has_no_turbo_instruments(self):
        o = settle("Rotax_912_ULS", P.Inputs(altitude_m=1000.0, throttle=0.7))
        self.assertEqual(o["wastegate_position"], 0.0)
        self.assertEqual(o["airbox_temp_c"], o["ambient_temp_c"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
