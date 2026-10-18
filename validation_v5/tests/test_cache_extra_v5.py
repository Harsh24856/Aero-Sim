"""A cache over the existing flights plus the P2 extra flights, on the LIVE scaler
(plan 2026-10-18-v5-real-world-fixes, P2.3). A fine-tune must see the same input
scaling the deployed models were trained on, so the export's contract does not move.
Builds two tiny datasets with the real generator - no GPU, no repo cache.

    validation/venv/bin/python validation_v5/tests/test_cache_extra_v5.py
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))

import generate_dataset_v5 as G  # noqa: E402
from pipeline_v5 import build_cache, Cache  # noqa: E402
import relabel_v5b  # noqa: E402
import run_v5  # noqa: E402

E = "Rotax_916_iS"


class TestExtraCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        G.generate_engine(E, G.engine_seed(E), 3, out_dir=os.path.join(d, "base"))
        G.generate_engine(E, G.engine_seed(E) + 100000, 2, out_dir=os.path.join(d, "steps"),
                          start_id=100000, missions=["throttle_steps"])
        cls.base, cls.steps = os.path.join(d, "base", "916"), os.path.join(d, "steps", "916")
        build_cache(cls.base, os.path.join(d, "c_base"))
        cls.live = os.path.join(d, "c_base", "contract_v5.json")
        build_cache([cls.base, cls.steps], os.path.join(d, "c_both"), scaler_from=cls.live)
        cls.a, cls.b = Cache(os.path.join(d, "c_base")), Cache(os.path.join(d, "c_both"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scaler_is_the_live_one(self):
        live = json.load(open(self.live))
        self.assertEqual(self.b.contract["scaler"], live["scaler"])
        self.assertEqual(self.b.contract["ctx_scaler"], live["ctx_scaler"])

    def test_old_flights_are_unchanged(self):
        n = int(self.a.flights.n_rows.sum())
        self.assertTrue(np.array_equal(np.asarray(self.a.X), np.asarray(self.b.X[:n])))
        k = len(self.a.ends)
        self.assertTrue(np.array_equal(np.asarray(self.a.ends), np.asarray(self.b.ends[:k])))

    def test_extra_flights_are_appended_and_recorded(self):
        fl = self.b.flights
        self.assertEqual(len(fl), 5)
        self.assertTrue((fl.scenario_id.iloc[3:] >= 100000).all())
        self.assertEqual(self.b.contract["source"], os.path.abspath(self.base))
        self.assertEqual(self.b.contract["extra_sources"], [os.path.abspath(self.steps)])

    def test_extra_weight_repeats_only_the_extra_flights_training_windows(self):
        # 6.6% of the 916's training windows held a step: too rare to move a fine-tune.
        splits = ["train", "cal", "val", "test"]
        extra = set(self.b.flights.flight[self.b.flights.scenario_id >= 100000])
        def count(ids):
            fl = self.b.ends[ids, 0].astype(int)
            return sum(f in extra for f in fl), sum(f not in extra for f in fl)
        n1 = count(self.b.end_ids(splits, seed=1))
        n2 = count(self.b.end_ids(splits, seed=1, extra_weight=2))
        # each random offset gives a flight +-1 window, so compare ratios, not counts
        self.assertAlmostEqual(n2[0] / n1[0], 2.0, delta=0.05)
        self.assertAlmostEqual(n2[1] / n1[1], 1.0, delta=0.05)
        self.assertEqual(len(self.b.end_ids(splits, jitter=False, extra_weight=2)),
                         len(self.b.end_ids(splits, jitter=False)))    # evaluation grid untouched

    def test_relabel_reads_every_source_and_keeps_the_live_thresholds(self):
        # The live 916 runs on v5b labels; the extra cache must be relabelled with the
        # SAME per-channel sensor thresholds, or the old flights' labels shift too.
        key = "_test_extra_relabel"
        cache = run_v5.paths(key)["cache"]
        os.symlink(os.path.join(self.tmp.name, "c_both"), cache)
        try:
            z = {c: 3.5 for c in relabel_v5b.FAULTABLE_CHANNELS}
            live = os.path.join(self.tmp.name, "live_b.json")
            json.dump({"relabel": {"z_sensor_per_channel": z}}, open(live, "w"))
            r = relabel_v5b.build(key, 1.0, 3.0, out_key=key + "_b", thresholds_from=live)
            out = Cache(r["cache"])
            self.assertEqual(out.contract["relabel"]["z_sensor_per_channel"], z)
            self.assertEqual(len(out.ends), len(self.b.ends))
        finally:
            os.unlink(cache)
            import shutil
            shutil.rmtree(run_v5.paths(key + "_b")["cache"], ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
