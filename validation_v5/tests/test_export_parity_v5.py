"""The export (assembly_v5.Deployed on backend/models_v5/<key>) must answer exactly
as the training-time Specialists do, on real test windows: every head and RUL.

    validation/venv/bin/python validation_v5/tests/test_export_parity_v5.py [run_key]
    AERO_TEST_CPU=1 ...   same check with TensorFlow on the CPU (device parity)
"""
import os
import sys
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
RUN = sys.argv.pop(1) if len(sys.argv) > 1 else "914b"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if os.environ.get("AERO_TEST_CPU"):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

import run_v5  # noqa: E402
from assembly_v5 import Deployed  # noqa: E402
from export_v5 import engine_key  # noqa: E402

EXPORT = os.path.join(os.path.dirname(V5), "backend", "models_v5", engine_key(RUN))
REF = os.path.join(V5, "logs", f"parity_ref_{RUN}.npz")          # git-ignored scratch
N = 512


@unittest.skipUnless(os.path.exists(os.path.join(EXPORT, "manifest.json")), "no export")
class TestExportParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline_v5 import Cache
        cls.cache = Cache(run_v5.paths(RUN)["cache"])
        cls.dep = Deployed(EXPORT)
        rng = np.random.default_rng(0)
        cls.ids = np.sort(rng.choice(cls.cache.end_ids(["test"], jitter=False), N, replace=False))
        cls.seq, cls.ctx, cls.E = cls.cache.batch(cls.ids)

    def test_heads_match_training_specialists(self):
        if os.environ.get("AERO_TEST_CPU"):
            self.skipTest("training Specialists need the GPU; the CPU run checks against saved GPU outputs")
        import specialists_v5 as S
        sp = S.Specialists(self.cache, S.paths(RUN)["art"])
        ref = sp({"seq": self.seq, "ctx": self.ctx})
        got = self.dep({"seq": self.seq, "ctx": self.ctx})
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        np.savez(REF, _exported=np.array(self.dep.manifest["exported"]), **{k: np.asarray(v) for k, v in ref.items()})
        for k in ref:      # relative too: sensor logits reach ~200, where float32 rounding is ~1e-5
            np.testing.assert_allclose(np.asarray(got[k]), np.asarray(ref[k]), rtol=1e-6, atol=1e-5, err_msg=k)

    def test_device_parity_against_saved_reference(self):
        f = REF
        if not os.path.exists(f):
            self.skipTest("run once on the GPU first to save the reference")
        ref = np.load(f)
        if "_exported" not in ref.files or str(ref["_exported"]) != self.dep.manifest["exported"]:
            self.skipTest("reference is from an older export - rerun on the GPU to refresh it")
        got = self.dep({"seq": self.seq, "ctx": self.ctx})
        for k in [k for k in ref.files if k != "_exported"]:
            np.testing.assert_allclose(np.asarray(got[k]), ref[k], rtol=1e-5, atol=1e-4, err_msg=k)

    def test_scaling_matches_cache(self):
        """Raw 1 Hz rows scaled by Deployed == what the cache stored."""
        import pandas as pd
        import pyarrow.parquet as pq
        fr = self.cache.flights.iloc[int(self.E[0, 0])]
        idx = pd.read_parquet(os.path.join(self.cache.contract["source"], "index.parquet"))
        r = idx[idx.scenario_id == fr.scenario_id].iloc[0]
        g = pq.ParquetFile(os.path.join(self.cache.contract["source"], r.file)).read_row_group(int(r.row_group)).to_pandas()
        g = g.sort_values("t")
        import features_v5 as F
        rows = g[F.FEATURE_COLS].to_numpy()
        got = self.dep.scale_seq(rows)
        ref = np.asarray(self.cache.X[int(fr.row0):int(fr.row0 + fr.n_rows)], np.float32)
        self.assertLess(float(np.abs(got - ref).max()), 1e-6)

    def test_rul_runs(self):
        out = self.dep.predict_window(self.seq[0], self.ctx[0], engine_hours=500.0)
        self.assertGreaterEqual(out["rul_hours"], 0.0)
        self.assertLessEqual(out["rul_hours"], out["rul_calendar_hours"])


if __name__ == "__main__":
    unittest.main()
