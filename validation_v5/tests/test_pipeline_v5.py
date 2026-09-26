"""Phase 3 checks on a built cache (default: the 914 pilot).

    validation/venv/bin/python validation_v5/tests/test_pipeline_v5.py [cache_dir]
"""
import os
import sys
import time
import unittest

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.dirname(HERE)
ROOT = os.path.dirname(V5)
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import features_v5 as F  # noqa: E402
from pipeline_v5 import RES_SIGMA, RES_SLICE, WINDOW, Cache, _context  # noqa: E402

CACHE = sys.argv.pop(1) if len(sys.argv) > 1 else os.path.join(ROOT, "validation_v5", "cache", "pilot_914")


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = Cache(CACHE)

    def test_same_seed_same_epoch_identical_windows(self):
        a = self.c.end_ids(["train"], seed=3, epoch=5)
        b = self.c.end_ids(["train"], seed=3, epoch=5)
        self.assertTrue(np.array_equal(a, b))
        sa, ca, _ = self.c.batch(a[:64])
        sb, cb, _ = self.c.batch(b[:64])
        self.assertTrue(np.array_equal(sa, sb) and np.array_equal(ca, cb))

    def test_epochs_differ(self):
        a = self.c.end_ids(["train"], seed=3, epoch=0)
        b = self.c.end_ids(["train"], seed=3, epoch=1)
        self.assertFalse(np.array_equal(a, b))

    def test_splits_disjoint(self):
        fl = self.c.flights
        for s1, s2 in (("train", "test"), ("train", "val"), ("cal", "test")):
            self.assertFalse(set(fl[fl.split == s1].flight) & set(fl[fl.split == s2].flight))

    def test_cache_matches_raw_features_and_context(self):
        con = self.c.contract
        mean, std = np.array(con["scaler"]["mean"]), np.array(con["scaler"]["std"])
        cm, cs = np.array(con["ctx_scaler"]["mean"]), np.array(con["ctx_scaler"]["std"])
        src = con["source"]
        import pandas as pd
        idx = pd.read_parquet(os.path.join(src, "index.parquet"))
        worst_x = worst_c = 0.0
        for fi in (0, 7, 42):
            r = idx.iloc[fi]
            g = pq.ParquetFile(os.path.join(src, r.file)).read_row_group(int(r.row_group)).to_pandas().sort_values("t")
            A = g[F.FEATURE_COLS].to_numpy(np.float64, copy=True)
            A[:, RES_SLICE] /= RES_SIGMA
            f = self.c.flights.iloc[fi]
            X = np.asarray(self.c.X[f.row0:f.row0 + f.n_rows], np.float64) * std + mean
            worst_x = max(worst_x, float(np.max(np.abs(X - A) / (np.abs(A) + std))))
            ends = np.arange(WINDOW - 1, len(g), 16)
            C = _context(g, con["tbo_hours"])[ends]
            E = np.asarray(self.c.ends[f.end0:f.end0 + f.n_ends])
            Cc = E[:, self.c.ctx_slice] * cs + cm
            worst_c = max(worst_c, float(np.max(np.abs(Cc - C) / (np.abs(C) + cs))))
        self.assertLess(worst_x, 2e-3)       # float16 storage
        self.assertLess(worst_c, 1e-5)       # float32 storage

    def test_batch_shapes_and_throughput(self):
        ids = self.c.end_ids(["train"], seed=0, epoch=0)
        t0 = time.time()
        n = 0
        for i in range(0, min(len(ids), 20 * 256), 256):
            seq, ctx, E = self.c.batch(ids[i:i + 256])
            n += len(seq)
        rate = n / (time.time() - t0)
        self.assertEqual(seq.shape[1:], (WINDOW, F.N_FEATURES))
        self.assertEqual(ctx.shape[1], F.N_LONG + 3)
        print(f"\n  batch assembly: {rate:,.0f} windows/s")
        self.assertGreater(rate, 2000)


if __name__ == "__main__":
    unittest.main(verbosity=1)
