"""The v5 network must compute the same thing inside a tf.function (every training
step) as eagerly (predict, scoring, serving). tensorflow-metal broke this for ReLU
after a matmul - the v5 models were trained on a different network from the one
they were scored as. Run on the GPU the models train on.

    validation/venv/bin/python validation_v5/tests/test_graph_parity_v5.py
"""
import os
import sys
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
import features_v5 as F  # noqa: E402
import model_architectures_v5 as M  # noqa: E402


class TestGraphParity(unittest.TestCase):
    def test_eager_equals_graph(self):
        rng = np.random.default_rng(0)
        x = {"seq": tf.constant(rng.normal(size=(32, F.WINDOW, F.N_FEATURES)).astype("float32")),
             "ctx": tf.constant(rng.normal(size=(32, M.N_CTX)).astype("float32"))}
        for kw in ({}, {"linear_regression": True, "enc_norm": True}):
            model = M.build_model(**kw)
            eager = model(x, training=False)
            graph = tf.function(lambda z: model(z, training=False))(x)
            for k in eager:
                d = float(np.abs(np.asarray(eager[k]) - np.asarray(graph[k])).max())
                self.assertLess(d, 1e-4, f"{k} {kw}: eager and tf.function differ by {d}")


if __name__ == "__main__":
    unittest.main()
