"""Federated learning demo (plan phase 10): FedAvg across engines as separate clients.

Each engine's dataset stays on its "client". Only model weights are exchanged:
  round r:  every client starts from the global weights, trains `--local-steps` on
            its own train split (its own scaler), and returns weights; the server
            averages them weighted by steps (FedAvg).
The shared task is fault DETECTION (phase-1 architecture: TCN encoder + sigmoid head),
because its label means the same thing on every engine.

Compared on each client's own test split (AUC):
  local-only  - trained on that client alone for the same total number of steps
  federated   - the global FedAvg model
A federated model that matches local-only while never seeing other clients' data is
the point; one that beats it shows the benefit of pooling knowledge across a fleet.

Run only when no other training is using the GPU (8 GB machine):
  validation/venv/bin/python3 validation/federated_demo.py [--clients 915,916] [--rounds 5] [--local-steps 150]
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tensorflow import keras                      # noqa: E402
from sklearn.metrics import roc_auc_score         # noqa: E402
import model_architectures as A                   # noqa: E402
import train_common as C                          # noqa: E402


def build():
    xi = A.build_encoder_input()
    m = keras.Model(xi, {"y_detection": A.build_detection_head(A.build_encoder(xi))})
    m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
              loss={"y_detection": keras.losses.BinaryCrossentropy()})
    return m


def evaluate(model, test_iter_factory, steps):
    yt, yp = [], []
    for i, (x, y) in enumerate(test_iter_factory()):
        yp.append(np.asarray(model.predict(x["x"], verbose=0)["y_detection"]).ravel())
        yt.append(y["y_detection"].numpy().ravel())
        if i + 1 >= steps: break
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return float(roc_auc_score(yt > 0.5, yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default="915,916")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local-steps", type=int, default=150)
    ap.add_argument("--eval-steps", type=int, default=40)
    a = ap.parse_args()
    by_key = {v: k for k, v in C.ENGINE_KEY.items()}
    keys = [k.strip() for k in a.clients.split(",") if k.strip()]
    t0 = time.time()

    data = {}
    for k in keys:
        tr, _, te = C.datasets(by_key[k])
        data[k] = {"train": iter(tr.map(C.split_labels(["y_detection"])).repeat()), "test": lambda te=te: te}

    global_model = build()
    history = []
    for r in range(1, a.rounds + 1):
        weights, sizes = [], []
        for k in keys:
            client = build(); client.set_weights(global_model.get_weights())
            client.fit(data[k]["train"], steps_per_epoch=a.local_steps, epochs=1, verbose=0)
            weights.append(client.get_weights()); sizes.append(a.local_steps)
        total = float(sum(sizes))
        global_model.set_weights([sum(w[i] * (s / total) for w, s in zip(weights, sizes))
                                  for i in range(len(weights[0]))])
        aucs = {k: evaluate(global_model, data[k]["test"], a.eval_steps) for k in keys}
        history.append({"round": r, "auc": aucs})
        print(f"round {r}: federated AUC " + "  ".join(f"{k} {v:.4f}" for k, v in aucs.items()))

    local = {}
    for k in keys:
        m = build()
        m.fit(data[k]["train"], steps_per_epoch=a.local_steps * a.rounds, epochs=1, verbose=0)
        local[k] = evaluate(m, data[k]["test"], a.eval_steps)
    print("local-only AUC  " + "  ".join(f"{k} {v:.4f}" for k, v in local.items()))

    out = {"clients": keys, "rounds": a.rounds, "local_steps": a.local_steps,
           "federated_history": history, "federated_final": history[-1]["auc"], "local_only": local,
           "minutes": round((time.time() - t0) / 60, 1),
           "note": "weights only leave each client; each client keeps its own data and scaler"}
    path = os.path.join(HERE, "models", "federated_report.json")
    json.dump(out, open(path, "w"), indent=2)
    print("report ->", path)


if __name__ == "__main__":
    main()
