"""Edge export of the deployed v3 heads (plan phase 10, PS "Edge AI / onboard analytics").

For every head in backend/models_v3/<key>/:
  1. convert to TensorFlow Lite (float32, and float16-weight quantised)
  2. parity: TFLite output vs Keras .predict() on real test windows
  3. size on disk and single-sample latency (TFLite interpreter, 1 CPU thread) vs Keras
Writes backend/models_v3/<key>/edge/*.tflite and validation/models/edge_report.json.

The RUL head (LSTM) may need SELECT_TF_OPS; the converter retries with them and the
report records which models needed the Flex delegate (heavier on a real edge target).

Run only when no training is using the GPU:
  validation/venv/bin/python3 validation/export_edge.py [--engines 914,915,916] [--samples 64]
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tensorflow as tf                     # noqa: E402
from tensorflow import keras                # noqa: E402
import train_common as C                    # noqa: E402

ROOT = os.path.join(os.path.dirname(HERE), "backend", "models_v3")


def convert(model, fp16, allow_flex):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    if allow_flex:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        conv._experimental_lower_tensor_list_ops = False
    return conv.convert()


def run_tflite(blob, feeds):
    interp = tf.lite.Interpreter(model_content=blob, num_threads=1)
    ins = interp.get_input_details()
    out = interp.get_output_details()[0]
    by_name = {d["name"]: d for d in ins}
    results, times = [], []
    for sample in feeds:
        for name, arr in sample.items():
            d = next((v for k, v in by_name.items() if name in k), ins[0] if len(ins) == 1 else None)
            if d["shape"][0] != 1 or list(d["shape"][1:]) != list(arr.shape):
                interp.resize_tensor_input(d["index"], [1, *arr.shape]); interp.allocate_tensors()
            interp.set_tensor(d["index"], arr[None].astype(np.float32))
        if not times:
            interp.allocate_tensors()
            for name, arr in sample.items():
                d = next((v for k, v in by_name.items() if name in k), ins[0])
                interp.set_tensor(d["index"], arr[None].astype(np.float32))
        t0 = time.perf_counter(); interp.invoke(); times.append(time.perf_counter() - t0)
        results.append(interp.get_tensor(out["index"])[0])
    return np.array(results), np.array(times[1:] or times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="914,915,916")
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args()
    by_key = {v: k for k, v in C.ENGINE_KEY.items()}
    report = {}
    for key in [k.strip() for k in a.engines.split(",") if k.strip()]:
        engine = by_key[key]
        man = json.load(open(os.path.join(ROOT, key, "manifest.json")))
        edge_dir = os.path.join(ROOT, key, "edge"); os.makedirs(edge_dir, exist_ok=True)
        _, _, test_ds = C.datasets(engine)
        xs, auxs = [], []
        for x, _ in test_ds:
            xs.append(x["x"].numpy()); auxs.append(x["x_rul_aux"].numpy())
            if sum(len(v) for v in xs) >= a.samples: break
        X = np.concatenate(xs)[: a.samples]; AUX = np.concatenate(auxs)[: a.samples]
        report[key] = {}
        for head, info in man["heads"].items():
            model = keras.models.load_model(os.path.join(ROOT, key, info["file"]), safe_mode=False, compile=False)
            needs_aux = "x_rul_aux" in info["inputs"]
            feed = {"x": X, "x_rul_aux": AUX} if needs_aux else X
            ref = np.asarray(model.predict(feed, verbose=0))
            t_keras = []
            for i in range(min(16, a.samples)):
                one = {"x": X[i:i+1], "x_rul_aux": AUX[i:i+1]} if needs_aux else X[i:i+1]
                t0 = time.perf_counter(); model.predict(one, verbose=0); t_keras.append(time.perf_counter() - t0)
            samples = [{"x": X[i], **({"x_rul_aux": AUX[i]} if needs_aux else {})} for i in range(a.samples)]
            row = {"keras_ms_p50": round(1000 * float(np.median(t_keras[1:])), 2)}
            for variant, fp16 in (("float32", False), ("float16", True)):
                flex = False
                try:
                    blob = convert(model, fp16, allow_flex=False)
                except Exception:
                    blob, flex = convert(model, fp16, allow_flex=True), True
                path = os.path.join(edge_dir, f"{key}_{head}_{variant}.tflite")
                open(path, "wb").write(blob)
                try:
                    got, times = run_tflite(blob, samples)
                    diff = float(np.max(np.abs(got.reshape(ref.shape) - ref)))
                    rel = diff / max(1e-9, float(np.max(np.abs(ref))))
                    row[variant] = {"size_kb": round(len(blob) / 1024, 1), "needs_flex": flex,
                                    "max_abs_diff": diff, "max_rel_diff": rel,
                                    "ms_p50": round(1000 * float(np.median(times)), 3),
                                    "ms_p95": round(1000 * float(np.percentile(times, 95)), 3)}
                except Exception as e:
                    row[variant] = {"size_kb": round(len(blob) / 1024, 1), "needs_flex": flex, "error": str(e)[:200]}
            report[key][head] = row
            f32, f16 = row["float32"], row["float16"]
            print(f"{key} {head:13s} keras {row['keras_ms_p50']:7.2f} ms | tflite f32 {f32.get('size_kb')} KB "
                  f"{f32.get('ms_p50', 'err')} ms diff {f32.get('max_abs_diff', f32.get('error'))} | "
                  f"f16 {f16.get('size_kb')} KB {f16.get('ms_p50', 'err')} ms diff {f16.get('max_abs_diff', f16.get('error'))}"
                  + (" [flex]" if f32.get("needs_flex") else ""))
    out = os.path.join(HERE, "models", "edge_report.json")
    json.dump(report, open(out, "w"), indent=2)
    print("report ->", out)


if __name__ == "__main__":
    main()
