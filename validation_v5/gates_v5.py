"""The v5 gates (docs/v5_model_improvement_plan.md §2) as data, and their check.

A gate passes only when the point estimate meets the target AND, for the head's
PRIMARY metric, the 95% flight-bootstrap CI of (model - do-nothing baseline)
excludes zero in the favourable direction. Secondary targets are point checks.
"""
from __future__ import annotations

import operator

# head -> [(metric path in evaluate() output, op, target, primary)]
GATES = {
    "detection": [("recall_at_p95", ">=", 0.85, True),
                  ("auc", ">=", 0.95, False)],
    "diagnosis": [("macro_f1", ">=", 0.70, True),
                  # §2 applies the 0.5 floor to faults Phase 1 marks observable. Until
                  # Phase 1 exists every applicable fault with test positives counts.
                  ("min_fault_recall", ">=", 0.50, False),
                  ("family_macro_f1", ">=", 0.85, False),
                  ("ece", "<=", 0.05, False)],
    "severity": [("mae_on_fault", "<=", 0.12, True),
                 ("mae_clean", "<=", 0.02, False),
                 ("abs:bias_on_fault", "<=", 0.03, False)],
    "sensor_fault": [("macro_f1", ">=", 0.60, True),
                     ("per_kind.dropout.recall", ">=", 0.90, False),
                     ("per_kind.stuck.recall", ">=", 0.70, False),
                     ("per_kind.spike.recall", ">=", 0.60, False),
                     ("per_kind.noise.recall", ">=", 0.60, False),
                     ("per_kind.bias.recall", ">=", 0.50, False),
                     ("per_kind.drift.recall", ">=", 0.40, False),
                     ("false_alarm_rate", "<=", 0.005, False)],
    "health": [("mae", "<=", 0.05, True)],
    "rul": [("mae_pct_tbo_wear_limited", "<=", 12.0, True),
            ("gain_vs_calendar", ">=", 0.25, False),
            ("mae_pct_tbo_tbo_limited", "<=", 4.0, False),
            ("frac_flights_nonmonotone", "<=", 0.0, False)],
    # Scored by tests/test_parity_v4.py, not by evaluate(); listed so the target lives here.
    "parity": [("max_abs_cpu_gpu", "<=", 1e-3, False)],
}
_OPS = {">=": operator.ge, "<=": operator.le}


def _get(head: dict, path: str):
    node = head
    for part in path.removeprefix("abs:").split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check(heads: dict) -> dict:
    """heads: evaluate()['heads'] (plus an optional 'parity' {'max_abs_cpu_gpu': {'point': x}}).
    Returns {head: {'pass': bool, 'reasons': [...]}} for every head present."""
    out = {}
    for head, gates in GATES.items():
        if head not in heads:
            continue
        reasons, ok = [], True
        for path, op, target, primary in gates:
            m = _get(heads[head], path)
            if m is None or m.get("point") is None:
                ok = False
                reasons.append(f"{path}: not measured")
                continue
            val = abs(m["point"]) if path.startswith("abs:") else m["point"]
            if not _OPS[op](val, target):
                ok = False
                reasons.append(f"{path} = {val:.4g} fails {op} {target:g} (CI {m['ci'][0]}, {m['ci'][1]})")
            if primary:
                d = m.get("diff")
                if d is None:
                    ok = False
                    reasons.append(f"{path}: no baseline to compare with")
                    continue
                lo, hi = d["ci"]
                beats = (lo is not None and lo > 0) if op == ">=" else (hi is not None and hi < 0)
                if not beats:
                    ok = False
                    reasons.append(f"{path}: CI of difference from baseline [{lo}, {hi}] does not exclude 0 "
                                   f"in the right direction (baseline {m['baseline']['point']})")
        out[head] = {"pass": ok, "reasons": reasons}
    return out
