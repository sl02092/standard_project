"""
threshold_sweep_inout.py — sweeps the in/out (offscreen) decision
threshold against ALREADY-COMPUTED probabilities (from eval_static_v1.py's
own inout_raw_probs_{condition}.json export), and reports precision/
recall/F1 at each. No model, no GPU, no re-inference -- this is why it's
essentially free once the export exists: the expensive part (running the
model on 70K+ images) already happened during the normal eval run.

Converts a named-but-untested limitation (default 0.5 threshold likely
poorly calibrated for a rare-positive-class problem -- offscreen is only
~8.7% of rows) into an actual, citable result: the best achievable F1 at
the right threshold, not just "recall was low at 0.5."

RUN LOCALLY, or on HPC with plain python3 -- no apptainer/venv needed,
this only needs the json module.
"""

import os
import json


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_prob):
        pred = 1 if p >= threshold else 0
        if pred == 1 and t == 1: tp += 1
        elif pred == 1 and t == 0: fp += 1
        elif pred == 0 and t == 1: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    raw_path = os.environ.get("INOUT_RAW_PROBS_PATH")
    if not raw_path:
        raise ValueError("INOUT_RAW_PROBS_PATH not set -- point at an "
                          "inout_raw_probs_{condition}.json file from eval_static_v1.py.")

    with open(raw_path) as f:
        data = json.load(f)
    y_true, y_prob = data["inout_true"], data["inout_prob"]
    n_positive = sum(y_true)
    print(f"Loaded {len(y_true)} rows ({n_positive} genuine offscreen, "
          f"{100*n_positive/len(y_true):.1f}% positive rate) from {raw_path}\n")

    thresholds = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                  0.0006, 0.0007, 0.0008, 0.0009,
                  0.001, 0.002, 0.005, 0.01, 0.02,
                  0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print(f"{'threshold':<12}{'precision':<12}{'recall':<12}{'f1':<12}{'tp':<8}{'fp':<8}{'fn':<8}{'tn':<8}")
    results = []
    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_prob, t)
        results.append(m)
        print(f"{t:<12.2f}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1']:<12.4f}"
              f"{m['tp']:<8}{m['fp']:<8}{m['fn']:<8}{m['tn']:<8}")

    best_f1 = max(results, key=lambda m: m["f1"])
    default_result = next(m for m in results if abs(m["threshold"] - 0.5) < 1e-9)

    print(f"\nDefault threshold (0.5): F1={default_result['f1']:.4f}  "
          f"precision={default_result['precision']:.4f}  recall={default_result['recall']:.4f}")
    print(f"Best F1 at threshold={best_f1['threshold']}: F1={best_f1['f1']:.4f}  "
          f"precision={best_f1['precision']:.4f}  recall={best_f1['recall']:.4f}")
    improvement = best_f1['f1'] - default_result['f1']
    if default_result['f1'] > 0:
        print(f"Improvement over default: {improvement:+.4f} "
              f"({100*improvement/default_result['f1']:+.1f}% relative)")

    out_path = raw_path.replace("inout_raw_probs_", "inout_threshold_sweep_")
    with open(out_path, "w") as f:
        json.dump({"n_rows": len(y_true), "n_positive": n_positive,
                    "sweep": results, "default_threshold_result": default_result,
                    "best_f1_result": best_f1}, f, indent=2)
    print(f"\nFull sweep saved: {out_path}")


if __name__ == "__main__":
    main()
