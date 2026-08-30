"""
eval_static_v1.py — scores already-trained static checkpoints against a
dataset, WITHOUT training anything. Fills the one real gap
distillation_v1.py has: it only ever trains, with no "just evaluate an
existing checkpoint" mode. Needed specifically for UCO-LAEO -- the
zero-shot holdout dataset, never trained on, never split into
_TRAIN/_VAL/_TEST -- but works generically for VAT/ChildPlay/VACATION's
own _TEST populations too (the genuinely held-out, MTGS-unseen splits
already established this project), giving a real, held-out number for
those three as well, not just the main run's own VAL breakdown.

DATASET_FILES below mirrors distillation_v1.py's own table, extended
with UCO-LAEO and an explicit split_suffix per dataset: "TEST" for
VAT/ChildPlay/VACATION (their already-existing, genuinely clean held-out
split), None for UCO-LAEO (never split -- the full population IS the
test set, by design).

METRICS: compute_regression_metrics / compute_inout_metrics are copied
verbatim from distillation_v1.py's own versions (2026-08-05) rather than
imported, to keep this script fully independent of distillation_v1.py's
module-level config requirements -- same reasoning as the
GazeStudent/GazeDataset extraction into gaze_student_model.py. If you
change one, change both.

REQUIRES the in/out head to exist in the checkpoint (every
distillation_v1.py checkpoint has one) -- in/out classification metrics
are always computed, not optional.
"""

import os
import json
import math

import torch
from torch.utils.data import DataLoader

from gaze_student_model import load_gaze_student, GazeDataset

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("WARNING: scikit-learn not available -- ROC-AUC will be null.")

PCK_THRESHOLDS = [0.05, 0.10, 0.15]

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

_datasets_env = os.environ.get("EVAL_DATASETS")
if not _datasets_env:
    raise ValueError("EVAL_DATASETS not set -- e.g. 'ucolaeo' or 'vat,childplay,vacation,ucolaeo'.")
EVAL_DATASETS = [d.strip() for d in _datasets_env.split(",")]

_conditions_env = os.environ.get("EVAL_CONDITIONS")
if not _conditions_env:
    raise ValueError("EVAL_CONDITIONS not set -- e.g. 'gt' or 'gt,teacher_hybrid,teacher_internvl3_alone,mtgs'.")
EVAL_CONDITIONS = [c.strip() for c in _conditions_env.split(",")]

STATIC_CHECKPOINT_ROOT = os.environ.get("STATIC_CHECKPOINT_ROOT")
if not STATIC_CHECKPOINT_ROOT:
    raise ValueError("STATIC_CHECKPOINT_ROOT not set.")

VIT_MODEL = os.environ.get("STATIC_VIT_MODEL", "vit_tiny_patch16_224")
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "32"))

OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("EVAL_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mirrors distillation_v1.py's own DATASET_FILES exactly, plus UCO-LAEO
# and an explicit split_suffix per dataset -- "TEST" for the three
# already-split datasets' genuinely clean held-out population, None for
# UCO-LAEO (never split -- the full population IS the eval set).
DATASET_FILES = {
    "vat": {
        "full": "labels_vat_full_{split}.jsonl",
        "hybrid": "labels_vat_hybrid_targeted_{split}.jsonl",
        "mtgs": "mtgs_teacher_vat_merged_{split}.jsonl",
        "split_suffix": "TEST",
    },
    "childplay": {
        "full": "labels_childplay_full_{split}.jsonl",
        "hybrid": "labels_childplay_hybrid_targeted_{split}.jsonl",
        "mtgs": "mtgs_teacher_childplay_{split}.jsonl",
        "split_suffix": "TEST",
    },
    "vacation": {
        "full": "labels_vacation_full_{split}.jsonl",
        "hybrid": "labels_vacation_hybrid_targeted_{split}.jsonl",
        "mtgs": "mtgs_teacher_vacation_{split}.jsonl",
        "split_suffix": "TEST",
    },
    "ucolaeo": {
        "full": "labels_ucolaeo_full.jsonl",
        "hybrid": "labels_ucolaeo_hybrid_targeted.jsonl",  # the patched/FINAL version, renamed to canonical
        "mtgs": "mtgs_teacher_ucolaeo.jsonl",
        "split_suffix": None,  # never split -- full population used as-is
    },
}

DATASET_ROOTS = {}
for _ds in EVAL_DATASETS:
    if _ds not in DATASET_FILES:
        raise ValueError(f"Unknown EVAL_DATASETS entry '{_ds}' -- known: {list(DATASET_FILES)}")
    env_key = f"{_ds.upper()}_LIBRARY_ROOT"
    root = os.environ.get(env_key)
    if root is None:
        raise ValueError(f"{env_key} not set.")
    DATASET_ROOTS[_ds] = root


# ══════════════════════════════════════════════════════════════════════
# ── LOAD + JOIN — same logic as distillation_v1.py's build_dataset_split_population,
# ── copied verbatim rather than imported (see module docstring) ────────
# ══════════════════════════════════════════════════════════════════════

def load_labels_file(path):
    records = {}
    n_missing = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            required = ["pred_x", "pred_y", "gt_x", "gt_y", "img_path",
                        "head_x1", "head_y1", "head_x2", "head_y2",
                        "show", "clip", "fname", "subject", "gaze_type"]
            if any(rec.get(k) is None for k in required):
                n_missing += 1
                continue
            key = (str(rec["show"]), str(rec["clip"]), str(rec["fname"]), str(rec["subject"]))
            records[key] = rec
    print(f"    Loaded {path}: {len(records)} valid records (skipped {n_missing} missing-field)")
    return records


def build_dataset_population(dataset):
    files = DATASET_FILES[dataset]
    root = DATASET_ROOTS[dataset]
    split = files["split_suffix"]

    def path_for(kind):
        template = files[kind]
        return os.path.join(root, template.format(split=split) if split else template)

    print(f"  [{dataset}]" + (f" (split={split})" if split else " (full population, never split)"))
    full_recs = load_labels_file(path_for("full"))
    hybrid_recs = load_labels_file(path_for("hybrid"))
    mtgs_recs = load_labels_file(path_for("mtgs"))

    common = set(full_recs) & set(hybrid_recs) & set(mtgs_recs)
    print(f"    Key alignment: {len(common)} common to all three files "
          f"(full={len(full_recs)}, hybrid={len(hybrid_recs)}, mtgs={len(mtgs_recs)})")

    gt_mismatches, missing_images = 0, 0
    joined = []
    GT_MISMATCH_PIXEL_TOLERANCE = 2.0  # see distillation_v1.py's own comment on why pixel-space, not normalized
    for key in common:
        fu, h, m = full_recs[key], hybrid_recs[key], mtgs_recs[key]
        d_fh_px = math.hypot(fu["gt_px_x"] - h["gt_px_x"], fu["gt_px_y"] - h["gt_px_y"])
        d_fm_px = math.hypot(fu["gt_px_x"] - m["gt_px_x"], fu["gt_px_y"] - m["gt_px_y"])
        if d_fh_px > GT_MISMATCH_PIXEL_TOLERANCE or d_fm_px > GT_MISMATCH_PIXEL_TOLERANCE:
            gt_mismatches += 1
            continue
        if not os.path.isfile(fu["img_path"]):
            missing_images += 1
            continue

        joined.append({
            "dataset": dataset,
            "show": fu["show"], "clip": fu["clip"], "subject": fu["subject"], "fname": fu["fname"],
            "gaze_type": fu["gaze_type"],
            "is_offscreen": fu["gaze_type"] == "offscreen",
            "img_path": fu["img_path"],
            "head_x1": fu["head_x1"], "head_y1": fu["head_y1"],
            "head_x2": fu["head_x2"], "head_y2": fu["head_y2"],
            "gt_x": fu["gt_x"], "gt_y": fu["gt_y"],
            "hybrid_pred_x": h["pred_x"], "hybrid_pred_y": h["pred_y"],
            "internvl3_alone_pred_x": fu["pred_x"], "internvl3_alone_pred_y": fu["pred_y"],
            "mtgs_pred_x": m["pred_x"], "mtgs_pred_y": m["pred_y"],
        })

    print(f"    GT disagreements excluded (>{GT_MISMATCH_PIXEL_TOLERANCE}px): {gt_mismatches}")
    print(f"    Missing images excluded: {missing_images}")
    print(f"    Final [{dataset}] eval population: {len(joined)} records")
    return joined


def build_full_eval_population(datasets):
    records = []
    for dataset in datasets:
        records.extend(build_dataset_population(dataset))
    print(f"\nEVAL population: {len(records)} total records")
    return records


# ══════════════════════════════════════════════════════════════════════
# ── METRICS — copied verbatim from distillation_v1.py (see module docstring) ──
# ══════════════════════════════════════════════════════════════════════

def compute_regression_metrics(dists, errs_x, errs_y):
    n = len(dists)
    if n == 0:
        return None
    ade = sum(dists) / n
    mae_x = sum(abs(e) for e in errs_x) / n
    mae_y = sum(abs(e) for e in errs_y) / n
    rmse = math.sqrt(sum(d ** 2 for d in dists) / n)
    pck = {f"pck@{t}": sum(1 for d in dists if d <= t) / n for t in PCK_THRESHOLDS}
    return {"n": n, "ade": ade, "mae_x": mae_x, "mae_y": mae_y, "rmse": rmse, **pck}


def compute_inout_metrics(y_true, y_prob, threshold=0.5):
    n = len(y_true)
    if n == 0:
        return None
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    roc_auc = None
    if HAVE_SKLEARN and len(set(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc = None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


def evaluate(model, loader, device, condition=None):
    """Same logic as distillation_v1.py's evaluate() -- pooled + by-dataset
    + by-gaze_type regression metrics, plus in/out classification metrics.

    ADDED 2026-08-16: if condition is given, also saves the raw per-row
    (inout_true, inout_prob) pairs to a separate file -- enables a
    threshold sweep later with zero re-inference (the expensive part,
    running the model on 70K+ images, stays separate from the cheap part,
    trying many thresholds against numbers already computed). Existing
    return signature is completely unchanged -- this is purely additive,
    an internal save step plus one new optional parameter, so the one
    existing caller works unmodified if condition is left as None."""
    model.eval()
    dists, errs_x, errs_y = [], [], []
    by_dataset_dists, by_gtype_dists = {}, {}
    inout_true, inout_prob = [], []
    by_dataset_inout_true, by_dataset_inout_prob = {}, {}
    per_record_dataset, per_record_gaze_type = [], []  # parallel to dists -- see export below
    # ADDED (Opus review, frame-boundary instability check): predicted
    # coordinates + identity, parallel to dists -- lets a later script
    # measure frame-to-frame movement of the STUDENT'S OWN predictions,
    # not just labels, without any re-inference.
    per_record_pred_x, per_record_pred_y = [], []
    per_record_clip, per_record_subject, per_record_fname = [], [], []
    per_record_show = []  # Opus review: (show, clip, subject) is the real track identity in VAT --
                           # clip IDs alone can collide across different shows.
    per_record_gt_x, per_record_gt_y = [], []  # Opus review: required for the near-stationary
                                                # ground-truth filter -- per_record_ade's scalar
                                                # distance can't be un-collapsed back into these.

    with torch.inference_mode():
        for batch in loader:
            scene = batch["scene"].to(device)
            head = batch["head"].to(device)
            target = batch["target"]
            is_offscreen = batch["is_offscreen"]
            gaze_types = batch["gaze_type"]
            datasets_ = batch["dataset"]
            clips_ = batch["clip"]
            subjects_ = batch["subject"]
            fnames_ = batch["fname"]
            shows_ = batch["show"]

            coord_pred, inout_logit = model(scene, head)
            coord_pred = coord_pred.cpu()
            inout_prob_batch = torch.sigmoid(inout_logit).cpu()

            on_screen_mask = is_offscreen < 0.5
            if on_screen_mask.any():
                d = torch.sqrt(((coord_pred - target) ** 2).sum(dim=1))
                ex = coord_pred[:, 0] - target[:, 0]
                ey = coord_pred[:, 1] - target[:, 1]
                for i in range(len(d)):
                    if not on_screen_mask[i]:
                        continue
                    dists.append(d[i].item()); errs_x.append(ex[i].item()); errs_y.append(ey[i].item())
                    per_record_dataset.append(datasets_[i]); per_record_gaze_type.append(gaze_types[i])
                    per_record_pred_x.append(coord_pred[i][0].item())
                    per_record_pred_y.append(coord_pred[i][1].item())
                    per_record_clip.append(clips_[i]); per_record_subject.append(subjects_[i])
                    per_record_fname.append(fnames_[i]); per_record_show.append(shows_[i])
                    per_record_gt_x.append(target[i][0].item()); per_record_gt_y.append(target[i][1].item())
                    by_dataset_dists.setdefault(datasets_[i], ([], [], []))
                    by_dataset_dists[datasets_[i]][0].append(d[i].item())
                    by_dataset_dists[datasets_[i]][1].append(ex[i].item())
                    by_dataset_dists[datasets_[i]][2].append(ey[i].item())
                    by_gtype_dists.setdefault(gaze_types[i], ([], [], []))
                    by_gtype_dists[gaze_types[i]][0].append(d[i].item())
                    by_gtype_dists[gaze_types[i]][1].append(ex[i].item())
                    by_gtype_dists[gaze_types[i]][2].append(ey[i].item())

            for i in range(len(is_offscreen)):
                t = int(is_offscreen[i].item())
                p = float(inout_prob_batch[i].item())
                inout_true.append(t); inout_prob.append(p)
                by_dataset_inout_true.setdefault(datasets_[i], []).append(t)
                by_dataset_inout_prob.setdefault(datasets_[i], []).append(p)

    pooled = compute_regression_metrics(dists, errs_x, errs_y)
    by_dataset = {k: compute_regression_metrics(*v) for k, v in by_dataset_dists.items()}
    by_gaze_type = {k: compute_regression_metrics(*v) for k, v in by_gtype_dists.items()}
    inout_pooled = compute_inout_metrics(inout_true, inout_prob)
    inout_by_dataset = {k: compute_inout_metrics(by_dataset_inout_true[k], by_dataset_inout_prob[k])
                         for k in by_dataset_inout_true}

    if condition is not None:
        raw_path = os.path.join(OUTPUT_DIR, f"inout_raw_probs_{condition}.json")
        with open(raw_path, "w") as f:
            json.dump({"inout_true": inout_true, "inout_prob": inout_prob}, f)
        print(f"  Saved raw in/out probabilities for threshold sweep: {raw_path}")

        # Per-record static distances -- enables a genuine paired bootstrap CI
        # on the static four-way ranking, matching the same rigor already
        # applied to every temporal claim (rather than a variance estimate
        # borrowed from a different model/condition). Same "expensive part
        # stays separate from cheap part" split as the threshold sweep.
        per_record_path = os.path.join(OUTPUT_DIR, f"per_record_ade_{condition}.json")
        with open(per_record_path, "w") as f:
            json.dump({"per_record_ade": dists, "dataset": per_record_dataset,
                        "gaze_type": per_record_gaze_type}, f)
        print(f"  Saved per-record static distances for bootstrap CI: {per_record_path}")

        # Predicted coordinates + identity -- enables measuring frame-to-frame
        # movement of the STUDENT'S OWN predictions (Opus review: does the
        # student inherit the teacher's frame-boundary instability, dampen
        # it, or amplify it?). Same records, same order as per_record_ade
        # above -- can be joined by position if needed.
        per_record_pred_path = os.path.join(OUTPUT_DIR, f"per_record_predictions_{condition}.json")
        with open(per_record_pred_path, "w") as f:
            json.dump({"pred_x": per_record_pred_x, "pred_y": per_record_pred_y,
                        "gt_x": per_record_gt_x, "gt_y": per_record_gt_y,
                        "show": per_record_show, "clip": per_record_clip,
                        "subject": per_record_subject, "gaze_type": per_record_gaze_type,
                        "fname": per_record_fname, "dataset": per_record_dataset}, f)
        print(f"  Saved per-record predicted coordinates + identity: {per_record_pred_path}")

    return pooled, by_dataset, by_gaze_type, inout_pooled, inout_by_dataset


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Datasets: {EVAL_DATASETS}")
    print(f"Conditions: {EVAL_CONDITIONS}")
    print(f"Device: {DEVICE}\n")

    records = build_full_eval_population(EVAL_DATASETS)
    if len(records) < 10:
        print("ERROR: too few eval records -- check paths/schema before rerunning.")
        return

    all_results = {}
    for condition in EVAL_CONDITIONS:
        print(f"\n{'='*70}\nEvaluating condition: {condition}\n{'='*70}")
        ckpt_path = os.path.join(STATIC_CHECKPOINT_ROOT, f"model_{condition}", "best_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  SKIPPED -- no checkpoint found at {ckpt_path}.")
            continue

        model = load_gaze_student(ckpt_path, vit_model=VIT_MODEL, device=DEVICE)
        eval_ds = GazeDataset(records, condition, augment=False)
        eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        pooled, by_dataset, by_gaze_type, inout_pooled, inout_by_dataset = evaluate(model, eval_loader, DEVICE, condition=condition)

        print(f"  Pooled ADE: {pooled['ade']:.4f}  (n={pooled['n']})")
        for ds, m in sorted(by_dataset.items()):
            print(f"    {ds:<12} ADE={m['ade']:.4f}  n={m['n']}")
        for gt, m in sorted(by_gaze_type.items()):
            print(f"    {gt:<12} ADE={m['ade']:.4f}  n={m['n']}")
        if inout_pooled:
            print(f"  In/out: precision={inout_pooled['precision']:.3f} "
                  f"recall={inout_pooled['recall']:.3f} f1={inout_pooled['f1']:.3f} "
                  f"roc_auc={inout_pooled['roc_auc']}")

        all_results[condition] = {
            "checkpoint": ckpt_path, "pooled": pooled, "by_dataset": by_dataset,
            "by_gaze_type": by_gaze_type, "inout_pooled": inout_pooled,
            "inout_by_dataset": inout_by_dataset,
        }

    output = {
        "run_metadata": {"datasets": EVAL_DATASETS, "conditions": EVAL_CONDITIONS,
                          "architecture": VIT_MODEL, "n_eval_records": len(records)},
        "results": all_results,
    }
    out_path = os.path.join(OUTPUT_DIR, "eval_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
