"""
compute_trivial_baselines.py — image-centre, head-box-centre, and
training-mean ADE against the same joined _TEST population every real
condition is scored against. No GPU, no model inference — pure
arithmetic over the existing label files, reusing eval_static_v1.py's
own population-joining logic (copied, not imported, same reasoning as
every other verbatim copy in this project: avoids its module-level env
var requirements firing just to reuse a function).

WHY THIS MATTERS: every static ADE reported in this dissertation (Table
4.1 etc.) is compared only against other conditions -- never against an
absolute floor showing the numbers are meaningfully better than doing
nothing informed at all. This closes that gap directly.

THREE BASELINES:
  image-centre:     constant (0.5, 0.5) prediction, every record.
  head-box-centre:  the subject's own head-box centre, normalised by
                     that record's real image dimensions (requires
                     opening each image once -- PIL is lazy, this reads
                     just the header, not full pixel data).
  training-mean:     the mean (gt_x, gt_y) across the POOLED VAT+
                     ChildPlay+VACATION training population -- matching
                     exactly what the real student models are trained
                     on -- applied as one constant to every test record,
                     including UCO-LAEO. This is deliberate, not an
                     oversight: the real student never saw UCO-LAEO's
                     own training distribution either, so using the
                     same pooled training-mean there is the correct,
                     apples-to-apples comparison, not a per-dataset one.

Usage: same env vars as eval_static_v1.py's own population-building
step (EVAL_DATASETS, and *_LIBRARY_ROOT per dataset), no
STATIC_CHECKPOINT_ROOT or condition needed since no model is loaded.
"""

import os
import json
import math

from PIL import Image

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION — same convention as eval_static_v1.py ───────────────
# ══════════════════════════════════════════════════════════════════════

_datasets_env = os.environ.get("EVAL_DATASETS")
if not _datasets_env:
    raise ValueError("EVAL_DATASETS not set -- e.g. 'vat,childplay,vacation,ucolaeo'.")
EVAL_DATASETS = [d.strip() for d in _datasets_env.split(",")]

OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("EVAL_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCK_THRESHOLDS = [0.05, 0.10, 0.15]

DATASET_FILES = {
    "vat": {"full": "labels_vat_full_{split}.jsonl", "hybrid": "labels_vat_hybrid_targeted_{split}.jsonl",
            "mtgs": "mtgs_teacher_vat_merged_{split}.jsonl", "split_suffix": "TEST"},
    "childplay": {"full": "labels_childplay_full_{split}.jsonl", "hybrid": "labels_childplay_hybrid_targeted_{split}.jsonl",
                  "mtgs": "mtgs_teacher_childplay_{split}.jsonl", "split_suffix": "TEST"},
    "vacation": {"full": "labels_vacation_full_{split}.jsonl", "hybrid": "labels_vacation_hybrid_targeted_{split}.jsonl",
                 "mtgs": "mtgs_teacher_vacation_{split}.jsonl", "split_suffix": "TEST"},
    "ucolaeo": {"full": "labels_ucolaeo_full.jsonl", "hybrid": "labels_ucolaeo_hybrid_targeted.jsonl",
                "mtgs": "mtgs_teacher_ucolaeo.jsonl", "split_suffix": None},
}
# Only these three ever have a real TRAIN split for the training-mean baseline.
TRAINING_MEAN_SOURCE_DATASETS = ["vat", "childplay", "vacation"]

DATASET_ROOTS = {}
for _ds in EVAL_DATASETS:
    if _ds not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{_ds}' -- known: {list(DATASET_FILES)}")
    env_key = f"{_ds.upper()}_LIBRARY_ROOT"
    root = os.environ.get(env_key)
    if root is None:
        raise ValueError(f"{env_key} not set.")
    DATASET_ROOTS[_ds] = root


# ══════════════════════════════════════════════════════════════════════
# ── LOAD + JOIN — copied verbatim from eval_static_v1.py ───────────────
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


def build_dataset_population(dataset, split_override=None):
    files = DATASET_FILES[dataset]
    root = DATASET_ROOTS[dataset]
    split = split_override if split_override is not None else files["split_suffix"]

    def path_for(kind):
        template = files[kind]
        return os.path.join(root, template.format(split=split) if split else template)

    print(f"  [{dataset}]" + (f" (split={split})" if split else " (full population, never split)"))
    full_recs = load_labels_file(path_for("full"))
    hybrid_recs = load_labels_file(path_for("hybrid"))
    mtgs_recs = load_labels_file(path_for("mtgs"))

    common = set(full_recs) & set(hybrid_recs) & set(mtgs_recs)
    print(f"    Key alignment: {len(common)} common to all three files")

    gt_mismatches, missing_images = 0, 0
    joined = []
    GT_MISMATCH_PIXEL_TOLERANCE = 2.0
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
            "dataset": dataset, "gaze_type": fu["gaze_type"],
            "is_offscreen": fu["gaze_type"] == "offscreen",
            "img_path": fu["img_path"],
            "head_x1": fu["head_x1"], "head_y1": fu["head_y1"],
            "head_x2": fu["head_x2"], "head_y2": fu["head_y2"],
            "gt_x": fu["gt_x"], "gt_y": fu["gt_y"],
        })

    print(f"    GT disagreements excluded (>{GT_MISMATCH_PIXEL_TOLERANCE}px): {gt_mismatches}")
    print(f"    Missing images excluded: {missing_images}")
    print(f"    Final [{dataset}] population: {len(joined)} records")
    return joined


def build_full_eval_population(datasets):
    records = []
    for dataset in datasets:
        records.extend(build_dataset_population(dataset))
    print(f"\nEVAL population: {len(records)} total records")
    return records


# ══════════════════════════════════════════════════════════════════════
# ── METRICS — same as eval_static_v1.py's own compute_regression_metrics ──
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


def score_baseline(records, pred_fn, name, needs_image_dims=False):
    """pred_fn(record, img_w, img_h) -> (pred_x, pred_y) in normalised
    [0,1] space. img_w/img_h are None if needs_image_dims=False -- saves
    opening every image for baselines that don't need real dimensions."""
    dists, errs_x, errs_y = [], [], []
    by_dataset_dists, by_gtype_dists = {}, {}
    n_onscreen = 0
    img_open_failures = 0

    for rec in records:
        if rec["is_offscreen"]:
            continue
        n_onscreen += 1

        img_w, img_h = None, None
        if needs_image_dims:
            try:
                with Image.open(rec["img_path"]) as img:
                    img_w, img_h = img.size
            except Exception:
                img_open_failures += 1
                continue

        pred_x, pred_y = pred_fn(rec, img_w, img_h)
        gt_x, gt_y = rec["gt_x"], rec["gt_y"]
        d = math.hypot(pred_x - gt_x, pred_y - gt_y)
        ex, ey = pred_x - gt_x, pred_y - gt_y

        dists.append(d); errs_x.append(ex); errs_y.append(ey)
        ds, gt = rec["dataset"], rec["gaze_type"]
        by_dataset_dists.setdefault(ds, ([], [], []))
        by_dataset_dists[ds][0].append(d); by_dataset_dists[ds][1].append(ex); by_dataset_dists[ds][2].append(ey)
        by_gtype_dists.setdefault(gt, ([], [], []))
        by_gtype_dists[gt][0].append(d); by_gtype_dists[gt][1].append(ex); by_gtype_dists[gt][2].append(ey)

    if img_open_failures:
        print(f"    WARNING [{name}]: {img_open_failures}/{n_onscreen} images failed to open, excluded from this baseline")

    pooled = compute_regression_metrics(dists, errs_x, errs_y)
    by_dataset = {k: compute_regression_metrics(*v) for k, v in by_dataset_dists.items()}
    by_gaze_type = {k: compute_regression_metrics(*v) for k, v in by_gtype_dists.items()}
    return pooled, by_dataset, by_gaze_type


def compute_training_mean(datasets_for_mean):
    """Pooled mean (gt_x, gt_y) across VAT+ChildPlay+VACATION's real
    TRAIN split -- matching exactly what the real student models are
    trained on. UCO-LAEO deliberately excluded here (no TRAIN split
    exists), but the resulting constant IS applied to UCO-LAEO's own
    test records later, since the real student never saw UCO-LAEO
    training data either -- that's the correct comparison, not an
    oversight."""
    all_x, all_y = [], []
    print("\nComputing training-mean baseline source (TRAIN split, on-screen only):")
    for dataset in datasets_for_mean:
        recs = build_dataset_population(dataset, split_override="TRAIN")
        for r in recs:
            if not r["is_offscreen"]:
                all_x.append(r["gt_x"]); all_y.append(r["gt_y"])
    mean_x, mean_y = sum(all_x) / len(all_x), sum(all_y) / len(all_y)
    print(f"  Pooled training mean (n={len(all_x)}): ({mean_x:.4f}, {mean_y:.4f})\n")
    return mean_x, mean_y


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Datasets: {EVAL_DATASETS}\n")
    records = build_full_eval_population(EVAL_DATASETS)
    if len(records) < 10:
        print("ERROR: too few eval records -- check paths/schema before rerunning.")
        return

    train_mean_sources = [d for d in TRAINING_MEAN_SOURCE_DATASETS if d in EVAL_DATASETS] or TRAINING_MEAN_SOURCE_DATASETS
    mean_x, mean_y = compute_training_mean(train_mean_sources)

    baselines = {
        "image_centre": (lambda rec, w, h: (0.5, 0.5), False),
        "head_box_centre": (lambda rec, w, h: (
            ((rec["head_x1"] + rec["head_x2"]) / 2) / w,
            ((rec["head_y1"] + rec["head_y2"]) / 2) / h,
        ), True),
        "training_mean": (lambda rec, w, h: (mean_x, mean_y), False),
    }

    all_results = {}
    for name, (pred_fn, needs_dims) in baselines.items():
        print(f"\n{'='*70}\nBaseline: {name}\n{'='*70}")
        pooled, by_dataset, by_gaze_type = score_baseline(records, pred_fn, name, needs_image_dims=needs_dims)
        print(f"  Pooled ADE: {pooled['ade']:.4f}  (n={pooled['n']})")
        for ds, m in sorted(by_dataset.items()):
            print(f"    {ds:<12} ADE={m['ade']:.4f}  n={m['n']}")
        for gt, m in sorted(by_gaze_type.items()):
            print(f"    {gt:<12} ADE={m['ade']:.4f}  n={m['n']}")
        all_results[name] = {"pooled": pooled, "by_dataset": by_dataset, "by_gaze_type": by_gaze_type}

    output = {
        "run_metadata": {"datasets": EVAL_DATASETS, "n_eval_records": len(records),
                          "training_mean_xy": [mean_x, mean_y],
                          "training_mean_source_datasets": train_mean_sources},
        "results": all_results,
    }
    out_path = os.path.join(OUTPUT_DIR, "trivial_baselines.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
