"""
distillation_v1.py — student training v1.0. This is the TRUE source of
truth for student training going forward, not the older
distillation_four_way_real.py / distillation_joint_vat_childplay_gt.py
scripts, which are kept purely as reference for what worked (the
GazeStudent architecture, the CONDITION_FIELDS join pattern, confidence-
thresholding, resumability) — not as a compatibility target. Per-project
decision (2026-08-04): prior student training was exploratory/PoC only,
never intended for the paper, so this rewrite owes no reproducibility
debt to the old scripts' specific split logic or behaviour.

WHAT'S GENUINELY NEW vs. the old scripts, and why:

1. READS THE NEW _TRAIN/_VAL/_TEST FILE STRUCTURE DIRECTLY. The old
   scripts read one arbitrary file per condition and built their own
   from-scratch random 80/20 clip split, with zero awareness of MTGS
   contamination. This script reads the already-canonical, already-clean
   per-dataset split files (built 2026-08-04, dataset-native boundaries:
   VAT's TRAIN_SHOWS/VAL_SHOWS/TEST_SHOWS, ChildPlay's own
   childplay_{train,val,test}.h5 -- confirmed identical to ChildPlay's
   own clips.csv split, exact set match, both same day -- and VACATION's
   official readme.txt split). No splitting logic lives in this script
   at all; the files themselves already encode the correct split.

2. THREE DATASETS, NOT ONE. build_dataset_population() is written once,
   generically, and called per dataset via the DATASETS env var (see
   DATASET_FILES below) -- this is what makes the SAME script serve both
   the joint VAT+ChildPlay+VACATION main run and a single-dataset
   confidence-threshold sweep run, just by changing which datasets are
   listed.

3. IN/OUT HEAD -- genuinely new model output, not just new logging.
   gaze_type=="offscreen" is a real, already-present ground-truth label
   in every row of every file (it describes the true annotated state of
   the frame, independent of any teacher condition) -- costs zero new
   data engineering. GazeStudent gets a second output (a single logit)
   trained with BCEWithLogitsLoss. The coordinate regression loss is
   MASKED to exclude offscreen rows (their gt_x/gt_y are a sentinel
   fraction, not a real target -- training the regression head against
   them would inject meaningless targets into that task) -- but offscreen
   rows ARE included in the population and DO contribute to the in/out
   loss, which needs real positive examples to learn from. This is why
   offscreen rows are no longer filtered out entirely the way the old
   scripts filtered them (their `n_offscreen` exclusion made sense for a
   regression-only model; it would silently starve this model's in/out
   head of its only positive class).

4. FULL DETERMINISM, not just seed_everything(). The old scripts' seed
   only covers random/numpy/torch.manual_seed -- given this project has
   now found and fixed real GPU/library non-determinism TWICE (the
   temporal PoC's cuDNN fix, and this week's UCO-LAEO inference-noise
   investigation), seed-only is not trusted here. See
   setup_full_determinism() below.

5. EXTENDED METRICS, regression-appropriate (not classification metrics
   bolted onto a regression task -- see conversation 2026-08-04 for why
   accuracy/precision/recall/F1/ROC-AUC/mAP/IoU don't fit a coordinate
   regression output, but DO legitimately apply to the new in/out head
   specifically). Computed pooled + by-dataset + by-gaze_type:
     - ADE (mean Euclidean distance -- unchanged core metric)
     - MAE_x, MAE_y (per-axis, not just pooled)
     - RMSE
     - PCK@{0.05, 0.10, 0.15} (% of predictions within that normalized
       distance of GT -- the standard way to turn a regression result
       into an accuracy-style number for a gaze/pose paper)
   Plus in/out classification metrics (precision, recall, F1, ROC-AUC,
   TP/FP/FN/TN) computed ONCE, pooled + by-dataset (gaze_type doesn't
   apply here since offscreen IS the label being predicted).
   All written to structured, incrementally-updated JSON -- built to be
   parsed programmatically for plotting/statistical analysis later, not
   just human-readable.

6. CONFIDENCE_THRESHOLD kept exactly as before (0.55, hybrid-condition-
   only) but promoted to a real, clearly-labelled top-level constant so a
   sweep is a pure env-var override, no code changes -- see the
   accompanying .sub files for the sweep runs (VAT/ChildPlay/VACATION x
   0.35/0.55/0.65, single-dataset each, plus 3 optional joint-population
   confirmatory runs).

REUSED, DELIBERATELY UNCHANGED FROM THE OLD SCRIPTS (already validated,
no reason to re-derive):
  - GazeStudent's dual-ViT-encoder architecture (scene + head crop)
  - The CONDITION_FIELDS join pattern (one record carries all four
    conditions' predictions; a single Dataset class picks the right
    field per condition -- this is what prevented the old
    confidence-thresholding-leaks-across-conditions bug from recurring)
  - Confidence-thresholding logic itself (get_gdino_score /
    passes_confidence_threshold), hybrid-condition-only
  - Resumability (skip a condition if its best_model.pt already exists)
  - Incremental comparison.json writes after every condition
"""

import os
import json
import math
import time
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gaze_student_model import GazeStudent, GazeDataset, count_params, setup_full_determinism, seeded_generator

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("WARNING: scikit-learn not available -- ROC-AUC will be reported "
          "as null in output JSON. `pip install scikit-learn` to enable it.")


# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

# Which datasets to load -- comma-separated, e.g. "vat,childplay,vacation"
# for the main joint run, or just "childplay" for a single-dataset
# confidence-threshold sweep run. No default: deliberately explicit,
# matching this project's established "no silent fallback on a path/scope
# that matters" convention.
_datasets_env = os.environ.get("DISTILL_DATASETS")
if not _datasets_env:
    raise ValueError(
        "DISTILL_DATASETS not set -- e.g. 'vat,childplay,vacation' for the "
        "joint run, or 'childplay' alone for a single-dataset sweep run. "
        "Pass via the .sub file, not a code default."
    )
DATASETS = [d.strip() for d in _datasets_env.split(",")]

ALL_CONDITIONS = ["gt", "teacher_hybrid", "teacher_internvl3_alone", "mtgs"]
_condition_filter = os.environ.get("DISTILL_CONDITIONS")
CONDITIONS = ([c.strip() for c in _condition_filter.split(",")]
              if _condition_filter else ALL_CONDITIONS)

# Per-dataset file roots -- each must point at the directory containing
# that dataset's already-split _TRAIN/_VAL/_TEST files. No defaults, same
# reasoning as DISTILL_DATASETS above.
DATASET_ROOTS = {}
for _ds in DATASETS:
    env_key = f"{_ds.upper()}_LIBRARY_ROOT"
    root = os.environ.get(env_key)
    if root is None:
        raise ValueError(
            f"{env_key} not set -- point it at the directory containing "
            f"{_ds}'s _TRAIN/_VAL/_TEST label + mtgs_teacher files."
        )
    DATASET_ROOTS[_ds] = root

# Per-dataset filenames -- NOT a single string formula. VAT's mtgs file
# carries an extra "_merged" token (from 2026-08-04's train/testshows
# merge) that ChildPlay's and VACATION's don't -- confirmed directly
# against real filenames, not assumed. Getting this wrong silently loads
# nothing or the wrong file, so it's spelled out explicitly per dataset
# rather than templated.
DATASET_FILES = {
    "vat": {
        "full": "labels_vat_full_{split}.jsonl",
        "hybrid": "labels_vat_hybrid_targeted_{split}.jsonl",
        "mtgs": "mtgs_teacher_vat_merged_{split}.jsonl",
    },
    "childplay": {
        "full": "labels_childplay_full_{split}.jsonl",
        "hybrid": "labels_childplay_hybrid_targeted_{split}.jsonl",
        "mtgs": "mtgs_teacher_childplay_{split}.jsonl",
    },
    "vacation": {
        "full": "labels_vacation_full_{split}.jsonl",
        "hybrid": "labels_vacation_hybrid_targeted_{split}.jsonl",
        "mtgs": "mtgs_teacher_vacation_{split}.jsonl",
    },
}
for _ds in DATASETS:
    if _ds not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{_ds}' -- no filename mapping "
                          f"defined. Known: {list(DATASET_FILES.keys())}")

OUTPUT_DIR = os.environ.get("DISTILL_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("DISTILL_OUTPUT_DIR not set.")

VIT_MODEL = os.environ.get("DISTILL_VIT_MODEL", "vit_tiny_patch16_224")
IMG_SIZE = 224
HEAD_CROP_SIZE = 112

EPOCHS = int(os.environ.get("DISTILL_EPOCHS", "15"))
BATCH_SIZE = int(os.environ.get("DISTILL_BATCH_SIZE", "32"))
LR = float(os.environ.get("DISTILL_LR", "3e-5"))
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6
SEED = int(os.environ.get("DISTILL_SEED", "42"))
NUM_WORKERS = int(os.environ.get("DISTILL_NUM_WORKERS", "4"))
COLLAPSE_WARNING_RATIO = 0.3
RESUME = os.environ.get("DISTILL_RESUME", "0") == "1"

CONFIDENCE_THRESHOLD_RAW = os.environ.get("DISTILL_CONFIDENCE_THRESHOLD", "0.55")
CONFIDENCE_THRESHOLD = None if CONFIDENCE_THRESHOLD_RAW.lower() == "none" else float(CONFIDENCE_THRESHOLD_RAW)

# GT-consistency check tolerance, in PIXELS, not normalized fraction.
# Originally a normalized-space check (fixed 1e-3), inherited directly
# from UCO-LAEO's own sub-pixel-noise investigation without re-deriving
# it per-dataset -- confirmed via diagnose_vacation_gt_mismatch.py
# (2026-08-04) that this incorrectly flagged 46.8% of VACATION's TRAIN
# population as "GT disagreement" when it was genuine benign rounding
# noise (every example within 1px per axis), misfiring only because
# VACATION's videos are lower resolution than what that threshold
# assumed. Pixel-space comparison is resolution-independent by
# construction -- the same 1-2px of real rounding noise reads as the
# same small number regardless of a video being 640x360 or 1920x1080,
# so one tolerance value genuinely works across all three datasets.
GT_MISMATCH_PIXEL_TOLERANCE = float(os.environ.get("DISTILL_GT_MISMATCH_PIXEL_TOLERANCE", "2.0"))

# In/out head loss weight -- coordinate regression (SmoothL1, scale ~0.01-
# 0.3 typically for normalized coords) and BCE (scale ~0.1-1.0 typically)
# are not naturally on the same scale. 1.0 is a reasonable starting
# point, NOT independently validated -- worth watching both loss curves
# separately in the calibration run and adjusting if one dominates.
INOUT_LOSS_WEIGHT = float(os.environ.get("DISTILL_INOUT_LOSS_WEIGHT", "1.0"))

PCK_THRESHOLDS = [0.05, 0.10, 0.15]

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for cond in CONDITIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, f"model_{cond}"), exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# ── DETERMINISM ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
# setup_full_determinism / seeded_generator now live in
# gaze_student_model.py (shared, dependency-free) -- imported above.
# Moved out so temporal scripts can reuse them without needing this
# file's own module-level config checks to be satisfied. See that
# module's docstring for the full reasoning.


# ══════════════════════════════════════════════════════════════════════
# ── LOAD + JOIN — per dataset, per split ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_labels_file(path):
    """Loads one label file. UNLIKE the old scripts, does NOT filter out
    offscreen rows here -- they're needed for the in/out head. Filtering
    (regression-loss masking) happens per-batch in the training loop
    instead, not at load time."""
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
    print(f"    Loaded {path}: {len(records)} valid records "
          f"(skipped {n_missing} missing-field)")
    return records


def get_gdino_score(rec):
    if rec.get("hybrid_label_source") != "teacher_hybrid":
        return None
    try:
        audit = json.loads(rec.get("hybrid_raw_response") or "{}")
        return audit.get("gdino_score")
    except (json.JSONDecodeError, TypeError):
        return None


def passes_confidence_threshold(rec, threshold):
    if threshold is None:
        return True
    score = get_gdino_score(rec)
    if score is None:
        return True
    return score >= threshold


def build_dataset_split_population(dataset, split):
    """Joins full/hybrid/mtgs for ONE dataset, ONE split (already-clean,
    already-correctly-split file -- no splitting logic here at all, the
    file itself IS the split). Returns a list of unified records, one per
    key, carrying all four conditions' predictions -- same
    CONDITION_FIELDS join pattern as the old scripts, extended to a
    generic per-dataset function instead of being VAT-specific.
    """
    files = DATASET_FILES[dataset]
    root = DATASET_ROOTS[dataset]
    full_path = os.path.join(root, files["full"].format(split=split))
    hybrid_path = os.path.join(root, files["hybrid"].format(split=split))
    mtgs_path = os.path.join(root, files["mtgs"].format(split=split))

    print(f"  [{dataset}:{split}]")
    full_recs = load_labels_file(full_path)
    hybrid_recs = load_labels_file(hybrid_path)
    mtgs_recs = load_labels_file(mtgs_path)

    full_keys, hybrid_keys, mtgs_keys = set(full_recs), set(hybrid_recs), set(mtgs_recs)
    common = full_keys & hybrid_keys & mtgs_keys
    print(f"    Key alignment: {len(common)} common to all three files "
          f"(full={len(full_keys)}, hybrid={len(hybrid_keys)}, mtgs={len(mtgs_keys)})")
    if len(common) < 0.99 * len(full_keys):
        print(f"    WARNING: more than 1% of records failed to align -- "
              f"verify these three files really describe the same "
              f"population before trusting anything downstream.")

    gt_mismatches = 0
    missing_images = 0
    joined = []
    # sorted(), NOT bare iteration -- found and fixed 2026-08-07. Set
    # iteration order isn't stable across separate process launches
    # (depends on per-process-randomized string hashes), so two
    # independent runs of this exact function, same data, same seed,
    # would still build train_records in a different STARTING order --
    # and shuffling a differently-ordered list with the same seed produces
    # a genuinely different final order, not the same one. Confirmed real,
    # not theoretical: the main run's teacher_hybrid (this process) and
    # the standalone joint sweep's teacher_hybrid (a separate process,
    # identical config, identical confirmed reclassification count)
    # produced different best_epoch/ADE (8/0.2321 vs 1/0.2196) despite
    # "full determinism" being enabled -- this was the actual cause.
    # This exact bug class was already found and fixed once before in
    # this project, for the temporal pipeline's build_clip_split()
    # ("step52's non-deterministic list(set(...)) bug") -- the same fix
    # (sort before shuffling) wasn't carried over to this file originally.
    for key in sorted(common):
        fu, h, m = full_recs[key], hybrid_recs[key], mtgs_recs[key]
        # PIXEL-SPACE comparison -- see GT_MISMATCH_PIXEL_TOLERANCE's own
        # comment for why this replaced an earlier normalized-space check
        # that incorrectly dropped ~46.8% of VACATION's population.
        # Offscreen sentinel rows carry gt_px_x=gt_px_y=-1 in every file
        # (confirmed project-wide) -- these compare equal (0 distance)
        # here as expected, no special-casing needed.
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
            "show": fu["show"], "clip": fu["clip"], "gaze_type": fu["gaze_type"],
            "is_offscreen": fu["gaze_type"] == "offscreen",
            "img_path": fu["img_path"],
            "head_x1": fu["head_x1"], "head_y1": fu["head_y1"],
            "head_x2": fu["head_x2"], "head_y2": fu["head_y2"],
            "gt_x": fu["gt_x"], "gt_y": fu["gt_y"],
            "hybrid_pred_x": h["pred_x"], "hybrid_pred_y": h["pred_y"],
            "hybrid_label_source": h.get("label_source"),
            "hybrid_raw_response": h.get("raw_response"),
            "internvl3_alone_pred_x": fu["pred_x"], "internvl3_alone_pred_y": fu["pred_y"],
            "mtgs_pred_x": m["pred_x"], "mtgs_pred_y": m["pred_y"],
        })

    print(f"    GT disagreements excluded (>{GT_MISMATCH_PIXEL_TOLERANCE}px): {gt_mismatches}")
    print(f"    Missing images excluded: {missing_images}")
    print(f"    Final [{dataset}:{split}] population: {len(joined)} records")
    return joined


def build_full_population():
    """Loads + joins every dataset in DATASETS, for both train and val
    splits, and concatenates across datasets -- this is what makes the
    joint run 'train on all three every epoch, validate on all three
    every epoch' rather than anything more elaborate."""
    train_records, val_records = [], []
    for dataset in DATASETS:
        train_records.extend(build_dataset_split_population(dataset, "TRAIN"))
        val_records.extend(build_dataset_split_population(dataset, "VAL"))

    random.Random(SEED).shuffle(train_records)

    print(f"\nJOINT population: train={len(train_records)}  val={len(val_records)}")
    by_dataset_train = Counter(r["dataset"] for r in train_records)
    by_dataset_val = Counter(r["dataset"] for r in val_records)
    print(f"  train by dataset: {dict(by_dataset_train)}")
    print(f"  val by dataset:   {dict(by_dataset_val)}")
    gaze_counts = Counter(r["gaze_type"] for r in train_records)
    print(f"  train gaze_type counts: {dict(gaze_counts)}")

    return train_records, val_records


# ══════════════════════════════════════════════════════════════════════
# ── DATASET ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
# GazeDataset now lives in gaze_student_model.py (shared, dependency-
# free) -- imported above. Kept out of this file for the same reason
# GazeStudent was: eval_static_v1.py (score an existing checkpoint
# against a dataset, no training) needs it without needing this file's
# own module-level config checks satisfied.


# ══════════════════════════════════════════════════════════════════════
# ── MODEL ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
# GazeStudent now lives in gaze_student_model.py (shared, dependency-
# free) -- imported above. Kept out of this file so temporal scripts can
# import the exact same class without needing this file's own module-
# level config checks satisfied. See that module's docstring.


# ══════════════════════════════════════════════════════════════════════
# ── METRICS ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def compute_regression_metrics(dists, errs_x, errs_y):
    """dists: list of Euclidean distances. errs_x/errs_y: SIGNED per-axis
    errors (pred - gt), for MAE/RMSE."""
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
    """y_true: 0/1 ground truth (1 = offscreen). y_prob: predicted
    probability (post-sigmoid) of offscreen. Pure-Python TP/FP/FN/TN +
    precision/recall/F1; ROC-AUC via sklearn if available, else null."""
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
    if HAVE_SKLEARN and len(set(y_true)) > 1:  # AUC undefined with only one class present
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc = None

    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


def evaluate(model, loader, device):
    """Returns (pooled, by_dataset, by_gaze_type, inout_pooled,
    inout_by_dataset, collapse_ratio). collapse_ratio is
    pred_std_x/gt_std_x over ON-SCREEN rows only (unchanged in spirit
    from the old scripts' collapse diagnostic -- offscreen coord
    predictions are meaningless and would distort the std ratio).
    """
    model.eval()
    dists, errs_x, errs_y = [], [], []
    by_dataset_dists = defaultdict(lambda: ([], [], []))
    by_gtype_dists = defaultdict(lambda: ([], [], []))
    inout_true, inout_prob = [], []
    by_dataset_inout_true, by_dataset_inout_prob = defaultdict(list), defaultdict(list)
    on_screen_pred_x, on_screen_gt_x = [], []

    with torch.inference_mode():
        for batch in loader:
            scene = batch["scene"].to(device)
            head = batch["head"].to(device)
            target = batch["target"]
            is_offscreen = batch["is_offscreen"]
            gaze_types = batch["gaze_type"]
            datasets_ = batch["dataset"]

            coord_pred, inout_logit = model(scene, head)
            coord_pred = coord_pred.cpu()
            inout_prob_batch = torch.sigmoid(inout_logit).cpu()

            on_screen_mask = is_offscreen < 0.5
            if on_screen_mask.any():
                d = torch.sqrt(((coord_pred - target) ** 2).sum(dim=1))
                ex = (coord_pred[:, 0] - target[:, 0])
                ey = (coord_pred[:, 1] - target[:, 1])
                for i in range(len(d)):
                    if not on_screen_mask[i]:
                        continue
                    dists.append(d[i].item()); errs_x.append(ex[i].item()); errs_y.append(ey[i].item())
                    ds_lists = by_dataset_dists[datasets_[i]]
                    ds_lists[0].append(d[i].item()); ds_lists[1].append(ex[i].item()); ds_lists[2].append(ey[i].item())
                    gt_lists = by_gtype_dists[gaze_types[i]]
                    gt_lists[0].append(d[i].item()); gt_lists[1].append(ex[i].item()); gt_lists[2].append(ey[i].item())
                    on_screen_pred_x.append(coord_pred[i, 0].item())
                    on_screen_gt_x.append(target[i, 0].item())

            for i in range(len(is_offscreen)):
                t = int(is_offscreen[i].item())
                p = float(inout_prob_batch[i].item())
                inout_true.append(t); inout_prob.append(p)
                by_dataset_inout_true[datasets_[i]].append(t)
                by_dataset_inout_prob[datasets_[i]].append(p)

    pooled = compute_regression_metrics(dists, errs_x, errs_y)
    by_dataset = {k: compute_regression_metrics(*v) for k, v in by_dataset_dists.items()}
    by_gaze_type = {k: compute_regression_metrics(*v) for k, v in by_gtype_dists.items()}

    inout_pooled = compute_inout_metrics(inout_true, inout_prob)
    inout_by_dataset = {k: compute_inout_metrics(by_dataset_inout_true[k], by_dataset_inout_prob[k])
                         for k in by_dataset_inout_true}

    pred_std_x = float(np.std(on_screen_pred_x)) if on_screen_pred_x else 0.0
    gt_std_x = float(np.std(on_screen_gt_x)) if on_screen_gt_x else 0.0
    collapse_ratio = pred_std_x / gt_std_x if gt_std_x > 1e-6 else float("inf")

    return pooled, by_dataset, by_gaze_type, inout_pooled, inout_by_dataset, collapse_ratio


# ══════════════════════════════════════════════════════════════════════
# ── TRAINING ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def worker_init_fn(worker_id):
    """PyTorch's DataLoader automatically seeds each worker's torch RNG
    from the main process's seed, but NOT stdlib random or numpy -- a
    real gap, since GazeDataset's flip-augmentation decision uses
    random.random() (stdlib, not torch), which runs inside the worker
    subprocess. Without this, that specific randomness wouldn't be
    reproducibly seeded across runs, on top of (not a substitute for) the
    sorted() fix above. Derives a distinct, deterministic seed per worker
    from torch's own per-worker base seed, matching the standard
    documented PyTorch pattern for this exact gap."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_condition(condition, train_records, val_records, ckpt_dir, device):
    print(f"\n{'='*70}\nTraining condition: {condition}\n{'='*70}")
    setup_full_determinism(SEED)

    if condition == "teacher_hybrid" and CONFIDENCE_THRESHOLD is not None:
        hybrid_recs = [r for r in train_records + val_records
                       if r.get("hybrid_label_source") == "teacher_hybrid"]
        n_below = sum(1 for r in hybrid_recs if not passes_confidence_threshold(r, CONFIDENCE_THRESHOLD))
        if hybrid_recs:
            print(f"    Confidence-thresholding ACTIVE (threshold={CONFIDENCE_THRESHOLD}): "
                  f"{n_below}/{len(hybrid_recs)} teacher_hybrid records "
                  f"({100*n_below/len(hybrid_recs):.1f}%) reclassified to GT")

    model = GazeStudent(VIT_MODEL).to(device)
    print(f"    Parameters: {count_params(model)/1e6:.1f}M")
    train_ds = GazeDataset(train_records, condition, augment=True)
    val_ds = GazeDataset(val_records, condition, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=(NUM_WORKERS > 0),
                               generator=seeded_generator(SEED),
                               worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=(NUM_WORKERS > 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    coord_loss_fn = nn.SmoothL1Loss(reduction="none")
    inout_loss_fn = nn.BCEWithLogitsLoss()

    best_ade, best_epoch, no_improve = float("inf"), -1, 0
    best_metrics = {}
    start_epoch = 0
    epoch_times = []
    # Full per-epoch record, EVERY epoch, not just the best one -- was
    # missing entirely before 2026-08-07 (only the single best epoch's
    # metrics were ever saved anywhere structured; everything else only
    # ever existed as printed text). Needed for genuine epoch-by-epoch
    # charts in the paper -- comparison.json is the only artifact that
    # survives after a run finishes, so it needs to actually contain this.
    epoch_history = []

    # PER-EPOCH RESUME -- saved every epoch regardless of improvement, to
    # LATEST_CKPT_PATH (not best_model.pt, which only updates on
    # improvement and could lag behind the true last-completed epoch if a
    # job is killed mid-plateau). Includes optimizer/scheduler state so
    # the cosine LR schedule continues exactly where it left off, not
    # restarted at epoch 0's high LR against already-partially-trained
    # weights. This is exactly the mechanism
    # distillation_joint_vat_childplay_gt.py needed for real (that run
    # genuinely timed out mid-training on a population less than half
    # this one's size) -- kept in the rewrite for the same reason, not
    # speculatively.
    LATEST_CKPT_PATH = os.path.join(ckpt_dir, "latest_checkpoint.pt")
    if RESUME:
        if os.path.exists(LATEST_CKPT_PATH):
            ckpt = torch.load(LATEST_CKPT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_ade = ckpt["best_ade"]
            best_epoch = ckpt["best_epoch"]
            no_improve = ckpt["no_improve"]
            best_metrics = ckpt.get("best_metrics", {})
            epoch_times = ckpt.get("epoch_times", [])
            epoch_history = ckpt.get("epoch_history", [])
            print(f"    RESUMED from {LATEST_CKPT_PATH}: completed through epoch "
                  f"{ckpt['epoch']+1}, continuing at epoch {start_epoch+1}. "
                  f"best_ade so far={best_ade:.4f} (epoch {best_epoch+1}), "
                  f"no_improve={no_improve}")
        else:
            print(f"    DISTILL_RESUME=1 but no checkpoint found at {LATEST_CKPT_PATH} "
                  f"-- starting fresh from epoch 1 instead.")

    condition_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            scene = batch["scene"].to(device)
            head = batch["head"].to(device)
            target = batch["target"].to(device)
            is_offscreen = batch["is_offscreen"].to(device)

            optimizer.zero_grad()
            coord_pred, inout_logit = model(scene, head)

            # Coordinate loss MASKED to on-screen rows only -- offscreen
            # gt_x/gt_y is a sentinel fraction, not a real target. Guard
            # against a batch with zero on-screen rows (unlikely at
            # batch_size=32 given no dataset is anywhere near 100%
            # offscreen, but defensive rather than assumed).
            on_screen_mask = (is_offscreen < 0.5)
            if on_screen_mask.any():
                per_sample_coord_loss = coord_loss_fn(coord_pred, target).sum(dim=1)
                coord_loss = per_sample_coord_loss[on_screen_mask].mean()
            else:
                coord_loss = torch.tensor(0.0, device=device)

            inout_loss = inout_loss_fn(inout_logit, is_offscreen)
            loss = coord_loss + INOUT_LOSS_WEIGHT * inout_loss

            if not torch.isfinite(loss):
                print(f"    WARNING: non-finite loss at epoch {epoch+1} -- skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else None

        pooled, by_dataset, by_gaze_type, inout_pooled, inout_by_dataset, collapse_ratio = \
            evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        val_ade = pooled["ade"] if pooled else float("inf")
        improved = val_ade < best_ade
        marker = " <- best" if improved else f" (no improve {no_improve+1}/{EARLY_STOP_PATIENCE})"
        by_ds_str = "  ".join(f"{k}={v['ade']:.4f}" for k, v in sorted(by_dataset.items()) if v)
        inout_str = (f"inout_f1={inout_pooled['f1']:.3f}" if inout_pooled else "inout=n/a")
        collapse_flag = " <-- LOW: possible collapse" if collapse_ratio < COLLAPSE_WARNING_RATIO else ""
        print(f"  Epoch {epoch+1:02d}  train_loss={avg_train_loss:.4f}  val_ADE={val_ade:.4f}  [{by_ds_str}]  "
              f"{inout_str}  pred/gt_std_ratio_x={collapse_ratio:.2f}{collapse_flag}  "
              f"time={epoch_time:.1f}s{marker}")

        epoch_history.append({
            "epoch": epoch, "train_loss": avg_train_loss, "val_ade": val_ade,
            "by_dataset": by_dataset, "by_gaze_type": by_gaze_type,
            "inout_pooled": inout_pooled, "inout_by_dataset": inout_by_dataset,
            "collapse_ratio_pred_gt_std_x": collapse_ratio,
            "wall_clock_seconds": epoch_time, "is_best": improved,
        })

        if improved:
            best_ade, best_epoch = val_ade, epoch
            best_metrics = {
                "pooled": pooled, "by_dataset": by_dataset, "by_gaze_type": by_gaze_type,
                "inout_pooled": inout_pooled, "inout_by_dataset": inout_by_dataset,
                "collapse_ratio_pred_gt_std_x": collapse_ratio,
            }
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save({
                "model_state": best_state, "val_ade": best_ade, "epoch": best_epoch,
                "condition": condition, "architecture": VIT_MODEL,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "datasets": DATASETS, "metrics": best_metrics,
                "epoch_history": epoch_history,
            }, os.path.join(ckpt_dir, "best_model.pt"))
        else:
            no_improve += 1

        # Saved EVERY epoch regardless of improvement -- see RESUME
        # block's comment above for why. Placed unconditionally, before
        # the early-stop check, so the final epoch's state is always
        # captured exactly once, not duplicated or skipped depending on
        # which branch triggered the break.
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch,
            "best_ade": best_ade, "best_epoch": best_epoch, "no_improve": no_improve,
            "best_metrics": best_metrics, "epoch_times": epoch_times,
            "epoch_history": epoch_history,
            "training_complete": False,
        }, LATEST_CKPT_PATH)

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Explicit completion marker, checked by main()'s resumability logic --
    # best_model.pt's mere EXISTENCE isn't a safe "this condition is done"
    # signal, since it gets written on the FIRST improving epoch, not on
    # completion. Without this, a job that improves once then dies before
    # finishing would be wrongly skipped as "already done" on resubmit,
    # silently keeping a severely undertrained checkpoint.
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "epoch": epoch,
        "best_ade": best_ade, "best_epoch": best_epoch, "no_improve": no_improve,
        "best_metrics": best_metrics, "epoch_times": epoch_times,
        "epoch_history": epoch_history,
        "training_complete": True,
    }, LATEST_CKPT_PATH)

    total_time = time.time() - condition_start
    return {
        "condition": condition, "best_epoch": best_epoch, "best_ade": best_ade,
        "metrics": best_metrics,
        "epoch_history": epoch_history,
        "n_train": len(train_records), "n_val": len(val_records),
        "wall_clock_seconds_total": total_time,
        "wall_clock_seconds_per_epoch": epoch_times,
        "wall_clock_seconds_mean_epoch": sum(epoch_times) / len(epoch_times) if epoch_times else None,
    }


def write_comparison(results, output_dir, run_metadata):
    """Structured JSON, built to be parsed programmatically -- not just a
    human-readable dump. Also writes a plain-text summary alongside for
    quick eyeballing, same as the old scripts."""
    output = {"run_metadata": run_metadata, "results": results}
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    lines = [f"DISTILLATION RUN — {run_metadata['datasets']} — "
             f"confidence_threshold={run_metadata['confidence_threshold']}",
             "=" * 70, ""]
    for name, r in results.items():
        pooled = r.get("metrics", {}).get("pooled") or {}
        lines.append(f"  {name:<24} best_epoch={r['best_epoch']:<4} "
                      f"ADE={r['best_ade']:.4f}  n_train={r['n_train']} n_val={r['n_val']}  "
                      f"wall_clock={r.get('wall_clock_seconds_total', 0):.0f}s")
        inout = r.get("metrics", {}).get("inout_pooled") or {}
        if inout:
            lines.append(f"      in/out: precision={inout['precision']:.3f} "
                          f"recall={inout['recall']:.3f} f1={inout['f1']:.3f} "
                          f"roc_auc={inout['roc_auc']}")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(output_dir, "experiment_summary.txt"), "w") as f:
        f.write(summary)


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {DATASETS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Architecture: {VIT_MODEL}")
    print(f"CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"INOUT_LOSS_WEIGHT: {INOUT_LOSS_WEIGHT}")
    print(f"EPOCHS: {EPOCHS}  BATCH_SIZE: {BATCH_SIZE}  LR: {LR}")
    print(f"NUM_WORKERS: {NUM_WORKERS}  SEED: {SEED}  RESUME: {RESUME}\n")

    train_records, val_records = build_full_population()
    if len(train_records) < 10:
        print("ERROR: too few training records -- check paths/schema before rerunning.")
        return

    # Calibration-run support -- caps population size for a fast smoke
    # test (confirm the pipeline runs end-to-end cleanly) without needing
    # a separate code path. Unset (default) uses the full population.
    max_train = os.environ.get("DISTILL_MAX_TRAIN_SAMPLES")
    max_val = os.environ.get("DISTILL_MAX_VAL_SAMPLES")
    if max_train:
        random.Random(SEED).shuffle(train_records)
        train_records = train_records[: int(max_train)]
        print(f"DISTILL_MAX_TRAIN_SAMPLES set -- truncated train to {len(train_records)}")
    if max_val:
        val_records = val_records[: int(max_val)]
        print(f"DISTILL_MAX_VAL_SAMPLES set -- truncated val to {len(val_records)}")

    run_metadata = {
        "datasets": DATASETS, "conditions": CONDITIONS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "architecture": VIT_MODEL, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "seed": SEED, "n_train": len(train_records), "n_val": len(val_records),
    }

    results = {}
    for condition in CONDITIONS:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"model_{condition}")
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pt")

        # Completion check via latest_checkpoint.pt's explicit
        # training_complete marker -- NOT best_model.pt's mere existence,
        # which gets written on the first improving epoch and would
        # otherwise wrongly look "done" if a job dies before finishing.
        already_complete = False
        if os.path.exists(latest_path):
            latest_ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
            already_complete = latest_ckpt.get("training_complete", False)
        elif os.path.exists(ckpt_path) and not os.path.exists(latest_path):
            # best_model.pt exists but latest_checkpoint.pt doesn't -- an
            # old-style checkpoint from before this completion-marker
            # existed. Treat as complete rather than silently retraining
            # something that already finished under the old scheme.
            already_complete = True

        if already_complete:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"\n[{condition}] already complete "
                  f"(val_ade={ckpt.get('val_ade')}, epoch={ckpt.get('epoch')}) "
                  f"-- skipping, not retraining. Delete this condition's "
                  f"checkpoint files first for a genuine retrain.")
            results[condition] = {
                "condition": condition, "best_epoch": ckpt.get("epoch"),
                "best_ade": ckpt.get("val_ade"), "metrics": ckpt.get("metrics", {}),
                "epoch_history": ckpt.get("epoch_history", []),
                "n_train": len(train_records), "n_val": len(val_records), "resumed": True,
            }
            write_comparison(results, OUTPUT_DIR, run_metadata)
            continue

        result = train_one_condition(condition, train_records, val_records, ckpt_dir, device)
        results[condition] = result
        write_comparison(results, OUTPUT_DIR, run_metadata)

    write_comparison(results, OUTPUT_DIR, run_metadata)
    print(f"\nAll outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
