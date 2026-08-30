"""
compute_static_same_frame_baseline.py — the one diagnostic number Opus's
dissertation review needs to settle §5.2.1: for the temporal-matched
population, the static student's own predicted (x,y) at the LAST CONTEXT
FRAME, compared against the TRUE gaze at that SAME frame (not the future
target static-copy-forward compares against).

WHY THIS IS A SEPARATE SCRIPT, NOT AN ADDITION TO train_temporal_v1.py's
own main(): that script always trains all four temporal architectures
from scratch every time it runs -- hours on GPU, and completely
unrelated to this specific number. Both ingredients this diagnostic
needs already exist inside that script (the predicted_xy is already
used for static-copy-forward; get_last_known_gaze() is already called
for the matched mask) -- so no static-model inference is needed, but a
full temporal training run would be needed if this were bolted onto
main() as written. This script reuses the SAME population-building and
baseline-lookup code (copied verbatim from train_temporal_v1.py, not
reimplemented) but stops before any model gets constructed or trained.

WHAT THIS SETTLES: teacher_internvl3_alone's static-copy-forward
baseline (0.2849) is far worse than its own pooled static ADE (0.1813,
Table 4.1) -- two possible explanations, and this number decides which:
  - near 0.285: the temporal-matched population is just a harder subset
    of VAT than average -- the static model's own same-frame accuracy on
    exactly these clips is already poor, explaining the bad baseline
    without any real temporal effect. Section 5.2.1's existing framing
    (a real, held-out generalisation finding) stands.
  - near 0.21 (matching iv3's known held-out VAT accuracy elsewhere in
    this dissertation): the static model is fine on these exact clips
    judged same-frame, and something genuinely breaks specifically when
    carried across a frame boundary -- a real temporal effect survives,
    and Section 5.2.1 needs re-examining.

USAGE: same env vars as train_temporal_v1.py's own population-building
step (TEMPORAL_DATASETS, STATIC_CONDITION, FEATURE_CACHE_ROOT,
TARGET_HORIZON_MS, {DATASET}_TEMPORAL_MANIFEST_PATH per dataset, plus
VAT_IMAGES_BASE_PATH / VAT_ROOT / CHILDPLAY_ROOT /
VACATION_LOCAL_ANNOTATIONS_ROOT as needed by the baseline modules) --
no STATIC_CHECKPOINT_ROOT needed, since no static inference happens
here either; only the already-cached predicted_xy is read.
"""

import os
import json
import importlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION — identical to train_temporal_v1.py's own ────────────
# ══════════════════════════════════════════════════════════════════════

_datasets_env = os.environ.get("TEMPORAL_DATASETS")
if not _datasets_env:
    raise ValueError("TEMPORAL_DATASETS not set -- e.g. 'vat,childplay,vacation'.")
DATASETS = [d.strip() for d in _datasets_env.split(",")]

STATIC_CONDITION = os.environ.get("STATIC_CONDITION")
if not STATIC_CONDITION:
    raise ValueError("STATIC_CONDITION not set.")

FEATURE_CACHE_ROOT = os.environ.get("FEATURE_CACHE_ROOT")
if not FEATURE_CACHE_ROOT:
    raise ValueError("FEATURE_CACHE_ROOT not set.")

TARGET_HORIZON_MS = int(os.environ.get("TARGET_HORIZON_MS", "300"))
VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}

OUTPUT_DIR = os.environ.get("TEMPORAL_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("TEMPORAL_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_CONFIG = {}
for _ds in DATASETS:
    manifest_env = f"{_ds.upper()}_TEMPORAL_MANIFEST_PATH"
    manifest_path = os.environ.get(manifest_env)
    if not manifest_path:
        raise ValueError(f"{manifest_env} not set.")
    DATASET_CONFIG[_ds] = {"manifest_path": manifest_path}

VAT_IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", "")
PATH_REMAP = {
    "childplay": (
        os.environ.get("CHILDPLAY_LOCAL_PATH_PREFIX", r"C:\repo\ChildPlay-gaze\images"),
        os.environ.get("CHILDPLAY_HPC_PATH_PREFIX",
                        "/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze/images"),
    ),
    "vacation": (
        os.environ.get("VACATION_LOCAL_PATH_PREFIX", r"C:\repo\VACATION\images"),
        os.environ.get("VACATION_HPC_PATH_PREFIX",
                        "/parallel_scratch/sl02092/standard_project/data/vacation/images"),
    ),
    "ucolaeo": (
        os.environ.get("UCOLAEO_LOCAL_PATH_PREFIX", r"C:\repo\UCO-LAEO\ucolaeodb\frames"),
        os.environ.get("UCOLAEO_HPC_PATH_PREFIX",
                        "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames"),
    ),
}
VAT_WINDOW_DATASET_MODULE = os.environ.get("VAT_WINDOW_DATASET_MODULE", "temporal_window_dataset_vat")
VAT_BASELINE_MODULE = os.environ.get("VAT_BASELINE_MODULE", "compute_last_known_position_baseline_vat")


# ══════════════════════════════════════════════════════════════════════
# ── PER-DATASET ADAPTERS — copied verbatim from train_temporal_v1.py ───
# ══════════════════════════════════════════════════════════════════════

def build_cache_key(dataset, clip_id, subject, fname):
    if dataset == "vat":
        show, clip = clip_id.split("/", 1)
        return f"{show}/{clip}/{subject}/{fname}"
    return f"{clip_id}/{subject}/{fname}"


def resolve_last_context_image_path(dataset, clip_id, raw_path):
    if dataset == "vat":
        if not VAT_IMAGES_BASE_PATH:
            raise ValueError("VAT_IMAGES_BASE_PATH not set -- required when TEMPORAL_DATASETS includes vat.")
        show, clip = clip_id.split("/", 1)
        return os.path.join(VAT_IMAGES_BASE_PATH, show, clip, raw_path)

    local_prefix, hpc_prefix = PATH_REMAP.get(dataset, (None, None))
    if local_prefix and hpc_prefix:
        normalized = raw_path.replace("\\", "/")
        normalized_prefix = local_prefix.replace("\\", "/")
        if normalized.startswith(normalized_prefix):
            return hpc_prefix.rstrip("/") + "/" + normalized[len(normalized_prefix):].lstrip("/")
    return raw_path


_split_fn_cache = {}

def get_dataset_split(dataset, manifest_path):
    if dataset not in _split_fn_cache:
        module_name = VAT_WINDOW_DATASET_MODULE if dataset == "vat" else f"temporal_window_dataset_{dataset}"
        mod = importlib.import_module(module_name)
        _split_fn_cache[dataset] = mod.build_clip_split
    return _split_fn_cache[dataset](manifest_path)


_baseline_fn_cache = {}

def get_last_known_gaze(dataset, clip_id, subject, fname):
    if dataset not in _baseline_fn_cache:
        if dataset == "vat":
            mod = importlib.import_module(VAT_BASELINE_MODULE)
            _baseline_fn_cache[dataset] = ("vat", mod.get_subject_gaze, None)
        else:
            mod = importlib.import_module(f"compute_last_known_position_baseline_{dataset}")
            _baseline_fn_cache[dataset] = ("generic", mod.get_subject_gaze, mod.parse_frame_number)

    kind, get_gaze, parse_frame_number = _baseline_fn_cache[dataset]
    if kind == "vat":
        show, clip = clip_id.split("/", 1)
        return get_gaze(show, clip, subject, fname)
    abs_frame = parse_frame_number(fname)
    return get_gaze(clip_id, subject, abs_frame)


def horizon_filter_matches(dataset, row, target_horizon_ms):
    if dataset == "vat":
        return row.get("horizon_frames") == VAT_HORIZON_FRAMES_MAP[target_horizon_ms]
    return row.get("target_horizon_ms") == target_horizon_ms


# ══════════════════════════════════════════════════════════════════════
# ── JOINT DATASET — copied verbatim from train_temporal_v1.py ──────────
# ══════════════════════════════════════════════════════════════════════

class JointTemporalDataset(Dataset):
    def __init__(self, dataset_split_ids, condition, target_horizon_ms, split_name="(unfiltered)"):
        self.rows = []
        counts = defaultdict(int)

        for dataset, clip_ids in dataset_split_ids.items():
            cache_dir = os.path.join(FEATURE_CACHE_ROOT, dataset, condition)
            index_path = os.path.join(cache_dir, "frame_features_index.jsonl")
            embeddings = np.load(os.path.join(cache_dir, "frame_features.npy"))
            predicted_xy = np.load(os.path.join(cache_dir, "frame_predicted_xy.npy"))

            key_to_idx = {}
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    key_to_idx[r["key"]] = r["embedding_idx"]

            manifest_path = DATASET_CONFIG[dataset]["manifest_path"]
            with open(manifest_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    counts["total"] += 1

                    if not horizon_filter_matches(dataset, row, target_horizon_ms):
                        counts["wrong_horizon"] += 1
                        continue
                    if not row["window_valid"]:
                        counts["invalid_window"] += 1
                        continue
                    if row["target_gaze_x"] is None:
                        counts["offscreen_target"] += 1
                        continue
                    if row["clip_id"] not in clip_ids:
                        counts["not_in_split"] += 1
                        continue

                    clip_id = row["clip_id"]
                    subject = row["subject_id"]
                    raw_paths = row["context_frame_paths"]
                    fnames = [os.path.basename(p) for p in raw_paths]
                    keys = [build_cache_key(dataset, clip_id, subject, fn) for fn in fnames]
                    if not all(k in key_to_idx for k in keys):
                        counts["missing_embedding"] += 1
                        continue

                    embed_indices = [key_to_idx[k] for k in keys]
                    self.rows.append({
                        "dataset": dataset, "clip": clip_id, "subject": subject,
                        "last_context_fname": fnames[-1],
                        "last_context_raw_path": raw_paths[-1],
                        "embed_seq": np.concatenate(
                            [embeddings[embed_indices], predicted_xy[embed_indices]], axis=1
                        ),
                        "target_gaze_x": float(row["target_gaze_x"]),
                        "target_gaze_y": float(row["target_gaze_y"]),
                        "target_gaze_type": row["target_gaze_type"],
                    })
                    counts["kept"] += 1

        self._resolve_all_dimensions()

        by_dataset = defaultdict(int)
        for r in self.rows:
            by_dataset[r["dataset"]] += 1
        print(f"[JointTemporalDataset:{split_name}] {counts['kept']} usable rows "
              f"(target_horizon_ms={target_horizon_ms}). By dataset: {dict(by_dataset)}. "
              f"Dropped: {counts['wrong_horizon']} wrong-horizon, "
              f"{counts['invalid_window']} invalid-window, "
              f"{counts['offscreen_target']} offscreen-target, "
              f"{counts['not_in_split']} not-in-split, "
              f"{counts['missing_embedding']} missing-embedding.")

    def _resolve_all_dimensions(self):
        dim_cache = {}
        bad = set()
        needed = {(r["dataset"], r["clip"], r["last_context_raw_path"]) for r in self.rows}

        for dataset, clip, raw_path in needed:
            cache_key = (dataset, clip)
            if cache_key in dim_cache:
                continue
            try:
                path = resolve_last_context_image_path(dataset, clip, raw_path)
                with Image.open(path) as img:
                    dim_cache[cache_key] = img.size
            except Exception as e:
                print(f"[JointTemporalDataset] Could not read dimensions for "
                      f"{dataset}/{clip}: {e}")
                bad.add(cache_key)

        self._dim_cache = dim_cache
        before = len(self.rows)
        self.rows = [r for r in self.rows if (r["dataset"], r["clip"]) not in bad]
        if len(self.rows) < before:
            print(f"[JointTemporalDataset] Dropped {before - len(self.rows)} rows "
                  f"from {len(bad)} (dataset,clip) pair(s) with unreadable dimensions.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        w, h = self._dim_cache[(r["dataset"], r["clip"])]
        target = torch.tensor([r["target_gaze_x"] / w, r["target_gaze_y"] / h], dtype=torch.float32)
        return {
            "embeddings": torch.from_numpy(r["embed_seq"]).float(),
            "target": target,
            "dataset": r["dataset"], "clip": r["clip"], "subject": r["subject"],
            "target_gaze_type": r["target_gaze_type"],
        }


def build_joint_split(datasets, condition, target_horizon_ms):
    train_split_ids, val_split_ids = {}, {}
    for dataset in datasets:
        manifest_path = DATASET_CONFIG[dataset]["manifest_path"]
        train_ids, val_ids = get_dataset_split(dataset, manifest_path)
        train_split_ids[dataset] = train_ids
        val_split_ids[dataset] = val_ids
        print(f"  [{dataset}] split: {len(train_ids)} train clips, {len(val_ids)} val clips")

    # Only VAL is needed for this diagnostic -- train split is built and
    # discarded to keep get_dataset_split()'s own caching behaviour
    # identical to train_temporal_v1.py's, in case that matters for any
    # dataset's own split logic, but nothing downstream touches it.
    val_ds = JointTemporalDataset(val_split_ids, condition, target_horizon_ms, split_name="val")
    return val_ds


# ══════════════════════════════════════════════════════════════════════
# ── THE NEW DIAGNOSTIC ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def compute_static_same_frame_baseline(val_ds):
    """For every row where the last-known-gaze baseline is computable,
    TWO distances from the same population, same rows, same run:
      - static-same-frame: predicted (x,y) at the last context frame vs.
        TRUE gaze at that SAME frame.
      - static-copy-forward (recomputed here, not read from a separate
        run's temporal_comparison.json): predicted (x,y) at the last
        context frame vs. the FUTURE target -- item["target"], already
        loaded. Opus review: reporting both from one pass means the
        carry-forward penalty needs no cross-run population-matching
        assumption -- both numbers are guaranteed to be over the exact
        same rows, not merely expected to be."""
    dists, by_dataset, by_gtype = [], defaultdict(list), defaultdict(list)
    d_forward_list, by_dataset_fwd, by_gtype_fwd = [], defaultdict(list), defaultdict(list)
    n_skipped_no_gaze, n_total = 0, 0

    for i, r in enumerate(val_ds.rows):
        n_total += 1
        gaze_px = get_last_known_gaze(r["dataset"], r["clip"], r["subject"], r["last_context_fname"])
        if gaze_px is None:
            n_skipped_no_gaze += 1
            continue

        w, h = val_ds._dim_cache[(r["dataset"], r["clip"])]
        gaze_norm = torch.tensor([gaze_px[0] / w, gaze_px[1] / h], dtype=torch.float32)

        item = val_ds[i]
        pred_xy_last_frame = item["embeddings"][-1, -2:]  # same slice static-copy-forward already uses

        d = torch.sqrt(((pred_xy_last_frame - gaze_norm) ** 2).sum()).item()
        d_forward = torch.sqrt(((pred_xy_last_frame - item["target"]) ** 2).sum()).item()

        dists.append(d)
        by_dataset[r["dataset"]].append(d)
        by_gtype[r["target_gaze_type"]].append(d)
        d_forward_list.append(d_forward)
        by_dataset_fwd[r["dataset"]].append(d_forward)
        by_gtype_fwd[r["target_gaze_type"]].append(d_forward)

    print(f"\n[static-same-frame] {len(dists)}/{n_total} rows had a computable "
          f"last-known-gaze value ({n_skipped_no_gaze} skipped).")
    return dists, by_dataset, by_gtype, d_forward_list, by_dataset_fwd, by_gtype_fwd


# ══════════════════════════════════════════════════════════════════════
# ── MAIN — no model construction, no training, eval-only diagnostic ────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Datasets: {DATASETS}")
    print(f"Static condition: {STATIC_CONDITION}")
    print(f"Target horizon: {TARGET_HORIZON_MS}ms\n")

    val_ds = build_joint_split(DATASETS, STATIC_CONDITION, TARGET_HORIZON_MS)

    dists, by_dataset, by_gtype, d_forward, by_dataset_fwd, by_gtype_fwd = compute_static_same_frame_baseline(val_ds)
    if not dists:
        raise RuntimeError(
            "0 rows had a computable static-same-frame baseline -- almost "
            "certainly a raw-annotation-root env var isn't set correctly "
            "for this machine (same failure mode train_temporal_v1.py "
            "already guards against for the matched mask)."
        )

    pooled_ade = sum(dists) / len(dists)
    pooled_fwd_ade = sum(d_forward) / len(d_forward)
    penalty = pooled_fwd_ade - pooled_ade
    print(f"\n{'='*70}\nSTATIC-SAME-FRAME vs. STATIC-COPY-FORWARD (same rows, one run)\n{'='*70}")
    print(f"  static-same-frame ADE:    {pooled_ade:.4f}  (n={len(dists)})")
    print(f"  static-copy-forward ADE:  {pooled_fwd_ade:.4f}  (same rows)")
    print(f"  carry-forward penalty:    {penalty:+.4f}")
    print(f"\n  By dataset:")
    for ds in sorted(by_dataset):
        v, vf = by_dataset[ds], by_dataset_fwd[ds]
        print(f"    {ds:<12} same-frame={sum(v)/len(v):.4f}  copy-forward={sum(vf)/len(vf):.4f}  "
              f"penalty={sum(vf)/len(vf)-sum(v)/len(v):+.4f}  n={len(v)}")
    print(f"\n  By gaze_type:")
    for gt in sorted(by_gtype):
        v, vf = by_gtype[gt], by_gtype_fwd[gt]
        print(f"    {gt:<12} same-frame={sum(v)/len(v):.4f}  copy-forward={sum(vf)/len(vf):.4f}  "
              f"penalty={sum(vf)/len(vf)-sum(v)/len(v):+.4f}  n={len(v)}")

    output = {
        "run_metadata": {"datasets": DATASETS, "static_condition": STATIC_CONDITION,
                          "target_horizon_ms": TARGET_HORIZON_MS, "n": len(dists)},
        "pooled_ade": pooled_ade,
        "pooled_copy_forward_ade_same_rows": pooled_fwd_ade,
        "carry_forward_penalty": penalty,
        "by_dataset": {k: sum(v)/len(v) for k, v in by_dataset.items()},
        "by_dataset_copy_forward": {k: sum(v)/len(v) for k, v in by_dataset_fwd.items()},
        "by_gaze_type": {k: sum(v)/len(v) for k, v in by_gtype.items()},
        "by_gaze_type_copy_forward": {k: sum(v)/len(v) for k, v in by_gtype_fwd.items()},
    }
    out_path = os.path.join(OUTPUT_DIR, f"static_same_frame_baseline_{STATIC_CONDITION}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
