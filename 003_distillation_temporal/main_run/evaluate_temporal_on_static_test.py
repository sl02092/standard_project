"""
evaluate_temporal_on_static_test.py — scores the ALREADY-TRAINED temporal
head checkpoints (model_{architecture}.pt, produced by train_temporal_v1.py
and confirmed to exist on disk since its 2026-08-10 fix) against the NEW,
genuinely held-out test manifests built directly from each dataset's own
static frame_manifest (build_temporal_from_static_v1.py's output).

NO RETRAINING. This loads real, already-trained weights and runs a single
forward pass over new data -- the whole point of this script.

WHY THIS ISN'T JUST "IMPORT train_temporal_v1.py AND CALL A FUNCTION":
that file requires several env vars (TEMPORAL_DATASETS, STATIC_CONDITION,
FEATURE_CACHE_ROOT, TEMPORAL_OUTPUT_DIR) at MODULE level, not inside a
function -- importing it would force satisfying all of its own training-
time config just to reuse a class. Same reason gaze_student_model.py was
split out of distillation_v1.py originally (see that file's own
docstring). JointTemporalDataset, build_cache_key(),
resolve_last_context_image_path(), get_last_known_gaze(), and the four
model architecture classes are copied here, not imported -- kept
logically identical to train_temporal_v1.py's own versions on purpose;
if you change the model architectures there, change them here too.

WHAT'S GENUINELY DIFFERENT FROM train_temporal_v1.py's OWN val-set logic:
  - No split_clip_ids filtering needed -- each *_test_from_static_
    temporal_manifest.jsonl file already contains ONLY test-split data
    (confirmed directly, dataset by dataset, before these files existed
    at all), so every row in the manifest is used.
  - Feature cache path points at temporal_feature_cache_from_static_*,
    not the old temporal_feature_cache_* the original training runs used
    -- this is the NEW cache extract_temporal_features_v1.py just built
    against the from-static manifests.
  - No training loop, no train/val split, no bootstrap significance
    testing against a matched subset that needs recomputing per-run --
    just: load weights, forward pass, report ADE.

USAGE (env vars, matching train_temporal_v1.py's own naming convention
wherever there's a direct equivalent):
    TEMPORAL_DATASETS=vat,childplay,vacation
    STATIC_CONDITION=gt
    TARGET_HORIZON_MS=300
    FEATURE_CACHE_ROOT=/path/to/temporal_feature_cache_from_static_vittiny
    CHECKPOINT_DIR=/path/to/temporal_output_vittiny/gt/horizon_300ms
      (wherever train_temporal_v1.py's OUTPUT_DIR for this condition/
      horizon actually wrote model_{name}.pt -- point this at the run
      whose checkpoints you want to evaluate)
    VAT_TEST_MANIFEST_PATH=/path/to/vat_test_from_static_temporal_manifest.jsonl
    CHILDPLAY_TEST_MANIFEST_PATH=/path/to/childplay_test_from_static_temporal_manifest.jsonl
    VACATION_TEST_MANIFEST_PATH=/path/to/vacation_test_from_static_temporal_manifest.jsonl
    VAT_IMAGES_BASE_PATH=/path/to/videoattentiontarget/images   (VAT only)
    OUTPUT_PATH=/path/to/temporal_test_evaluation.json

    python evaluate_temporal_on_static_test.py
"""

import os
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
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
    raise ValueError("FEATURE_CACHE_ROOT not set -- point this at the NEW "
                      "temporal_feature_cache_from_static_* directory, not "
                      "the old one.")

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR")
if not CHECKPOINT_DIR:
    raise ValueError("CHECKPOINT_DIR not set -- wherever train_temporal_v1.py's "
                      "OUTPUT_DIR for this condition/horizon wrote model_{name}.pt.")

TARGET_HORIZON_MS = int(os.environ.get("TARGET_HORIZON_MS", "300"))
VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}

OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "temporal_test_evaluation.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-dataset TEST manifest paths -- deliberately a DIFFERENT env var
# name pattern (*_TEST_MANIFEST_PATH) from train_temporal_v1.py's own
# *_TEMPORAL_MANIFEST_PATH, so the two scripts can never be pointed at
# the same env vars by accident and silently mix up train/val/test data.
TEST_MANIFEST_PATHS = {}
for _ds in DATASETS:
    env_name = f"{_ds.upper()}_TEST_MANIFEST_PATH"
    path = os.environ.get(env_name)
    if not path:
        raise ValueError(f"{env_name} not set.")
    TEST_MANIFEST_PATHS[_ds] = path

VAT_IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", "")
# Same PATH_REMAP purpose as extract_temporal_features_v1.py and
# train_temporal_v1.py's own copies -- kept identical, not imported, same
# reasoning as everywhere else this pattern appears in this project.
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
}

VAT_BASELINE_MODULE = os.environ.get("VAT_BASELINE_MODULE", "compute_last_known_position_baseline_vat")


# ══════════════════════════════════════════════════════════════════════
# ── PER-DATASET ADAPTERS — copied from train_temporal_v1.py, kept
# ── logically identical on purpose (see module docstring) ──────────────
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


_baseline_fn_cache = {}

def get_last_known_gaze(dataset, clip_id, subject, fname):
    """VAT: get_subject_gaze(show, clip, subject, fname) -- takes the
    filename directly. Others: get_subject_gaze(clip_id, subject,
    abs_frame) -- takes an integer frame number via that dataset's own
    parse_frame_number(). See module docstring point 3. Returns
    (gx, gy) pixel-space or None.

    VAT FIX (found via this script's own by-dataset breakdown showing
    n(matched)=0 for VAT specifically, everything else non-zero):
    compute_last_known_position_baseline_vat.py's own get_subject_gaze()
    hardcodes TRAIN_ANN_DIR = annotations/train -- correct for the
    ORIGINAL, train-only manifest that module was built for, but wrong
    here, where VAT rows can genuinely be test-split, living under
    annotations/test instead. That mismatch silently returned None for
    every VAT row rather than erroring, exactly the kind of failure mode
    this project has hit before (raw-annotation-root env vars pointing
    at the wrong place). Fixed here by trying BOTH split directories
    directly, not by editing the shared baseline module (which the
    original train-only evaluation still depends on as-is)."""
    if dataset == "vat":
        return _get_vat_last_known_gaze(clip_id, subject, fname)

    if dataset not in _baseline_fn_cache:
        import importlib
        mod = importlib.import_module(f"compute_last_known_position_baseline_{dataset}")
        _baseline_fn_cache[dataset] = (mod.get_subject_gaze, mod.parse_frame_number)

    get_gaze, parse_frame_number = _baseline_fn_cache[dataset]
    abs_frame = parse_frame_number(fname)
    return get_gaze(clip_id, subject, abs_frame)


_vat_ann_cache = {}

def _load_vat_annotations(ann_path):
    """Copied from compute_last_known_position_baseline_vat.py's own
    load_annotations(), verbatim -- same parsing, not reimplemented."""
    frames = []
    with open(ann_path, "r") as f:
        import csv as _csv
        reader = _csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            fname = row[0].strip()
            try:
                gx, gy = int(row[5]), int(row[6])
            except ValueError:
                continue
            gaze = (gx, gy) if gx != -1 and gy != -1 else None
            frames.append({"fname": fname, "gaze": gaze})
    return frames


def _get_vat_last_known_gaze(clip_id, subject, fname):
    """VAT_ROOT/annotations/{train,test}/{show}/{clip}/{subject}.txt --
    tries test first (this script's whole purpose is test-split
    evaluation) then train, rather than assuming one fixed split like
    the original module does."""
    show, clip = clip_id.split("/", 1)
    key = (show, clip, subject)
    if key not in _vat_ann_cache:
        vat_root = os.environ.get("VAT_ROOT")
        if not vat_root:
            raise ValueError("VAT_ROOT not set -- required for VAT's static-copy-forward baseline lookup.")
        found = {}
        for split in ("test", "train"):
            ann_path = os.path.join(vat_root, "annotations", split, show, clip, f"{subject}.txt")
            if os.path.isfile(ann_path):
                frames = _load_vat_annotations(ann_path)
                found = {row["fname"]: row["gaze"] for row in frames}
                break
        _vat_ann_cache[key] = found
    return _vat_ann_cache[key].get(fname)


def horizon_filter_matches(dataset, row, target_horizon_ms):
    if dataset == "vat":
        return row.get("horizon_frames") == VAT_HORIZON_FRAMES_MAP[target_horizon_ms]
    return row.get("target_horizon_ms") == target_horizon_ms


# ══════════════════════════════════════════════════════════════════════
# ── TEST DATASET — same shape as train_temporal_v1.py's own
# ── JointTemporalDataset, but reads every row in the manifest (no
# ── split_clip_ids filtering -- the file itself is already test-only) ──
# ══════════════════════════════════════════════════════════════════════

class JointTemporalTestDataset(Dataset):
    def __init__(self, manifest_paths, condition, target_horizon_ms):
        self.rows = []
        counts = defaultdict(int)

        for dataset, manifest_path in manifest_paths.items():
            cache_dir = os.path.join(FEATURE_CACHE_ROOT, dataset, condition)
            index_path = os.path.join(cache_dir, "frame_features_index.jsonl")
            embeddings = np.load(os.path.join(cache_dir, "frame_features.npy"))
            predicted_xy = np.load(os.path.join(cache_dir, "frame_predicted_xy.npy"))

            key_to_idx = {}
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    key_to_idx[r["key"]] = r["embedding_idx"]

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
                    # NO not-in-split check -- every row in a
                    # *_test_from_static_temporal_manifest.jsonl file is
                    # already test-split by construction.

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
        print(f"[JointTemporalTestDataset] {counts['kept']} usable TEST rows "
              f"(target_horizon_ms={target_horizon_ms}). By dataset: {dict(by_dataset)}. "
              f"Dropped: {counts['wrong_horizon']} wrong-horizon, "
              f"{counts['invalid_window']} invalid-window, "
              f"{counts['offscreen_target']} offscreen-target, "
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
                print(f"[JointTemporalTestDataset] Could not read dimensions for "
                      f"{dataset}/{clip}: {e}")
                bad.add(cache_key)

        self._dim_cache = dim_cache
        before = len(self.rows)
        self.rows = [r for r in self.rows if (r["dataset"], r["clip"]) not in bad]
        if len(self.rows) < before:
            print(f"[JointTemporalTestDataset] Dropped {before - len(self.rows)} rows "
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


# ══════════════════════════════════════════════════════════════════════
# ── MODELS — copied verbatim from train_temporal_v1.py, kept logically
# ── identical on purpose (see module docstring) ─────────────────────────
# ══════════════════════════════════════════════════════════════════════

def make_regression_tail(input_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2), nn.Sigmoid(),
    )


def make_residual_tail(input_dim, dropout=0.1):
    tail = nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2),
    )
    nn.init.zeros_(tail[-1].weight)
    nn.init.zeros_(tail[-1].bias)
    return tail


def pick_nhead(input_dim, max_heads=8):
    for h in range(min(max_heads, input_dim), 0, -1):
        if input_dim % h == 0:
            return h
    return 1


class GRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.tail = make_regression_tail(hidden_dim, dropout)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.tail(h_n[-1])


class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256,
                 dropout=0.1, window_length=8):
        super().__init__()
        if input_dim % nhead != 0:
            raise ValueError(f"input_dim={input_dim} not divisible by nhead={nhead}.")
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_regression_tail(input_dim, dropout)

    def forward(self, x):
        x = x + self.pos_embed
        out = self.encoder(x)
        return self.tail(out[:, -1, :])


class ResidualGRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        assert input_dim > 2, "ResidualGRUHead needs predicted_xy in the input."
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.tail = make_residual_tail(hidden_dim, dropout)

    def forward(self, x):
        reference = x[:, -1, -2:]
        _, h_n = self.gru(x)
        delta = self.tail(h_n[-1])
        return torch.clamp(reference + delta, 0.0, 1.0)


class ResidualTransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256,
                 dropout=0.1, window_length=8):
        super().__init__()
        assert input_dim > 2, "ResidualTransformerHead needs predicted_xy in the input."
        if input_dim % nhead != 0:
            raise ValueError(f"input_dim={input_dim} not divisible by nhead={nhead}.")
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_residual_tail(input_dim, dropout)

    def forward(self, x):
        reference = x[:, -1, -2:]
        x2 = x + self.pos_embed
        out = self.encoder(x2)
        delta = self.tail(out[:, -1, :])
        return torch.clamp(reference + delta, 0.0, 1.0)


MODEL_CLASSES = {
    "GRU": GRUHead, "Transformer": TransformerHead,
    "ResidualGRU": ResidualGRUHead, "ResidualTransformer": ResidualTransformerHead,
}


def load_trained_model(name, checkpoint_path, input_dim):
    """Loads an ALREADY-TRAINED checkpoint -- no training happens here.
    Re-derives nhead the same way train_temporal_v1.py's own model_specs
    did at training time (pick_nhead(input_dim)) -- the checkpoint itself
    doesn't store nhead, so this must match exactly or state_dict loading
    will fail with a shape mismatch, which is itself a useful sanity
    check that input_dim matches what this checkpoint was trained with."""
    cls = MODEL_CLASSES[name]
    if name in ("Transformer", "ResidualTransformer"):
        model = cls(input_dim=input_dim, nhead=pick_nhead(input_dim))
    else:
        model = cls(input_dim=input_dim)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"[load_trained_model] Loaded {checkpoint_path}")
    print(f"  architecture={ckpt.get('architecture')}  best_ade(orig val)={ckpt.get('best_ade')}  "
          f"best_epoch={ckpt.get('best_epoch')}  static_condition={ckpt.get('static_condition')}  "
          f"target_horizon_ms={ckpt.get('target_horizon_ms')}")
    return model


@torch.no_grad()
def evaluate(model, loader):
    dists = []
    for batch in loader:
        emb = batch["embeddings"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        pred = model(emb)
        d = torch.sqrt(((pred - target) ** 2).sum(dim=1))
        dists.extend(d.cpu().tolist())
    return dists


def breakdown_by(dists, keys, matched_mask, label):
    by_key = defaultdict(list)
    by_key_matched = defaultdict(list)
    for d, k, m in zip(dists, keys, matched_mask):
        by_key[k].append(d)
        if m:
            by_key_matched[k].append(d)
    print(f"\n[{label}]")
    for k in sorted(by_key):
        full, matched = by_key[k], by_key_matched.get(k, [])
        m_str = f"{sum(matched)/len(matched):.4f}" if matched else "n/a"
        print(f"  {k:<14} n(full)={len(full):<7} ADE(full)={sum(full)/len(full):.4f}  "
              f"n(matched)={len(matched):<7} ADE(matched)={m_str}")


def compute_baseline_computable_mask(test_ds):
    mask = []
    for r in test_ds.rows:
        gaze = get_last_known_gaze(r["dataset"], r["clip"], r["subject"], r["last_context_fname"])
        mask.append(gaze is not None)
    return mask


# ══════════════════════════════════════════════════════════════════════
# ── SIGNIFICANCE TESTING — copied verbatim from train_temporal_v1.py,
# ── same 10,000-resample paired bootstrap, same 95% CI, same REAL+/
# ── REAL-/n.s. verdict convention used throughout this whole project ───
# ══════════════════════════════════════════════════════════════════════

def paired_bootstrap_ci(model_dists, baseline_dists, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(baseline_dists) - np.array(model_dists)  # positive = model better
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diffs.mean(), lo, hi


def verdict(lo, hi):
    if lo > 0:
        return "REAL+"
    if hi < 0:
        return "REAL-"
    return "n.s."


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Datasets: {DATASETS}")
    print(f"Static condition: {STATIC_CONDITION}")
    print(f"Target horizon: {TARGET_HORIZON_MS}ms")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Feature cache root (from-static): {FEATURE_CACHE_ROOT}")
    print(f"Device: {DEVICE}\n")

    test_ds = JointTemporalTestDataset(TEST_MANIFEST_PATHS, STATIC_CONDITION, TARGET_HORIZON_MS)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    if len(test_ds) == 0:
        raise RuntimeError("0 usable test rows -- check manifest paths and that the "
                            "feature cache was extracted for this condition/dataset combo.")

    input_dim = test_ds[0]["embeddings"].shape[-1]
    print(f"\nInferred per-frame input_dim = {input_dim}")

    matched_mask = compute_baseline_computable_mask(test_ds)
    n_matched = sum(matched_mask)
    test_datasets_list = [r["dataset"] for r in test_ds.rows]
    test_gaze_types = [r["target_gaze_type"] for r in test_ds.rows]
    print(f"{n_matched}/{len(matched_mask)} test rows are baseline-computable.\n")

    if n_matched == 0:
        raise RuntimeError(
            "0 test rows are baseline-computable -- the static-copy-forward "
            "baseline would be undefined. Check each dataset's raw-annotation "
            "root env var (VAT_ROOT, CHILDPLAY_ROOT, VACATION_LOCAL_ANNOTATIONS_ROOT) "
            "points at a real, existing path on this machine."
        )

    # ── Baseline: static-copy-forward, on TEST this time ────────────────
    copy_fwd_dists = []
    for i in range(len(test_ds)):
        item = test_ds[i]
        last_pred_xy = item["embeddings"][-1, -2:]
        copy_fwd_dists.append(torch.sqrt(((last_pred_xy - item["target"]) ** 2).sum()).item())
    breakdown_by(copy_fwd_dists, test_datasets_list, matched_mask, "static-copy-forward (fair floor) -- by dataset")
    breakdown_by(copy_fwd_dists, test_gaze_types, matched_mask, "static-copy-forward (fair floor) -- by gaze_type")

    results = {
        "static-copy-forward": {
            "full_ade": sum(copy_fwd_dists) / len(copy_fwd_dists),
            "matched_ade": sum(d for d, m in zip(copy_fwd_dists, matched_mask) if m) / n_matched,
        },
    }

    # ── Load and evaluate each already-trained architecture ─────────────
    for name in MODEL_CLASSES:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_{name}.pt")
        if not os.path.isfile(ckpt_path):
            print(f"=== SKIPPING {name} -- no checkpoint at {ckpt_path} ===")
            continue

        model = load_trained_model(name, ckpt_path, input_dim)
        dists = evaluate(model, test_loader)
        matched_ade = sum(d for d, m in zip(dists, matched_mask) if m) / n_matched
        results[name] = {"full_ade": sum(dists) / len(dists), "matched_ade": matched_ade}
        breakdown_by(dists, test_datasets_list, matched_mask, f"{name} -- by dataset (TEST)")
        breakdown_by(dists, test_gaze_types, matched_mask, f"{name} -- by gaze_type (TEST)")

        # ── Paired bootstrap significance, matched subset, pooled across
        # ── the whole test population (not split by gaze_type here --
        # ── this is a single pooled verdict per model, matching the
        # ── headline "does the model beat the baseline" question) ───────
        model_matched = [d for d, m in zip(dists, matched_mask) if m]
        baseline_matched = [d for d, m in zip(copy_fwd_dists, matched_mask) if m]
        mean_diff, lo, hi = paired_bootstrap_ci(model_matched, baseline_matched)
        v = verdict(lo, hi)
        results[name]["significance_vs_baseline"] = {
            "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi, "verdict": v,
        }
        print(f"  Paired bootstrap 95% CI vs static-copy-forward (matched, pooled): "
              f"mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {v}")

    output = {
        "run_metadata": {
            "datasets": DATASETS, "static_condition": STATIC_CONDITION,
            "target_horizon_ms": TARGET_HORIZON_MS, "n_test": len(test_ds), "n_matched": n_matched,
            "checkpoint_dir": CHECKPOINT_DIR, "feature_cache_root": FEATURE_CACHE_ROOT,
            "note": "Evaluation only -- no training happened in this run. Checkpoints "
                    "were trained by an earlier train_temporal_v1.py run on the old "
                    "population; this manifest is the new, genuinely held-out test "
                    "split built directly from the static frame manifest.",
        },
        "results": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 78 + "\nSUMMARY (TEST)\n" + "=" * 78)
    for name, r in results.items():
        verdict_str = f"  -> {r['significance_vs_baseline']['verdict']}" if "significance_vs_baseline" in r else ""
        print(f"  {name:<22} matched_ADE={r['matched_ade']:.4f}{verdict_str}")
    print(f"\nFull results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
