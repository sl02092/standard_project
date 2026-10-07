"""
train_temporal_v1.py — temporal training, v1.0. Joint multi-dataset
version of the proven single-dataset PoC pattern (train_temporal_poc.py
and its VACATION/ChildPlay/UCO-LAEO ports), mirroring how distillation_v1.py
generalized the static training scripts. Reads cached embeddings from
extract_temporal_features_v1.py, combines multiple datasets into one
training population, trains the four proven temporal head architectures,
and reports pooled + by-dataset + by-gaze-type results with paired
bootstrap significance testing.

MODEL LAYER: GRUHead, TransformerHead, ResidualGRUHead,
ResidualTransformerHead, make_regression_tail, make_residual_tail,
pick_nhead, train_one_model, evaluate -- copied VERBATIM from
train_temporal_poc.py (confirmed directly, 2026-08-05, not reconstructed
by analogy). Zero changes to the model architectures or training loop --
only the data layer (which datasets' embeddings feed the same training
run) and the evaluation layer (pooled + by-dataset breakdown, not just
pooled) are new.

DELIBERATELY NOT INCLUDED YET: the EmbeddingZeroedWrapper shortcut-
collapse ablation (VACATION's own extension) -- its exact implementation
hasn't been directly confirmed yet, unlike everything else here. Add it
once that code is in hand rather than guess at it now.

PER-DATASET DIFFERENCES, HANDLED EXPLICITLY:
1. HORIZON FIELD. VAT's manifest has ONLY horizon_frames (2/7/12, no
   per-clip fps). The other three have target_horizon_ms (100/300/500,
   the cross-clip-comparable bucket). TARGET_HORIZON_MS below is the one
   canonical parameter this script takes; VAT_HORIZON_FRAMES_MAP
   translates it to VAT's own field for that one dataset.
2. TRAIN/VAL SPLIT. Each dataset's own build_clip_split()/
   build_video_split() is dynamically imported and used as-is (VAT: a
   custom seeded shuffle; ChildPlay: official clips.csv; VACATION:
   official readme.txt; UCO-LAEO: H5-origin) -- NOT reimplemented here,
   avoiding any risk of silently diverging from each dataset's own
   already-correct, already-verified split logic.
3. BASELINE LOOKUP. VAT's get_subject_gaze(show, clip, subject, fname)
   takes a bare filename directly; the other three's
   get_subject_gaze(clip_id, subject, abs_frame) take an integer frame
   number via their own parse_frame_number(path). Both shapes handled
   explicitly in get_last_known_gaze() below, not assumed uniform.
4. CACHE KEY FORMAT. Must match extract_temporal_features_v1.py's own
   build_cache_key() EXACTLY (VAT: 4-part "show/clip/subject/fname";
   others: 3-part "clip/subject/fname") -- duplicated here rather than
   imported, since these are meant to be two independently-runnable
   scripts, but kept byte-identical in logic. If you ever change one,
   change both.
5. IMAGE PATH FOR DIMENSION RESOLUTION. VAT's manifest stores BARE
   filenames for context_frame_paths (needs VAT_IMAGES_BASE_PATH +
   show/clip reconstruction); the other three store already-full,
   directly-openable paths. The raw manifest value for the LAST context
   frame specifically is kept (not just its basename) so real image
   dimensions can be resolved correctly for both cases -- see
   _resolve_all_dimensions().

REQUIRES: every dataset listed in TEMPORAL_DATASETS must have its own
compute_last_known_position_baseline_{dataset}.py and
temporal_window_dataset_{dataset}.py (VAT specifically:
temporal_window_dataset.py / compute_last_known_position_baseline_003.py
-- VAT's baseline module has a non-dataset-suffixed name, a file-
versioning artifact rather than a deliberate convention; worth renaming
to compute_last_known_position_baseline_vat.py for consistency with the
other three, but not required for this script -- the exact module name
is a config value below, not a hardcoded assumption) importable from the
working directory.
"""

import os
import json
import random
import importlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from gaze_student_model import setup_full_determinism, seeded_generator

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

_datasets_env = os.environ.get("TEMPORAL_DATASETS")
if not _datasets_env:
    raise ValueError(
        "TEMPORAL_DATASETS not set -- e.g. 'vat,childplay,vacation' for the "
        "joint run, or 'vacation' alone for a single-dataset run."
    )
DATASETS = [d.strip() for d in _datasets_env.split(",")]

STATIC_CONDITION = os.environ.get("STATIC_CONDITION")
if not STATIC_CONDITION:
    raise ValueError("STATIC_CONDITION not set -- which cached embeddings to "
                      "train on (must match what extract_temporal_features_v1.py produced).")

FEATURE_CACHE_ROOT = os.environ.get("FEATURE_CACHE_ROOT")
if not FEATURE_CACHE_ROOT:
    raise ValueError("FEATURE_CACHE_ROOT not set.")

TARGET_HORIZON_MS = int(os.environ.get("TARGET_HORIZON_MS", "300"))
VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}  # VAT's own nominal-fps mapping

EPOCHS = int(os.environ.get("TEMPORAL_EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("TEMPORAL_BATCH_SIZE", "64"))
LR = float(os.environ.get("TEMPORAL_LR", "1e-5"))  # VAT's established value -- see
                                                      # distillation_v1.py's own comment
                                                      # history on why this isn't re-swept
                                                      # by default
SEED = 42
WINDOW_LENGTH = 8

OUTPUT_DIR = os.environ.get("TEMPORAL_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("TEMPORAL_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-dataset manifest paths -- env-var-per-dataset, same "no silent
# default on a path that matters" convention as distillation_v1.py.
DATASET_CONFIG = {}
for _ds in DATASETS:
    manifest_env = f"{_ds.upper()}_TEMPORAL_MANIFEST_PATH"
    manifest_path = os.environ.get(manifest_env)
    if not manifest_path:
        raise ValueError(f"{manifest_env} not set.")
    DATASET_CONFIG[_ds] = {"manifest_path": manifest_path}

VAT_IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", "")
# Rewrites a local-machine path prefix baked into a manifest at build time
# to the real HPC path -- MUST match extract_temporal_features_v1.py's own
# PATH_REMAP exactly (found necessary 2026-08-10; see that script's own
# comment for the full story). Confirmed values, not guessed:
# ChildPlay="C:\repo\ChildPlay-gaze\images", VACATION="C:\repo\VACATION\images".
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
# Both VAT support modules are now consistently named
# (temporal_window_dataset_vat.py / compute_last_known_position_baseline_vat.py)
# -- matching the other three datasets. Overridable in case your local
# copy still uses the older, inconsistent names
# (temporal_window_dataset.py / compute_last_known_position_baseline_003.py).
VAT_WINDOW_DATASET_MODULE = os.environ.get("VAT_WINDOW_DATASET_MODULE", "temporal_window_dataset_vat")
VAT_BASELINE_MODULE = os.environ.get("VAT_BASELINE_MODULE", "compute_last_known_position_baseline_vat")


# ══════════════════════════════════════════════════════════════════════
# ── PER-DATASET ADAPTERS ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_cache_key(dataset, clip_id, subject, fname):
    """MUST match extract_temporal_features_v1.py's own build_cache_key()
    exactly -- see module docstring point 4."""
    if dataset == "vat":
        show, clip = clip_id.split("/", 1)
        return f"{show}/{clip}/{subject}/{fname}"
    return f"{clip_id}/{subject}/{fname}"


def resolve_last_context_image_path(dataset, clip_id, raw_path):
    """VAT's raw_path is a bare filename (needs show/clip +
    VAT_IMAGES_BASE_PATH reconstruction). The other three datasets'
    raw_path is SUPPOSED to already be a full, directly-openable path --
    but see PATH_REMAP's own comment for why that assumption was wrong
    for ChildPlay/VACATION specifically."""
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

# Retrain-on-test-population mode: when set, IGNORES each dataset's own
# official train/val split entirely (ChildPlay/VACATION's build_clip_split()
# filters by the manifest's OWN "split" column, which is uniformly "test"
# for these manifests -- pointed there directly, that returns EMPTY
# train/val sets for those two datasets, not a real split). Instead builds
# one fresh, seeded 85/15 clip-level shuffle from the manifest's own
# distinct clip_id values -- same logic VAT's ORIGINAL build_clip_split()
# already used (sorted(set(...)) before shuffling, seed=42), generalized
# here since that function was already dataset-agnostic (just reads
# clip_id from any manifest, no VAT-specific logic at all).
RETRAIN_ON_TEST_POPULATION = os.environ.get("RETRAIN_ON_TEST_POPULATION", "0") == "1"


def _fresh_clip_shuffle_split(manifest_path, val_frac=0.15, seed=SEED):
    clip_ids = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            clip_ids.add(row["clip_id"])
    clip_ids = sorted(clip_ids)
    rng = random.Random(seed)
    rng.shuffle(clip_ids)
    n_val = max(1, int(round(len(clip_ids) * val_frac)))
    val_ids = set(clip_ids[:n_val])
    train_ids = set(clip_ids[n_val:])
    return train_ids, val_ids


def get_dataset_split(dataset, manifest_path):
    """Dynamically imports and calls each dataset's OWN build_clip_split
    -- not reimplemented here. See module docstring point 2.

    UNLESS RETRAIN_ON_TEST_POPULATION=1, in which case every dataset uses
    a fresh, seeded clip-level shuffle instead (see module-level comment
    above) -- this is the one change needed to train on the new,
    genuinely-held-out test manifests rather than the original train
    population, since the official-split-column approach silently breaks
    for ChildPlay/VACATION there."""
    if RETRAIN_ON_TEST_POPULATION:
        return _fresh_clip_shuffle_split(manifest_path)

    if dataset not in _split_fn_cache:
        module_name = VAT_WINDOW_DATASET_MODULE if dataset == "vat" else f"temporal_window_dataset_{dataset}"
        mod = importlib.import_module(module_name)
        _split_fn_cache[dataset] = mod.build_clip_split
    return _split_fn_cache[dataset](manifest_path)


_baseline_fn_cache = {}
_vat_ann_cache = {}


def _load_vat_annotations(ann_path):
    """Copied from compute_last_known_position_baseline_vat.py's own
    load_annotations(), verbatim -- same parsing, not reimplemented."""
    import csv
    frames = []
    with open(ann_path, "r") as f:
        reader = csv.reader(f)
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
    """VAT_ROOT/annotations/{test,train}/{show}/{clip}/{subject}.txt --
    tries BOTH split directories, not just train. FIX (same one already
    verified in evaluate_temporal_on_static_test.py): the original
    compute_last_known_position_baseline_vat.py hardcodes
    TRAIN_ANN_DIR=annotations/train, which silently returns None for
    every row when a VAT clip is genuinely test-split -- exactly the
    case here, since RETRAIN_ON_TEST_POPULATION runs use test-split VAT
    clips throughout. Without this fix, VAT would silently contribute
    zero rows to the baseline-matched significance comparison, same
    failure this exact bug caused in the evaluation script before it was
    caught and fixed there."""
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


def get_last_known_gaze(dataset, clip_id, subject, fname):
    """VAT: get_subject_gaze(show, clip, subject, fname) -- takes the
    filename directly. Others: get_subject_gaze(clip_id, subject,
    abs_frame) -- takes an integer frame number via that dataset's own
    parse_frame_number(). See module docstring point 3. Returns
    (gx, gy) pixel-space or None."""
    if dataset == "vat":
        return _get_vat_last_known_gaze(clip_id, subject, fname)

    if dataset not in _baseline_fn_cache:
        mod = importlib.import_module(f"compute_last_known_position_baseline_{dataset}")
        _baseline_fn_cache[dataset] = ("generic", mod.get_subject_gaze, mod.parse_frame_number)

    kind, get_gaze, parse_frame_number = _baseline_fn_cache[dataset]
    if kind == "vat":
        show, clip = clip_id.split("/", 1)
        return get_gaze(show, clip, subject, fname)
    abs_frame = parse_frame_number(fname)
    return get_gaze(clip_id, subject, abs_frame)


def horizon_filter_matches(dataset, row, target_horizon_ms):
    """VAT filters by horizon_frames (a fixed global mapping); the other
    three filter by target_horizon_ms directly. See module docstring
    point 1."""
    if dataset == "vat":
        return row.get("horizon_frames") == VAT_HORIZON_FRAMES_MAP[target_horizon_ms]
    return row.get("target_horizon_ms") == target_horizon_ms


# ══════════════════════════════════════════════════════════════════════
# ── JOINT DATASET ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class JointTemporalDataset(Dataset):
    """Generic across all four datasets -- reads each dataset's own
    manifest + cached embeddings (from extract_temporal_features_v1.py),
    filters to one horizon, tags every row with its source dataset for
    the by-dataset breakdown. NOT any single dataset's own
    TemporalWindowDataset class (those have incompatible internal field
    names) -- this is new, generic code sitting on top of all of them
    uniformly, reusing only their FILTERING LOGIC (confirmed identical
    across all four existing per-dataset versions), not their classes
    directly.
    """
    def __init__(self, dataset_split_ids, condition, target_horizon_ms, split_name="(unfiltered)"):
        """dataset_split_ids: {dataset_name: set_of_clip_ids} -- which
        clips from EACH dataset belong to this split (train or val)."""
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
                        "last_context_raw_path": raw_paths[-1],  # for dimension resolution -- see point 5
                        "embed_seq": np.concatenate(
                            [embeddings[embed_indices], predicted_xy[embed_indices]], axis=1
                        ),  # (8, 130)
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
        """Real per-clip image dimensions, needed to normalize
        target_gaze_x/y (raw pixels in every dataset's manifest). Uses
        each row's last_context_raw_path via resolve_last_context_image_path()
        -- correct for both VAT (reconstruction needed) and the other
        three (already a full path) -- rather than the basename-only
        value used for cache keys, which isn't independently resolvable
        to a real file for non-VAT datasets."""
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
    """Builds train/val JointTemporalDataset instances by first getting
    EACH dataset's own official/established clip split (not a fresh
    shuffle), then loading only those clips' cached embeddings."""
    train_split_ids, val_split_ids = {}, {}
    for dataset in datasets:
        manifest_path = DATASET_CONFIG[dataset]["manifest_path"]
        train_ids, val_ids = get_dataset_split(dataset, manifest_path)
        train_split_ids[dataset] = train_ids
        val_split_ids[dataset] = val_ids
        print(f"  [{dataset}] split: {len(train_ids)} train clips, {len(val_ids)} val clips")

    train_ds = JointTemporalDataset(train_split_ids, condition, target_horizon_ms, split_name="train")
    val_ds = JointTemporalDataset(val_split_ids, condition, target_horizon_ms, split_name="val")
    return train_ds, val_ds


# ══════════════════════════════════════════════════════════════════════
# ── MODELS — copied verbatim from train_temporal_poc.py, confirmed ─────
# ── directly 2026-08-05, not reconstructed by analogy ──────────────────
# ══════════════════════════════════════════════════════════════════════

def make_regression_tail(input_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2), nn.Sigmoid(),
    )


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
                 dropout=0.1, window_length=WINDOW_LENGTH):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_nhead(input_dim, max_heads=8):
    for h in range(min(max_heads, input_dim), 0, -1):
        if input_dim % h == 0:
            return h
    return 1


def make_residual_tail(input_dim, dropout=0.1):
    tail = nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2),
    )
    nn.init.zeros_(tail[-1].weight)
    nn.init.zeros_(tail[-1].bias)
    return tail


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
                 dropout=0.1, window_length=WINDOW_LENGTH):
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


def seed_everything(seed=SEED):
    """Kept as a thin wrapper for call-site compatibility, but now
    delegates to gaze_student_model's setup_full_determinism() rather
    than duplicating a weaker version here. FOUND AND FIXED 2026-08-07:
    the original version of this function only seeded random/numpy/torch
    -- never torch.use_deterministic_algorithms / cudnn.deterministic.
    That gap has direct, already-proven precedent in this exact project
    (the UCO-LAEO inference-noise investigation confirmed real,
    measurable GPU/cuDNN-level non-determinism at this scale, not
    theoretical) -- GRU/Transformer layers have cuDNN-accelerated
    implementations that can exhibit exactly this. Not found as an actual
    problem here yet (unlike distillation_v1.py's confirmed
    set-iteration-order bug), but fixed proactively rather than waited on
    given the precedent."""
    setup_full_determinism(seed)


def evaluate(model, loader):
    """Returns a list of per-row Euclidean distances, in dataset order."""
    model.eval()
    dists = []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            pred = model(emb)
            d = torch.sqrt(((pred - target) ** 2).sum(dim=1))
            dists.extend(d.cpu().tolist())
    return dists


def train_one_model(name, model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    print(f"\n[{name}] {count_params(model):,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val_ade, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            emb = batch["embeddings"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            pred = model(emb)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_dists = evaluate(model, val_loader)
        val_ade = sum(val_dists) / len(val_dists)
        if val_ade < best_val_ade:
            best_val_ade, best_epoch = val_ade, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{name}] epoch {epoch+1:>3}/{epochs}  "
                  f"train_loss={sum(train_losses)/len(train_losses):.4f}  val_ADE={val_ade:.4f}")

    model.load_state_dict(best_state)
    print(f"[{name}] Best val_ADE={best_val_ade:.4f} at epoch {best_epoch+1}/{epochs}")
    return model, best_val_ade, best_epoch


# ══════════════════════════════════════════════════════════════════════
# ── BASELINES + SIGNIFICANCE TESTING ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def compute_baseline_computable_mask(val_ds):
    """True where the oracle last-known-position baseline could also
    produce an answer, per-dataset-adapted get_last_known_gaze()."""
    mask = []
    for r in val_ds.rows:
        gaze = get_last_known_gaze(r["dataset"], r["clip"], r["subject"], r["last_context_fname"])
        mask.append(gaze is not None)
    return mask


def paired_bootstrap_ci(model_dists, baseline_dists, n_boot=10000, seed=SEED):
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


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Datasets: {DATASETS}")
    print(f"Static condition: {STATIC_CONDITION}")
    print(f"Target horizon: {TARGET_HORIZON_MS}ms")
    print(f"Device: {DEVICE}\n")

    train_ds, val_ds = build_joint_split(DATASETS, STATIC_CONDITION, TARGET_HORIZON_MS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               generator=seeded_generator(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)  # order preserved -- required for masks below

    input_dim = next(iter(train_loader))["embeddings"].shape[-1]
    print(f"\nInferred per-frame input_dim = {input_dim}")

    matched_mask = compute_baseline_computable_mask(val_ds)
    n_matched = sum(matched_mask)
    val_datasets = [r["dataset"] for r in val_ds.rows]
    val_gaze_types = [r["target_gaze_type"] for r in val_ds.rows]
    print(f"{n_matched}/{len(matched_mask)} val rows are baseline-computable.\n")

    # FAIL LOUD, FAIL HERE -- not several divisions later as a cryptic
    # ZeroDivisionError. Found 2026-08-10: n_matched=0 almost always means
    # a dataset's baseline module can't find its raw annotation files --
    # each of VAT/ChildPlay/VACATION's own baseline/utils modules default
    # their annotation-root env var to a LOCAL WINDOWS PATH
    # (VAT_ROOT, CHILDPLAY_ROOT, VACATION_LOCAL_ANNOTATIONS_ROOT
    # respectively) that doesn't exist on HPC -- get_subject_gaze()
    # silently returns None for every row rather than erroring, so this
    # is the first point that can catch it clearly.
    if n_matched == 0:
        raise RuntimeError(
            "0 val rows are baseline-computable -- the static-copy-forward "
            "baseline (and every downstream ADE using it) would be "
            "undefined. Almost certainly means a dataset's raw-annotation "
            "root env var isn't set correctly for THIS machine: check "
            "VAT_ROOT, CHILDPLAY_ROOT, and VACATION_LOCAL_ANNOTATIONS_ROOT "
            "all point at real, existing HPC paths -- their defaults are "
            "local Windows paths and will silently fail here otherwise."
        )

    # ── Baselines ────────────────────────────────────────────────────
    train_targets = torch.stack([train_ds[i]["target"] for i in range(len(train_ds))])
    global_mean_xy = train_targets.mean(dim=0)
    val_targets = torch.stack([val_ds[i]["target"] for i in range(len(val_ds))])
    global_mean_dists = torch.sqrt(((val_targets - global_mean_xy) ** 2).sum(dim=1)).tolist()

    copy_fwd_dists = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        last_pred_xy = item["embeddings"][-1, -2:]
        copy_fwd_dists.append(torch.sqrt(((last_pred_xy - item["target"]) ** 2).sum()).item())

    breakdown_by(global_mean_dists, val_datasets, matched_mask, "global-mean (trivial) -- by dataset")
    breakdown_by(copy_fwd_dists, val_datasets, matched_mask, "static-copy-forward (fair floor) -- by dataset")
    breakdown_by(copy_fwd_dists, val_gaze_types, matched_mask, "static-copy-forward (fair floor) -- by gaze_type")

    results = {
        "global-mean": {"full_ade": sum(global_mean_dists) / len(global_mean_dists),
                         "matched_ade": sum(d for d, m in zip(global_mean_dists, matched_mask) if m) / n_matched},
        "static-copy-forward": {"full_ade": sum(copy_fwd_dists) / len(copy_fwd_dists),
                                 "matched_ade": sum(d for d, m in zip(copy_fwd_dists, matched_mask) if m) / n_matched},
    }
    copy_fwd_matched_by_type = {
        g: [d for d, gg, m in zip(copy_fwd_dists, val_gaze_types, matched_mask) if gg == g and m]
        for g in set(val_gaze_types)
    }

    # ── Models ───────────────────────────────────────────────────────
    model_specs = [
        ("GRU", lambda: GRUHead(input_dim=input_dim)),
        ("Transformer", lambda: TransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))),
    ]
    if input_dim > 2:
        model_specs.append(("ResidualGRU", lambda: ResidualGRUHead(input_dim=input_dim)))
        model_specs.append(("ResidualTransformer", lambda: ResidualTransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))))

    all_dists = {}
    for name, ctor in model_specs:
        seed_everything(SEED)
        model = ctor()
        model, best_ade, best_epoch = train_one_model(name, model, train_loader, val_loader)
        dists = evaluate(model, val_loader)
        all_dists[name] = dists
        matched_ade = sum(d for d, m in zip(dists, matched_mask) if m) / n_matched
        results[name] = {"full_ade": sum(dists) / len(dists), "matched_ade": matched_ade,
                          "best_epoch": best_epoch + 1, "n_params": count_params(model)}
        breakdown_by(dists, val_datasets, matched_mask, f"{name} -- by dataset")
        breakdown_by(dists, val_gaze_types, matched_mask, f"{name} -- by gaze_type")

        # FOUND MISSING 2026-08-10: nothing previously saved the trained
        # temporal head to disk -- only metrics ever survived past the
        # process exiting. Needed for any later eval-only use (e.g.
        # scoring against UCO-LAEO with no further training) and for
        # genuine reproducibility of these specific weights.
        model_ckpt_path = os.path.join(OUTPUT_DIR, f"model_{name}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "architecture": name, "input_dim": input_dim,
            "best_ade": best_ade, "best_epoch": best_epoch + 1,
            "static_condition": STATIC_CONDITION, "datasets": DATASETS,
            "target_horizon_ms": TARGET_HORIZON_MS,
        }, model_ckpt_path)
        print(f"  Saved: {model_ckpt_path}")

        print(f"  Paired bootstrap 95% CI vs static-copy-forward, per gaze_type:")
        sig_by_type = {}
        for g in sorted(copy_fwd_matched_by_type):
            model_matched = [d for d, gg, m in zip(dists, val_gaze_types, matched_mask) if gg == g and m]
            mean_diff, lo, hi = paired_bootstrap_ci(model_matched, copy_fwd_matched_by_type[g])
            v = verdict(lo, hi)
            sig_by_type[g] = {"mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi, "verdict": v}
            print(f"    {g:<10} mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {v}")
        results[name]["significance_by_gaze_type"] = sig_by_type

    # ── Output ───────────────────────────────────────────────────────
    output = {
        "run_metadata": {"datasets": DATASETS, "static_condition": STATIC_CONDITION,
                          "target_horizon_ms": TARGET_HORIZON_MS, "epochs": EPOCHS,
                          "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
                          "n_train": len(train_ds), "n_val": len(val_ds), "n_matched": n_matched},
        "results": results,
    }
    with open(os.path.join(OUTPUT_DIR, "temporal_comparison.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    for name, r in results.items():
        print(f"  {name:<22} matched_ADE={r['matched_ade']:.4f}")
    print(f"\nFull results: {os.path.join(OUTPUT_DIR, 'temporal_comparison.json')}")


if __name__ == "__main__":
    main()
