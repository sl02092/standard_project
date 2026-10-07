"""
extract_temporal_features_v1.py — temporal feature extraction, v1.0.
Generic across all four datasets (VAT/ChildPlay/VACATION/UCO-LAEO) and
all four static teacher conditions (gt/teacher_hybrid/
teacher_internvl3_alone/mtgs) -- run once per (dataset, condition) pair
you actually need cached embeddings for.

WHAT THIS DOES: loads a frozen GazeStudent checkpoint (from
distillation_v1.py's output), reads a dataset's already-built temporal
window manifest, and for every unique (clip, subject, frame) referenced
as CONTEXT anywhere in that manifest, runs one forward pass and caches
three things: the 128-dim penultimate embedding, the model's predicted
(x, y) coordinate, and — NEW, not in any prior version of this script —
the model's predicted in/out (offscreen) probability. All three come
from the SAME forward pass (see gaze_student_model.PenultimateFeatureExtractor),
so caching all three costs nothing beyond caching one.

WHY IN/OUT PROBABILITY IS WORTH CACHING NOW: the old temporal scripts
only ever had predicted_xy to feed the residual heads (ResidualGRUHead
etc.) as an explicit position signal. The new GazeStudent also predicts
whether the target is offscreen at all -- a genuinely new, cheap-to-add
signal a temporal head could use (e.g. "the static model was very
uncertain this frame" is itself informative for anticipation). Not
required by any existing model class, but available for anyone who wants
to experiment with it later, without needing to re-run extraction.

PER-DATASET DIFFERENCES, HANDLED EXPLICITLY (not assumed uniform):
1. IMAGE PATH FORMAT. VAT's context_frame_paths are BARE, zero-padded
   filenames ("00011121.jpg") needing reconstruction via
   VAT_IMAGES_BASE_PATH/show/clip/fname. ChildPlay/VACATION/UCO-LAEO's
   context_frame_paths are already FULL, directly-openable paths (a
   deliberate schema improvement made after VAT's original design --
   confirmed via each dataset's own manifest schema doc). Handled via
   DATASET_ADAPTERS[dataset]["needs_path_reconstruction"] below, not
   assumed to be the same for every dataset.
2. CACHE KEY FORMAT. VAT's clip_id is "show/clip" (needs splitting);
   the other three datasets' clip_id has no further subdivision. Handled
   via build_cache_key() below.
3. HORIZON FIELD. VAT's manifest has ONLY horizon_frames (a fixed global
   per-manifest constant: 2/7/12 -- no per-clip fps, no target_horizon_ms
   field exists at all). The other three have BOTH target_horizon_ms
   (100/300/500, the cross-clip-comparable bucket) and horizon_frames
   (real, per-clip, fps-derived). This script itself doesn't need to
   filter by horizon at all -- it extracts embeddings for every context
   frame referenced by ANY row, valid or not, at ANY horizon, since the
   same frame's embedding is reused across horizons and overlapping
   windows regardless. Horizon filtering happens downstream, in
   train_temporal_v1.py.

CHECKPOINT COMPATIBILITY: GazeStudent is imported from
gaze_student_model.py (shared, dependency-free module -- see that file's
own docstring for why NOT importing from distillation_v1.py directly).
find_final_regression_layer()'s dynamic search means this works
correctly regardless of which GazeStudent architecture revision produced
the checkpoint being loaded.
"""

import os
import sys
import json
from collections import OrderedDict

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from gaze_student_model import (
    load_gaze_student, PenultimateFeatureExtractor, IMG_SIZE, NORM_MEAN, NORM_STD,
)

HEAD_CROP_SIZE = 112

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

DATASET = os.environ.get("EXTRACT_DATASET")
if not DATASET:
    raise ValueError(
        "EXTRACT_DATASET not set -- one of: vat, childplay, vacation, ucolaeo. "
        "Run once per dataset you need embeddings for."
    )

STATIC_CONDITION = os.environ.get("STATIC_CONDITION")
if not STATIC_CONDITION:
    raise ValueError(
        "STATIC_CONDITION not set -- one of: gt, teacher_hybrid, "
        "teacher_internvl3_alone, mtgs. Determines which frozen checkpoint "
        "produces these embeddings."
    )

STATIC_CHECKPOINT_ROOT = os.environ.get("STATIC_CHECKPOINT_ROOT")
if not STATIC_CHECKPOINT_ROOT:
    raise ValueError(
        "STATIC_CHECKPOINT_ROOT not set -- the DISTILL_OUTPUT_DIR of "
        "whichever distillation_v1.py run produced the checkpoint you "
        "want (e.g. the main run's output dir)."
    )
CHECKPOINT_PATH = os.path.join(STATIC_CHECKPOINT_ROOT, f"model_{STATIC_CONDITION}", "best_model.pt")

VIT_MODEL = os.environ.get("STATIC_VIT_MODEL", "vit_tiny_patch16_224")

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH")
if not MANIFEST_PATH:
    raise ValueError(
        "TEMPORAL_MANIFEST_PATH not set -- path to this dataset's own "
        "already-built temporal window manifest (e.g. vat_temporal_manifest.jsonl)."
    )

# Only needed for VAT (see module docstring point 1). Ignored for the
# other three datasets, whose context_frame_paths are already full paths.
IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", "")

OUTPUT_ROOT = os.environ.get("FEATURE_CACHE_ROOT")
if not OUTPUT_ROOT:
    raise ValueError("FEATURE_CACHE_ROOT not set -- base directory; this run "
                      "writes to {FEATURE_CACHE_ROOT}/{DATASET}/{STATIC_CONDITION}/.")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATASET, STATIC_CONDITION)
INDEX_PATH = os.path.join(OUTPUT_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(OUTPUT_DIR, "frame_predicted_xy.npy")
PREDICTED_INOUT_PATH = os.path.join(OUTPUT_DIR, "frame_predicted_inout.npy")
CACHE_SCHEMA_VERSION = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ADAPTERS = {
    "vat": {"needs_path_reconstruction": True},
    "childplay": {"needs_path_reconstruction": False},
    "vacation": {"needs_path_reconstruction": False},
    "ucolaeo": {"needs_path_reconstruction": False},
}
if DATASET not in DATASET_ADAPTERS:
    raise ValueError(f"Unknown EXTRACT_DATASET '{DATASET}' -- known: {list(DATASET_ADAPTERS)}")

# Rewrites a local-machine path prefix baked into a manifest at build
# time to the real HPC path -- found necessary 2026-08-10 for ChildPlay
# (confirmed: manifest paths start with "C:\repo\ChildPlay-gaze\images\").
# Empty string for either side of a dataset = no remapping attempted,
# path used as-is (VAT never needs this at all; VACATION's own local
# prefix hasn't been directly confirmed yet -- CONFIRM the real value
# before trusting this default, same discipline as everywhere else a
# path was guessed this project).
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
        # HPC side NOT confirmed the way ChildPlay/VACATION's were -- this is
        # inferred from the established data/{dataset}/... convention, not
        # verified against a real file. CONFIRM with the same standalone
        # os.path.isfile() check used for VACATION before trusting this.
        os.environ.get("UCOLAEO_HPC_PATH_PREFIX",
                        "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames"),
    ),
}


# ══════════════════════════════════════════════════════════════════════
# ── PER-DATASET ADAPTERS ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_cache_key(clip_id, subject, fname):
    """VAT's clip_id is 'show/clip' -- splits into a 4-part key matching
    how VAT's own prior scripts always keyed their cache
    ('{show}/{clip}/{subject}/{fname}'). The other three datasets' clip_id
    has no further subdivision -- 3-part key. Confirmed against each
    dataset's own extract_temporal_frame_features_*.py precedent, not
    assumed uniform."""
    if DATASET == "vat":
        show, clip = clip_id.split("/", 1)
        return f"{show}/{clip}/{subject}/{fname}"
    return f"{clip_id}/{subject}/{fname}"


def resolve_image_path(clip_id, fname):
    """VAT needs reconstruction (bare filenames); the other three
    datasets' context_frame_paths are supposed to already be full,
    resolvable paths -- but found 2026-08-10: ChildPlay's (and very
    likely VACATION's) manifest has LOCAL WINDOWS PATHS baked in from
    when it was originally built (e.g.
    "C:\\repo\\ChildPlay-gaze\\images\\..."), permanently -- a static
    JSONL file doesn't re-resolve paths, so whatever the path resolved
    to at build time is what's in there forever. PATH_REMAP below
    rewrites a known local prefix to the real HPC one for any dataset
    that needs it -- empty/unset for a dataset means passed through
    unchanged (the original, "already correct" assumption), same
    behavior as before for any dataset not affected."""
    if DATASET_ADAPTERS[DATASET]["needs_path_reconstruction"]:
        if not IMAGES_BASE_PATH:
            raise ValueError("VAT_IMAGES_BASE_PATH not set -- required for VAT.")
        show, clip = clip_id.split("/", 1)
        return os.path.join(IMAGES_BASE_PATH, show, clip, fname)

    local_prefix, hpc_prefix = PATH_REMAP.get(DATASET, (None, None))
    if local_prefix and hpc_prefix:
        normalized = fname.replace("\\", "/")
        normalized_prefix = local_prefix.replace("\\", "/")
        if normalized.startswith(normalized_prefix):
            return hpc_prefix.rstrip("/") + "/" + normalized[len(normalized_prefix):].lstrip("/")
        # Prefix didn't match what was expected -- fall through and let
        # sanity_check_image_paths() catch it with a clear failure,
        # rather than silently returning an unrewritten, still-broken path.
    return fname  # already a full, correct path (or remap not configured for this dataset)


# ══════════════════════════════════════════════════════════════════════
# ── STEP 1: read the manifest, collect unique context (clip, subject, frame) ──
# ══════════════════════════════════════════════════════════════════════

def collect_unique_context_frames(manifest_path):
    """Scans EVERY row (not just window_valid=True ones -- an offscreen
    or invalid TARGET doesn't mean the CONTEXT frames aren't perfectly
    good, reusable embeddings for some other row), collects every unique
    (clip, subject, frame) referenced as context anywhere.

    Rows with an empty/mismatched context_head_boxes are SKIPPED (logged,
    not silently dropped) rather than crashing the whole run -- found
    2026-08-10: real VAT rows exist where context_head_boxes is empty
    while context_frame_paths is populated (e.g. a subject who is
    genuinely offscreen throughout the window -- confirmed against this
    project's own static labels for the same show/clip/subject). A
    single such row shouldn't be able to take down an entire extraction
    run; the row is simply unusable as head-crop input (no box to crop),
    same "invalidate, never pad" treatment as everywhere else in this
    project."""
    unique = OrderedDict()
    n_rows = 0
    n_skipped_mismatch = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1

            clip = row["clip_id"]
            subject = row["subject_id"]
            paths = row["context_frame_paths"]
            head_boxes = row["context_head_boxes"]

            if len(head_boxes) != len(paths):
                n_skipped_mismatch += 1
                if n_skipped_mismatch <= 5:
                    print(f"[extract] SKIPPING row -- context_head_boxes/"
                          f"context_frame_paths length mismatch for {clip} "
                          f"subject={subject} ({len(head_boxes)} vs {len(paths)}).")
                continue

            for path, head_box in zip(paths, head_boxes):
                fname = os.path.basename(path)
                key = build_cache_key(clip, subject, fname)
                if key not in unique:
                    unique[key] = {
                        "clip": clip, "subject": subject, "fname": fname,
                        "image_path": resolve_image_path(clip, path),
                        "head_box": head_box,
                    }

    print(f"[extract] Scanned {n_rows} manifest rows.")
    if n_skipped_mismatch:
        pct = 100 * n_skipped_mismatch / n_rows
        flag = " -- WARNING: unexpectedly high, worth investigating" if pct > 1 else " (expected to be rare -- e.g. genuinely offscreen-throughout subjects)"
        print(f"[extract] Skipped {n_skipped_mismatch} rows ({pct:.3f}%) with "
              f"mismatched context_head_boxes{flag}.")
    print(f"[extract] {len(unique)} unique (clip, subject, frame) context "
          f"positions need embeddings.")
    return unique


def sanity_check_image_paths(unique_frames, n_check=10):
    sample_keys = list(unique_frames.keys())[:n_check]
    missing = [unique_frames[k]["image_path"] for k in sample_keys
               if not os.path.isfile(unique_frames[k]["image_path"])]
    if missing:
        print("=" * 60)
        print("IMAGE PATH CHECK FAILED -- stopping before doing real work.")
        print(f"{len(missing)}/{len(sample_keys)} sample paths do not exist, e.g.:")
        for m in missing[:5]:
            print(f"  {m}")
        if DATASET == "vat":
            print("VAT requires VAT_IMAGES_BASE_PATH -- confirm it's set correctly.")
        print("=" * 60)
        sys.exit(1)
    print(f"[extract] Image paths verified against {len(sample_keys)} sample files -- proceeding.")


# ══════════════════════════════════════════════════════════════════════
# ── STEP 2: load + crop a frame's inputs ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_transforms():
    to_tensor_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    scene_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), to_tensor_norm])
    head_transform = transforms.Compose([transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE)), to_tensor_norm])
    return scene_transform, head_transform


def load_scene_and_head_crop(image_path, head_box, scene_transform, head_transform):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = head_box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1: x1, x2 = max(0, x1 - 1), min(w, x2 + 1)
    if y2 <= y1: y1, y2 = max(0, y1 - 1), min(h, y2 + 1)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h
    head_img = img.crop((x1, y1, x2, y2))

    scene_tensor = scene_transform(img).unsqueeze(0).to(DEVICE)
    head_tensor = head_transform(head_img).unsqueeze(0).to(DEVICE)
    return scene_tensor, head_tensor


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Dataset: {DATASET}  Condition: {STATIC_CONDITION}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_frames = collect_unique_context_frames(MANIFEST_PATH)
    sanity_check_image_paths(unique_frames)

    already_cached = set()
    existing_rows = []
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_rows.append(row)
                already_cached.add(row["key"])

        old_schema = (
            not existing_rows
            or existing_rows[0].get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or not os.path.exists(PREDICTED_XY_PATH)
            or not os.path.exists(PREDICTED_INOUT_PATH)
        )
        if old_schema:
            print(f"[extract] Existing cache at {OUTPUT_DIR} has a different "
                  f"schema version or is missing an expected file. NOT resuming "
                  f"-- rename/delete {OUTPUT_DIR} and rerun for a clean "
                  f"re-extraction. If you're swapping checkpoints/conditions, "
                  f"delete the cache even if the schema version matches -- "
                  f"resume does not track which checkpoint produced the cache.")
            sys.exit(1)
        print(f"[extract] Resuming -- {len(already_cached)} keys already "
              f"cached (schema v{CACHE_SCHEMA_VERSION}), skipping those.")

    todo = [(k, v) for k, v in unique_frames.items() if k not in already_cached]
    print(f"[extract] {len(todo)} keys remaining to embed.")
    if not todo:
        print("Nothing to do -- cache already covers every referenced frame.")
        return

    print(f"[*] Device: {DEVICE}")
    model = load_gaze_student(CHECKPOINT_PATH, vit_model=VIT_MODEL, device=DEVICE)
    scene_transform, head_transform = build_transforms()
    extractor = PenultimateFeatureExtractor(model)

    existing_embeddings = np.load(EMBEDDINGS_PATH) if os.path.exists(EMBEDDINGS_PATH) else None
    existing_xy = np.load(PREDICTED_XY_PATH) if os.path.exists(PREDICTED_XY_PATH) else None
    existing_inout = np.load(PREDICTED_INOUT_PATH) if os.path.exists(PREDICTED_INOUT_PATH) else None

    new_rows = list(existing_rows)
    embed_chunks = [existing_embeddings] if existing_embeddings is not None else []
    xy_chunks = [existing_xy] if existing_xy is not None else []
    inout_chunks = [existing_inout] if existing_inout is not None else []

    batch_embed, batch_xy, batch_inout = [], [], []
    for i, (key, info) in enumerate(todo):
        try:
            scene_t, head_t = load_scene_and_head_crop(
                info["image_path"], info["head_box"], scene_transform, head_transform
            )
            feat, pred_xy, pred_inout = extractor.extract(scene_t, head_t)
        except Exception as e:
            print(f"[extract] FAILED on {key}: {e}")
            continue

        batch_embed.append(feat)
        batch_xy.append(pred_xy)
        batch_inout.append(pred_inout)
        new_rows.append({
            "key": key, "clip": info["clip"], "subject": info["subject"],
            "fname": info["fname"], "head_box": info["head_box"],
            "embedding_idx": len(new_rows), "cache_schema_version": CACHE_SCHEMA_VERSION,
        })

        if (i + 1) % 500 == 0:
            print(f"[extract] {i + 1}/{len(todo)} done.")

    if batch_embed:
        embed_chunks.append(np.stack(batch_embed).astype(np.float32))
        xy_chunks.append(np.stack(batch_xy).astype(np.float32))
        inout_chunks.append(np.array(batch_inout, dtype=np.float32).reshape(-1))

    final_embeddings = np.concatenate(embed_chunks, axis=0) if embed_chunks else np.zeros((0, 128))
    final_xy = np.concatenate(xy_chunks, axis=0) if xy_chunks else np.zeros((0, 2))
    final_inout = np.concatenate(inout_chunks, axis=0) if inout_chunks else np.zeros((0,))

    np.save(EMBEDDINGS_PATH, final_embeddings)
    np.save(PREDICTED_XY_PATH, final_xy)
    np.save(PREDICTED_INOUT_PATH, final_inout)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n[extract] Done. {final_embeddings.shape[0]} embeddings "
          f"({final_embeddings.shape[1]}-dim) + predicted_xy + predicted_inout "
          f"written to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
