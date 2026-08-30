"""
extract_temporal_frame_features_childplay.py — ChildPlay port of
extract_temporal_frame_features.py (VAT). Same frozen GT-trained static
GazeStudent checkpoint, same model/transform code verbatim — this is
cross-dataset feature extraction (the SAME static model applied to a
DIFFERENT dataset's images), not a new ChildPlay-specific model.

WHAT'S SIMPLER HERE THAN THE VAT VERSION, AND WHY:
1. NO resolve_image_path() NEEDED AT ALL. VAT's manifest stored bare
   fnames — a schema wrinkle discovered partway through VAT's own work
   (context_frame_paths' name promised a path but only ever held a
   filename, requiring reconstruction via show/clip/VAT_IMAGES_BASE_PATH
   at extraction time). ChildPlay's manifest schema was deliberately
   designed to avoid repeating that wrinkle (see
   childplay_temporal_manifest_schema.md) — context_frame_paths already
   stores fully resolvable paths, built once by the window-builder via
   childplay_annotation_utils.py's resolve_image_path(). This script just
   uses them directly.
2. NO show/clip SPLIT. VAT's clip_id was "show/clip"; ChildPlay's is a
   single flat clip name. Cache keys are "{clip}/{subject}/{fname}", one
   fewer component than VAT's.

WHAT'S IDENTICAL, COPIED VERBATIM: GazeStudent, load_gaze_student,
build_transforms, find_final_regression_layer, PenultimateFeatureExtractor,
load_scene_and_head_crop (including the degenerate-box clamping logic) —
all of this is about the STATIC MODEL, which doesn't change between
datasets. head_box format ((x1,y1,x2,y2), pixel space) matches VAT's
convention exactly — confirmed against real ChildPlay data earlier this
session via childplay_annotation_utils.py's validate_against_real_clip
output.

ONE REAL TRADE-OFF WORTH KNOWING ABOUT: because ChildPlay's manifest
bakes in FULL, ABSOLUTE paths (built wherever the window-builder was run —
currently Scott's local machine), this cache isn't automatically portable
to a different machine the way VAT's bare-fname approach was. If this
ever needs to run somewhere the manifest's baked-in root doesn't exist
(e.g. HPC), either rebuild the manifest with CHILDPLAY_ROOT set for that
machine, or use the optional PATH_REMAP_FROM/PATH_REMAP_TO env vars below
(default: no-op) rather than assuming this needs solving now.

KNOWN HARMLESS NOISE: same torch.load weights_only=False FutureWarning as
every other script that loads this checkpoint format.

RESUMABLE: same schema-versioned resumable-cache discipline as the VAT
version. CACHE_SCHEMA_VERSION starts at 1 here (not 2) — unlike VAT,
there's no legacy ChildPlay cache that predates predicted_xy to guard
against; this cache includes it from the very first run.
"""

import os
import json
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "childplay_temporal_manifest.jsonl")

#CHECKPOINT_PATH = os.environ.get("GAZE_STUDENT_CHECKPOINT", None)  # SAME frozen GT-trained VAT checkpoint — required
# PENDING (2026-07-30): was pointing at the STALE confidence_threshold_055
# checkpoint (pre-population-expansion VAT-only, ~23,096 rows) --
# deliberately left unset now rather than guessing a new default. Once
# the joint VAT+ChildPlay retrain finishes, download
# distillation_joint_vat_childplay_run/model_joint_gt/best_model.pt from
# the HPC to a local path and either set that path below or pass it via
# GAZE_STUDENT_CHECKPOINT at runtime -- whichever's more convenient.
# IMPORTANT: also delete/rename the existing childplay_temporal_feature_cache/
# folder before running with the new checkpoint -- the resume logic below
# only checks cache_schema_version, not which checkpoint produced the
# cached embeddings, so it would silently keep old-checkpoint embeddings
# mixed in with new ones otherwise.
CHECKPOINT_PATH = os.environ.get("GAZE_STUDENT_CHECKPOINT", r"C:\Users\scott\Desktop\dissertation\091_distillation_vat_childplay\distillation_joint_vat_childplay_001_resumed\distillation_joint_vat_childplay_run\model_joint_gt\best_model.pt")  # GT-trained checkpoint — required
VIT_MODEL = os.environ.get("GAZE_STUDENT_VIT_MODEL", "vit_tiny_patch16_224")

OUTPUT_DIR = os.environ.get("FEATURE_CACHE_DIR", "childplay_temporal_feature_cache")
INDEX_PATH = os.path.join(OUTPUT_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(OUTPUT_DIR, "frame_predicted_xy.npy")
CACHE_SCHEMA_VERSION = 1  # first version for ChildPlay — see module docstring

# Optional, defaults to no-op: remap the manifest's baked-in absolute path
# prefix if this ever runs on a different machine than the one the
# manifest was built on. See module docstring's portability note.
PATH_REMAP_FROM = os.environ.get("CHILDPLAY_PATH_REMAP_FROM", "")
PATH_REMAP_TO = os.environ.get("CHILDPLAY_PATH_REMAP_TO", "")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONLY_VALID_ROWS = True

IMG_SIZE = 224
HEAD_CROP_SIZE = 112
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════
# ── MODEL — copied verbatim from extract_temporal_frame_features.py (VAT) ──
# ══════════════════════════════════════════════════════════════════════

class GazeStudent(nn.Module):
    def __init__(self, vit_model="vit_tiny_patch16_224"):
        super().__init__()
        self.encoder = timm.create_model(
            vit_model, pretrained=False, num_classes=0, global_pool="token",
        )
        feat_dim = self.encoder.num_features
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 2), nn.Sigmoid(),
        )

    def forward(self, scene, head):
        if head.shape[-1] != IMG_SIZE:
            head = F.interpolate(head, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return self.fusion(torch.cat([self.encoder(scene), self.encoder(head)], dim=1))


def load_gaze_student(checkpoint_path, vit_model=VIT_MODEL):
    model = GazeStudent(vit_model=vit_model).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)  # FutureWarning expected, harmless
    state = ckpt.get("model_state", ckpt)  # tolerates raw state_dict or wrapped form
    model.load_state_dict(state)
    model.eval()
    print(f"[extract_features] Loaded checkpoint (val_ade={ckpt.get('val_ade', 'n/a')}, "
          f"epoch={ckpt.get('epoch', 'n/a')}, label_source={ckpt.get('label_source', 'n/a')})")
    return model


def build_transforms():
    """Returns (scene_transform, head_transform) — deliberately two
    different resize targets (224 vs 112), matching the VAT version and
    check_shortcut_collapse.py exactly. Do not collapse into one shared
    transform."""
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    scene_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), to_tensor_norm])
    head_transform = transforms.Compose([transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE)), to_tensor_norm])
    return scene_transform, head_transform


def find_final_regression_layer(model, expected_out_features=2):
    """Locates GazeStudent's final nn.Linear(*, expected_out_features) by
    walking the module tree — same generic approach as the VAT version,
    already verified to land correctly on fusion.8 with no hardcoded
    attribute name needed."""
    candidates = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == expected_out_features
    ]
    if not candidates:
        raise RuntimeError(
            f"No nn.Linear(out_features={expected_out_features}) found in "
            f"the model — inspect manually before proceeding."
        )
    name, module = candidates[-1]
    print(f"[extract_features] Using '{name}' as the final regression layer "
          f"(hooking its INPUT — the 128-dim penultimate embedding).")
    return module


class PenultimateFeatureExtractor:
    """Wraps a loaded GazeStudent + a forward-pre-hook on its final
    regression layer, so a normal forward() call also yields the
    penultimate feature vector as a side effect."""

    def __init__(self, model):
        self.model = model
        self._captured = None
        final_layer = find_final_regression_layer(model)
        final_layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._captured = inputs[0].detach()

    @torch.no_grad()
    def extract(self, scene_t, head_t):
        """Returns (penultimate_embedding, predicted_xy) — same design as
        the VAT version, both captured from one forward pass."""
        pred_xy = self.model(scene_t, head_t)
        if self._captured is None:
            raise RuntimeError("Hook never fired — forward() didn't reach the regression layer.")
        feat = self._captured
        self._captured = None
        return feat.squeeze(0).cpu().numpy(), pred_xy.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# ── STEP 1: read the manifest, collect unique (subject, frame) keys ────
# ══════════════════════════════════════════════════════════════════════

def remap_path(path):
    """No-op unless CHILDPLAY_PATH_REMAP_FROM/TO are set. See module
    docstring's portability note."""
    if PATH_REMAP_FROM and path.startswith(PATH_REMAP_FROM):
        return PATH_REMAP_TO + path[len(PATH_REMAP_FROM):]
    return path


def collect_unique_context_frames(manifest_path):
    """Returns an OrderedDict: key -> {clip, subject, fname, image_path,
    head_box} for every distinct (clip, subject, image_path) referenced
    anywhere in context_frame_paths across the manifest. Dedup matters
    the same way it did for VAT: the same (subject, frame) pair recurs
    across overlapping windows and all 3 horizon rows built from the same
    window. No show/clip split needed (ChildPlay's clip_id is already
    flat) and no path reconstruction needed (context_frame_paths already
    stores full, resolvable paths — see module docstring point 1)."""
    unique = OrderedDict()
    n_rows = 0
    n_skipped_invalid = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1
            if ONLY_VALID_ROWS and not row["window_valid"]:
                n_skipped_invalid += 1
                continue

            clip = row["clip_id"]
            subject = row["subject_id"]
            paths = row["context_frame_paths"]
            head_boxes = row["context_head_boxes"]

            if len(head_boxes) != len(paths):
                raise ValueError(
                    f"context_head_boxes/context_frame_paths length mismatch "
                    f"for {clip} subject={subject} "
                    f"({len(head_boxes)} vs {len(paths)}) — investigate "
                    f"before trusting this row."
                )

            for path, head_box in zip(paths, head_boxes):
                fname = os.path.basename(path)
                key = f"{clip}/{subject}/{fname}"
                if key not in unique:
                    unique[key] = {
                        "clip": clip, "subject": subject,
                        "fname": fname, "image_path": remap_path(path),
                        "head_box": head_box,
                    }

    print(f"[extract_features] Scanned {n_rows} manifest rows "
          f"({n_skipped_invalid} invalid, skipped).")
    print(f"[extract_features] {len(unique)} unique (subject, frame) pairs "
          f"need embeddings.")
    return unique


# ══════════════════════════════════════════════════════════════════════
# ── STEP 2: sanity-check image paths ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def sanity_check_image_paths(unique_frames, n_check=10):
    """Fail fast and loud if the manifest's baked-in paths don't resolve
    on this machine (e.g. running somewhere other than where the
    manifest was built — see module docstring's portability note),
    rather than silently failing partway through a long run."""
    sample_keys = list(unique_frames.keys())[:n_check]
    missing = []
    for key in sample_keys:
        info = unique_frames[key]
        if not os.path.isfile(info["image_path"]):
            missing.append(info["image_path"])

    if missing:
        print("=" * 60)
        print("IMAGE PATH CHECK FAILED — stopping before doing real work.")
        print(f"{len(missing)}/{len(sample_keys)} sample paths do not exist, e.g.:")
        for m in missing[:5]:
            print(f"  {m}")
        print("If running on a different machine than where the manifest was "
              "built, set CHILDPLAY_PATH_REMAP_FROM/CHILDPLAY_PATH_REMAP_TO, "
              "or rebuild the manifest with CHILDPLAY_ROOT set correctly here.")
        print("=" * 60)
        sys.exit(1)

    print(f"[extract_features] Image paths verified against "
          f"{len(sample_keys)} sample files — proceeding.")


# ══════════════════════════════════════════════════════════════════════
# ── STEP 3: load + crop a frame's inputs — copied verbatim from VAT ─────
# ══════════════════════════════════════════════════════════════════════

def load_scene_and_head_crop(image_path, head_box, scene_transform, head_transform):
    """Degenerate-box clamping logic ported unchanged from the VAT
    version (originally from check_shortcut_collapse.py's
    load_image_pair) — same handling for boxes that touch/exceed image
    bounds or collapse to zero width/height."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = head_box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1:
        x1, x2 = max(0, x1 - 1), min(w, x2 + 1)
    if y2 <= y1:
        y1, y2 = max(0, y1 - 1), min(h, y2 + 1)
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
    if not CHECKPOINT_PATH:
        print("ERROR: set GAZE_STUDENT_CHECKPOINT to the GT-trained static "
              "checkpoint path before running (the SAME VAT checkpoint used "
              "throughout Objective 4 — this extracts embeddings for "
              "ChildPlay's images using that already-frozen model, not a "
              "newly trained one).")
        sys.exit(1)

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
            or "cache_schema_version" not in existing_rows[0]
            or existing_rows[0].get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or not os.path.exists(PREDICTED_XY_PATH)
        )
        if old_schema:
            print(f"[extract_features] Existing cache at {OUTPUT_DIR} has a "
                  f"different schema version or is missing {PREDICTED_XY_PATH}. "
                  f"NOT resuming from it — rename/delete {OUTPUT_DIR} and "
                  f"rerun for a clean full re-extraction.")
            sys.exit(1)

        print(f"[extract_features] Resuming — {len(already_cached)} keys "
              f"already cached (schema v{CACHE_SCHEMA_VERSION}), skipping those.")

    todo = [(k, v) for k, v in unique_frames.items() if k not in already_cached]
    print(f"[extract_features] {len(todo)} keys remaining to embed.")

    if not todo:
        print("Nothing to do — cache already covers every referenced frame.")
        return

    print(f"[*] Device: {DEVICE}")
    model = load_gaze_student(CHECKPOINT_PATH)
    scene_transform, head_transform = build_transforms()
    extractor = PenultimateFeatureExtractor(model)

    existing_embeddings = None
    existing_predicted_xy = None
    if os.path.exists(EMBEDDINGS_PATH):
        existing_embeddings = np.load(EMBEDDINGS_PATH)
    if os.path.exists(PREDICTED_XY_PATH):
        existing_predicted_xy = np.load(PREDICTED_XY_PATH)

    new_rows = list(existing_rows)
    new_embedding_chunks = [existing_embeddings] if existing_embeddings is not None else []
    new_predicted_xy_chunks = [existing_predicted_xy] if existing_predicted_xy is not None else []
    embed_dim = None

    batch_embed = []
    batch_xy = []
    for i, (key, info) in enumerate(todo):
        try:
            scene_t, head_t = load_scene_and_head_crop(
                info["image_path"], info["head_box"], scene_transform, head_transform
            )
            feat, pred_xy = extractor.extract(scene_t, head_t)
        except Exception as e:
            print(f"[extract_features] FAILED on {key}: {e}")
            continue

        if embed_dim is None:
            embed_dim = feat.shape[0]
        batch_embed.append(feat)
        batch_xy.append(pred_xy)
        new_rows.append({
            "key": key, "clip": info["clip"],
            "subject": info["subject"], "fname": info["fname"],
            "head_box": info["head_box"],
            "embedding_idx": len(new_rows),
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        })

        if (i + 1) % 500 == 0:
            print(f"[extract_features] {i + 1}/{len(todo)} done.")

    if batch_embed:
        new_embedding_chunks.append(np.stack(batch_embed).astype(np.float32))
        new_predicted_xy_chunks.append(np.stack(batch_xy).astype(np.float32))

    final_embeddings = (
        np.concatenate(new_embedding_chunks, axis=0)
        if new_embedding_chunks else np.zeros((0, embed_dim or 0))
    )
    final_predicted_xy = (
        np.concatenate(new_predicted_xy_chunks, axis=0)
        if new_predicted_xy_chunks else np.zeros((0, 2))
    )
    np.save(EMBEDDINGS_PATH, final_embeddings)
    np.save(PREDICTED_XY_PATH, final_predicted_xy)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")

    dim_str = final_embeddings.shape[1] if final_embeddings.size else "?"
    print(f"[extract_features] Done. {final_embeddings.shape[0]} embeddings "
          f"({dim_str}-dim) + predicted_xy written to {OUTPUT_DIR}, "
          f"index at {INDEX_PATH}.")


if __name__ == "__main__":
    main()
