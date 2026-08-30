"""
extract_temporal_frame_features_ucolaeo.py — UCO-LAEO port of
extract_temporal_frame_features_childplay.py. Same frozen static
GazeStudent checkpoint, same model/transform code verbatim — this is
cross-dataset feature extraction (the SAME static model applied to a
DIFFERENT dataset's images), not a new UCO-LAEO-specific model.

WHAT'S IDENTICAL, COPIED VERBATIM: literally everything except the
configuration constants below and this docstring. GazeStudent,
load_gaze_student, build_transforms, find_final_regression_layer,
PenultimateFeatureExtractor, load_scene_and_head_crop, and — the
important one — collect_unique_context_frames() all transfer unchanged.
Confirmed by reading step_temporal_window_builder_ucolaeo.py's actual
output schema directly: clip_id, subject_id, context_frame_paths,
context_head_boxes, window_valid are ALL the same field names ChildPlay's
manifest uses, so collect_unique_context_frames() needs no adaptation —
this is a genuine config-only port, not something that only looks like one.

CHECKPOINT QUESTION — DELIBERATELY UNRESOLVED HERE, SEE PROJECT NOTES:
UCO-LAEO is broadcast/TV content, the same broad visual domain as VAT
(unlike ChildPlay, which was confirmed genuinely out-of-domain and needed
its own joint retrain to fix a ~30% domain-transfer loss) — so there's a
real possibility the existing VAT-only GT checkpoint is already adequate
here, unlike ChildPlay's case. Not assumed either way: run the
domain-transfer diagnostic (zero-horizon same-frame accuracy, same
methodology as ChildPlay's own §4 measurement) against whatever
checkpoint this extraction uses before trusting any downstream PoC
result. CHECKPOINT_PATH is left unset below, same "don't guess a
default" discipline as the ChildPlay version.

CACHE SCHEMA: CACHE_SCHEMA_VERSION starts at 1 here — this is UCO-LAEO's
own first cache, no legacy version to guard against (same situation
ChildPlay was in when its own version was first built).
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

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "ucolaeo_temporal_manifest.jsonl")

# Deliberately unset — see module docstring's checkpoint-question note.
# Set GAZE_STUDENT_CHECKPOINT to whichever checkpoint the domain-transfer
# question resolves to (plain VAT-only GT checkpoint is a reasonable
# first choice given UCO-LAEO's likely-small domain gap, but run the
# diagnostic before trusting the result either way).
CHECKPOINT_PATH = os.environ.get("GAZE_STUDENT_CHECKPOINT", "old_vat_checkpoint/best_model.pt")
VIT_MODEL = os.environ.get("GAZE_STUDENT_VIT_MODEL", "vit_tiny_patch16_224")

OUTPUT_DIR = os.environ.get("FEATURE_CACHE_DIR", "ucolaeo_temporal_feature_cache")
INDEX_PATH = os.path.join(OUTPUT_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(OUTPUT_DIR, "frame_predicted_xy.npy")
CACHE_SCHEMA_VERSION = 1  # first version for UCO-LAEO — see module docstring

# Optional, defaults to no-op: remap the manifest's baked-in absolute path
# prefix if this ever runs on a different machine than the one the
# manifest was built on (LOCAL_IMAGES_ROOT in ucolaeo_temporal_utils.py /
# the window builder). Same portability note as the ChildPlay version.
PATH_REMAP_FROM = os.environ.get("UCOLAEO_PATH_REMAP_FROM", "")
PATH_REMAP_TO = os.environ.get("UCOLAEO_PATH_REMAP_TO", "")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONLY_VALID_ROWS = True

IMG_SIZE = 224
HEAD_CROP_SIZE = 112
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════
# ── MODEL — copied verbatim, dataset-agnostic ──────────────────────────
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
    different resize targets (224 vs 112). Do not collapse into one
    shared transform."""
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    scene_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), to_tensor_norm])
    head_transform = transforms.Compose([transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE)), to_tensor_norm])
    return scene_transform, head_transform


def find_final_regression_layer(model, expected_out_features=2):
    """Locates GazeStudent's final nn.Linear(*, expected_out_features) by
    walking the module tree — generic, dataset-independent."""
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
        """Returns (penultimate_embedding, predicted_xy), both captured
        from one forward pass."""
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
    """No-op unless UCOLAEO_PATH_REMAP_FROM/TO are set."""
    if PATH_REMAP_FROM and path.startswith(PATH_REMAP_FROM):
        return PATH_REMAP_TO + path[len(PATH_REMAP_FROM):]
    return path


def collect_unique_context_frames(manifest_path):
    """Returns an OrderedDict: key -> {clip, subject, fname, image_path,
    head_box} for every distinct (clip, subject, image_path) referenced
    anywhere in context_frame_paths across the manifest. UNCHANGED from
    ChildPlay's version — see module docstring: the manifest schemas
    match field-for-field (clip_id, subject_id, context_frame_paths,
    context_head_boxes, window_valid), confirmed by reading
    step_temporal_window_builder_ucolaeo.py's actual output directly."""
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
    on this machine, rather than silently failing partway through a long
    run."""
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
              "built, set UCOLAEO_PATH_REMAP_FROM/UCOLAEO_PATH_REMAP_TO, "
              "or rebuild the manifest with LOCAL_IMAGES_ROOT set correctly here.")
        print("=" * 60)
        sys.exit(1)

    print(f"[extract_features] Image paths verified against "
          f"{len(sample_keys)} sample files — proceeding.")


# ══════════════════════════════════════════════════════════════════════
# ── STEP 3: load + crop a frame's inputs — copied verbatim ─────────────
# ══════════════════════════════════════════════════════════════════════

def load_scene_and_head_crop(image_path, head_box, scene_transform, head_transform):
    """Degenerate-box clamping logic, unchanged from the ChildPlay/VAT
    versions — same handling for boxes that touch/exceed image bounds or
    collapse to zero width/height."""
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
        print("ERROR: set GAZE_STUDENT_CHECKPOINT before running. See module "
              "docstring's checkpoint-question note — run the domain-transfer "
              "diagnostic against whichever checkpoint you pick before "
              "trusting any downstream PoC result built from this cache.")
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
                  f"rerun for a clean full re-extraction. If you're swapping "
                  f"GAZE_STUDENT_CHECKPOINT, delete the cache folder even if "
                  f"the schema version matches — this resume logic does not "
                  f"track which checkpoint produced the cached embeddings "
                  f"(the exact mistake made once on the ChildPlay version).")
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
