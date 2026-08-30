"""
temporal_window_dataset.py — PyTorch Dataset for the Temporal PoC (Objective 4)

WHAT THIS DOES: reads vat_temporal_manifest.jsonl, filters to a single
horizon (300ms/7-frame by default, per the agreed PoC scope), looks up
each window's 8 cached context embeddings from frame_features_index.jsonl
/ frame_features.npy (built by extract_temporal_frame_features.py), and
yields (embedding_sequence, normalized_target_xy) pairs ready for a GRU or
Transformer head. No pixels touched at __getitem__ time — everything
after this point trains on cached feature vectors.

FILTERING, IN ORDER: horizon_frames == target horizon; window_valid ==
True; target_gaze_x is not None (excludes off-screen targets — same
convention as every other ADE number in this project, e.g.
check_shortcut_collapse.py's load_records and
ade_distance_by_gaze_type_comparison.py's on_screen_df split). Rows whose
8 context embeddings aren't all present in the cache (e.g. an image that
failed during extraction) are dropped with a logged count rather than
crashing or silently zero-filling.

TARGET NORMALIZATION: target_gaze_x/y in the manifest are raw pixels.
Normalized here by dividing by each row's own CLIP's actual image
dimensions (not a fixed reference) — the same convention confirmed via
ade_distance_by_gaze_type_comparison.py's "-1/width" bug description and
cross-checked against compute_last_known_position_baseline.py's corrected
baseline_ADE_norm. Keeps the temporal target on the same 0-1-ish scale as
the static student's Sigmoid output and makes any resulting ADE directly
comparable to baseline_ADE_norm (~0.0446 at horizon_frames=7) rather than
silently reporting in different units.

TRAIN/VAL SPLIT: by clip, never by window — matches the project's
existing rule (no data leakage between overlapping windows of the same
clip). build_clip_split() deliberately does sorted(set(...)) BEFORE
shuffling, replicating the exact fix already made once in this project
for step52's non-deterministic list(set(...)) bug (Python's hash
randomization made that split different every process launch) — not
re-introducing a bug that's already been found and fixed elsewhere.
"""

import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "vat_temporal_manifest.jsonl")
MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", r"C:\repo\standard_project\vat_missing_annotation_investigation\vat_temporal_manifest.jsonl")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "temporal_feature_cache")
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

# If True, each context frame's static-model-predicted (x, y) is
# concatenated onto its 128-dim embedding (-> 130-dim per frame) before
# being handed to the temporal head. Added after the first PoC run showed
# both GRU and Transformer badly underperforming baseline with a fast-
# overfit signature -- working hypothesis: the temporal head had no
# geometrically-meaningful position signal to build on, only an opaque
# pre-regression embedding. Requires a v2-schema feature cache (see
# extract_temporal_frame_features.py's CACHE_SCHEMA_VERSION) -- falls
# back to embeddings-only with a clear warning if frame_predicted_xy.npy
# isn't present, rather than failing silently.
INCLUDE_PREDICTED_XY = os.environ.get("INCLUDE_PREDICTED_XY", "1") == "1"

VAT_ROOT = os.environ.get(
    "VAT_ROOT",
    r"C:\Users\scott\Desktop\dissertation\0_dataset\videoattentiontarget",
)
VAT_IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", os.path.join(VAT_ROOT, "images"))

PRIMARY_HORIZON_FRAMES = 7  # 300ms — the sole PoC scope for now, per the agreed plan


# ══════════════════════════════════════════════════════════════════════
# ── CLIP-LEVEL TRAIN/VAL SPLIT ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_clip_split(manifest_path, val_frac=0.15, seed=42):
    """Deterministic clip-level split. sorted(set(...)) BEFORE shuffling —
    see module docstring for why this matters (step52's exact bug)."""
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


# ══════════════════════════════════════════════════════════════════════
# ── DATASET ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class TemporalWindowDataset(Dataset):
    def __init__(self, manifest_path=MANIFEST_PATH, index_path=INDEX_PATH,
                 embeddings_path=EMBEDDINGS_PATH, images_base_path=VAT_IMAGES_BASE_PATH,
                 horizon_frames=PRIMARY_HORIZON_FRAMES, split_clip_ids=None,
                 split_name="(unfiltered)"):
        self.images_base_path = images_base_path
        self.embeddings = np.load(embeddings_path)

        self.predicted_xy = None
        if INCLUDE_PREDICTED_XY:
            if os.path.exists(PREDICTED_XY_PATH):
                self.predicted_xy = np.load(PREDICTED_XY_PATH)
                if self.predicted_xy.shape[0] != self.embeddings.shape[0]:
                    raise ValueError(
                        f"frame_predicted_xy.npy has {self.predicted_xy.shape[0]} rows "
                        f"but frame_features.npy has {self.embeddings.shape[0]} -- "
                        f"caches are out of sync, re-run extraction cleanly."
                    )
                print(f"[TemporalWindowDataset] Including predicted_xy -- "
                      f"per-frame feature dim is {self.embeddings.shape[1]} + 2 "
                      f"= {self.embeddings.shape[1] + 2}.")
            else:
                print(f"[TemporalWindowDataset] INCLUDE_PREDICTED_XY=1 but "
                      f"{PREDICTED_XY_PATH} not found -- falling back to "
                      f"embeddings-only ({self.embeddings.shape[1]}-dim). "
                      f"Re-run extraction with the updated script for the "
                      f"explicit-position input.")

        self.key_to_idx = {}
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.key_to_idx[row["key"]] = row["embedding_idx"]

        self.rows = []
        counts = defaultdict(int)

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                counts["total"] += 1

                if row["horizon_frames"] != horizon_frames:
                    counts["wrong_horizon"] += 1
                    continue
                if not row["window_valid"]:
                    counts["invalid_window"] += 1
                    continue
                if row["target_gaze_x"] is None:
                    counts["offscreen_target"] += 1
                    continue
                if split_clip_ids is not None and row["clip_id"] not in split_clip_ids:
                    counts["not_in_split"] += 1
                    continue

                show, clip = row["clip_id"].split("/", 1)
                subject = row["subject_id"]
                keys = [f"{show}/{clip}/{subject}/{fn}" for fn in row["context_frame_paths"]]
                if not all(k in self.key_to_idx for k in keys):
                    counts["missing_embedding"] += 1
                    continue

                self.rows.append({
                    "show": show, "clip": clip, "subject": subject,
                    "clip_id": row["clip_id"],
                    "embed_indices": [self.key_to_idx[k] for k in keys],
                    "last_context_fname": row["context_frame_paths"][-1],
                    "target_gaze_x": float(row["target_gaze_x"]),
                    "target_gaze_y": float(row["target_gaze_y"]),
                    "target_gaze_type": row["target_gaze_type"],
                })
                counts["kept"] += 1

        self._dim_cache = {}
        self._resolve_all_dimensions()

        print(f"[TemporalWindowDataset:{split_name}] {counts['kept']} usable rows "
              f"(horizon_frames={horizon_frames}). Dropped: "
              f"{counts['wrong_horizon']} wrong-horizon, "
              f"{counts['invalid_window']} invalid-window, "
              f"{counts['offscreen_target']} offscreen-target, "
              f"{counts['not_in_split']} not-in-split, "
              f"{counts['missing_embedding']} missing-embedding.")

    def _resolve_all_dimensions(self):
        """Fail loud, fail early, per-clip — not silently mid-training.
        Any clip whose real image dimensions can't be read has its rows
        dropped, with a clear count, rather than crashing __getitem__
        deep into an epoch."""
        needed_clips = {(r["show"], r["clip"], r["last_context_fname"]) for r in self.rows}
        bad_clips = set()
        for show, clip, sample_fname in needed_clips:
            key = (show, clip)
            if key in self._dim_cache:
                continue
            path = os.path.join(self.images_base_path, show, clip, sample_fname)
            try:
                with Image.open(path) as img:
                    self._dim_cache[key] = img.size  # (w, h)
            except Exception as e:
                print(f"[TemporalWindowDataset] Could not read dimensions for "
                      f"{show}/{clip}: {e}")
                bad_clips.add(key)

        if bad_clips:
            before = len(self.rows)
            self.rows = [r for r in self.rows if (r["show"], r["clip"]) not in bad_clips]
            print(f"[TemporalWindowDataset] Dropped {before - len(self.rows)} rows "
                  f"from {len(bad_clips)} clip(s) with unreadable dimensions.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        embed_seq = self.embeddings[r["embed_indices"]]  # (8, D)
        if self.predicted_xy is not None:
            xy_seq = self.predicted_xy[r["embed_indices"]]  # (8, 2)
            embed_seq = np.concatenate([embed_seq, xy_seq], axis=1)  # (8, D+2)

        w, h = self._dim_cache[(r["show"], r["clip"])]
        target = torch.tensor(
            [r["target_gaze_x"] / w, r["target_gaze_y"] / h], dtype=torch.float32
        )

        return {
            "embeddings": torch.from_numpy(embed_seq).float(),  # (8, D) or (8, D+2)
            "target": target,                                    # (2,) normalized
            "clip_id": r["clip_id"],
            "subject": r["subject"],
            "target_gaze_type": r["target_gaze_type"],
        }


# ══════════════════════════════════════════════════════════════════════
# ── SMOKE TEST ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_ids, val_ids = build_clip_split(MANIFEST_PATH)
    print(f"Clip split: {len(train_ids)} train clips, {len(val_ids)} val clips.")

    train_ds = TemporalWindowDataset(split_clip_ids=train_ids, split_name="train")
    val_ds = TemporalWindowDataset(split_clip_ids=val_ids, split_name="val")

    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print("\nSample batch shapes:")
    print(f"  embeddings : {batch['embeddings'].shape}  (expect [B, 8, 128] or [B, 8, 130] if predicted_xy is included)")
    print(f"  target     : {batch['target'].shape}  (expect [B, 2])")
    print(f"  target range: x in [{batch['target'][:,0].min():.3f}, "
          f"{batch['target'][:,0].max():.3f}], "
          f"y in [{batch['target'][:,1].min():.3f}, {batch['target'][:,1].max():.3f}] "
          f"(expect roughly within 0-1, confirms normalization is sane)")
