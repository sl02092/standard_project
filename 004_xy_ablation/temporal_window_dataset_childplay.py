"""
temporal_window_dataset_childplay.py — ChildPlay port of
temporal_window_dataset.py (VAT). PyTorch Dataset for the ChildPlay
Temporal PoC.

WHAT'S IDENTICAL: __getitem__'s core shape (embeddings + optional
predicted_xy concatenation, normalized target), the "resolve all
dimensions up front, fail loud per-clip" pattern, the off-screen/missing-
embedding filtering logic, and INCLUDE_PREDICTED_XY's graceful fallback.

THREE REAL DIFFERENCES FROM VAT, NOT JUST RENAMES:
1. SPLIT SOURCE: replaces build_clip_split()'s custom seeded shuffle with
   ChildPlay's own official train/val/test split (clips.csv), matching
   the same adaptation already made in
   compute_last_known_position_baseline_childplay.py. "test" clips are
   deliberately excluded from both train_ids and val_ids here — reserved,
   unused for PoC-level experimentation, same as the static distillation
   work's own held-out test population.
2. HORIZON FILTER: PRIMARY_HORIZON_FRAMES (a single global frame-count
   constant) doesn't make sense for ChildPlay, where horizon_frames is
   computed per clip from that clip's real fps. Filters by
   target_horizon_ms instead (the fixed, cross-clip-comparable bucket) —
   same adaptation already made in the ChildPlay baseline script.
3. KEYS AND PATHS: no show/clip split (ChildPlay's clip_id is already
   flat) and no VAT_IMAGES_BASE_PATH reconstruction needed —
   context_frame_paths are already full, resolvable paths (see
   childplay_temporal_manifest_schema.md). Cache keys are
   "{clip}/{subject}/{fname}", matching
   extract_temporal_frame_features_childplay.py's key format exactly —
   confirmed identical (both built from the same basename convention),
   not just assumed to match.
"""

import os
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from childplay_annotation_utils import load_clip_metadata

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "childplay_temporal_manifest.jsonl")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "childplay_temporal_feature_cache")
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

INCLUDE_PREDICTED_XY = os.environ.get("INCLUDE_PREDICTED_XY", "1") == "1"

PRIMARY_HORIZON_MS = 300  # matches the VAT PoC's own scope


# ══════════════════════════════════════════════════════════════════════
# ── CLIP-LEVEL TRAIN/VAL SPLIT — ChildPlay's OWN official split ────────
# ══════════════════════════════════════════════════════════════════════

def build_clip_split(manifest_path=MANIFEST_PATH):
    """Unlike VAT's build_clip_split(), this is NOT a custom seeded
    shuffle — ChildPlay ships its own author-defined train/val/test split
    (clips.csv), used directly (see schema doc: "ChildPlay's own official
    splits.csv, not a custom clip-level shuffle"). "test" clips are
    deliberately excluded from both returned sets — reserved, not used
    for PoC-level experimentation. manifest_path kept as a parameter for
    signature-compatibility with VAT's version, but unused here — the
    split comes from clips.csv, not the manifest."""
    split_map = {cid: meta["split"] for cid, meta in load_clip_metadata().items()}
    train_ids = {cid for cid, s in split_map.items() if s == "train"}
    val_ids = {cid for cid, s in split_map.items() if s == "val"}
    return train_ids, val_ids


# ══════════════════════════════════════════════════════════════════════
# ── DATASET ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class TemporalWindowDataset(Dataset):
    def __init__(self, manifest_path=MANIFEST_PATH, index_path=INDEX_PATH,
                 embeddings_path=EMBEDDINGS_PATH,
                 horizon_ms=PRIMARY_HORIZON_MS, split_clip_ids=None,
                 split_name="(unfiltered)"):
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
                      f"embeddings-only ({self.embeddings.shape[1]}-dim).")

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

                if row["target_horizon_ms"] != horizon_ms:
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

                clip = row["clip_id"]
                subject = row["subject_id"]
                fnames = [os.path.basename(p) for p in row["context_frame_paths"]]
                keys = [f"{clip}/{subject}/{fn}" for fn in fnames]
                if not all(k in self.key_to_idx for k in keys):
                    counts["missing_embedding"] += 1
                    continue

                self.rows.append({
                    "clip": clip, "subject": subject,
                    "clip_id": clip,
                    "embed_indices": [self.key_to_idx[k] for k in keys],
                    "last_context_path": row["context_frame_paths"][-1],
                    "target_gaze_x": float(row["target_gaze_x"]),
                    "target_gaze_y": float(row["target_gaze_y"]),
                    "target_gaze_type": row["target_gaze_type"],
                })
                counts["kept"] += 1

        self._dim_cache = {}
        self._resolve_all_dimensions()

        print(f"[TemporalWindowDataset:{split_name}] {counts['kept']} usable rows "
              f"(target_horizon_ms={horizon_ms}). Dropped: "
              f"{counts['wrong_horizon']} wrong-horizon, "
              f"{counts['invalid_window']} invalid-window, "
              f"{counts['offscreen_target']} offscreen-target, "
              f"{counts['not_in_split']} not-in-split, "
              f"{counts['missing_embedding']} missing-embedding.")

    def _resolve_all_dimensions(self):
        """Fail loud, fail early, per-clip — same discipline as VAT's
        version. context_frame_paths are already full paths, so no
        path-reconstruction is needed here, just an open."""
        needed = {(r["clip"], r["last_context_path"]) for r in self.rows}
        bad_clips = set()
        for clip, path in needed:
            if clip in self._dim_cache:
                continue
            try:
                with Image.open(path) as img:
                    self._dim_cache[clip] = img.size  # (w, h)
            except Exception as e:
                print(f"[TemporalWindowDataset] Could not read dimensions for "
                      f"{clip}: {e}")
                bad_clips.add(clip)

        if bad_clips:
            before = len(self.rows)
            self.rows = [r for r in self.rows if r["clip"] not in bad_clips]
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

        w, h = self._dim_cache[r["clip"]]
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
    train_ids, val_ids = build_clip_split()
    print(f"Official clip split: {len(train_ids)} train clips, {len(val_ids)} val clips "
          f"('test' clips excluded -- reserved).")

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
