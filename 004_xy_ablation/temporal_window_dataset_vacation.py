"""
temporal_window_dataset_vacation.py — VACATION port of
temporal_window_dataset_ucolaeo.py. PyTorch Dataset for the VACATION
Temporal PoC.

WHAT'S IDENTICAL, UNCHANGED: __getitem__'s core shape, the "resolve all
dimensions up front, fail loud per-video" pattern, the missing-embedding
filtering logic, and INCLUDE_PREDICTED_XY's graceful fallback. Possible
because step_temporal_window_builder_vacation.py's manifest schema
matches the same field names (clip_id, subject_id, context_frame_paths,
context_head_boxes, window_valid, target_horizon_ms, target_gaze_x/y,
target_gaze_type) used throughout this project's temporal pipelines.

THE ONE REAL DIFFERENCE: SPLIT SOURCE, AND IT'S GENUINELY DIFFERENT FROM
BOTH PRIOR DATASETS, NOT JUST A RENAME. ChildPlay has one official,
complete splits.csv. UCO-LAEO's split is per-row, already baked into the
H5 origin. VACATION's official readme.txt split covers only 237 of the
real ~299 videos -- the remaining ~62 "extra" videos (folded into train
only, per the static pipeline's own decision) aren't enumerable from the
readme alone. build_clip_split() below resolves this by scanning the
manifest's own distinct clip_id values (the real, actually-referenced
video population) and classifying each via
vacation_temporal_utils.get_video_split()'s fallback-to-train rule,
rather than assuming a fixed universe the way ChildPlay's version could.

target_gaze_type here is genuinely "social" OR "object" -- unlike
UCO-LAEO, this dataset's eventual PoC can ask the full object/social
breakdown question VAT and ChildPlay could.
"""

import os
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from vacation_temporal_utils import load_official_split, get_video_split

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "vacation_temporal_manifest.jsonl")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "vacation_temporal_feature_cache")
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

INCLUDE_PREDICTED_XY = os.environ.get("INCLUDE_PREDICTED_XY", "1") == "1"

PRIMARY_HORIZON_MS = 300  # matches every other dataset's PoC scope; also
                          # a real bucket in HORIZON_MS_LIST=[100,300,500]


# ══════════════════════════════════════════════════════════════════════
# ── VIDEO-LEVEL TRAIN/VAL SPLIT — from the manifest's real population + ──
# ── the official readme lists, NOT a fixed enumerable universe ─────────
# ══════════════════════════════════════════════════════════════════════

def build_clip_split(manifest_path=MANIFEST_PATH):
    """Returns (train_ids, val_ids) -- sets of video_id strings. Scans
    the manifest's own distinct clip_id values (the real population this
    pipeline actually produced windows for) rather than assuming a fixed
    universe, since VACATION's official split only covers part of the
    real video population -- see module docstring. "test" videos are
    excluded from both returned sets, same reserved-not-used convention
    as every other dataset; a video only ends up there via explicit
    official test_list membership, never via the extras fallback."""
    split_map = load_official_split()
    video_ids = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            video_ids.add(row["clip_id"])

    train_ids, val_ids = set(), set()
    for vid in video_ids:
        split = get_video_split(vid, split_map)
        if split == "train":
            train_ids.add(vid)
        elif split == "val":
            val_ids.add(vid)
        # "test" -> excluded from both, reserved
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
                    counts["unresolved_target"] += 1
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
              f"{counts['unresolved_target']} unresolved-target, "
              f"{counts['not_in_split']} not-in-split, "
              f"{counts['missing_embedding']} missing-embedding.")

    def _resolve_all_dimensions(self):
        """Fail loud, fail early, per-video."""
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
                  f"from {len(bad_clips)} video(s) with unreadable dimensions.")

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
            "embeddings": torch.from_numpy(embed_seq).float(),
            "target": target,
            "clip_id": r["clip_id"],
            "subject": r["subject"],
            "target_gaze_type": r["target_gaze_type"],
        }


# ══════════════════════════════════════════════════════════════════════
# ── SMOKE TEST ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_ids, val_ids = build_clip_split()
    print(f"Video split: {len(train_ids)} train videos, {len(val_ids)} val videos "
          f"('test' videos excluded -- reserved).")

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
