"""
temporal_window_dataset_ucolaeo.py — UCO-LAEO port of
temporal_window_dataset_childplay.py. PyTorch Dataset for the UCO-LAEO
Temporal PoC.

WHAT'S IDENTICAL, UNCHANGED FROM CHILDPLAY'S VERSION: __getitem__'s core
shape (embeddings + optional predicted_xy concatenation, normalized
target), the "resolve all dimensions up front, fail loud per-video"
pattern, the missing-embedding filtering logic, and
INCLUDE_PREDICTED_XY's graceful fallback. Possible because
step_temporal_window_builder_ucolaeo.py's manifest schema matches
ChildPlay's field-for-field (clip_id, subject_id, context_frame_paths,
context_head_boxes, window_valid, target_horizon_ms, target_gaze_x/y,
target_gaze_type) -- confirmed by reading both manifest schemas directly,
not assumed from the two datasets being "similar enough."

THE ONE REAL DIFFERENCE: SPLIT SOURCE. ChildPlay has an official
splits.csv, read via childplay_annotation_utils.load_clip_metadata().
UCO-LAEO's temporal manifest carries no "split" field of its own (unlike
its static manifest / the H5 files themselves), but
ucolaeo_temporal_utils.build_person_timelines() already tracks split per
(video_id, person_id) as a side effect of building the canonical
timelines -- split is inherently per-VIDEO here (a video's H5 origin
determines its split for every person in it, never mixed across splits),
so build_video_split() below just collapses that down to one entry per
video_id. This is a real read of already-resolved data, not a new
computation -- same "reuse, don't re-derive" reasoning as
compute_last_known_position_baseline_ucolaeo.py.

NAMING NOTE: build_clip_split() keeps ChildPlay's own function name
(rather than e.g. build_video_split() as the public interface) so
train_temporal_poc_ucolaeo.py's import line matches the ChildPlay
script's calling convention exactly -- `from
temporal_window_dataset_ucolaeo import TemporalWindowDataset,
build_clip_split, ...` -- even though "clip" doesn't really fit UCO-LAEO's
per-video unit. Terminology mismatch, not a design error.

KEYS AND PATHS: same as ChildPlay -- no show/clip split needed (UCO-LAEO's
clip_id IS the video_id, no finer subdivision), context_frame_paths are
already full LOCAL paths (built by build_image_path() in the window
builder), so no path reconstruction is needed here either. Cache keys are
"{clip}/{subject}/{fname}", matching
extract_temporal_frame_features_ucolaeo.py's own key format exactly (see
that script's collect_unique_context_frames(), unchanged from ChildPlay's).

"test" videos are deliberately excluded from both train_ids and val_ids
returned by build_clip_split() -- reserved, unused for PoC-level
experimentation, same convention as every other dataset in this project.
"""

import os
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ucolaeo_temporal_utils import build_person_timelines

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "ucolaeo_temporal_manifest.jsonl")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "ucolaeo_temporal_feature_cache")
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
EMBEDDINGS_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features.npy")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

INCLUDE_PREDICTED_XY = os.environ.get("INCLUDE_PREDICTED_XY", "1") == "1"

PRIMARY_HORIZON_MS = 300  # matches VAT/ChildPlay's own PoC scope; also
                          # present in the window builder's own
                          # HORIZON_MS_LIST=[100,300,500], so this is a
                          # real available bucket, not an assumption.


# ══════════════════════════════════════════════════════════════════════
# ── VIDEO-LEVEL TRAIN/VAL SPLIT — from build_person_timelines()'s own
#    already-resolved split tracking, NOT a custom shuffle ─────────────
# ══════════════════════════════════════════════════════════════════════

def build_video_split():
    """Returns {video_id: split}, one entry per distinct video_id.
    Split is inherently per-video (every person in a given video shares
    the same split, since it's determined by which H5 file the video's
    rows came from) -- collapsing build_person_timelines()'s
    per-(video,person) tracking down to per-video is safe, not a
    simplifying assumption, because there is nothing to disagree: no
    video can appear in two different H5 splits."""
    timelines, _ = build_person_timelines()
    video_split = {}
    for (video_id, _person_id), t in timelines.items():
        video_split[video_id] = t["split"]
    return video_split


def build_clip_split(manifest_path=MANIFEST_PATH):
    """Kept as ChildPlay's own function name for import-compatibility
    with train_temporal_poc_ucolaeo.py's calling convention -- see
    module docstring's naming note. manifest_path kept as a parameter
    for signature-compatibility, unused here (same as ChildPlay's own
    version) -- the split comes from build_video_split(), not re-derived
    from the manifest. "test" videos deliberately excluded from both
    returned sets -- reserved, not used for PoC-level experimentation."""
    video_split = build_video_split()
    train_ids = {v for v, s in video_split.items() if s == "train"}
    val_ids = {v for v, s in video_split.items() if s == "val"}
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
        """Fail loud, fail early, per-video -- same discipline as
        ChildPlay's version. context_frame_paths are already full local
        paths, so no path-reconstruction is needed here, just an open."""
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
