"""
build_vacation_vsgaze_h5.py — converts VACATION's already-verified
per-person temporal data (vacation_temporal_utils.py) into MTGS's own
VSGaze per-frame H5 schema (uco_laeo_{split}.h5 / childplay_{split}.h5
convention), so a new, minimal MTGS dataset class can read it the same
way VideoLAEODataset_temporal reads UCO-LAEO's.

WHY THIS IS A REGROUPING, NOT A RE-RESOLUTION: every coordinate here
comes from vacation_temporal_utils.build_person_timelines(), already
independently verified twice (hand-traced against real raw annotation
lines at NewAnt_11.txt frames 9/15/21/214, and cross-checked at
206,774-instance scale against the static pipeline -- exact match,
integer for integer). This script only reshapes that already-correct
data from "one row per (video, person), frames nested inside" into "one
row per (video, frame), all present people nested inside" -- the access
pattern MTGS's own dataset classes expect. It does one genuinely NEW
piece of work: normalizing to [0,1], since vacation_temporal_utils.py
deliberately stays in pixel space (see schema doc) and VSGaze's own H5
schema is normalized.

SCHEMA (matches README.md's VSGaze format, confirmed against real
UCO-LAEO/ChildPlay H5 consumption patterns):
  path: "images/{video_id}/{video_id}_{frame}.jpg" -- matches
    extract_frames_vacation.py's real naming convention exactly.
  person_ids: first entry ALWAYS -1 (the placeholder/padding slot every
    VSGaze H5 carries -- confirmed via UCO-LAEO's own "if pid == -1:
    continue" consumption), remaining entries are VACATION's own
    "Person1"/"Person2" labels with the prefix stripped and cast to int,
    matching every other VSGaze dataset's numeric person_id convention.
  head_bboxes: normalized [0,1], first row [0,0,0,0] (placeholder).
  gaze_points: normalized [0,1], first row [-1,-1] (never read, paired
    with person_ids[0]==-1).
  inout: 1 for a resolved target (social OR object -- VSGaze's inout
    field doesn't distinguish gaze TYPE, only whether a target exists at
    all), -1 for unresolved. NEVER 0 ("confirmed outside frame") --
    VACATION's annotation scheme has no such concept, only "resolved"
    vs "unknown" (build_vacation_frame_manifest.py's own "NO OFFSCREEN
    CONCEPT" note) -- writing 0 here would assert something not actually
    annotated.
  pairs/lah_pairs/laeo_pairs/coatt_pairs: present but empty per row --
    VACATION has no pair-level LAEO-style annotation at all, unlike
    ChildPlay. Included only so a dataset class that happens to read
    these columns won't hit a missing-column error; the new VACATION
    dataset class being built alongside this converter is not expected
    to need them.

REAL IMAGE DIMENSIONS, NOT ASSUMED: one frame per video_id opened via
PIL to get (w, h) for normalization -- same "measure, don't assume"
discipline as every other per-video dimension lookup in this project.
"""

import os
import re

import numpy as np
import pandas as pd
from PIL import Image

from vacation_temporal_utils import (
    build_person_timelines, load_official_split, get_video_split, IMAGES_ROOT,
)

OUTPUT_DIR = os.environ.get("VACATION_H5_OUT_DIR", ".")
IMAGE_PATH_PREFIX = "images"  # matches extract_frames_vacation.py's own convention

_dim_cache = {}


def get_video_dimensions(video_id, sample_frame_path):
    if video_id not in _dim_cache:
        try:
            with Image.open(sample_frame_path) as img:
                _dim_cache[video_id] = img.size
        except Exception as e:
            print(f"  [build_h5] could not read dimensions for {video_id} "
                  f"via {sample_frame_path}: {e}")
            _dim_cache[video_id] = None
    return _dim_cache[video_id]


def build_frame_rows():
    """Regroups build_person_timelines()'s per-person output into
    per-(video_id, frame_num) buckets -- pure reshaping of already-
    verified data, no new resolution logic."""
    timelines, _ = build_person_timelines()
    split_map = load_official_split()

    frame_data = {}  # (video_id, frame_num) -> list of (person_id, head_box_px, gaze_px_or_None)

    for (video_id, person_label), t in timelines.items():
        m = re.match(r"Person(\d+)$", person_label)
        if not m:
            raise ValueError(
                f"Unexpected person_label format: {person_label!r} for "
                f"video {video_id} -- expected 'Person<N>'. Stop and check "
                f"rather than silently coercing something unexpected."
            )
        person_id = int(m.group(1))

        for frame_num, entry in t["frames"].items():
            key = (video_id, frame_num)
            frame_data.setdefault(key, []).append(
                (person_id, entry["head_box"], entry["gaze"])
            )

    return frame_data, split_map


def build_rows_for_split(frame_data, split_map, target_split):
    rows = []
    n_frames_no_dims = 0

    for (video_id, frame_num), people in sorted(frame_data.items()):
        if get_video_split(video_id, split_map) != target_split:
            continue

        fname = f"{video_id}_{frame_num}.jpg"
        img_path = os.path.join(IMAGES_ROOT, video_id, fname)
        dims = get_video_dimensions(video_id, img_path)
        if dims is None:
            n_frames_no_dims += 1
            continue
        w, h = dims

        # Placeholder slot first (person_id=-1), matching every VSGaze H5's
        # own convention -- confirmed via UCO-LAEO's "if pid == -1: continue"
        # consumption pattern.
        person_ids = [-1]
        head_bboxes = [(0.0, 0.0, 0.0, 0.0)]
        gaze_points = [(-1.0, -1.0)]
        inout = [-1]

        for person_id, (x1, y1, x2, y2), gaze in people:
            person_ids.append(person_id)
            head_bboxes.append((x1 / w, y1 / h, x2 / w, y2 / h))
            if gaze is not None:
                gx, gy = gaze
                gaze_points.append((gx / w, gy / h))
                inout.append(1)
            else:
                gaze_points.append((-1.0, -1.0))
                inout.append(-1)

        rows.append({
            "path": os.path.join(IMAGE_PATH_PREFIX, video_id, fname).replace("\\", "/"),
            "person_ids": np.array(person_ids, dtype=np.int64),
            "head_bboxes": np.array(head_bboxes, dtype=np.float32),
            "gaze_points": np.array(gaze_points, dtype=np.float32),
            "inout": np.array(inout, dtype=np.float32),
            "pairs": np.zeros((0, 2), dtype=np.int64),
            "lah_pairs": np.zeros((0,), dtype=np.float32),
            "laeo_pairs": np.zeros((0,), dtype=np.float32),
            "coatt_pairs": np.zeros((0,), dtype=np.float32),
        })

    if n_frames_no_dims:
        print(f"  [{target_split}] {n_frames_no_dims} frames skipped -- "
              f"could not read image dimensions.")
    return rows


def main():
    print("Building per-person timelines and regrouping to per-frame rows...")
    frame_data, split_map = build_frame_rows()
    print(f"  {len(frame_data)} distinct (video, frame) combinations found.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ("train", "val", "test"):
        rows = build_rows_for_split(frame_data, split_map, split)
        if not rows:
            print(f"[{split}] 0 rows -- skipping file write.")
            continue
        df = pd.DataFrame(rows)
        out_path = os.path.join(OUTPUT_DIR, f"vacation_{split}.h5")
        df.to_hdf(out_path, key="data", mode="w")
        print(f"[{split}] {len(df)} frame-rows written to {out_path}")

    print("\nDone. Recommended before trusting this: read a row back with "
          "pd.read_hdf() and cross-check against an already hand-verified "
          "frame (e.g. video 11, frame 0 or frame 15) -- see the smoke "
          "test below.")


# ══════════════════════════════════════════════════════════════════════
# ── SMOKE TEST — run against real data before trusting the H5 files ────
# ══════════════════════════════════════════════════════════════════════

def validate_against_known_frame(split, video_id, frame_num, expected_person_id, expected_gaze_type):
    """Reads back a freshly-written H5 and checks one row against a
    frame already hand-verified earlier in this project (e.g. video 11
    frame 0: Person1 -> Object1 centroid (338.5, 272.5) in pixel space;
    frame 15: Person1/Person2 mutual social). Only confirms this
    converter's OWN correctness (regrouping + normalization) -- the
    underlying resolution was already verified independently before this
    script was written."""
    path = os.path.join(OUTPUT_DIR, f"vacation_{split}.h5")
    df = pd.read_hdf(path, "data")
    df = df.set_index("path")
    key = os.path.join(IMAGE_PATH_PREFIX, video_id, f"{video_id}_{frame_num}.jpg").replace("\\", "/")
    if key not in df.index:
        print(f"VALIDATION FAILED: {key} not found in {path}")
        return
    row = df.loc[key]
    idx = np.where(row["person_ids"] == expected_person_id)[0]
    if len(idx) == 0:
        print(f"VALIDATION FAILED: person {expected_person_id} not found at {key}")
        return
    gaze = row["gaze_points"][idx[0]]
    inout = row["inout"][idx[0]]
    print(f"Row for {key}, person {expected_person_id}:")
    print(f"  gaze (normalized) = {gaze}, inout = {inout}")
    print(f"  Manually multiply by this video's real (w, h) and compare "
          f"against the known pixel-space answer for this frame -- "
          f"expected gaze_type was {expected_gaze_type!r}.")


if __name__ == "__main__":
    main()
    #validate_against_known_frame('train', '211', 15, 1, 'social')
