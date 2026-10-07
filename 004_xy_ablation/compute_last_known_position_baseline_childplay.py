"""
compute_last_known_position_baseline_childplay.py — ChildPlay port of
compute_last_known_position_baseline.py (VAT).

WHY THIS EXISTS: same reasoning as the VAT version — gaze is temporally
autocorrelated, so any half-decent GRU/Transformer will show a plausible
ADE just by leaning on that autocorrelation. This computes the trivial
"did nothing" baseline (predict the subject's own TRUE gaze at the last
context frame, unchanged, as the answer at every horizon) — the PRIMARY
pass/fail criterion any trained model needs to clear, more important than
which architecture wins.

WHAT'S SIMPLER HERE THAN THE VAT VERSION: no annotation parser needed —
reuses childplay_annotation_utils.py's load_clip_annotations,
load_clip_metadata, and parse_clip_start_frame directly rather than
re-implementing CSV parsing (which the VAT version had to do itself,
copying load_annotations() from the window-builder). Same for image
paths — context_frame_paths are already fully resolvable, no
VAT_IMAGES_BASE_PATH-style reconstruction needed.

THREE REAL STRUCTURAL DIFFERENCES FROM VAT, NOT JUST RENAMES:
1. SPLIT SOURCE: VAT's baseline imported build_clip_split() — a custom,
   seeded clip-shuffle the training script also uses. ChildPlay does NOT
   use a custom split; per the schema doc, it uses its own official
   train/val/test split (clips.csv's "split" column, surfaced via
   childplay_annotation_utils.load_clip_metadata()). The "matched to
   model eval" table below is therefore restricted to clip_id's whose
   OWN split is "val", not a locally-computed shuffle — the eventual
   ChildPlay dataset-loader port needs the same adaptation, not
   build_clip_split().
2. GROUPING KEY: VAT grouped by horizon_frames directly, since it was one
   of three fixed global constants (2/7/12) for the whole manifest.
   ChildPlay's horizon_frames is computed PER CLIP from that clip's real
   fps (schema doc's three-field horizon design) — the same
   horizon_frames value can mean different real elapsed time on different
   clips. Grouping here is by target_horizon_ms (the fixed 100/300/500
   bucket), which is the value that's actually comparable across clips of
   different fps.
3. FRAME-NUMBER CONVERSION: context_frame_paths' filenames encode the
   ABSOLUTE frame number in the original video (resolve_image_path's
   frame_start + relative - 1 formula), but load_clip_annotations()
   returns a dict keyed by the CSV's own RELATIVE frame number. Cached
   per clip: convert once at cache-build time (relative -> absolute,
   using parse_clip_start_frame), not on every lookup.

NORMALIZATION: per-clip actual pixel dimensions, read directly via PIL —
the CORRECT convention from the start here (no fixed-reference mistake to
discover and fix, unlike the VAT version's original REF_W/REF_H=1280x720
guess that under-normalized real results). ChildPlay head-box/gaze values
were already confirmed pixel-space, not normalized, during earlier
validation (childplay_annotation_utils.py's validate_against_real_clip
output).

OFF-SCREEN HANDLING: identical policy to the VAT version. window_valid=True
does not guarantee target_gaze_x/y is real (a subject can be validly
annotated while genuinely looking off-screen — target_gaze_type=
"offscreen", gaze=None). Excluded here the same way every other ADE
number in this project excludes off-screen frames. Rows where the LAST
CONTEXT frame's own gaze is off-screen are also excluded — no position to
carry forward.
"""

import os
import json
from collections import defaultdict

from PIL import Image

from childplay_annotation_utils import (
    load_clip_metadata, load_clip_annotations, parse_clip_start_frame,
)

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "childplay_temporal_manifest.jsonl")

_clip_split_map = None
_ann_cache = {}   # clip_id -> {absolute_frame: {subject_id: (gx,gy) or None}}
_dim_cache = {}    # clip_id -> (w, h) or None


def get_clip_split_map():
    """clip_id -> official split ("train"/"val"/"test"), from clips.csv
    via childplay_annotation_utils.load_clip_metadata() — NOT a custom
    shuffle. See module docstring point 1."""
    global _clip_split_map
    if _clip_split_map is None:
        _clip_split_map = {cid: meta["split"] for cid, meta in load_clip_metadata().items()}
    return _clip_split_map


def parse_frame_number(path):
    """Extract the ABSOLUTE frame number from a full image path's
    filename ('{video_id}_{frame}.jpg') — same convention as
    verify_temporal_contiguity_childplay.py's parse_frame_idx."""
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]
    return int(stem.rsplit("_", 1)[1])


def get_clip_annotations(clip_id):
    """Returns {absolute_frame: {subject_id: (gx,gy) or None}}, cached
    per clip. Converts load_clip_annotations()'s RELATIVE frame keys to
    ABSOLUTE ones once here (matching the numbering image filenames
    actually use — see module docstring point 3), not on every lookup."""
    if clip_id not in _ann_cache:
        split = get_clip_split_map().get(clip_id)
        if split is None:
            print(f"[baseline] {clip_id} not found in clips.csv -- skipping.")
            _ann_cache[clip_id] = {}
            return _ann_cache[clip_id]
        _, clip_start, _ = parse_clip_start_frame(clip_id)
        raw = load_clip_annotations(clip_id, split)  # {relative_frame: {subject: {...}}}
        remapped = {}
        for rel_frame, people in raw.items():
            abs_frame = clip_start + rel_frame - 1
            remapped[abs_frame] = {pid: ann["gaze"] for pid, ann in people.items()}
        _ann_cache[clip_id] = remapped
    return _ann_cache[clip_id]


def get_subject_gaze(clip_id, subject, abs_frame):
    return get_clip_annotations(clip_id).get(abs_frame, {}).get(subject)


def get_clip_dimensions(sample_image_path):
    """Returns (width, height) or None. One PIL open per clip, cached --
    assumes constant resolution within a clip (same video, a few frames
    apart), not across clips."""
    clip_dir = os.path.dirname(sample_image_path)
    if clip_dir not in _dim_cache:
        try:
            with Image.open(sample_image_path) as img:
                _dim_cache[clip_dir] = img.size
        except Exception as e:
            print(f"[baseline] Could not read dimensions via {sample_image_path}: {e}")
            _dim_cache[clip_dir] = None
    return _dim_cache[clip_dir]


def main():
    split_map = get_clip_split_map()
    n_val_clips = sum(1 for s in split_map.values() if s == "val")
    print(f"[baseline] {n_val_clips} clips in ChildPlay's own official val split.\n")

    stats = defaultdict(int)
    dists_by_horizon_ms = defaultdict(list)
    val_dists_by_horizon_ms = defaultdict(list)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stats["total_rows"] += 1

            if not row["window_valid"]:
                stats["skipped_invalid_window"] += 1
                continue
            if row["target_gaze_x"] is None:
                stats["skipped_offscreen_target"] += 1
                continue

            clip_id = row["clip_id"]
            subject = row["subject_id"]
            last_context_path = row["context_frame_paths"][-1]
            last_abs_frame = parse_frame_number(last_context_path)

            last_known_gaze = get_subject_gaze(clip_id, subject, last_abs_frame)
            if last_known_gaze is None:
                stats["skipped_offscreen_last_context"] += 1
                continue

            dims = get_clip_dimensions(last_context_path)
            if dims is None:
                stats["skipped_no_dimensions"] += 1
                continue
            clip_w, clip_h = dims

            lkx, lky = last_known_gaze
            tx, ty = row["target_gaze_x"], row["target_gaze_y"]
            dist_px = ((lkx - tx) ** 2 + (lky - ty) ** 2) ** 0.5
            dist_norm = (((lkx - tx) / clip_w) ** 2 + ((lky - ty) / clip_h) ** 2) ** 0.5

            target_horizon_ms = row["target_horizon_ms"]
            dists_by_horizon_ms[target_horizon_ms].append((dist_px, dist_norm))
            stats["scored_rows"] += 1

            if split_map.get(clip_id) == "val":
                val_dists_by_horizon_ms[target_horizon_ms].append((dist_px, dist_norm))

    print("=" * 70)
    print("LAST-KNOWN-POSITION BASELINE -- CHILDPLAY -- FULL MANIFEST")
    print("=" * 70)
    print(f"  Total manifest rows            : {stats['total_rows']}")
    print(f"  Skipped (window_valid=False)   : {stats['skipped_invalid_window']}")
    print(f"  Skipped (off-screen target)    : {stats['skipped_offscreen_target']}")
    print(f"  Skipped (off-screen last-ctx)  : {stats['skipped_offscreen_last_context']}")
    print(f"  Skipped (no image dimensions)  : {stats['skipped_no_dimensions']}")
    print(f"  Scored rows                    : {stats['scored_rows']}")
    print()
    print(f"  {'target_horizon_ms':<20}{'n':<10}{'baseline_ADE_px':<18}{'baseline_ADE_norm':<20}")
    for h in sorted(dists_by_horizon_ms):
        d = dists_by_horizon_ms[h]
        ade_px = sum(x[0] for x in d) / len(d)
        ade_norm = sum(x[1] for x in d) / len(d)
        print(f"  {h:<20}{len(d):<10}{ade_px:<18.2f}{ade_norm:<20.4f}")

    print()
    print("=" * 70)
    print("LAST-KNOWN-POSITION BASELINE -- CHILDPLAY -- VAL-SPLIT ONLY")
    print("(ChildPlay's own official val split, not a custom shuffle)")
    print("=" * 70)
    print("  Restricted to clips whose OWN split (clips.csv) is 'val'. Also a")
    print("  further-restricted subset: requires the SUBJECT'S OWN last-context")
    print("  gaze to be on-screen (the baseline has no position to carry forward")
    print("  otherwise), which a trained model's own eval population does not")
    print("  require. THIS is the fair, apples-to-apples number to compare a")
    print("  trained model's matched-subset ADE against.")
    print()
    print(f"  {'target_horizon_ms':<20}{'n':<10}{'baseline_ADE_px':<18}{'baseline_ADE_norm':<20}")
    for h in sorted(val_dists_by_horizon_ms):
        d = val_dists_by_horizon_ms[h]
        ade_px = sum(x[0] for x in d) / len(d)
        ade_norm = sum(x[1] for x in d) / len(d)
        print(f"  {h:<20}{len(d):<10}{ade_px:<18.2f}{ade_norm:<20.4f}")

    print()
    print("  baseline_ADE_norm uses each CLIP's ACTUAL pixel dimensions. Any")
    print("  trained model must beat the VAL-SPLIT ONLY baseline_ADE_norm at")
    print("  target_horizon_ms=300, on the matched subset, by a real margin to")
    print("  be considered non-trivial (matching the VAT PoC's own scope).")


if __name__ == "__main__":
    main()
