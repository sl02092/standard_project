"""
compute_last_known_position_baseline.py — Last-Known-Position Baseline
for the Temporal PoC (Objective 4)

WHY THIS EXISTS: gaze is temporally autocorrelated, so any half-decent
GRU/Transformer will show a plausible ADE just by leaning on that
autocorrelation -- the temporal analog of the static student's per-show
shortcut collapse. This computes the trivial "did nothing" baseline
(predict the subject's own gaze at the LAST context frame, unchanged, as
the answer at every horizon). Per the agreed plan: clearing this baseline
by a real margin is the PRIMARY pass/fail criterion for the whole PoC --
more important than which architecture (GRU vs Transformer) wins.

WHY THIS NEEDS A SEPARATE ANNOTATION LOOKUP, NOT JUST THE MANIFEST:
vat_temporal_manifest.jsonl's context_head_boxes gives each context
frame's HEAD BOX (for cropping), never the subject's own historical GAZE
POINT -- confirmed by direct reading of step_temporal_window_builder_vat.py
(context_head_boxes comes from ann[fn]["head"], never ann[fn]["gaze"]).
Rather than modify and rerun the already-validated manifest, this does a
lightweight join back to the raw annotation files for only the specific
(show, clip, subject, last_context_fname) tuples actually needed --
reusing load_annotations() verbatim from the window-builder script.

UNIT MISMATCH — FLAGGED, NOW RESOLVED WITH EVIDENCE (not a fixed-reference
guess anymore): target_gaze_x/y in the manifest are RAW PIXEL coordinates
(load_annotations does a plain int() parse, no normalization step
anywhere in the window-builder script). ade_distance_by_gaze_type_comparison.py's
docstring describes a real bug where off-screen rows got mis-normalized
as "-1/width" instead of a clean sentinel -- direct evidence the project's
actual convention divides by each frame's own ACTUAL pixel width/height,
not a fixed reference resolution. The original REF_W/REF_H=1280x720
version of this script produced 0.0490 at horizon_frames=7, well under
the project's usual 0.15-0.31 ADE range -- consistent with under-
normalization from real VAT frames being smaller than the guessed fixed
reference. Fixed here: each clip's actual (width, height) is read from
one representative frame per clip (cached -- ~214 opens total, not one
per scored row) and used to normalize that clip's rows. This assumes
resolution is constant within a clip (same video, a few frames apart --
essentially certain) but not necessarily across clips/shows -- which is
exactly the kind of per-show variation the "-1/width" bug implies exists.
Reports BOTH:
  - baseline_ADE_px  : raw pixel-distance ADE, no assumption, always correct
  - baseline_ADE_norm: per-clip actual-dimension normalized ADE -- should
    now be directly comparable to the project's other ADE numbers. Still
    worth a final sanity glance against the 0.15-0.31 range before fully
    trusting "beats baseline by X%" claims built on it.

OFF-SCREEN HANDLING: window_valid=True does NOT guarantee target_gaze_x/y
is real -- a subject can be validly annotated at the target frame while
genuinely looking off-screen (target_gaze_type="offscreen", gaze=None).
Excluded here the same way every other ADE number in this project
excludes off-screen frames. Rows where the LAST CONTEXT frame's own gaze
is off-screen are also excluded -- no position to carry forward.

SCOPE: reports all 3 horizons for context, even though the PoC itself is
scoped to 300ms/7-frame only -- cheap to compute, and whether the
baseline degrades with horizon the way a genuine anticipation task should
is itself informative.
"""

import os
import csv
import json
import sys
from collections import defaultdict

from PIL import Image

# Reuses the EXACT same deterministic split the model training script uses
# -- not reimplementing this logic a second time, which risks the two
# silently drifting apart (same discipline as reusing load_annotations
# verbatim rather than re-deriving it).
try:
    from temporal_window_dataset import build_clip_split
except ImportError:
    build_clip_split = None
    print("[baseline] temporal_window_dataset.py not importable from this "
          "directory -- val-split-restricted breakdown will be skipped. "
          "Full-manifest breakdown still runs.")

#MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "vat_temporal_manifest.jsonl")
MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", r"C:\repo\standard_project\vat_missing_annotation_investigation\vat_temporal_manifest.jsonl")
VAT_ROOT = os.environ.get(
    "VAT_ROOT",
    r"C:\Users\scott\Desktop\dissertation\0_dataset\videoattentiontarget",
)
# Manifest was built from TRAIN_ANN_DIR only (see window-builder script) --
# matching that here, not guessing a different split.
TRAIN_ANN_DIR = os.path.join(VAT_ROOT, "annotations", "train")
# No train/val/test level under images -- confirmed in an earlier session
# against a real MTGS-work path. Same convention as the feature-extraction
# script.
VAT_IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", os.path.join(VAT_ROOT, "images"))


# ── copied verbatim from step_temporal_window_builder_vat.py ───────────
def load_annotations(ann_path):
    frames = []
    with open(ann_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            fname = row[0].strip()
            try:
                x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                gx, gy = int(row[5]), int(row[6])
            except ValueError:
                continue
            gaze = (gx, gy) if gx != -1 and gy != -1 else None
            frames.append({"fname": fname, "head": (x1, y1, x2, y2), "gaze": gaze})
    return frames


_ann_cache = {}

def get_subject_gaze(show, clip, subject, fname):
    """Returns (gx, gy) or None. Caches one subject's full annotation file
    per (show, clip, subject) so repeated lookups (same subject's frame
    reused across overlapping windows) don't re-parse the .txt file."""
    key = (show, clip, subject)
    if key not in _ann_cache:
        ann_path = os.path.join(TRAIN_ANN_DIR, show, clip, f"{subject}.txt")
        if not os.path.isfile(ann_path):
            _ann_cache[key] = {}
        else:
            frames = load_annotations(ann_path)
            _ann_cache[key] = {row["fname"]: row["gaze"] for row in frames}
    return _ann_cache[key].get(fname)


_dim_cache = {}

def get_clip_dimensions(show, clip, sample_fname):
    """Returns (width, height) or None. One PIL open per (show, clip),
    cached -- assumes constant resolution within a clip (same video, a
    few frames apart), not across clips. See module docstring for why a
    fixed reference resolution was replaced with this."""
    key = (show, clip)
    if key not in _dim_cache:
        path = os.path.join(VAT_IMAGES_BASE_PATH, show, clip, sample_fname)
        try:
            with Image.open(path) as img:
                _dim_cache[key] = img.size  # (w, h)
        except Exception as e:
            print(f"[baseline] Could not read dimensions for {show}/{clip} "
                  f"via {sample_fname}: {e}")
            _dim_cache[key] = None
    return _dim_cache[key]


def main():
    val_clip_ids = None
    if build_clip_split is not None:
        _, val_clip_ids = build_clip_split(MANIFEST_PATH)
        print(f"[baseline] Loaded val split: {len(val_clip_ids)} clips "
              f"(same seed=42 split the training script will use).\n")

    stats = defaultdict(int)
    dists_by_horizon = defaultdict(list)
    val_dists_by_horizon = defaultdict(list)

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

            show, clip = row["clip_id"].split("/", 1)
            subject = row["subject_id"]
            last_context_fname = row["context_frame_paths"][-1]

            last_known_gaze = get_subject_gaze(show, clip, subject, last_context_fname)
            if last_known_gaze is None:
                stats["skipped_offscreen_last_context"] += 1
                continue

            dims = get_clip_dimensions(show, clip, last_context_fname)
            if dims is None:
                stats["skipped_no_dimensions"] += 1
                continue
            clip_w, clip_h = dims

            lkx, lky = last_known_gaze
            tx, ty = row["target_gaze_x"], row["target_gaze_y"]
            dist_px = ((lkx - tx) ** 2 + (lky - ty) ** 2) ** 0.5
            dist_norm = (((lkx - tx) / clip_w) ** 2 + ((lky - ty) / clip_h) ** 2) ** 0.5
            dists_by_horizon[row["horizon_frames"]].append((dist_px, dist_norm))
            stats["scored_rows"] += 1

            if val_clip_ids is not None and row["clip_id"] in val_clip_ids:
                val_dists_by_horizon[row["horizon_frames"]].append((dist_px, dist_norm))

    print("=" * 70)
    print("LAST-KNOWN-POSITION BASELINE — FULL MANIFEST")
    print("=" * 70)
    print(f"  Total manifest rows            : {stats['total_rows']}")
    print(f"  Skipped (window_valid=False)   : {stats['skipped_invalid_window']}")
    print(f"  Skipped (off-screen target)    : {stats['skipped_offscreen_target']}")
    print(f"  Skipped (off-screen last-ctx)  : {stats['skipped_offscreen_last_context']}")
    print(f"  Skipped (no image dimensions)  : {stats['skipped_no_dimensions']}")
    print(f"  Scored rows                    : {stats['scored_rows']}")
    print()
    print(f"  {'horizon_frames':<16}{'n':<10}{'baseline_ADE_px':<18}{'baseline_ADE_norm':<20}")
    for hf in sorted(dists_by_horizon):
        d = dists_by_horizon[hf]
        ade_px = sum(x[0] for x in d) / len(d)
        ade_norm = sum(x[1] for x in d) / len(d)
        print(f"  {hf:<16}{len(d):<10}{ade_px:<18.2f}{ade_norm:<20.4f}")

    if val_clip_ids is not None:
        print()
        print("=" * 70)
        print("LAST-KNOWN-POSITION BASELINE — VAL-SPLIT ONLY (matched to model eval)")
        print("=" * 70)
        print("  Restricted to the same clip-level val split "
              "temporal_window_dataset.py uses. Note this is a further-")
        print("  restricted subset of the model's val rows: it also requires the ")
        print("  SUBJECT'S OWN last-context gaze to be on-screen (the baseline has")
        print("  no position to carry forward otherwise), which the model's own")
        print("  eval population does not require. THIS is the fair, apples-to-")
        print("  apples number to compare a trained model's matched-subset ADE")
        print("  against -- not the full-manifest number above.")
        print()
        print(f"  {'horizon_frames':<16}{'n':<10}{'baseline_ADE_px':<18}{'baseline_ADE_norm':<20}")
        for hf in sorted(val_dists_by_horizon):
            d = val_dists_by_horizon[hf]
            ade_px = sum(x[0] for x in d) / len(d)
            ade_norm = sum(x[1] for x in d) / len(d)
            print(f"  {hf:<16}{len(d):<10}{ade_px:<18.2f}{ade_norm:<20.4f}")

    print()
    print("  baseline_ADE_norm uses each CLIP's ACTUAL pixel dimensions (not a")
    print("  fixed reference). Any trained model must beat the VAL-SPLIT ONLY")
    print("  baseline_ADE_norm at horizon_frames=7, on the matched subset, by a")
    print("  real margin to be considered non-trivial.")


if __name__ == "__main__":
    main()
