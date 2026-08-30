"""
build_childplay_frame_manifest.py — Adapts ChildPlay's raw annotations into
frame_manifest.csv, the exact schema step40_teacher_pipeline_006_final.py and
step46_hybrid_teacher_pipeline_final.py both expect.

WHY NOT labels_childplay_gt_sorted.jsonl: verified directly against a real
clip (1Ab4vLMMAbY_2354-2439) that the flat GT file silently drops each clip's
first ~9 frames, inherited from the temporal window-builder's "needs 8 frames
of context before a frame is eligible as a target" requirement -- a
constraint that has nothing to do with teacher-labeling, which just needs a
valid per-frame annotation. Confirmed: frames 1-9 of that clip are all
genuinely valid targets (offscreen/object/object.../social...) when
classified fresh via classify_target(), zero skip reasons -- yet none of
them appear in the flat file. Building fresh from the raw CSVs avoids this
and any inherited clip-level quality-gate exclusion too.

REQUIRED MANIFEST COLUMNS (confirmed via direct inspection of step40/step46 --
BASE_PATH is dead code in both scripts, img_path is read directly from the
manifest row, fully resolved, so it's the single source of truth here):
  show, clip, fname, subject, head_x1, head_y1, head_x2, head_y2,
  gaze_x, gaze_y, gaze_type, use_teacher, img_path
Extra columns (split, is_child) are carried along for free -- harmless to
step40/step46 (pandas just ignores unused columns), useful for later
analysis (e.g. the is_child x gaze_type breakdown flagged as well-motivated
future work in childplay_temporal_poc_bible.md).

SHOW/CLIP/SUBJECT CONVENTION: matches labels_childplay_gt.jsonl's own
established mapping (show=video_id, clip=clip_id, subject=person_id) for
consistency with everything already built.

DOWNSAMPLED CLIPS: excluded, same reasoning as the temporal pipeline --
resolve_image_path() raises on these since the frame-step formula was never
verified for them (9 of 401 clips, ~2%).

img_path: built via childplay_annotation_utils.resolve_image_path()'s
already-confirmed formula, rooted at HPC_IMAGES_ROOT rather than local
CHILDPLAY_ROOT/images -- decouples "where this script reads annotations
from" (local, to build the manifest) from "where the actual images will be
read from at inference time" (HPC, once uploaded).
"""

import os
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import childplay_annotation_utils as cu

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

# Where THIS SCRIPT reads clips.csv / annotations from (local checkout or
# wherever it's run from -- must contain clips.csv and annotations/{split}/).
CHILDPLAY_ROOT = os.environ.get("CHILDPLAY_ROOT", cu.CHILDPLAY_ROOT)
CLIPS_CSV_PATH = os.environ.get(
    "CHILDPLAY_CLIPS_CSV", os.path.join(CHILDPLAY_ROOT, "clips.csv")
)

# Where the IMAGES will actually live at inference time -- the real HPC path,
# confirmed 2026-07-27. This is what gets baked into img_path, regardless of
# where this script itself is run from.
HPC_IMAGES_ROOT = os.environ.get(
    "CHILDPLAY_HPC_IMAGES_ROOT",
    "/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze/images",
)

OUTPUT_PATH = os.environ.get("CHILDPLAY_MANIFEST_OUT", "frame_manifest_childplay.csv")

MANIFEST_COLUMNS = [
    "show", "clip", "fname", "subject",
    "head_x1", "head_y1", "head_x2", "head_y2",
    "gaze_x", "gaze_y", "gaze_type", "use_teacher", "img_path",
    "split", "is_child",  # extra, harmless, useful later
]


def build_hpc_img_path(clip_id, video_id, absolute_frame):
    fname = f"{video_id}_{absolute_frame}.jpg"
    return os.path.join(HPC_IMAGES_ROOT, clip_id, fname), fname


def build_manifest():
    all_meta = cu.load_clip_metadata(CLIPS_CSV_PATH)
    print(f"Loaded {len(all_meta)} clips from {CLIPS_CSV_PATH}")

    downsampled = [cid for cid, m in all_meta.items() if m["downsampled"]]
    print(f"Excluding {len(downsampled)} downsampled clips (unconfirmed frame-step formula)")

    rows = []
    n_clips_ok, n_clips_missing = 0, 0
    skip_totals = {}
    gaze_type_totals = {"offscreen": 0, "social": 0, "object": 0}

    for clip_id, meta in sorted(all_meta.items()):
        if meta["downsampled"]:
            continue
        split = meta["split"]
        video_id, start_frame, end_frame = cu.parse_clip_start_frame(clip_id)

        ann_path = os.path.join(CHILDPLAY_ROOT, "annotations", split, f"{clip_id}.csv")
        if not os.path.isfile(ann_path):
            n_clips_missing += 1
            continue
        n_clips_ok += 1

        frames = cu.load_clip_annotations(clip_id, split, ann_root=os.path.join(CHILDPLAY_ROOT, "annotations"))

        for frame_num, people in sorted(frames.items()):
            head_boxes = {pid: ann["head_box"] for pid, ann in people.items()}
            absolute_frame = start_frame + frame_num - 1
            img_path, fname = build_hpc_img_path(clip_id, video_id, absolute_frame)

            for pid, ann in people.items():
                gtype, skip_reason = cu.classify_target(
                    ann["gaze_class"], ann["gaze"], pid, head_boxes
                )
                if gtype is None:
                    skip_totals[skip_reason] = skip_totals.get(skip_reason, 0) + 1
                    continue

                x1, y1, x2, y2 = ann["head_box"]
                if gtype == "offscreen":
                    gaze_x, gaze_y = -1, -1
                else:
                    gaze_x, gaze_y = round(ann["gaze"][0]), round(ann["gaze"][1])
                    gaze_type_totals[gtype] += 1
                if gtype == "offscreen":
                    gaze_type_totals["offscreen"] += 1

                rows.append({
                    "show": video_id,
                    "clip": clip_id,
                    "fname": fname,
                    "subject": pid,
                    "head_x1": round(x1), "head_y1": round(y1),
                    "head_x2": round(x2), "head_y2": round(y2),
                    "gaze_x": gaze_x, "gaze_y": gaze_y,
                    "gaze_type": gtype,
                    "use_teacher": (gtype != "offscreen"),
                    "img_path": img_path,
                    "split": split,
                    "is_child": ann["is_child"],
                })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nClips processed: {n_clips_ok} ok, {n_clips_missing} missing annotation file")
    print(f"Total manifest rows: {len(rows)}")
    print(f"gaze_type totals: {gaze_type_totals}")
    print(f"skip_reason totals (excluded from manifest): {skip_totals}")
    print(f"use_teacher=True (InternVL3/GDINO/MTGS targets): "
          f"{sum(1 for r in rows if r['use_teacher'])}")
    print(f"use_teacher=False (GT-direct, offscreen): "
          f"{sum(1 for r in rows if not r['use_teacher'])}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_manifest()
