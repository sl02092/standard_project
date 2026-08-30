"""
fix_mtgs_gaze_type.py — re-derives the correct gaze_type for an
already-completed mtgs_teacher_*.jsonl export, WITHOUT re-running MTGS
inference. Confirmed bug (2026-08-02): export_childplay_pseudolabels.py's
and export_vacation_pseudolabels.py's own process_frame() built
head_boxes_px as a SINGLE-ENTRY dict containing only the CURRENT
subject's own box, not every other subject present in the same frame --
so classify_gaze()'s own "is there another person's box at this point"
check always had zero real candidates, unconditionally returning
"object" (or "offscreen" if gt was unresolved), never "social".
Confirmed directly against real ChildPlay output: 189,049/189,049
"valid" rows labeled "object", zero "social".

WHAT THIS DOES NOT NEED TO FIX: pred_x/pred_y (the actual MTGS model
predictions) are computed independently of this classification step and
are NOT affected by the bug -- this is purely a post-hoc LABELING
error, not an error in the model's own output. No re-run needed; this
just re-derives the one field that was wrong, using the same static
manifest every other part of this project already trusts for real
per-frame subject boxes.

USES PIXEL-SPACE THROUGHOUT (gt_px_x/gt_px_y, already present in every
row, against the manifest's own pixel-space head boxes) -- deliberately
NOT comparing normalized gt_x/gt_y against pixel-space boxes, which
would silently produce nonsense from a unit mismatch.

USAGE: set MTGS_TEACHER_IN / MTGS_TEACHER_OUT / MANIFEST_PATH via env
vars for whichever dataset you're fixing (same script for ChildPlay and
VACATION, both share the identical bug and fix).
"""

import csv
import json
import os
from collections import defaultdict

MTGS_TEACHER_IN = os.environ.get("MTGS_TEACHER_IN", "mtgs_teacher_childplay.jsonl")
MTGS_TEACHER_OUT = os.environ.get("MTGS_TEACHER_OUT", "mtgs_teacher_childplay_gazetype_fixed.jsonl")
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "frame_manifest_childplay.csv")


def load_frame_boxes(manifest_path):
    """(show, fname) -> {subject: (x1,y1,x2,y2)} in PIXEL space, for
    EVERY subject present in that frame -- what head_boxes_px SHOULD
    have contained, built once here from the manifest rather than
    reconstructed per-row from a single subject's own data."""
    frame_boxes = defaultdict(dict)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["show"], row["fname"])
            frame_boxes[key][row["subject"]] = (
                float(row["head_x1"]), float(row["head_y1"]),
                float(row["head_x2"]), float(row["head_y2"]),
            )
    return dict(frame_boxes)


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_gaze(gaze_px, subject, head_boxes_px):
    if gaze_px is None or gaze_px == (-1, -1):
        return "offscreen"
    is_social = any(
        point_in_box(gaze_px, box)
        for other_subject, box in head_boxes_px.items()
        if other_subject != subject
    )
    return "social" if is_social else "object"


def main():
    frame_boxes = load_frame_boxes(MANIFEST_PATH)
    print(f"Loaded frame-level boxes for {len(frame_boxes)} frames.")

    n_total = 0
    n_changed = 0
    n_no_frame_match = 0
    old_counts = defaultdict(int)
    new_counts = defaultdict(int)

    with open(MTGS_TEACHER_IN, encoding="utf-8") as f_in, \
         open(MTGS_TEACHER_OUT, "w", encoding="utf-8") as f_out:
        for line in f_in:
            r = json.loads(line)
            n_total += 1
            old_counts[r.get("gaze_type")] += 1

            key = (r["show"], r["fname"])
            boxes = frame_boxes.get(key)
            if boxes is None:
                n_no_frame_match += 1
                new_counts[r.get("gaze_type")] += 1
                f_out.write(line if line.endswith("\n") else line + "\n")
                continue

            gt_px_x, gt_px_y = r.get("gt_px_x"), r.get("gt_px_y")
            gaze_px = None if gt_px_x is None else (gt_px_x, gt_px_y)
            new_gaze_type = classify_gaze(gaze_px, r["subject"], boxes)

            if new_gaze_type != r.get("gaze_type"):
                n_changed += 1
            new_counts[new_gaze_type] += 1
            r["gaze_type"] = new_gaze_type
            f_out.write(json.dumps(r) + "\n")

    print(f"\n{n_total} rows processed, {n_changed} gaze_type values changed, "
          f"{n_no_frame_match} rows had no manifest frame match (left unchanged).")
    print(f"Old gaze_type distribution: {dict(old_counts)}")
    print(f"New gaze_type distribution: {dict(new_counts)}")
    print(f"\nWritten to: {MTGS_TEACHER_OUT}")


if __name__ == "__main__":
    main()
