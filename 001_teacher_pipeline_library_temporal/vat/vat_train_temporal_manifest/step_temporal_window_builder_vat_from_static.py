"""
step_temporal_window_builder_vat_from_static.py — builds a VAT temporal
manifest DIRECTLY from a static frame_manifest_vat_{SPLIT}.csv file,
rather than re-scanning raw annotations and guessing a quality filter.

WHY THIS IS DIFFERENT FROM step_temporal_window_builder_vat_test.py, AND
WHY IT'S MORE RELIABLE: that earlier script tried to reconstruct the
original temporal builder's own, undocumented per-subject quality filter
(MIN_FRAMES, MAX_UNRESOLVED_PCT) and clip-level rule (>=2 qualifying
subjects) from scratch, through several rounds of guess-and-check against
the real train-split manifest -- and never converged (227, then 341
clips, against a target of 214, each round revealing a new wrinkle).

This script sidesteps that problem entirely. frame_manifest_vat_{SPLIT}.csv
is the STATIC pipeline's own, already-validated frame selection --
confirmed directly against real data:
  - row counts match the known static population exactly (TRAIN: 103877,
    VAL: 3864 rows after the header)
  - frames per (show,clip,subject) are essentially 100% consecutive
    (103,139 of 103,141 measured gaps are exactly 1 -- confirmed directly
    against the real TRAIN file)
  - gaze_type (social/object/offscreen) is already correctly labelled,
    something the raw-annotation reconstruction could never determine
So every (show,clip,subject) in this file already represents a
genuinely-qualifying, densely-sampled timeline -- no separate quality
filter is needed, and none is applied here.

WHAT THIS PRODUCES IS A DIFFERENT POPULATION FROM THE ORIGINAL
vat_temporal_manifest.jsonl, NOT A REPRODUCTION OF IT. That's expected
and, for this purpose, fine: the goal is a genuinely test-split temporal
evaluation with a consistent, fully-traceable methodology -- not an exact
match to a population whose own construction logic was never available
to check. For a fair comparison against the already-trained heads' TRAIN-
split numbers, consider ALSO rebuilding train from frame_manifest_vat_
TRAIN.csv with this same script, so both sides of the comparison share
one consistent methodology rather than comparing an old, differently-built
train manifest against a new, differently-built test one.

NO >=2-SUBJECTS RULE: that rule in the earlier script was a proxy for
"could this clip possibly contain social gaze", needed because gaze_type
wasn't known. It's not needed here -- gaze_type is read directly from the
static manifest, so a single-subject clip with valid object-gaze targets
is correctly included rather than dropped.

USAGE:
    Set FRAME_MANIFEST_PATH to point at the static CSV for whichever
    split you're building (TRAIN, VAL, or TEST), then run.
"""

import os
import json
import csv
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

FRAME_MANIFEST_PATH = os.environ.get(
    "FRAME_MANIFEST_PATH", "frame_manifest_vat_TEST.csv"
)
SPLIT_NAME = os.environ.get("VAT_SPLIT_NAME", "test")

WINDOW_LENGTH = 8
STRIDE = 4
HORIZON_MS_LIST = [100, 300, 500]
VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}  # confirmed fixed
                                                      # mapping, from
                                                      # train_temporal_v1.py

OUTPUT_MANIFEST_JSONL = f"vat_{SPLIT_NAME}_from_static_temporal_manifest.jsonl"


# ══════════════════════════════════════════════════════════════════════
# ── LOADING — build per-(show,clip,subject) canonical timelines directly
# ── from the static frame manifest, no separate quality filter ─────────
# ══════════════════════════════════════════════════════════════════════

def load_static_timelines(path):
    """Returns {(show, clip, subject): sorted list of (frame_num, row_dict)}."""
    timelines = defaultdict(list)
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            key = (row["show"], row["clip"], row["subject"])
            frame_num = int(row["fname"].split(".")[0])
            timelines[key].append((frame_num, row))

    for key in timelines:
        timelines[key].sort(key=lambda x: x[0])

    print(f"  Rows read                          : {n_rows}")
    print(f"  Distinct (show,clip,subject) timelines: {len(timelines)}")
    print(f"  Distinct clips                       : {len({(k[0], k[1]) for k in timelines})}")
    return dict(timelines)


# ══════════════════════════════════════════════════════════════════════
# ── WINDOW CONSTRUCTION ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_windows(timelines):
    manifest_rows = []
    stats = defaultdict(int)

    for (show, clip, subject), timeline in sorted(timelines.items()):
        frame_nums = [fn for fn, _ in timeline]
        idx_to_row = {fn: r for fn, r in timeline}
        n = len(frame_nums)

        for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
            window_frame_nums = frame_nums[start:start + WINDOW_LENGTH]
            if len(window_frame_nums) < WINDOW_LENGTH:
                continue
            last_idx_pos = start + WINDOW_LENGTH - 1

            non_contiguous = (window_frame_nums[-1] - window_frame_nums[0]) != (WINDOW_LENGTH - 1)
            window_ok = not non_contiguous

            context_frame_paths, context_head_boxes = [], []
            if window_ok:
                for fn in window_frame_nums:
                    r = idx_to_row[fn]
                    context_frame_paths.append(r["fname"])
                    context_head_boxes.append([
                        int(r["head_x1"]), int(r["head_y1"]),
                        int(r["head_x2"]), int(r["head_y2"]),
                    ])

            for target_ms in HORIZON_MS_LIST:
                horizon_frames = VAT_HORIZON_FRAMES_MAP[target_ms]
                target_pos = last_idx_pos + horizon_frames
                row_valid = window_ok
                skip_reason = None
                target_frame_path = target_gaze_x = target_gaze_y = None
                target_gaze_type = None

                if not window_ok:
                    skip_reason = "non-contiguous timeline span (context window)"
                elif target_pos >= n:
                    row_valid = False
                    skip_reason = "target frame beyond available timeline"
                else:
                    target_frame_num = frame_nums[target_pos]
                    real_gap = target_frame_num - window_frame_nums[-1]
                    if real_gap != horizon_frames:
                        row_valid = False
                        skip_reason = "non-contiguous span (context to target)"
                    else:
                        target_row = idx_to_row[target_frame_num]
                        target_gaze_type = target_row["gaze_type"]
                        gx, gy = int(target_row["gaze_x"]), int(target_row["gaze_y"])
                        if gx == -1 and gy == -1:
                            row_valid = False
                            skip_reason = "unresolved (offscreen target)"
                        else:
                            target_frame_path = target_row["fname"]
                            target_gaze_x, target_gaze_y = gx, gy

                manifest_rows.append({
                    "source_dataset": "vat",
                    "clip_id": f"{show}/{clip}",
                    "subject_id": subject,
                    "context_frame_paths": context_frame_paths,
                    "context_head_boxes": context_head_boxes,
                    "target_frame_path": target_frame_path,
                    "target_gaze_x": target_gaze_x,
                    "target_gaze_y": target_gaze_y,
                    "target_gaze_type": target_gaze_type,
                    "target_horizon_ms": target_ms,
                    "horizon_frames": horizon_frames,
                    "window_valid": row_valid,
                    "skip_reason": skip_reason,
                })
                stats["valid" if row_valid else "invalid"] += 1
                if skip_reason:
                    stats[f"skip:{skip_reason}"] += 1

    with open(OUTPUT_MANIFEST_JSONL, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n  Total rows (windows x horizons) : {len(manifest_rows)}")
    print(f"  Valid                            : {stats['valid']}")
    print(f"  Invalid                          : {stats['invalid']}")
    print(f"\n  Invalid breakdown:")
    for reason, n in sorted(stats.items(), key=lambda x: -x[1]):
        if reason.startswith("skip:"):
            print(f"    {n:>6}  {reason[5:]}")
    print(f"\n  Manifest written to: {OUTPUT_MANIFEST_JSONL}")

    n_clips = len({(row["clip_id"]) for row in manifest_rows})
    n_windows = len({(row["clip_id"], row["subject_id"], tuple(row["context_frame_paths"]))
                      for row in manifest_rows if row["context_frame_paths"]})
    print(f"\n  Distinct clips in output   : {n_clips}")
    print(f"  Distinct windows in output : {n_windows}")

    return manifest_rows


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"FRAME_MANIFEST_PATH : {FRAME_MANIFEST_PATH}")
    print(f"SPLIT_NAME           : {SPLIT_NAME}\n")

    print("Loading static frame manifest and building timelines...")
    timelines = load_static_timelines(FRAME_MANIFEST_PATH)
    print()

    if not timelines:
        print("ERROR: no timelines found. Check FRAME_MANIFEST_PATH.")
        return

    print("Building temporal windows...")
    build_windows(timelines)

    print("\nDone. Next steps:")
    print("  1. Spot-check a handful of real windows by hand against the raw")
    print("     annotation files (or just against this same CSV), same discipline")
    print("     as every other dataset's window-builder in this project.")
    print("  2. This population will NOT exactly match the original")
    print("     vat_temporal_manifest.jsonl's own clip/row counts -- that's")
    print("     expected, not a bug (see the module docstring).")
    print("  3. Consider building TRAIN the same way too, so both sides of any")
    print("     train-vs-test comparison share one consistent methodology.")


if __name__ == "__main__":
    main()
