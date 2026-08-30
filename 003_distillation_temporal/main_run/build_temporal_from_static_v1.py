"""
build_temporal_from_static_v1.py — generalized temporal window builder,
sourced directly from each dataset's own static frame_manifest_{DATASET}_
{SPLIT}.csv, for VAT, ChildPlay, and VACATION. One script, not three
near-duplicates -- same reasoning as verify_temporal_contiguity_v1.py's
own generalization.

WHY THIS APPROACH, FOR ALL THREE, NOT JUST VAT: the original per-dataset
temporal builders (childplay_annotation_utils.py, vacation_temporal_
utils.py, and VAT's own raw-annotation reconstruction) all re-parse raw
annotations and apply their own per-subject quality filter to decide
which subjects/frames qualify. For VAT this proved impossible to
reconstruct reliably (five rounds of guess-and-check against the known
real manifest, never converged). ChildPlay and VACATION's SPLIT
assignment already matches static (confirmed: both read the same
clips.csv / readme.txt the static pipeline uses) -- but their per-subject
QUALIFICATION logic is independent of static's own, already-validated
frame_manifest, so the same risk applies even though it hasn't been
caught the same way yet. Building directly from the static manifest
sidesteps this for all three: every row in frame_manifest_{DATASET}_
{SPLIT}.csv already represents a validated selection, confirmed against
real population counts, so no separate quality filter is needed or
applied here.

CONFIRMED DIRECTLY, BEFORE TRUSTING THIS:
  - VAT: frames per (show,clip,subject) are ~100% consecutive (103,139 of
    103,141 measured gaps == 1, checked against the real TRAIN file).
  - gaze_type (social/object/offscreen) is read directly from the static
    manifest for all three datasets -- not inferred, not "unknown".
  - VAT's horizon_frames uses a FIXED map ({100:2, 300:7, 500:12},
    confirmed from train_temporal_v1.py's own VAT_HORIZON_FRAMES_MAP).
    ChildPlay and VACATION use PER-CLIP horizon_frames = round(target_ms
    / 1000 * clip_fps), matching the "three-field horizon design"
    already established for those two datasets (target_horizon_ms,
    horizon_frames, horizon_ms_actual) -- NOT yet independently
    re-confirmed against real per-clip data the way VAT's fixed map was;
    worth a spot-check once this runs.

WHAT THIS DOES NOT YET DO: contiguity across VAT/ChildPlay/VACATION was
only independently, empirically verified for VAT before this script was
written. Run the contiguity check (this script logs gap counts itself)
against ChildPlay/VACATION output before trusting it at scale, same
discipline as everywhere else in this project.

USAGE:
    python build_temporal_from_static_v1.py <vat|childplay|vacation> <split> <frame_manifest.csv> [clips_csv_for_fps]

    clips_csv_for_fps is REQUIRED for childplay (its own clips.csv, for
    per-clip fps) and IGNORED for vat. For vacation, pass the fps source
    file (whatever format it's in) -- NOT YET IMPLEMENTED, see
    load_vacation_fps() below; vacation runs will fail loudly until a
    real fps source is wired in.
"""

import os
import sys
import csv
import json
from collections import defaultdict

WINDOW_LENGTH = 8
STRIDE = 4
HORIZON_MS_LIST = [100, 300, 500]
VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}  # fixed, confirmed


# ══════════════════════════════════════════════════════════════════════
# ── DATASET-SPECIFIC CONFIG ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def get_context_frame_path(dataset, row):
    """Returns what belongs in context_frame_paths / target_frame_path
    for this dataset's rows. VAT's downstream consumers (extract_
    temporal_features_v1.py's resolve_image_path(), temporal_window_
    dataset_vat.py) expect BARE filenames, reconstructing the full path
    themselves via show/clip/fname. ChildPlay and VACATION's downstream
    consumers expect ALREADY-FULL, directly-openable paths -- confirmed
    directly against extract_temporal_features_v1.py's own explicit,
    documented per-dataset assumption ("ChildPlay/VACATION/UCO-LAEO's
    context_frame_paths are already FULL, directly-openable paths").
    Static manifests' own "fname" column is bare for all three datasets;
    the full path lives in the separate "img_path" column for ChildPlay/
    VACATION -- read from there for those two, not from "fname"."""
    if dataset == "vat":
        return row["fname"]
    return row["img_path"]


def get_clip_id(dataset, row):
    """Returns the clip_id string this dataset's downstream scripts
    (temporal_window_dataset_{dataset}.py, compute_last_known_position_
    baseline_{dataset}.py) already expect."""
    if dataset == "vat":
        return f"{row['show']}/{row['clip']}"          # confirmed: VAT's
                                                          # hierarchical
                                                          # "show/clip" form
    return row["clip"]                                   # childplay,
                                                          # vacation: flat,
                                                          # confirmed


def parse_frame_num(dataset, fname):
    """Extracts the numeric frame index from fname, per each dataset's
    own confirmed convention."""
    stem = os.path.splitext(fname)[0]
    if dataset == "vat":
        if not stem.isdigit():
            raise ValueError(f"VAT fname '{fname}' doesn't parse as a "
                              f"zero-padded frame index.")
        return int(stem)
    # childplay, vacation: "{video_id}_{frame}.jpg" -- frame is the part
    # after the LAST underscore (video_id itself may contain underscores).
    if "_" not in stem:
        raise ValueError(f"'{fname}' -- stem '{stem}' has no underscore "
                          f"to split off a frame number.")
    frame_str = stem.rsplit("_", 1)[1]
    if not frame_str.isdigit():
        raise ValueError(f"'{fname}' -- '{frame_str}' doesn't parse as a "
                          f"frame number.")
    return int(frame_str)


def load_childplay_fps(clips_csv_path):
    """Returns {clip_id: fps}, from ChildPlay's own clips.csv -- the
    exact source childplay_annotation_utils.load_clip_metadata() already
    reads for this purpose."""
    fps_map = {}
    with open(clips_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fps_map[row["clip"]] = float(row["fps"])
    return fps_map


def load_vacation_fps(fps_source_path):
    """Returns {video_id: fps} -- reused, not reinvented, matching
    vacation_temporal_utils.load_fps_map()'s own regex exactly against
    the real video_fps.txt format ('{video_id}.mp4: {fps} FPS' per
    line), confirmed directly against the real file."""
    import re
    fps_map = {}
    with open(fps_source_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(\d+)\.mp4:\s*([\d.]+)\s*FPS", line)
            if not m:
                raise ValueError(f"Could not parse fps line: {line!r}")
            fps_map[m.group(1)] = float(m.group(2))
    return fps_map


def get_horizon_frames(dataset, target_ms, clip_id=None, fps_map=None):
    if dataset == "vat":
        return VAT_HORIZON_FRAMES_MAP[target_ms]
    # childplay, vacation: per-clip, fps-derived
    fps = fps_map[clip_id]
    return round(target_ms / 1000 * fps)


# ══════════════════════════════════════════════════════════════════════
# ── LOADING ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_static_timelines(dataset, frame_manifest_path):
    """Returns {(clip_id, subject): sorted [(frame_num, row_dict), ...]}."""
    timelines = defaultdict(list)
    n_rows = 0
    with open(frame_manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            clip_id = get_clip_id(dataset, row)
            subject = row["subject"]
            frame_num = parse_frame_num(dataset, row["fname"])
            timelines[(clip_id, subject)].append((frame_num, row))

    for key in timelines:
        timelines[key].sort(key=lambda x: x[0])

    print(f"  Rows read                              : {n_rows}")
    print(f"  Distinct (clip,subject) timelines      : {len(timelines)}")
    print(f"  Distinct clips                         : {len({k[0] for k in timelines})}")
    return dict(timelines)


# ══════════════════════════════════════════════════════════════════════
# ── WINDOW CONSTRUCTION ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_windows(dataset, timelines, fps_map=None):
    manifest_rows = []
    stats = defaultdict(int)
    gap_violations = 0

    for (clip_id, subject), timeline in sorted(timelines.items()):
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
            if non_contiguous:
                gap_violations += 1

            context_frame_paths, context_head_boxes = [], []
            if window_ok:
                for fn in window_frame_nums:
                    r = idx_to_row[fn]
                    context_frame_paths.append(get_context_frame_path(dataset, r))
                    context_head_boxes.append([
                        int(float(r["head_x1"])), int(float(r["head_y1"])),
                        int(float(r["head_x2"])), int(float(r["head_y2"])),
                    ])

            for target_ms in HORIZON_MS_LIST:
                horizon_frames = get_horizon_frames(dataset, target_ms, clip_id, fps_map)
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
                        gx, gy = int(float(target_row["gaze_x"])), int(float(target_row["gaze_y"]))
                        if gx == -1 and gy == -1:
                            row_valid = False
                            skip_reason = "unresolved (offscreen target)"
                        else:
                            target_frame_path = get_context_frame_path(dataset, target_row)
                            target_gaze_x, target_gaze_y = gx, gy

                manifest_rows.append({
                    "source_dataset": dataset,
                    "clip_id": clip_id,
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

    print(f"\n  Total rows (windows x horizons) : {len(manifest_rows)}")
    print(f"  Valid                            : {stats['valid']}")
    print(f"  Invalid                          : {stats['invalid']}")
    print(f"  Context-window gap violations    : {gap_violations}")
    print(f"\n  Invalid breakdown:")
    for reason, n in sorted(stats.items(), key=lambda x: -x[1]):
        if reason.startswith("skip:"):
            print(f"    {n:>6}  {reason[5:]}")

    n_clips = len({row["clip_id"] for row in manifest_rows})
    print(f"\n  Distinct clips in output   : {n_clips}")

    return manifest_rows


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 4:
        print("Usage: python build_temporal_from_static_v1.py <vat|childplay|vacation> "
              "<split> <frame_manifest.csv> [clips_csv_for_fps]")
        sys.exit(1)

    dataset, split, frame_manifest_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if dataset not in ("vat", "childplay", "vacation"):
        print("dataset must be one of: vat, childplay, vacation")
        sys.exit(1)

    fps_map = None
    if dataset == "childplay":
        if len(sys.argv) < 5:
            print("childplay requires clips_csv_for_fps as the 4th argument.")
            sys.exit(1)
        fps_map = load_childplay_fps(sys.argv[4])
        print(f"Loaded fps for {len(fps_map)} ChildPlay clips.\n")
    elif dataset == "vacation":
        if len(sys.argv) < 5:
            print("vacation requires an fps source as the 4th argument (see module docstring).")
            sys.exit(1)
        fps_map = load_vacation_fps(sys.argv[4])  # will raise until implemented

    output_path = f"{dataset}_{split}_from_static_temporal_manifest.jsonl"

    print(f"Dataset            : {dataset}")
    print(f"Split               : {split}")
    print(f"Frame manifest      : {frame_manifest_path}\n")

    print("Loading static frame manifest and building timelines...")
    timelines = load_static_timelines(dataset, frame_manifest_path)
    print()

    if not timelines:
        print("ERROR: no timelines found.")
        return

    print("Building temporal windows...")
    manifest_rows = build_windows(dataset, timelines, fps_map)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\n  Manifest written to: {output_path}")

    print("\nDone. Next steps:")
    print("  1. Spot-check a handful of real windows by hand against the CSV.")
    print("  2. For childplay/vacation specifically: independently verify the")
    print("     per-clip horizon_frames computation against a few real clips'")
    print("     own fps -- this hasn't had the same direct empirical check VAT's")
    print("     fixed mapping already got.")
    print("  3. This population will differ from any pre-existing temporal")
    print("     manifest for this dataset -- expected, not a bug.")


if __name__ == "__main__":
    main()
