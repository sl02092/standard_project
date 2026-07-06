"""
step_temporal_window_builder_vat.py — Temporal Window Builder (VAT)
Objective 4 PoC — builds the temporal manifest for VAT, per the finalized
schema (temporal_manifest_schema.md, 2026-07-06).

WHY THIS SCRIPT EXISTS (2026-07-06): the static pipeline's clip scanner
(Step30_clip_scanner_2.py) selects and budgets *individual* frames for the
teacher-labelling pipeline — correct for that job, wrong for this one.
Its FRAME_SAMPLE_EVERY_N (keeps every 3rd frame) and MAX_FRAMES_PER_SHOW
(can truncate a clip mid-stream) both break the frame-to-frame contiguity
that temporal windows need. This script reuses Step30's clip-quality
filter (should_include) and per-frame gaze classification unchanged, but
replaces the frame-sampling/budget logic entirely with contiguous window
construction.

RUN ORDER — TWO PHASES, IN ONE SCRIPT:
  Phase 1 (scan_gaps): a dry-run pass over every qualifying clip/subject
    that reports how often a SUBJECT-LEVEL gap occurs mid-window — i.e.
    the clip's frame timeline continues, other subjects may still be
    annotated, but THIS subject has no annotation row for one or more
    frames inside what would otherwise be a valid context window. This
    was flagged as a real, expected possibility (VAT subjects go on/off
    screen) but not yet measured. Always run before Phase 2, and read the
    printed summary before trusting Phase 2's output — if gaps turn out
    to be common/clustered, the default policy below may need revisiting
    with real numbers in hand, not a guess.
  Phase 2 (build_windows): constructs the actual temporal manifest rows.
    DEFAULT POLICY (matches the project's existing "no fallback labels"
    rule): if a subject has any mid-window annotation gap, or a required
    horizon target is missing/off-clip, INVALIDATE — never substitute,
    interpolate, or pad. window_valid=False + skip_reason records why.
    This keeps every row either fully genuine or explicitly excluded.

WHAT COUNTS AS "CONTIGUOUS" HERE: the clip's own canonical frame timeline
(union of every subject's annotated fnames in that clip, sorted) is
treated as the ground truth frame sequence — this matches the 2026-07-06
visual/flip-book confirmation that VAT's provided frames are contiguous
with no clip-wide skips. A "gap" this script detects is never a clip-wide
gap (that would fail the should_include filter or corrupt the whole
clip's frame count) — it is specifically ONE subject missing an
annotation for a frame that the clip's timeline otherwise contains.

NOMINAL FPS: VAT has no authoritative fps (confirmed against the VAT
paper and ejcgt/attention-target-detection, which is frame-index only).
24fps is used here as a STATED, CAVEATED assumption for horizon_ms_nominal
only — horizon_frames (the real, exact quantity) never depends on it.

DESIGNED TO BE REUSED: this script is deliberately written so the
Phase 1 / Phase 2 split, the per-subject-gap detection logic, and the
per-horizon independent validity check can be lifted for ChildPlay and
VACATION with mostly just a different clip/annotation loader swapped in.
Keep that in mind before dataset-specific logic creeps into the shared
functions below.
"""

import os
import csv
import json
import random
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

BASE_PATH     = os.environ.get("VAT_BASE_PATH", r"C:\repo\standard_project\videoattentiontarget")
TRAIN_ANN_DIR = os.path.join(BASE_PATH, "annotations", "train")

# ── Clip selection thresholds — UNCHANGED from Step30_clip_scanner_2.py
# ── (same clip-quality bar as the static pipeline; not the frame-budget
# ── controls, which are deliberately NOT reused here — see docstring).
MIN_FRAMES         = 20
MIN_SUBJECTS       = 2
MIN_SOCIAL_PCT     = 0.20
MAX_OFFSCREEN_PCT  = 0.60
MIN_FRAMES_PER_SHOW = 200

# ── Temporal window parameters (finalized 2026-07-06) ──────────────────
WINDOW_LENGTH = 8      # context frames per window
STRIDE        = 4      # half-window overlap
NOMINAL_FPS   = 24     # VAT: stated nominal assumption, ms-figures only
HORIZON_MS_LIST = [100, 300, 500]
HORIZON_FRAMES_LIST = [max(1, round(ms / 1000 * NOMINAL_FPS)) for ms in HORIZON_MS_LIST]
# At 24fps nominal: [2, 7, 12]

RANDOM_SEED = 42

OUTPUT_GAP_REPORT     = "vat_temporal_gap_report.csv"
OUTPUT_MANIFEST_JSONL = "vat_temporal_manifest.jsonl"

# ══════════════════════════════════════════════════════════════════════
# ── REUSED FROM Step30_clip_scanner_2.py (unchanged logic) ─────────────
# ══════════════════════════════════════════════════════════════════════

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


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def should_include(row):
    if row["n_subjects"] < MIN_SUBJECTS:
        return False, f"only {row['n_subjects']} subject(s)"
    if row["total_ann_frames"] < MIN_FRAMES:
        return False, f"only {row['total_ann_frames']} annotated frames"
    if row["social_pct"] < MIN_SOCIAL_PCT:
        return False, f"social gaze only {row['social_pct']*100:.0f}%"
    if row["offscreen_pct"] > MAX_OFFSCREEN_PCT:
        return False, f"off-screen gaze {row['offscreen_pct']*100:.0f}%"
    return True, "ok"


def classify_gaze(gaze, label, head_boxes):
    """Same social/object/offscreen test as Step30 — reused so temporal
    target_gaze_type is classified identically to the static pipeline."""
    if gaze is None:
        return "offscreen"
    is_social = any(
        point_in_box(gaze, box)
        for other_label, box in head_boxes.items()
        if other_label != label
    )
    return "social" if is_social else "object"


def scan_clip(show, clip, ann_dir):
    """Loads one clip's subjects + builds the canonical frame timeline
    (union of all subjects' annotated fnames, sorted). Returns None if
    the clip has no usable annotation files."""
    clip_path = os.path.join(ann_dir, show, clip)
    ann_files = sorted(f for f in os.listdir(clip_path) if f.endswith(".txt"))
    if not ann_files:
        return None

    subjects = {}
    for fname in ann_files:
        label = fname.replace(".txt", "")
        frames = load_annotations(os.path.join(clip_path, fname))
        subjects[label] = {row["fname"]: row for row in frames}

    all_fnames = set()
    for frames in subjects.values():
        all_fnames.update(frames.keys())
    if not all_fnames:
        return None
    canonical_timeline = sorted(all_fnames)

    total_frames = social_frames = object_frames = offscreen_frames = 0
    for fname in canonical_timeline:
        head_boxes = {lbl: fr[fname]["head"] for lbl, fr in subjects.items() if fname in fr}
        for label, frames in subjects.items():
            if fname not in frames:
                continue
            total_frames += 1
            gtype = classify_gaze(frames[fname]["gaze"], label, head_boxes)
            if gtype == "social":
                social_frames += 1
            elif gtype == "object":
                object_frames += 1
            else:
                offscreen_frames += 1

    if total_frames == 0:
        return None

    return {
        "show": show, "clip": clip,
        "subjects": subjects,
        "canonical_timeline": canonical_timeline,
        "n_subjects": len(subjects),
        "total_ann_frames": total_frames,
        "social_pct": round(social_frames / total_frames, 3),
        "offscreen_pct": round(offscreen_frames / total_frames, 3),
    }


def get_qualifying_clips():
    """Scans all clips, applies should_include (clip-quality filter),
    then Step30's per-show minimum-frame filter — adapted to use
    total_ann_frames (every annotated frame in the clip) since this
    script has no frame-sampling step to define a "sampled_frames" count
    the way Step30 does. No frame-sampling, no per-show frame CAP —
    those two are deliberately not reused (see module docstring)."""
    shows = sorted(os.listdir(TRAIN_ANN_DIR))
    candidates = []
    for show in shows:
        show_path = os.path.join(TRAIN_ANN_DIR, show)
        if not os.path.isdir(show_path):
            continue
        for clip in sorted(os.listdir(show_path)):
            if not os.path.isdir(os.path.join(show_path, clip)):
                continue
            row = scan_clip(show, clip, TRAIN_ANN_DIR)
            if row is None:
                continue
            inc, reason = should_include(row)
            if inc:
                candidates.append(row)

    show_totals = defaultdict(int)
    for row in candidates:
        show_totals[row["show"]] += row["total_ann_frames"]
    low_shows = {s for s, n in show_totals.items() if n < MIN_FRAMES_PER_SHOW}
    if low_shows:
        print(f"  Dropping {len(low_shows)} show(s) below MIN_FRAMES_PER_SHOW "
              f"({MIN_FRAMES_PER_SHOW}): {sorted(low_shows)}")

    return [row for row in candidates if row["show"] not in low_shows]


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 1 — GAP LOGGING (dry run, no windows built yet) ──────────────
# ══════════════════════════════════════════════════════════════════════

def scan_gaps(qualifying_clips):
    """For every qualifying clip x subject, slides the same 8/stride-4
    window pattern Phase 2 will use, but only CHECKS whether the subject
    has an annotation for every frame in each candidate window — logging
    every mid-window gap found. Builds no manifest rows. Read the printed
    summary before trusting Phase 2's skip-rate."""
    gap_rows = []
    total_candidate_windows = 0
    windows_with_gap = 0

    for clip in qualifying_clips:
        timeline = clip["canonical_timeline"]
        n = len(timeline)
        for subject, ann in clip["subjects"].items():
            for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
                window_fnames = timeline[start:start + WINDOW_LENGTH]
                if len(window_fnames) < WINDOW_LENGTH:
                    continue  # too short — not a candidate window at all
                total_candidate_windows += 1
                missing = [fn for fn in window_fnames if fn not in ann]
                if missing:
                    windows_with_gap += 1
                    gap_rows.append({
                        "show": clip["show"], "clip": clip["clip"],
                        "subject": subject,
                        "window_start_fname": window_fnames[0],
                        "window_end_fname": window_fnames[-1],
                        "n_missing_in_window": len(missing),
                        "missing_fnames": "|".join(missing),
                    })

    with open(OUTPUT_GAP_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "show", "clip", "subject", "window_start_fname", "window_end_fname",
            "n_missing_in_window", "missing_fnames",
        ])
        writer.writeheader()
        writer.writerows(gap_rows)

    pct = (windows_with_gap / total_candidate_windows * 100) if total_candidate_windows else 0
    print("=" * 60)
    print("PHASE 1 — Gap scan (dry run, no manifest rows written)")
    print("=" * 60)
    print(f"  Qualifying clips scanned      : {len(qualifying_clips)}")
    print(f"  Candidate windows (all subj)  : {total_candidate_windows}")
    print(f"  Windows with a mid-window gap : {windows_with_gap} ({pct:.1f}%)")

    by_show = defaultdict(lambda: [0, 0])
    for clip in qualifying_clips:
        pass
    gap_by_show = defaultdict(int)
    for r in gap_rows:
        gap_by_show[r["show"]] += 1
    if gap_by_show:
        print("\n  Gaps by show (top 10):")
        for show, n in sorted(gap_by_show.items(), key=lambda x: -x[1])[:10]:
            print(f"    {show:<35} {n} affected windows")
    print(f"\n  Full detail written to: {OUTPUT_GAP_REPORT}")
    print("  >>> Review this before trusting Phase 2's skip_reason counts. <<<\n")

    return windows_with_gap, total_candidate_windows


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 2 — WINDOW CONSTRUCTION ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_windows(qualifying_clips):
    """Builds the actual temporal manifest, per temporal_manifest_schema.md.
    DEFAULT POLICY: any mid-window subject gap, or any horizon target that
    is missing/off-clip, invalidates that row (window_valid=False) with an
    explicit skip_reason. No padding, no interpolation, no fallback label."""
    manifest_rows = []
    stats = defaultdict(int)

    for clip in qualifying_clips:
        timeline = clip["canonical_timeline"]
        n = len(timeline)
        subjects = clip["subjects"]

        for subject, ann in subjects.items():
            for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
                window_fnames = timeline[start:start + WINDOW_LENGTH]
                if len(window_fnames) < WINDOW_LENGTH:
                    continue
                last_idx = start + WINDOW_LENGTH - 1

                missing_in_window = [fn for fn in window_fnames if fn not in ann]
                window_ok = len(missing_in_window) == 0

                context_frame_paths = window_fnames
                context_head_boxes = []
                other_people_boxes = []
                if window_ok:
                    for fn in window_fnames:
                        context_head_boxes.append(list(ann[fn]["head"]))
                        others = {
                            lbl: subjects[lbl][fn]["head"]
                            for lbl in subjects
                            if lbl != subject and fn in subjects[lbl]
                        }
                        other_people_boxes.append(
                            [list(box) for box in others.values()]
                        )

                for horizon_frames, horizon_ms in zip(HORIZON_FRAMES_LIST, HORIZON_MS_LIST):
                    target_pos = last_idx + horizon_frames
                    row_valid = window_ok
                    skip_reason = None
                    target_fname = None
                    target_gaze_x = target_gaze_y = None
                    target_gaze_type = None

                    if not window_ok:
                        skip_reason = (
                            f"mid-window annotation gap for subject "
                            f"({len(missing_in_window)} frame(s) missing)"
                        )
                    elif target_pos >= n:
                        row_valid = False
                        skip_reason = "target frame beyond clip end"
                    else:
                        target_fname = timeline[target_pos]
                        if target_fname not in ann:
                            row_valid = False
                            skip_reason = "target frame missing (subject not annotated)"
                        else:
                            head_boxes = {
                                lbl: subjects[lbl][target_fname]["head"]
                                for lbl in subjects
                                if target_fname in subjects[lbl]
                            }
                            gaze = ann[target_fname]["gaze"]
                            target_gaze_type = classify_gaze(gaze, subject, head_boxes)
                            if gaze is not None:
                                target_gaze_x, target_gaze_y = gaze

                    manifest_rows.append({
                        "source_dataset": "vat",
                        "clip_id": f"{clip['show']}/{clip['clip']}",
                        "subject_id": subject,
                        "context_frame_paths": context_frame_paths if row_valid else context_frame_paths,
                        "context_head_boxes": context_head_boxes if row_valid else [],
                        "other_people_boxes": other_people_boxes if row_valid else [],
                        "target_frame_path": target_fname,
                        "target_gaze_x": target_gaze_x,
                        "target_gaze_y": target_gaze_y,
                        "target_gaze_type": target_gaze_type,
                        "horizon_frames": horizon_frames,
                        "horizon_ms_nominal": horizon_ms,
                        "window_valid": row_valid,
                        "skip_reason": skip_reason,
                    })
                    stats["valid" if row_valid else "invalid"] += 1
                    if skip_reason:
                        stats[skip_reason] += 1

    with open(OUTPUT_MANIFEST_JSONL, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    print("=" * 60)
    print("PHASE 2 — Window construction")
    print("=" * 60)
    print(f"  Total rows (windows x horizons) : {len(manifest_rows)}")
    print(f"  Valid                            : {stats['valid']}")
    print(f"  Invalid                          : {stats['invalid']}")
    print(f"\n  Manifest written to: {OUTPUT_MANIFEST_JSONL}")

    return manifest_rows


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    print("Scanning clips and applying clip-quality filter (should_include)...")
    qualifying_clips = get_qualifying_clips()
    print(f"Qualifying clips: {len(qualifying_clips)}\n")

    if not qualifying_clips:
        print("ERROR: no qualifying clips found. Check VAT_BASE_PATH and "
              "clip-quality thresholds.")
        return

    scan_gaps(qualifying_clips)

    print("Proceeding to Phase 2 (window construction) using default policy: "
          "invalidate on any mid-window gap or missing horizon target.\n")
    build_windows(qualifying_clips)

    print("\nDone. Recommended next step: read vat_temporal_gap_report.csv "
          "and the valid/invalid breakdown above before treating "
          "vat_temporal_manifest.jsonl as final — if invalid-rate is high "
          "or concentrated in a few shows, that's worth a conscious decision "
          "rather than an assumed default.")


if __name__ == "__main__":
    main()
