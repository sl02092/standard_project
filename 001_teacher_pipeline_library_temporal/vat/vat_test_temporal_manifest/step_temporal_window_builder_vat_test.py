"""
step_temporal_window_builder_vat_test.py — VAT temporal window builder,
TEST SPLIT ONLY.

WHY THIS SCRIPT EXISTS, AND HOW MUCH CONFIDENCE TO PLACE IN IT: this
project's real VAT temporal manifest (vat_temporal_manifest.jsonl) was
built entirely from annotations/train, confirmed directly against the
real filesystem via check_vat_manifest_has_test_clips.py (0 of 214
manifest clips resolve to annotations/test). This script is a
RECONSTRUCTION of that same window-building logic, pointed at
annotations/test instead, assembled from pieces that ARE directly
confirmed rather than guessed:
  - load_annotations() is copied verbatim from
    compute_last_known_position_baseline_vat.py, which itself states it
    was "copied verbatim from step_temporal_window_builder_vat.py" --
    this is the one piece of the original builder's own code directly
    confirmed to exist, not inferred.
  - The manifest schema (bare zero-padded fname strings in
    context_frame_paths/target_frame_path, clip_id="show/clip",
    horizon_frames from a FIXED {100:2,300:7,500:12} map rather than
    per-clip fps) is confirmed from temporal_window_dataset.py,
    verify_temporal_contiguity.py, and train_temporal_v1.py's
    VAT_HORIZON_FRAMES_MAP directly.
  - The sliding-window mechanics (8-frame window, stride 4, "invalidate
    never pad", two-phase gap-scan-then-build design) are NOT VAT-specific
    knowledge -- they're the structure step_temporal_window_builder_
    vacation.py's own docstring states was "ported directly from
    step_temporal_window_builder_ucolaeo.py, unchanged in structure",
    and that whole family of builders traces back to VAT's own original
    design (VAT was this project's first temporal dataset). Reused here
    on that basis, not reinvented.

WHAT IS NOT independently confirmed, and is the single biggest risk in
this script: whether the ORIGINAL builder's own MIN_FRAMES / quality-
filter thresholds, or its exact horizon-frames-to-assumed-fps derivation,
match what's used below exactly. MIN_FRAMES=20 and the {100:2,300:7,500:12}
mapping are taken directly from confirmed project-wide constants (the
former is stated as "the same constant used across every dataset in this
project" in step_temporal_window_builder_vacation.py; the latter is
VAT_HORIZON_FRAMES_MAP itself, confirmed in train_temporal_v1.py) -- not
guessed, but also not read from VAT's own original builder script
directly, since that file was never available to read. If the real
train-split manifest is available, RUN THE VALIDATION BLOCK AT THE
BOTTOM OF THIS FILE FIRST, pointed at TRAIN instead of TEST, and confirm
it reproduces the known reference figures (214 clips, 21397 windows,
64191 rows, 59138 valid, from verify_vat_temporal_completeness.py's own
REFERENCE dict) before trusting this script's TEST output at all. If it
doesn't reproduce those figures exactly, something about this
reconstruction has drifted from the real original and needs fixing
before the test-split output can be trusted for anything.

OUTPUT: vat_test_temporal_manifest.jsonl -- a SEPARATE file, deliberately
not overwriting or touching the existing, already-validated
vat_temporal_manifest.jsonl (train-split) in any way.
"""

import os
import csv
import json
import random
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

VAT_ROOT = os.environ.get(
    "VAT_ROOT",
    r"C:\Users\scott\Desktop\dissertation\000_datasets\videoattentiontarget",
)
TEST_ANN_DIR = os.path.join(VAT_ROOT, "annotations", "test")

MIN_FRAMES = 20            # same constant used across every dataset in this project
MAX_UNRESOLVED_PCT = 0.60  # same value as every other dataset's quality filter --
                            # NOT independently re-derived for VAT test specifically;
                            # check the printed exclusion stats below once this runs
                            # at real scale, same discipline as every other dataset.

WINDOW_LENGTH = 8
STRIDE = 4
HORIZON_MS_LIST = [100, 300, 500]
# Confirmed directly from train_temporal_v1.py's own VAT_HORIZON_FRAMES_MAP --
# a FIXED mapping, not a per-clip fps computation (unlike VACATION/ChildPlay).
VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}

RANDOM_SEED = 42

OUTPUT_GAP_REPORT = "vat_test_temporal_gap_report.csv"
OUTPUT_MANIFEST_JSONL = "vat_test_temporal_manifest.jsonl"


# ══════════════════════════════════════════════════════════════════════
# ── ANNOTATION LOADING — copied verbatim from
# ── compute_last_known_position_baseline_vat.py (itself copied verbatim
# ── from the original step_temporal_window_builder_vat.py) ─────────────
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


# ══════════════════════════════════════════════════════════════════════
# ── SCANNING — build per-(show,clip,subject) canonical timelines ───────
# ══════════════════════════════════════════════════════════════════════

def scan_test_annotations():
    """Walks TEST_ANN_DIR/{show}/{clip}/{subject}.txt, building one
    canonical frame timeline per (show, clip, subject). Mirrors
    vacation_temporal_utils.build_person_timelines()'s own role: a
    verified-against-real-data foundational layer the window construction
    sits on top of, kept separate so a bug here is caught against real
    printed stats before Phase 2 runs at full scale."""
    if not os.path.isdir(TEST_ANN_DIR):
        raise FileNotFoundError(
            f"No annotations/test directory found at {TEST_ANN_DIR} -- "
            f"confirm VAT_ROOT is set correctly before proceeding."
        )

    timelines = {}  # (show, clip, subject) -> sorted list of {"fname","head","gaze"}
    n_shows, n_clips, n_subjects = 0, 0, 0

    for show in sorted(os.listdir(TEST_ANN_DIR)):
        show_dir = os.path.join(TEST_ANN_DIR, show)
        if not os.path.isdir(show_dir):
            continue
        n_shows += 1
        for clip in sorted(os.listdir(show_dir)):
            clip_dir = os.path.join(show_dir, clip)
            if not os.path.isdir(clip_dir):
                continue
            n_clips += 1
            for fname in sorted(os.listdir(clip_dir)):
                if not fname.endswith(".txt"):
                    continue
                subject = fname[:-4]
                n_subjects += 1
                ann_path = os.path.join(clip_dir, fname)
                rows = load_annotations(ann_path)
                # Canonical timeline: sorted by the frame's own numeric
                # value (parsed from its zero-padded fname), same
                # convention verify_temporal_contiguity.py's
                # parse_frame_idx() already established as correct for
                # this dataset.
                rows_with_idx = []
                for r in rows:
                    stem = os.path.splitext(r["fname"])[0]
                    if not stem.isdigit():
                        continue  # malformed fname -- skip rather than crash
                    rows_with_idx.append((int(stem), r))
                rows_with_idx.sort(key=lambda x: x[0])
                timelines[(show, clip, subject)] = rows_with_idx

    print(f"  Shows scanned    : {n_shows}")
    print(f"  Clips scanned    : {n_clips}")
    print(f"  Subject files    : {n_subjects}")
    print(f"  Timelines built  : {len(timelines)}")
    return timelines


def build_candidates(timelines):
    """Quality filter -- same MIN_FRAMES/MAX_UNRESOLVED_PCT convention as
    every other dataset in this project."""
    candidates = []
    n_excluded = defaultdict(int)

    for (show, clip, subject), rows_with_idx in sorted(timelines.items()):
        n_total = len(rows_with_idx)
        n_resolved = sum(1 for _, r in rows_with_idx if r["gaze"] is not None)
        unresolved_pct = 1 - (n_resolved / n_total) if n_total else 1.0

        if n_total < MIN_FRAMES:
            n_excluded[f"only {n_total} annotated frames"] += 1
            continue
        if unresolved_pct > MAX_UNRESOLVED_PCT:
            n_excluded[f"unresolved gaze {unresolved_pct*100:.0f}%"] += 1
            continue

        candidates.append({
            "show": show, "clip": clip, "subject": subject,
            "timeline": rows_with_idx,  # [(frame_idx, {"fname","head","gaze"}), ...]
        })

    print(f"  {sum(n_excluded.values())} (show,clip,subject) timelines excluded by quality filter:")
    for reason, n in sorted(n_excluded.items(), key=lambda x: -x[1]):
        print(f"    {n:>4}  {reason}")
    return candidates


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 1 — GAP LOGGING (dry run) ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def scan_gaps(candidates):
    gap_rows = []
    total_candidate_windows = 0
    windows_with_gap = 0

    for c in candidates:
        frame_nums = [idx for idx, _ in c["timeline"]]
        n = len(frame_nums)
        for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
            window_frame_nums = frame_nums[start:start + WINDOW_LENGTH]
            if len(window_frame_nums) < WINDOW_LENGTH:
                continue
            total_candidate_windows += 1
            non_contiguous = (window_frame_nums[-1] - window_frame_nums[0]) != (WINDOW_LENGTH - 1)
            if non_contiguous:
                windows_with_gap += 1
                gap_rows.append({
                    "show": c["show"], "clip": c["clip"], "subject": c["subject"],
                    "window_start_frame": window_frame_nums[0],
                    "window_end_frame": window_frame_nums[-1],
                    "reason": "non-contiguous timeline span",
                })

    with open(OUTPUT_GAP_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "show", "clip", "subject", "window_start_frame", "window_end_frame", "reason",
        ])
        writer.writeheader()
        writer.writerows(gap_rows)

    pct = (windows_with_gap / total_candidate_windows * 100) if total_candidate_windows else 0
    print("=" * 60)
    print("PHASE 1 — Gap scan (dry run, no manifest rows written)")
    print("=" * 60)
    print(f"  Qualifying (show,clip,subject) timelines: {len(candidates)}")
    print(f"  Candidate windows                        : {total_candidate_windows}")
    print(f"  Windows with a gap                       : {windows_with_gap} ({pct:.1f}%)")
    print(f"\n  Full detail written to: {OUTPUT_GAP_REPORT}")
    print("  >>> Review before trusting Phase 2's skip_reason counts. <<<\n")


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 2 — WINDOW CONSTRUCTION ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_windows(candidates):
    manifest_rows = []
    stats = defaultdict(int)

    for c in candidates:
        show, clip, subject = c["show"], c["clip"], c["subject"]
        timeline = c["timeline"]
        frame_nums = [idx for idx, _ in timeline]
        idx_to_row = {idx: r for idx, r in timeline}
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
                    context_frame_paths.append(r["fname"])  # bare zero-padded fname,
                                                              # matching VAT's own
                                                              # confirmed schema
                    context_head_boxes.append(list(r["head"]))

            for target_ms in HORIZON_MS_LIST:
                horizon_frames = VAT_HORIZON_FRAMES_MAP[target_ms]
                target_pos = last_idx_pos + horizon_frames
                row_valid = window_ok
                skip_reason = None
                target_frame_path = target_gaze_x = target_gaze_y = None
                target_gaze_type = None

                if not window_ok:
                    skip_reason = (
                        f"non-contiguous timeline span (window spans "
                        f"{window_frame_nums[-1] - window_frame_nums[0] + 1} real "
                        f"frames, not {WINDOW_LENGTH})"
                    )
                elif target_pos >= n:
                    row_valid = False
                    skip_reason = "target frame beyond available timeline"
                else:
                    target_frame_num = frame_nums[target_pos]
                    real_gap = target_frame_num - window_frame_nums[-1]
                    if real_gap != horizon_frames:
                        row_valid = False
                        skip_reason = (
                            f"non-contiguous span between context and target "
                            f"(real gap {real_gap}, not horizon_frames={horizon_frames})"
                        )
                    else:
                        target_row = idx_to_row[target_frame_num]
                        if target_row["gaze"] is None:
                            row_valid = False
                            skip_reason = "unresolved"
                            target_gaze_type = "offscreen"
                        else:
                            target_frame_path = target_row["fname"]
                            target_gaze_x, target_gaze_y = target_row["gaze"]
                            # NOTE: gaze_type ("social"/"object") is not derivable from
                            # this annotation format alone the way it is for
                            # VACATION/ChildPlay (their entity labels distinguish
                            # Person/Object directly) -- VAT's own gaze_type comes from
                            # a SEPARATE classification step against other subjects'
                            # boxes in the same frame, not present in this raw
                            # per-subject annotation file. Left as "unknown" here
                            # rather than guessed; if the real original builder
                            # resolved this differently, this is the one other place
                            # (besides the quality-filter thresholds) worth checking
                            # against a genuine reference if one becomes available.
                            target_gaze_type = "unknown"

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
                    if skip_reason.startswith("non-contiguous timeline span"):
                        coarse = "non-contiguous timeline span (context window)"
                    elif skip_reason.startswith("non-contiguous span between context and target"):
                        coarse = "non-contiguous span (context to target)"
                    else:
                        coarse = skip_reason
                    stats[f"skip:{coarse}"] += 1

    with open(OUTPUT_MANIFEST_JSONL, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    print("=" * 60)
    print("PHASE 2 — Window construction")
    print("=" * 60)
    print(f"  Total rows (windows x horizons) : {len(manifest_rows)}")
    print(f"  Valid                            : {stats['valid']}")
    print(f"  Invalid                          : {stats['invalid']}")
    print(f"\n  Invalid breakdown (coarse categories):")
    for reason, n in sorted(stats.items(), key=lambda x: -x[1]):
        if not reason.startswith("skip:"):
            continue
        print(f"    {n:>6}  {reason[5:]}")
    print(f"\n  Manifest written to: {OUTPUT_MANIFEST_JSONL}")
    return manifest_rows


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    print(f"VAT_ROOT           : {VAT_ROOT}")
    print(f"TEST_ANN_DIR        : {TEST_ANN_DIR}\n")

    print("Scanning annotations/test...")
    timelines = scan_test_annotations()
    print()

    print("Applying quality filter...")
    candidates = build_candidates(timelines)
    print(f"\nQualifying (show,clip,subject) timelines: {len(candidates)}\n")

    if not candidates:
        print("ERROR: no qualifying timelines found. Check VAT_ROOT and thresholds.")
        return

    scan_gaps(candidates)

    print("Proceeding to Phase 2 (window construction)...\n")
    build_windows(candidates)

    print("\nDone. IMPORTANT NEXT STEP, before trusting this for anything:")
    print("  1. Run check_vat_manifest_has_test_clips.py against")
    print("     vat_test_temporal_manifest.jsonl -- every clip should now resolve")
    print("     to annotations/test, not annotations/train (sanity check that this")
    print("     script actually read the right directory).")
    print("  2. Spot-check a handful of real windows by hand against the raw")
    print("     annotation files, the same way this project has verified every")
    print("     other dataset's window-builder before trusting it at scale.")
    print("  3. target_gaze_type is 'unknown' for every valid row here (see the")
    print("     code comment in build_windows()) -- this means object/social")
    print("     breakdown analysis won't be possible on this test population")
    print("     unless that classification step is added separately.")


if __name__ == "__main__":
    main()
