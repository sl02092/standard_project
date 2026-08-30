"""
step_temporal_window_builder_vacation.py — Temporal Window Builder
(VACATION), per vacation_temporal_manifest_schema.md.

WHAT'S PORTED DIRECTLY FROM step_temporal_window_builder_ucolaeo.py,
UNCHANGED IN STRUCTURE: the two-phase design (Phase 1 scan_gaps dry-run
before trusting Phase 2), the sliding 8-frame/stride-4 window mechanics,
the "invalidate, never pad" policy, and both contiguity checks (context-
window internal-span check, target-frame real-gap check) -- built in
here from the start, not discovered the same way twice.

WHAT'S ADAPTED, AND WHY:
1. Canonical timeline: per (video_id, person_label), from
   vacation_temporal_utils.build_person_timelines() -- includes frames
   whether or not that person's own gaze resolved that frame, same
   "annotated regardless of target validity" property as every other
   dataset.
2. Quality filter: MIN_FRAMES / MAX_UNRESOLVED_PCT port directly from
   UCO-LAEO's version. Deliberately NO MIN_SUBJECTS filter -- see schema
   doc: VACATION's object-gaze targets need no other person present at
   all, so a "needs 2+ people" filter would exclude real, valid
   solo-with-object content. Not rediscovering ChildPlay's/UCO-LAEO's
   own lesson a third time.
3. Target validity: THREE real states here, not UCO-LAEO's two --
   resolved-social, resolved-object, or unresolved. gaze_type carries
   through directly from build_person_timelines() rather than being
   collapsed to a single "resolved" flag, so this dataset's eventual PoC
   can ask the full object/social breakdown question VAT and ChildPlay
   could, unlike UCO-LAEO.
4. Horizon: real per-video fps (video_fps.txt), same three-field design
   and same compute_horizon_frames() formula as ChildPlay/UCO-LAEO's.
5. other_people_boxes: from frame_people, self excluded explicitly.
   frame_people already excludes objects by construction (see
   vacation_temporal_utils.py) -- this field is genuinely "other people"
   here, same meaning as ChildPlay/UCO-LAEO's schema, not "other
   entities."
6. Images: LOCAL paths (this pipeline runs on the local machine, not the
   HPC), filename convention `{video_id}_{frame}.jpg` -- matches
   extract_frames_vacation.py's real output, NOT UCO-LAEO's
   zero-padded `{frame:06d}.jpg` convention. Different dataset,
   different naming -- confirmed against the real source, not assumed
   to match UCO-LAEO's.
"""

import os
import csv
import json
import random
from collections import defaultdict

from vacation_temporal_utils import (
    build_person_timelines, load_fps_map, IMAGES_ROOT,
)

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

MIN_FRAMES = 20            # same constant used across every dataset in this project
MAX_UNRESOLVED_PCT = 0.60  # same value as ChildPlay's MAX_NONTARGETABLE_PCT
                            # and UCO-LAEO's MAX_UNRESOLVED_PCT, kept for
                            # cross-dataset consistency. Not yet retuned
                            # against VACATION's own real exclusion stats
                            # -- check the quality-filter print below
                            # once this runs at real scale, same as every
                            # other dataset's thresholds were.

# NO MIN_SUBJECTS -- see module docstring point 2 and schema doc. This
# is a deliberate omission, not an unfinished port.

WINDOW_LENGTH = 8
STRIDE = 4
HORIZON_MS_LIST = [100, 300, 500]

RANDOM_SEED = 42

OUTPUT_GAP_REPORT = "vacation_temporal_gap_report.csv"
OUTPUT_MANIFEST_JSONL = "vacation_temporal_manifest.jsonl"


# ══════════════════════════════════════════════════════════════════════
# ── SCANNING + QUALITY FILTER ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_candidates():
    """Returns qualifying (video_id, person_label) timelines, each
    carrying its own fps and canonical frame list -- analogous to
    UCO-LAEO's build_candidates(), same per-person granularity (VACATION's
    natural unit is per-person, like UCO-LAEO, not per-clip like
    ChildPlay)."""
    timelines, frame_people = build_person_timelines()
    fps_map = load_fps_map()

    candidates = []
    n_excluded = defaultdict(int)

    for (video_id, person_label), t in sorted(timelines.items()):
        frames = t["frames"]
        n_total = len(frames)
        n_resolved = sum(1 for f in frames.values() if f["gaze_type"] is not None)
        unresolved_pct = 1 - (n_resolved / n_total) if n_total else 1.0

        if n_total < MIN_FRAMES:
            n_excluded[f"only {n_total} annotated frames"] += 1
            continue
        if unresolved_pct > MAX_UNRESOLVED_PCT:
            n_excluded[f"unresolved gaze {unresolved_pct*100:.0f}%"] += 1
            continue

        fps = fps_map.get(video_id)
        if fps is None:
            n_excluded["no fps found"] += 1
            continue

        candidates.append({
            "video_id": video_id, "person_label": person_label, "fps": fps,
            "frames": frames, "canonical_timeline": sorted(frames.keys()),
        })

    print(f"  {sum(n_excluded.values())} (video, person) timelines excluded by quality filter:")
    for reason, n in sorted(n_excluded.items(), key=lambda x: -x[1]):
        print(f"    {n:>4}  {reason}")
    return candidates, frame_people


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 1 — GAP LOGGING (dry run) ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def scan_gaps(candidates):
    gap_rows = []
    total_candidate_windows = 0
    windows_with_gap = 0

    for c in candidates:
        timeline = c["canonical_timeline"]
        n = len(timeline)
        for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
            window_frame_nums = timeline[start:start + WINDOW_LENGTH]
            if len(window_frame_nums) < WINDOW_LENGTH:
                continue
            total_candidate_windows += 1
            # Built in from the start (ChildPlay's round-1 fix, applied
            # here up front rather than discovered later): presence at
            # each selected position doesn't guarantee those positions
            # are numerically contiguous.
            non_contiguous = (window_frame_nums[-1] - window_frame_nums[0]) != (WINDOW_LENGTH - 1)
            if non_contiguous:
                windows_with_gap += 1
                gap_rows.append({
                    "video_id": c["video_id"], "person_label": c["person_label"],
                    "window_start_frame": window_frame_nums[0],
                    "window_end_frame": window_frame_nums[-1],
                    "reason": "non-contiguous timeline span",
                })

    with open(OUTPUT_GAP_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "person_label", "window_start_frame", "window_end_frame", "reason",
        ])
        writer.writeheader()
        writer.writerows(gap_rows)

    pct = (windows_with_gap / total_candidate_windows * 100) if total_candidate_windows else 0
    print("=" * 60)
    print("PHASE 1 — Gap scan (dry run, no manifest rows written)")
    print("=" * 60)
    print(f"  Qualifying (video,person) timelines: {len(candidates)}")
    print(f"  Candidate windows                  : {total_candidate_windows}")
    print(f"  Windows with a gap                 : {windows_with_gap} ({pct:.1f}%)")
    print(f"\n  Full detail written to: {OUTPUT_GAP_REPORT}")
    print("  >>> Review before trusting Phase 2's skip_reason counts. <<<\n")


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 2 — WINDOW CONSTRUCTION ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def compute_horizon_frames(fps, target_ms):
    horizon_frames = max(1, round(target_ms / 1000 * fps))
    horizon_ms_actual = horizon_frames / fps * 1000
    return horizon_frames, horizon_ms_actual


def build_image_path(video_id, frame_num):
    """Matches extract_frames_vacation.py's real output convention --
    NOT UCO-LAEO's zero-padded frame naming, confirmed against the real
    source rather than assumed to match."""
    return os.path.join(IMAGES_ROOT, video_id, f"{video_id}_{frame_num}.jpg")


def build_windows(candidates, frame_people):
    manifest_rows = []
    stats = defaultdict(int)

    for c in candidates:
        video_id, person_label, fps = c["video_id"], c["person_label"], c["fps"]
        timeline = c["canonical_timeline"]
        n = len(timeline)
        frames = c["frames"]

        for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
            window_frame_nums = timeline[start:start + WINDOW_LENGTH]
            if len(window_frame_nums) < WINDOW_LENGTH:
                continue
            last_idx = start + WINDOW_LENGTH - 1

            # Both checks built in from the start -- see module docstring.
            non_contiguous = (window_frame_nums[-1] - window_frame_nums[0]) != (WINDOW_LENGTH - 1)
            window_ok = not non_contiguous

            context_frame_paths, context_head_boxes, other_people_boxes = [], [], []
            if window_ok:
                for fn in window_frame_nums:
                    context_frame_paths.append(build_image_path(video_id, fn))
                    context_head_boxes.append(list(frames[fn]["head_box"]))
                    others = {
                        label: box for label, box in frame_people.get((video_id, fn), {}).items()
                        if label != person_label
                    }
                    other_people_boxes.append([list(box) for box in others.values()])

            for target_ms in HORIZON_MS_LIST:
                horizon_frames, horizon_ms_actual = compute_horizon_frames(fps, target_ms)
                target_pos = last_idx + horizon_frames
                row_valid = window_ok
                skip_reason = None
                target_frame_path = target_gaze_x = target_gaze_y = target_gaze_type = None

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
                    target_frame_num = timeline[target_pos]
                    real_gap = target_frame_num - window_frame_nums[-1]
                    if real_gap != horizon_frames:
                        row_valid = False
                        skip_reason = (
                            f"non-contiguous span between context and target "
                            f"(real gap {real_gap}, not horizon_frames={horizon_frames})"
                        )
                    else:
                        target_ann = frames.get(target_frame_num)
                        if target_ann is None:
                            row_valid = False
                            skip_reason = "target frame missing from timeline"
                        elif target_ann["gaze_type"] is None:
                            row_valid = False
                            skip_reason = "unresolved"
                        else:
                            target_frame_path = build_image_path(video_id, target_frame_num)
                            target_gaze_x, target_gaze_y = target_ann["gaze"]
                            target_gaze_type = target_ann["gaze_type"]  # "social" or "object"

                manifest_rows.append({
                    "source_dataset": "vacation",
                    "clip_id": video_id,
                    "subject_id": person_label,
                    "context_frame_paths": context_frame_paths,
                    "context_head_boxes": context_head_boxes,
                    "other_people_boxes": other_people_boxes,
                    "target_frame_path": target_frame_path,
                    "target_gaze_x": target_gaze_x,
                    "target_gaze_y": target_gaze_y,
                    "target_gaze_type": target_gaze_type,
                    "target_horizon_ms": target_ms,
                    "horizon_frames": horizon_frames,
                    "horizon_ms_actual": round(horizon_ms_actual, 1),
                    "window_valid": row_valid,
                    "skip_reason": skip_reason,
                })
                stats["valid" if row_valid else "invalid"] += 1
                if row_valid:
                    stats[f"valid_{target_gaze_type}"] += 1
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
    print(f"  Valid                            : {stats['valid']} "
          f"(social={stats['valid_social']}, object={stats['valid_object']})")
    print(f"  Invalid                          : {stats['invalid']}")
    print(f"\n  Invalid breakdown (coarse categories, full detail in manifest rows):")
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
    print("Building person timelines and applying quality filter...")
    candidates, frame_people = build_candidates()
    print(f"\nQualifying (video, person) timelines: {len(candidates)}\n")

    if not candidates:
        print("ERROR: no qualifying timelines found. Check paths and thresholds.")
        return

    scan_gaps(candidates)

    print("Proceeding to Phase 2 (window construction)...\n")
    build_windows(candidates, frame_people)

    print("\nDone. Recommended next step: read vacation_temporal_gap_report.csv "
          "and the invalid breakdown above before treating "
          "vacation_temporal_manifest.jsonl as final. MAX_UNRESOLVED_PCT is "
          "carried over from ChildPlay/UCO-LAEO, not yet retuned against "
          "this run's real numbers -- check the quality-filter exclusion "
          "stats above for that signal, same as every other dataset's "
          "threshold was.")


if __name__ == "__main__":
    main()
