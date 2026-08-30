"""
step_temporal_window_builder_childplay.py — Temporal Window Builder (ChildPlay)
Objective 4 extension — builds the temporal manifest for ChildPlay, per
childplay_temporal_manifest_schema.md.

WHAT'S PORTED DIRECTLY FROM step_temporal_window_builder_vat.py, UNCHANGED
IN STRUCTURE: the two-phase design (Phase 1 scan_gaps dry-run before
trusting Phase 2), the "canonical timeline = union of all subjects'
annotated frame numbers, sorted" construction (defensive — not assumed
dense even though real samples suggest ChildPlay usually is), the sliding
8-frame/stride-4 window mechanics, and the "invalidate, never pad" policy
for both mid-window gaps and missing horizon targets. These are the parts
the original README explicitly said were written to be dataset-agnostic,
and they are — this file changes none of that logic, only what it's
pointed at.

FIX (2026-07-18): verify_temporal_contiguity_childplay.py found 11 real
violations (out of 151,886 valid rows -- 0.007%) where a window's 8
selected timeline POSITIONS were each individually valid (subject present
at every one) but not numerically CONTIGUOUS -- a full-clip annotation
gap (nobody annotated, not just one subject, confirmed via one traced
example: iWYy8n6mSwc_12172-12757, an 11-frame gap at relative frames
12218-12228) sitting between two runs of otherwise-valid frames, straddled
by a window because the position-based subject-presence check never
verified the selected positions were themselves temporally contiguous.
Fixed in both scan_gaps() and build_windows() with an explicit span check
(window's last frame_num minus first must equal exactly WINDOW_LENGTH-1).

FIX ROUND 2 (2026-07-18, same day): the first fix was incomplete -- it
only checked the CONTEXT window's own internal contiguity, not the
TARGET frame's distance from it. target_pos = last_idx + horizon_frames
is ALSO a list-position offset, not a verified real-frame one, so the
exact same class of bug remained in the target lookup (still visible via
verify_temporal_contiguity_childplay.py: 0 context violations after round
1, but 5 horizon-offset violations remained, all traced to the same one
gap). Fixed by verifying target_frame_num - window_frame_nums[-1] ==
horizon_frames exactly before trusting the target; if it doesn't match,
the row is invalidated with an explicit skip_reason rather than silently
accepted. The manifest needs rebuilding again to purge these rows;
rerun verify_temporal_contiguity_childplay.py afterward -- this time
expecting BOTH violation counts at zero.

WHAT'S ADAPTED, AND WHY:
1. Annotation source: childplay_annotation_utils.py's load_clip_annotations
   (raw CSVs) instead of VAT's load_annotations (raw .txt files) — same
   role, different parser, already independently validated against real
   clips.
2. Quality filter (should_include): same four-criterion SHAPE as VAT's
   (min subjects, min frames, min social%, max non-targetable%). Started
   with VAT's threshold VALUES, flagged for retuning against ChildPlay's
   real numbers once this ran — it did, and MIN_SOCIAL_PCT was retuned
   from 0.20 to 0.0 as a direct result (see its own inline comment below
   for the full reasoning; short version: it was excluding 75% of every
   clip this filter rejected, almost entirely object-gaze-heavy content
   this project's own findings make valuable, not disposable).
   MAX_NONTARGETABLE_PCT stayed at VAT's original 0.60 — barely binding
   in the first real run, no comparable evidence it needs changing.
   "max_offscreen_pct" is generalised to "offscreen + invalid %" since
   ChildPlay has several additional non-targetable categories VAT never
   had (see point 4).
3. Downsampled clips (9 of 401) excluded entirely at the metadata stage —
   resolve_image_path() would raise on them as a second safety net if one
   somehow reached that far, but they're filtered out before that.
4. Target validity has a THIRD failure mode VAT never needed: a subject
   can be genuinely annotated at the target frame (not absent — that's
   the existing "gap" case) but have gaze_class in
   {inside_occluded, inside_uncertain, eyes_closed, gaze_shift} — cases
   where VAT's classify_gaze() had no equivalent (VAT's occlusion-like
   cases always showed up as the subject being ENTIRELY ABSENT from that
   frame, never present-but-marked-invalid). skip_reason carries the
   specific category rather than a generic reason, richer than VAT's gap
   analysis could offer.
5. Horizon computation: VAT used one fixed HORIZON_FRAMES_LIST for the
   whole dataset (single nominal fps). ChildPlay's fps is real and
   per-clip, so horizon_frames is computed per clip from its own fps —
   three fields (target_horizon_ms, horizon_frames, horizon_ms_actual),
   not two — see schema doc §4 for the full reasoning.
6. context_frame_paths stores genuinely RESOLVABLE full paths (via
   resolve_image_path), not bare filenames. VAT's own manifest named this
   field "paths" but actually stored bare fnames, requiring a
   reconstruction step later (extract_temporal_frame_features.py had to
   work around this). Deliberately not repeating that wrinkle here now
   that it's a known, already-paid-for lesson.
"""

import os
import csv
import json
import random
from collections import defaultdict

from childplay_annotation_utils import (
    load_clip_metadata, load_clip_annotations, resolve_image_path,
    classify_target,
)

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

# Same criteria shape as VAT's should_include. MIN_SOCIAL_PCT dropped to
# 0.0 (2026-07-16), CHANGED from the VAT-ported 0.20 starting point --
# not a formatting tweak, a real, evidenced retuning decision:
#
# The first real run against ChildPlay's actual data showed this single
# threshold responsible for 190 of 254 excluded clips (75% of every
# exclusion combined) -- MAX_NONTARGETABLE_PCT, by contrast, excluded
# exactly one. MIN_SOCIAL_PCT=0.20 was ported unchanged from VAT, where
# it made sense: VAT is scripted TV/film drama, built around dialogue and
# eye contact. ChildPlay is children playing with toys, often
# solo-absorbed even with a supervising adult in frame -- a clip with low
# social-gaze content there isn't low-quality data, it's ChildPlay doing
# exactly what it's for. Filtering it out would import VAT's content
# assumptions into a dataset that was never meant to match them.
#
# This project's primary framing is social gaze -- which is likely why
# this threshold existed in the first place, copied from VAT without
# question. But object gaze was added to the project's scope later on,
# and the static/temporal work's own central finding (temporal modelling
# adds real, statistically-confirmed value specifically on OBJECT-directed
# gaze, where the static encoder's estimates are noisiest -- see the
# Objective 4 write-up) makes ChildPlay's object-gaze-heavy clips
# disproportionately valuable for this project specifically, not merely
# tolerable. Staying focused on social gaze as the primary lens doesn't
# require minimising the object-gaze data that a real secondary finding
# already depends on. Set to 0.0 (not removed as a field, in case a
# future reason to reinstate a floor arises) -- MIN_FRAMES and
# MAX_NONTARGETABLE_PCT remain as the real quality gates.
MIN_SOCIAL_PCT = 0.0
MIN_FRAMES = 20
MAX_NONTARGETABLE_PCT = 0.60  # unchanged -- barely binding in the first real
                               # run (1/254 exclusions), no comparable evidence
                               # this one needs retuning

# MIN_SUBJECTS REMOVED (2026-07-30) -- confirmed against a real run to
# exclude 63 of 392 candidate clips (16.1%) for having only one subject,
# essentially all otherwise usable (MIN_FRAMES excluded ZERO clips that
# same run). Same VAT-ported "needs 2+ people" assumption removed once
# already this session for UCO-LAEO -- there it had zero real effect;
# here it's the opposite, a substantial exclusion of genuinely valid
# solo-child object-gaze data. That's not low-quality data -- solo-child
# scenes are exactly what ChildPlay is FOR (same reasoning MIN_SOCIAL_PCT's
# own 2026-07-16 retuning already established), and object-gaze content
# is specifically what this project's temporal findings already lean on
# most. Never a deliberate choice for ChildPlay -- an unexamined carryover
# from VAT's own multi-person-drama framing.

WINDOW_LENGTH = 8
STRIDE = 4
HORIZON_MS_LIST = [100, 300, 500]

RANDOM_SEED = 42

OUTPUT_GAP_REPORT = "childplay_temporal_gap_report.csv"
OUTPUT_MANIFEST_JSONL = "childplay_temporal_manifest.jsonl"


# ══════════════════════════════════════════════════════════════════════
# ── CLIP SCANNING + QUALITY FILTER ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def scan_clip(clip_id, meta):
    """Loads one clip's annotations, builds the canonical frame timeline
    (union of all subjects' annotated relative-frame numbers, sorted --
    same defensive construction as VAT's scan_clip). Returns None if the
    clip is downsampled, unreadable, or has no usable annotations."""
    if meta["downsampled"]:
        return None

    try:
        frames = load_clip_annotations(clip_id, meta["split"])
    except FileNotFoundError:
        return None

    all_frame_nums = sorted(frames.keys())
    if not all_frame_nums:
        return None

    subjects = set()
    for people in frames.values():
        subjects.update(people.keys())

    total = social = obj = offscreen = invalid = 0
    invalid_by_category = defaultdict(int)
    for frame_num, people in frames.items():
        head_boxes = {pid: ann["head_box"] for pid, ann in people.items()}
        for pid, ann in people.items():
            total += 1
            gtype, skip = classify_target(ann["gaze_class"], ann["gaze"], pid, head_boxes)
            if gtype == "social":
                social += 1
            elif gtype == "object":
                obj += 1
            elif gtype == "offscreen":
                offscreen += 1
            else:
                invalid += 1
                invalid_by_category[skip] += 1

    if total == 0:
        return None

    return {
        "clip_id": clip_id, "split": meta["split"], "fps": meta["fps"],
        "frames_by_num": frames,
        "canonical_timeline": all_frame_nums,
        "n_subjects": len(subjects),
        "total_ann_frames": total,
        "social_pct": round(social / total, 3),
        "offscreen_pct": round(offscreen / total, 3),
        "invalid_pct": round(invalid / total, 3),
        "invalid_by_category": dict(invalid_by_category),
    }


def should_include(row):
    # n_subjects retained in the row dict as passive metadata (e.g. a
    # future solo-vs-multi-person breakdown), just no longer gates
    # inclusion -- see MIN_SUBJECTS removal note above.
    if row["total_ann_frames"] < MIN_FRAMES:
        return False, f"only {row['total_ann_frames']} annotated frames"
    if row["social_pct"] < MIN_SOCIAL_PCT:
        return False, f"social gaze only {row['social_pct']*100:.0f}%"
    nontargetable = row["offscreen_pct"] + row["invalid_pct"]
    if nontargetable > MAX_NONTARGETABLE_PCT:
        return False, f"offscreen+invalid gaze {nontargetable*100:.0f}%"
    return True, "ok"


def get_qualifying_clips():
    all_meta = load_clip_metadata()
    candidates = []
    n_downsampled = 0
    n_excluded = defaultdict(int)

    for clip_id, meta in sorted(all_meta.items()):
        if meta["downsampled"]:
            n_downsampled += 1
            continue
        row = scan_clip(clip_id, meta)
        if row is None:
            n_excluded["unreadable/empty"] += 1
            continue
        inc, reason = should_include(row)
        if inc:
            candidates.append(row)
        else:
            n_excluded[reason] += 1

    print(f"  {n_downsampled} downsampled clips excluded (see schema doc).")
    print(f"  {sum(n_excluded.values())} clips excluded by quality filter:")
    for reason, n in sorted(n_excluded.items(), key=lambda x: -x[1]):
        print(f"    {n:>4}  {reason}")
    return candidates


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 1 — GAP LOGGING (dry run) ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def scan_gaps(qualifying_clips):
    gap_rows = []
    total_candidate_windows = 0
    windows_with_gap = 0

    for clip in qualifying_clips:
        timeline = clip["canonical_timeline"]
        n = len(timeline)
        frames_by_num = clip["frames_by_num"]
        subjects = set()
        for people in frames_by_num.values():
            subjects.update(people.keys())

        for subject in subjects:
            for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
                window_frame_nums = timeline[start:start + WINDOW_LENGTH]
                if len(window_frame_nums) < WINDOW_LENGTH:
                    continue
                total_candidate_windows += 1
                missing = [fn for fn in window_frame_nums if subject not in frames_by_num.get(fn, {})]
                # FIX (2026-07-18, found via verify_temporal_contiguity_childplay.py):
                # subject-presence at each selected TIMELINE POSITION doesn't
                # guarantee those positions are numerically CONTIGUOUS -- a
                # full-clip annotation gap (nobody annotated, not just this
                # subject) can sit between two runs of otherwise-valid frames,
                # and a window can straddle it while every individual position
                # still passes the presence check above. Explicit span check:
                # a genuinely contiguous 8-frame window's last frame_num minus
                # its first must equal exactly WINDOW_LENGTH-1.
                non_contiguous = (window_frame_nums[-1] - window_frame_nums[0]) != (WINDOW_LENGTH - 1)
                if missing or non_contiguous:
                    windows_with_gap += 1
                    reason = "subject missing" if missing else "non-contiguous timeline span"
                    gap_rows.append({
                        "clip_id": clip["clip_id"], "subject": subject,
                        "window_start_frame": window_frame_nums[0],
                        "window_end_frame": window_frame_nums[-1],
                        "n_missing_in_window": len(missing),
                        "missing_frames": "|".join(str(f) for f in missing),
                        "reason": reason,
                    })

    with open(OUTPUT_GAP_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "subject", "window_start_frame", "window_end_frame",
            "n_missing_in_window", "missing_frames", "reason",
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
    print(f"\n  Full detail written to: {OUTPUT_GAP_REPORT}")
    print("  >>> Review before trusting Phase 2's skip_reason counts. <<<\n")
    return windows_with_gap, total_candidate_windows


# ══════════════════════════════════════════════════════════════════════
# ── PHASE 2 — WINDOW CONSTRUCTION ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def compute_horizon_frames(clip_fps, target_ms):
    """Three-field horizon design (schema doc §4): horizon_frames is the
    real, clip-specific integer actually used; horizon_ms_actual is what
    that frame-count really corresponds to at this clip's real fps --
    usually close to target_ms but not always exact (integer rounding),
    kept visible rather than silently absorbed."""
    horizon_frames = max(1, round(target_ms / 1000 * clip_fps))
    horizon_ms_actual = horizon_frames / clip_fps * 1000
    return horizon_frames, horizon_ms_actual


def build_windows(qualifying_clips):
    manifest_rows = []
    stats = defaultdict(int)

    for clip in qualifying_clips:
        timeline = clip["canonical_timeline"]
        n = len(timeline)
        frames_by_num = clip["frames_by_num"]
        clip_fps = clip["fps"]
        subjects = set()
        for people in frames_by_num.values():
            subjects.update(people.keys())

        for subject in subjects:
            for start in range(0, max(1, n - WINDOW_LENGTH + 1), STRIDE):
                window_frame_nums = timeline[start:start + WINDOW_LENGTH]
                if len(window_frame_nums) < WINDOW_LENGTH:
                    continue
                last_idx = start + WINDOW_LENGTH - 1

                missing_in_window = [fn for fn in window_frame_nums if subject not in frames_by_num.get(fn, {})]
                # Same fix as scan_gaps() -- see its comment for the full
                # mechanism. Subject-presence at each selected position
                # doesn't guarantee those positions are numerically
                # contiguous; a full-clip annotation gap can be straddled
                # by a window while every individual frame still passes
                # the presence check.
                non_contiguous = (window_frame_nums[-1] - window_frame_nums[0]) != (WINDOW_LENGTH - 1)
                window_ok = len(missing_in_window) == 0 and not non_contiguous

                context_frame_paths = []
                context_head_boxes = []
                other_people_boxes = []
                is_child_value = None
                if window_ok:
                    for fn in window_frame_nums:
                        context_frame_paths.append(resolve_image_path(clip["clip_id"], fn))
                        context_head_boxes.append(list(frames_by_num[fn][subject]["head_box"]))
                        others = {
                            pid: ann["head_box"]
                            for pid, ann in frames_by_num[fn].items()
                            if pid != subject
                        }
                        other_people_boxes.append([list(box) for box in others.values()])
                    is_child_value = frames_by_num[window_frame_nums[-1]][subject]["is_child"]

                for target_ms in HORIZON_MS_LIST:
                    horizon_frames, horizon_ms_actual = compute_horizon_frames(clip_fps, target_ms)
                    target_pos = last_idx + horizon_frames
                    row_valid = window_ok
                    skip_reason = None
                    target_frame_path = None
                    target_gaze_x = target_gaze_y = None
                    target_gaze_type = None
                    raw_gaze_class = None

                    if not window_ok:
                        if missing_in_window:
                            skip_reason = (
                                f"mid-window annotation gap for subject "
                                f"({len(missing_in_window)} frame(s) missing)"
                            )
                        else:
                            skip_reason = (
                                f"non-contiguous timeline span (window spans "
                                f"{window_frame_nums[-1] - window_frame_nums[0] + 1} "
                                f"real frames, not {WINDOW_LENGTH} -- a full-clip "
                                f"annotation gap sits inside this window)"
                            )
                    elif target_pos >= n:
                        row_valid = False
                        skip_reason = "target frame beyond clip end"
                    else:
                        target_frame_num = timeline[target_pos]
                        # FIX (2026-07-18, round 2): same class of bug as the
                        # context-window fix above, in a second location the
                        # first fix didn't cover. target_pos is a LIST
                        # POSITION, not a verified real-frame offset -- if a
                        # full-clip gap sits between the window's end and the
                        # target's timeline position, target_frame_num can
                        # land further away than horizon_frames actually
                        # means, exactly like the context-window case. Verify
                        # the REAL frame-number gap matches horizon_frames
                        # exactly, using window_frame_nums[-1] (now guaranteed
                        # correct by the window_ok check above) as the
                        # trustworthy reference point.
                        real_gap = target_frame_num - window_frame_nums[-1]
                        if real_gap != horizon_frames:
                            row_valid = False
                            skip_reason = (
                                f"non-contiguous timeline span between context and "
                                f"target (real gap is {real_gap} frames, not "
                                f"horizon_frames={horizon_frames} -- a full-clip "
                                f"annotation gap sits between them)"
                            )
                        else:
                            target_people = frames_by_num.get(target_frame_num, {})
                            if subject not in target_people:
                                row_valid = False
                                skip_reason = "target frame missing (subject not annotated)"
                            else:
                                target_ann = target_people[subject]
                                raw_gaze_class = target_ann["gaze_class"]
                                head_boxes = {pid: a["head_box"] for pid, a in target_people.items()}
                                gtype, cat_skip = classify_target(
                                    target_ann["gaze_class"], target_ann["gaze"], subject, head_boxes
                                )
                                target_frame_path = resolve_image_path(clip["clip_id"], target_frame_num)
                                if gtype is not None:
                                    target_gaze_type = gtype
                                    if target_ann["gaze"] is not None:
                                        target_gaze_x, target_gaze_y = target_ann["gaze"]
                                else:
                                    row_valid = False
                                    skip_reason = cat_skip  # e.g. "gaze_shift", "inside_occluded", ...

                    manifest_rows.append({
                        "source_dataset": "childplay",
                        "clip_id": clip["clip_id"],
                        "subject_id": subject,
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
                        "raw_gaze_class": raw_gaze_class,
                        "is_child": is_child_value,
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
    print(f"\n  Invalid breakdown:")
    for reason, n in sorted(stats.items(), key=lambda x: -x[1]):
        if reason in ("valid", "invalid"):
            continue
        print(f"    {n:>6}  {reason}")
    print(f"\n  Manifest written to: {OUTPUT_MANIFEST_JSONL}")
    return manifest_rows


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    print("Scanning clips and applying clip-quality filter (should_include)...")
    qualifying_clips = get_qualifying_clips()
    print(f"\nQualifying clips: {len(qualifying_clips)}\n")

    if not qualifying_clips:
        print("ERROR: no qualifying clips found. Check CHILDPLAY_ROOT and "
              "clip-quality thresholds.")
        return

    scan_gaps(qualifying_clips)

    print("Proceeding to Phase 2 (window construction) using default policy: "
          "invalidate on any mid-window gap, missing horizon target, or "
          "non-targetable gaze_class at the target frame.\n")
    build_windows(qualifying_clips)

    print("\nDone. Recommended next step: read childplay_temporal_gap_report.csv "
          "and the invalid breakdown above before treating "
          "childplay_temporal_manifest.jsonl as final. MIN_SOCIAL_PCT has "
          "already been retuned (0.20 -> 0.0) based on the first real run; "
          "if MAX_NONTARGETABLE_PCT is now excluding an unexpectedly large "
          "fraction with more clips in the population, that would be the next "
          "retuning signal worth a real look, not a bug to chase.")


if __name__ == "__main__":
    main()
