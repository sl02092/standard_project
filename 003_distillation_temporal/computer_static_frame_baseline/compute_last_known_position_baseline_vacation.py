"""
compute_last_known_position_baseline_vacation.py — VACATION oracle-baseline
lookup, matching the ChildPlay/UCO-LAEO versions' get_subject_gaze(clip_id,
subject, abs_frame) / parse_frame_number(path) signatures exactly, so
train_temporal_poc's compute_baseline_computable_mask() needs only an
import-line change.

Thin read over vacation_temporal_utils.build_person_timelines()'s own
already-resolved structure, same "reuse, don't re-derive" reasoning as the
UCO-LAEO version -- that function's resolution logic is independently
verified at full scale (206,774-instance run matched
build_vacation_frame_manifest.py's own totals exactly, integer for
integer).

FRAME NUMBERS: VACATION's filenames are '{video_id}_{frame}.jpg' -- unlike
UCO-LAEO's bare '{frame:06d}.jpg', the frame number is the part after the
LAST underscore, not the whole basename stem (video_id itself may already
contain no underscores here, but splitting from the right is robust
either way).
"""

import os

from vacation_temporal_utils import build_person_timelines

_timelines_cache = None


def parse_frame_number(path):
    """'.../11/11_9.jpg' or '11_9.jpg' -> 9."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("_", 1)[-1])


def _get_timelines():
    global _timelines_cache
    if _timelines_cache is None:
        _timelines_cache, _ = build_person_timelines()
    return _timelines_cache


def get_subject_gaze(video_id, person_label, abs_frame):
    """Mirrors get_subject_gaze(clip_id, subject, abs_frame) exactly.
    Returns the resolved (gx, gy) PIXEL-space target for this exact
    (video, person, frame) if attention_focus resolved there, else None.
    person_label is used as-is (already the raw "Person1"-style string,
    no int coercion needed unlike UCO-LAEO's numeric ids)."""
    timelines = _get_timelines()
    t = timelines.get((video_id, person_label))
    if t is None:
        return None
    frame = t["frames"].get(abs_frame)
    if frame is None:
        return None
    return frame["gaze"]  # (x, y) pixel-space, or None if unresolved at this exact frame


def get_subject_gaze_type(video_id, person_label, abs_frame):
    """Sibling to get_subject_gaze() -- returns 'social'/'object'/None for
    the same (video, person, frame), reusing the SAME cached timelines
    rather than rebuilding them. Added to let diagnostics split results
    by gaze_type without needing gaze_type to be threaded through every
    caller's own data structures (e.g. the feature-cache index, which
    doesn't carry it)."""
    timelines = _get_timelines()
    t = timelines.get((video_id, person_label))
    if t is None:
        return None
    frame = t["frames"].get(abs_frame)
    if frame is None:
        return None
    return frame["gaze_type"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python compute_last_known_position_baseline_vacation.py "
              "<video_id> <person_label> <frame_path_or_number>")
        print("Example: python compute_last_known_position_baseline_vacation.py "
              "11 Person1 11_9.jpg")
        sys.exit(1)

    video_id, person_label, frame_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    abs_frame = parse_frame_number(frame_arg) if not frame_arg.isdigit() else int(frame_arg)

    gaze = get_subject_gaze(video_id, person_label, abs_frame)
    print(f"get_subject_gaze({video_id!r}, {person_label!r}, {abs_frame}) -> {gaze}")
    if gaze is None:
        print("(None is a valid answer -- means this subject's attention_focus "
              "wasn't resolved at this exact frame, e.g. NA or target absent.)")
