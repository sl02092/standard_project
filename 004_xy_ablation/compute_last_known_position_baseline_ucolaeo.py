"""
compute_last_known_position_baseline_ucolaeo.py — UCO-LAEO oracle-baseline
lookup, matching compute_last_known_position_baseline_childplay.py's own
get_subject_gaze(clip_id, subject, abs_frame) / parse_frame_number(path)
signatures exactly, so train_temporal_poc_childplay.py's own
compute_baseline_computable_mask() needs only an import-line change to
work against UCO-LAEO -- no logic changes there.

SIMPLER THAN THE CHILDPLAY VERSION, AND WHY: ChildPlay's version had to
re-parse annotation CSVs itself to answer "was this subject's gaze
resolved at this exact frame". UCO-LAEO doesn't need that --
ucolaeo_temporal_utils.build_person_timelines() already resolved exactly
this lookup (video_id, person_id) -> {frame_num: {"gaze": ..., ...}} as
part of building the temporal manifest in the first place. This module
is a thin, cached read over that existing structure, not a fresh
computation -- if the two ever disagree, that's a real bug worth
investigating, not two independent sources of truth to keep in sync.

FRAME NUMBERS: UCO-LAEO's frame filenames ARE the frame number
("000046.jpg" -> 46) -- no video-relative-to-clip-absolute arithmetic
needed the way ChildPlay's "{video}_{absolute_frame}.jpg" convention
required. parse_frame_number() here is correspondingly simpler, and
deliberately kept independent of ucolaeo_temporal_utils.parse_video_and_frame()
(which returns (video_id, frame_num) as a pair) since call sites here
only ever have a bare frame path, not a full video/frame tuple.
"""

import os

from ucolaeo_temporal_utils import build_person_timelines

_timelines_cache = None


def parse_frame_number(path):
    """'.../got01/000046.jpg' or '000046.jpg' -> 46."""
    return int(os.path.splitext(os.path.basename(path))[0])


def _get_timelines():
    """Built once, reused across every get_subject_gaze() call --
    build_person_timelines() reads all three H5 files, not something to
    redo per-row inside compute_baseline_computable_mask()'s loop."""
    global _timelines_cache
    if _timelines_cache is None:
        _timelines_cache, _ = build_person_timelines()
    return _timelines_cache


def get_subject_gaze(video_id, person_id, abs_frame):
    """Mirrors ChildPlay's get_subject_gaze(clip_id, subject, abs_frame)
    signature exactly (three positional args, same order) -- video_id
    plays ChildPlay's "clip_id" role, since UCO-LAEO's clip_id IS the
    video_id (no finer subdivision, same convention used throughout the
    static manifest and window builder).

    Returns the resolved (gx, gy) PIXEL-space gaze point for this exact
    (video, person, frame) if the mutual LAEO pair was resolved there,
    else None. person_id is coerced to str to match
    build_person_timelines()'s own key convention (bare numeric string,
    e.g. "3" -- consistent with build_ucolaeo_frame_manifest.py's
    str(int(pid))), so this works whether the caller passes an int or a
    str.
    """
    timelines = _get_timelines()
    t = timelines.get((video_id, str(person_id)))
    if t is None:
        return None
    frame = t["frames"].get(abs_frame)
    if frame is None:
        return None
    return frame["gaze"]  # (x, y) pixel-space, or None if unresolved at this exact frame


# ══════════════════════════════════════════════════════════════════════
# ── VALIDATION — run against real data before trusting anything ────────
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python compute_last_known_position_baseline_ucolaeo.py "
              "<video_id> <person_id> <frame_path_or_number>")
        print("Example: python compute_last_known_position_baseline_ucolaeo.py "
              "got01 3 000046.jpg")
        sys.exit(1)

    video_id, person_id, frame_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    abs_frame = parse_frame_number(frame_arg) if not frame_arg.isdigit() else int(frame_arg)

    gaze = get_subject_gaze(video_id, person_id, abs_frame)
    print(f"get_subject_gaze({video_id!r}, {person_id!r}, {abs_frame}) -> {gaze}")
    if gaze is None:
        print("(None is a valid answer -- means this subject's gaze wasn't "
              "resolved at this exact frame, e.g. no active mutual pair.)")
