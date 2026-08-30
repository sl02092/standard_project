"""
ucolaeo_temporal_utils.py — Foundational layer for the UCO-LAEO temporal
manifest builder, mirroring childplay_annotation_utils.py's role: small,
independently-verifiable building blocks the window builder sits on top
of, checked against real data here first.

SOURCE: the three H5 files (ucolaeo_temporal_manifest_schema.md has the
full reasoning for using these instead of the raw got0X.txt files).

PIXEL VS NORMALIZED: the H5 stores normalized [0,1] coordinates, but
ChildPlay's own temporal manifest stores PIXEL-space boxes/targets
(confirmed directly against real manifest rows -- e.g. context_head_boxes
values in the hundreds, not 0-1), with normalization deferred to
TemporalWindowDataset's own dimension-resolution step at __getitem__
time. Matched here for consistency: this module resolves H5 normalized
coordinates to real pixel values (via a real per-video image open,
cached), same as build_ucolaeo_frame_manifest.py already did for the
static manifest.

FPS: real, per-video, individually measured (uco-laeo_fps.txt) -- video
IDs there come from source filenames ("Got01.mp4") while the H5's own
video_id convention is lowercase ("got01") -- confirmed case mismatch,
normalized to lowercase here rather than assumed to already match.

CROSS-SPLIT DEDUPLICATION (added 2026-08-02): the same (video_id,
person_id) key can genuinely appear in more than one of the three H5
files -- confirmed for real (2,858 such keys found at the manifest
level; see dedupe_ucolaeo_cross_split.py and
build_ucolaeo_frame_manifest.py's own matching fix for the full
investigation). This function's last-write-wins dict-overwrite on
timelines[key] already resolved this correctly, but only because the
literal split iteration order ("train","val","test") happened to
already match the required test>val>train priority -- an accident of
tuple order, not a guaranteed property. SPLIT_PRIORITY and the
assertion in build_person_timelines() make that dependency explicit and
fail loudly if it's ever broken, rather than silently reintroducing the
duplication this project already found and fixed once.
"""

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = higher priority --
                                                       # matches
                                                       # dedupe_ucolaeo_cross_split.py
                                                       # and
                                                       # build_ucolaeo_frame_manifest.py's
                                                       # own rule exactly.

import os
import re
from collections import defaultdict

import pandas as pd
from PIL import Image

H5_DIR = os.environ.get("UCOLAEO_H5_DIR", ".")
FPS_FILE = os.environ.get("UCOLAEO_FPS_FILE", "uco-laeo_fps.txt")
LOCAL_IMAGES_ROOT = os.environ.get(
    "UCOLAEO_LOCAL_IMAGES_ROOT", r"C:\repo\UCO-LAEO\ucolaeodb\frames"
)


def load_fps_map(fps_path=FPS_FILE):
    """Returns {video_id: fps}, e.g. {'got01': 23.98, ...}. Fails loud on
    any unparseable line rather than silently skipping it."""
    fps_map = {}
    with open(fps_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"([^\\/:]+)\.mp4:\s*([\d.]+)\s*FPS", line)
            if not m:
                raise ValueError(f"Could not parse fps line: {line!r}")
            fps_map[m.group(1).lower()] = float(m.group(2))
    return fps_map


def parse_video_and_frame(h5_path):
    """'frames/got01/000046.jpg' -> ('got01', 46)."""
    parts = h5_path.split("/")
    video_id = parts[-2]
    frame_num = int(os.path.splitext(parts[-1])[0])
    return video_id, frame_num


_dim_cache = {}


def get_video_dimensions(video_id, images_root=LOCAL_IMAGES_ROOT, sample_frame_path=None):
    """Real pixel dimensions, cached per video -- same discipline as
    every other per-clip/per-video dimension cache in this project, not
    assumed from a reference constant."""
    if video_id not in _dim_cache:
        try:
            with Image.open(sample_frame_path) as img:
                _dim_cache[video_id] = img.size
        except Exception as e:
            print(f"  [ucolaeo_temporal] could not read dimensions for {video_id}: {e}")
            _dim_cache[video_id] = None
    return _dim_cache[video_id]


def build_person_timelines(h5_dir=H5_DIR, images_root=LOCAL_IMAGES_ROOT):
    """Reads all three H5 splits, returns:
      timelines: {(video_id, person_id): {"split": str,
                   "frames": {frame_num: {"head_box": (x1,y1,x2,y2) px,
                                            "gaze": (x,y) px or None,
                                            "gaze_type": "social" or None}}}}
      frame_people: {(video_id, frame_num): {person_id: (x1,y1,x2,y2) px}}
        -- every real person's box in that exact frame, for
        other_people_boxes lookups.
    Pixel conversion uses each video's REAL dimensions (opened once per
    video from its first-encountered frame), not assumed.
    """
    split_order = ("train", "val", "test")
    assert list(split_order) == sorted(split_order, key=lambda s: -SPLIT_PRIORITY[s]), (
        f"split_order {split_order} no longer matches the required "
        f"test>val>train priority (SPLIT_PRIORITY={SPLIT_PRIORITY}) -- "
        f"this function's correctness for cross-split-duplicated "
        f"(video_id, person_id) keys depends on this exact order, since "
        f"it relies on last-write-wins dict overwrite rather than an "
        f"explicit per-key check. Fix the order, or restructure this "
        f"function to check SPLIT_PRIORITY explicitly per key instead."
    )

    timelines = defaultdict(lambda: {"split": None, "frames": {}})
    frame_people = defaultdict(dict)
    stats = defaultdict(int)
    keys_seen = set()

    for split in split_order:
        path = os.path.join(h5_dir, f"uco_laeo_{split}.h5")
        df = pd.read_hdf(path, "data")
        for _, row in df.iterrows():
            video_id, frame_num = parse_video_and_frame(row["path"])
            img_path = os.path.join(images_root, video_id, os.path.basename(row["path"]))
            dims = get_video_dimensions(video_id, images_root, img_path)
            if dims is None:
                stats["skipped_no_dims"] += 1
                continue
            w, h = dims

            for pid, box, gp, io in zip(row["person_ids"], row["head_bboxes"],
                                          row["gaze_points"], row["inout"]):
                if pid == -1:
                    continue
                pid = str(int(pid))
                key = (video_id, pid)

                if key in keys_seen and timelines[key]["split"] != split:
                    # A LATER split (per split_order, asserted above to be
                    # priority-ascending) is about to overwrite this
                    # key's data from an EARLIER, lower-priority split --
                    # confirmed real, not hypothetical. Counted for
                    # visibility; the overwrite itself is already correct
                    # given the assertion above.
                    stats["cross_split_duplicate_instances"] += 1
                keys_seen.add(key)

                x1, y1, x2, y2 = box
                # Cast to native float -- H5 values are numpy.float32,
                # not JSON-serializable as-is (caught by actually running
                # the window builder end-to-end, not assumed safe).
                head_box_px = (float(x1 * w), float(y1 * h), float(x2 * w), float(y2 * h))
                frame_people[(video_id, frame_num)][pid] = head_box_px

                key = (video_id, pid)
                timelines[key]["split"] = split
                stats["total_person_instances"] += 1

                if io == 1.0:
                    gx, gy = gp
                    gaze_px = (float(gx * w), float(gy * h))
                    timelines[key]["frames"][frame_num] = {
                        "head_box": head_box_px, "gaze": gaze_px, "gaze_type": "social",
                    }
                    stats["resolved"] += 1
                else:
                    timelines[key]["frames"][frame_num] = {
                        "head_box": head_box_px, "gaze": None, "gaze_type": None,
                    }
                    stats["unresolved"] += 1

    print(f"  Person-instances scanned: {stats['total_person_instances']}")
    print(f"  Resolved (social target) : {stats['resolved']}")
    print(f"  Unresolved (present, no target): {stats['unresolved']}")
    print(f"  Skipped (no image dims)  : {stats['skipped_no_dims']}")
    print(f"  Cross-split duplicate instances (later split's data kept, "
          f"per assertion above): {stats['cross_split_duplicate_instances']}")
    return dict(timelines), dict(frame_people)


# ══════════════════════════════════════════════════════════════════════
# ── VALIDATION — run against real data before trusting anything ────────
# ══════════════════════════════════════════════════════════════════════

def validate_against_real_video(video_id, person_id):
    fps_map = load_fps_map()
    print(f"fps for {video_id}: {fps_map.get(video_id, 'NOT FOUND')}")

    timelines, frame_people = build_person_timelines()
    key = (video_id, person_id)
    if key not in timelines:
        print(f"ERROR: {key} not found in any H5 split")
        return
    t = timelines[key]
    frames = sorted(t["frames"].keys())
    print(f"\n{key}: split={t['split']}, {len(frames)} frames, "
          f"range {frames[0]}-{frames[-1]}")

    resolved = [f for f in frames if t["frames"][f]["gaze_type"] is not None]
    print(f"  {len(resolved)}/{len(frames)} frames have a resolved target")

    f0 = frames[0]
    print(f"\n  First frame ({f0}): {t['frames'][f0]}")
    print(f"  Other people in that same frame: {frame_people.get((video_id, f0), {})}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python ucolaeo_temporal_utils.py <video_id> <person_id>")
        print('Example: python ucolaeo_temporal_utils.py got01 3')
        sys.exit(1)
    validate_against_real_video(sys.argv[1], sys.argv[2])
