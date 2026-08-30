"""
build_ucolaeo_frame_manifest.py — builds frame_manifest_ucolaeo.csv from
the three MTGS H5 files (uco_laeo_train/val/test.h5), in the exact schema
step40/step46 expect.

WHY THE H5, NOT THE RAW got0X.txt/pair_got0X.txt FILES: verified directly
(2026-07-29) against real got01 data that the H5's person_ids are a
genuine cross-frame identity tracking on top of the raw per-frame head
list (not just the raw file's positional ordering carried through
unchanged) -- confirmed by solving for the real image size and matching
boxes frame-by-frame. Re-deriving this ourselves from the raw files would
mean re-implementing that same tracking logic; the H5 already has it
done and independently confirmed correct.

GAZE_TYPE: always "social", never "object" or "offscreen" -- this
dataset's schema has no concept of either. Every valid target comes from
a resolved mutual LAEO pair (inout=1); there is no gaze-at-object
annotation and no gaze-off-camera annotation, only "resolvable" or "not".

INOUT=-1 ROWS ARE DROPPED, NOT MAPPED TO "offscreen": confirmed directly
against real data (frame got01/000046.jpg) that inout=-1 includes
genuinely on-screen, clearly visible people with real head boxes -- they
simply aren't in an active mutual pair at that moment. Labelling them
"offscreen" would assert something false (that we know they're looking
off-camera) rather than the true state (we don't know where they're
looking at all). These rows carry no usable target and are excluded
entirely, same treatment as VAT/ChildPlay's genuinely-invalid categories
(never padded with an invented value).

SHOW/CLIP: no finer subdivision exists below the video level for this
dataset (unlike VAT's show/clip or ChildPlay's video/clip split) -- both
fields are set to the video_id (e.g. "got01"). Kept as two fields rather
than one for schema consistency with step40/step46's expected columns,
not because there's a real distinction here.

SPLIT: taken directly from which H5 file a row came from (train/val/test)
-- this IS the dataset's own official split, not a custom shuffle.

IMAGE DIMENSIONS: read via real PIL opens, cached per video folder (same
"assume constant resolution within one video, verify don't assume"
discipline as ChildPlay's baseline script) -- NOT assumed as a fixed
1280x720 reference. That value was confirmed correct for got01
specifically during verification; other videos are not assumed to match.

"NEGATIVES" VIDEOS: negatives01-negatives30 are, by construction, videos
with no real LAEO pairs -- expect these to contribute few or zero usable
rows after inout=-1 filtering. Not a bug if the printed per-video summary
shows zero for these; that's the correct behaviour for a negative example
with no resolvable target to output.

CROSS-SPLIT DEDUPLICATION (added 2026-08-02): the same (video_id, fname,
subject) key can genuinely appear in more than one of the three H5 files
-- confirmed for real against a from-scratch run of this exact script,
not hypothetical (2,858 such keys found, e.g. video got05 present in
both train and test). The original version of this script appended every
occurrence without checking, silently duplicating 2,858 rows in the
output manifest -- caught and fixed downstream via
dedupe_ucolaeo_cross_split.py, but that fix only cleaned the files that
already existed at the time; without this change, a fresh rebuild from
the raw H5s would reintroduce the exact same duplication. Explicit
test>val>train priority now applied here, matching
dedupe_ucolaeo_cross_split.py's own rule exactly, so a rebuild agrees
with the already-cleaned files rather than silently diverging from them.
"""

import os
import pandas as pd
from PIL import Image
from collections import Counter

HPC_IMAGES_ROOT = os.environ.get(
    "UCOLAEO_HPC_IMAGES_ROOT",
    "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames",
)
H5_DIR = os.environ.get("UCOLAEO_H5_DIR", ".")
OUTPUT_PATH = os.environ.get("UCOLAEO_MANIFEST_OUT", "frame_manifest_ucolaeo.csv")

MANIFEST_COLUMNS = [
    "show", "clip", "fname", "subject",
    "head_x1", "head_y1", "head_x2", "head_y2",
    "gaze_x", "gaze_y", "gaze_type", "use_teacher", "img_path", "split",
]

# lower = higher priority -- matches dedupe_ucolaeo_cross_split.py's own
# rule exactly. See module docstring's "CROSS-SPLIT DEDUPLICATION" note.
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}

_dim_cache = {}  # video_id -> (w, h) or None


def get_video_dimensions(img_path, video_id):
    if video_id not in _dim_cache:
        try:
            with Image.open(img_path) as img:
                _dim_cache[video_id] = img.size
        except Exception as e:
            print(f"  [ucolaeo] could not read dimensions via {img_path}: {e}")
            _dim_cache[video_id] = None
    return _dim_cache[video_id]


def parse_video_and_fname(h5_path):
    """'frames/got01/000046.jpg' -> ('got01', '000046.jpg')"""
    parts = h5_path.split("/")
    return parts[-2], parts[-1]


def build_manifest():
    rows_by_key = {}  # (show, fname, subject) -> row dict -- see module
                       # docstring's "CROSS-SPLIT DEDUPLICATION" note.
                       # Was a plain list in the pre-2026-08-02 version;
                       # that's exactly what let duplicates through.
    stats = Counter()
    per_video_kept = Counter()

    all_videos_seen = set()
    for split in ("train", "val", "test"):
        h5_path = os.path.join(H5_DIR, f"uco_laeo_{split}.h5")
        df = pd.read_hdf(h5_path, "data")
        print(f"Loaded {h5_path}: {len(df)} frames")

        for _, r in df.iterrows():
            video_id, fname = parse_video_and_fname(r["path"])
            all_videos_seen.add(video_id)
            img_path = os.path.join(HPC_IMAGES_ROOT, video_id, fname)

            dims = get_video_dimensions(img_path, video_id)
            if dims is None:
                stats["skipped_no_dims"] += len(r["person_ids"])
                continue
            w, h = dims

            for pid, box, gp, io in zip(r["person_ids"], r["head_bboxes"],
                                          r["gaze_points"], r["inout"]):
                if pid == -1:
                    continue  # padding slot
                stats["total_person_instances"] += 1
                if io != 1.0:
                    stats["excluded_no_target"] += 1
                    continue

                x1, y1, x2, y2 = box
                gx, gy = gp
                subject = str(int(pid))
                key = (video_id, fname, subject)

                # CROSS-SPLIT DEDUPLICATION -- explicit priority check,
                # not reliance on which order this row happened to be
                # processed in. Confirmed real: 2,858 such keys exist
                # (see module docstring).
                if key in rows_by_key:
                    if SPLIT_PRIORITY[split] >= SPLIT_PRIORITY[rows_by_key[key]["split"]]:
                        continue  # existing row already has equal/higher priority
                    stats["cross_split_duplicates_resolved"] += 1
                else:
                    stats["kept"] += 1
                    per_video_kept[video_id] += 1

                rows_by_key[key] = {
                    "show": video_id, "clip": video_id, "fname": fname,
                    "subject": subject,
                    "head_x1": round(float(x1) * w), "head_y1": round(float(y1) * h),
                    "head_x2": round(float(x2) * w), "head_y2": round(float(y2) * h),
                    "gaze_x": round(float(gx) * w), "gaze_y": round(float(gy) * h),
                    "gaze_type": "social",
                    "use_teacher": True,
                    "img_path": img_path,
                    "split": split,
                }

    rows = list(rows_by_key.values())

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal person-instances scanned : {stats['total_person_instances']}")
    print(f"Excluded (no resolvable target) : {stats['excluded_no_target']}")
    print(f"Excluded (image dims unreadable): {stats['skipped_no_dims']}")
    print(f"Cross-split duplicates resolved : {stats['cross_split_duplicates_resolved']} "
          f"(same key present in >1 H5 file -- kept only the highest-"
          f"priority split's row, test>val>train)")
    print(f"Kept (written to manifest)      : {stats['kept']}")
    print(f"Manifest written to: {OUTPUT_PATH}")

    zero_contributing = sorted(all_videos_seen - set(per_video_kept))
    negatives_kept = sum(c for v, c in per_video_kept.items() if v.startswith("negatives"))
    print(f"\nVideos scanned total          : {len(all_videos_seen)}")
    print(f"Videos contributing zero rows  : {len(zero_contributing)} "
          f"(expected to include most/all 'negatives*' videos)")
    print(f"Rows from 'negatives*' videos  : {negatives_kept} "
          f"(expected near-zero -- these are negative LAEO examples by construction)")
    n_negatives_zero = sum(1 for v in zero_contributing if v.startswith("negatives"))
    n_negatives_total = sum(1 for v in all_videos_seen if v.startswith("negatives"))
    print(f"'negatives*' videos contributing zero rows: {n_negatives_zero}/{n_negatives_total}")


if __name__ == "__main__":
    build_manifest()
