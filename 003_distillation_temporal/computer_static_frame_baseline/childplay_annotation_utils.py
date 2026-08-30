"""
childplay_annotation_utils.py — Foundational parsing/classification layer
for the ChildPlay temporal manifest builder (Objective 4 extension).

WHAT THIS IS: the verified building blocks the actual window-builder will
be built on top of — CSV loading, frame-path resolution, and gaze-category
classification. Deliberately NOT the full window-builder (VAT's own
version, step_temporal_window_builder_vat.py, is 600+ lines) — built as a
separate, smaller, independently-verifiable layer first, so a bug here
gets caught against one real clip rather than discovered mid-way through
a full-scale run. See childplay_temporal_manifest_schema.md for the full
design this implements.

ONE OPEN QUESTION, DOESN'T BLOCK THIS LAYER: whether bbox_x/y/width/height
in the raw CSV are pixel coordinates or already normalized [0,1] -- the
README's "in the image frame" phrasing doesn't settle it. Doesn't matter
for classify_target() below (point-in-box comparisons work identically
either way, since gaze point and head boxes are always in the same units).
Matters for the NEXT layer (writing normalized coordinates to the
manifest) -- run the validation block at the bottom against one real clip
and check the printed bbox values before building that layer.

FRAME-PATH FORMULA: confirmed against real H5 data (see schema doc) --
absolute_frame = clip_start_frame + relative_frame - 1. Only valid for
non-downsampled clips; resolve_image_path() raises rather than silently
guessing if called on one of the 9 excluded clips.
"""

import os
import csv
from collections import defaultdict

CHILDPLAY_ROOT = os.environ.get(
    "CHILDPLAY_ROOT",
    r"C:\repo\ChildPlay-gaze",
)
ANNOTATIONS_ROOT = os.path.join(CHILDPLAY_ROOT, "annotations")
IMAGES_ROOT = os.path.join(CHILDPLAY_ROOT, "images")
CLIPS_CSV_PATH = os.path.join(CHILDPLAY_ROOT, "clips.csv")

# Confirmed real values from the ACTUAL shipped CSVs (not just the
# README, which turned out to be incomplete) -- fail loudly on anything
# else rather than silently mishandling an unexpected category.
# `not_annotated`, `inside_ambiguous`, and bare `uncertain` were NOT in
# the README's documented six; found via a full-dataset scan (2026-07-16)
# after `not_annotated` first surfaced a ValueError here. `uncertain` and
# `inside_uncertain` are plausibly related (an earlier/legacy label vs. a
# later, more specific one -- see the paper's supplementary definition of
# a bare "uncertain" category) but this is NOT confirmed, and they are
# deliberately kept as separate categories below rather than merged on
# that guess. All three new categories combined are 883/257,927 (0.34%)
# of the full dataset -- low-stakes either way.
VALID_GAZE_CLASSES = {
    "inside_visible", "outside_frame", "gaze_shift",
    "inside_occluded", "inside_uncertain", "eyes_closed",
    "inside_ambiguous", "uncertain", "not_annotated",
}

# The categories that invalidate a frame as a TARGET (present, but no
# usable point) -- same "invalidate, never pad" treatment as VAT, each
# kept as its own distinct skip_reason rather than merged into a generic
# "invalid" bucket.
INVALID_TARGET_CLASSES = {
    "inside_occluded", "inside_uncertain", "eyes_closed",
    "gaze_shift", "inside_ambiguous", "uncertain", "not_annotated",
}


# ══════════════════════════════════════════════════════════════════════
# ── CLIP METADATA ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_clip_metadata(clips_csv_path=CLIPS_CSV_PATH):
    """Returns {clip_id: {video_id, channel_id, split, frame_count, fps,
    resolution, downsampled}}. `downsampled` is derived from the clip_id
    string itself (contains "-downsampled"), not a separate column."""
    metadata = {}
    with open(clips_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["clip"]
            metadata[clip_id] = {
                "video_id": row["video_id"],
                "channel_id": row["channel_id"],
                "split": row["split"],
                "frame_count": int(row["frame_count"]),
                "fps": float(row["fps"]),
                "resolution": row["resolution"],
                "downsampled": clip_id.endswith("-downsampled"),
            }
    return metadata


def parse_clip_start_frame(clip_id):
    """'DGls-Z4e6OI_8031-8107' -> (video_id='DGls-Z4e6OI', start=8031, end=8107).
    Splits on the LAST underscore, not the first -- YouTube video IDs can
    themselves contain underscores/hyphens, so a naive first-split would
    misparse some clips."""
    base = clip_id[: -len("-downsampled")] if clip_id.endswith("-downsampled") else clip_id
    video_id, frame_range = base.rsplit("_", 1)
    start_str, end_str = frame_range.split("-")
    return video_id, int(start_str), int(end_str)


def resolve_image_path(clip_id, relative_frame, images_root=IMAGES_ROOT):
    """CONFIRMED formula (see schema doc): absolute = clip_start + relative - 1.
    Raises on downsampled clips rather than guessing -- their step size was
    never verified, and they're excluded from the pipeline anyway."""
    if clip_id.endswith("-downsampled"):
        raise ValueError(
            f"resolve_image_path called on downsampled clip {clip_id} -- "
            f"these are excluded from the pipeline (unconfirmed frame-step "
            f"formula), not expected to reach this function in normal use."
        )
    video_id, start_frame, _ = parse_clip_start_frame(clip_id)
    absolute_frame = start_frame + relative_frame - 1
    return os.path.join(images_root, clip_id, f"{video_id}_{absolute_frame}.jpg")


# ══════════════════════════════════════════════════════════════════════
# ── ANNOTATION LOADING ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_clip_annotations(clip_id, split, ann_root=ANNOTATIONS_ROOT):
    """Reads one clip's raw CSV. Returns {relative_frame: {person_id: {...}}}.
    head_box stored as (x1, y1, x2, y2) -- converted from the CSV's own
    (bbox_x, bbox_y, bbox_width, bbox_height) upper-left+size convention,
    matching VAT's box representation for downstream consistency."""
    path = os.path.join(ann_root, split, f"{clip_id}.csv")
    frames = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            person_id = row["person_id"]
            gaze_class = row["gaze_class"]
            if gaze_class not in VALID_GAZE_CLASSES:
                raise ValueError(
                    f"Unexpected gaze_class '{gaze_class}' in {path} "
                    f"(frame {frame}, person {person_id}) -- not one of "
                    f"{sorted(VALID_GAZE_CLASSES)}. Stop and check rather "
                    f"than silently mishandling a possibly-new category."
                )
            bbox_x, bbox_y = float(row["bbox_x"]), float(row["bbox_y"])
            bbox_w, bbox_h = float(row["bbox_width"]), float(row["bbox_height"])
            gaze_x, gaze_y = float(row["gaze_x"]), float(row["gaze_y"])
            frames[frame][person_id] = {
                "head_box": (bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h),
                "gaze_class": gaze_class,
                "gaze": (gaze_x, gaze_y) if gaze_class == "inside_visible" else None,
                "is_child": row["is_child"].strip().lower() in ("1", "true"),
            }
    return dict(frames)


# ══════════════════════════════════════════════════════════════════════
# ── CLASSIFICATION ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def point_in_box(point, box):
    """Identical logic to VAT's classifier -- reused, not reimplemented."""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_target(gaze_class, gaze_point, this_person_id, head_boxes_this_frame):
    """Maps one annotation row to (target_gaze_type, skip_reason) -- exactly
    one is non-None. See schema doc's mapping table for the full reasoning,
    especially why gaze_shift (and the other invalid categories) are kept
    distinct from each other rather than merged into one generic reason."""
    if gaze_class == "outside_frame":
        return "offscreen", None
    if gaze_class in INVALID_TARGET_CLASSES:
        return None, gaze_class
    if gaze_class == "inside_visible":
        is_social = any(
            point_in_box(gaze_point, box)
            for other_id, box in head_boxes_this_frame.items()
            if other_id != this_person_id
        )
        return ("social" if is_social else "object"), None
    raise ValueError(
        f"Unhandled gaze_class '{gaze_class}' -- should be unreachable given "
        f"load_clip_annotations' own validation against VALID_GAZE_CLASSES."
    )


# ══════════════════════════════════════════════════════════════════════
# ── VALIDATION — run this against one real clip before trusting anything ──
# ══════════════════════════════════════════════════════════════════════

def validate_against_real_clip(clip_id, split):
    """Loads one real clip and prints enough to sanity-check every
    assumption above by eye: does the image path resolve to a real file,
    are bbox values pixel-scale or normalized-scale, does classification
    produce a sane mix of categories. Prints, doesn't assert -- meant for
    a human to read, not an automated test."""
    print(f"Loading clip metadata...")
    all_meta = load_clip_metadata()
    if clip_id not in all_meta:
        print(f"ERROR: {clip_id} not found in clips.csv")
        return
    meta = all_meta[clip_id]
    print(f"  {clip_id}: fps={meta['fps']}, frame_count={meta['frame_count']}, "
          f"split={meta['split']}, downsampled={meta['downsampled']}")

    print(f"\nLoading annotations for {clip_id}...")
    frames = load_clip_annotations(clip_id, split)
    print(f"  {len(frames)} annotated relative-frame entries")

    first_frame_num = sorted(frames.keys())[0]
    first_frame = frames[first_frame_num]
    print(f"\nFirst annotated frame ({first_frame_num}), "
          f"{len(first_frame)} person(s):")
    for pid, ann in first_frame.items():
        print(f"  person {pid}: head_box={ann['head_box']} "
              f"(CHECK: values <2 => likely normalized, values in the "
              f"hundreds/thousands => likely pixel space)")
        print(f"    gaze_class={ann['gaze_class']}, gaze={ann['gaze']}, "
              f"is_child={ann['is_child']}")

    img_path = resolve_image_path(clip_id, first_frame_num)
    exists = os.path.isfile(img_path)
    print(f"\nResolved image path: {img_path}")
    print(f"  File exists: {exists}  "
          f"{'-- formula confirmed against this clip' if exists else '-- MISMATCH, check the formula/paths before trusting anything else'}")

    print(f"\nClassification over all frames in this clip:")
    from collections import Counter
    type_counts = Counter()
    skip_counts = Counter()
    for frame_num, people in frames.items():
        head_boxes = {pid: ann["head_box"] for pid, ann in people.items()}
        for pid, ann in people.items():
            gtype, skip = classify_target(
                ann["gaze_class"], ann["gaze"], pid, head_boxes
            )
            if gtype:
                type_counts[gtype] += 1
            else:
                skip_counts[skip] += 1
    print(f"  target_gaze_type counts: {dict(type_counts)}")
    print(f"  skip_reason counts     : {dict(skip_counts)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python childplay_annotation_utils.py <clip_id> <split>")
        print('Example: python childplay_annotation_utils.py "DGls-Z4e6OI_8031-8107" train')
        sys.exit(1)
    validate_against_real_clip(sys.argv[1], sys.argv[2])
