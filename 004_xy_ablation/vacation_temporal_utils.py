"""
vacation_temporal_utils.py — Foundational layer for the VACATION temporal
manifest builder, mirroring ucolaeo_temporal_utils.py's role: small,
independently-verifiable building blocks the window builder sits on top
of, checked against real data here first.

SOURCE: raw NewAnt_{video_id}.txt files (vacation_temporal_manifest_schema.md
has the full reasoning for using these instead of frame_manifest_vacation.csv).

parse_annotation_file(), focus_to_label(), resolve_target(), and
load_official_split() below are reused near-verbatim from
build_vacation_frame_manifest.py -- deliberately, not reimplemented from
scratch, since that script's resolution algorithm was independently
hand-verified against real data before this file was written (frame 15 of
NewAnt_11.txt, see schema doc). Re-deriving this logic fresh would risk
quietly diverging from something already confirmed correct.

PIXEL SPACE: VATIC-style annotations are native pixel coordinates
already -- no normalization step here, unlike UCO-LAEO's H5-derived
[0,1] values. Everything in this module stays in pixel space; the
eventual dataset class defers normalization to its own per-video
dimension cache, same point ChildPlay and UCO-LAEO both defer it to.
"""

import os
import re
import glob
from collections import defaultdict

ANNOTATIONS_ROOT = os.environ.get(
    "VACATION_LOCAL_ANNOTATIONS_ROOT", r"C:\repo\VACATION\annotations"
)
IMAGES_ROOT = os.environ.get(
    "VACATION_LOCAL_IMAGES_ROOT", r"C:\repo\VACATION\images"
)
README_PATH = os.environ.get(
    "VACATION_README_PATH", os.path.join(ANNOTATIONS_ROOT, "readme.txt")
)
FPS_FILE = os.environ.get("VACATION_FPS_FILE", "video_fps.txt")

# NOTE: ANNOTATIONS_ROOT/IMAGES_ROOT above are UNCONFIRMED placeholders --
# see schema doc's "genuinely unresolved" section. Override via the env
# vars above, or edit the defaults once the real local paths are known.


def load_fps_map(fps_path=FPS_FILE):
    """Returns {video_id: fps}, e.g. {'11': 25.0, ...}. Fails loud on any
    unparseable line rather than silently skipping it -- same discipline
    as UCO-LAEO's version. Simpler than UCO-LAEO's parser: VACATION's
    video IDs are bare numeric already, no case-mismatch to normalize."""
    fps_map = {}
    with open(fps_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(\d+)\.mp4:\s*([\d.]+)\s*FPS", line)
            if not m:
                raise ValueError(f"Could not parse fps line: {line!r}")
            fps_map[m.group(1)] = float(m.group(2))
    return fps_map


# ══════════════════════════════════════════════════════════════════════
# ── SPLIT — reused verbatim from build_vacation_frame_manifest.py ──────
# ══════════════════════════════════════════════════════════════════════

def load_official_split(readme_path=README_PATH):
    """Returns {video_id: 'train'/'val'/'test'} for the 237 officially-
    listed videos. Anything not in this dict is one of the 62 extra
    videos -- caller folds those into train, matching the static
    pipeline's own decision exactly (see schema doc)."""
    text = open(readme_path, encoding="utf-8").read()

    def extract(name):
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
        return re.findall(r"'(\d+)'", m.group(1))

    split_map = {}
    for vid in extract("train_list"):
        split_map[vid] = "train"
    for vid in extract("val_list"):
        split_map[vid] = "val"
    for vid in extract("test_list"):
        split_map[vid] = "test"
    return split_map


def get_video_split(video_id, split_map):
    """video_id not in split_map -> one of the 62 extra videos -> train.
    Small wrapper so every call site applies this fallback identically,
    rather than each re-writing `split_map.get(video_id, "train")` and
    risking one of them forgetting the fallback."""
    return split_map.get(video_id, "train")


# ══════════════════════════════════════════════════════════════════════
# ── PARSING — reused verbatim from build_vacation_frame_manifest.py ────
# ══════════════════════════════════════════════════════════════════════

def parse_annotation_file(path):
    """Returns {frame_id: {label: {...}}}. label is the raw bbx_label
    string ("Person1", "Object1", ...), used directly as the lookup key
    for attention_focus resolution -- unchanged from the verified static
    version."""
    frames = {}
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) not in (10, 13):
                raise ValueError(
                    f"{path}:{line_num} -- {len(tokens)} tokens, expected 10 "
                    f"(object) or 13 (person). Stop and check rather than "
                    f"silently mishandling an unexpected row shape."
                )
            xmin, ymin, xmax, ymax = (float(tokens[i]) for i in range(1, 5))
            frame_id = int(tokens[5])
            lost, occluded, generated = (int(tokens[i]) for i in range(6, 9))
            label = tokens[9]

            entry = {
                "box": (xmin, ymin, xmax, ymax),
                "lost": lost, "occluded": occluded, "generated": generated,
            }
            if len(tokens) == 13:
                entry["event_attribute"] = tokens[10]
                entry["atomic_attribute"] = tokens[11]
                entry["attention_focus"] = tokens[12]
            else:
                if not label.startswith("Object"):
                    raise ValueError(
                        f"{path}:{line_num} -- 10-token row with label "
                        f"'{label}', expected an Object* label. Stop and check."
                    )

            frames.setdefault(frame_id, {})[label] = entry
    return frames


def focus_to_label(focus):
    """'P2' -> 'Person2', 'O1' -> 'Object1', 'NA' -> None."""
    if focus == "NA":
        return None
    if focus.startswith("P"):
        return f"Person{focus[1:]}"
    if focus.startswith("O"):
        return f"Object{focus[1:]}"
    raise ValueError(f"Unrecognized attention_focus value: '{focus}'")


def resolve_target(entity, entities_this_frame):
    """Returns (gaze_x, gaze_y, gaze_type) in PIXEL space, or None if
    unresolvable (NA, referenced entity absent from this frame, or
    referenced entity itself lost). Unchanged from the verified static
    version -- see schema doc for the hand-verification against real
    data (NewAnt_11.txt frame 15)."""
    focus = entity.get("attention_focus")
    if focus is None or focus == "NA":
        return None
    target_label = focus_to_label(focus)
    target = entities_this_frame.get(target_label)
    if target is None or target["lost"] == 1:
        return None
    x1, y1, x2, y2 = target["box"]
    gaze_type = "social" if target_label.startswith("Person") else "object"
    return (x1 + x2) / 2, (y1 + y2) / 2, gaze_type


# ══════════════════════════════════════════════════════════════════════
# ── CANONICAL TIMELINES — new for the temporal pipeline ────────────────
# ══════════════════════════════════════════════════════════════════════

def build_person_timelines(annotations_root=ANNOTATIONS_ROOT, readme_path=README_PATH):
    """Reads every NewAnt_{video_id}.txt in annotations_root, returns:
      timelines: {(video_id, person_label): {"split": str,
                   "frames": {frame_id: {"head_box": (x1,y1,x2,y2) px,
                                           "gaze": (x,y) px or None,
                                           "gaze_type": "social"/"object"/None}}}}
      frame_people: {(video_id, frame_id): {person_label: (x1,y1,x2,y2) px}}
        -- every real PERSON's box in that exact frame (objects excluded
        -- this mirrors ChildPlay/UCO-LAEO's "other_people_boxes" schema
        field, which is about other people specifically, not objects),
        for other_people_boxes lookups, self excluded by the caller.

    Unlike UCO-LAEO, gaze_type here can be "social" OR "object" -- a real
    difference this dataset's temporal work can actually make use of,
    unlike UCO-LAEO's 100%-social-by-construction case.

    A person's own lost==1 rows are skipped entirely (no head_box, no
    timeline entry for that frame) -- their own box is untrustworthy in
    that frame, same exclusion build_vacation_frame_manifest.py applies.
    """
    split_map = load_official_split(readme_path)

    ann_files = sorted(
        glob.glob(os.path.join(annotations_root, "NewAnt_*.txt")),
        key=lambda p: int(re.search(r"NewAnt_(\d+)\.txt", p).group(1)),
    )
    if not ann_files:
        raise FileNotFoundError(
            f"No NewAnt_*.txt files found in {annotations_root} -- check "
            f"VACATION_LOCAL_ANNOTATIONS_ROOT is set to the real local path "
            f"(see schema doc's 'genuinely unresolved' section)."
        )

    timelines = defaultdict(lambda: {"split": None, "frames": {}})
    frame_people = defaultdict(dict)
    stats = defaultdict(int)

    for ann_path in ann_files:
        video_id = re.search(r"NewAnt_(\d+)\.txt", ann_path).group(1)
        split = get_video_split(video_id, split_map)
        frames = parse_annotation_file(ann_path)

        for frame_id, entities in frames.items():
            for label, entity in entities.items():
                if not label.startswith("Person"):
                    continue  # objects never have their own timeline
                if entity["lost"] == 1:
                    stats["excluded_lost"] += 1
                    continue

                stats["total_person_instances"] += 1
                head_box_px = entity["box"]
                frame_people[(video_id, frame_id)][label] = head_box_px

                key = (video_id, label)
                timelines[key]["split"] = split

                resolved = resolve_target(entity, entities)
                if resolved is not None:
                    gaze_x, gaze_y, gaze_type = resolved
                    timelines[key]["frames"][frame_id] = {
                        "head_box": head_box_px,
                        "gaze": (gaze_x, gaze_y),
                        "gaze_type": gaze_type,
                    }
                    stats[f"resolved_{gaze_type}"] += 1
                else:
                    timelines[key]["frames"][frame_id] = {
                        "head_box": head_box_px, "gaze": None, "gaze_type": None,
                    }
                    stats["unresolved"] += 1

    print(f"  Videos scanned            : {len(ann_files)}")
    print(f"  Person-instances scanned  : {stats['total_person_instances']}")
    print(f"  Excluded (own lost==1)    : {stats['excluded_lost']}")
    print(f"  Resolved (social target)  : {stats['resolved_social']}")
    print(f"  Resolved (object target)  : {stats['resolved_object']}")
    print(f"  Unresolved (no target)    : {stats['unresolved']}")
    return dict(timelines), dict(frame_people)


# ══════════════════════════════════════════════════════════════════════
# ── VALIDATION — run against real data before trusting anything ────────
# ══════════════════════════════════════════════════════════════════════

def validate_against_real_video(video_id, person_label):
    fps_map = load_fps_map()
    print(f"fps for {video_id}: {fps_map.get(video_id, 'NOT FOUND')}")

    timelines, frame_people = build_person_timelines()
    key = (video_id, person_label)
    if key not in timelines:
        print(f"ERROR: {key} not found -- check video_id/person_label and "
              f"that this video's annotation file was actually scanned.")
        return
    t = timelines[key]
    frames = sorted(t["frames"].keys())
    print(f"\n{key}: split={t['split']}, {len(frames)} frames, "
          f"range {frames[0]}-{frames[-1]}")

    resolved = [f for f in frames if t["frames"][f]["gaze_type"] is not None]
    print(f"  {len(resolved)}/{len(frames)} frames have a resolved target")

    f0 = frames[0]
    print(f"\n  First frame ({f0}): {t['frames'][f0]}")
    print(f"  Other people in that same frame: "
          f"{ {k: v for k, v in frame_people.get((video_id, f0), {}).items() if k != person_label} }")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python vacation_temporal_utils.py <video_id> <person_label>")
        print("Example: python vacation_temporal_utils.py 11 Person1")
        sys.exit(1)
    validate_against_real_video(sys.argv[1], sys.argv[2])
