"""
build_vacation_frame_manifest.py — builds frame_manifest_vacation.csv from
VACATION's raw NewAnt_{video_id}.txt annotation files.

WHY THIS NEEDS NEW LOGIC, NOT A CHILDPLAY PORT: ChildPlay's raw CSV already
has a direct (gaze_x, gaze_y) coordinate per row -- classify_target() there
only classifies an already-given point, it never resolves one from
scratch. VACATION's attention_focus is CATEGORICAL ("P2", "O1", "NA") --
resolving it into a coordinate means looking up the referenced entity's
OWN box in the SAME frame and taking its centroid. Confirmed as the right
algorithm days ago by tracing NewAnt_106.txt's Person1/Person2/Object1
state progression by hand (single -> follow -> share -> mutual).

ROW FORMAT (confirmed directly, awk NF check against real files):
  Object rows: 10 tokens -- bbx_id xmin ymin xmax ymax frame_id lost
    occluded generated label (no event/atomic/attention_focus -- an
    object doesn't look at anything).
  Person rows: 13 tokens -- same first 9 fields + label, event_attribute,
    atomic_attribute, attention_focus.

RESOLUTION RULE:
  attention_focus == "NA"        -> excluded (no usable target at all,
                                     same "invalidate, never pad"
                                     treatment as every other dataset's
                                     genuinely-unresolvable rows)
  attention_focus == "P{n}"      -> target = Person{n}'s box centroid,
                                     gaze_type = "social"
  attention_focus == "O{n}"      -> target = Object{n}'s box centroid,
                                     gaze_type = "object"
  referenced entity not present in this frame, or itself lost==1
                                  -> excluded (can't resolve a target
                                     against a box that doesn't exist)
  this row's own lost==1         -> excluded (this person's own box in
                                     this frame is untrustworthy, not
                                     just the target's)

NO OFFSCREEN CONCEPT: same as UCO-LAEO -- VATIC-style annotation only
covers entities that are actually being tracked. use_teacher is always
True for every row that survives the exclusions above.

SPLIT: official train_list/val_list/test_list (237 unique video IDs,
readme.txt) used as-is. The 62 additional annotated-but-unofficial videos
(238-301, confirmed a contiguous continuation of the original split, not
scattered) are folded into train only, per 2026-07-29 discussion --
doesn't touch val/test, so anything wanting to stay comparable to the
original published benchmark still can. Videos 255 and 291 have source
video but no annotation file (confirmed via the real tree listing) --
naturally absent since this script only processes videos it finds an
annotation file for.

IMAGE PATH: {images_root}/{video_id}/{video_id}_{frame}.jpg -- matches
extract_frames_vacation.py's own output convention exactly (confirmed
against the real uploaded tree listing), so no path-reconstruction
guessing needed here.
"""

import os
import re
import glob
import csv

ANNOTATIONS_ROOT = os.environ.get(
    "VACATION_ANNOTATIONS_ROOT",
    "/parallel_scratch/sl02092/standard_project/data/vacation/annotation",
)
HPC_IMAGES_ROOT = os.environ.get(
    "VACATION_HPC_IMAGES_ROOT",
    "/parallel_scratch/sl02092/standard_project/data/vacation/images",
)
README_PATH = os.environ.get(
    "VACATION_README_PATH",
    os.path.join(ANNOTATIONS_ROOT, "readme.txt"),
)
OUTPUT_PATH = os.environ.get("VACATION_MANIFEST_OUT", "frame_manifest_vacation.csv")

MANIFEST_COLUMNS = [
    "show", "clip", "fname", "subject",
    "head_x1", "head_y1", "head_x2", "head_y2",
    "gaze_x", "gaze_y", "gaze_type", "use_teacher", "img_path", "split",
]


# ══════════════════════════════════════════════════════════════════════
# ── SPLIT ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_official_split(readme_path):
    """Returns {video_id: 'train'/'val'/'test'} for the 237 officially-
    listed videos. Anything not in this dict is one of the 62 extra
    videos -- caller folds those into train."""
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


# ══════════════════════════════════════════════════════════════════════
# ── PARSING ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def parse_annotation_file(path):
    """Returns {frame_id: {label: {...}}}. label is the raw bbx_label
    string ("Person1", "Object1", ...) -- used directly as the lookup
    key for attention_focus resolution, not re-encoded."""
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
    """Returns (gaze_x, gaze_y, gaze_type) or None if unresolvable
    (NA, referenced entity absent from this frame, or referenced entity
    itself lost)."""
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
# ── BUILD ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_manifest():
    split_map = load_official_split(README_PATH)
    print(f"Official split loaded: {len(split_map)} video IDs")

    ann_files = sorted(
        glob.glob(os.path.join(ANNOTATIONS_ROOT, "NewAnt_*.txt")),
        key=lambda p: int(re.search(r"NewAnt_(\d+)\.txt", p).group(1)),
    )
    print(f"Found {len(ann_files)} annotation files")

    rows = []
    stats = {"total_person_rows": 0, "excluded_lost": 0, "excluded_unresolvable": 0,
              "kept": 0, "extra_videos_to_train": 0}
    gaze_type_totals = {"social": 0, "object": 0}

    for ann_path in ann_files:
        video_id = re.search(r"NewAnt_(\d+)\.txt", ann_path).group(1)
        split = split_map.get(video_id)
        if split is None:
            split = "train"  # one of the 62 extra videos
            stats["extra_videos_to_train"] += 1

        frames = parse_annotation_file(ann_path)

        for frame_id, entities in sorted(frames.items()):
            for label, entity in entities.items():
                if not label.startswith("Person"):
                    continue  # objects never have a gaze target of their own
                stats["total_person_rows"] += 1

                if entity["lost"] == 1:
                    stats["excluded_lost"] += 1
                    continue

                resolved = resolve_target(entity, entities)
                if resolved is None:
                    stats["excluded_unresolvable"] += 1
                    continue
                gaze_x, gaze_y, gaze_type = resolved

                x1, y1, x2, y2 = entity["box"]
                fname = f"{video_id}_{frame_id}.jpg"
                img_path = os.path.join(HPC_IMAGES_ROOT, video_id, fname)

                rows.append({
                    "show": video_id, "clip": video_id, "fname": fname,
                    "subject": label,
                    "head_x1": round(x1), "head_y1": round(y1),
                    "head_x2": round(x2), "head_y2": round(y2),
                    "gaze_x": round(gaze_x), "gaze_y": round(gaze_y),
                    "gaze_type": gaze_type, "use_teacher": True,
                    "img_path": img_path, "split": split,
                })
                gaze_type_totals[gaze_type] += 1
                stats["kept"] += 1

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    print(f"\nVideos processed: {len(ann_files)} ({stats['extra_videos_to_train']} "
          f"extra/unofficial, folded into train)")
    print(f"Total person-rows scanned : {stats['total_person_rows']}")
    print(f"Excluded (lost==1)        : {stats['excluded_lost']}")
    print(f"Excluded (unresolvable target: NA / target absent / target lost): "
          f"{stats['excluded_unresolvable']}")
    print(f"Kept                      : {stats['kept']}")
    print(f"gaze_type totals          : {gaze_type_totals}")
    print(f"By split                  : {split_counts}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_manifest()
