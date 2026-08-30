"""
build_childplay_flat_labels.py — Derive a VAT-schema flat static label file
(labels_childplay_gt.jsonl) from the already-built ChildPlay temporal manifest,
joined back to the raw per-clip annotation CSVs only for what the manifest
doesn't carry (the target frame's own head box).

WHY THIS EXISTS: ChildPlay never had a flat, per-frame-subject static label
file the way VAT has labels_hybrid_targeted.jsonl -- only the temporal window
manifest (childplay_temporal_manifest.jsonl, ~367MB) and the raw annotation
CSVs (annotations/{split}/{clip_id}.csv). The manifest already contains the
one genuinely new, nontrivial piece of derived logic (social/object gaze-type
classification via the point-in-headbox classifier) and the bbox-corner
conversion -- this script extracts and reshapes what's already there rather
than re-deriving any of it, and only touches the raw CSVs for the one field
the manifest never carried: the TARGET frame's own head box (the manifest
only stores head boxes for the 8 CONTEXT frames of each window, since the
target frame is a coordinate label only, never model input).

SOURCE OF TRUTH FOR EACH PIECE:
  - gaze_type (social/object/offscreen), gaze coordinates: temporal manifest
    (already computed, not re-derived)
  - clip metadata (split, fps, resolution): clips.csv (confirmed via direct
    check: 401/401 rows agree with splits.csv on split assignment, and
    clips.csv also carries fps/resolution which splits.csv lacks -- using
    clips.csv alone rather than both)
  - target frame's head box: raw per-clip annotation CSV (the one join this
    script actually performs)

NULL/OFFSCREEN MAPPING (confirmed against real manifest rows with real
outside_frame examples, not guessed):
  target_gaze_type == "social" | "object"  -> normal row, real coordinates
  target_gaze_type == "offscreen"          -> gt_px_x/gt_px_y = -1 (VAT's
                                               sentinel; the manifest itself
                                               uses null, not -1 -- this
                                               script introduces the -1
                                               explicitly at this step)
  window_valid == False (gaze_shift, inside_occluded, inside_uncertain,
    eyes_closed)                            -> row excluded entirely, never
                                               written -- matches VAT's own
                                               "no fallback labels, ever"
                                               policy, extended here.

RESOLUTION HANDLING: clips.csv only gives a categorical "720p"/"1080p"
string, not literal pixel dimensions -- confirmed via direct check this
covers 100% of clips (219 x 1080p, 182 x 720p, no other values). Standard
16:9 convention assumed (1080p=1920x1080, 720p=1280x720) but NOT trusted
blindly -- every resolved coordinate is bounds-checked against the assumed
frame size before being written, same discipline as every other pixel-
normalization step in this project. A violation is logged and the row is
SKIPPED, never silently written with an out-of-range normalized coordinate
(the exact failure mode already found and fixed twice in the VAT pipeline
this session -- the 1280x720 and 1,1 image-dimension fallback bugs).

DOWNSAMPLED CLIPS: the 9 clips at fps=60 (confirmed via direct check: this
is exactly the "9 downsampled clips, both from 60fps source" the schema doc
already flagged) are excluded entirely, matching the schema doc's own
decision -- not re-litigated here.

CROSS-CHECKS PERFORMED PER ROW (not blind trust of either source):
  - the raw CSV's own gaze_class must match the manifest's raw_gaze_class
    for the same (clip, frame, subject) -- mismatch is logged and the row
    is skipped, not silently written.
  - resolved pixel coordinates (head box corners, gaze target) must fall
    within [0, width] x [0, height] of the assumed resolution -- violation
    logged and skipped.
  - relative frame number must not exceed the clip's own frame_count in
    clips.csv -- sanity bound, violation logged and skipped.

NOT YET DONE, FLAGGED FOR THE CANONICAL DISTILLATION SCRIPT, NOT THIS ONE:
  step52b's own train/val split re-shuffles clips randomly -- for ChildPlay
  (and any joint VAT+ChildPlay run that wants to respect ChildPlay's own
  official split), that split logic needs to read the `split` field this
  script writes into every row, not re-derive its own. This script writes
  the field; nothing downstream honours it yet.

USAGE:
    python build_childplay_flat_labels.py \
        --manifest /path/to/childplay_temporal_manifest.jsonl \
        --clips_csv /path/to/clips.csv \
        --annotations_dir /path/to/ChildPlay/annotations \
        --images_root "C:\\repo\\ChildPlay-gaze\\images" \
        --out labels_childplay_gt.jsonl

FIRST RUN SHOULD BE TREATED AS A VALIDATION PASS, NOT TRUSTED BLINDLY:
this has been built carefully against confirmed schema facts, real sample
rows, and a real raw CSV -- but has not been run end-to-end against the
full 367MB manifest or the complete set of annotation CSVs (neither is
available in this environment). Spot-check a sample of output rows against
their source CSV rows by hand before this feeds into any distillation run,
same discipline as everything else in this project.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════
# ── CONFIG / CONSTANTS ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

RESOLUTION_MAP = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
}

# Confirmed via direct check against clips.csv: exactly these two values,
# 401/401 clips covered. Anything else encountered at runtime is a real
# surprise, not a silently-guessed default -- see resolve_resolution().

CLIP_ID_PATTERN = re.compile(r"^(.+)_(\d+)-(\d+)$")
# video_id confirmed always 11 chars in clips.csv, but this regex doesn't
# rely on that -- it just takes the trailing _start-end off the end,
# whatever precedes it. Matches the README's own stated naming convention.

FRAME_FILENAME_PATTERN = re.compile(r"_(\d+)\.jpg$")


# ══════════════════════════════════════════════════════════════════════
# ── CLIP METADATA (clips.csv) ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_clip_metadata(clips_csv_path):
    """
    Returns {clip_id: {video_id, split, fps, frame_count, width, height}}.
    Clips with fps==60 (the 9 downsampled clips) are dropped here, at the
    source -- matches the schema doc's decision, applied once rather than
    re-checked at every row downstream.
    """
    meta = {}
    n_downsampled_dropped = 0
    n_unknown_resolution = 0
    with open(clips_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fps = int(row["fps"])
            if fps == 60:
                n_downsampled_dropped += 1
                continue
            res = RESOLUTION_MAP.get(row["resolution"])
            if res is None:
                n_unknown_resolution += 1
                print(f"  [!] Unknown resolution '{row['resolution']}' for "
                      f"clip {row['clip']} -- skipping this clip entirely "
                      f"rather than guessing dimensions.", file=sys.stderr)
                continue
            width, height = res
            meta[row["clip"]] = {
                "video_id": row["video_id"],
                "split": row["split"],
                "fps": fps,
                "frame_count": int(row["frame_count"]),
                "width": width,
                "height": height,
            }
    print(f"Loaded metadata for {len(meta)} clips "
          f"(dropped {n_downsampled_dropped} downsampled @60fps, "
          f"{n_unknown_resolution} unknown resolution).")
    return meta


# ══════════════════════════════════════════════════════════════════════
# ── FRAME NUMBER RESOLUTION ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def parse_clip_start_frame(clip_id):
    """'1Ab4vLMMAbY_2354-2439' -> 2354. Confirmed pattern against real
    clip_ids in clips.csv and the manifest sample rows."""
    m = CLIP_ID_PATTERN.match(clip_id)
    if not m:
        return None
    return int(m.group(2))


def parse_absolute_frame_from_path(frame_path):
    """Extracts the absolute-in-source-video frame number from an image
    filename, e.g. '...\\1Ab4vLMMAbY_2363.jpg' -> 2363. Windows-style paths
    in the manifest (as uploaded) -- os.path.basename alone won't split
    backslashes correctly on a non-Windows host, so this regexes the
    filename directly regardless of path separator style."""
    m = FRAME_FILENAME_PATTERN.search(frame_path)
    if not m:
        return None
    return int(m.group(1))


def resolve_relative_frame(clip_id, target_frame_path, clip_start_frame):
    """
    absolute_frame = clip_start_frame + relative_frame - 1 (confirmed
    formula, childplay_temporal_manifest_schema.md, verified against real
    H5 data). Inverted here to get the relative frame number the raw CSV
    actually indexes by.
    """
    abs_frame = parse_absolute_frame_from_path(target_frame_path)
    if abs_frame is None:
        return None
    return abs_frame - clip_start_frame + 1


# ══════════════════════════════════════════════════════════════════════
# ── RAW CSV LOOKUP (per-clip, cached -- many manifest rows share a clip) ──
# ══════════════════════════════════════════════════════════════════════

class RawCsvCache:
    """
    Loads a clip's raw annotation CSV once, indexes by (frame, person_id)
    for fast repeated lookup. Manifest rows are processed in whatever order
    they stream in, not grouped by clip -- caching avoids re-reading the
    same CSV file many times for windows that share a clip.
    """
    def __init__(self, annotations_dir, clip_meta):
        self.annotations_dir = annotations_dir
        self.clip_meta = clip_meta
        self._cache = {}  # clip_id -> {(frame, person_id): row_dict}

    def get_row(self, clip_id, frame, person_id):
        if clip_id not in self._cache:
            self._cache[clip_id] = self._load(clip_id)
        return self._cache[clip_id].get((frame, str(person_id)))

    def _load(self, clip_id):
        meta = self.clip_meta.get(clip_id)
        if meta is None:
            return {}
        path = os.path.join(self.annotations_dir, meta["split"], f"{clip_id}.csv")
        index = {}
        if not os.path.exists(path):
            print(f"  [!] Annotation CSV not found for {clip_id}: {path}",
                  file=sys.stderr)
            return index
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                index[(int(row["frame"]), row["person_id"])] = row
        return index


# ══════════════════════════════════════════════════════════════════════
# ── MANIFEST STREAMING + DEDUP ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def extract_unique_targets(manifest_path):
    """
    Streams the (large) temporal manifest once. Every window whose target
    is usable (window_valid == True) contributes one candidate flat row.
    The same (clip, subject, frame) is very likely hit by multiple windows
    (different horizons, overlapping context spans) -- deduped here, first
    occurrence wins (all should agree on gaze_type/coords; a later
    cross-check would need to compare duplicates for disagreement, not
    done here since window_valid targets are only ever a snapshot of the
    same underlying raw annotation and shouldn't legitimately differ).
    """
    seen = {}
    n_rows_scanned = 0
    n_valid = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows_scanned += 1
            rec = json.loads(line)
            if not rec.get("window_valid"):
                continue
            n_valid += 1
            key = (rec["clip_id"], rec["subject_id"], rec["target_frame_path"])
            if key not in seen:
                seen[key] = rec
    print(f"Manifest scan: {n_rows_scanned} rows, {n_valid} valid-target "
          f"rows, {len(seen)} unique (clip, subject, frame) targets after dedup.")
    return list(seen.values())


# ══════════════════════════════════════════════════════════════════════
# ── ROW CONVERSION ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def convert_one(rec, clip_meta, csv_cache, stats):
    clip_id = rec["clip_id"]
    subject_id = rec["subject_id"]

    meta = clip_meta.get(clip_id)
    if meta is None:
        stats["skip_no_clip_meta"] += 1
        return None  # downsampled or unknown-resolution clip, already dropped

    clip_start = parse_clip_start_frame(clip_id)
    if clip_start is None:
        stats["skip_bad_clip_id"] += 1
        return None

    rel_frame = resolve_relative_frame(clip_id, rec["target_frame_path"], clip_start)
    if rel_frame is None or rel_frame < 1 or rel_frame > meta["frame_count"]:
        stats["skip_bad_frame_number"] += 1
        return None

    raw_row = csv_cache.get_row(clip_id, rel_frame, subject_id)
    if raw_row is None:
        stats["skip_no_raw_csv_row"] += 1
        return None

    # Cross-check: manifest and raw CSV must agree on gaze_class for this
    # exact (clip, frame, subject) -- not blind trust of either source.
    if raw_row["gaze_class"] != rec.get("raw_gaze_class"):
        stats["skip_gaze_class_mismatch"] += 1
        return None

    width, height = meta["width"], meta["height"]

    # Head box: x,y,w,h -> corners, confirmed conversion (matches the
    # manifest's own already-converted context_head_boxes exactly, checked
    # against raw CSV for the same frame earlier this session).
    bx, by = float(raw_row["bbox_x"]), float(raw_row["bbox_y"])
    bw, bh = float(raw_row["bbox_width"]), float(raw_row["bbox_height"])
    x1, y1, x2, y2 = bx, by, bx + bw, by + bh

    def in_bounds(px, py):
        return -1 <= px <= width + 1 and -1 <= py <= height + 1
        # +-1px slack for float rounding at the frame edge, not a
        # meaningfully looser check than exact bounds.

    if not (in_bounds(x1, y1) and in_bounds(x2, y2)):
        stats["skip_head_box_out_of_bounds"] += 1
        return None

    gaze_type = rec["target_gaze_type"]

    if gaze_type == "offscreen":
        gt_px_x, gt_px_y = -1, -1
        gt_x_norm, gt_y_norm = None, None
        pred_x_norm, pred_y_norm = None, None
    elif gaze_type in ("social", "object"):
        gt_px_x = float(rec["target_gaze_x"])
        gt_px_y = float(rec["target_gaze_y"])
        if not in_bounds(gt_px_x, gt_px_y):
            stats["skip_gaze_target_out_of_bounds"] += 1
            return None
        gt_x_norm = gt_px_x / width
        gt_y_norm = gt_px_y / height
        pred_x_norm, pred_y_norm = gt_x_norm, gt_y_norm
        # GT row convention matches VAT's own step40 exactly: pred == gt
        # for label_source=="gt" rows.
    else:
        stats["skip_unexpected_gaze_type"] += 1
        return None

    stats["written"] += 1
    return {
        "source_dataset": "childplay",
        "show": meta["video_id"],
        "clip": clip_id,
        "fname": os.path.basename(rec["target_frame_path"].replace("\\", "/")),
        "subject": str(subject_id),
        "img_path": rec["target_frame_path"],
        "frame_idx": rel_frame,
        "head_x1": round(x1), "head_y1": round(y1),
        "head_x2": round(x2), "head_y2": round(y2),
        "gaze_type": gaze_type if gaze_type != "offscreen" else "offscreen",
        "label_source": "gt",
        "teacher_version": None,
        "pred_x": pred_x_norm, "pred_y": pred_y_norm,
        "gt_x": gt_x_norm, "gt_y": gt_y_norm,
        "gt_px_x": gt_px_x, "gt_px_y": gt_px_y,
        "confidence": "GT",
        "raw_response": None,
        # ChildPlay-specific passthrough fields -- extra to VAT's schema,
        # harmless (step52-family scripts only read known fields), and
        # exactly what the "canonical script split logic" TODO above needs
        # once that's addressed there, not here.
        "split": meta["split"],
        "is_child": rec.get("is_child"),
    }


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="path to childplay_temporal_manifest.jsonl (~367MB)")
    ap.add_argument("--clips_csv", required=True, help="path to clips.csv")
    ap.add_argument("--annotations_dir", required=True,
                     help="path to ChildPlay/annotations/ (contains train/val/test subfolders)")
    ap.add_argument("--out", default="labels_childplay_gt.jsonl")
    args = ap.parse_args()

    clip_meta = load_clip_metadata(args.clips_csv)
    csv_cache = RawCsvCache(args.annotations_dir, clip_meta)

    targets = extract_unique_targets(args.manifest)

    stats = defaultdict(int)
    n_written = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for rec in targets:
            row = convert_one(rec, clip_meta, csv_cache, stats)
            if row is not None:
                out_f.write(json.dumps(row) + "\n")
                n_written += 1

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Rows written: {n_written}")
    for k, v in sorted(stats.items()):
        if k != "written":
            print(f"  {k}: {v}")
    gaze_types = defaultdict(int)
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            gaze_types[json.loads(line)["gaze_type"]] += 1
    print(f"\nGaze type breakdown: {dict(gaze_types)}")
    print(f"\nOutput: {args.out}")
    print("\nREMINDER: spot-check a sample of these rows against their "
          "source CSV rows by hand before this feeds any distillation run.")


if __name__ == "__main__":
    main()
