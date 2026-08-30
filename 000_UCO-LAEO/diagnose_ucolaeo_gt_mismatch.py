"""
diagnose_ucolaeo_gt_mismatch.py — traces WHY mtgs_teacher_ucolaeo.jsonl
and labels_ucolaeo_full.jsonl disagree on gt_x/gt_y for 14,203 of 18,726
shared rows (76%), confirmed via compute_ucolaeo_mtgs_ade.py. Both
files' GT should trace back to the identical underlying annotation data
-- a 76% disagreement rate means the ADE comparison built on them isn't
trustworthy until this is understood.

METHOD: for a few real mismatched keys, print both files' own gt_x/gt_y
and gt_px_x/gt_px_y, plus the manifest's own pixel-space GT for that
same key -- then back-calculate what image width/height each file's
normalization implicitly assumed (pixel_value / normalized_value). If
those implied dimensions differ between the two files, that pinpoints
the mismatch as a normalization-consistency issue, not a data-integrity
one -- worth knowing before assuming anything more serious.
"""

import csv
import json

MTGS_PATH = "mtgs_teacher_ucolaeo.jsonl"
INTERNVL3_PATH = "labels_ucolaeo_full.jsonl"
MANIFEST_PATH = "frame_manifest_ucolaeo.csv"

# Real mismatched keys from the actual run -- override below if you
# want to check a different one.
DEFAULT_KEYS = [
    ("mr12", "000152.jpg", "4"),
    ("twd14", "000170.jpg", "3"),
    ("sv27", "000005.jpg", "1"),
]


def load_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["show"], r["fname"], r["subject"])
            rows[key] = r
    return rows


def load_manifest(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["show"], row["fname"], row["subject"])
            rows[key] = row
    return rows


def investigate(key, mtgs, internvl3, manifest):
    print(f"\n{'='*70}\nInvestigating key: {key}\n{'='*70}")
    m = mtgs.get(key)
    i = internvl3.get(key)
    man = manifest.get(key)

    if m is None or i is None:
        print("  Key missing from one of the two files -- different problem, stop here.")
        return

    print("mtgs_teacher_ucolaeo.jsonl:")
    print(f"  gt_x={m['gt_x']:.6f}, gt_y={m['gt_y']:.6f}")
    print(f"  gt_px_x={m.get('gt_px_x')}, gt_px_y={m.get('gt_px_y')}")
    print(f"  img_path={m.get('img_path')}")

    print("\nlabels_ucolaeo_full.jsonl:")
    print(f"  gt_x={i['gt_x']:.6f}, gt_y={i['gt_y']:.6f}")
    print(f"  gt_px_x={i.get('gt_px_x')}, gt_px_y={i.get('gt_px_y')}")

    if man is None:
        print("\nWARNING: key not found in frame_manifest_ucolaeo.csv -- "
              "can't cross-check against pixel-space GT for this one.")
        return

    print("\nframe_manifest_ucolaeo.csv (pixel-space GT, source of truth):")
    print(f"  gaze_x={man['gaze_x']}, gaze_y={man['gaze_y']}")

    manifest_gx, manifest_gy = float(man["gaze_x"]), float(man["gaze_y"])

    # Back-calculate: what image width/height would make each file's
    # own gt_x/gt_y consistent with the manifest's pixel-space value?
    for label, row in (("MTGS", m), ("InternVL3", i)):
        gx, gy = row["gt_x"], row["gt_y"]
        implied_w = manifest_gx / gx if gx else float("nan")
        implied_h = manifest_gy / gy if gy else float("nan")
        print(f"  Implied image size from {label}'s gt_x/gt_y: "
              f"({implied_w:.1f}, {implied_h:.1f})")


def main():
    mtgs = load_rows(MTGS_PATH)
    internvl3 = load_rows(INTERNVL3_PATH)
    manifest = load_manifest(MANIFEST_PATH)

    for key in DEFAULT_KEYS:
        investigate(key, mtgs, internvl3, manifest)

    print(f"\n{'='*70}")
    print("If the 'implied image size' lines above disagree between MTGS "
          "and InternVL3 for the same key, that confirms a normalization-"
          "consistency mismatch between the two pipelines -- and whichever "
          "one disagrees with the REAL image dimensions (check by opening "
          "the actual file) is the one that's wrong.")


if __name__ == "__main__":
    main()
