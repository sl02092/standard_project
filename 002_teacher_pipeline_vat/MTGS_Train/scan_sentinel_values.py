#!/usr/bin/env python3
"""General-purpose sentinel-value scanner for this project's teacher
pipeline JSONL files. Checks whether gt_x/gt_y/pred_x/pred_y ever fall
OUTSIDE the normalized [0,1] range while still being treated as real
data -- i.e. present, non-null, but not a genuine coordinate.

Catches BOTH known sentinel conventions in this project:
  - ChildPlay/VACATION/UCO-LAEO (H5-native): raw (-1, -1), exact.
  - VAT: gt_px_x=-1 divided by img_w -> a small negative fraction like
    -1/1920, NOT exactly -1.0 -- an exact-match check would miss this
    entirely, which is why this scans for "out of [0,1]" generally
    rather than one hardcoded value.

Works on ANY of: labels_*_full.jsonl, labels_*_hybrid_targeted.jsonl,
mtgs_teacher_*.jsonl. Reports counts broken down by label_source and
gaze_type (if those fields exist in the file) so a real pattern is easy
to spot vs. isolated noise.

Usage:
    python3 scan_sentinel_values.py <file.jsonl>

    # e.g.
    python3 scan_sentinel_values.py mtgs_teacher_vat.jsonl
    python3 scan_sentinel_values.py mtgs_teacher_vat_test.jsonl
"""
import sys
import json
from collections import Counter

FIELDS = ("gt_x", "gt_y", "pred_x", "pred_y")


def is_out_of_range(v):
    return v is not None and (v < 0.0 or v > 1.0)


def main():
    path = sys.argv[1]

    n_total = 0
    hits_by_field = {f: [] for f in FIELDS}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            for field in FIELDS:
                if is_out_of_range(r.get(field)):
                    hits_by_field[field].append(r)

    print(f"Total rows: {n_total}")
    print(f"Checking for values outside normalized [0,1] range\n")

    any_hits = False
    for field, rows in hits_by_field.items():
        if not rows:
            print(f"{field}: 0 hits")
            continue
        any_hits = True
        print(f"{field}: {len(rows)} hits")

        by_source = Counter(r.get("label_source", "?") for r in rows)
        by_gaze = Counter(r.get("gaze_type", "?") for r in rows)
        print(f"  by label_source: {dict(by_source)}")
        print(f"  by gaze_type: {dict(by_gaze)}")

        vals = [r[field] for r in rows]
        print(f"  value range: min={min(vals):.4f} max={max(vals):.4f}")

        if field == "gt_x":
            both_gt = sum(1 for r in rows if is_out_of_range(r.get("gt_x")) and is_out_of_range(r.get("gt_y")))
            print(f"  rows where BOTH gt_x and gt_y are out of range: {both_gt}")

    if not any_hits:
        print("\nNo out-of-range values found -- clean.")
    else:
        print("\nOut-of-range values found -- inspect the breakdown above. If "
              "gt_x/gt_y hits are NOT all tagged with an appropriate "
              "gaze_type (e.g. 'offscreen') or an excluded label_source, "
              "this is the same class of bug found in ChildPlay's export.")


if __name__ == "__main__":
    main()

