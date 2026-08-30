#!/usr/bin/env python3
"""General-purpose sentinel-value scanner for this project's teacher
pipeline JSONL files. Checks whether gt_x/gt_y/pred_x/pred_y ever equal a
suspicious placeholder value (default -1.0) while still being treated as
real data -- i.e. present, non-null, but a sentinel.

Works on ANY of: labels_*_full.jsonl, labels_*_hybrid_targeted.jsonl,
mtgs_teacher_*.jsonl. Reports counts broken down by label_source and
gaze_type (if those fields exist in the file) so a real pattern is easy
to spot vs. isolated noise.

Usage:
    python3 scan_sentinel_values.py <file.jsonl> [sentinel_value]

    # e.g.
    python3 scan_sentinel_values.py mtgs_teacher_ucolaeo.jsonl
    python3 scan_sentinel_values.py mtgs_teacher_vat.jsonl
    python3 scan_sentinel_values.py mtgs_teacher_vat_test.jsonl
    python3 scan_sentinel_values.py labels_ucolaeo_full.jsonl
"""
import sys
import json
from collections import Counter

FIELDS = ("gt_x", "gt_y", "pred_x", "pred_y")


def main():
    path = sys.argv[1]
    sentinel = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0

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
                v = r.get(field)
                if v is not None and v == sentinel:
                    hits_by_field[field].append(r)

    print(f"Total rows: {n_total}")
    print(f"Checking for sentinel value: {sentinel}\n")

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

        both_gt = sum(1 for r in rows if r.get("gt_x") == sentinel and r.get("gt_y") == sentinel)
        if field == "gt_x":
            print(f"  rows where BOTH gt_x and gt_y == sentinel: {both_gt}")

    if not any_hits:
        print("\nNo sentinel values found -- clean.")
    else:
        print("\nSentinel values found -- inspect the breakdown above. If "
              "gt_x/gt_y hits are NOT all tagged with an appropriate "
              "gaze_type (e.g. 'offscreen') or an excluded label_source, "
              "this is the same class of bug found in ChildPlay's export.")


if __name__ == "__main__":
    main()
