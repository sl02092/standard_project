#!/usr/bin/env python3
"""Restricts multiple teacher label files to the INTERSECTION of rows
where every teacher genuinely succeeded (not a fallback), then computes
each teacher's pooled ADE_norm on that identical shared population.

This is for the standalone "which teacher is more accurate" comparison
table -- NOT for student training data, which correctly uses every row
(including gt_fallback) already and needs no changes.

Edit the TEACHERS list below for whichever comparison you're running --
e.g. VAT's 3-way (InternVL3-alone, hybrid, MTGS), or a 2-way subset.

IMPORTANT: the "success_sources" list for each teacher is my best
understanding of this project's label_source conventions, NOT verified
against real data yet:
    - InternVL3-alone: "teacher"
    - Hybrid: "teacher" (social frames, no GDINO involved) AND
      "teacher_hybrid" (object frames, GDINO succeeded)
    - MTGS: every row in its own file counts as "attempted" -- but a row
      is only a fair comparison point if its GT is real, not a sentinel
      (-1,-1 for ChildPlay/VACATION/UCO-LAEO's H5-native convention, or
      a small negative fraction for VAT's normalized convention -- see
      GT_SENTINEL_CHECK below).
Confirm the hybrid label_source values against a real file before
trusting this output -- run:
    python3 -c "import json,collections; print(collections.Counter(json.loads(l)['label_source'] for l in open('labels_hybrid_targeted.jsonl')))"
and check the actual values present.

Usage:
    python3 build_intersection_comparison.py
"""
import json
import math
from collections import Counter

# ---- EDIT THIS FOR THE COMPARISON YOU'RE RUNNING ----
TEACHERS = [
    {
        "name": "InternVL3-alone",
        "path": "labels_full.jsonl",
        "success_sources": {"teacher"},
    },
    {
        "name": "Hybrid",
        "path": "labels_hybrid_targeted.jsonl",
        "success_sources": {"teacher", "teacher_hybrid"},
    },
    {
        "name": "MTGS",
        "path": "mtgs_teacher_vat.jsonl",
        "success_sources": None,  # every row in the file counts as attempted
    },
]
# -------------------------------------------------------


def gt_is_sentinel(r):
    gt_x, gt_y = r.get("gt_x"), r.get("gt_y")
    if gt_x is None or gt_y is None:
        return True
    return gt_x < 0.0 or gt_x > 1.0 or gt_y < 0.0 or gt_y > 1.0


def load_success_rows(spec):
    rows = {}
    with open(spec["path"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if spec["success_sources"] is not None and \
               r.get("label_source") not in spec["success_sources"]:
                continue
            if gt_is_sentinel(r):
                continue
            key = (r.get("show"), r.get("clip"), r.get("subject"), r.get("fname"))
            rows[key] = r
    return rows


def ade(rows_dict, keys):
    dists = []
    for k in keys:
        r = rows_dict[k]
        dists.append(math.hypot(r["pred_x"] - r["gt_x"], r["pred_y"] - r["gt_y"]))
    return sum(dists) / len(dists) if dists else None


def main():
    per_teacher_rows = {}
    for spec in TEACHERS:
        rows = load_success_rows(spec)
        per_teacher_rows[spec["name"]] = rows
        print(f"{spec['name']} ({spec['path']}): {len(rows)} genuine-success, "
              f"real-GT rows")

    key_sets = [set(rows.keys()) for rows in per_teacher_rows.values()]
    intersection = set.intersection(*key_sets)
    print(f"\nIntersection (all {len(TEACHERS)} teachers succeeded, real GT): "
          f"{len(intersection)} rows")

    print(f"\n--- Pooled ADE_norm on the SAME {len(intersection)}-row population ---")
    for spec in TEACHERS:
        val = ade(per_teacher_rows[spec["name"]], intersection)
        print(f"  {spec['name']}: {val:.4f}" if val is not None else
              f"  {spec['name']}: n/a")

    # Breakdown of gaze_type within the intersection, for sanity -- should
    # roughly match the dataset's known real-world class balance.
    gaze_types = Counter()
    first_teacher_rows = next(iter(per_teacher_rows.values()))
    for k in intersection:
        gaze_types[first_teacher_rows[k].get("gaze_type")] += 1
    print(f"\ngaze_type distribution within intersection: {dict(gaze_types)}")


if __name__ == "__main__":
    main()
