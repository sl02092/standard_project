#!/usr/bin/env python3
"""Quick verification of mtgs_teacher_vacation.jsonl post gaze_type-fix export.
Pure stdlib -- run with plain python3, no venv_mtgs or GPU needed.

Usage:
    python3 check_vacation_mtgs_output.py /path/to/mtgs_teacher_vacation.jsonl
    python3 check_vacation_mtgs_output.py /path/to/mtgs_teacher_vacation.jsonl /path/to/labels_vacation_full.jsonl
        (2nd arg optional -- if given, also computes InternVL3-alone ADE
        on the matching population for a direct side-by-side comparison)
"""
import sys
import json
import math
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ade(rows):
    if not rows:
        return None
    total = sum(math.hypot(r["pred_x"] - r["gt_x"], r["pred_y"] - r["gt_y"]) for r in rows)
    return total / len(rows)


def main():
    mtgs_path = sys.argv[1]
    rows = load_jsonl(mtgs_path)
    print(f"Total rows: {len(rows)}")

    counts = Counter(r.get("gaze_type") for r in rows)
    print("\ngaze_type distribution:")
    for k, v in counts.most_common():
        print(f"  {k}: {v} ({100 * v / len(rows):.1f}%)")

    pooled = ade(rows)
    print(f"\nMTGS pooled ADE_norm (all rows): {pooled:.4f}")
    for gt in ("social", "object", "offscreen"):
        subset = [r for r in rows if r.get("gaze_type") == gt]
        if subset:
            print(f"  {gt} only (n={len(subset)}): {ade(subset):.4f}")

    dupe_counter = Counter((r["show"], r["fname"], r["subject"]) for r in rows)
    dupe_keys = {k: v for k, v in dupe_counter.items() if v > 1}
    print(f"\nDuplicate (show, fname, subject) keys: {len(dupe_keys)}")
    if dupe_keys:
        example = next(iter(dupe_keys.items()))
        print(f"  example: {example}")

    if len(sys.argv) > 2:
        internvl3_path = sys.argv[2]
        iv3_rows = []
        with open(internvl3_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (
                    r.get("label_source") == "teacher"
                    and r.get("pred_x") is not None
                    and r.get("gt_x") is not None
                ):
                    iv3_rows.append(r)
        print(f"\nInternVL3-alone comparison population "
              f"(label_source=='teacher', has pred+gt): {len(iv3_rows)}")
        print(f"InternVL3-alone pooled ADE_norm: {ade(iv3_rows):.4f}")
        print(f"\n(For reference, the pre-fix pooled MTGS ADE_norm reported "
              f"earlier this project was 0.0991 -- this run's pooled value "
              f"above should be close if the fix only changed gaze_type "
              f"labels and not predictions themselves.)")


if __name__ == "__main__":
    main()
