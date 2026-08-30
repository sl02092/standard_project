#!/usr/bin/env python3
"""Characterizes duplicate (show, clip, subject, fname) keys in any of
this project's JSONL label files. Same approach as
analyze_vat_manifest_dupes.py, adapted for JSONL.

Usage:
    python3 analyze_jsonl_dupes.py labels_full.jsonl
    python3 analyze_jsonl_dupes.py labels_hybrid_targeted.jsonl
"""
import sys
import json
from collections import defaultdict


def main():
    path = sys.argv[1]

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"Total rows: {len(rows)}")

    groups = defaultdict(list)
    for r in rows:
        key = (r.get("show"), r.get("clip"), r.get("subject"), r.get("fname"))
        groups[key].append(r)

    dupe_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Unique keys: {len(groups)}")
    print(f"Duplicate keys (appearing 2+ times): {len(dupe_groups)}")
    total_dupe_rows = sum(len(v) for v in dupe_groups.values())
    print(f"Total rows involved in duplicates: {total_dupe_rows}")

    if not dupe_groups:
        print("\nNo duplicates found in this file.")
        return

    exact_dupe_groups = 0
    differing_groups = 0
    differing_columns_seen = set()

    for key, group_rows in dupe_groups.items():
        first = group_rows[0]
        all_identical = all(r == first for r in group_rows[1:])
        if all_identical:
            exact_dupe_groups += 1
        else:
            differing_groups += 1
            all_cols = set()
            for r in group_rows:
                all_cols.update(r.keys())
            for col in all_cols:
                vals = set(json.dumps(r.get(col), sort_keys=True) for r in group_rows)
                if len(vals) > 1:
                    differing_columns_seen.add(col)

    print(f"\nGroups where all copies are byte-identical (every field): {exact_dupe_groups}")
    print(f"Groups where copies differ in at least one field: {differing_groups}")
    if differing_columns_seen:
        print(f"Fields that differ when they do: {sorted(differing_columns_seen)}")

    print("\n--- Example duplicate groups ---")
    shown = 0
    for key, group_rows in dupe_groups.items():
        if shown >= 5:
            break
        print(f"\nKey: show={key[0]!r} clip={key[1]!r} subject={key[2]!r} fname={key[3]!r}")
        print(f"  {len(group_rows)} copies:")
        for r in group_rows:
            print(f"    label_source={r.get('label_source')} "
                  f"teacher_version={r.get('teacher_version')} "
                  f"gaze_type={r.get('gaze_type')} "
                  f"confidence={r.get('confidence')}")
            print(f"    pred=({r.get('pred_x')},{r.get('pred_y')}) "
                  f"gt=({r.get('gt_x')},{r.get('gt_y')}) "
                  f"frame_idx={r.get('frame_idx')}")
        shown += 1

    dupe_shows = defaultdict(int)
    for key in dupe_groups:
        dupe_shows[key[0]] += 1
    print(f"\n--- Shows containing duplicate keys ({len(dupe_shows)} shows) ---")
    for show, count in sorted(dupe_shows.items(), key=lambda x: -x[1])[:15]:
        print(f"  {show}: {count} duplicate keys")

    # label_source / teacher_version breakdown across ALL dupe rows, not
    # just the first of each group -- useful to see if e.g. a resume
    # produced two rows with different teacher_version/confidence for the
    # "same" logical row, vs. genuinely identical reprocessing.
    src_counter = defaultdict(int)
    for group_rows in dupe_groups.values():
        for r in group_rows:
            src_counter[(r.get("label_source"), r.get("confidence"))] += 1
    print(f"\n--- (label_source, confidence) across all duplicate rows ---")
    for k, v in sorted(src_counter.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
