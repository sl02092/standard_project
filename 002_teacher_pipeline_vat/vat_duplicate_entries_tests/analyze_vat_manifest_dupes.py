#!/usr/bin/env python3
"""Characterizes the ~4,873 duplicate (show, clip, subject, fname) keys in
VAT's frame_manifest.csv. Doesn't assume a cause -- just reports what the
actual duplicate rows look like, so the real root cause can be read off
the evidence rather than guessed from code alone.

Usage:
    python3 analyze_vat_manifest_dupes.py frame_manifest.csv
"""
import sys
import csv
from collections import defaultdict


def main():
    path = sys.argv[1]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")

    groups = defaultdict(list)
    for r in rows:
        key = (r["show"], r["clip"], r["subject"], r["fname"])
        groups[key].append(r)

    dupe_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Unique keys: {len(groups)}")
    print(f"Duplicate keys (appearing 2+ times): {len(dupe_groups)}")
    total_dupe_rows = sum(len(v) for v in dupe_groups.values())
    print(f"Total rows involved in duplicates: {total_dupe_rows}")

    if not dupe_groups:
        print("\nNo duplicates found in this file.")
        return

    # Are duplicate rows byte-identical across ALL columns, or do they differ?
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
            for col in first.keys():
                vals = set(r[col] for r in group_rows)
                if len(vals) > 1:
                    differing_columns_seen.add(col)

    print(f"\nGroups where all copies are byte-identical (every column): {exact_dupe_groups}")
    print(f"Groups where copies differ in at least one column: {differing_groups}")
    if differing_columns_seen:
        print(f"Columns that differ when they do: {sorted(differing_columns_seen)}")

    # Show a handful of concrete examples either way
    print("\n--- Example duplicate groups ---")
    shown = 0
    for key, group_rows in dupe_groups.items():
        if shown >= 5:
            break
        print(f"\nKey: show={key[0]!r} clip={key[1]!r} subject={key[2]!r} fname={key[3]!r}")
        print(f"  {len(group_rows)} copies:")
        for r in group_rows:
            print(f"    img_path={r.get('img_path')}")
            print(f"    ann_path={r.get('ann_path')}")
            print(f"    head=({r.get('head_x1')},{r.get('head_y1')},{r.get('head_x2')},{r.get('head_y2')}) "
                  f"gaze=({r.get('gaze_x')},{r.get('gaze_y')}) gaze_type={r.get('gaze_type')}")
        shown += 1

    # Check for near-duplicate show names (case/whitespace variants) --
    # tests whether two differently-spelled folders could be colliding
    # after some downstream normalization.
    all_shows = sorted(set(r["show"] for r in rows))
    print(f"\n--- Show name check ---")
    print(f"Unique show values in this file: {len(all_shows)}")
    normalized = defaultdict(list)
    for s in all_shows:
        norm_key = s.strip().lower().replace(" ", "_")
        normalized[norm_key].append(s)
    collisions = {k: v for k, v in normalized.items() if len(v) > 1}
    if collisions:
        print(f"WARNING: {len(collisions)} normalized-name collisions found:")
        for norm_key, variants in collisions.items():
            print(f"  {norm_key}: {variants}")
    else:
        print("No case/whitespace-variant show name collisions found.")

    # Which shows do the duplicate keys actually belong to?
    dupe_shows = defaultdict(int)
    for key in dupe_groups:
        dupe_shows[key[0]] += 1
    print(f"\n--- Shows containing duplicate keys ({len(dupe_shows)} shows) ---")
    for show, count in sorted(dupe_shows.items(), key=lambda x: -x[1])[:15]:
        print(f"  {show}: {count} duplicate keys")


if __name__ == "__main__":
    main()
