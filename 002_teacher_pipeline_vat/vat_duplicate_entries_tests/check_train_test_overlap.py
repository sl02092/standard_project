#!/usr/bin/env python3
"""Checks for (show, clip, subject, fname) key overlap ACROSS two files --
e.g. the main VAT population vs. the testshows population. This is a
different question from within-file duplicates (already confirmed clean
on all four files): are any keys shared BETWEEN a "train/main" file and
its corresponding "testshows" file?

Handles both .csv and .jsonl transparently. Also checks a
space/underscore-normalized version of "show" in case the two files use
different naming conventions for the same underlying show (this project
uses both "show_spaces" and "show_us" conventions in different places).

Usage:
    python3 check_train_test_overlap.py file_a.csv_or_jsonl file_b.csv_or_jsonl
"""
import sys
import csv
import json


def load_keys(path):
    rows = []
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def make_key(r, normalize_show=False):
    show = r.get("show", "")
    if normalize_show:
        show = show.strip().lower().replace(" ", "_")
    return (show, r.get("clip"), r.get("subject"), r.get("fname"))


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]

    rows_a = load_keys(path_a)
    rows_b = load_keys(path_b)
    print(f"{path_a}: {len(rows_a)} rows")
    print(f"{path_b}: {len(rows_b)} rows")

    # Exact key match first
    keys_a = set(make_key(r) for r in rows_a)
    keys_b = set(make_key(r) for r in rows_b)
    overlap = keys_a & keys_b
    print(f"\nExact key overlap (show as-is): {len(overlap)}")

    # Normalized-show key match, in case of a naming-convention mismatch
    norm_keys_a = set(make_key(r, normalize_show=True) for r in rows_a)
    norm_keys_b = set(make_key(r, normalize_show=True) for r in rows_b)
    norm_overlap = norm_keys_a & norm_keys_b
    print(f"Normalized-show key overlap: {len(norm_overlap)}")

    if norm_overlap and not overlap:
        print("\n  -> Overlap ONLY appears after normalizing show names -- "
              "the two files are using different naming conventions for "
              "the SAME shows, and something downstream that doesn't "
              "normalize would silently miss this collision.")

    if overlap:
        print(f"\n--- Example overlapping keys ---")
        for k in list(overlap)[:10]:
            print(f"  {k}")

        show_counts = {}
        for k in overlap:
            show_counts[k[0]] = show_counts.get(k[0], 0) + 1
        print(f"\n--- Shows involved in overlap ({len(show_counts)} shows) ---")
        for show, count in sorted(show_counts.items(), key=lambda x: -x[1]):
            print(f"  {show}: {count} overlapping keys")
    elif not norm_overlap:
        print("\nNo overlap found, exact or normalized -- these two "
              "populations are genuinely disjoint.")


if __name__ == "__main__":
    main()
