#!/usr/bin/env python3
"""Builds VACATION's train/val/test populations from the dataset's OWN
official readme.txt split (not a newly-invented percentage), filtering
the EXISTING manifest/labels/MTGS files. No new labeling -- every
teacher's predictions already exist for the full population.

load_official_split()/get_video_split() logic below is copied verbatim
from vacation_temporal_utils.py, deliberately, not re-derived --
consistent with this project's own stated discipline for reusing
already-verified parsing logic rather than risking silent divergence.

Runs entirely locally -- no HPC/H5 needed, unlike ChildPlay's version.

Usage:
    python3 build_vacation_clean_population.py readme.txt \\
        frame_manifest_vacation.csv \\
        labels_vacation_full.jsonl \\
        labels_vacation_hybrid_targeted.jsonl \\
        mtgs_teacher_vacation.jsonl
"""
import sys
import re
import csv
import json
from collections import Counter


def load_official_split(readme_path):
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


def get_video_split(video_id, split_map):
    return split_map.get(video_id, "train")  # extras fold into train


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    readme_path = sys.argv[1]
    file_paths = sys.argv[2:6]

    split_map = load_official_split(readme_path)
    n_train = sum(1 for v in split_map.values() if v == "train")
    n_val = sum(1 for v in split_map.values() if v == "val")
    n_test = sum(1 for v in split_map.values() if v == "test")
    print(f"Official split: {n_train} train, {n_val} val, {n_test} test "
          f"({n_train + n_val + n_test} officially listed)")

    for path in file_paths:
        is_csv = path.lower().endswith(".csv")
        rows = load_csv(path) if is_csv else load_jsonl(path)
        print(f"\n{path}: {len(rows)} total rows")

        # Verification, not assumption -- show a few real 'show' values and
        # how they resolve, so a naming mismatch is visible immediately
        # rather than silently producing a wrong (possibly zero) split.
        sample_shows = list({r.get("show") for r in rows[:200]})[:5]
        print(f"  Sample 'show' values found: {sample_shows}")
        for s in sample_shows:
            print(f"    {s!r} -> {get_video_split(s, split_map)}"
                  f"{' (in official list)' if s in split_map else ' (extra -> train)'}")

        buckets = {"train": [], "val": [], "test": []}
        for r in rows:
            split = get_video_split(r.get("show"), split_map)
            buckets[split].append(r)

        for split, split_rows in buckets.items():
            if not split_rows:
                print(f"  {split}: 0 rows -- skipping file write.")
                continue
            base = path.rsplit(".", 1)[0]
            ext = "csv" if is_csv else "jsonl"
            out_path = f"{base}_{split.upper()}.{ext}"
            (write_csv if is_csv else write_jsonl)(split_rows, out_path)
            print(f"  {split}: {len(split_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
