#!/usr/bin/env python3
"""ChildPlay 3-way version of build_childplay_clean_population.py --
outputs TRAIN/VAL/TEST (not just TEST), using the same clip-disjoint H5
split already confirmed today (68 train / 6 val / 21 test clips, zero
overlap). Matches VACATION's _TRAIN/_VAL/_TEST naming exactly.

Run inside venv_mtgs on HPC (needs pandas+pytables for the H5s).

Usage:
    python3 build_childplay_3way_population.py /path/to/ann_root \\
        frame_manifest_childplay.csv \\
        labels_childplay_full.jsonl \\
        labels_childplay_hybrid_targeted.jsonl \\
        mtgs_teacher_childplay.CLEAN.jsonl
"""
import sys
import os
import csv
import json
import pandas as pd


def clip_name_from_path(path):
    clip, _frame = path.split("/")[1:]
    return "_".join(clip.split("_")[:-1])


def load_clips(h5_path):
    df = pd.read_hdf(h5_path, "data")
    paths = df.groupby("path").groups.keys()
    return set(clip_name_from_path(p) for p in paths)


def bucket_for_show(show, train_clips, val_clips, test_clips):
    # Precedence matches the priority already established for VAT/
    # VACATION: an explicit test/val listing wins; ambiguity (a clip
    # appearing in more than one H5, which today's checks showed doesn't
    # happen here) would fall through to train.
    if show in test_clips:
        return "test"
    if show in val_clips:
        return "val"
    return "train"


def filter_csv(path, train_clips, val_clips, test_clips, out_prefix):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[bucket_for_show(r.get("show"), train_clips, val_clips, test_clips)].append(r)
    for split, split_rows in buckets.items():
        if not split_rows:
            print(f"  {split}: 0 rows -- skipping.")
            continue
        out_path = f"{out_prefix}_{split.upper()}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"  {split}: {len(split_rows)} rows -> {out_path}")
    return len(rows)


def filter_jsonl(path, train_clips, val_clips, test_clips, out_prefix):
    n_total = 0
    buckets = {"train": [], "val": [], "test": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            buckets[bucket_for_show(r.get("show"), train_clips, val_clips, test_clips)].append(r)
    for split, split_rows in buckets.items():
        if not split_rows:
            print(f"  {split}: 0 rows -- skipping.")
            continue
        out_path = f"{out_prefix}_{split.upper()}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {split}: {len(split_rows)} rows -> {out_path}")
    return n_total


def main():
    ann_root = sys.argv[1]
    manifest_path, labels_full_path, labels_hybrid_path, mtgs_path = sys.argv[2:6]

    train_clips = load_clips(os.path.join(ann_root, "childplay_train.h5"))
    val_clips = load_clips(os.path.join(ann_root, "childplay_val.h5"))
    test_clips = load_clips(os.path.join(ann_root, "childplay_test.h5"))
    print(f"train clips: {len(train_clips)}, val clips: {len(val_clips)}, "
          f"test clips: {len(test_clips)}")

    for path, loader in [
        (manifest_path, filter_csv),
        (labels_full_path, filter_jsonl),
        (labels_hybrid_path, filter_jsonl),
        (mtgs_path, filter_jsonl),
    ]:
        print(f"\n{path}:")
        out_prefix = path.rsplit(".", 1)[0]
        n_total = loader(path, train_clips, val_clips, test_clips, out_prefix)
        print(f"  ({n_total} total rows read)")


if __name__ == "__main__":
    main()
