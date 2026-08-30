#!/usr/bin/env python3
"""Builds ChildPlay's genuinely MTGS-unseen population by filtering the
EXISTING manifest/labels/MTGS files down to rows whose 'show' falls in a
test-only clip (never seen by MTGS in train OR val). No new labeling --
every teacher's predictions already exist for the full population; this
just carves out the clean subset, the same way VAT's TEST_SHOWS files
already exist as a filtered subset of the full pipeline run.

Run inside venv_mtgs on HPC (needs pandas+pytables to read the H5s).

Usage:
    python3 build_childplay_clean_population.py /path/to/ann_root \\
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


def filter_csv(path, test_only_clips, out_path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r.get("show") in test_only_clips]
    if kept:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
            writer.writeheader()
            writer.writerows(kept)
    return len(rows), len(kept)


def filter_jsonl(path, test_only_clips, out_path):
    n_total = 0
    kept = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            if r.get("show") in test_only_clips:
                kept.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    return n_total, len(kept)


def main():
    ann_root = sys.argv[1]
    manifest_path, labels_full_path, labels_hybrid_path, mtgs_path = sys.argv[2:6]

    train_clips = load_clips(os.path.join(ann_root, "childplay_train.h5"))
    val_clips = load_clips(os.path.join(ann_root, "childplay_val.h5"))
    test_clips = load_clips(os.path.join(ann_root, "childplay_test.h5"))
    test_only_clips = test_clips - train_clips - val_clips
    print(f"Test-only clips (genuinely MTGS-unseen): {len(test_only_clips)}")
    print(f"  {sorted(test_only_clips)}")

    def out_name(path, filetype):
        base = path.rsplit(".", 1)[0]
        return f"{base}_TESTONLY.{filetype}"

    for path, loader, ext in [
        (manifest_path, filter_csv, "csv"),
        (labels_full_path, filter_jsonl, "jsonl"),
        (labels_hybrid_path, filter_jsonl, "jsonl"),
        (mtgs_path, filter_jsonl, "jsonl"),
    ]:
        out_path = out_name(path, ext)
        n_total, n_kept = loader(path, test_only_clips, out_path)
        print(f"{path}: {n_total} rows -> {n_kept} kept -> {out_path}")


if __name__ == "__main__":
    main()
