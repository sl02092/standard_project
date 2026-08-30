#!/usr/bin/env python3
"""Checks whether ChildPlay's MTGS train/val/test H5 split is genuinely
clip-disjoint (VAT-style, a real held-out population) or just a
frame-level split within the same clips (same underlying video content
in both train and test, much weaker as a "clean" holdout).

Clip-identifier extraction is copied EXACTLY from childplay_temporal.py's
own __getitem__ logic, so this reads the split the same way MTGS itself
does -- not a reconstructed guess.

Run inside venv_mtgs on HPC, wherever childplay_{train,val,test}.h5 live
(MTGS-main's ann_root).

Usage:
    python3 inspect_childplay_split.py /path/to/ann_root [frame_manifest_childplay.csv]

    Second arg optional -- if given, also reports what fraction of the
    CURRENT teacher-pipeline population (your library's own manifest)
    falls into test-only clips, i.e. how big a genuinely-clean population
    could actually be built from what already exists.
"""
import sys
import os
import csv
import pandas as pd


def clip_name_from_path(path):
    # Copied exactly from ChildPlayDataset_temporal.__getitem__:
    #   clip, frame = path.split("/")[1:]
    #   clip_name = "_".join(clip.split("_")[:-1])
    clip, _frame = path.split("/")[1:]
    clip_name = "_".join(clip.split("_")[:-1])
    return clip_name


def load_clips(h5_path):
    df = pd.read_hdf(h5_path, "data")
    paths = df.groupby("path").groups.keys()
    clips = set(clip_name_from_path(p) for p in paths)
    return clips, len(paths)


def main():
    ann_root = sys.argv[1]
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else None

    splits = {}
    for split in ("train", "val", "test"):
        h5_path = os.path.join(ann_root, f"childplay_{split}.h5")
        clips, n_paths = load_clips(h5_path)
        splits[split] = clips
        print(f"{split}: {n_paths} unique paths, {len(clips)} unique clips")

    print()
    train_test_overlap = splits["train"] & splits["test"]
    val_test_overlap = splits["val"] & splits["test"]
    train_val_overlap = splits["train"] & splits["val"]

    print(f"train ∩ test clip overlap: {len(train_test_overlap)}")
    print(f"val ∩ test clip overlap:   {len(val_test_overlap)}")
    print(f"train ∩ val clip overlap:  {len(train_val_overlap)}")

    if train_test_overlap or val_test_overlap:
        print("\n-> NOT clip-disjoint: test shares clips with train and/or val.")
        print("   This is a frame-level split, not a VAT-style clean holdout --")
        print("   test frames may be near-duplicates of trained frames from the")
        print("   same clip.")
        if train_test_overlap:
            print(f"   Example overlapping clips: {list(train_test_overlap)[:5]}")
    else:
        print("\n-> Clip-disjoint: test's clips never appear in train or val.")
        print("   This IS a genuine held-out population, structurally the same")
        print("   as VAT's TEST_SHOWS -- worth building a clean comparison from.")

    if manifest_path:
        print(f"\n--- Cross-check against {manifest_path} ---")
        test_only_clips = splits["test"] - splits["train"] - splits["val"]
        with open(manifest_path, newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        print(f"Manifest rows: {len(manifest_rows)}")

        # IMPORTANT: the clip_name computed above is the STRIPPED,
        # show-level identifier (childplay_temporal.py's own confusing
        # internal naming calls this "clip_name", but per
        # export_childplay_pseudolabels.py's parse_childplay_path, this
        # value corresponds to the manifest's "show" column -- NOT its
        # "clip" column, which holds the full, unstripped folder name.
        # Comparing against "clip" here would silently show zero matches
        # regardless of real overlap.
        clip_col = None
        for candidate in ("show", "clip"):
            if candidate in manifest_rows[0]:
                clip_col = candidate
                break
        if clip_col is None:
            print("Could not find a clip/show column in the manifest to "
                  "cross-check -- inspect its columns manually: "
                  f"{list(manifest_rows[0].keys())}")
            return

        in_test_only = sum(1 for r in manifest_rows if r[clip_col] in test_only_clips)
        print(f"Manifest rows whose '{clip_col}' matches a test-only clip: {in_test_only}")
        if clip_col == "clip":
            print("(WARNING: using 'clip' as fallback since 'show' wasn't found -- "
                  "this comparison is likely comparing stripped vs. unstripped "
                  "identifiers and may silently under-count. Prefer 'show' if "
                  "the manifest has it under a different name.)")


if __name__ == "__main__":
    main()
