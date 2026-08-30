"""
check_vat_manifest_has_test_clips.py — one-shot check: does
vat_temporal_manifest.jsonl contain windows for any clip whose real
annotation file lives under VAT's TEST split directory, rather than
TRAIN?

WHY THIS IS THE RIGHT CHECK FOR VAT SPECIFICALLY: unlike ChildPlay/
VACATION, VAT has no single CSV or readme listing split membership --
split is determined entirely by which top-level annotations/ subfolder
(train/ or test/) a given show/clip's own .txt files live under. This
checks the real filesystem directly, the same ground truth
compute_last_known_position_baseline_vat.py's own TRAIN_ANN_DIR constant
already assumes, rather than trusting a comment about what an
unavailable window-builder script does.

USAGE:
    python check_vat_manifest_has_test_clips.py <manifest.jsonl> <vat_root>

<vat_root> is the directory containing "annotations/train" and
(if it exists) "annotations/test" -- e.g. the same VAT_ROOT value used
throughout the rest of this project's VAT scripts.
"""

import os
import sys
import json
from collections import Counter


def main():
    if len(sys.argv) != 3:
        print("Usage: python check_vat_manifest_has_test_clips.py <manifest.jsonl> <vat_root>")
        sys.exit(1)

    manifest_path, vat_root = sys.argv[1], sys.argv[2]
    train_ann_dir = os.path.join(vat_root, "annotations", "train")
    test_ann_dir = os.path.join(vat_root, "annotations", "test")

    print(f"Checking for a real test annotations directory at: {test_ann_dir}")
    test_dir_exists = os.path.isdir(test_ann_dir)
    print(f"  Exists: {test_dir_exists}")
    if not test_dir_exists:
        print("\n>>> RESULT: there is no annotations/test directory at all on this")
        print(">>> machine/path. Either VAT_ROOT is wrong, or this VAT copy genuinely")
        print(">>> only has train annotations available locally. Can't proceed further")
        print(">>> without confirming the correct VAT_ROOT first.")
        return

    manifest_clip_ids = Counter()
    total_rows = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            manifest_clip_ids[row["clip_id"]] += 1
            total_rows += 1

    print(f"\nManifest: {total_rows} total rows, {len(manifest_clip_ids)} distinct clip_ids")

    in_train, in_test, in_neither = [], [], []
    for clip_id in manifest_clip_ids:
        show, clip = clip_id.split("/", 1)
        train_path = os.path.join(train_ann_dir, show, clip)
        test_path = os.path.join(test_ann_dir, show, clip)
        train_exists = os.path.isdir(train_path)
        test_exists = os.path.isdir(test_path)
        if test_exists:
            in_test.append(clip_id)
        elif train_exists:
            in_train.append(clip_id)
        else:
            in_neither.append(clip_id)

    print(f"\nOf the manifest's {len(manifest_clip_ids)} distinct clips:")
    print(f"  Found under annotations/train : {len(in_train)}")
    print(f"  Found under annotations/test  : {len(in_test)}")
    print(f"  Found under neither           : {len(in_neither)}")

    if in_test:
        test_rows = sum(manifest_clip_ids[c] for c in in_test)
        print(f"\n>>> RESULT: the manifest DOES contain {len(in_test)} test-split clips "
              f"({test_rows} manifest rows).")
        print(">>> Re-evaluating already-trained heads on real VAT test data is viable")
        print(">>> without rebuilding the manifest.")
        print(f"\nSample test clip_ids found: {sorted(in_test)[:5]}")
    else:
        print("\n>>> RESULT: every single clip in the manifest resolves to annotations/train.")
        print(">>> The manifest genuinely contains NO test-split windows -- confirms the")
        print(">>> comment in compute_last_known_position_baseline_vat.py directly, from")
        print(">>> the real filesystem rather than trusting that comment alone.")
        print(">>> A real fix requires rebuilding the manifest from annotations/test,")
        print(">>> which needs the actual window-builder script.")

    if in_neither:
        print(f"\nNOTE: {len(in_neither)} clips were found under neither directory --")
        print("worth checking VAT_ROOT is set correctly, or these clips no longer")
        print("exist on disk at their expected location.")
        print(f"Examples: {sorted(in_neither)[:5]}")


if __name__ == "__main__":
    main()
