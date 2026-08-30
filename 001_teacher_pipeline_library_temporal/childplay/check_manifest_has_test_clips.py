"""
check_manifest_has_test_clips.py — one-shot check: does
childplay_temporal_manifest.jsonl actually contain windows for any of
ChildPlay's real, official test-split clips?

This is the one fact that determines whether re-evaluating the already-
trained temporal heads on genuine test data is possible without rebuilding
the manifest from scratch. Everything else (the split-function changes,
the new eval script) is straightforward once this is known either way.

USAGE: python check_manifest_has_test_clips.py <path_to_manifest.jsonl> <path_to_clips.csv>
"""

import sys
import csv
import json
from collections import Counter


def main():
    if len(sys.argv) != 3:
        print("Usage: python check_manifest_has_test_clips.py <manifest.jsonl> <clips.csv>")
        sys.exit(1)

    manifest_path, clips_csv_path = sys.argv[1], sys.argv[2]

    test_clip_ids = set()
    with open(clips_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                test_clip_ids.add(row["clip"])
    print(f"Real test-split clips in {clips_csv_path}: {len(test_clip_ids)}")

    manifest_clip_ids = Counter()
    total_rows = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            manifest_clip_ids[row["clip_id"]] += 1
            total_rows += 1

    found_test_clips = set(manifest_clip_ids) & test_clip_ids
    test_rows = sum(manifest_clip_ids[c] for c in found_test_clips)

    print(f"\nManifest: {total_rows} total rows, {len(manifest_clip_ids)} distinct clip_ids")
    print(f"Of those, {len(found_test_clips)} are real test-split clips "
          f"({test_rows} manifest rows belonging to them)")

    if found_test_clips:
        print("\n>>> RESULT: the manifest DOES contain test-split windows.")
        print(">>> Re-evaluating already-trained heads on real test data is viable")
        print(">>> without rebuilding the manifest.")
        print(f"\nSample test clip_ids found in the manifest: {sorted(found_test_clips)[:5]}")
    else:
        print("\n>>> RESULT: the manifest contains NO test-split windows.")
        print(">>> Same situation as VAT -- the window-builder never referenced")
        print(">>> test-split annotations, so there's no cached data to evaluate on")
        print(">>> without rebuilding the manifest from scratch.")


if __name__ == "__main__":
    main()
