"""
verify_temporal_contiguity_v1.py — generic contiguity/horizon-offset
verification, for datasets that don't yet have their own dedicated
version (VACATION, UCO-LAEO) -- same two checks as
verify_temporal_contiguity.py (VAT) and
verify_temporal_contiguity_childplay.py, generalized rather than
duplicated a third and fourth time.

WHY THIS WASN'T ALREADY DONE: VACATION and UCO-LAEO's temporal pipelines
have strong indirect evidence of correctness (VACATION's resolution
counts matched the static pipeline exactly at full scale;
ucolaeo_temporal_utils.py has explicit, assertion-guarded cross-split-
duplication handling) -- but neither had this SPECIFIC direct check run
against real data, the same one that already caught real bugs earlier in
this project's history. Worth confirming rather than trusting lineage
alone, especially for UCO-LAEO given its own documented bug history.

PER-DATASET FRAME-NUMBER PARSING, the one real difference:
  vacation: filenames are "{video_id}_{frame}.jpg" -- frame is the part
    after the LAST underscore (video_id itself never contains one, but
    parsing from the right is robust regardless).
  ucolaeo: filenames ARE the frame number directly ("000046.jpg" -> 46),
    no video_id prefix to strip.

Usage:
    python3 verify_temporal_contiguity_v1.py <vacation|ucolaeo> <manifest_path>
"""

import os
import sys
import json
import random

N_EXAMPLES = 5
SEED = 42


def parse_frame_idx(path, dataset):
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]
    if dataset == "vacation":
        if "_" not in stem:
            raise ValueError(f"path '{path}' -- basename stem '{stem}' has no "
                              f"underscore to split video_id from frame number.")
        frame_str = stem.rsplit("_", 1)[1]
    else:  # ucolaeo
        frame_str = stem
    if not frame_str.isdigit():
        raise ValueError(f"path '{path}' -- '{frame_str}' (from stem '{stem}') "
                          f"doesn't parse as a frame number.")
    return int(frame_str)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("vacation", "ucolaeo"):
        print("Usage: python3 verify_temporal_contiguity_v1.py <vacation|ucolaeo> <manifest_path>")
        sys.exit(1)
    dataset, manifest_path = sys.argv[1], sys.argv[2]

    n_total = 0
    n_checked = 0
    n_context_gap_violation = 0
    n_horizon_offset_violation = 0
    violation_examples = []
    all_valid_rows = []

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            if not row["window_valid"]:
                continue
            n_checked += 1

            try:
                context_idxs = [parse_frame_idx(p, dataset) for p in row["context_frame_paths"]]
                target_idx = parse_frame_idx(row["target_frame_path"], dataset)
            except ValueError as e:
                violation_examples.append((row["clip_id"], row["subject_id"], f"PARSE ERROR: {e}"))
                continue

            # Check 1: strict +1 contiguity across all 8 context frames.
            diffs = [b - a for a, b in zip(context_idxs, context_idxs[1:])]
            if any(d != 1 for d in diffs):
                n_context_gap_violation += 1
                if len(violation_examples) < 10:
                    violation_examples.append(
                        (row["clip_id"], row["subject_id"],
                         f"context frame_idx diffs = {diffs} (expected all 1s): {context_idxs}")
                    )

            # Check 2: target sits exactly horizon_frames past the last context frame.
            expected_target_idx = context_idxs[-1] + row["horizon_frames"]
            if target_idx != expected_target_idx:
                n_horizon_offset_violation += 1
                if len(violation_examples) < 10:
                    violation_examples.append(
                        (row["clip_id"], row["subject_id"],
                         f"target_idx={target_idx}, expected={expected_target_idx} "
                         f"(last_context={context_idxs[-1]}, horizon_frames={row['horizon_frames']})")
                    )

            all_valid_rows.append(row)

    print("=" * 78)
    print(f"TEMPORAL CONTIGUITY VERIFICATION -- {dataset.upper()}")
    print("=" * 78)
    print(f"Total manifest rows            : {n_total}")
    print(f"Valid rows checked             : {n_checked}")
    print(f"Context-frame gap violations   : {n_context_gap_violation} "
          f"({100 * n_context_gap_violation / max(n_checked,1):.3f}%)")
    print(f"Horizon-offset violations      : {n_horizon_offset_violation} "
          f"({100 * n_horizon_offset_violation / max(n_checked,1):.3f}%)")

    if violation_examples:
        print(f"\nFirst {len(violation_examples)} violation(s):")
        for clip_id, subject, detail in violation_examples:
            print(f"  {clip_id} / {subject}: {detail}")

    if n_context_gap_violation == 0 and n_horizon_offset_violation == 0:
        print("\nBoth checks pass with ZERO violations across the entire valid "
              "population -- frame contiguity and horizon offsets are correct.")
    else:
        print("\nVIOLATIONS FOUND -- worth root-causing before trusting ANY "
              "temporal result built on these windows.")

    print("\n" + "=" * 78)
    print(f"{N_EXAMPLES} RANDOM VALID WINDOWS, IN FULL")
    print("=" * 78)
    rng = random.Random(SEED)
    sample = rng.sample(all_valid_rows, min(N_EXAMPLES, len(all_valid_rows)))
    for row in sample:
        print(f"\n{row['clip_id']} / {row['subject_id']}  "
              f"(target_horizon_ms={row.get('target_horizon_ms')}, "
              f"horizon_frames={row['horizon_frames']}, "
              f"target_gaze_type={row['target_gaze_type']})")
        for path in row["context_frame_paths"]:
            print(f"    context {os.path.basename(path)}")
        print(f"    TARGET  {os.path.basename(row['target_frame_path'])}  "
              f"gaze=({row.get('target_gaze_x')}, {row.get('target_gaze_y')}) [raw px]")


if __name__ == "__main__":
    main()
