"""
verify_temporal_contiguity.py — Closes an open item flagged since
2026-07-06 and never actually checked in code: are context_frame_paths
genuinely contiguous, correctly ordered frames, and does target_frame_path
really sit exactly horizon_frames past the last context frame?

WHY THIS MATTERS NOW, SPECIFICALLY: two of the three live hypotheses for
why no temporal model beats the static-copy-forward baseline have now
been tested (explicit predicted_xy input, pretrained embeddings) without
closing the gap. Before concluding this is a genuine ceiling -- a
legitimate finding -- it's worth ruling out a pipeline correctness issue,
given this project's own precedent (MTGS subject-alignment off-by-one,
frame_manifest.csv walltime-as-fact) of surprising results turning out to
be instrumentation bugs rather than genuine findings. The original
schema doc said this exact check was "cheap now and expensive to
discover wrong later" and flagged it as not yet done -- still true.

WHAT THIS CHECKS, over the ENTIRE valid population (not a sample --
cheap, filename-only, no images opened):
  1. Within each window, do the 8 context_frame_paths' frame indices
     increase by EXACTLY 1 each step (true contiguity, no skips/gaps/
     duplicates/reordering)?
  2. Does target_frame_path's frame index equal the last context frame's
     index + horizon_frames EXACTLY (correct horizon offset, not off by
     one or using the wrong reference frame)?
Both assumptions are foundational to the entire temporal-modeling
premise -- if either is violated at any real rate, there's no genuine
"recent video" for a GRU/Transformer to learn motion from, which would
directly explain an inability to beat naive last-position copying
regardless of architecture, LR, or embedding quality.

ALSO prints a handful of real windows in full (frame indices,
predicted_xy trajectory if available, target) for manual eyeballing --
systematic checks catch violations of the *rule*; a human glance at real
examples catches things that satisfy the rule but still look wrong
(duplicate-looking frames, a static-model prediction trajectory that
never moves, etc.).

FRAME INDEX PARSING: VAT fnames are zero-padded frame numbers, e.g.
"00011121.jpg" -> 11121 (confirmed against a real path from MTGS work:
.../images/CBS This Morning/11118_11359/00011121.jpg). Fails loudly if a
filename doesn't parse this way rather than silently skipping it.
"""

import os
import json
import random
from collections import defaultdict

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", r"C:\repo\standard_project\vat_missing_annotation_investigation\vat_temporal_manifest.jsonl")
#MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "vat_temporal_manifest.jsonl")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "temporal_feature_cache")
N_EXAMPLES = 5
SEED = 42


def parse_frame_idx(fname):
    stem = os.path.splitext(fname)[0]
    if not stem.isdigit():
        raise ValueError(f"fname '{fname}' doesn't parse as a zero-padded frame index.")
    return int(stem)


def main():
    # Optional: load predicted_xy cache for the example printout, if present.
    index_path = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
    predicted_xy_path = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")
    key_to_pred = {}
    if os.path.exists(index_path) and os.path.exists(predicted_xy_path):
        import numpy as np
        pred_arr = np.load(predicted_xy_path)
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key_to_pred[row["key"]] = tuple(pred_arr[row["embedding_idx"]].tolist())
        print(f"[verify] Loaded predicted_xy cache ({len(key_to_pred)} keys) for example printout.\n")
    else:
        print("[verify] No predicted_xy cache found -- example printout will omit it.\n")

    n_total = 0
    n_checked = 0
    n_context_gap_violation = 0
    n_horizon_offset_violation = 0
    violation_examples = []
    all_valid_rows = []

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            if not row["window_valid"]:
                continue
            n_checked += 1

            try:
                context_idxs = [parse_frame_idx(fn) for fn in row["context_frame_paths"]]
                target_idx = parse_frame_idx(row["target_frame_path"])
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
    print("TEMPORAL CONTIGUITY VERIFICATION")
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
              "population -- frame contiguity and horizon offsets are correct. "
              "This rules out mis-ordered/non-contiguous windows as an "
              "explanation for the temporal PoC's ceiling.")
    else:
        print("\nVIOLATIONS FOUND -- this is a real, previously-unverified "
              "assumption failing. Worth root-causing before trusting ANY "
              "temporal PoC result built on these windows, past or future.")

    # ── Qualitative spot-check: a few real windows in full ──────────────
    print("\n" + "=" * 78)
    print(f"{N_EXAMPLES} RANDOM VALID WINDOWS, IN FULL")
    print("=" * 78)
    rng = random.Random(SEED)
    sample = rng.sample(all_valid_rows, min(N_EXAMPLES, len(all_valid_rows)))
    for row in sample:
        show, clip = row["clip_id"].split("/", 1)
        subject = row["subject_id"]
        print(f"\n{row['clip_id']} / {subject}  (horizon_frames={row['horizon_frames']}, "
              f"target_gaze_type={row['target_gaze_type']})")
        for fn in row["context_frame_paths"]:
            key = f"{show}/{clip}/{subject}/{fn}"
            pred = key_to_pred.get(key)
            pred_str = f"predicted_xy=({pred[0]:.3f}, {pred[1]:.3f})" if pred else "predicted_xy=n/a"
            print(f"    context {fn}  {pred_str}")
        print(f"    TARGET  {row['target_frame_path']}  "
              f"gaze=({row['target_gaze_x']}, {row['target_gaze_y']}) [raw px]")


if __name__ == "__main__":
    main()
