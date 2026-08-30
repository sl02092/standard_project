"""
verify_temporal_contiguity_childplay.py — ChildPlay adaptation of
verify_temporal_contiguity.py (VAT). Same two checks, same reasoning for
why they matter (see the original VAT script's docstring for the full
history) -- adapted here only where ChildPlay's manifest genuinely
differs from VAT's.

WHAT ACTUALLY DIFFERS FROM THE VAT VERSION, AND WHY:
1. FRAME-INDEX PARSING: VAT's context_frame_paths were bare, zero-padded
   fnames ("00011121.jpg"). ChildPlay's manifest stores FULL resolvable
   paths (a deliberate improvement over VAT's schema -- see
   childplay_temporal_manifest_schema.md -- to avoid needing the same
   path-reconstruction step extract_temporal_frame_features.py had to
   work around for VAT), with filenames of the form
   "{video_id}_{frame_number}.jpg" -- video_id itself can contain
   underscores/hyphens (real YouTube IDs), so the frame number is taken
   from the LAST underscore-separated component, not assumed to be the
   whole stem (same subtlety already handled in
   childplay_annotation_utils.py's parse_clip_start_frame).
2. horizon_frames NEEDS NO CHANGE to the actual check logic -- VAT's
   script already read row["horizon_frames"] per-row rather than a
   global constant, which happens to already be exactly correct for
   ChildPlay's per-clip, real-fps-derived horizon values (schema doc
   §4's three-field design). Nothing to fix here.
3. clip_id STRUCTURE: VAT's clip_id was "show/clip" (split on "/") --
   ChildPlay's is a single flat clip name (e.g.
   "3mXFJ8S-boU_4218-4655"), no split needed or possible.
4. EXAMPLE PRINTOUT: also surfaces target_horizon_ms / horizon_ms_actual
   (sanity-check the fps-based computation looks reasonable) and
   raw_gaze_class / is_child (ChildPlay-specific fields with no VAT
   equivalent).
5. predicted_xy CACHE KEY: ChildPlay's own extract_temporal_frame_features.py
   equivalent hasn't been built yet, so the exact cache key convention
   used there is unknown. The lookup below is a reasonable placeholder
   (clip_id/subject_id/fname), NOT confirmed against a real cache --
   reconcile this against whatever key format that script actually uses
   once it exists. Harmless either way right now: with no cache built,
   this section just reports "not found" and the rest of the script is
   unaffected.
"""

import os
import json
import random
from collections import defaultdict

MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH", "childplay_temporal_manifest.jsonl")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "childplay_temporal_feature_cache")
N_EXAMPLES = 5
SEED = 42


def parse_frame_idx(path):
    """ChildPlay paths are FULL paths: .../images/{clip}/{video_id}_{frame}.jpg
    -- extract just the frame number, splitting on the LAST underscore
    since video_id can itself contain underscores/hyphens."""
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]
    if "_" not in stem:
        raise ValueError(f"path '{path}' -- basename stem '{stem}' has no "
                          f"underscore to split video_id from frame number.")
    frame_str = stem.rsplit("_", 1)[1]
    if not frame_str.isdigit():
        raise ValueError(f"path '{path}' -- '{frame_str}' (from stem '{stem}') "
                          f"doesn't parse as a frame number.")
    return int(frame_str)


def main():
    # Optional: load predicted_xy cache for the example printout, if present.
    # See module docstring point 5 -- key format here is a placeholder,
    # not yet confirmed against a real ChildPlay feature cache.
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
        print("[verify] No predicted_xy cache found (expected -- ChildPlay "
              "feature extraction hasn't run yet) -- example printout will "
              "omit it.\n")

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
                context_idxs = [parse_frame_idx(p) for p in row["context_frame_paths"]]
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

            # Check 2: target sits exactly horizon_frames past the last
            # context frame. row["horizon_frames"] is already the real,
            # per-clip, fps-derived value (schema doc §4) -- no VAT-style
            # global constant assumption here, same as the original script.
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
    print("TEMPORAL CONTIGUITY VERIFICATION -- CHILDPLAY")
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
              "Same clean result as the VAT manifest's own verification.")
    else:
        print("\nVIOLATIONS FOUND -- this is a real, previously-unverified "
              "assumption failing. Worth root-causing before trusting ANY "
              "ChildPlay temporal result built on these windows.")

    # ── Qualitative spot-check: a few real windows in full ──────────────
    print("\n" + "=" * 78)
    print(f"{N_EXAMPLES} RANDOM VALID WINDOWS, IN FULL")
    print("=" * 78)
    rng = random.Random(SEED)
    sample = rng.sample(all_valid_rows, min(N_EXAMPLES, len(all_valid_rows)))
    for row in sample:
        clip_id = row["clip_id"]  # flat clip name -- no "show/clip" split for ChildPlay
        subject = row["subject_id"]
        print(f"\n{clip_id} / {subject}  (target_horizon_ms={row['target_horizon_ms']}, "
              f"horizon_frames={row['horizon_frames']}, "
              f"horizon_ms_actual={row['horizon_ms_actual']}, "
              f"target_gaze_type={row['target_gaze_type']}, "
              f"raw_gaze_class={row.get('raw_gaze_class')}, "
              f"is_child={row.get('is_child')})")
        for path in row["context_frame_paths"]:
            fname = os.path.basename(path)
            key = f"{clip_id}/{subject}/{fname}"  # placeholder key -- see module docstring point 5
            pred = key_to_pred.get(key)
            pred_str = f"predicted_xy=({pred[0]:.3f}, {pred[1]:.3f})" if pred else "predicted_xy=n/a"
            print(f"    context {fname}  {pred_str}")
        print(f"    TARGET  {os.path.basename(row['target_frame_path'])}  "
              f"gaze=({row['target_gaze_x']}, {row['target_gaze_y']}) [raw px]")


if __name__ == "__main__":
    main()
