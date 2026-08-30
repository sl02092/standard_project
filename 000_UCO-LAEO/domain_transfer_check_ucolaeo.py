"""
domain_transfer_check_ucolaeo.py — UCO-LAEO port of
static_forward_copy_test.py (ChildPlay). Measures the frozen static
encoder's SAME-FRAME accuracy on UCO-LAEO -- zero temporal component at
all -- the exact diagnostic that caught ChildPlay's ~30% domain-transfer
loss (objective4_childplay_extension_writeup.md §4).

WHY THIS MATTERS RIGHT NOW: UCO-LAEO is broadcast/TV content, the same
broad visual domain as VAT (confirmed in that write-up's §7.3), unlike
ChildPlay which was genuinely out-of-domain. That's a real reason to
expect a smaller gap here -- but a reason to expect it, not evidence of
it. This script produces the actual number rather than continuing to
reason from analogy. Run this against whatever cache
extract_temporal_frame_features_ucolaeo.py produces before trusting any
downstream PoC result.

ONE REAL ADAPTATION FROM THE CHILDPLAY VERSION: that script imports
get_clip_dimensions() from compute_last_known_position_baseline_childplay
-- a function not available here (not part of
compute_last_known_position_baseline_ucolaeo.py, which only exposes
get_subject_gaze/parse_frame_number). Rather than guess at porting a
function I haven't seen the real implementation of, this reads image
dimensions directly via PIL, matching the same pattern already used in
temporal_window_dataset_ucolaeo.py's own _resolve_all_dimensions() and
ucolaeo_temporal_utils.py's get_video_dimensions(). Everything else is a
direct read of get_subject_gaze/parse_frame_number, whose signatures
were deliberately built to match the ChildPlay version exactly.

HOW TO READ THE RESULT: compare the printed ADE_norm against this
checkpoint's own VAT val_ade (printed by load_gaze_student() during
extraction, ~0.1996 for the current GT-trained checkpoint at time of
writing). A result close to that VAT number suggests a small domain gap
(good news for UCO-LAEO as a clean replication case). A result
substantially worse -- in ChildPlay's case, same-frame ADE roughly
doubled -- would mean the same domain-transfer problem applies here too,
despite the shared broadcast-content domain, and should change the
checkpoint plan accordingly rather than being explained away.
"""

import os
import json

import numpy as np
from PIL import Image

from ucolaeo_temporal_utils import LOCAL_IMAGES_ROOT
from compute_last_known_position_baseline_ucolaeo import get_subject_gaze, parse_frame_number

FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "ucolaeo_temporal_feature_cache")
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

_dim_cache = {}


def get_image_dimensions(img_path):
    """Direct PIL read, cached per path -- see module docstring on why
    this doesn't reuse a same-named ChildPlay helper whose real
    implementation wasn't available to port from."""
    if img_path not in _dim_cache:
        try:
            with Image.open(img_path) as img:
                _dim_cache[img_path] = img.size
        except Exception as e:
            print(f"  [domain_transfer_check] could not read {img_path}: {e}")
            _dim_cache[img_path] = None
    return _dim_cache[img_path]


def main():
    predicted_xy = np.load(PREDICTED_XY_PATH)

    dists = []
    skipped_unresolved = 0
    skipped_no_dims = 0

    with open(INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            clip, subject, fname = row["clip"], row["subject"], row["fname"]
            abs_frame = parse_frame_number(fname)
            gaze = get_subject_gaze(clip, subject, abs_frame)
            if gaze is None:
                skipped_unresolved += 1
                continue

            img_path = os.path.join(LOCAL_IMAGES_ROOT, clip, fname)
            dims = get_image_dimensions(img_path)
            if dims is None:
                skipped_no_dims += 1
                continue
            w, h = dims

            gx, gy = gaze[0] / w, gaze[1] / h
            px, py = predicted_xy[row["embedding_idx"]]
            dists.append(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)

    print("Same-frame static model accuracy on UCO-LAEO (zero-horizon):")
    print(f"  n = {len(dists)}, skipped (unresolved={skipped_unresolved}, "
          f"no-dims={skipped_no_dims})")
    if dists:
        print(f"  ADE_norm = {sum(dists)/len(dists):.4f}")
    else:
        print("  No usable rows -- check that extraction actually ran and "
              "produced a non-empty cache before investigating further.")
    print("  Compare against the checkpoint's own VAT val_ade (printed "
          "during extraction, ~0.1996 for the current GT-trained checkpoint) "
          "-- see module docstring for how to read the comparison.")


if __name__ == "__main__":
    main()
