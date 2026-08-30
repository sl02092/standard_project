"""
domain_transfer_check_vacation.py — VACATION port of
domain_transfer_check_ucolaeo.py. Measures the frozen static encoder's
SAME-FRAME accuracy on VACATION -- zero temporal component -- the same
diagnostic that caught ChildPlay's ~30% domain-transfer loss and
confirmed UCO-LAEO's was small (~3.5%).

WHY THIS MATTERS: per objective4_ucolaeo_extension_writeup.md §7.3,
VACATION is broadcast/TV content like VAT and UCO-LAEO, not ChildPlay's
out-of-domain case -- reason to expect a small gap here too, but a
reason to expect it, not evidence of it. Run this before trusting any
downstream PoC result, same discipline as every other dataset.

ONE REAL DIFFERENCE FROM THE UCO-LAEO VERSION: frame-number parsing.
VACATION's filenames are '{video_id}_{frame}.jpg' (frame number after
the LAST underscore), not UCO-LAEO's bare '{frame:06d}.jpg' --
compute_last_known_position_baseline_vacation.py's parse_frame_number()
already handles this correctly, reused directly rather than re-derived.
"""

import os
import json

import numpy as np
from PIL import Image

from vacation_temporal_utils import IMAGES_ROOT
from compute_last_known_position_baseline_vacation import (
    get_subject_gaze, get_subject_gaze_type, parse_frame_number,
)

FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "vacation_temporal_feature_cache")
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

_dim_cache = {}


def get_image_dimensions(img_path):
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
    dists_by_type = {"social": [], "object": []}
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

            img_path = os.path.join(IMAGES_ROOT, clip, fname)
            dims = get_image_dimensions(img_path)
            if dims is None:
                skipped_no_dims += 1
                continue
            w, h = dims

            gx, gy = gaze[0] / w, gaze[1] / h
            px, py = predicted_xy[row["embedding_idx"]]
            d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            dists.append(d)

            gtype = get_subject_gaze_type(clip, subject, abs_frame)
            if gtype in dists_by_type:
                dists_by_type[gtype].append(d)

    print("Same-frame static model accuracy on VACATION (zero-horizon):")
    print(f"  n = {len(dists)}, skipped (unresolved={skipped_unresolved}, "
          f"no-dims={skipped_no_dims})")
    if dists:
        print(f"  ADE_norm (pooled) = {sum(dists)/len(dists):.4f}")
        for gtype, d_list in dists_by_type.items():
            if d_list:
                print(f"  ADE_norm ({gtype:>6}, n={len(d_list)}) = "
                      f"{sum(d_list)/len(d_list):.4f}")
    else:
        print("  No usable rows -- check that extraction actually ran and "
              "produced a non-empty cache before investigating further.")
    print("  Compare against the checkpoint's own VAT val_ade (printed "
          "during extraction) and against UCO-LAEO's own result (0.1911 "
          "vs. checkpoint val_ade 0.1846, ~3.5% relative) as a second "
          "reference point on the same checkpoint. If object ADE here is "
          "notably worse than social ADE, that's evidence the pooled gap "
          "reflects task composition (more object-gaze than VAT's own mix) "
          "rather than a genuine domain-transfer problem.")


if __name__ == "__main__":
    main()
