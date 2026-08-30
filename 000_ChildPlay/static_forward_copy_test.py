import os, json
import numpy as np
from childplay_annotation_utils import IMAGES_ROOT
from compute_last_known_position_baseline_childplay import get_subject_gaze, parse_frame_number, get_clip_dimensions

FEATURE_CACHE_DIR = "childplay_temporal_feature_cache"
INDEX_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_features_index.jsonl")
PREDICTED_XY_PATH = os.path.join(FEATURE_CACHE_DIR, "frame_predicted_xy.npy")

predicted_xy = np.load(PREDICTED_XY_PATH)

dists = []
skipped = 0
with open(INDEX_PATH, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        clip, subject, fname = row["clip"], row["subject"], row["fname"]
        abs_frame = parse_frame_number(fname)
        gaze = get_subject_gaze(clip, subject, abs_frame)
        if gaze is None:
            skipped += 1
            continue
        img_path = os.path.join(IMAGES_ROOT, clip, fname)
        dims = get_clip_dimensions(img_path)
        if dims is None:
            skipped += 1
            continue
        w, h = dims
        gx, gy = gaze[0] / w, gaze[1] / h
        px, py = predicted_xy[row["embedding_idx"]]
        dists.append(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)

print(f"Same-frame static model accuracy on ChildPlay (zero-horizon):")
print(f"  n = {len(dists)}, skipped (off-screen/no-dims) = {skipped}")
print(f"  ADE_norm = {sum(dists)/len(dists):.4f}")
print(f"  Compare against the checkpoint's own VAT val_ade (~0.1996).")