"""
check_train_gaze_type_composition_childplay.py — direct check of the
TRAINING set's target_gaze_type composition, to test whether the large
object/social asymmetry in the LR=1e-5 gaze-type breakdown is driven by
training-set class imbalance rather than a stronger denoising effect.
"""
from collections import Counter
from temporal_window_dataset_childplay import TemporalWindowDataset, build_clip_split

train_ids, val_ids = build_clip_split()
train_ds = TemporalWindowDataset(split_clip_ids=train_ids, split_name="train")

counts = Counter(r["target_gaze_type"] for r in train_ds.rows)
total = sum(counts.values())
print(f"\nTraining set target_gaze_type composition (n={total}):")
for gtype, n in counts.most_common():
    print(f"  {gtype:10s}: {n:6d}  ({n/total*100:.1f}%)")