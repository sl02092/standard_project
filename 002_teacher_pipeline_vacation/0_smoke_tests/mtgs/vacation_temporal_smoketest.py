# coding=utf-8

"""
vacation_temporal_smoketest.py — minimal, self-contained check that
VacationDataset_temporal actually instantiates and produces a real item.
Deliberately has NO hydra/config dependency (unlike the export scripts)
-- avoids the relative-config_path trap entirely by just hardcoding the
two real path strings directly. No GPU, no model, no checkpoint needed.
"""

import os

# Import order matters here, not just style -- mtgs.datasets/__init__.py
# and mtgs.train/__init__.py have a real circular dependency between
# them (confirmed via the actual traceback: importing anything under
# mtgs.datasets first triggers mtgs.datasets/__init__.py, which pulls in
# mtgs.train.transforms, which pulls in mtgs.train/__init__.py, which
# pulls back into mtgs.datasets while it's still mid-initialization).
# Every working export script sidesteps this by importing
# mtgs.config/mtgs.networks.models FIRST, which fully loads mtgs.train
# before mtgs.datasets is ever touched -- matched here for the same
# reason, not just copied out of habit.
from mtgs.config import ConfigManager
from mtgs.networks.models import MTGSModel
from mtgs.datasets.vacation_temporal import VacationDataset_temporal
from mtgs.utils.image import IMG_MEAN, IMG_STD
from mtgs.train.transforms import Compose, Resize, ToTensor, Normalize

# Hardcoded directly -- matches config.yaml's current values. Update here
# if either changes, this script deliberately does not read config.yaml.
VACATION_ROOT = "/parallel_scratch/sl02092/standard_project/data/vacation"
ANN_ROOT = "/parallel_scratch/sl02092/MTGS-main/00_vgaze/VSGaze"  # confirm this is really where the 3 vacation_*.h5 files were uploaded
IMAGE_SIZE = (448, 448)
HEAD_SIZE = (224, 224)


def main():
    print(f"VACATION_ROOT = {VACATION_ROOT}")
    print(f"ANN_ROOT      = {ANN_ROOT}")
    for split in ("train", "val", "test"):
        h5_path = os.path.join(ANN_ROOT, f"vacation_{split}.h5")
        print(f"  {split}: {h5_path} -- exists: {os.path.isfile(h5_path)}")

    transform = Compose([
        Resize(img_size=IMAGE_SIZE, head_size=HEAD_SIZE),
        ToTensor(),
        Normalize(img_mean=list(IMG_MEAN), img_std=list(IMG_STD)),
    ])

    print("\nInstantiating VacationDataset_temporal(split='val')...")
    ds = VacationDataset_temporal(
        root=VACATION_ROOT,
        ann_root=ANN_ROOT,
        split="val",
        stride=1,
        transform=transform,
        tr=(0.0, 0.0),
        image_size=IMAGE_SIZE,
        head_size=HEAD_SIZE,
        heatmap_size=64,
        num_people="all",
        temporal_stride=3,
        temporal_context=0,
    )
    print(f"  {len(ds)} val paths found.")

    print("\nPulling item 0...")
    sample = ds[0]
    print(f"  path         = {sample['path']}")
    print(f"  pids         = {sample['pids']}")
    print(f"  image.shape  = {sample['image'].shape}")
    print(f"  gaze_pts.shape = {sample['gaze_pts'].shape}")
    print(f"  inout        = {sample['inout']}")
    print(f"  lah_labels (should be all -1) = {sample['lah_labels'].unique().tolist()}")

    print("\nPulling a few more, checking for crashes across a real range...")
    for i in [1, 2, len(ds) // 2, len(ds) - 1]:
        s = ds[i]
        print(f"  item {i}: path={s['path']}, pids={s['pids'].tolist()}")

    print("\nDone -- no exceptions raised.")


if __name__ == "__main__":
    main()
