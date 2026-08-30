from mtgs.datasets.vacation_temporal import VacationDataset_temporal
ds = VacationDataset_temporal(root=cfg.data.vacation.root, ann_root=cfg.data.ann_root, split="val", num_people="all", temporal_context=0, temporal_stride=3, transform=your_transform)
sample = ds[0]
print(sample["pids"], sample["gaze_pts"].shape, sample["path"])