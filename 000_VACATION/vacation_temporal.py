# coding=utf-8

"""
vacation_temporal.py — new MTGS dataset class for VACATION, adapted from
VideoLAEODataset_temporal (UCO-LAEO's). Reads vacation_{split}.h5, built
by build_vacation_vsgaze_h5.py from already-independently-verified data
(regrouping + normalization spot-checked against three real frames across
three videos, all confirmed by manual image inspection).

WHAT'S IDENTICAL TO VideoLAEODataset_temporal, UNCHANGED: the overall
__getitem__ shape, the num_missing_heads padding convention
(unconditionally 1 when num_people="all" -- confirmed identical across
three MTGS dataset classes read directly, no longer a guess for any of
them), the train-split frame-jitter + person-id-shuffle side effects and
the split="val" override trick to disable them for deterministic export,
and the general per-frame annotation-lookup/pad/transform pipeline.

TWO REAL ADAPTATIONS, NOT JUST RENAMES:

1. PATH FORMAT. UCO-LAEO's H5 paths are 2-segment with a bare-number
   filename ("{clip}/{frame:06d}.jpg"), parsed by taking the whole final
   path segment as an int directly. build_vacation_vsgaze_h5.py's own
   paths are 3-segment with a compound filename
   ("images/{video_id}/{video_id}_{frame}.jpg", confirmed against real
   spot-checked rows) -- the frame number is the token after the LAST
   underscore, not the whole filename stem. Parsing here matches that
   real format, not UCO-LAEO's simpler one. video_id plays UCO-LAEO's
   "clip" role directly (VACATION has no finer subdivision below video
   level, confirmed in vacation_temporal_manifest_schema.md).

2. NO PAIR-LEVEL ANNOTATION AT ALL. UCO-LAEO's and ChildPlay's H5 files
   carry REAL pairs/lah_pairs/laeo_pairs/coatt_pairs data (both datasets'
   whole purpose involves studying pairwise social gaze), consumed via
   get_lah_labels(). VACATION's H5 deliberately leaves these columns
   empty (build_vacation_vsgaze_h5.py's own docstring) -- VACATION's
   attention_focus is a per-person pointer to a target, not a symmetric
   pair-level relationship annotation, and there is genuinely nothing
   real to compute here. Rather than call get_lah_labels() against empty
   data and trust it degrades gracefully (unverified), lah_labels/
   laeo_labels/coatt_labels are set directly to the SAME -1-filled
   placeholder shape/dtype every class already uses for frames with no
   real data (confirmed: is_child and speaking are hardcoded -1 in
   EVERY class checked so far, VACATION's or not -- this follows that
   same already-established "not annotated, and that's fine" pattern,
   not a new one invented for this dataset.

ROOT PATH: unlike UCO-LAEO's class, which appends a hardcoded
"images_Idiap" folder between root and path, this joins root directly
against path (os.path.join(self.root, path)), matching ChildPlay's own
simpler convention -- because build_vacation_vsgaze_h5.py's own path
values already start with "images/...", self-contained. This means
cfg.data.vacation.root must be the directory that directly CONTAINS
"images/", matching the existing static pipeline's own
HPC_IMAGES_ROOT convention (".../vacation/images") -- i.e. root should
resolve to ".../vacation", not ".../vacation/images".

NO DataModule BUILT HERE, DELIBERATELY: every other class in this file
family also defines a PyTorch Lightning DataModule for full MTGS
training on that dataset. VACATION is being added purely for zero-shot
INFERENCE via an export script (mirroring export_ucolaeo_pseudolabels.py/
export_childplay_pseudolabels.py, which both instantiate the Dataset
class directly and never touch Lightning at all) -- there is no plan in
this project to actually train MTGS on VACATION. Skipped rather than
built speculatively; add one later if that plan ever changes.
"""

import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mtgs.utils import (
    generate_gaze_heatmap,
    generate_mask,
    square_bbox,
)
from mtgs.utils.social_gaze import get_shuffle_idx
from mtgs.train.transforms import (
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
)


class VacationDataset_temporal(Dataset):
    def __init__(
        self,
        root: str,
        ann_root: str,
        split: str = "train",
        stride: int = 1,
        transform=None,
        tr: Tuple[float, float] = (-0.1, 0.1),
        image_size: Tuple[int, int] = (224, 224),
        head_size: Tuple[int, int] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: Union[int, str] = 5,
        temporal_stride: int = 3,
        temporal_context: int = 2,
        aspect: bool = False,
    ):
        super().__init__()
        self.root = root
        self.ann_root = ann_root
        self.split = split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.image_size = image_size
        self.head_size = head_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.temporal_stride = temporal_stride
        self.temporal_context = temporal_context
        self.aspect = aspect

        # load annotations
        self.annotations = pd.read_hdf(
            os.path.join(ann_root, f"vacation_{self.split}.h5"), "data"
        )
        self.annotations = self.annotations.groupby("path")
        self.paths = list(self.annotations.groups.keys())
        self.paths = np.array(self.paths)
        if self.stride > 1:
            index_keep = np.arange(len(self.paths), step=self.stride)
            self.paths = self.paths[index_keep]

    @staticmethod
    def _parse_path(path):
        """'images/11/11_9.jpg' -> ('11', 9). See module docstring
        point 1 -- deliberately NOT UCO-LAEO's simpler bare-number
        parsing, confirmed against real spot-checked rows instead."""
        parts = path.split("/")
        video_id = parts[1]
        fname = parts[-1]
        frame = int(fname.split("_")[-1][:-4])
        return video_id, frame

    @staticmethod
    def _build_path(video_id, frame):
        return f"images/{video_id}/{video_id}_{frame}.jpg"

    def __getitem__(self, index):
        path = self.paths[index]
        video_id, frame = self._parse_path(path)

        # jitter frame number during training
        if self.split == "train":
            if self.temporal_context == 0:
                frame_shift = torch.randint(
                    -(self.temporal_stride // 2), self.temporal_stride // 2, (1,)
                ).item()
            else:
                frame_shift = torch.randint(
                    -(self.temporal_context * self.temporal_stride - 1),
                    self.temporal_context * self.temporal_stride,
                    (1,),
                ).item()
            frame_tmp = frame + frame_shift
            path = self._build_path(video_id, frame_tmp)
            while path not in self.annotations.groups.keys():
                if self.temporal_context == 0:
                    frame_shift = torch.randint(
                        -(self.temporal_stride // 2), self.temporal_stride // 2, (1,)
                    ).item()
                else:
                    frame_shift = torch.randint(
                        -(self.temporal_context * self.temporal_stride - 1),
                        self.temporal_context * self.temporal_stride,
                        (1,),
                    ).item()
                frame_tmp = frame + frame_shift
                path = self._build_path(video_id, frame_tmp)
            frame = frame_tmp
        img_annotations = self.annotations.get_group(path).iloc[0]

        curr_frame_nb = frame

        img_path = os.path.join(self.root, path)  # see module docstring -- no "images_Idiap" detour
        image = Image.open(img_path)
        img_w, img_h = image.size
        if self.split == "test" and self.aspect:
            dummy_sample = {}
            dummy_sample["image"] = image
            dummy_sample["heads"] = []
            dummy_sample = Resize(
                img_size=self.image_size[1], head_size=self.head_size
            )(dummy_sample)
            self.image_size = dummy_sample["image"].size

        frame_nbs = np.arange(
            curr_frame_nb - (self.temporal_stride * self.temporal_context),
            curr_frame_nb + (self.temporal_stride * self.temporal_context) + 1,
            self.temporal_stride,
        )

        person_ids = img_annotations["person_ids"]
        inout = img_annotations["inout"]

        if self.split == "train":
            shuffle_idx = get_shuffle_idx(inout)
            person_ids = person_ids[shuffle_idx]

        num_heads = len(person_ids)
        num_keep = num_heads
        if self.num_people != "all":
            batch_num_heads = self.num_people
            if num_heads > 1:
                num_keep = np.random.randint(2, min(num_heads, self.num_people) + 1)
        else:
            batch_num_heads = num_heads
        person_ids = person_ids[-num_keep:]
        num_pairs = batch_num_heads * (batch_num_heads + 1)

        self.horizontal_flip = False
        if self.split == "train" and torch.rand(1) <= 0.5:
            self.horizontal_flip = RandomHorizontalFlip(p=1)

        t_sample = {
            "image": [], "heads": [], "head_centers": [], "head_masks": [],
            "head_bboxes": [], "inout": [], "gaze_pts": [], "gaze_vecs": [],
            "gaze_heatmaps": [], "lah_labels": [], "laeo_labels": [],
            "coatt_labels": [], "speaking": [], "is_child": [],
            "num_valid_people": [], "img_size": [], "path": [],
            "dataset": "vacation",
        }
        t_sample["pids"] = torch.cat(
            [
                torch.zeros((batch_num_heads + 1 - len(person_ids),)) - 1,
                torch.from_numpy(person_ids),
            ]
        )
        for frame_nb in frame_nbs:
            path = self._build_path(video_id, frame_nb)
            if path not in self.annotations.groups.keys():
                t_sample["image"].append(
                    torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32)
                )
                t_sample["heads"].append(
                    torch.zeros((batch_num_heads + 1, 3, 224, 224), dtype=torch.float32)
                )
                t_sample["head_centers"].append(
                    torch.zeros((batch_num_heads + 1, 2), dtype=torch.float32)
                )
                t_sample["head_masks"].append(
                    torch.zeros(
                        (batch_num_heads + 1, 1, self.image_size[1], self.image_size[0]),
                        dtype=torch.float32,
                    )
                )
                t_sample["head_bboxes"].append(
                    torch.zeros((batch_num_heads + 1, 4), dtype=torch.float32)
                )
                t_sample["gaze_pts"].append(
                    torch.zeros((batch_num_heads + 1, 2), dtype=torch.float32) - 1
                )
                t_sample["gaze_vecs"].append(
                    torch.zeros((batch_num_heads + 1, 2), dtype=torch.float32)
                )
                t_sample["gaze_heatmaps"].append(
                    torch.zeros(
                        (batch_num_heads + 1, self.heatmap_size, self.heatmap_size),
                        dtype=torch.float32,
                    )
                )
                t_sample["inout"].append(
                    torch.zeros((batch_num_heads + 1), dtype=torch.float32) - 1
                )
                t_sample["lah_labels"].append(torch.zeros((num_pairs), dtype=torch.float32) - 1)
                t_sample["laeo_labels"].append(torch.zeros((num_pairs), dtype=torch.float32) - 1)
                t_sample["coatt_labels"].append(torch.zeros((num_pairs), dtype=torch.float32) - 1)
                t_sample["speaking"].append(
                    torch.zeros((batch_num_heads + 1), dtype=torch.float32) - 1
                )
                t_sample["is_child"].append(
                    torch.zeros((batch_num_heads + 1), dtype=torch.float32) - 1
                )
                t_sample["num_valid_people"].append(torch.zeros(1, dtype=torch.long))
                t_sample["img_size"].append(torch.zeros(2, dtype=torch.long))
                t_sample["path"].append("")
            else:
                img_path = os.path.join(self.root, path)
                image = Image.open(img_path)
                img_w, img_h = image.size

                img_annotations = self.annotations.get_group(path).iloc[0]
                pids_frame = img_annotations["person_ids"]
                pids_selected = []
                head_bboxes = []
                gaze_pts = []
                inout = []
                for pi, pid in enumerate(person_ids):
                    pid_idx = np.where(pids_frame == pid)[0]
                    if len(pid_idx) == 0:
                        pids_selected.append(-1)
                        head_bboxes.append(torch.zeros(4, dtype=torch.float32))
                        gaze_pts.append(torch.zeros(2, dtype=torch.float32) - 1)
                        inout.append(-1)
                    else:
                        pids_selected.append(pid)
                        head_bbox = img_annotations["head_bboxes"][pid_idx]
                        head_bbox = torch.from_numpy(head_bbox.astype(np.float32)).squeeze()
                        head_bboxes.append(head_bbox)
                        gaze_pt = img_annotations["gaze_points"][pid_idx]
                        gaze_pt = torch.from_numpy(gaze_pt.astype(np.float32)).squeeze()
                        gaze_pts.append(gaze_pt)
                        io = img_annotations["inout"][pid_idx].item()
                        inout.append(io)

                pids_selected = torch.tensor(pids_selected, dtype=torch.long)
                head_bboxes = torch.stack(head_bboxes)
                head_bboxes = head_bboxes * torch.tensor(
                    [img_w, img_h, img_w, img_h], dtype=torch.float
                )
                gaze_pts = torch.stack(gaze_pts)
                inout = torch.tensor(inout, dtype=torch.float)

                if self.split == "train":
                    head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

                head_bboxes = square_bbox(head_bboxes, img_w, img_h)

                heads = []
                for head_bbox in head_bboxes:
                    heads.append(image.crop(head_bbox.int().tolist()))
                num_valid_heads = len(heads)
                num_missing_heads = (
                    max(self.num_people + 1 - num_valid_heads, 1)
                    if self.num_people != "all"
                    else 1
                )  # pad at least one person -- confirmed identical formula
                   # across VAT/ChildPlay/UCO-LAEO, third-party-verified
                   # pattern, not a guess for this fourth dataset either.

                if len(gaze_pts) > 0:
                    head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

                sample = {
                    "image": image, "heads": heads, "head_bboxes": head_bboxes,
                    "inout": inout, "gaze_pts": gaze_pts,
                    "num_valid_people": torch.tensor([num_valid_heads]),
                    "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
                    "path": path, "dataset": "vacation",
                }

                if self.transform:
                    sample = self.transform(sample)
                if self.horizontal_flip:
                    sample = self.horizontal_flip(sample)

                head_bboxes = sample["head_bboxes"]
                gaze_pts = sample["gaze_pts"]
                heads = sample["heads"]

                if num_missing_heads > 0:
                    pids_selected = torch.cat(
                        [torch.zeros((num_missing_heads,), dtype=torch.long) - 1, pids_selected]
                    )
                    head_bboxes = torch.cat(
                        [torch.zeros((num_missing_heads, 4), dtype=torch.float32), head_bboxes]
                    )
                    heads = torch.cat(
                        [torch.zeros((num_missing_heads, 3, 224, 224), dtype=torch.float32), heads]
                    )
                    gaze_pts = torch.cat(
                        [torch.zeros((num_missing_heads, 2), dtype=torch.float32) - 1, gaze_pts]
                    )
                    inout = torch.cat(
                        [torch.zeros((num_missing_heads,), dtype=torch.float32) - 1, inout]
                    )

                # NO get_lah_labels() CALL -- see module docstring point 2.
                # VACATION has no real pair-level annotation to compute
                # from; -1-filled placeholders here, same convention
                # already used everywhere for genuinely-unannotated fields.
                lah_labels = torch.zeros((num_pairs,), dtype=torch.float32) - 1
                laeo_labels = torch.zeros((num_pairs,), dtype=torch.float32) - 1
                coatt_labels = torch.zeros((num_pairs,), dtype=torch.float32) - 1

                head_centers = torch.hstack(
                    [
                        (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2,
                        (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2,
                    ]
                )
                sample["head_centers"] = head_centers
                sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)

                _, img_h2, img_w2 = sample["image"].shape
                sample["head_masks"] = generate_mask(head_bboxes, img_w2, img_h2)

                sample["gaze_heatmaps"] = generate_gaze_heatmap(
                    gaze_pts, sigma=self.heatmap_sigma, size=self.heatmap_size
                )

                is_child = torch.zeros(len(heads), dtype=torch.float) - 1
                speaking_scores = torch.zeros(len(heads), dtype=torch.float) - 1

                sample["lah_labels"] = lah_labels
                sample["laeo_labels"] = laeo_labels
                sample["coatt_labels"] = coatt_labels
                sample["head_bboxes"] = head_bboxes
                sample["gaze_pts"] = gaze_pts
                sample["heads"] = heads
                sample["inout"] = inout
                sample["speaking"] = speaking_scores

                t_sample["image"].append(sample["image"])
                t_sample["heads"].append(sample["heads"])
                t_sample["head_centers"].append(sample["head_centers"])
                t_sample["head_masks"].append(sample["head_masks"])
                t_sample["head_bboxes"].append(sample["head_bboxes"])
                t_sample["gaze_pts"].append(sample["gaze_pts"])
                t_sample["gaze_vecs"].append(sample["gaze_vecs"])
                t_sample["gaze_heatmaps"].append(sample["gaze_heatmaps"])
                t_sample["inout"].append(sample["inout"])
                t_sample["lah_labels"].append(sample["lah_labels"])
                t_sample["laeo_labels"].append(sample["laeo_labels"])
                t_sample["coatt_labels"].append(sample["coatt_labels"])
                t_sample["speaking"].append(sample["speaking"])
                t_sample["is_child"].append(is_child)
                t_sample["num_valid_people"].append(sample["num_valid_people"])
                t_sample["img_size"].append(sample["img_size"])
                t_sample["path"].append(path)

        for key, item in t_sample.items():
            if key not in ["dataset", "path", "pids"]:
                t_sample[key] = torch.stack(t_sample[key], axis=0).squeeze()
                if self.temporal_context == 0:
                    t_sample[key] = t_sample[key].unsqueeze(0)
        return t_sample

    def __len__(self):
        return len(self.paths)
