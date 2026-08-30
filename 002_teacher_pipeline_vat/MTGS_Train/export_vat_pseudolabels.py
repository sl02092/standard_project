# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Standalone MTGS inference script.

Generates one JSONL pseudo-label row per (show, clip, subject, frame_idx)
already present in the reference InternVL3(+GDINO) teacher labels file, so
the MTGS "specialist" teacher covers the exact same frame-subject
population as the existing teachers for a genuinely comparable evaluation.

Does not modify test_vat.sh / config.yaml / main.py. Reuses MTGSModel and
VideoAttentionTargetDataset_temporal exactly as test_vat.sh's validated
`test` task does.
"""

import csv
import json
import os

import torch

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load

import hydra
from omegaconf import DictConfig

from mtgs.config import ConfigManager
from mtgs.networks.models import MTGSModel
from mtgs.datasets.videoattentiontarget_temporal import (
    VideoAttentionTargetDataset_temporal,
    TRAIN_SHOWS,
    VAL_SHOWS,
    TEST_SHOWS,
)
from mtgs.train.transforms import Compose, Resize, ToTensor, Normalize
from mtgs.utils import spatial_argmax2d, pair

import logging

logger = logging.getLogger("ExportVATPseudolabels")

# v1, frozen production hybrid teacher. Previously pointed at a
# rejected "v2" (face-append) variant -- harmless in practice, since
# this script only reads (show,clip,fname,subject,frame_idx) here,
# never pred_x/pred_y, and v1/v2 share identical frame selection --
# but corrected for clarity.
REFERENCE_LABELS_PATH = (
    "/parallel_scratch/sl02092/MTGS-main/labels_hybrid_targeted.jsonl"
)
VAT_IMAGES_ROOT = (
    "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget/images"
)
TEACHER_VERSION = "mtgs-static-vsgaze"
LABEL_SOURCE = "mtgs_teacher"


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_gaze(gaze, label, head_boxes):
    if gaze is None:
        return "offscreen"
    is_social = any(
        point_in_box(gaze, box)
        for other_label, box in head_boxes.items()
        if other_label != label
    )
    return "social" if is_social else "object"


_clip_dir_cache = {}
_clip_subjects_cache = {}


def find_clip_annotation_dir(vat_root, show_spaces, clip):
    """Raw VAT annotations only have train/test dirs (MTGS's "val" shows are
    a curated subset of the official VAT train split) — check both."""
    key = (show_spaces, clip)
    if key in _clip_dir_cache:
        return _clip_dir_cache[key]
    for split_dir in ("train", "test"):
        candidate = os.path.join(vat_root, "annotations", split_dir, show_spaces, clip)
        if os.path.isdir(candidate):
            _clip_dir_cache[key] = candidate
            return candidate
    raise FileNotFoundError(f"No raw annotation dir found for {show_spaces}/{clip}")


def list_clip_subjects(vat_root, show_spaces, clip):
    """Every real subject track (sNN) annotated anywhere in this clip —
    scanned from the actual sNN.txt files on disk, exactly like
    Step30_clip_scanner_2.py's `subjects` dict. This deliberately does NOT
    use the H5's person_ids: some H5 ids (e.g. 1001, 1002, ...) don't
    correspond to any real sNN.txt track file."""
    key = (show_spaces, clip)
    if key in _clip_subjects_cache:
        return _clip_subjects_cache[key]
    clip_dir = find_clip_annotation_dir(vat_root, show_spaces, clip)
    subjects = sorted(
        fn[:-4] for fn in os.listdir(clip_dir) if fn.endswith(".txt")
    )
    _clip_subjects_cache[key] = subjects
    return subjects


_annotation_cache = {}


def load_subject_annotations(vat_root, show_spaces, clip, subject):
    """Read a raw VAT sNN.txt track file (fname,x1,y1,x2,y2,gx,gy per row) —
    the same source Step30_clip_scanner_2.py and the reference teacher
    labels file use — not the MTGS H5, whose head_bboxes are pre-squared
    for the model's own crop input and no longer match the original
    annotation box."""
    key = (show_spaces, clip, subject)
    if key in _annotation_cache:
        return _annotation_cache[key]

    clip_dir = find_clip_annotation_dir(vat_root, show_spaces, clip)
    ann_path = os.path.join(clip_dir, f"{subject}.txt")

    frames = {}
    with open(ann_path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            fname = row[0].strip()
            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            gx, gy = int(row[5]), int(row[6])
            gaze = (gx, gy) if gx != -1 and gy != -1 else None
            frames[fname] = {"head": (x1, y1, x2, y2), "gaze": gaze}

    _annotation_cache[key] = frames
    return frames


def load_reference_population(path):
    """Return {(show_us, clip, fname): [(subject, frame_idx), ...]}."""
    needed = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            show_us = r["show"].replace(" ", "_")
            key = (show_us, r["clip"], r["fname"])
            needed.setdefault(key, []).append((r["subject"], r["frame_idx"]))
    return needed


def scan_split_population(vat_root, dataset):
    """Every (show_us, clip, subject, fname) quadruple where `subject` has a
    genuine annotation entry for that frame in its own raw sNN.txt track
    file, restricted to the frames `dataset` itself contains (i.e. driven by
    vat_test.h5 directly, since -- unlike train/val -- there is no reference
    teacher-labels file to intersect against for the held-out test split).
    Same subject source (raw files on disk, not H5 person_ids) as the
    reference-driven GT extraction above, for the same reason: some H5 ids
    don't correspond to any real sNN.txt track."""
    pairs = []
    for path in dataset.paths:
        show_us, clip, fname = path.split("/")
        show_spaces = show_us.replace("_", " ")
        for subject in list_clip_subjects(vat_root, show_spaces, clip):
            ann = load_subject_annotations(vat_root, show_spaces, clip, subject)
            if fname in ann:
                pairs.append((show_us, clip, subject, fname))
    return pairs


def assign_frame_idx(pairs):
    """{(show_us, clip, fname): [(subject, frame_idx), ...]} where frame_idx
    is each subject's own 0-indexed sequence position within (show_us, clip,
    subject), sorted by fname -- byte-identical to
    step46_hybrid_teacher_pipeline.py's own
    `df.sort_values([...]).groupby(["show","clip","subject"]).cumcount()`,
    so a CSV manifest built with the same convention lines up with this
    export's frame_idx row-for-row for the same population."""
    pairs = sorted(pairs, key=lambda r: (r[0], r[1], r[2], r[3]))
    needed = {}
    counters = {}
    for show_us, clip, subject, fname in pairs:
        ctr_key = (show_us, clip, subject)
        frame_idx = counters.get(ctr_key, 0)
        counters[ctr_key] = frame_idx + 1
        needed.setdefault((show_us, clip, fname), []).append((subject, frame_idx))
    return needed


def build_dataset(cfg, split):
    # Match VideoAttentionTargetDataModule exactly: image_size is squared via
    # pair() before being used by either the Resize transform or the dataset
    # itself (passing the raw int instead breaks Resize's aspect-preserving
    # vs. forced-square code path and produces non-multiple-of-14 widths).
    image_size = pair(cfg.data.image_size)
    test_transform = Compose(
        [
            Resize(img_size=image_size, head_size=cfg.model.head_size),
            ToTensor(),
            Normalize(
                img_mean=[0.31072, 0.25703, 0.24182],
                img_std=[0.26028, 0.24050, 0.23851],
            ),
        ]
    )
    dataset = VideoAttentionTargetDataset_temporal(
        root=cfg.data.vat.root,
        ann_root=cfg.data.ann_root,
        split=split,
        stride=1,
        transform=test_transform,
        tr=(0.0, 0.0),
        heatmap_size=cfg.data.heatmap_size,
        num_people="all",
        temporal_context=cfg.data.temporal_context,
        temporal_stride=cfg.data.temporal_stride,
        image_size=image_size,
        head_size=cfg.model.head_size,
    )
    if split == "train":
        # Disable the two self.split=="train"-gated side effects in
        # __getitem__ (random frame jitter, person-id shuffle) so every
        # frame is processed deterministically against its own true
        # annotation. vat_train.h5 was already loaded into
        # dataset.annotations above, at construction time.
        dataset.split = "val"
    return dataset


def process_frame(model, dataset, path, needed_subjects, device, vat_root):
    idx = dataset.path_to_index[path]
    sample = dataset[idx]
    batch = {
        k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v]
        for k, v in sample.items()
    }

    with torch.no_grad():
        _, _, gaze_hm_pred, inout_pred, _, _, _ = model(batch)
        b, t, n, hm_h, hm_w = gaze_hm_pred.shape
        middle_frame_idx = t // 2
        gaze_hm_pred = gaze_hm_pred[:, middle_frame_idx, :, :, :]
        gaze_pt_pred = spatial_argmax2d(
            gaze_hm_pred.reshape(b * n, hm_h, hm_w), normalize=True
        ).view(b, n, -1)
        inout_pred = torch.sigmoid(inout_pred[:, middle_frame_idx, :])

    gaze_pt_pred = gaze_pt_pred[0].cpu().numpy()  # (n, 2)
    inout_pred = inout_pred[0].cpu().numpy()  # (n,)

    img_w, img_h = sample["img_size"][0].tolist()

    # Model-output slot index for a given person_id, ONLY used to align
    # gaze_pt_pred[pi]/inout_pred[pi] to a subject — NOT used for any
    # ground-truth value below.
    #
    # This must be sample["pids"], not the raw H5 row's person_ids array:
    # __getitem__ (videoattentiontarget_temporal.py) ALWAYS prepends one
    # extra zero-padding slot ahead of the H5's own person_ids when
    # num_people="all" (num_missing_heads=1, unconditionally), so the
    # model's actual per-person slot order is offset by +1 from the H5
    # array's own index. Using the H5 array's index directly (as an earlier
    # version of this script did) silently reads every subject's PREVIOUS
    # neighbor's prediction instead of their own — confirmed both by a
    # direct dump of sample["pids"] and by visual overlays showing two
    # different subjects in the same frame producing near-duplicate
    # predictions pointing at one of their neighbor's faces.
    slot_ids = sample["pids"].tolist()

    row = dataset.annotations.get_group(path).iloc[0]

    show_us, clip, fname = path.split("/")
    show_spaces = show_us.replace("_", " ")

    # Ground truth (head box, gaze point, gaze_type) comes straight from the
    # raw VAT sNN.txt annotation files, in their native pixel space, for
    # every real subject track annotated anywhere in this CLIP (scanned from
    # disk, not from the H5's person_ids — some H5 ids don't correspond to
    # any real sNN.txt track) that has an entry for this specific FRAME —
    # matching Step30_clip_scanner_2.py's own convention exactly, and
    # guaranteeing byte-identical GT fields to the reference teacher labels
    # file.
    pixel_head_boxes = {}  # subject label -> (x1,y1,x2,y2) pixel ints
    pixel_gaze = {}  # subject label -> (gx,gy) pixel ints, or None if offscreen
    for subject_label in list_clip_subjects(vat_root, show_spaces, clip):
        ann = load_subject_annotations(vat_root, show_spaces, clip, subject_label)
        if fname not in ann:
            continue
        frame_ann = ann[fname]
        pixel_head_boxes[subject_label] = frame_ann["head"]
        pixel_gaze[subject_label] = frame_ann["gaze"]  # None short-circuits classify_gaze

    img_path = os.path.join(VAT_IMAGES_ROOT, show_spaces, clip, fname)

    out_rows = []
    for subject, frame_idx in needed_subjects:
        pid = int(subject[1:])
        pi = slot_ids.index(float(pid))

        gaze = pixel_gaze[subject]
        if gaze is not None:
            gt_px_x, gt_px_y = gaze
        else:
            gt_px_x, gt_px_y = -1, -1
        # -1/img_w, -1/img_h — the reference file's offscreen sentinel is
        # relative to each frame's own resolution, not a fixed canonical
        # size (verified: some clips aren't 1920x1080).
        gt_x = gt_px_x / img_w
        gt_y = gt_px_y / img_h

        hx1, hy1, hx2, hy2 = pixel_head_boxes[subject]
        gaze_type = classify_gaze(gaze, subject, pixel_head_boxes)

        pred_x, pred_y = gaze_pt_pred[pi].tolist()

        out_rows.append(
            {
                "show": show_spaces,
                "clip": clip,
                "fname": fname,
                "subject": subject,
                "img_path": img_path,
                "frame_idx": frame_idx,
                "head_x1": hx1,
                "head_y1": hy1,
                "head_x2": hx2,
                "head_y2": hy2,
                "gaze_type": gaze_type,
                "label_source": LABEL_SOURCE,
                "teacher_version": TEACHER_VERSION,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_px_x": gt_px_x,
                "gt_px_y": gt_px_y,
                "confidence": "MTGS_INFERENCE",
                "raw_response": json.dumps({"inout_pred": float(inout_pred[pi])}),
            }
        )

    return out_rows


@hydra.main(
    config_path="./../mtgs/config/", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)

    device = cfg.device
    model = MTGSModel(cfg)
    ckpt = torch.load(cfg.test.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    logger.info(f"Loaded checkpoint from {cfg.test.checkpoint}")

    needed = load_reference_population(REFERENCE_LABELS_PATH)
    logger.info(f"Reference population: {len(needed)} unique frames")

    needed_by_split = {"train": {}, "val": {}}
    for (show_us, clip, fname), subjects in needed.items():
        if show_us in TRAIN_SHOWS:
            split = "train"
        elif show_us in VAL_SHOWS:
            split = "val"
        else:
            raise ValueError(f"Show {show_us} is not a train or val show")
        path = f"{show_us}/{clip}/{fname}"
        needed_by_split[split][path] = subjects

    datasets = {}

    def get_dataset(split):
        if split not in datasets:
            datasets[split] = build_dataset(cfg, split)
        return datasets[split]

    # Test split: genuinely held out, so (unlike train/val above) there is
    # no reference teacher-labels file to intersect against. vat_test.h5
    # itself IS the population -- every real subject/frame pair it contains,
    # scanned from the raw sNN.txt files on disk exactly like the train/val
    # path's own GT extraction already does (see scan_split_population).
    test_pairs = scan_split_population(cfg.data.vat.root, get_dataset("test"))
    test_needed = assign_frame_idx(test_pairs)
    logger.info(
        f"Test-split population (from vat_test.h5): {len(test_needed)} unique "
        f"frames, {sum(len(v) for v in test_needed.values())} subject-rows"
    )
    needed_by_split["test"] = {
        f"{show_us}/{clip}/{fname}": subjects
        for (show_us, clip, fname), subjects in test_needed.items()
    }

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
        "vat_pseudolabels",
    )
    os.makedirs(output_dir, exist_ok=True)
    # Train+val (reference-driven, train/test-contaminated population -- see
    # CLAUDE.md) keep writing to the existing file, byte-for-byte as before.
    # Test (vat_test.h5-driven, genuinely held-out population) is NEW and
    # goes to its own file so the two populations are never conflated.
    output_path_main = os.path.join(output_dir, "mtgs_teacher_vat.jsonl")
    output_path_test = os.path.join(output_dir, "mtgs_teacher_vat_test.jsonl")

    n_written = {"main": 0, "test": 0}
    with open(output_path_main, "w", encoding="utf-8") as out_f_main, open(
        output_path_test, "w", encoding="utf-8"
    ) as out_f_test:
        for split, split_needed in needed_by_split.items():
            if not split_needed:
                continue
            dataset = get_dataset(split)
            dataset.path_to_index = {p: i for i, p in enumerate(dataset.paths)}

            missing = [p for p in split_needed if p not in dataset.path_to_index]
            if missing:
                raise ValueError(
                    f"{len(missing)} needed paths missing from {split} dataset, "
                    f"e.g. {missing[:5]}"
                )

            items = list(split_needed.items())
            limit = os.environ.get("VAT_EXPORT_LIMIT")
            if limit is not None:
                items = items[: int(limit)]

            target_f, bucket = (
                (out_f_test, "test") if split == "test" else (out_f_main, "main")
            )

            logger.info(f"Processing {len(items)} frames from {split} split")
            for i, (path, subjects) in enumerate(items):
                rows = process_frame(
                    model, dataset, path, subjects, device, cfg.data.vat.root
                )
                for row in rows:
                    target_f.write(json.dumps(row) + "\n")
                    n_written[bucket] += 1
                if (i + 1) % 500 == 0:
                    logger.info(f"  {split}: {i + 1}/{len(items)} frames")

    logger.info(f"Wrote {n_written['main']} rows to {output_path_main}")
    logger.info(f"Wrote {n_written['test']} rows to {output_path_test}")


if __name__ == "__main__":
    main()
