# coding=utf-8

"""Standalone MTGS inference script for VACATION.

Ported from export_childplay_pseudolabels.py (post-fix version) --
VacationDataset_temporal reads from vacation_{split}.h5, the same
H5-native VSGaze family as ChildPlay's and UCO-LAEO's classes.

NORMALIZATION: mtgs.utils.image.IMG_MEAN/IMG_STD, same confirmed source
used by ChildPlay's real class and matching config.yaml's own block
exactly (see the ChildPlay script's own docstring for the full
reasoning -- VAT is the one dataset with genuinely different,
deliberately separate normalization; VACATION is not VAT).

PATH FORMAT: "images/{video_id}/{video_id}_{frame}.jpg" -- confirmed
against VacationDataset_temporal's own _parse_path() (video_id =
path.split("/")[1], frame = the token after the LAST underscore in the
filename, not the whole stem). Mirrored here exactly, not
reconstructed independently, to guarantee agreement with what the
dataset class itself expects.

SUBJECT FIELD TYPE MISMATCH -- A REAL ADAPTATION, NOT A COPY-PASTE:
labels_vacation_full.jsonl's own "subject" field is the raw
"Person1"-style string (copied straight from frame_manifest_vacation.csv,
which itself carries build_vacation_frame_manifest.py's own `"subject":
label` convention) -- NOT a bare integer the way UCO-LAEO's reference
file's subject field is. The H5's own person_ids array, however, IS
integers (build_vacation_vsgaze_h5.py's own `int(m.group(1))`
conversion). This script converts "PersonN" -> N via the same regex
convention already used in vacation_temporal_utils.py and
build_vacation_vsgaze_h5.py, rather than assuming a UCO-LAEO-style
direct int() cast would work here -- it would raise on "Person1"
directly.

REFERENCE POPULATION: labels_vacation_full.jsonl, confirmed complete and
internally consistent (162,204 total rows: 139,498 "teacher" + 22,706
"gt_fallback_solo" + 0 "gt_fallback" + 0 "gt" -- arithmetic verified to
close exactly against the manifest total, see 2026-08-02 session notes).
FILTERED to exclude "gt" and "gt_fallback_solo" (teacher never invoked
for either), same corrected logic as the ChildPlay script -- keeps
"teacher", "teacher_offscreen", "gt_fallback" (teacher invoked
regardless of outcome). For this specific completed run the practical
effect is just excluding the 22,706 solo-fallback rows, since the other
two never-invoked/edge categories are empirically zero here -- but the
filter is written generally, not hardcoded to this run's specific counts.

======================================================================
PRE-FLIGHT CHECKLIST -- confirm before submitting:
======================================================================

1. IMPORT PATH: `from mtgs.datasets.vacation_temporal import
   VacationDataset_temporal` -- CONFIRMED correct (this is the actual
   file/class just built and smoke-tested successfully on HPC), not a
   guess this time.

2. config.yaml's `data.vacation.root` -- CONFIRMED correct
   (/parallel_scratch/sl02092/standard_project/data/vacation), verified
   both by the smoke test's own successful image loads and by matching
   the static pipeline's own already-proven HPC_IMAGES_ROOT.

3. REFERENCE_LABELS_PATH / MANIFEST_PATH below assume the standard
   MTGS-main-root locations, matching every other export script's own
   default convention -- confirm these are really where the files live
   if they were generated somewhere else.

4. LABEL_SOURCE FILTER: same corrected NEVER_INVOKED set as the fixed
   ChildPlay script ({"gt", "gt_fallback_solo"}), confirmed via reading
   step40_teacher_pipeline_vacation.py's own actual label_source
   assignments, not guessed.

The pids/slot-alignment fix is confirmed via VacationDataset_temporal's
own code (built this session), not re-derived here -- fourth dataset
where this exact pattern holds.
"""

import csv
import json
import os
import re

import numpy as np
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
from mtgs.datasets.vacation_temporal import VacationDataset_temporal  # see checklist item 1
from mtgs.utils.image import IMG_MEAN, IMG_STD
from mtgs.train.transforms import Compose, Resize, ToTensor, Normalize
from mtgs.utils import spatial_argmax2d, pair

import logging

logger = logging.getLogger("ExportVacationPseudolabels")

REFERENCE_LABELS_PATH = os.environ.get(
    "VACATION_REFERENCE_LABELS",
    "/parallel_scratch/sl02092/MTGS-main/labels_vacation_full.jsonl",
)
MANIFEST_PATH = os.environ.get(
    "VACATION_MANIFEST_PATH",
    "/parallel_scratch/sl02092/MTGS-main/frame_manifest_vacation.csv",
)
TEACHER_VERSION = "mtgs-static-vsgaze"
LABEL_SOURCE = "mtgs_teacher"

# Both are cases where the teacher was NEVER invoked at all -- confirmed
# via step40_teacher_pipeline_vacation.py directly, not guessed. See
# module docstring.
NEVER_INVOKED = {"gt", "gt_fallback_solo"}


def parse_vacation_path(h5_path):
    """'images/11/11_9.jpg' -> ('11', 9). Mirrors
    VacationDataset_temporal._parse_path() EXACTLY -- reused, not
    reconstructed, to guarantee agreement with the dataset class."""
    parts = h5_path.split("/")
    video_id = parts[1]
    fname = parts[-1]
    frame_num = int(fname.split("_")[-1][:-4])
    return video_id, frame_num


def subject_to_person_id(subject):
    """'Person1' -> 1. Matches build_vacation_vsgaze_h5.py's own
    conversion exactly -- see module docstring's "subject field type
    mismatch" note. Fails loud on an unexpected format rather than
    silently miscoercing."""
    m = re.match(r"Person(\d+)$", subject)
    if not m:
        raise ValueError(
            f"Unexpected subject format: {subject!r} -- expected "
            f"'Person<N>'. Stop and check rather than silently coercing."
        )
    return int(m.group(1))


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_gaze(gaze, label, head_boxes):
    """Reused pattern from every other export script -- computed
    generically rather than trusted from the reference file."""
    if gaze is None:
        return "offscreen"
    is_social = any(
        point_in_box(gaze, box)
        for other_label, box in head_boxes.items()
        if other_label != label
    )
    return "social" if is_social else "object"


def load_split_lookup(manifest_path):
    """(video_id, fname) -> split, from frame_manifest_vacation.csv's
    own split column."""
    lookup = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[(row["show"], row["fname"])] = row["split"]
    return lookup


def load_reference_population(path):
    """(video_id, fname) -> [(subject_str, frame_idx), ...], filtered to
    exclude never-invoked rows. subject_str stays as the raw "PersonN"
    string here -- converted to an integer person_id only at the point
    of use in process_frame(), keeping this function's output directly
    comparable to how the other export scripts' populations look."""
    needed = {}
    n_skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("label_source") in NEVER_INVOKED:
                n_skipped += 1
                continue
            key = (r["show"], r["fname"])
            needed.setdefault(key, []).append((r["subject"], r["frame_idx"]))
    logger.info(f"Reference population: {sum(len(v) for v in needed.values())} "
                f"teacher-eligible rows across {len(needed)} frames "
                f"({n_skipped} never-invoked rows excluded: offscreen + solo-fallback).")
    return needed


def build_dataset(cfg, split):
    image_size = pair(cfg.data.image_size)
    transform = Compose(
        [
            Resize(img_size=image_size, head_size=cfg.model.head_size),
            ToTensor(),
            Normalize(img_mean=list(IMG_MEAN), img_std=list(IMG_STD)),
        ]
    )
    dataset = VacationDataset_temporal(
        root=cfg.data.vacation.root,
        ann_root=cfg.data.ann_root,
        split=split,
        stride=1,
        transform=transform,
        tr=(0.0, 0.0),
        image_size=image_size,
        head_size=cfg.model.head_size,
        heatmap_size=cfg.data.heatmap_size,
        num_people=cfg.data.num_people,
        temporal_stride=cfg.data.temporal_stride,
        temporal_context=cfg.data.temporal_context,
    )
    if split == "train":
        # Disables BOTH self.split=="train"-gated side effects confirmed
        # present in VacationDataset_temporal's own __getitem__ (checked
        # directly, not assumed): frame jitter AND the get_shuffle_idx()
        # person-id shuffle -- same two side effects every other class
        # in this family has, same override trick disables both here too.
        dataset.split = "val"
    return dataset


def index_dataset_paths(dataset):
    """(video_id, fname) -> the real path string used internally by this
    dataset object."""
    lookup = {}
    for p in dataset.paths:
        video_id, frame_num = parse_vacation_path(p)
        fname = p.split("/")[-1]
        lookup[(video_id, fname)] = p
    return lookup


def process_frame(model, dataset, real_path, video_id, fname, needed_subjects, device):
    idx = list(dataset.paths).index(real_path)
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

    gaze_pt_pred = gaze_pt_pred[0].cpu().numpy()
    inout_pred = inout_pred[0].cpu().numpy()

    img_w, img_h = sample["img_size"][0].tolist()

    # CONFIRMED via VacationDataset_temporal's own code (built this
    # session) -- num_missing_heads unconditionally 1 when
    # num_people="all". Fourth dataset, same pattern, no longer a guess
    # for any of them.
    slot_ids = sample["pids"].tolist()

    img_annotations = dataset.annotations.get_group(real_path).iloc[0]
    pids_frame = img_annotations["person_ids"]

    # BUG FIX (2026-08-02): same root cause as the identical fix in
    # export_childplay_pseudolabels.py -- this used to be rebuilt INSIDE
    # the per-subject loop below as a single-entry dict containing only
    # that one subject's own box, so classify_gaze()'s own "is there
    # another person's box at this point" check always had zero real
    # candidates, and gaze_type was unconditionally "object" (or
    # "offscreen"), never "social". Built ONCE here instead, covering
    # every real subject in this frame, keyed by the same "PersonN"
    # string convention subject itself uses (pids_frame holds integers,
    # converted here for a consistent comparison).
    all_head_boxes_px = {}
    for other_pid, other_box in zip(pids_frame, img_annotations["head_bboxes"]):
        ox1, oy1, ox2, oy2 = [float(v) for v in other_box]
        all_head_boxes_px[f"Person{int(other_pid)}"] = (
            ox1 * img_w, oy1 * img_h, ox2 * img_w, oy2 * img_h
        )

    out_rows = []
    unmatched = 0
    for subject, frame_idx in needed_subjects:
        pid = subject_to_person_id(subject)  # see module docstring -- NOT int(subject) directly
        pid_idx = np.where(pids_frame == pid)[0]
        if len(pid_idx) == 0:
            unmatched += 1
            continue

        head_box = img_annotations["head_bboxes"][pid_idx[0]]
        gaze_pt = img_annotations["gaze_points"][pid_idx[0]]

        try:
            pi = slot_ids.index(float(pid))
        except ValueError:
            unmatched += 1
            continue
        pred_x, pred_y = gaze_pt_pred[pi].tolist()

        gt_x, gt_y = float(gaze_pt[0]), float(gaze_pt[1])
        hx1, hy1, hx2, hy2 = [float(v) for v in head_box]

        gaze_type = classify_gaze((gt_x * img_w, gt_y * img_h), subject, all_head_boxes_px)

        out_rows.append(
            {
                "show": video_id,
                "clip": video_id,
                "fname": fname,
                "subject": subject,
                "img_path": os.path.join(dataset.root, real_path),
                "frame_idx": frame_idx,
                "head_x1": round(hx1 * img_w),
                "head_y1": round(hy1 * img_h),
                "head_x2": round(hx2 * img_w),
                "head_y2": round(hy2 * img_h),
                "gaze_type": gaze_type,
                "label_source": LABEL_SOURCE,
                "teacher_version": TEACHER_VERSION,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_px_x": round(gt_x * img_w),
                "gt_px_y": round(gt_y * img_h),
                "confidence": "MTGS_INFERENCE",
                "raw_response": json.dumps({"inout_pred": float(inout_pred[pi])}),
            }
        )

    if unmatched:
        logger.warning(f"  {real_path}: {unmatched} needed subject(s) not matched, skipped")

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

    split_lookup = load_split_lookup(MANIFEST_PATH)
    needed = load_reference_population(REFERENCE_LABELS_PATH)

    needed_by_split = {"train": {}, "val": {}, "test": {}}
    unresolved = 0
    for (video_id, fname), subjects in needed.items():
        split = split_lookup.get((video_id, fname))
        if split is None:
            unresolved += 1
            continue
        needed_by_split[split][(video_id, fname)] = subjects

    if unresolved:
        logger.warning(
            f"{unresolved} reference frames had no manifest split match -- "
            f"check for a show/fname key mismatch before trusting the "
            f"rest of this run"
        )

    datasets = {}
    path_indices = {}

    def get_dataset(split):
        if split not in datasets:
            datasets[split] = build_dataset(cfg, split)
            path_indices[split] = index_dataset_paths(datasets[split])
        return datasets[split], path_indices[split]

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
        "vacation_pseudolabels",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mtgs_teacher_vacation.jsonl")

    # Resume-aware, same discipline as every other export script -- gpu/
    # gpu_risk jobs can be preempted at any time.
    already_done = set()
    last_good_offset = 0
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                already_done.add((r["show"], r["fname"], r["subject"]))
                last_good_offset += len(line)
        with open(output_path, "r+b") as f:
            f.truncate(last_good_offset)
        if already_done:
            logger.info(f"Resuming: {len(already_done)} rows already written")

    n_written = 0
    n_skipped_done = 0
    n_frames_missing = 0
    limit = os.environ.get("VACATION_EXPORT_LIMIT")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for split, split_needed in needed_by_split.items():
            if not split_needed:
                continue
            dataset, path_index = get_dataset(split)

            items = list(split_needed.items())
            if limit is not None:
                items = items[: int(limit)]

            logger.info(f"Processing {len(items)} frames from {split} split")
            for i, ((video_id, fname), subjects) in enumerate(items):
                remaining_subjects = [
                    (s, fi) for s, fi in subjects
                    if (video_id, fname, s) not in already_done
                ]
                n_skipped_done += len(subjects) - len(remaining_subjects)
                if not remaining_subjects:
                    continue

                real_path = path_index.get((video_id, fname))
                if real_path is None:
                    n_frames_missing += 1
                    logger.warning(
                        f"  ({video_id}, {fname}) not found in {split} dataset's "
                        f"own path list -- skipped"
                    )
                    continue
                rows = process_frame(
                    model, dataset, real_path, video_id, fname, remaining_subjects, device
                )
                for row in rows:
                    row["split"] = split
                    out_f.write(json.dumps(row) + "\n")
                    n_written += 1
                out_f.flush()
                if (i + 1) % 500 == 0:
                    logger.info(f"  {split}: {i + 1}/{len(items)} frames")

    logger.info(
        f"Wrote {n_written} new rows to {output_path} "
        f"({n_skipped_done} rows already done, "
        f"{n_frames_missing} reference frames not found in any dataset object)"
    )


if __name__ == "__main__":
    main()
