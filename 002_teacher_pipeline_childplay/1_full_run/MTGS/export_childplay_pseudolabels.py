# coding=utf-8

"""Standalone MTGS inference script for ChildPlay.

Ported from export_ucolaeo_pseudolabels.py -- ChildPlayDataset_temporal
reads from childplay_{split}.h5, same H5-native VSGaze family as
UCO-LAEO's class, NOT VAT's raw-file-scanning approach. Confirmed by
reading childplay_temporal.py directly: same pids-offset gotcha
(num_missing_heads unconditionally 1 when num_people="all" -- third
confirmation of this pattern across VAT/UCO-LAEO/ChildPlay, no longer a
guess), same train-split frame-jitter + person-id-shuffle side effects
disabled via the same split="val" override trick, same normalized-[0,1]
H5 fields (head_bboxes, gaze_points, person_ids, inout).

NORMALIZATION -- RESOLVED, NOT A GUESS: ChildPlayDataset_temporal imports
mtgs.utils.image.IMG_MEAN/IMG_STD directly (confirmed by reading the
class's own import statement), and those values match config.yaml's own
image.normalization block exactly (both [0.44232, 0.40506, 0.36457] /
[0.28674, 0.27776, 0.27995]). Imported directly from mtgs.utils.image
here for maximum fidelity to what the real class does, rather than going
through config. VAT's own class hardcodes DIFFERENT values
([0.31072, 0.25703, 0.24182] / [0.26028, 0.24050, 0.23851]) -- confirmed
deliberate, not a bug (VAT's own MTGS results are independently verified
bit-exact against known reference figures using those values) -- do not
"fix" VAT to match this file's values.

PATH FORMAT -- A REAL DIFFERENCE FROM UCO-LAEO, HANDLED THE SAME WAY:
ChildPlayDataset_temporal's own H5 "path" column decomposes as
"images/{clip}/{show}_{frame}.jpg" where clip is the full clip
identifier (e.g. "1Ab4vLMMAbY_2354-2439") and show is clip with its
frame-range suffix stripped (confirmed against a real row: img_path
".../images/1Ab4vLMMAbY_2354-2439/1Ab4vLMMAbY_2354.jpg", frame_idx=0 for
frame 2354, frame_idx=1 for 2355 -- sequential, per-subject, matches the
established convention exactly). Same defensive approach as the UCO-LAEO
script: parse dataset.paths' own real strings using the exact logic
ChildPlayDataset_temporal itself uses, rather than reconstructing path
strings from the manifest and hoping they match.

REFERENCE POPULATION: labels_childplay_full.jsonl (step40's own output,
confirmed via a real head -5 pull to have show/clip/fname/subject/
frame_idx already present -- no assign_frame_idx() needed, same win
UCO-LAEO's port got). FILTERED to exclude BOTH "gt" and "gt_fallback_solo"
label_source values -- offscreen frames (gt) and solo frames
(gt_fallback_solo) are both cases where the teacher was NEVER INVOKED at
all (confirmed by reading step40_teacher_pipeline_childplay.py directly:
neither branch touches n_success/n_fail, and the arithmetic
n_success+n_fail+GT-direct closing exactly against the manifest total
with nothing left over confirms these are the only two such categories).
A real bug in an earlier version of this filter (fixed here before
submission) excluded only "gt", silently letting 23,152 never-invoked
solo-fallback rows into MTGS's population -- breaking the population-
consistency this filter exists to enforce. "teacher", "teacher_offscreen",
and "gt_fallback" are all kept: the teacher WAS invoked in all three,
regardless of outcome, so MTGS should get the same chance to predict on
those rows too.

======================================================================
PRE-FLIGHT CHECKLIST -- confirm before submitting:
======================================================================

1. IMPORT PATH: `from mtgs.datasets.childplay_temporal import
   ChildPlayDataset_temporal` is inferred from the filename and this
   project's own naming convention, same as the UCO-LAEO script's
   equivalent guess. One-line fix if wrong.

2. config.yaml's `data.childplay.root` is STILL THE LITERAL PLACEHOLDER
   (".../ChildPlay # update path if needed") as last seen. Needs a real
   path -- and per ChildPlayDataset_temporal's own path construction
   (`os.path.join(self.root, path)` where path already starts with
   "images/..."), this root should be the directory that directly
   CONTAINS the "images/" folder, not the images folder itself.

3. REFERENCE_LABELS_PATH below is a guessed location
   (labels_childplay_full.jsonl was being actively written by an HPC job
   at the time this was drafted, not yet complete) -- confirm the real
   final path once the job finishes.

4. LABEL_SOURCE FILTER: CONFIRMED, not guessed, via reading
   step40_teacher_pipeline_childplay.py directly and verifying the
   arithmetic closes exactly against the manifest total. Excludes
   "gt" and "gt_fallback_solo" (never invoked); keeps "teacher",
   "teacher_offscreen", "gt_fallback" (invoked regardless of outcome).

5. MANIFEST_PATH (split lookup) assumes frame_manifest_childplay.csv's
   real location and that it still has a "split" column (confirmed
   present in build_childplay_frame_manifest.py's own MANIFEST_COLUMNS
   at draft time).

The pids/slot-alignment fix is NOT a guess here -- confirmed via reading
childplay_temporal.py directly, third confirmation of this exact pattern
across three separate MTGS dataset classes now.
"""

import csv
import json
import os

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
from mtgs.datasets.childplay_temporal import ChildPlayDataset_temporal  # see checklist item 1
from mtgs.utils.image import IMG_MEAN, IMG_STD  # confirmed matches config.yaml's own block -- see module docstring
from mtgs.train.transforms import Compose, Resize, ToTensor, Normalize
from mtgs.utils import spatial_argmax2d, pair

import logging

logger = logging.getLogger("ExportChildPlayPseudolabels")

# See checklist items 2 and 3.
REFERENCE_LABELS_PATH = os.environ.get(
    "CHILDPLAY_REFERENCE_LABELS",
    "/parallel_scratch/sl02092/MTGS-main/labels_childplay_full.jsonl",
)
MANIFEST_PATH = os.environ.get(
    "CHILDPLAY_MANIFEST_PATH",
    "/parallel_scratch/sl02092/MTGS-main/frame_manifest_childplay.csv",
)
TEACHER_VERSION = "mtgs-static-vsgaze"
LABEL_SOURCE = "mtgs_teacher"


def parse_childplay_path(h5_path):
    """'images/1Ab4vLMMAbY_2354-2439/1Ab4vLMMAbY_2354.jpg' ->
    ('1Ab4vLMMAbY_2354-2439', '1Ab4vLMMAbY', 2354, '1Ab4vLMMAbY_2354.jpg').
    Mirrors ChildPlayDataset_temporal.__getitem__'s own decomposition
    EXACTLY (clip, frame = path.split('/')[1:]; clip_name =
    '_'.join(clip.split('_')[:-1]); frame = frame.split('_')[-1][:-4]) --
    reused rather than re-derived, and applied to the H5 dataset's
    ACTUAL path strings (dataset.paths), not to a path this script
    reconstructs itself."""
    _, clip, fname = h5_path.split("/")
    show = "_".join(clip.split("_")[:-1])
    frame_num = int(fname.split("_")[-1][:-4])
    return clip, show, frame_num, fname


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_gaze(gaze, label, head_boxes):
    """Reused pattern from every other export script -- computed
    generically rather than trusted from the reference file, so a
    surprising result surfaces rather than being silently masked."""
    if gaze is None:
        return "offscreen"
    is_social = any(
        point_in_box(gaze, box)
        for other_label, box in head_boxes.items()
        if other_label != label
    )
    return "social" if is_social else "object"


def load_split_lookup(manifest_path):
    """(show, fname) -> split, read from frame_manifest_childplay.csv's
    own split column -- ChildPlay's real, official split (not a
    recomputation)."""
    lookup = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[(row["show"], row["fname"])] = row["split"]
    return lookup


def load_reference_population(path):
    """(show, fname) -> [(subject, frame_idx), ...], filtered to exclude
    "gt" and "gt_fallback_solo" -- both are cases where the teacher was
    never invoked at all (see module docstring; confirmed via the real
    pipeline code, not guessed). "teacher", "teacher_offscreen", and
    "gt_fallback" are all kept -- real invocations regardless of outcome."""
    NEVER_INVOKED = {"gt", "gt_fallback_solo"}
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
    dataset = ChildPlayDataset_temporal(
        root=cfg.data.childplay.root,
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
        # Disables the SAME two self.split=="train"-gated side effects
        # (frame jitter, person-id shuffle) confirmed present in
        # ChildPlayDataset_temporal's own __getitem__, identical
        # mechanism to the UCO-LAEO and VAT scripts.
        dataset.split = "val"
    return dataset


def index_dataset_paths(dataset):
    """(show, fname) -> the real path string used internally by this
    dataset object. Built once per dataset so lookups never reconstruct
    that string themselves."""
    lookup = {}
    for p in dataset.paths:
        clip, show, frame_num, fname = parse_childplay_path(p)
        lookup[(show, fname)] = p
    return lookup


def process_frame(model, dataset, real_path, show, fname, needed_subjects, device):
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

    # CONFIRMED (not guessed) via reading childplay_temporal.py directly:
    # num_missing_heads is unconditionally 1 when num_people="all" --
    # third dataset where this exact pattern has now been confirmed by
    # reading the real code, not assumed to transfer.
    slot_ids = sample["pids"].tolist()

    img_annotations = dataset.annotations.get_group(real_path).iloc[0]
    pids_frame = img_annotations["person_ids"]

    out_rows = []
    unmatched = 0
    for subject, frame_idx in needed_subjects:
        pid = int(subject)
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

        head_boxes_px = {
            subject: (hx1 * img_w, hy1 * img_h, hx2 * img_w, hy2 * img_h)
        }
        gaze_type = classify_gaze((gt_x * img_w, gt_y * img_h), subject, head_boxes_px)

        clip_id, show_id, frame_num, _ = parse_childplay_path(real_path)

        out_rows.append(
            {
                "show": show_id,
                "clip": clip_id,
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
    for (show, fname), subjects in needed.items():
        split = split_lookup.get((show, fname))
        if split is None:
            unresolved += 1
            continue
        needed_by_split[split][(show, fname)] = subjects

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
        "childplay_pseudolabels",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mtgs_teacher_childplay.jsonl")

    # Resume-aware, same discipline as export_ucolaeo_pseudolabels.py --
    # gpu_risk/gpu jobs can be preempted at any time.
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
    limit = os.environ.get("CHILDPLAY_EXPORT_LIMIT")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for split, split_needed in needed_by_split.items():
            if not split_needed:
                continue
            dataset, path_index = get_dataset(split)

            items = list(split_needed.items())
            if limit is not None:
                items = items[: int(limit)]

            logger.info(f"Processing {len(items)} frames from {split} split")
            for i, ((show, fname), subjects) in enumerate(items):
                remaining_subjects = [
                    (s, fi) for s, fi in subjects
                    if (show, fname, s) not in already_done
                ]
                n_skipped_done += len(subjects) - len(remaining_subjects)
                if not remaining_subjects:
                    continue

                real_path = path_index.get((show, fname))
                if real_path is None:
                    n_frames_missing += 1
                    logger.warning(
                        f"  ({show}, {fname}) not found in {split} dataset's "
                        f"own path list -- skipped"
                    )
                    continue
                rows = process_frame(
                    model, dataset, real_path, show, fname, remaining_subjects, device
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
