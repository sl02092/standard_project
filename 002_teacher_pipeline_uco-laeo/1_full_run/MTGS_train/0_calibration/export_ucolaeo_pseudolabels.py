# coding=utf-8

"""Standalone MTGS inference script for UCO-LAEO.

Ported from export_vat_pseudolabels.py, simplified because UCO-LAEO's own
dataset class (VideoLAEODataset_temporal) reads directly from the H5 files
with split baked in by which H5 a row came from -- no TRAIN_SHOWS/VAL_SHOWS
membership logic needed, and no raw-annotation-file scanning needed (GT
comes straight from the H5's own head_bboxes/gaze_points/person_ids, the
same source build_ucolaeo_frame_manifest.py already uses).

Also simpler than the VAT script in one more way: InternVL3's step40 ran
over the ENTIRE UCO-LAEO manifest (all three splits) in a single pass, so
every row in labels_ucolaeo_full.jsonl already carries a real frame_idx --
no VAT-style separate "recompute frame_idx for the held-out test split"
step is needed here.

Per the flagship-vs-clean-split discussion: unlike VAT, MTGS was pretrained
on UCO-LAEO's FULL dataset (train+val+test), so there is no genuinely clean
subset available no matter how this is split -- there is deliberately no
VAT-style dual-output-file (main vs held-out-test) split here. One file,
full population, with the caveat disclosed in the write-up rather than
engineered around.

======================================================================
PRE-FLIGHT CHECKLIST -- confirm these before submitting, none of them
were verifiable from the files available while drafting this:
======================================================================

1. IMPORT PATH: `from mtgs.datasets.uco_laeo_temporal import
   VideoLAEODataset_temporal` is inferred from the filename you uploaded
   and the project's own naming convention (videoattentiontarget_temporal.py
   -> mtgs.datasets.videoattentiontarget_temporal). Not confirmed against
   the real package layout. One-line fix if wrong.

2. config.yaml's `data.uco_laeo.root` is STILL THE LITERAL PLACEHOLDER
   (".../UCO-LAEO # update path if needed") as uploaded. It needs a real
   path before this will run at all. Importantly, it is NOT simply the
   same UCOLAEO_HPC_IMAGES_ROOT used for the InternVL3 pipeline
   (".../standard_project/data/ucolaeodb/frames") -- VideoLAEODataset_temporal
   joins root with an extra "images_Idiap" folder internally
   (img_path = os.path.join(self.root, "images_Idiap", path)), a structure
   build_ucolaeo_frame_manifest.py's own image-loading code does not use.
   Confirm what real HPC path has an images_Idiap/ subfolder with real
   frames in it -- may be a different location entirely from the
   InternVL3 pipeline's frames folder.

3. REFERENCE_LABELS_PATH below is a guessed location + guessed field
   names (show, clip, fname, subject, frame_idx -- assumed identical to
   VAT's reference file schema per the project's "v4 schema, locked"
   convention). Not confirmed against a real row of labels_ucolaeo_full.jsonl.

4. NORMALIZATION MISMATCH, genuinely unresolved, do not silently pick one:
   config.yaml's own `image.normalization` block (mean=[0.44232, 0.40506,
   0.36457], std=[0.28674, 0.27776, 0.27995]) does NOT match the values
   export_vat_pseudolabels.py hardcodes (mean=[0.31072, 0.25703, 0.24182],
   std=[0.26028, 0.24050, 0.23851]). This script defaults to config.yaml's
   own values on the reasoning that MTGS is one shared multi-dataset
   checkpoint and config.yaml's block is presumably what it was actually
   trained with -- but VAT's script deliberately hardcoding a different
   pair suggests that choice may have been verified once and is worth
   preserving, not overridden by default here. Wrong normalization
   produces plausible-looking but systematically wrong predictions --
   this is a silent-wrongness risk, not a crash risk. CONFIRM before
   trusting output, e.g. by finding whichever value the actual MTGS
   training run used.

5. MANIFEST_PATH: assumes frame_manifest_ucolaeo.csv is unchanged from
   the version build_ucolaeo_frame_manifest.py produces (columns: show,
   clip, fname, subject, ..., split). Only used here for its split column
   -- if that script has been rerun/changed since, re-check the column
   names still match.

6. OUTPUT_DIR: mirrors VAT script's relative-path convention (two
   directories up from wherever this script itself lives on disk). Not
   confirmed where you'll actually place this file -- check the printed
   output path on a short test run before trusting it landed somewhere
   sensible.

The one thing that IS confirmed, not guessed: the pids/slot-alignment
gotcha. Read directly from VideoLAEODataset_temporal's own code --
num_missing_heads is unconditionally 1 whenever num_people="all" (which
config.yaml sets it to), identical to VAT's dataset class. slot_ids =
sample["pids"] is used below for exactly the reason VAT's script's own
comment describes: using the raw H5 person_ids array index directly would
silently read a neighboring subject's prediction instead of the intended
subject's, with no error and no obviously-wrong output.
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
from mtgs.datasets.uco_laeo_temporal import VideoLAEODataset_temporal  # see checklist item 1
from mtgs.train.transforms import Compose, Resize, ToTensor, Normalize
from mtgs.utils import spatial_argmax2d, pair

import logging

logger = logging.getLogger("ExportUcoLaeoPseudolabels")

# See checklist items 2 and 3.
REFERENCE_LABELS_PATH = os.environ.get(
    "UCOLAEO_REFERENCE_LABELS",
    "/parallel_scratch/sl02092/MTGS-main/labels_ucolaeo_full.jsonl",
)
MANIFEST_PATH = os.environ.get(
    "UCOLAEO_MANIFEST_PATH",
    "/parallel_scratch/sl02092/MTGS-main/frame_manifest_ucolaeo.csv",
)
TEACHER_VERSION = "mtgs-static-vsgaze"
LABEL_SOURCE = "mtgs_teacher"


def parse_video_and_fname(h5_path):
    """'frames/got01/000046.jpg' -> ('got01', '000046.jpg'). Deliberately
    reused verbatim from build_ucolaeo_frame_manifest.py's own already-
    verified parser rather than re-derived, and applied here to the H5
    dataset's *actual* path strings (dataset.paths) rather than to a path
    this script reconstructs itself -- sidesteps needing to know the exact
    prefix format (e.g. whether a "frames/" segment is really there) by
    never having to build that string from scratch."""
    parts = h5_path.split("/")
    return parts[-2], parts[-1]


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_gaze(gaze, label, head_boxes):
    """Reused from the VAT script rather than hardcoding "social" for every
    row. UCO-LAEO is 100% social by construction (confirmed elsewhere in
    the project), so this should always resolve to "social" -- but computing
    it generically means a row that DIDN'T resolve that way would actually
    surface as a visible surprise worth investigating, rather than being
    silently masked by an assumption baked into the code."""
    if gaze is None:
        return "offscreen"
    is_social = any(
        point_in_box(gaze, box)
        for other_label, box in head_boxes.items()
        if other_label != label
    )
    return "social" if is_social else "object"


def load_split_lookup(manifest_path):
    """(video_id, fname) -> split ("train"/"val"/"test"), read straight from
    frame_manifest_ucolaeo.csv's own split column -- which IS the dataset's
    real split (taken directly from which H5 file each row came from), not
    a recomputation. Unlike VAT's TRAIN_SHOWS/VAL_SHOWS/TEST_SHOWS
    membership check, no show-name logic is needed here at all."""
    lookup = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # show == clip == video_id for this dataset (no finer
            # subdivision exists below video level) -- either works as key.
            lookup[(row["show"], row["fname"])] = row["split"]
    return lookup


def load_reference_population(path):
    """(video_id, fname) -> [(subject, frame_idx), ...], read straight from
    labels_ucolaeo_full.jsonl. Unlike VAT's script, no separate
    assign_frame_idx() pass is needed for any split -- step40 ran over the
    full manifest (all three splits) in one pass, so every row already
    carries its own real frame_idx directly."""
    needed = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["show"], r["fname"])
            needed.setdefault(key, []).append((r["subject"], r["frame_idx"]))
    return needed


def load_existing_progress(output_path):
    """(video_id, fname, subject) -> True for every row already written by a
    prior (possibly preempted) run of this script against this same
    output_path.

    Added after checking Eureka2's own docs: gpu_risk jobs -- the partition
    this project uses throughout -- "may be preempted at any time" and are
    simply re-queued to run again later. Without this, a preempted run
    restarted from scratch would either silently redo already-completed
    work or, if main() were later changed to append blindly, produce
    duplicate rows. Every other production pipeline in this project already
    has an equivalent mechanism (step40/step46's recover_progress_from_jsonl,
    the joint retrain's latest_checkpoint.pt) -- this brings the export
    script in line with that same standard rather than being the one
    exception.

    Tolerates a partially-written final line (the realistic failure mode if
    the job is killed mid-write) by truncating the file back to the end of
    the last successfully-parsed line before any further writing happens.
    """
    done = set()
    if not os.path.exists(output_path):
        return done

    last_good_offset = 0
    with open(output_path, "rb") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break  # partial/corrupt trailing line -- stop here, don't count it
            done.add((r["show"], r["fname"], r["subject"]))
            last_good_offset += len(line)

    with open(output_path, "r+b") as f:
        f.truncate(last_good_offset)

    return done


def build_dataset(cfg, split):
    image_size = pair(cfg.data.image_size)
    # Uses config.yaml's own shared image.normalization block -- see
    # checklist item 4 before trusting this over VAT script's own hardcoded
    # (different) values.
    transform = Compose(
        [
            Resize(img_size=image_size, head_size=cfg.model.head_size),
            ToTensor(),
            Normalize(
                img_mean=list(cfg.image.normalization.mean),
                img_std=list(cfg.image.normalization.std),
            ),
        ]
    )
    dataset = VideoLAEODataset_temporal(
        root=cfg.data.uco_laeo.root,
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
        # Same trick as the VAT script: load the train H5 (needed since
        # that's where these rows live), but disable the two
        # self.split=="train"-gated side effects in __getitem__ (frame
        # jitter, person-id shuffle) so every frame is processed
        # deterministically against its own true annotation.
        dataset.split = "val"
    return dataset


def index_dataset_paths(dataset):
    """(video_id, fname) -> the real path string used internally by this
    dataset object (dataset.paths / dataset.annotations groupby key).
    Built once per dataset so lookups never have to reconstruct that
    string themselves -- see parse_video_and_fname's docstring."""
    lookup = {}
    for p in dataset.paths:
        vid, fn = parse_video_and_fname(p)
        lookup[(vid, fn)] = p
    return lookup


def process_frame(model, dataset, real_path, video_id, fname, needed_subjects, device):
    idx = list(dataset.paths).index(real_path)  # dataset has no path_to_index of its own
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

    # CONFIRMED (not guessed) via reading VideoLAEODataset_temporal directly:
    # num_missing_heads is unconditionally 1 when num_people="all", so the
    # model's per-person output slot is offset from the H5's own person_ids
    # array index by the same +1 the VAT script had to work around. Match
    # via sample["pids"], never the raw H5 array index.
    slot_ids = sample["pids"].tolist()

    # GT pulled straight from the H5 group in its native normalized [0,1]
    # space -- the same source build_ucolaeo_frame_manifest.py already uses
    # and has verified, not re-derived from that script's own already-
    # denormalized CSV output (avoids a second, possibly-inconsistent,
    # pixel-conversion step).
    img_annotations = dataset.annotations.get_group(real_path).iloc[0]
    pids_frame = img_annotations["person_ids"]

    out_rows = []
    unmatched = 0
    for subject, frame_idx in needed_subjects:
        pid = int(subject)  # UCO-LAEO subjects are bare numeric ids, unlike VAT's "sNN" labels
        pid_idx = np.where(pids_frame == pid)[0]
        if len(pid_idx) == 0:
            # Shouldn't happen -- the reference population is itself derived
            # from this same manifest/H5 population -- but skip defensively
            # and count it rather than crash on a single bad row.
            unmatched += 1
            continue

        head_box = img_annotations["head_bboxes"][pid_idx[0]]  # normalized [0,1], (x1,y1,x2,y2)
        gaze_pt = img_annotations["gaze_points"][pid_idx[0]]  # normalized [0,1], (gx,gy)

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

        out_rows.append(
            {
                "show": video_id,
                "clip": video_id,
                "fname": fname,
                "subject": subject,
                "img_path": os.path.join(dataset.root, "images_Idiap", real_path),
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
    logger.info(f"Reference population: {len(needed)} unique frames")

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
            f"check for a show/fname key mismatch between "
            f"labels_ucolaeo_full.jsonl and frame_manifest_ucolaeo.csv "
            f"before trusting the rest of this run"
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
        "ucolaeo_pseudolabels",
    )
    os.makedirs(output_dir, exist_ok=True)
    # Single output file, full population, all three splits, "split" column
    # carried per-row -- see module docstring for why there is deliberately
    # no VAT-style main/held-out-test split here.
    output_path = os.path.join(output_dir, "mtgs_teacher_ucolaeo.jsonl")

    already_done = load_existing_progress(output_path)
    if already_done:
        logger.info(
            f"Resuming from a prior run: {len(already_done)} rows already "
            f"written to {output_path}, will skip re-processing them"
        )

    n_written = 0
    n_skipped_done = 0
    n_frames_missing = 0
    limit = os.environ.get("UCOLAEO_EXPORT_LIMIT")

    # "a" (append), not "w" -- a fresh run creates the file same as "w"
    # would, but a resumed run must not truncate the already-verified rows
    # load_existing_progress just confirmed are there.
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
                    continue  # every subject in this frame already written

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
                out_f.flush()  # so a preemption mid-run loses at most the
                                # current frame, not everything since the
                                # last OS-level buffer flush
                if (i + 1) % 500 == 0:
                    logger.info(f"  {split}: {i + 1}/{len(items)} frames")

    logger.info(
        f"Wrote {n_written} new rows to {output_path} "
        f"({n_skipped_done} rows already done from a prior run, "
        f"{n_frames_missing} reference frames not found in any dataset object)"
    )


if __name__ == "__main__":
    main()
