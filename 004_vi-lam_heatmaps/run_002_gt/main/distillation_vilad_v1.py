"""
distillation_vilad_v1.py — Vi-LAD: trains GazeStudent WITH the auxiliary
heatmap head (include_heatmap_head=True), adding an SSIM loss against
cached Gaze-LLE heatmap targets alongside the existing point-regression
and in/out losses. A genuinely separate script from distillation_v1.py
rather than a modification of it -- distillation_v1.py is validated,
dissertation-critical code, and this is new, untested code; keeping them
separate means zero risk to the original regardless of what happens here.

Population-building logic (load_labels_file, build_dataset_split_population,
build_full_population) is copied from distillation_v1.py, NOT imported --
same reasoning as every other verbatim copy this project: avoids
distillation_v1.py's own module-level config requirements firing just to
reuse a function. One real addition to the copy: the joined record now
also carries a "cache_key" field (matching cache_gazelle_heatmap_targets.py's
own key format exactly) so heatmap lookup is a direct dict hit, not a
re-derivation.

SCOPE (first test): gt condition, ViT-Tiny only, trained from scratch --
not fine-tuned from the existing checkpoint, since the new head starts
random regardless, and training from scratch keeps this a clean,
uncontaminated comparison against the original point-only gt result.

OFFSCREEN HANDLING: the heatmap cache has NO entry for offscreen rows (no
real gaze location to draw a heatmap for -- see the caching script's own
docstring). The SSIM loss is masked to on-screen rows only, same
philosophy as the existing coord_loss mask, computed separately since the
two masks aren't guaranteed identical (a row can be on-screen but still
missing from the cache for other reasons, e.g. a GT-mismatch that
differed slightly between the TRAIN-split cache build and this run's own
population build -- checked for and logged, not assumed away).

ACADEMIC FRAMING: this is NOT a reimplementation of ViLAM (arXiv:2503.09820,
a robot-navigation costmap paper) -- it adapts ViLAM's general principle
(SSIM-based distillation of spatial structure from a stronger source)
to gaze anticipation, using a domain-appropriate teacher (Gaze-LLE) in
place of ViLAM's own VLM-navigation source. State it that way in the
write-up, not as a direct application.
"""

import os
import json
import math
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ssim

from gaze_student_model import (GazeStudent, count_params, setup_full_determinism, seeded_generator,
                                 IMG_SIZE, HEAD_CROP_SIZE, NORM_MEAN, NORM_STD)

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

DATASETS = [d.strip() for d in os.environ.get("VILAD_DATASETS", "vat,childplay,vacation").split(",")]
CONDITION = os.environ.get("VILAD_CONDITION", "gt")  # first test: gt only
VIT_MODEL = os.environ.get("VILAD_VIT_MODEL", "vit_tiny_patch16_224")

DATASET_ROOTS = {}
for _ds in DATASETS:
    env_key = f"{_ds.upper()}_LIBRARY_ROOT"
    root = os.environ.get(env_key)
    if not root:
        raise ValueError(f"{env_key} not set -- required for dataset '{_ds}'.")
    DATASET_ROOTS[_ds] = root

DATASET_FILES = {
    "vat": {"full": "labels_vat_full_{split}.jsonl", "hybrid": "labels_vat_hybrid_targeted_{split}.jsonl",
            "mtgs": "mtgs_teacher_vat_merged_{split}.jsonl"},
    "childplay": {"full": "labels_childplay_full_{split}.jsonl", "hybrid": "labels_childplay_hybrid_targeted_{split}.jsonl",
                  "mtgs": "mtgs_teacher_childplay_{split}.jsonl"},
    "vacation": {"full": "labels_vacation_full_{split}.jsonl", "hybrid": "labels_vacation_hybrid_targeted_{split}.jsonl",
                 "mtgs": "mtgs_teacher_vacation_{split}.jsonl"},
}

OUTPUT_DIR = os.environ.get("VILAD_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("VILAD_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEATMAP_CACHE_DIR = os.environ.get("VILAD_HEATMAP_CACHE_DIR", "/parallel_scratch/sl02092/standard_project/data/vilad")
HEATMAP_SIZE = 64

EPOCHS = int(os.environ.get("VILAD_EPOCHS", "15"))
BATCH_SIZE = int(os.environ.get("VILAD_BATCH_SIZE", "32"))
LR = float(os.environ.get("VILAD_LR", "3e-5"))
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6
SEED = int(os.environ.get("VILAD_SEED", "42"))
NUM_WORKERS = int(os.environ.get("VILAD_NUM_WORKERS", "4"))

GT_MISMATCH_PIXEL_TOLERANCE = float(os.environ.get("VILAD_GT_MISMATCH_PIXEL_TOLERANCE", "2.0"))
INOUT_LOSS_WEIGHT = float(os.environ.get("VILAD_INOUT_LOSS_WEIGHT", "1.0"))
# SSIM loss weight -- NOT independently validated, matches distillation_v1.py's
# own honest framing of INOUT_LOSS_WEIGHT. Started modest so the auxiliary
# task can't dominate and disrupt the already-proven point-regression
# performance; watch both loss curves separately and adjust if one dominates.
SSIM_LOSS_WEIGHT = float(os.environ.get("VILAD_SSIM_LOSS_WEIGHT", "0.2"))

os.makedirs(os.path.join(OUTPUT_DIR, f"model_{CONDITION}"), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# ── LOAD + JOIN — copied from distillation_v1.py, + cache_key added ────
# ══════════════════════════════════════════════════════════════════════

def load_labels_file(path):
    records = {}
    n_missing = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            required = ["pred_x", "pred_y", "gt_x", "gt_y", "img_path",
                        "head_x1", "head_y1", "head_x2", "head_y2",
                        "show", "clip", "fname", "subject", "gaze_type"]
            if any(rec.get(k) is None for k in required):
                n_missing += 1
                continue
            key = (str(rec["show"]), str(rec["clip"]), str(rec["fname"]), str(rec["subject"]))
            records[key] = rec
    print(f"    Loaded {path}: {len(records)} valid records (skipped {n_missing} missing-field)")
    return records


def build_dataset_split_population(dataset, split):
    files = DATASET_FILES[dataset]
    root = DATASET_ROOTS[dataset]
    full_path = os.path.join(root, files["full"].format(split=split))
    hybrid_path = os.path.join(root, files["hybrid"].format(split=split))
    mtgs_path = os.path.join(root, files["mtgs"].format(split=split))

    print(f"  [{dataset}:{split}]")
    full_recs = load_labels_file(full_path)
    hybrid_recs = load_labels_file(hybrid_path)
    mtgs_recs = load_labels_file(mtgs_path)

    common = set(full_recs) & set(hybrid_recs) & set(mtgs_recs)
    print(f"    Key alignment: {len(common)} common (full={len(full_recs)}, "
          f"hybrid={len(hybrid_recs)}, mtgs={len(mtgs_recs)})")

    gt_mismatches, missing_images = 0, 0
    joined = []
    for key in sorted(common):  # sorted, not bare iteration -- see distillation_v1.py's own comment on why
        fu, h, m = full_recs[key], hybrid_recs[key], mtgs_recs[key]
        d_fh_px = math.hypot(fu["gt_px_x"] - h["gt_px_x"], fu["gt_px_y"] - h["gt_px_y"])
        d_fm_px = math.hypot(fu["gt_px_x"] - m["gt_px_x"], fu["gt_px_y"] - m["gt_px_y"])
        if d_fh_px > GT_MISMATCH_PIXEL_TOLERANCE or d_fm_px > GT_MISMATCH_PIXEL_TOLERANCE:
            gt_mismatches += 1
            continue
        if not os.path.isfile(fu["img_path"]):
            missing_images += 1
            continue
        joined.append({
            "dataset": dataset, "cache_key": "/".join(key),
            "show": fu["show"], "clip": fu["clip"], "gaze_type": fu["gaze_type"],
            "is_offscreen": fu["gaze_type"] == "offscreen",
            "img_path": fu["img_path"],
            "head_x1": fu["head_x1"], "head_y1": fu["head_y1"],
            "head_x2": fu["head_x2"], "head_y2": fu["head_y2"],
            "gt_x": fu["gt_x"], "gt_y": fu["gt_y"],
            "hybrid_pred_x": h["pred_x"], "hybrid_pred_y": h["pred_y"],
            "internvl3_alone_pred_x": fu["pred_x"], "internvl3_alone_pred_y": fu["pred_y"],
            "mtgs_pred_x": m["pred_x"], "mtgs_pred_y": m["pred_y"],
        })

    print(f"    GT disagreements excluded (>{GT_MISMATCH_PIXEL_TOLERANCE}px): {gt_mismatches}")
    print(f"    Missing images excluded: {missing_images}")
    print(f"    Final [{dataset}:{split}] population: {len(joined)} records")
    return joined


def build_full_population():
    train_records, val_records = [], []
    for dataset in DATASETS:
        train_records.extend(build_dataset_split_population(dataset, "TRAIN"))
        val_records.extend(build_dataset_split_population(dataset, "VAL"))
    random.Random(SEED).shuffle(train_records)
    print(f"\nJOINT population: train={len(train_records)}  val={len(val_records)}")
    print(f"  train by dataset: {dict(Counter(r['dataset'] for r in train_records))}")
    return train_records, val_records


# ══════════════════════════════════════════════════════════════════════
# ── HEATMAP CACHE — memory-mapped, one dict lookup per row ─────────────
# ══════════════════════════════════════════════════════════════════════

def load_heatmap_cache():
    index_path = os.path.join(HEATMAP_CACHE_DIR, "heatmap_index.jsonl")
    heatmaps_path = os.path.join(HEATMAP_CACHE_DIR, "heatmaps.npy")
    if not os.path.isfile(index_path) or not os.path.isfile(heatmaps_path):
        raise ValueError(f"Heatmap cache not found at {HEATMAP_CACHE_DIR} "
                          f"(expected heatmap_index.jsonl and heatmaps.npy).")
    key_to_idx = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key_to_idx[row["key"]] = row["index"]
    heatmaps = np.load(heatmaps_path, mmap_mode="r")  # memory-mapped -- 1.3GB stays on disk, not all in RAM
    print(f"Loaded heatmap cache: {len(key_to_idx)} keys, array shape {heatmaps.shape}")
    return key_to_idx, heatmaps


# ══════════════════════════════════════════════════════════════════════
# ── DATASET — extends GazeDataset's own logic with heatmap lookup ──────
# ══════════════════════════════════════════════════════════════════════

class GazeDatasetWithHeatmap(torch.utils.data.Dataset):
    CONDITION_FIELDS = {
        "gt": ("gt_x", "gt_y"),
        "teacher_hybrid": ("hybrid_pred_x", "hybrid_pred_y"),
        "teacher_internvl3_alone": ("internvl3_alone_pred_x", "internvl3_alone_pred_y"),
        "mtgs": ("mtgs_pred_x", "mtgs_pred_y"),
    }

    def __init__(self, records, condition, augment, key_to_idx, heatmaps):
        assert condition in self.CONDITION_FIELDS
        self.records = records
        self.condition = condition
        self.field_x, self.field_y = self.CONDITION_FIELDS[condition]
        self.augment = augment
        self.key_to_idx = key_to_idx
        self.heatmaps = heatmaps
        self.colour_jitter_scene = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        self.colour_jitter_head = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.resize_scene = transforms.Resize((IMG_SIZE, IMG_SIZE))
        self.resize_head = transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE))
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])

        n_have_heatmap = sum(1 for r in records if r["cache_key"] in key_to_idx)
        n_onscreen = sum(1 for r in records if not r["is_offscreen"])
        print(f"    Heatmap coverage: {n_have_heatmap}/{len(records)} rows have a cached heatmap "
              f"({n_onscreen} on-screen rows total -- mismatch beyond a small margin is worth checking)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["img_path"]).convert("RGB")
        img_w, img_h = img.size
        x1 = max(0, int(rec["head_x1"])); y1 = max(0, int(rec["head_y1"]))
        x2 = min(img_w, int(rec["head_x2"])); y2 = min(img_h, int(rec["head_y2"]))
        if x2 <= x1: x1, x2 = max(0, x1 - 1), min(img_w, x2 + 1)
        if y2 <= y1: y1, y2 = max(0, y1 - 1), min(img_h, y2 + 1)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, img_w, img_h
        head_img = img.crop((x1, y1, x2, y2))

        do_flip = self.augment and random.random() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            head_img = head_img.transpose(Image.FLIP_LEFT_RIGHT)

        gt_x_raw = float(rec[self.field_x])
        gt_y_raw = float(rec[self.field_y])
        target_x = (1.0 - gt_x_raw) if do_flip else gt_x_raw
        target_y = gt_y_raw

        if self.augment:
            scene_t = self.to_tensor(self.resize_scene(self.colour_jitter_scene(img)))
            head_t = self.to_tensor(self.resize_head(self.colour_jitter_head(head_img)))
        else:
            scene_t = self.to_tensor(self.resize_scene(img))
            head_t = self.to_tensor(self.resize_head(head_img))

        target = torch.tensor([target_x, target_y], dtype=torch.float32)
        is_offscreen = torch.tensor(float(rec["is_offscreen"]), dtype=torch.float32)

        cache_idx = self.key_to_idx.get(rec["cache_key"])
        if cache_idx is not None:
            hm = np.array(self.heatmaps[cache_idx], dtype=np.float32) / 255.0  # uint8 -> [0,1]
            if do_flip:
                hm = np.ascontiguousarray(np.fliplr(hm))  # MUST flip to match the image, or the
                    # heatmap target would be misaligned with the augmented image on this sample.
            heatmap_target = torch.from_numpy(hm)
            has_heatmap = torch.tensor(1.0, dtype=torch.float32)
        else:
            heatmap_target = torch.zeros(HEATMAP_SIZE, HEATMAP_SIZE, dtype=torch.float32)
            has_heatmap = torch.tensor(0.0, dtype=torch.float32)

        return {
            "scene": scene_t, "head": head_t, "target": target,
            "is_offscreen": is_offscreen,
            "heatmap_target": heatmap_target, "has_heatmap": has_heatmap,
            "gaze_type": rec.get("gaze_type", "unknown"), "dataset": rec.get("dataset", "unknown"),
        }


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ══════════════════════════════════════════════════════════════════════
# ── EVAL (point metrics only -- same convention as distillation_v1.py) ─
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dists = []
    for batch in loader:
        scene, head, target = batch["scene"].to(device), batch["head"].to(device), batch["target"].to(device)
        is_offscreen = batch["is_offscreen"].to(device)
        coord_pred, inout_logit, _ = model(scene, head)
        on_screen_mask = is_offscreen < 0.5
        d = torch.sqrt(((coord_pred - target) ** 2).sum(dim=1))
        dists.extend(d[on_screen_mask].cpu().tolist())
    return sum(dists) / len(dists) if dists else float("inf")


# ══════════════════════════════════════════════════════════════════════
# ── TRAIN ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Condition: {CONDITION}  Model: {VIT_MODEL}")
    print(f"SSIM_LOSS_WEIGHT={SSIM_LOSS_WEIGHT}  INOUT_LOSS_WEIGHT={INOUT_LOSS_WEIGHT}")

    setup_full_determinism(SEED)

    train_records, val_records = build_full_population()
    key_to_idx, heatmaps = load_heatmap_cache()

    model = GazeStudent(VIT_MODEL, include_heatmap_head=True).to(device)
    print(f"Parameters: {count_params(model)/1e6:.2f}M")

    train_ds = GazeDatasetWithHeatmap(train_records, CONDITION, augment=True, key_to_idx=key_to_idx, heatmaps=heatmaps)
    val_ds = GazeDatasetWithHeatmap(val_records, CONDITION, augment=False, key_to_idx=key_to_idx, heatmaps=heatmaps)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), generator=seeded_generator(SEED),
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    coord_loss_fn = nn.SmoothL1Loss(reduction="none")
    inout_loss_fn = nn.BCEWithLogitsLoss()

    best_ade, best_epoch, no_improve = float("inf"), -1, 0
    epoch_history = []
    ckpt_dir = os.path.join(OUTPUT_DIR, f"model_{CONDITION}")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        coord_losses, inout_losses, ssim_losses, total_losses = [], [], [], []

        for batch in train_loader:
            scene = batch["scene"].to(device)
            head = batch["head"].to(device)
            target = batch["target"].to(device)
            is_offscreen = batch["is_offscreen"].to(device)
            heatmap_target = batch["heatmap_target"].to(device)
            has_heatmap = batch["has_heatmap"].to(device)

            optimizer.zero_grad()
            coord_pred, inout_logit, heatmap_pred = model(scene, head)

            on_screen_mask = is_offscreen < 0.5
            if on_screen_mask.any():
                per_sample_coord_loss = coord_loss_fn(coord_pred, target).sum(dim=1)
                coord_loss = per_sample_coord_loss[on_screen_mask].mean()
            else:
                coord_loss = torch.tensor(0.0, device=device)

            inout_loss = inout_loss_fn(inout_logit, is_offscreen)

            # SSIM loss, masked to rows that actually have a cached
            # heatmap -- NOT the same mask as on_screen_mask (a row can
            # be on-screen but still miss the cache for other reasons;
            # has_heatmap is the ground truth for "do we have a real
            # target here", checked directly rather than assumed).
            heatmap_mask = has_heatmap > 0.5
            if heatmap_mask.any():
                pred_masked = heatmap_pred[heatmap_mask].unsqueeze(1)    # (N,1,64,64)
                target_masked = heatmap_target[heatmap_mask].unsqueeze(1)
                ssim_val = ssim(pred_masked, target_masked, data_range=1.0, size_average=True)
                ssim_loss = 1.0 - ssim_val
            else:
                ssim_loss = torch.tensor(0.0, device=device)

            loss = coord_loss + INOUT_LOSS_WEIGHT * inout_loss + SSIM_LOSS_WEIGHT * ssim_loss

            if not torch.isfinite(loss):
                print(f"    WARNING: non-finite loss at epoch {epoch+1} -- skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            coord_losses.append(coord_loss.item())
            inout_losses.append(inout_loss.item())
            ssim_losses.append(ssim_loss.item())
            total_losses.append(loss.item())

        scheduler.step()
        val_ade = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        improved = val_ade < best_ade
        marker = " <- best" if improved else f" (no improve {no_improve+1}/{EARLY_STOP_PATIENCE})"
        print(f"  Epoch {epoch+1:02d}  coord={sum(coord_losses)/len(coord_losses):.4f}  "
              f"inout={sum(inout_losses)/len(inout_losses):.4f}  "
              f"ssim_loss={sum(ssim_losses)/len(ssim_losses):.4f}  "
              f"total={sum(total_losses)/len(total_losses):.4f}  "
              f"val_ADE={val_ade:.4f}  time={epoch_time:.1f}s{marker}")

        epoch_history.append({
            "epoch": epoch, "val_ade": val_ade,
            "coord_loss": sum(coord_losses)/len(coord_losses),
            "inout_loss": sum(inout_losses)/len(inout_losses),
            "ssim_loss": sum(ssim_losses)/len(ssim_losses),
            "wall_clock_seconds": epoch_time, "is_best": improved,
        })

        if improved:
            best_ade, best_epoch, no_improve = val_ade, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "val_ade": best_ade, "epoch": best_epoch,
                "condition": CONDITION, "architecture": VIT_MODEL, "include_heatmap_head": True,
                "ssim_loss_weight": SSIM_LOSS_WEIGHT, "datasets": DATASETS,
                "epoch_history": epoch_history,
            }, os.path.join(ckpt_dir, "best_model.pt"))
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nDone. Best val_ADE={best_ade:.4f} at epoch {best_epoch+1}")
    print(f"Compare against the original point-only {CONDITION} result via eval_static_v1.py -- "
          f"the point output is unchanged, purely additive, so that script works as-is.")


if __name__ == "__main__":
    main()
