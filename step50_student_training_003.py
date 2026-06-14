"""
Step 5 - Student Gaze Estimation Model
Trains a lightweight Vision Transformer student on the teacher labels
generated in Step 4. Evaluates against VAT ground truth annotations.

ARCHITECTURE OVERVIEW
─────────────────────
This is Stage 1: single-frame gaze estimation.
The student learns: (image, head_box) → (gaze_x, gaze_y)

Stage 2 (next week): temporal anticipation — extend to sequences.
The architecture is designed so Stage 2 is an extension, not a rewrite:
- The image encoder (ViT backbone) stays identical
- A temporal module slots in between encoder and prediction head
- The prediction head stays identical

MODEL
─────
- Backbone : ViT-Small patch16 (pretrained on ImageNet via timm)
- Head crop : subject's head region is cropped and encoded separately
- Fusion    : concat(scene_features, head_features) → MLP → (x, y)
- Parameters: ~22M (ViT-Small) — adjust VIT_MODEL below to go smaller
- Alternative: 'vit_tiny_patch16_224' (~5M params, faster, less accurate)

TRAINING
────────
- Loss      : SmoothL1 (Huber) — robust to teacher label noise
- Optimiser : AdamW with cosine LR schedule
- Augment   : random horizontal flip + colour jitter only
              (RandomResizedCrop removed — it invalidates normalised
               gaze coordinates and was only applied to the scene,
               not the target, causing silent label corruption)
- Early stopping : patience-based, halts when val ADE stops improving

EVALUATION
──────────
- ADE (Average Displacement Error) in normalised [0,1] coords
- ADE in pixels (rescaled to 1280x720)
- Per-source breakdown: teacher labels vs GT labels vs gt_fallback

KEY FIXES FROM previous version
────────────────────────────────
- FIX 1: Removed RandomResizedCrop from augmentation pipeline.
         The crop was applied to the scene image but the normalised
         gaze target coordinate was NOT updated to match the crop
         window. This silently corrupted labels — the model was
         learning to predict gaze in the uncropped frame while
         seeing a cropped frame. Flip + colour jitter are sufficient
         augmentation at this stage.
- FIX 2: Augmentation flip now uses a shared random state so that
         scene and head crop are always flipped together. Previously
         they were flipped independently (each with p=0.5), meaning
         25% of the time only one was flipped — another silent label
         corruption.
- FIX 3: Added patience-based early stopping. In the test run, best
         epoch was 2 and the model then overfitted for 8 more epochs.
         Early stopping halts training automatically and saves time.
- FIX 4: gt_fallback_solo is now included in label source breakdown
         so solo-frame labels from Step 4 Fix 4 are visible in eval.
- FIX 5: Degenerate head box (x2 <= x1 or y2 <= y1) now pads the
         box by 1px on each side before falling back, rather than
         using the whole image — avoids encoding the full frame as
         a "head crop" which confused the head encoder.

─────────────────────────────────────────────────────────────────────
TEST_MODE flag
─────────────────────────────────────────────────────────────────────
TEST_MODE = True  → trains on labels_test.jsonl  (fast, ~5-10 min)
TEST_MODE = False → trains on labels_full.jsonl  (full training run)

Usage:
    python step5_student_training.py

Requirements:
    pip install torch torchvision timm pandas tqdm matplotlib
"""

import os
import json
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    raise ImportError("Please run: pip install timm")

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

TEST_MODE = False          # ← set False when running on full labels

LABELS_FILE  = "labels_test.jsonl" if TEST_MODE else "labels_full.jsonl"
OUTPUT_DIR   = "student_test"      if TEST_MODE else "student_full"

# Model
# VIT_MODEL  = "vit_small_patch16_224"   # ~22M params
VIT_MODEL    = "vit_tiny_patch16_224"    # ~5M params — faster alternative
IMG_SIZE     = 224
HEAD_CROP_SIZE = 112    # size to resize head crop to

# Training
EPOCHS        = 10 if TEST_MODE else 30
BATCH_SIZE    = 16
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15     # fraction of clips held out for validation
SEED          = 42

# FIX 3: Early stopping patience (epochs without val ADE improvement)
EARLY_STOP_PATIENCE = 5

# Image normalisation (ImageNet stats — matches ViT pretraining)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Reference image size for pixel-space ADE reporting
REF_W, REF_H = 1280, 720

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# ── DATASET ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class GazeDataset(Dataset):
    """
    Loads frame-subject pairs from a JSONL label file.
    Each item returns:
        scene_img  : full frame, resized to IMG_SIZE x IMG_SIZE
        head_crop  : head region crop, resized to HEAD_CROP_SIZE
        target     : (pred_x, pred_y) normalised gaze coordinate
        gt         : (gt_x, gt_y) ground truth (for eval only)
        meta       : dict of metadata for analysis
    """

    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

        # Base transforms (no augmentation) — used for val and as
        # the final step in the augmentation path
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

        # FIX 1: RandomResizedCrop removed from augmentation.
        # It was only applied to the scene image, not the gaze target,
        # silently corrupting normalised coordinates.
        # Augmentation is now: shared flip + independent colour jitter.
        self.colour_jitter_scene = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2
        )
        self.colour_jitter_head = transforms.ColorJitter(
            brightness=0.2, contrast=0.2
        )

        self.resize_scene = transforms.Resize((IMG_SIZE, IMG_SIZE))
        self.resize_head  = transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load image
        try:
            img = Image.open(rec["img_path"]).convert("RGB")
        except Exception:
            # Return zeros if image missing (shouldn't happen in practice)
            scene  = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            head   = torch.zeros(3, HEAD_CROP_SIZE, HEAD_CROP_SIZE)
            target = torch.tensor([0.5, 0.5], dtype=torch.float32)
            gt     = torch.tensor([0.5, 0.5], dtype=torch.float32)
            return scene, head, target, gt, rec

        img_w, img_h = img.size

        # Head crop — clamp to image bounds
        x1 = max(0, int(rec["head_x1"]))
        y1 = max(0, int(rec["head_y1"]))
        x2 = min(img_w, int(rec["head_x2"]))
        y2 = min(img_h, int(rec["head_y2"]))

        # FIX 5: pad degenerate boxes by 1px rather than falling back
        # to the whole image — avoids encoding the full scene as "head"
        if x2 <= x1:
            x1 = max(0, x1 - 1)
            x2 = min(img_w, x2 + 1)
        if y2 <= y1:
            y1 = max(0, y1 - 1)
            y2 = min(img_h, y2 + 1)
        # Final fallback if still degenerate (very rare edge case)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, img_w, img_h

        head_img = img.crop((x1, y1, x2, y2))

        # FIX 2: shared random flip state — scene and head crop are
        # always flipped together. Previously each had independent
        # p=0.5, meaning 25% of the time only one was flipped.
        do_flip = self.augment and random.random() < 0.5

        if do_flip:
            img      = img.transpose(Image.FLIP_LEFT_RIGHT)
            head_img = head_img.transpose(Image.FLIP_LEFT_RIGHT)
            pred_x   = 1.0 - float(rec["pred_x"])
            gt_x     = 1.0 - float(rec["gt_x"]) if rec["gt_x"] is not None else 0.5
        else:
            pred_x = float(rec["pred_x"])
            gt_x   = float(rec["gt_x"]) if rec["gt_x"] is not None else 0.5

        pred_y = float(rec["pred_y"])
        gt_y   = float(rec["gt_y"]) if rec["gt_y"] is not None else 0.5

        # Apply transforms
        if self.augment:
            # Colour jitter applied independently to scene and head crop
            # (different jitter is fine — they're different regions)
            scene_img = self.colour_jitter_scene(img)
            head_img  = self.colour_jitter_head(head_img)
        else:
            scene_img = img

        scene_t = self.to_tensor_norm(self.resize_scene(scene_img))
        head_t  = self.to_tensor_norm(self.resize_head(head_img))

        target = torch.tensor([pred_x, pred_y], dtype=torch.float32)
        gt     = torch.tensor([gt_x,   gt_y],   dtype=torch.float32)

        return scene_t, head_t, target, gt, rec


def collate_fn(batch):
    """Custom collate to handle the meta dict."""
    scenes, heads, targets, gts, metas = zip(*batch)
    return (
        torch.stack(scenes),
        torch.stack(heads),
        torch.stack(targets),
        torch.stack(gts),
        list(metas),
    )

# ══════════════════════════════════════════════════════════════════════
# ── MODEL ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class GazeStudent(nn.Module):
    """
    Lightweight gaze estimation student.

    Architecture:
        scene_encoder  : ViT backbone (pretrained) → 384-dim features
        head_encoder   : same ViT backbone, shared weights → 384-dim
        fusion_mlp     : concat(768-dim) → 256 → 128 → 2 (x, y)

    Designed for Stage 2 extension:
        To add temporal modelling, wrap the encoder call in a loop
        over T frames, then pass the T×768 sequence to a temporal
        module before the fusion MLP.

    Parameters (ViT-Small): ~22M
    Parameters (ViT-Tiny):  ~5M
    """

    def __init__(self, vit_model=VIT_MODEL):
        super().__init__()

        # Shared ViT encoder for both scene and head crop
        self.encoder = timm.create_model(
            vit_model,
            pretrained=True,
            num_classes=0,          # remove classification head
            global_pool="token",    # use [CLS] token as representation
        )
        feat_dim = self.encoder.num_features  # 192 (tiny) or 384 (small)

        # ── Fusion MLP ──────────────────────────────────────────────
        # Input: concat(scene_feat, head_feat) = 2 * feat_dim
        # Output: (x, y) in [0, 1]
        #
        # NOTE for Stage 2:
        # Insert temporal module here, before the MLP.
        # e.g. self.temporal = nn.TransformerEncoder(...)
        # and change forward() to accept a sequence of fused features.

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Sigmoid(),           # clamp output to [0, 1]
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  GazeStudent initialised — {n_params/1e6:.1f}M parameters")

    def encode(self, img_tensor):
        """Encode a batch of images → feature vectors."""
        return self.encoder(img_tensor)

    def forward(self, scene, head):
        """
        scene : (B, 3, IMG_SIZE, IMG_SIZE)
        head  : (B, 3, HEAD_CROP_SIZE, HEAD_CROP_SIZE)
        returns: (B, 2) — predicted (x, y) gaze coordinates
        """
        # Resize head crop to match ViT input if needed
        if head.shape[-1] != IMG_SIZE:
            head = F.interpolate(
                head, size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear", align_corners=False
            )

        scene_feat = self.encode(scene)                          # (B, feat_dim)
        head_feat  = self.encode(head)                           # (B, feat_dim)
        fused      = torch.cat([scene_feat, head_feat], dim=1)  # (B, feat_dim*2)
        return self.fusion(fused)                                # (B, 2)

# ══════════════════════════════════════════════════════════════════════
# ── TRAINING UTILITIES ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def ade(pred, gt):
    """Average Displacement Error in normalised [0,1] space."""
    return torch.sqrt(((pred - gt) ** 2).sum(dim=1)).mean().item()


def ade_pixels(pred, gt, w=REF_W, h=REF_H):
    """ADE rescaled to pixel space."""
    pred_px = pred * torch.tensor([w, h], device=pred.device)
    gt_px   = gt   * torch.tensor([w, h], device=gt.device)
    return torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=1)).mean().item()


def save_training_plot(train_losses, val_losses, val_ades, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train loss", color="#2E75B6")
    ax1.plot(val_losses,   label="Val loss",   color="#ED7D31")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SmoothL1 Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(val_ades, label="Val ADE (norm)", color="#70AD47")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ADE (normalised)")
    ax2.set_title("Validation ADE")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120)
    plt.close()
    print(f"  Training curves saved: {out_dir}/training_curves.png")

# ══════════════════════════════════════════════════════════════════════
# ── EVALUATION ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device):
    """
    Full evaluation pass. Returns dict of metrics broken down
    by label source (teacher / gt / gt_fallback / gt_fallback_solo)
    and by gaze type (social / object / offscreen).
    """
    model.eval()
    all_pred, all_gt, all_sources, all_types = [], [], [], []

    with torch.inference_mode():
        for scene, head, target, gt, metas in loader:
            scene, head = scene.to(device), head.to(device)
            pred = model(scene, head).cpu()
            all_pred.append(pred)
            all_gt.append(gt)
            all_sources.extend([m["label_source"] for m in metas])
            all_types.extend([m["gaze_type"]    for m in metas])

    all_pred = torch.cat(all_pred)
    all_gt   = torch.cat(all_gt)

    results = {}

    # Overall
    results["ADE_norm"] = ade(all_pred, all_gt)
    results["ADE_px"]   = ade_pixels(all_pred, all_gt)
    results["n_total"]  = len(all_pred)

    # FIX 4: gt_fallback_solo added to per-source breakdown
    for source in ["teacher", "gt", "gt_fallback", "gt_fallback_solo",
                   "teacher_offscreen"]:
        mask = torch.tensor([s == source for s in all_sources])
        if mask.sum() > 0:
            results[f"ADE_norm_{source}"] = ade(all_pred[mask], all_gt[mask])
            results[f"n_{source}"]         = int(mask.sum())

    # Per gaze type
    for gtype in ["social", "object", "offscreen"]:
        mask = torch.tensor([t == gtype for t in all_types])
        if mask.sum() > 0:
            results[f"ADE_norm_{gtype}"] = ade(all_pred[mask], all_gt[mask])
            results[f"n_{gtype}"]         = int(mask.sum())

    return results

# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "TEST MODE" if TEST_MODE else "FULL MODE"

    print("=" * 60)
    print(f"Step 5 — Student Gaze Estimation  [{mode_str}]")
    print("=" * 60)
    print(f"  Device           : {device}")
    print(f"  Labels           : {LABELS_FILE}")
    print(f"  Output           : {OUTPUT_DIR}/")
    print(f"  Model            : {VIT_MODEL}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Early stop pat.  : {EARLY_STOP_PATIENCE}")
    print()

    # ── 1. Load labels ─────────────────────────────────────────────────
    print("Loading labels...")
    records = []
    with open(LABELS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Skip records with missing coordinates
                if rec.get("pred_x") is None or rec.get("pred_y") is None:
                    continue
                if rec.get("gt_x") is None or rec.get("gt_y") is None:
                    continue
                records.append(rec)

    print(f"  Loaded {len(records)} valid records")

    # Label source breakdown
    from collections import Counter
    sources = Counter(r["label_source"] for r in records)
    types   = Counter(r["gaze_type"]    for r in records)
    print(f"  Label sources : {dict(sources)}")
    print(f"  Gaze types    : {dict(types)}")

    if len(records) < 10:
        print("ERROR: Too few records to train. Check labels file.")
        return

    # ── 2. Train / val split by clip ───────────────────────────────────
    # Splitting by clip prevents the same conversation appearing in both
    # train and val, which would cause data leakage.
    clips = list(set((r["show"], r["clip"]) for r in records))
    random.shuffle(clips)
    n_val_clips   = max(1, int(len(clips) * VAL_SPLIT))
    val_clips     = set(clips[:n_val_clips])
    train_clips   = set(clips[n_val_clips:])

    train_records = [r for r in records if (r["show"], r["clip"]) in train_clips]
    val_records   = [r for r in records if (r["show"], r["clip"]) in val_clips]

    print(f"\n  Train : {len(train_records)} records ({len(train_clips)} clips)")
    print(f"  Val   : {len(val_records)} records ({len(val_clips)} clips)")

    # ── 3. Datasets and loaders ────────────────────────────────────────
    train_ds = GazeDataset(train_records, augment=True)
    val_ds   = GazeDataset(val_records,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    # ── 4. Model, optimiser, scheduler ────────────────────────────────
    print(f"\nInitialising model...")
    model = GazeStudent(vit_model=VIT_MODEL).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=LR * 0.01
    )
    loss_fn = nn.SmoothL1Loss()   # Huber loss — robust to teacher noise

    # ── 5. Training loop ───────────────────────────────────────────────
    print(f"\nTraining for up to {EPOCHS} epochs "
          f"(early stop patience={EARLY_STOP_PATIENCE})...\n")

    train_losses, val_losses, val_ades = [], [], []
    best_val_ade     = float("inf")
    best_ckpt        = os.path.join(OUTPUT_DIR, "best_model.pt")
    epochs_no_improve = 0   # FIX 3: early stopping counter

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for scene, head, target, gt, _ in tqdm(
            train_loader, desc=f"Epoch {epoch:02d} train", leave=False
        ):
            scene, head = scene.to(device), head.to(device)
            target      = target.to(device)

            optimiser.zero_grad()
            pred = model(scene, head)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        n_val_batches = 0

        with torch.inference_mode():
            for scene, head, target, gt, _ in val_loader:
                scene, head = scene.to(device), head.to(device)
                target      = target.to(device)
                pred        = model(scene, head)
                val_loss   += loss_fn(pred, target).item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_metrics  = evaluate(model, val_loader, device)
        current_ade  = val_metrics["ADE_norm"]

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ades.append(current_ade)

        # Save best checkpoint + early stopping bookkeeping
        if current_ade < best_val_ade:
            best_val_ade      = current_ade
            epochs_no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_ade":     best_val_ade,
                "val_ade_px":  val_metrics["ADE_px"],
                "config": {
                    "vit_model":      VIT_MODEL,
                    "img_size":       IMG_SIZE,
                    "head_crop_size": HEAD_CROP_SIZE,
                },
            }, best_ckpt)
            ckpt_marker = " ← best"
        else:
            epochs_no_improve += 1
            ckpt_marker = f" (no improve {epochs_no_improve}/{EARLY_STOP_PATIENCE})"

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_ADE={current_ade:.4f}  "
            f"val_ADE_px={val_metrics['ADE_px']:.1f}px"
            f"{ckpt_marker}"
        )

        # FIX 3: early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # ── 6. Final evaluation on best checkpoint ─────────────────────────
    print(f"\nLoading best checkpoint (ADE={best_val_ade:.4f})...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    final_metrics = evaluate(model, val_loader, device)

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION  [{mode_str}]")
    print(f"{'='*60}")
    print(f"  Best epoch               : {ckpt['epoch']}")
    print(f"  Overall ADE (normalised) : {final_metrics['ADE_norm']:.4f}")
    print(f"  Overall ADE (pixels)     : {final_metrics['ADE_px']:.1f} px  (ref: 1280x720)")
    print(f"  Total val samples        : {final_metrics['n_total']}")

    print(f"\n  By label source:")
    for source in ["teacher", "gt", "gt_fallback", "gt_fallback_solo",
                   "teacher_offscreen"]:
        if f"n_{source}" in final_metrics:
            print(f"    {source:<20} n={final_metrics[f'n_{source}']:<6} "
                  f"ADE={final_metrics.get(f'ADE_norm_{source}', float('nan')):.4f}")

    print(f"\n  By gaze type:")
    for gtype in ["social", "object", "offscreen"]:
        if f"n_{gtype}" in final_metrics:
            print(f"    {gtype:<20} n={final_metrics[f'n_{gtype}']:<6} "
                  f"ADE={final_metrics.get(f'ADE_norm_{gtype}', float('nan')):.4f}")

    # ── 7. Save training curves and results ────────────────────────────
    save_training_plot(train_losses, val_losses, val_ades, OUTPUT_DIR)

    results = {
        "mode":            mode_str,
        "vit_model":       VIT_MODEL,
        "epochs_trained":  len(train_losses),
        "best_epoch":      ckpt["epoch"],
        "final_metrics":   final_metrics,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "val_ades":        val_ades,
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Best model saved : {best_ckpt}")
    print(f"  Results saved    : {results_path}")

    # ── 8. Verdict ─────────────────────────────────────────────────────
    print(f"\n  Verdict:")
    ade_norm = final_metrics["ADE_norm"]
    if ade_norm < 0.10:
        print("  STRONG — ADE < 0.10, distillation is working well.")
        print("  Ready to proceed to Stage 2 (temporal anticipation).")
    elif ade_norm < 0.20:
        print("  REASONABLE — ADE 0.10-0.20, student is learning.")
        print("  Consider more data before Stage 2.")
    else:
        print("  WEAK — ADE > 0.20, student is struggling.")
        print("  Check label quality and consider rerunning Step 4.")

    if TEST_MODE:
        print(f"\n  TEST MODE complete.")
        print(f"  If results look sensible, set TEST_MODE = False in")
        print(f"  both step4 and step5, run the full teacher pipeline,")
        print(f"  then retrain the student on the full label set.")

    print("\nDone.")


if __name__ == "__main__":
    main()
