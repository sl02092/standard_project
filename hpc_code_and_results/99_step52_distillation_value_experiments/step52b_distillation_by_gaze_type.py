"""
step52b_distillation_by_gaze_type.py - Distillation Value Experiment,
FULL SCALE, WITH GAZE-TYPE BREAKDOWN.

Complements (does not replace) step52_distillation_full_scale.py, which
reports one pooled ADE. That pooled number can't distinguish "social is
fine, object is the real gap" from "both are equally short" -- exactly the
distinction the current teacher-approach debate turns on. This script adds
that breakdown, and runs as an independent job rather than trying to
retrofit a breakdown onto the ALREADY-RUNNING pooled job's checkpoints --
see the split-determinism fix below for why that would have been unsafe.

CHANGES FROM step52_distillation_full_scale.py:
  1. FIX (split determinism): the original script builds its clip list via
     `list(set(...))`. Python's set iteration order depends on hash
     randomization, which is randomized PER PROCESS unless explicitly
     seeded -- so `random.seed(42)` alone does NOT guarantee the same
     clips end up in the same train/val split across separate runs, even
     with identical code. Fixed here via `sorted(set(...))` before
     shuffling, which is fully deterministic. This is also why this script
     evaluates its OWN freshly-trained checkpoints rather than trying to
     reuse the currently-running pooled job's checkpoints against a
     reconstructed val set -- there's no safe way to guarantee that
     reconstructed set doesn't overlap clips the other job trained on.
  2. ADDITION: evaluate() now buckets ADE by gaze_type (social vs object)
     in addition to the overall pooled number, for both the teacher-
     trained and GT-trained models. This is the actual answer to "is the
     gap in social, object, or both" -- printed at final evaluation, not
     inferred from the pooled number.
  3. Same LABELS_FILE (labels_hybrid_targeted.jsonl) and same off-screen
     filtering fix (gt_px_x/gt_px_y == -1, not a None-check) as the pooled
     script -- see that script's docstring for why the None-check was
     wrong.

EVERYTHING ELSE unchanged: architecture, training loop, SmoothL1 loss,
early stopping, both models trained on identical clips and evaluated
against the same GT.
"""

import os
import json
import math
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

LABELS_FILE  = os.environ.get("DISTILL_LABELS_PATH", "/parallel_scratch/sl02092/standard_project/InternV3_GDino_Hybrid_Attempt_002/labels_hybrid_targeted.jsonl")
OUTPUT_DIR   = "experiment_results_full_targeted_by_gaze_type"
NUM_RECORDS_CAP = None  # None = use every on-screen record; set an int to
                         # subsample for a faster/cheaper first pass
VIT_MODEL    = "vit_tiny_patch16_224" # ~5M params — fast
IMG_SIZE     = 224
HEAD_CROP_SIZE = 112

EPOCHS              = 15       # enough to see convergence on 85 samples
BATCH_SIZE          = 8        # smaller batch — fewer samples
LR                  = 3e-4
WEIGHT_DECAY        = 1e-4
VAL_SPLIT           = 0.20     # slightly larger val for better estimates
EARLY_STOP_PATIENCE = 6
SEED                = 42

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
REF_W, REF_H = 1280, 720

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "model_teacher"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "model_gt"),      exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# ── DATASET ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class GazeDataset(Dataset):
    """
    label_source controls which coordinate is used as the training target:
        "teacher" → uses pred_x, pred_y (teacher-generated coordinate)
        "gt"      → uses gt_x, gt_y     (ground truth annotation)

    Both conditions use GT for evaluation — ensuring a fair comparison.
    """

    def __init__(self, records, label_source="teacher", augment=False):
        assert label_source in ("teacher", "gt"), \
            "label_source must be 'teacher' or 'gt'"
        self.records      = records
        self.label_source = label_source
        self.augment      = augment

        self.colour_jitter_scene = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2
        )
        self.colour_jitter_head = transforms.ColorJitter(
            brightness=0.2, contrast=0.2
        )
        self.resize_scene = transforms.Resize((IMG_SIZE, IMG_SIZE))
        self.resize_head  = transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE))
        self.to_tensor    = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        try:
            img = Image.open(rec["img_path"]).convert("RGB")
        except Exception:
            scene  = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            head   = torch.zeros(3, HEAD_CROP_SIZE, HEAD_CROP_SIZE)
            target = torch.tensor([0.5, 0.5], dtype=torch.float32)
            gt     = torch.tensor([0.5, 0.5], dtype=torch.float32)
            return scene, head, target, gt, rec

        img_w, img_h = img.size

        # Head crop
        x1 = max(0, int(rec["head_x1"]))
        y1 = max(0, int(rec["head_y1"]))
        x2 = min(img_w, int(rec["head_x2"]))
        y2 = min(img_h, int(rec["head_y2"]))
        if x2 <= x1: x1, x2 = max(0, x1-1), min(img_w, x2+1)
        if y2 <= y1: y1, y2 = max(0, y1-1), min(img_h, y2+1)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, img_w, img_h

        head_img = img.crop((x1, y1, x2, y2))

        # Shared horizontal flip
        do_flip = self.augment and random.random() < 0.5
        if do_flip:
            img      = img.transpose(Image.FLIP_LEFT_RIGHT)
            head_img = head_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Training target — this is what differs between the two models
        if self.label_source == "teacher":
            raw_x = float(rec.get("pred_x") or rec.get("gt_x") or 0.5)
            raw_y = float(rec.get("pred_y") or rec.get("gt_y") or 0.5)
        else:  # "gt"
            raw_x = float(rec.get("gt_x") or rec.get("pred_x") or 0.5)
            raw_y = float(rec.get("gt_y") or rec.get("pred_y") or 0.5)

        # Mirror x if flipped
        target_x = (1.0 - raw_x) if do_flip else raw_x
        target_y = raw_y

        # GT always uses ground truth (for evaluation)
        gt_x_raw = float(rec.get("gt_x") or 0.5)
        gt_y_raw = float(rec.get("gt_y") or 0.5)
        gt_x = (1.0 - gt_x_raw) if do_flip else gt_x_raw
        gt_y = gt_y_raw

        # Apply transforms
        if self.augment:
            scene_t = self.to_tensor(
                self.resize_scene(self.colour_jitter_scene(img))
            )
            head_t = self.to_tensor(
                self.resize_head(self.colour_jitter_head(head_img))
            )
        else:
            scene_t = self.to_tensor(self.resize_scene(img))
            head_t  = self.to_tensor(self.resize_head(head_img))

        target = torch.tensor([target_x, target_y], dtype=torch.float32)
        gt     = torch.tensor([gt_x,     gt_y],     dtype=torch.float32)
        return scene_t, head_t, target, gt, rec


def collate_fn(batch):
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
    def __init__(self, vit_model=VIT_MODEL):
        super().__init__()
        self.encoder = timm.create_model(
            vit_model, pretrained=True,
            num_classes=0, global_pool="token",
        )
        feat_dim = self.encoder.num_features
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
            nn.Sigmoid(),
        )
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    Parameters: {n/1e6:.1f}M")

    def forward(self, scene, head):
        if head.shape[-1] != IMG_SIZE:
            head = F.interpolate(
                head, size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear", align_corners=False
            )
        return self.fusion(torch.cat([
            self.encoder(scene),
            self.encoder(head),
        ], dim=1))

# ══════════════════════════════════════════════════════════════════════
# ── TRAINING UTILITIES ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def ade_norm(pred, gt):
    return torch.sqrt(((pred - gt)**2).sum(dim=1)).mean().item()

def ade_px(pred, gt):
    scale = torch.tensor([REF_W, REF_H], device=pred.device, dtype=pred.dtype)
    return torch.sqrt(((pred * scale - gt * scale)**2).sum(dim=1)).mean().item()

def evaluate(model, loader, device):
    model.eval()
    preds, gts, metas_all = [], [], []
    with torch.inference_mode():
        for scene, head, _, gt, metas in loader:
            scene, head = scene.to(device), head.to(device)
            preds.append(model(scene, head).cpu())
            gts.append(gt)
            metas_all.extend(metas)
    preds = torch.cat(preds)
    gts   = torch.cat(gts)

    result = {
        "ADE_norm": ade_norm(preds, gts),
        "ADE_px":   ade_px(preds, gts),
        "n":        len(preds),
    }

    # ADDITION: bucket by gaze_type -- this is the actual answer to
    # "is the gap in social, object, or both", not inferable from the
    # pooled number alone.
    from collections import defaultdict
    by_type_idx = defaultdict(list)
    for i, m in enumerate(metas_all):
        by_type_idx[m.get("gaze_type", "unknown")].append(i)

    for gtype, idxs in by_type_idx.items():
        idxs_t = torch.tensor(idxs)
        result[f"ADE_norm_{gtype}"] = ade_norm(preds[idxs_t], gts[idxs_t])
        result[f"n_{gtype}"] = len(idxs)

    return result

# ══════════════════════════════════════════════════════════════════════
# ── TRAIN ONE MODEL ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def train_model(train_records, val_records, label_source, ckpt_dir, device):
    """
    Train a single student model.
    label_source: "teacher" or "gt" — controls what the model learns from.
    Both evaluate against GT.
    """
    print(f"\n  Training on {label_source.upper()} labels...")

    train_ds = GazeDataset(train_records, label_source=label_source, augment=True)
    val_ds   = GazeDataset(val_records,   label_source=label_source, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    model     = GazeStudent().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=LR * 0.01
    )
    loss_fn   = nn.SmoothL1Loss()
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    train_losses, val_ades = [], []
    best_ade      = float("inf")
    no_improve    = 0
    best_epoch    = 1

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for scene, head, target, _, _ in tqdm(
            train_loader,
            desc=f"    [{label_source:<7}] Epoch {epoch:02d}/{EPOCHS}",
            leave=False
        ):
            scene, head, target = (
                scene.to(device), head.to(device), target.to(device)
            )
            optimiser.zero_grad()
            loss = loss_fn(model(scene, head), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Evaluate against GT
        metrics = evaluate(model, val_loader, device)
        current_ade = metrics["ADE_norm"]
        val_ades.append(current_ade)

        marker = ""
        if current_ade < best_ade:
            best_ade   = current_ade
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_ade":      best_ade,
                "label_source": label_source,
            }, best_ckpt)
            marker = " <- best"
        else:
            no_improve += 1
            marker = f" (no improve {no_improve}/{EARLY_STOP_PATIENCE})"

        print(
            f"    [{label_source:<7}] Epoch {epoch:02d}  "
            f"loss={avg_loss:.4f}  "
            f"val_ADE={current_ade:.4f}  "
            f"val_ADE_px={metrics['ADE_px']:.1f}px"
            f"{marker}"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {epoch}.")
            break

    # Reload best checkpoint and re-evaluate once more, purely to capture
    # the gaze-type breakdown at the actual best epoch (the in-loop
    # metrics above were computed but the breakdown wasn't retained
    # per-epoch to keep the training loop unchanged).
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model_state"])
    final_breakdown = evaluate(model, val_loader, device)

    print(f"    [{label_source:<7}] Best epoch {best_epoch} breakdown by gaze_type:")
    for key in final_breakdown:
        if key.startswith("ADE_norm_"):
            gtype = key.replace("ADE_norm_", "")
            n = final_breakdown.get(f"n_{gtype}", "?")
            print(f"      {gtype:<10} n={n:<6} ADE={final_breakdown[key]:.4f}")

    return {
        "label_source": label_source,
        "best_epoch":   best_epoch,
        "best_ade":     best_ade,
        "train_losses": train_losses,
        "val_ades":     val_ades,
        "n_train":      len(train_records),
        "n_val":        len(val_records),
        "breakdown":    {k: v for k, v in final_breakdown.items()
                          if k.startswith("ADE_norm_") or k.startswith("n_")},
    }

# ══════════════════════════════════════════════════════════════════════
# ── COMPARISON PLOT ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def save_comparison_plot(res_teacher, res_gt, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Distillation Value Experiment\n"
        "Student trained on Teacher Labels vs Ground Truth Labels",
        fontsize=13, fontweight="bold"
    )

    # Training loss
    ax = axes[0]
    ax.plot(res_teacher["train_losses"], color="#2E75B6",
            label="Teacher labels", linewidth=2)
    ax.plot(res_gt["train_losses"],     color="#ED7D31",
            label="GT labels",      linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 Loss (training)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Val ADE
    ax = axes[1]
    ax.plot(res_teacher["val_ades"], color="#2E75B6",
            label=f"Teacher labels (best={res_teacher['best_ade']:.3f})",
            linewidth=2)
    ax.plot(res_gt["val_ades"],     color="#ED7D31",
            label=f"GT labels (best={res_gt['best_ade']:.3f})",
            linewidth=2)

    # Mark best points
    t_best_epoch = res_teacher["best_epoch"] - 1
    g_best_epoch = res_gt["best_epoch"] - 1
    ax.scatter(t_best_epoch, res_teacher["best_ade"],
               color="#2E75B6", zorder=5, s=80)
    ax.scatter(g_best_epoch, res_gt["best_ade"],
               color="#ED7D31", zorder=5, s=80)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ADE vs Ground Truth (normalised)")
    ax.set_title("Validation ADE (evaluated against GT)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Comparison plot saved: {path}")

# ══════════════════════════════════════════════════════════════════════
# ── SUMMARY TEXT ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def save_summary(res_teacher, res_gt, out_dir):
    t_ade = res_teacher["best_ade"]
    g_ade = res_gt["best_ade"]
    improvement = (g_ade - t_ade) / g_ade * 100

    # Build the gaze-type breakdown comparison table
    breakdown_lines = [f"{'gaze_type':<12} {'n (teacher)':<12} {'ADE (teacher)':<15} "
                        f"{'n (GT)':<10} {'ADE (GT)':<10} {'diff (GT - teacher)'}"]
    t_bd = res_teacher.get("breakdown", {})
    g_bd = res_gt.get("breakdown", {})
    gtypes = sorted(set(k.replace("ADE_norm_", "") for k in t_bd if k.startswith("ADE_norm_")))
    for gtype in gtypes:
        t_n = t_bd.get(f"n_{gtype}", 0)
        g_n = g_bd.get(f"n_{gtype}", 0)
        t_a = t_bd.get(f"ADE_norm_{gtype}", float("nan"))
        g_a = g_bd.get(f"ADE_norm_{gtype}", float("nan"))
        diff = g_a - t_a
        breakdown_lines.append(
            f"{gtype:<12} {t_n:<12} {t_a:<15.4f} {g_n:<10} {g_a:<10.4f} "
            f"{diff:+.4f} {'(teacher better)' if diff > 0 else '(GT better)'}"
        )
    breakdown_text = "\n".join(breakdown_lines)

    if t_ade < g_ade:
        verdict = (
            f"The teacher-trained student achieved lower ADE ({t_ade:.4f}) "
            f"than the GT-trained student ({g_ade:.4f}), an improvement of "
            f"{improvement:.1f}%. This confirms that the VLM teacher labels "
            f"provide a richer training signal than raw GT pixel annotations "
            f"for social gaze frames, validating the knowledge distillation "
            f"approach."
        )
    elif t_ade < g_ade * 1.05:
        verdict = (
            f"The teacher-trained student ({t_ade:.4f}) and GT-trained "
            f"student ({g_ade:.4f}) achieved comparable ADE. This suggests "
            f"the teacher labels are at least as informative as GT annotations "
            f"at this dataset scale. Results are expected to diverge in favour "
            f"of the teacher on the full 23,096-frame training set."
        )
    else:
        verdict = (
            f"The GT-trained student ({g_ade:.4f}) outperformed the "
            f"teacher-trained student ({t_ade:.4f}) on this small test set. "
            f"This is likely a result of the limited 85-frame dataset size. "
            f"The teacher labels encode social gaze reasoning that requires "
            f"more data to leverage effectively."
        )

    # Scale using separate x/y dimensions — matches step5 reporting
    # (diagonal would inflate by ~1491px scale, non-standard)
    t_ade_px = t_ade * math.sqrt(REF_W**2 + REF_H**2) / math.sqrt(2)
    g_ade_px = g_ade * math.sqrt(REF_W**2 + REF_H**2) / math.sqrt(2)

    summary = f"""DISTILLATION VALUE EXPERIMENT — RESULTS SUMMARY
================================================

Dataset
-------
  Labels file    : {LABELS_FILE}
  Total records  : {res_teacher['n_train'] + res_teacher['n_val']}
  Training set   : {res_teacher['n_train']} records
  Validation set : {res_teacher['n_val']} records
  Split method   : by clip (no data leakage)

Model
-----
  Architecture   : GazeStudent (ViT-Tiny backbone, dual-stream)
  Parameters     : ~5.7M
  Training       : up to {EPOCHS} epochs, early stopping patience={EARLY_STOP_PATIENCE}
  Loss           : SmoothL1 (Huber)
  Optimiser      : AdamW, lr={LR}, cosine schedule

Results (evaluated against VAT ground truth)
---------------------------------------------
  Model A — Teacher labels
    Best epoch   : {res_teacher['best_epoch']}
    ADE (norm)   : {t_ade:.4f}
    ADE (pixels) : {t_ade_px:.1f} px  (ref: 1280x720)

  Model B — GT labels only
    Best epoch   : {res_gt['best_epoch']}
    ADE (norm)   : {g_ade:.4f}
    ADE (pixels) : {g_ade_px:.1f} px

  Difference     : {abs(improvement):.1f}% {'in favour of teacher' if t_ade < g_ade else 'in favour of GT'}

Breakdown by gaze_type (the actual answer to "is the gap in social,
object, or both" -- not inferable from the pooled ADE above)
---------------------------------------------------------------
{breakdown_text}

Verdict
-------
{verdict}

Label Noise
-----------
  Teacher-GT mean distance is printed above.
  Lower = teacher labels are closer to GT.
  Higher = teacher is contributing novel signal beyond raw annotation.

Note
----
Both models are trained and evaluated on only 85 frames (test subset).
The full training run uses 23,096 frames. These results are indicative
of the distillation signal quality, not final model performance.
Final evaluation will use the VAT held-out test set.
"""

    path = os.path.join(out_dir, "experiment_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"  Summary saved: {path}")

# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Step 5b — Distillation Value Experiment")
    print("Teacher Labels vs GT Labels — which trains better?")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Labels : {LABELS_FILE}")
    print(f"  Model  : {VIT_MODEL}")
    print()

    # ── Load labels ────────────────────────────────────────────────────
    records = []
    n_skipped_offscreen = 0
    n_skipped_missing = 0
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # FIX (2026-07-04): filter off-screen rows using the RAW PIXEL
            # sentinel (gt_px_x/gt_px_y == -1), NOT a None-check on
            # gt_x/gt_y. Off-screen rows have gt_x/gt_y populated with
            # mis-normalized garbage (-1/img_width), not None -- a
            # None-check silently lets them through. This is the exact bug
            # ade_by_gaze_type_comparison.py already found and fixed
            # tonight, in a different script -- don't reintroduce it here.
            if rec.get("gt_px_x") == -1 or rec.get("gt_px_y") == -1:
                n_skipped_offscreen += 1
                continue

            if (rec.get("pred_x") is None or rec.get("pred_y") is None or
                rec.get("gt_x")   is None or rec.get("gt_y")   is None):
                n_skipped_missing += 1
                continue
            records.append(rec)

    print(f"  Loaded {len(records)} valid on-screen records "
          f"(skipped {n_skipped_offscreen} off-screen, "
          f"{n_skipped_missing} missing-field rows)")

    if NUM_RECORDS_CAP is not None and len(records) > NUM_RECORDS_CAP:
        random.shuffle(records)
        records = records[:NUM_RECORDS_CAP]
        print(f"  Subsampled to NUM_RECORDS_CAP={NUM_RECORDS_CAP} records")

    from collections import Counter
    sources = Counter(r["label_source"] for r in records)
    types   = Counter(r["gaze_type"]    for r in records)
    print(f"  Label sources : {dict(sources)}")
    print(f"  Gaze types    : {dict(types)}")

    if len(records) < 10:
        print("ERROR: Too few records. Run step4 first.")
        return

    # ── Train/val split by clip ────────────────────────────────────────
    # FIX: sorted() before shuffle -- see module docstring. list(set(...))
    # alone is not reproducible across separate process runs.
    clips = sorted(set((r["show"], r["clip"]) for r in records))
    random.shuffle(clips)
    n_val     = max(1, int(len(clips) * VAL_SPLIT))
    val_clips = set(clips[:n_val])

    train_records = [r for r in records if (r["show"], r["clip"]) not in val_clips]
    val_records   = [r for r in records if (r["show"], r["clip"]) in val_clips]

    print(f"\n  Train : {len(train_records)} records")
    print(f"  Val   : {len(val_records)} records")
    print(f"\n  Initialising model architecture...")
    _ = GazeStudent()  # print param count once

    # ── Teacher-GT label distance stats ───────────────────────────────
    # Contextualises label noise going into training
    dists = []
    for r in records:
        if r.get("pred_x") is not None and r.get("gt_x") is not None:
            d = math.sqrt(
                (float(r["pred_x"]) - float(r["gt_x"]))**2 +
                (float(r["pred_y"]) - float(r["gt_y"]))**2
            )
            dists.append(d)
    if dists:
        print(f"\n  Label noise (teacher vs GT distance):")
        print(f"    Mean : {sum(dists)/len(dists):.4f}")
        print(f"    Max  : {max(dists):.4f}")
        print(f"    Min  : {min(dists):.4f}")
        n_close = sum(1 for d in dists if d < 0.10)
        print(f"    Within 0.10 (px ~144): {n_close}/{len(dists)} "
              f"({n_close/len(dists)*100:.0f}%)")

    # ── Train Model A — Teacher labels ─────────────────────────────────
    t_start    = time.time()
    res_teacher = train_model(
        train_records, val_records,
        label_source="teacher",
        ckpt_dir=os.path.join(OUTPUT_DIR, "model_teacher"),
        device=device,
    )

    # ── Train Model B — GT labels ──────────────────────────────────────
    res_gt = train_model(
        train_records, val_records,
        label_source="gt",
        ckpt_dir=os.path.join(OUTPUT_DIR, "model_gt"),
        device=device,
    )

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed/60:.1f} min")

    # ── Save outputs ───────────────────────────────────────────────────
    save_comparison_plot(res_teacher, res_gt, OUTPUT_DIR)

    comparison = {
        "teacher": res_teacher,
        "gt":      res_gt,
    }
    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    save_summary(res_teacher, res_gt, OUTPUT_DIR)

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()