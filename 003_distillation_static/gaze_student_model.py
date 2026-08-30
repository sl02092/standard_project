"""
gaze_student_model.py — Shared, dependency-free home for GazeStudent and
the determinism helpers. Split out of distillation_v1.py specifically so
other scripts (temporal feature extraction, temporal training) can import
GazeStudent safely.

WHY THIS FILE EXISTS: distillation_v1.py has hard config checks at MODULE
level (e.g. `if not _datasets_env: raise ValueError(...)`) that fire the
instant it's imported, not just when run as a script -- discovered while
building the temporal extraction script, before it could cause a real
failure. Importing GazeStudent directly from distillation_v1.py would
have required every static-training env var to be set even for a script
that has nothing to do with static training. This module has zero
module-level side effects and zero required env vars -- safe to import
from anywhere.

distillation_v1.py now imports FROM this module rather than defining
GazeStudent inline -- one definition, not two copies that can drift
apart (the exact class of bug that motivated fixing
extract_temporal_frame_features_vacation.py's duplicated GazeStudent
class a few sessions ago).
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

IMG_SIZE = 224
HEAD_CROP_SIZE = 112
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# ── DATASET ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class GazeDataset(Dataset):
    """Reads a joined-record population (built by whichever caller --
    distillation_v1.py's build_full_population(), or a lightweight
    eval-only loader for a dataset that was never split, like UCO-LAEO)
    and yields (scene, head-crop, target, is_offscreen) tensors for one
    teacher condition at a time. augment=False (the eval-only case)
    skips colour-jitter and flip entirely -- deterministic, matching
    what any real evaluation needs.
    """
    CONDITION_FIELDS = {
        "gt": ("gt_x", "gt_y"),
        "teacher_hybrid": ("hybrid_pred_x", "hybrid_pred_y"),
        "teacher_internvl3_alone": ("internvl3_alone_pred_x", "internvl3_alone_pred_y"),
        "mtgs": ("mtgs_pred_x", "mtgs_pred_y"),
    }

    def __init__(self, records, condition, augment):
        assert condition in self.CONDITION_FIELDS
        self.records = records
        self.condition = condition
        self.field_x, self.field_y = self.CONDITION_FIELDS[condition]
        self.augment = augment
        self.colour_jitter_scene = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        self.colour_jitter_head = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.resize_scene = transforms.Resize((IMG_SIZE, IMG_SIZE))
        self.resize_head = transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE))
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])

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

        return {
            "scene": scene_t, "head": head_t, "target": target,
            "is_offscreen": is_offscreen,
            "gaze_type": rec.get("gaze_type", "unknown"),
            "dataset": rec.get("dataset", "unknown"),
        }


# ══════════════════════════════════════════════════════════════════════
# ── MODEL ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class GazeStudent(nn.Module):
    """Dual-ViT-encoder architecture (scene + head crop). Two outputs:
    a coordinate head (Linear(128,2) -> Sigmoid) and an in/out
    (offscreen) head (a single logit, BCEWithLogitsLoss applied by the
    caller). The coordinate head's own Linear(128,2) is what
    extract_temporal_frame_features scripts hook via
    find_final_regression_layer() (a dynamic search for any
    Linear(out_features=2), not a hardcoded attribute path) -- this
    remains correct across architecture changes precisely because that
    search is dynamic, not because the module structure is guaranteed
    stable.
    """
    def __init__(self, vit_model="vit_tiny_patch16_224"):
        super().__init__()
        self.encoder = timm.create_model(vit_model, pretrained=True, num_classes=0, global_pool="token")
        feat_dim = self.encoder.num_features
        self.shared = nn.Sequential(
            nn.Linear(feat_dim * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
        )
        self.coord_head = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())
        self.inout_head = nn.Linear(128, 1)  # raw logit -- BCEWithLogitsLoss applies sigmoid internally

    def forward(self, scene, head):
        if head.shape[-1] != IMG_SIZE:
            head = F.interpolate(head, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        feat = self.shared(torch.cat([self.encoder(scene), self.encoder(head)], dim=1))
        coord = self.coord_head(feat)
        inout_logit = self.inout_head(feat).squeeze(-1)
        return coord, inout_logit


def load_gaze_student(checkpoint_path, vit_model="vit_tiny_patch16_224", device=None):
    """Loads a GazeStudent checkpoint produced by distillation_v1.py.
    Prints the checkpoint's own recorded val_ade/epoch/condition for a
    quick sanity check that the right checkpoint was actually loaded."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GazeStudent(vit_model=vit_model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[load_gaze_student] Loaded {checkpoint_path}")
    print(f"  condition={ckpt.get('condition', 'n/a')}  val_ade={ckpt.get('val_ade', 'n/a')}  "
          f"epoch={ckpt.get('epoch', 'n/a')}  architecture={ckpt.get('architecture', 'n/a')}")
    return model


def find_final_regression_layer(model, expected_out_features=2):
    """Dynamic search for the coordinate-regression layer -- NOT a
    hardcoded attribute path. This is what keeps feature extraction
    correct even as GazeStudent's internal structure changes (e.g. the
    shared/coord_head/inout_head split vs. an earlier single-fusion-
    block version) -- confirmed compatible with both."""
    candidates = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == expected_out_features
    ]
    if not candidates:
        raise RuntimeError(
            f"No nn.Linear(out_features={expected_out_features}) found in "
            f"the model -- inspect manually before proceeding."
        )
    name, module = candidates[-1]
    print(f"[find_final_regression_layer] Using '{name}' as the coordinate "
          f"regression layer (hooking its INPUT -- the 128-dim penultimate embedding).")
    return module


class PenultimateFeatureExtractor:
    """Hooks the coordinate-regression layer's INPUT (the 128-dim shared
    embedding) via a forward-pre-hook, and also returns the model's real
    coord + in/out predictions from the same forward pass -- all three
    (embedding, predicted_xy, predicted in/out probability) come from
    ONE forward pass, no extra compute cost for caching all three."""
    def __init__(self, model):
        self.model = model
        self._captured = None
        final_layer = find_final_regression_layer(model)
        final_layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._captured = inputs[0].detach()

    @torch.no_grad()
    def extract(self, scene_t, head_t):
        coord_pred, inout_logit = self.model(scene_t, head_t)
        if self._captured is None:
            raise RuntimeError("Hook never fired -- forward() didn't reach the regression layer.")
        feat = self._captured
        self._captured = None
        inout_prob = torch.sigmoid(inout_logit)
        return (feat.squeeze(0).cpu().numpy(),
                coord_pred.squeeze(0).cpu().numpy(),
                inout_prob.squeeze(0).cpu().numpy())


# ══════════════════════════════════════════════════════════════════════
# ── DETERMINISM ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def setup_full_determinism(seed):
    """Full determinism, not just seeding random/numpy/torch. See
    distillation_v1.py's own comment history for why seed-only isn't
    trusted in this project (the temporal PoC's cuDNN fix; this week's
    UCO-LAEO inference-noise investigation)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"WARNING: use_deterministic_algorithms(True) raised: {e}")
        print("  Falling back to warn_only=True -- determinism is NOT "
              "guaranteed airtight for this run.")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
