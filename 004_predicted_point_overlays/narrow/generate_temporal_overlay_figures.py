"""
generate_temporal_overlay_figures.py — dissertation figures showing gaze
anticipation in action, on real images.

HOW TO READ EACH FIGURE: left panel is the last OBSERVED frame ("now"),
right panel is the real FUTURE target frame (horizon_ms later). The gray
square marker appears at the SAME position in both panels -- it's the
static model's prediction at "now", carried forward unchanged as the
naive "did nothing" guess. On the right panel, compare gray (naive
guess) and red (temporal-anticipated prediction) against green (ground
truth, where the person actually ended up). Red closer to green than
gray = the model correctly anticipated the change.

SELECTION: scores a POOL_SIZE candidate pool first, then renders only
N_GOOD (best improvement over the naive guess), N_BAD (worst), and
N_RANDOM (from the remaining middle) -- not just N_SAMPLES random draws.
Filenames are prefixed good_/bad_/random_ accordingly.

MARKERS: circle=ground truth, square=static copy-forward (naive),
diamond=temporal-anticipated prediction -- shape AND color both carry
meaning, so this survives black-and-white printing and is colorblind-
friendlier than color alone.

HEAD BOX: drawn on the left (context) panel only -- context_head_boxes
is a confirmed manifest field. No confirmed equivalent exists for the
target frame, so the right panel is left without one rather than guess
a field name.

RUN ON HPC (needs real images + real checkpoints + torch), via apptainer.
"""

import os
import json
import random

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter

from gaze_student_model import load_gaze_student, PenultimateFeatureExtractor, IMG_SIZE, HEAD_CROP_SIZE, NORM_MEAN, NORM_STD

WINDOW_LENGTH = 8

# Robustness testing (added 2026-08-15, folded into the existing figure
# pipeline rather than a separate script -- see project notes on why:
# genuinely cheap given this script's inference machinery already exists,
# and the honest comparison IS "clean vs corrupted", which this script's
# own existing clean-prediction pipeline is the natural baseline for.
# Applied to the LAST CONTEXT FRAME only (the most recent observation) --
# a realistic scenario (a hand briefly blocking the camera, motion blur
# on the newest frame), not corrupting the whole 8-frame history, which
# would be a different and less realistic test.
CORRUPTION_TYPE = os.environ.get("CORRUPTION_TYPE", "none")  # none | blur | occlusion
BLUR_RADIUS = float(os.environ.get("BLUR_RADIUS", "8"))
OCCLUSION_FRACTION = float(os.environ.get("OCCLUSION_FRACTION", "0.3"))  # fraction of image width/height covered

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

DATASET = os.environ.get("FIGURE_DATASET")
if not DATASET:
    raise ValueError("FIGURE_DATASET not set -- one of: vat, childplay, vacation, ucolaeo.")

STATIC_CONDITION = os.environ.get("STATIC_CONDITION")
if not STATIC_CONDITION:
    raise ValueError("STATIC_CONDITION not set.")

STATIC_CHECKPOINT_ROOT = os.environ.get("STATIC_CHECKPOINT_ROOT")
if not STATIC_CHECKPOINT_ROOT:
    raise ValueError("STATIC_CHECKPOINT_ROOT not set.")
STATIC_VIT_MODEL = os.environ.get("STATIC_VIT_MODEL", "vit_tiny_patch16_224")
CHECKPOINT_PATH = os.path.join(STATIC_CHECKPOINT_ROOT, f"model_{STATIC_CONDITION}", "best_model.pt")

TEMPORAL_CHECKPOINT_DIR = os.environ.get("TEMPORAL_CHECKPOINT_DIR")
if not TEMPORAL_CHECKPOINT_DIR:
    raise ValueError("TEMPORAL_CHECKPOINT_DIR not set -- e.g. temporal_output_vittiny/gt/horizon_300ms")
TEMPORAL_ARCHITECTURE = os.environ.get("TEMPORAL_ARCHITECTURE", "GRU")
TARGET_HORIZON_MS = int(os.environ.get("TARGET_HORIZON_MS", "300"))

TEMPORAL_MANIFEST_PATH = os.environ.get("TEMPORAL_MANIFEST_PATH")
if not TEMPORAL_MANIFEST_PATH:
    raise ValueError("TEMPORAL_MANIFEST_PATH not set.")

VAT_IMAGES_BASE_PATH = os.environ.get("VAT_IMAGES_BASE_PATH", "")

POOL_SIZE = int(os.environ.get("POOL_SIZE", "150"))
N_GOOD = int(os.environ.get("N_GOOD", "5"))
N_BAD = int(os.environ.get("N_BAD", "5"))
N_RANDOM = int(os.environ.get("N_RANDOM", "5"))
SEED = int(os.environ.get("FIGURE_SEED", "42"))

OUTPUT_DIR = os.environ.get("FIGURE_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("FIGURE_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH_REMAP = {
    "childplay": (r"C:\repo\ChildPlay-gaze\images",
                  "/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze/images"),
    "vacation": (r"C:\repo\VACATION\images",
                 "/parallel_scratch/sl02092/standard_project/data/vacation/images"),
    "ucolaeo": (r"C:\repo\UCO-LAEO\ucolaeodb\frames",
                "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames"),
}

VAT_HORIZON_FRAMES_MAP = {100: 2, 300: 7, 500: 12}

# shape: "circle" | "square" | "diamond". Color AND shape both carry
# meaning -- survives black-and-white printing, more colorblind-friendly
# than color alone.
MARKERS = {
    "gt": {"color": (46, 204, 113), "shape": "circle", "label": "Ground truth"},
    "copy_forward": {"color": (149, 165, 166), "shape": "square", "label": "Static copy-forward (no anticipation)"},
    "anticipated": {"color": (231, 76, 60), "shape": "diamond", "label": "Temporal-corrected prediction"},
    "anticipated_corrupted": {"color": (155, 89, 182), "shape": "diamond", "label": "Anticipated (corrupted input)"},
}


# ══════════════════════════════════════════════════════════════════════
# ── PATH RESOLUTION — same logic as the extraction/training scripts ────
# ══════════════════════════════════════════════════════════════════════

def resolve_image_path(clip_id, raw_path):
    if DATASET == "vat":
        if not VAT_IMAGES_BASE_PATH:
            raise ValueError("VAT_IMAGES_BASE_PATH not set -- required for VAT.")
        show, clip = clip_id.split("/", 1)
        return os.path.join(VAT_IMAGES_BASE_PATH, show, clip, raw_path)
    local_prefix, hpc_prefix = PATH_REMAP.get(DATASET, (None, None))
    if local_prefix and hpc_prefix:
        normalized = raw_path.replace("\\", "/")
        normalized_prefix = local_prefix.replace("\\", "/")
        if normalized.startswith(normalized_prefix):
            return hpc_prefix.rstrip("/") + "/" + normalized[len(normalized_prefix):].lstrip("/")
    return raw_path


def horizon_filter_matches(row, target_horizon_ms):
    if DATASET == "vat":
        return row.get("horizon_frames") == VAT_HORIZON_FRAMES_MAP[target_horizon_ms]
    return row.get("target_horizon_ms") == target_horizon_ms


# ══════════════════════════════════════════════════════════════════════
# ── TEMPORAL MODEL LAYER — copied verbatim, same source as
# ── eval_temporal_v1.py / benchmark_inference.py ────────────────────────
# ══════════════════════════════════════════════════════════════════════

def make_regression_tail(input_dim, dropout=0.1):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(64, 2), nn.Sigmoid())


def make_residual_tail(input_dim, dropout=0.1):
    tail = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))
    nn.init.zeros_(tail[-1].weight)
    nn.init.zeros_(tail[-1].bias)
    return tail


def pick_nhead(input_dim, max_heads=8):
    for h in range(min(max_heads, input_dim), 0, -1):
        if input_dim % h == 0:
            return h
    return 1


class GRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.tail = make_regression_tail(hidden_dim, dropout)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.tail(h_n[-1])


class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, window_length=WINDOW_LENGTH):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                     dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_regression_tail(input_dim, dropout)

    def forward(self, x):
        out = self.encoder(x + self.pos_embed)
        return self.tail(out[:, -1, :])


class ResidualGRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.tail = make_residual_tail(hidden_dim, dropout)

    def forward(self, x):
        reference = x[:, -1, -2:]
        _, h_n = self.gru(x)
        return torch.clamp(reference + self.tail(h_n[-1]), 0.0, 1.0)


class ResidualTransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, window_length=WINDOW_LENGTH):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                     dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_residual_tail(input_dim, dropout)

    def forward(self, x):
        reference = x[:, -1, -2:]
        out = self.encoder(x + self.pos_embed)
        return torch.clamp(reference + self.tail(out[:, -1, :]), 0.0, 1.0)


TEMPORAL_ARCHS = {"GRU": GRUHead, "Transformer": TransformerHead,
                   "ResidualGRU": ResidualGRUHead, "ResidualTransformer": ResidualTransformerHead}


def load_temporal_model(architecture, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    input_dim = ckpt["input_dim"]
    cls = TEMPORAL_ARCHS[architecture]
    if architecture in ("Transformer", "ResidualTransformer"):
        model = cls(input_dim=input_dim, nhead=pick_nhead(input_dim)).to(DEVICE)
    else:
        model = cls(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════
# ── SAMPLE POOL ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def sample_pool(pool_size):
    """Diverse pool: mixes gaze_type where present, so scoring isn't
    accidentally dominated by one category."""
    by_type = {}
    with open(TEMPORAL_MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not horizon_filter_matches(row, TARGET_HORIZON_MS):
                continue
            if not row["window_valid"] or row["target_gaze_x"] is None:
                continue
            by_type.setdefault(row["target_gaze_type"], []).append(row)

    rng = random.Random(SEED)
    types = list(by_type.keys())
    per_type = max(1, pool_size // len(types))
    pool = []
    for t in types:
        candidates = by_type[t]
        pool.extend(rng.sample(candidates, min(per_type, len(candidates))))
    rng.shuffle(pool)
    return pool[:pool_size]


# ══════════════════════════════════════════════════════════════════════
# ── DRAWING ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_transforms():
    to_tensor_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    scene_t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), to_tensor_norm])
    head_t = transforms.Compose([transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE)), to_tensor_norm])
    return scene_t, head_t


def apply_corruption(img):
    """Applied to a PIL image before it's fed to the model -- returns a
    NEW image, original untouched (needed since the clean version is
    still used for the existing good/bad/random comparison, corruption
    is an addition, not a replacement)."""
    if CORRUPTION_TYPE == "none":
        return img
    if CORRUPTION_TYPE == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    if CORRUPTION_TYPE == "occlusion":
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        patch_w, patch_h = int(w * OCCLUSION_FRACTION), int(h * OCCLUSION_FRACTION)
        # Fixed, deterministic placement (centre-ish) rather than random --
        # reproducible across re-runs, and a centred patch is a
        # reasonably realistic "something briefly in frame" scenario
        # without needing to justify a random seed choice for a figure.
        x1 = (w - patch_w) // 2
        y1 = (h - patch_h) // 2
        draw.rectangle([x1, y1, x1 + patch_w, y1 + patch_h], fill=(128, 128, 128))
        return img
    raise ValueError(f"Unknown CORRUPTION_TYPE '{CORRUPTION_TYPE}' -- expected none, blur, or occlusion.")


def load_scene_and_head(image_path, head_box, scene_transform, head_transform):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = head_box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h
    head_img = img.crop((x1, y1, x2, y2))
    return (scene_transform(img).unsqueeze(0).to(DEVICE),
            head_transform(head_img).unsqueeze(0).to(DEVICE), img)


def draw_marker(draw, x_px, y_px, marker_key, radius=14, width=3):
    m = MARKERS[marker_key]
    color = m["color"]
    if m["shape"] == "circle":
        draw.ellipse([x_px - radius, y_px - radius, x_px + radius, y_px + radius], outline=color, width=width)
    elif m["shape"] == "square":
        draw.rectangle([x_px - radius, y_px - radius, x_px + radius, y_px + radius], outline=color, width=width)
    elif m["shape"] == "diamond":
        draw.polygon([(x_px, y_px - radius), (x_px + radius, y_px), (x_px, y_px + radius), (x_px - radius, y_px)],
                     outline=color, width=width)
    draw.line([x_px - radius - 6, y_px, x_px + radius + 6, y_px], fill=color, width=2)
    draw.line([x_px, y_px - radius - 6, x_px, y_px + radius + 6], fill=color, width=2)


def draw_head_box(draw, box, color=(255, 0, 255), width=3, label="Subject"):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw.text((x1, max(0, y1 - 14)), label, fill=color)


def make_legend_strip(width, keys, height=40):
    strip = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    x = 15
    for key in keys:
        label = MARKERS[key]["label"]
        draw_marker(draw, x + 6, height // 2, key, radius=6, width=2)
        draw.text((x + 20, height // 2 - 7), label, fill=(0, 0, 0))
        x += 20 + len(label) * 7 + 30
    return strip


def render_one(row, last_predicted_xy, anticipated, category, idx, anticipated_corrupted=None):
    clip_id = row["clip_id"]
    context_paths = row["context_frame_paths"]

    target_path = resolve_image_path(clip_id, row["target_frame_path"])
    target_img = Image.open(target_path).convert("RGB")
    tw, th = target_img.size
    target_draw = ImageDraw.Draw(target_img)

    gt_px = (row["target_gaze_x"], row["target_gaze_y"])
    copy_fwd_px = (last_predicted_xy[0] * tw, last_predicted_xy[1] * th)
    anticipated_px = (anticipated[0] * tw, anticipated[1] * th)

    draw_marker(target_draw, *gt_px, "gt")
    draw_marker(target_draw, *copy_fwd_px, "copy_forward")
    draw_marker(target_draw, *anticipated_px, "anticipated")
    legend_keys = ["gt", "copy_forward", "anticipated"]
    if anticipated_corrupted is not None:
        anticipated_c_px = (anticipated_corrupted[0] * tw, anticipated_corrupted[1] * th)
        draw_marker(target_draw, *anticipated_c_px, "anticipated_corrupted")
        legend_keys.append("anticipated_corrupted")

    last_real_path = resolve_image_path(clip_id, context_paths[-1])
    last_img = Image.open(last_real_path).convert("RGB")
    # Show whatever was ACTUALLY fed to the model -- corrupted, if
    # corruption is active, so the figure honestly reflects the input,
    # not a clean image standing in for a different, corrupted one.
    if CORRUPTION_TYPE != "none" and anticipated_corrupted is not None:
        last_img = apply_corruption(last_img)
    lw, lh = last_img.size
    last_draw = ImageDraw.Draw(last_img)
    last_box = row["context_head_boxes"][-1]
    draw_head_box(last_draw, last_box)
    draw_marker(last_draw, last_predicted_xy[0] * lw, last_predicted_xy[1] * lh, "copy_forward")

    combined_w = last_img.width + target_img.width + 20
    combined_h = max(last_img.height, target_img.height)
    composite = Image.new("RGB", (combined_w, combined_h + 40), (255, 255, 255))
    composite.paste(last_img, (0, 0))
    composite.paste(target_img, (last_img.width + 20, 0))
    composite.paste(make_legend_strip(combined_w, legend_keys), (0, combined_h))

    dist_copy_fwd_px = ((copy_fwd_px[0] - gt_px[0]) ** 2 + (copy_fwd_px[1] - gt_px[1]) ** 2) ** 0.5
    dist_anticipated_px = ((anticipated_px[0] - gt_px[0]) ** 2 + (anticipated_px[1] - gt_px[1]) ** 2) ** 0.5
    improvement_px = dist_copy_fwd_px - dist_anticipated_px

    fname = f"{category}_{idx:02d}_{DATASET}_{STATIC_CONDITION}_{TEMPORAL_ARCHITECTURE}_{row['target_gaze_type']}.png"
    composite.save(os.path.join(OUTPUT_DIR, fname))
    print(f"  [{category}][{idx}] {row['clip_id']}/{row['subject_id']} ({row['target_gaze_type']}): "
          f"copy_fwd_err={dist_copy_fwd_px:.1f}px  anticipated_err={dist_anticipated_px:.1f}px  "
          f"improvement={improvement_px:+.1f}px  -> {fname}")
    return improvement_px


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Dataset: {DATASET}  Condition: {STATIC_CONDITION}  Architecture: {TEMPORAL_ARCHITECTURE}  Horizon: {TARGET_HORIZON_MS}ms")
    print(f"Pool size: {POOL_SIZE}  Selecting: {N_GOOD} good, {N_BAD} bad, {N_RANDOM} random\n")

    static_model = load_gaze_student(CHECKPOINT_PATH, vit_model=STATIC_VIT_MODEL, device=DEVICE)
    extractor = PenultimateFeatureExtractor(static_model)
    temporal_model = load_temporal_model(TEMPORAL_ARCHITECTURE, os.path.join(TEMPORAL_CHECKPOINT_DIR, f"model_{TEMPORAL_ARCHITECTURE}.pt"))
    scene_transform, head_transform = build_transforms()

    pool = sample_pool(POOL_SIZE)
    print(f"Scoring pool of {len(pool)} candidates...")

    # Score every candidate first (inference only, no image rendering
    # yet) -- rendering only happens for the selected good/bad/random
    # subset below, so scoring a large pool stays cheap.
    scored = []
    for row in pool:
        clip_id = row["clip_id"]
        context_paths = row["context_frame_paths"]
        head_boxes = row["context_head_boxes"]

        embed_seq = []
        last_predicted_xy = None
        ok = True
        for i, (path, box) in enumerate(zip(context_paths, head_boxes)):
            real_path = resolve_image_path(clip_id, path)
            try:
                scene, head, _ = load_scene_and_head(real_path, box, scene_transform, head_transform)
            except Exception:
                ok = False
                break
            feat, pred_xy, _ = extractor.extract(scene, head)
            embed_seq.append(list(feat) + list(pred_xy))
            last_predicted_xy = pred_xy
        if not ok:
            continue

        seq_tensor = torch.tensor([embed_seq], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            anticipated = temporal_model(seq_tensor).cpu().numpy()[0]

        # Corrupted-input variant: same 8-frame sequence, but the LAST
        # (most recent) context frame's scene image is corrupted before
        # feature extraction -- only computed when CORRUPTION_TYPE is set,
        # zero cost to the existing clean-only path otherwise.
        anticipated_corrupted = None
        if CORRUPTION_TYPE != "none":
            last_real_path = resolve_image_path(clip_id, context_paths[-1])
            try:
                clean_img = Image.open(last_real_path).convert("RGB")
                corrupted_img = apply_corruption(clean_img)
                last_box = head_boxes[-1]
                x1c, y1c, x2c, y2c = last_box
                head_crop = corrupted_img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))
                scene_c = scene_transform(corrupted_img).unsqueeze(0).to(DEVICE)
                head_c = head_transform(head_crop).unsqueeze(0).to(DEVICE)
                feat_c, pred_xy_c, _ = extractor.extract(scene_c, head_c)
                embed_seq_corrupted = embed_seq[:-1] + [list(feat_c) + list(pred_xy_c)]
                seq_tensor_c = torch.tensor([embed_seq_corrupted], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    anticipated_corrupted = temporal_model(seq_tensor_c).cpu().numpy()[0]
            except Exception as e:
                print(f"  WARNING: corruption inference failed for {clip_id}: {e}")

        gt = (row["target_gaze_x"], row["target_gaze_y"])
        # Approximate distances (mixed px/normalized scale) used ONLY for
        # RELATIVE ranking within this pool -- exact, correctly-scaled
        # pixel distances get recomputed at render time using each
        # image's real dimensions. Fine for picking best/worst since it's
        # a relative comparison, not an absolute claim.
        copy_fwd_proxy = ((last_predicted_xy[0] - gt[0]) ** 2 + (last_predicted_xy[1] - gt[1]) ** 2) ** 0.5
        anticipated_proxy = ((anticipated[0] - gt[0]) ** 2 + (anticipated[1] - gt[1]) ** 2) ** 0.5
        improvement_proxy = copy_fwd_proxy - anticipated_proxy

        scored.append((row, last_predicted_xy, anticipated, anticipated_corrupted, improvement_proxy))

    print(f"Scored {len(scored)}/{len(pool)} successfully.\n")
    scored.sort(key=lambda t: t[4], reverse=True)  # best improvement first -- t[4] is
        # improvement_proxy; shifted from t[3] when anticipated_corrupted was
        # inserted into the tuple. FOUND 2026-08-17: this was still reading
        # t[3] (anticipated_corrupted, always None without corruption active),
        # which crashed sorting 150 equal-None keys with a TypeError -- sent
        # to stderr, invisible in stdout, while the .sub's lack of `set -e`
        # let the job report "Finished" anyway despite the crash.

    good = scored[:N_GOOD]
    bad = scored[-N_BAD:] if N_BAD > 0 else []
    remaining = scored[N_GOOD:len(scored) - N_BAD] if len(scored) > N_GOOD + N_BAD else []
    rng = random.Random(SEED)
    rand_selection = rng.sample(remaining, min(N_RANDOM, len(remaining))) if remaining else []

    print("--- GOOD (largest improvement from anticipation) ---")
    for i, (row, lpx, ant, ant_c, _) in enumerate(good):
        render_one(row, lpx, ant, "good", i, ant_c)
    print("\n--- BAD (worst / most negative) ---")
    for i, (row, lpx, ant, ant_c, _) in enumerate(bad):
        render_one(row, lpx, ant, "bad", i, ant_c)
    print("\n--- RANDOM (typical, from the remaining middle) ---")
    for i, (row, lpx, ant, ant_c, _) in enumerate(rand_selection):
        render_one(row, lpx, ant, "random", i, ant_c)

    print(f"\n{len(good)+len(bad)+len(rand_selection)} figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
