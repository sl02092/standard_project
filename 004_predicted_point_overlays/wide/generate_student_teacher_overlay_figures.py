"""
generate_student_teacher_overlay_figures.py — dissertation figures
showing, on a single real frame: the ground truth gaze point, the
teacher's own prediction, and the distilled student's own prediction.

For STATIC_CONDITION="gt" there is no separate teacher (gt IS the
target) -- only ground truth + student are drawn.

SELECTION: scores a POOL_SIZE candidate pool first, then renders only
N_GOOD (student closest to GT), N_BAD (student furthest from GT), and
N_RANDOM (from the remaining middle). Filenames prefixed accordingly.

MARKERS: circle=ground truth, square=teacher, diamond=student -- shape
AND color both carry meaning (black-and-white printing, colorblind
accessibility).

Subject head box is drawn (confirmed field, unlike the temporal script's
target-frame case).

DATASET_FILES / the join logic is copied verbatim from eval_static_v1.py.
"""

import os
import json
import math
import random

import torch
from torchvision import transforms
from PIL import Image, ImageDraw

from gaze_student_model import load_gaze_student, IMG_SIZE, HEAD_CROP_SIZE, NORM_MEAN, NORM_STD

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

MARKERS = {
    "gt": {"color": (46, 204, 113), "shape": "circle", "label": "Ground truth"},
    "teacher": {"color": (241, 196, 15), "shape": "square", "label": f"Teacher ({STATIC_CONDITION})"},
    "student": {"color": (52, 152, 219), "shape": "diamond", "label": "Student prediction"},
}

CONDITION_FIELDS = {
    "teacher_hybrid": ("hybrid_pred_x", "hybrid_pred_y"),
    "teacher_internvl3_alone": ("internvl3_alone_pred_x", "internvl3_alone_pred_y"),
    "mtgs": ("mtgs_pred_x", "mtgs_pred_y"),
}

# ══════════════════════════════════════════════════════════════════════
# ── DATASET FILES + JOIN — copied verbatim from eval_static_v1.py ──────
# ══════════════════════════════════════════════════════════════════════

DATASET_FILES = {
    "vat": {"full": "labels_vat_full_{split}.jsonl", "hybrid": "labels_vat_hybrid_targeted_{split}.jsonl",
            "mtgs": "mtgs_teacher_vat_merged_{split}.jsonl", "split_suffix": "TEST"},
    "childplay": {"full": "labels_childplay_full_{split}.jsonl", "hybrid": "labels_childplay_hybrid_targeted_{split}.jsonl",
                  "mtgs": "mtgs_teacher_childplay_{split}.jsonl", "split_suffix": "TEST"},
    "vacation": {"full": "labels_vacation_full_{split}.jsonl", "hybrid": "labels_vacation_hybrid_targeted_{split}.jsonl",
                 "mtgs": "mtgs_teacher_vacation_{split}.jsonl", "split_suffix": "TEST"},
    "ucolaeo": {"full": "labels_ucolaeo_full.jsonl", "hybrid": "labels_ucolaeo_hybrid_targeted.jsonl",
                "mtgs": "mtgs_teacher_ucolaeo.jsonl", "split_suffix": None},
}
if DATASET not in DATASET_FILES:
    raise ValueError(f"Unknown FIGURE_DATASET '{DATASET}' -- known: {list(DATASET_FILES)}")

LIBRARY_ROOT = os.environ.get(f"{DATASET.upper()}_LIBRARY_ROOT")
if not LIBRARY_ROOT:
    raise ValueError(f"{DATASET.upper()}_LIBRARY_ROOT not set.")


def load_labels_file(path):
    records = {}
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
                continue
            key = (str(rec["show"]), str(rec["clip"]), str(rec["fname"]), str(rec["subject"]))
            records[key] = rec
    return records


def build_population():
    files = DATASET_FILES[DATASET]
    split = files["split_suffix"]

    def path_for(kind):
        template = files[kind]
        return os.path.join(LIBRARY_ROOT, template.format(split=split) if split else template)

    full_recs = load_labels_file(path_for("full"))
    hybrid_recs = load_labels_file(path_for("hybrid"))
    mtgs_recs = load_labels_file(path_for("mtgs"))
    common = set(full_recs) & set(hybrid_recs) & set(mtgs_recs)

    GT_MISMATCH_PIXEL_TOLERANCE = 2.0
    joined = []
    for key in common:
        fu, h, m = full_recs[key], hybrid_recs[key], mtgs_recs[key]
        d_fh = math.hypot(fu["gt_px_x"] - h["gt_px_x"], fu["gt_px_y"] - h["gt_px_y"])
        d_fm = math.hypot(fu["gt_px_x"] - m["gt_px_x"], fu["gt_px_y"] - m["gt_px_y"])
        if d_fh > GT_MISMATCH_PIXEL_TOLERANCE or d_fm > GT_MISMATCH_PIXEL_TOLERANCE:
            continue
        if not os.path.isfile(fu["img_path"]):
            continue
        if fu["gaze_type"] == "offscreen":
            continue
        joined.append({
            "gaze_type": fu["gaze_type"], "img_path": fu["img_path"],
            "head_x1": fu["head_x1"], "head_y1": fu["head_y1"], "head_x2": fu["head_x2"], "head_y2": fu["head_y2"],
            "gt_x": fu["gt_x"], "gt_y": fu["gt_y"],
            "hybrid_pred_x": h["pred_x"], "hybrid_pred_y": h["pred_y"],
            "internvl3_alone_pred_x": fu["pred_x"], "internvl3_alone_pred_y": fu["pred_y"],
            "mtgs_pred_x": m["pred_x"], "mtgs_pred_y": m["pred_y"],
        })
    print(f"[{DATASET}] {len(joined)} usable records (on-screen, GT-agreed, image present).")
    return joined


# ══════════════════════════════════════════════════════════════════════
# ── SAMPLING + DRAWING ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def sample_pool(records, pool_size):
    by_type = {}
    for r in records:
        by_type.setdefault(r["gaze_type"], []).append(r)
    rng = random.Random(SEED)
    types = list(by_type.keys())
    per_type = max(1, pool_size // len(types))
    pool = []
    for t in types:
        candidates = by_type[t]
        pool.extend(rng.sample(candidates, min(per_type, len(candidates))))
    rng.shuffle(pool)
    return pool[:pool_size]


def build_transforms():
    to_tensor_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    scene_t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), to_tensor_norm])
    head_t = transforms.Compose([transforms.Resize((HEAD_CROP_SIZE, HEAD_CROP_SIZE)), to_tensor_norm])
    return scene_t, head_t


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


def render_one(rec, student_px, category, idx, has_teacher, teacher_px=None):
    img = Image.open(rec["img_path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw_head_box(draw, (rec["head_x1"], rec["head_y1"], rec["head_x2"], rec["head_y2"]))

    gt_px = (rec["gt_x"] * img.width, rec["gt_y"] * img.height)
    draw_marker(draw, *gt_px, "gt")
    draw_marker(draw, *student_px, "student")
    keys = ["gt", "student"]
    if has_teacher:
        draw_marker(draw, *teacher_px, "teacher")
        keys = ["gt", "teacher", "student"]

    composite = Image.new("RGB", (img.width, img.height + 40), (255, 255, 255))
    composite.paste(img, (0, 0))
    composite.paste(make_legend_strip(img.width, keys), (0, img.height))

    student_dist = math.hypot(student_px[0] - gt_px[0], student_px[1] - gt_px[1])
    teacher_dist = math.hypot(teacher_px[0] - gt_px[0], teacher_px[1] - gt_px[1]) if has_teacher else None

    fname = f"{category}_{idx:02d}_{DATASET}_{STATIC_CONDITION}_{rec['gaze_type']}.png"
    composite.save(os.path.join(OUTPUT_DIR, fname))
    teacher_str = f"  teacher_err={teacher_dist:.1f}px" if teacher_dist is not None else ""
    print(f"  [{category}][{idx}] {rec['gaze_type']}: student_err={student_dist:.1f}px{teacher_str}  -> {fname}")


def main():
    has_teacher = STATIC_CONDITION in CONDITION_FIELDS
    print(f"Dataset: {DATASET}  Condition: {STATIC_CONDITION}  "
          f"({'student vs teacher vs GT' if has_teacher else 'student vs GT only'})")
    print(f"Pool size: {POOL_SIZE}  Selecting: {N_GOOD} good, {N_BAD} bad, {N_RANDOM} random\n")

    model = load_gaze_student(CHECKPOINT_PATH, vit_model=STATIC_VIT_MODEL, device=DEVICE)
    scene_transform, head_transform = build_transforms()

    records = build_population()
    if len(records) < 10:
        print("ERROR: too few usable records -- check paths before trusting this.")
        return
    pool = sample_pool(records, POOL_SIZE)
    print(f"Scoring pool of {len(pool)} candidates...")

    scored = []
    for rec in pool:
        img = Image.open(rec["img_path"]).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = rec["head_x1"], rec["head_y1"], rec["head_x2"], rec["head_y2"]
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(w, int(x2)), min(h, int(y2))
        if x2c <= x1c or y2c <= y1c:
            x1c, y1c, x2c, y2c = 0, 0, w, h
        head_crop = img.crop((x1c, y1c, x2c, y2c))

        scene_t = scene_transform(img).unsqueeze(0).to(DEVICE)
        head_t = head_transform(head_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            coord_pred, _ = model(scene_t, head_t)
        student_x, student_y = coord_pred.cpu().numpy()[0]

        gt_px = (rec["gt_x"] * w, rec["gt_y"] * h)
        student_px = (student_x * w, student_y * h)
        student_dist = math.hypot(student_px[0] - gt_px[0], student_px[1] - gt_px[1])

        teacher_px = None
        if has_teacher:
            fx, fy = CONDITION_FIELDS[STATIC_CONDITION]
            teacher_px = (rec[fx] * w, rec[fy] * h)

        # Rank by student's own accuracy (distance to GT) -- "good" =
        # student very close to ground truth, "bad" = student far off.
        scored.append((rec, student_px, teacher_px, student_dist))

    print(f"Scored {len(scored)} successfully.\n")
    scored.sort(key=lambda t: t[3])  # smallest error (best) first

    good = scored[:N_GOOD]
    bad = scored[-N_BAD:] if N_BAD > 0 else []
    remaining = scored[N_GOOD:len(scored) - N_BAD] if len(scored) > N_GOOD + N_BAD else []
    rng = random.Random(SEED)
    rand_selection = rng.sample(remaining, min(N_RANDOM, len(remaining))) if remaining else []

    print("--- GOOD (student closest to ground truth) ---")
    for i, (rec, spx, tpx, _) in enumerate(good):
        render_one(rec, spx, "good", i, has_teacher, tpx)
    print("\n--- BAD (student furthest from ground truth) ---")
    for i, (rec, spx, tpx, _) in enumerate(bad):
        render_one(rec, spx, "bad", i, has_teacher, tpx)
    print("\n--- RANDOM (typical, from the remaining middle) ---")
    for i, (rec, spx, tpx, _) in enumerate(rand_selection):
        render_one(rec, spx, "random", i, has_teacher, tpx)

    print(f"\n{len(good)+len(bad)+len(rand_selection)} figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
