"""
Step 2 - Teacher VLM Probe (v3 — Multi-Subject Social Gaze)
Runs InternVL3-8B-hf against 5 frames, now providing BOTH subjects'
head bounding boxes and asking the model to reason about who is
looking at whom — a much more grounded social gaze task.

Usage:
    python step2_teacher_probe.py

Requirements:
    pip install -U transformers accelerate pillow torch torchvision
"""

import os
import csv
import re
import math
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_PATH      = r"C:\repo\standard_project_feasability\videoattentiontarget"
SHOW           = "Band of Brothers"
CLIP           = "5910_5970"
PRIMARY_SUBJ   = "s00.txt"   # the subject whose gaze we're predicting
SECONDARY_SUBJ = "s01.txt"   # the other person in the scene
NUM_FRAMES     = 5
MODEL_NAME     = "OpenGVLab/InternVL3-8B-hf"
OUTPUT_FILE    = "step2_probe_results.jpg"
PASS_THRESHOLD = 0.15        # normalised Euclidean distance

# ── Annotation loader ─────────────────────────────────────────────────────────

def load_annotations(ann_path):
    annotations = {}
    if not os.path.exists(ann_path):
        return annotations
    with open(ann_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            fname = row[0].strip()
            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            gx, gy = int(row[5]), int(row[6])
            gaze = (gx, gy) if gx != -1 and gy != -1 else None
            annotations[fname] = (x1, y1, x2, y2, gaze)
    return annotations

def pick_frames(all_frames, n):
    if len(all_frames) <= n:
        return all_frames
    step = (len(all_frames) - 1) / (n - 1)
    return [all_frames[round(i * step)] for i in range(n)]

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_social_prompt(img_width, img_height, primary_box, secondary_box):
    """
    Multi-subject social gaze prompt.
    Tells the model exactly how many people are in the scene,
    gives both head bounding boxes, and asks it to reason about
    who Person A is looking at before estimating the coordinate.
    """
    # Primary subject (whose gaze we predict)
    ax1, ay1, ax2, ay2 = primary_box
    acx = (ax1 + ax2) / 2 / img_width
    acy = (ay1 + ay2) / 2 / img_height
    ah_pos = "left" if acx < 0.4 else ("right" if acx > 0.6 else "centre")
    av_pos = "upper" if acy < 0.4 else ("lower" if acy > 0.6 else "middle")

    # Normalised boxes
    a_norm = (round(ax1/img_width,3), round(ay1/img_height,3),
              round(ax2/img_width,3), round(ay2/img_height,3))

    # Secondary subject
    bx1, by1, bx2, by2 = secondary_box
    b_norm = (round(bx1/img_width,3), round(by1/img_height,3),
              round(bx2/img_width,3), round(by2/img_height,3))
    bcx = (bx1 + bx2) / 2 / img_width
    bcy = (by1 + by2) / 2 / img_height
    bh_pos = "left" if bcx < 0.4 else ("right" if bcx > 0.6 else "centre")
    bv_pos = "upper" if bcy < 0.4 else ("lower" if bcy > 0.6 else "middle")

    # Face centre of Person B (likely gaze target if social gaze)
    b_face_cx = round((bx1 + bx2) / 2 / img_width, 3)
    b_face_cy = round((by1 + by2) / 2 / img_height, 3)

    prompt = f"""You are a gaze estimation expert analysing a social interaction scene.

IMAGE SIZE: {img_width} x {img_height} pixels.

SCENE DESCRIPTION: There are exactly TWO people visible in this image.
This is a social interaction — they are likely engaged in conversation.

PERSON A (whose gaze you must predict):
  Position: {av_pos}-{ah_pos} of image
  Head box (pixels): top-left=({ax1},{ay1}), bottom-right=({ax2},{ay2})
  Head box (normalised): top-left=({a_norm[0]},{a_norm[1]}), bottom-right=({a_norm[2]},{a_norm[3]})

PERSON B (the other person in the scene):
  Position: {bv_pos}-{bh_pos} of image
  Head box (pixels): top-left=({bx1},{by1}), bottom-right=({bx2},{by2})
  Head box (normalised): top-left=({b_norm[0]},{b_norm[1]}), bottom-right=({b_norm[2]},{b_norm[3]})
  Person B's face centre (normalised): approximately ({b_face_cx}, {b_face_cy})

INSTRUCTIONS — answer each step on its own line:

Step 1 — Head pose of Person A: Describe the direction Person A's head \
and eyes are pointing (e.g. "facing right toward Person B", \
"looking down", "facing camera").

Step 2 — Social gaze decision: Is Person A looking at Person B's face, \
at the camera, at an object, or off-screen? State your choice clearly.

Step 3 — Gaze target coordinate: Based on your analysis, estimate the \
(x, y) normalised coordinate (0.0–1.0) of what Person A is looking at. \
If looking at Person B, use Person B's face centre as your estimate.

You MUST end your response with this exact line:
GAZE_XY: (x, y)

where x and y are decimal values between 0.0 and 1.0.

Step 4 — Confidence: HIGH / MEDIUM / LOW"""

    return prompt

# ── Parse model output ────────────────────────────────────────────────────────

def parse_gaze_xy(response_text):
    match = re.search(r"GAZE_XY:\s*\(([0-9.]+),\s*([0-9.]+)\)", response_text)
    if match:
        x = max(0.0, min(1.0, float(match.group(1))))
        y = max(0.0, min(1.0, float(match.group(2))))
        return (x, y)
    return None

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_comparison(img_path, primary_ann, secondary_ann, pred_xy, frame_name):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Person A head box — green
    ax1, ay1, ax2, ay2, gaze = primary_ann
    draw.rectangle([ax1, ay1, ax2, ay2], outline=(0, 220, 80), width=3)
    draw.text((ax1, max(0, ay1-18)), "A", fill=(0, 220, 80))

    # Person B head box — orange
    if secondary_ann:
        bx1, by1, bx2, by2, _ = secondary_ann
        draw.rectangle([bx1, by1, bx2, by2], outline=(255, 165, 0), width=3)
        draw.text((bx1, max(0, by1-18)), "B", fill=(255, 165, 0))

    # GT gaze — red crosshair
    if gaze:
        gx, gy = gaze
        r = 12
        draw.ellipse([gx-r, gy-r, gx+r, gy+r], outline=(255, 60, 60), width=3)
        draw.line([gx-r*2, gy, gx+r*2, gy], fill=(255, 60, 60), width=2)
        draw.line([gx, gy-r*2, gx, gy+r*2], fill=(255, 60, 60), width=2)
        draw.text((gx+r+2, gy-8), "GT", fill=(255, 60, 60))

    # Predicted gaze — blue crosshair
    if pred_xy:
        px, py = int(pred_xy[0]*w), int(pred_xy[1]*h)
        r = 12
        draw.ellipse([px-r, py-r, px+r, py+r], outline=(80, 160, 255), width=3)
        draw.line([px-r*2, py, px+r*2, py], fill=(80, 160, 255), width=2)
        draw.line([px, py-r*2, px, py+r*2], fill=(80, 160, 255), width=2)
        draw.text((px+r+2, py-8), "PRED", fill=(80, 160, 255))
        if gaze:
            draw.line([gaze[0], gaze[1], px, py], fill=(255, 255, 0), width=2)

    # Frame label
    draw.rectangle([0, h-22, w, h], fill=(0, 0, 0))
    draw.text((4, h-18), frame_name, fill=(255, 255, 255))

    return img

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    images_dir  = os.path.join(BASE_PATH, "images", SHOW, CLIP)
    ann_dir     = os.path.join(BASE_PATH, "annotations", "train", SHOW, CLIP)
    primary_ann = load_annotations(os.path.join(ann_dir, PRIMARY_SUBJ))
    second_ann  = load_annotations(os.path.join(ann_dir, SECONDARY_SUBJ))

    if not second_ann:
        print(f"Warning: {SECONDARY_SUBJ} not found — falling back to single-subject prompt")

    # Pick frames where BOTH subjects are annotated and primary gaze is on-screen
    all_frames = sorted(os.listdir(images_dir))
    candidates = [
        f for f in all_frames
        if f in primary_ann
        and primary_ann[f][4] is not None      # on-screen gaze
        and (not second_ann or f in second_ann) # secondary subject present
    ]
    candidates = ["00005924.jpg", "00005925.jpg", "00005926.jpg"]
    chosen = pick_frames(candidates, NUM_FRAMES)
    print(f"Probing {len(chosen)} frames with both subjects annotated")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n── Loading model: {MODEL_NAME} ──────────────")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"  Model loaded ✓  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Run inference ───────────────────────────────────────────────────────
    results = []
    annotated_imgs = []

    for frame_name in chosen:
        img_path = os.path.join(images_dir, frame_name)
        img_raw  = Image.open(img_path).convert("RGB")
        img_w, img_h = img_raw.size

        ax1, ay1, ax2, ay2, gaze = primary_ann[frame_name]
        primary_box = (ax1, ay1, ax2, ay2)

        sec = second_ann.get(frame_name) if second_ann else None
        if sec:
            secondary_box = (sec[0], sec[1], sec[2], sec[3])
            prompt_text = build_social_prompt(img_w, img_h, primary_box, secondary_box)
        else:
            # Fallback: no secondary subject data
            secondary_box = None
            prompt_text = build_social_prompt(img_w, img_h, primary_box, primary_box)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_raw},
                    {"type": "text",  "text": prompt_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        response  = processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True
        )[0]

        pred_xy = parse_gaze_xy(response)
        #gt_norm = (gaze[0]/img_w, gaze[1]/img_h)
        gt_norm = (gaze[0]/img_w, gaze[1]/img_h) if gaze is not None else None
        #dist    = euclidean_distance(pred_xy, gt_norm) if pred_xy else None
        dist = euclidean_distance(pred_xy, gt_norm) if (pred_xy is not None and gt_norm is not None) else None
        passed  = dist is not None and dist <= PASS_THRESHOLD

        results.append({
            "frame": frame_name,
            "gt_norm": gt_norm,
            "pred_norm": pred_xy,
            "dist": dist,
            "passed": passed,
            "response": response,
        })

        print(f"\n── {frame_name} ──────────────────────────────")
        print(f"  GT   (norm) : {gt_norm}")
        print(f"  PRED (norm) : {pred_xy}")
        print(f"  Distance    : {f'{dist:.3f}' if dist is not None else 'parse failed'}")
        print(f"  Pass        : {'✓' if passed else '✗'} (threshold={PASS_THRESHOLD})")
        print(f"  Response    :\n{response}")

        annotated_imgs.append(
            draw_comparison(img_path, primary_ann[frame_name], sec, pred_xy, frame_name)
        )

    # ── Summary ────────────────────────────────────────────────────────────
    n_parsed  = sum(1 for r in results if r["pred_norm"] is not None)
    n_passed  = sum(1 for r in results if r["passed"])
    dists     = [r["dist"] for r in results if r["dist"] is not None]
    mean_dist = sum(dists)/len(dists) if dists else None

    print(f"\n{'='*52}")
    print(f"SUMMARY  {SHOW} / {CLIP} / {PRIMARY_SUBJ}")
    print(f"{'='*52}")
    print(f"  Frames probed      : {len(results)}")
    print(f"  Coordinates parsed : {n_parsed} / {len(results)}")
    print(f"  Passed (≤{PASS_THRESHOLD})      : {n_passed} / {len(results)}")
    if mean_dist:
        print(f"  Mean distance      : {mean_dist:.3f}")
    print(f"\n  Verdict:")
    if n_passed >= 3:
        print("  ✓ PASS — Teacher signal is strong enough for distillation")
    elif n_passed >= 1:
        print("  ~ PARTIAL — Useful signal but prompt needs further refinement")
    else:
        print("  ✗ FAIL — Teacher cannot localise social gaze reliably")

    # ── Save visualisation ──────────────────────────────────────────────────
    if annotated_imgs:
        widths, heights = zip(*(i.size for i in annotated_imgs))
        combined = Image.new("RGB", (sum(widths), max(heights)), (20, 20, 20))
        x_off = 0
        for img in annotated_imgs:
            combined.paste(img, (x_off, 0))
            x_off += img.width
        combined.save(OUTPUT_FILE)
        print(f"\n  Saved : {OUTPUT_FILE}")
        print("  Green = Person A (primary), Orange = Person B")
        print("  Red crosshair = GT, Blue crosshair = Prediction")
        print("  Yellow line = error vector")

    print("\nDone.")

if __name__ == "__main__":
    main()
