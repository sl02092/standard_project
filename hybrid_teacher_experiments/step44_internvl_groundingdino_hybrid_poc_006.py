"""
step44_internvl_groundingdino_hybrid_poc.py — Proof-of-concept, NO TRAINING.

Tests the hypothesis from the literature (Tafasca et al., NeurIPS 2024 --
semantic identification and spatial localization are different skills, and
specialist localizers + VLM semantics are complementary) directly against
our own object-gaze coordinate problem:

  InternVL3-8B-hf: produces the verbal gaze-target description (already
  reliable per production data -- e.g. "looking towards the mannequin").
  -> extract_object_phrase(): pulls the object noun phrase out of that text.
  Grounding DINO: takes that phrase as a TEXT-CONDITIONED QUERY and returns
  a bounding box for it directly -- no separate matching/re-ranking step
  needed (unlike a generic detector + CLIP re-rank pipeline).
  -> box centroid = predicted (x, y), compared against GT via the same
     ADE/distance metric used throughout this investigation.

Both models are used frozen, at inference only. No training, no gradients,
no checkpointing needed -- this is meant to run in minutes, not hours.

FIRST RUN NOTE: grounding-dino-base is a new dependency, not used anywhere
else in this project so far -- expect a one-time HF weight download
(~700MB, much smaller than InternVL3-8B) the first time this runs.
"""

import os
import re
import math
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText, AutoProcessor as InternVLProcessor,
    AutoProcessor as GDinoProcessor, AutoModelForZeroShotObjectDetection,
)

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH        = os.environ.get("VAT_BASE_PATH", r"C:\repo\standard_project\videoattentiontarget")
INTERNVL_NAME    = "OpenGVLab/InternVL3-8B-hf"
GDINO_NAME       = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD    = 0.30   # Grounding DINO confidence threshold for keeping a box
TEXT_THRESHOLD   = 0.25

# Reuse the SAME multi-person object-gaze frames already picked from the
# manifest earlier in this investigation -- so results are directly
# comparable against the existing object-gaze ADE figures (0.436 production
# baseline; 0.554/0.452 from the DINOv2+InternVL hybrid attempts).
TARGET_FRAMES = [
    ("17742_17893", 17826, "s00"),
    ("10239_10740", 10270, "s01"),
    ("14250_14430", 14250, "s01"),
    ("19636_19829", 19658, "s00"),
    ("1710_1890",   1737,  "s02"),
    ("1348_1469",   1382,  "s00"),
]

DESCRIBE_PROMPT_001 = (
    "Look at the person indicated. What single object, thing or person are they looking at? "
    "Answer with ONLY the object's name (2-4 words maximum), no extra description, "
    "no mention of the person, no full sentence. For example: 'newspaper' or "
    "'red coffee mug' or 'laptop screen'. If they are looking at another person's "
    "face or body (not holding/using an object), answer 'face'. "
    "If looking off-screen, answer 'off-screen'."
)

# <<< this one is the best so far!
DESCRIBE_PROMPT = (
    "Analysis Steps:"
    "1. Examine the subject's face orientation and precise eye/pupil direction."
    "2. Check if the vector of their eye gaze points at a person, object or entirely outside the visible image frame boundaries."
    "3. Identify what object or person they are looking at."
    "4. Write 1-2 sentences of step-by-step reasoning."
    " "
    "If looking at an object: Answer with ONLY the object name plus a brief location hint if multiple similar objects exist "
    "(e.g. 'newspaper', 'red coffee mug', 'plate on the left'). 2-5 words maximum."
    "If looking at a person: Answer with a brief visual descriptor of that person only "
    "(e.g. 'person in gray suit', 'woman on the right', 'man with beard'). 2-5 words maximum."
    "If looking off-screen: answer 'off-screen'."
)

# ── LOOKUP ENGINE (copied verbatim from the proven step43d7_prompt_compare.py,
# NOT reinvented -- this is the real annotation-folder convention: each clip's
# directory contains one .txt per subject, GT coords + the real image filename
# both live inside that .txt, keyed by frame number) ─────────────────────────
def find_dir_by_clip_id(root_path, clip_id):
    """Exact match on directory name, not substring."""
    target = str(clip_id).strip()
    for root, dirs, _ in os.walk(root_path):
        for d in dirs:
            if d.strip() == target:
                return os.path.join(root, d)
    return None


def parse_txt_file_for_frame(filepath, frame_num):
    """Exact frame-number match only."""
    target_idx = int(frame_num)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        for line in f:
            tokens = line.strip().split(',')
            if not tokens or len(tokens) < 7:
                continue
            try:
                file_idx_str = re.sub(r"\D", "", os.path.basename(tokens[0]))
                line_idx = int(file_idx_str)
                if line_idx == target_idx:
                    return {
                        "fname_actual": os.path.basename(tokens[0]),
                        "head_x1": float(tokens[1]), "head_y1": float(tokens[2]),
                        "head_x2": float(tokens[3]), "head_y2": float(tokens[4]),
                        "gaze_x": float(tokens[5]),  "gaze_y": float(tokens[6])
                    }
            except (ValueError, IndexError):
                continue
    return None


def resolve_context(clip, frame_num, subject, base_path):
    ann_dir = find_dir_by_clip_id(os.path.join(base_path, "annotations"), clip)
    img_dir = find_dir_by_clip_id(os.path.join(base_path, "images"), clip)
    if not ann_dir or not img_dir:
        return None
    subj_file = f"{subject}.txt"
    primary_data = parse_txt_file_for_frame(os.path.join(ann_dir, subj_file), frame_num)
    if not primary_data:
        return None
    img_path = os.path.join(img_dir, primary_data["fname_actual"])
    if not os.path.exists(img_path):
        return None
    
    return {"img_path": img_path, "primary": primary_data}


def extract_object_phrase(description_text):
    """
    Light cleanup of InternVL3's direct-name response.
    Key change from earlier versions: person descriptors are now PRESERVED
    rather than collapsed to bare 'person' -- so Grounding DINO receives
    'person in gray suit' or 'woman on the right' rather than 'person',
    which is spatially ambiguous in multi-person frames.
    The location-hint instruction also covers identical-object disambiguation
    ('plate on the left' rather than bare 'plate').
    """
    cleaned = description_text.strip().strip('."\'').strip()

    if "off-screen" in cleaned.lower() or "off screen" in cleaned.lower():
        return None

    # Safety net: model ignored the brevity instruction and produced a full
    # sentence -- extract the core noun phrase rather than sending the whole
    # sentence to Grounding DINO.
    if len(cleaned.split()) > 7:
        # Try person-looking-at pattern first
        match = re.search(
            r"looking (?:at|towards?|toward) (?:the |a |an )?([^.,;]+)",
            cleaned, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        # Fall back: just use the first 5 words, which is usually the core noun
        return " ".join(cleaned.split()[:5])

    return cleaned


def main():
    print("[*] Loading InternVL3-8B-hf (semantic description)...")
    internvl_processor = InternVLProcessor.from_pretrained(INTERNVL_NAME, trust_remote_code=True)
    internvl_model = AutoModelForImageTextToText.from_pretrained(
        INTERNVL_NAME, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    internvl_model.eval()

    print("[*] Loading Grounding DINO (text-conditioned localization)...")
    gdino_processor = GDinoProcessor.from_pretrained(GDINO_NAME)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_NAME).to("cuda:0")
    gdino_model.eval()

    print("\n[+] Both models loaded. Running hybrid pipeline...\n")
    print(f"{'Frame / Subject':<28} | {'Object Phrase':<40} | {'Pred (Norm)':<18} | {'Box Confidence'}")
    print("-" * 115)

    for seq_dir, frame_idx, subject_id in TARGET_FRAMES:
        context = resolve_context(seq_dir, frame_idx, subject_id, BASE_PATH)
        frame_label = f"{seq_dir}/{frame_idx} ({subject_id})"

        if context is None:
            print(f"{frame_label:<28} | SKIP -- lookup failure (clip/annotation/frame not found)")
            continue

        frame_path = context["img_path"]
        primary = context["primary"]

        img = Image.open(frame_path).convert("RGB")
        w, h = img.size

        gt_x, gt_y = primary["gaze_x"], primary["gaze_y"]
        gt_is_offscreen = (gt_x == -1 and gt_y == -1)
        gt_norm = None if gt_is_offscreen else (gt_x / w, gt_y / h)

        # --- Step 1: InternVL3 describes the gaze target ---
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DESCRIBE_PROMPT}]}]
        inputs = internvl_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(internvl_model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            gen_ids = internvl_model.generate(**inputs, max_new_tokens=40, do_sample=False)
        description = internvl_processor.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        object_phrase = extract_object_phrase(description)

        if object_phrase is None:
            print(f"{frame_label:<28} | (off-screen per VLM)     | {'(-1, -1)':<18} | n/a")
            continue

        # --- Step 2: Grounding DINO localizes that exact phrase ---
        # Grounding DINO expects lowercase, period-separated phrases per its
        # training convention.
        gdino_query = object_phrase.lower().strip().rstrip(".") + "."
        gdino_inputs = gdino_processor(images=img, text=gdino_query, return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
            gdino_out = gdino_model(**gdino_inputs)

        results = gdino_processor.post_process_grounded_object_detection(
            gdino_out, gdino_inputs.input_ids,
            threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]

        if len(results["boxes"]) == 0:
            print(f"{frame_label:<28} | {object_phrase[:40]:<40} | NO BOX FOUND       | (below threshold)")
            continue

        # Take the highest-confidence box.
        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].tolist()  # [x1, y1, x2, y2] in pixel coords
        score = results["scores"][best_idx].item()

        cx = ((box[0] + box[2]) / 2) / w
        cy = ((box[1] + box[3]) / 2) / h

        if gt_norm is not None:
            dist = math.sqrt((gt_norm[0] - cx) ** 2 + (gt_norm[1] - cy) ** 2)
            dist_str = f" | dist={dist:.3f}"
        else:
            dist_str = " | (GT is off-screen -- not comparable to an object-coord prediction)"

        print(f"{frame_label:<28} | {object_phrase[:40]:<40} | ({cx:.3f}, {cy:.3f})    | {score:.3f}{dist_str}")

if __name__ == "__main__":
    main()
