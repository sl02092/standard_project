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

DESCRIBE_PROMPT = (
    "Look at the person indicated. Describe in one short sentence what they "
    "are looking at. If they are looking at a specific object, name it clearly "
    "(e.g. 'looking at the mannequin'). If off-screen, say 'looking off-screen'."
)

# ── PATH RESOLUTION (mirrors the pattern already used elsewhere in this project) ──
def resolve_frame_path(seq_dir, frame_idx, base_path):
    for candidate in [
        os.path.join(base_path, "images", seq_dir, f"{frame_idx:06d}.jpg"),
        os.path.join(base_path, "images", seq_dir, f"{frame_idx}.jpg"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_gt(seq_dir, frame_idx, subject_id, base_path):
    """
    Placeholder GT lookup -- mirrors the existing project's annotation-file
    convention. If you already have a working resolve_context()/GT lookup
    from step43d6/d7, swap it in here directly; this stub returns None so
    the script still runs end-to-end and reports predictions even without
    GT wired up, rather than crashing on a missing lookup.
    """
    return None


def extract_object_phrase(description_text):
    """
    Pulls the object noun phrase out of InternVL3's description, e.g.
    "The person is looking towards the mannequin." -> "the mannequin"
    Deliberately simple regex first (cheap, no extra model call) --
    production raw_responses consistently follow "looking at/towards the X"
    phrasing per the existing handover notes. Falls back to the full
    description if no pattern matches, so Grounding DINO still gets SOME
    query rather than nothing.
    """
    patterns = [
        r"looking (?:at|towards|toward) (?:the |a |an )?([^.,;]+)",
        r"gaze(?:s|ing)? (?:at|towards|toward) (?:the |a |an )?([^.,;]+)",
    ]
    for pat in patterns:
        match = re.search(pat, description_text, re.IGNORECASE)
        if match:
            phrase = match.group(1).strip()
            if "off-screen" in phrase.lower() or "off screen" in phrase.lower():
                return None  # signal off-screen, don't run detection at all
            return phrase
    return description_text.strip()


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
    print(f"{'Frame / Subject':<28} | {'Object Phrase':<25} | {'Pred (Norm)':<18} | {'Box Confidence'}")
    print("-" * 100)

    for seq_dir, frame_idx, subject_id in TARGET_FRAMES:
        frame_path = resolve_frame_path(seq_dir, frame_idx, BASE_PATH)
        frame_label = f"{seq_dir}/{frame_idx} ({subject_id})"

        if frame_path is None:
            print(f"{frame_label:<28} | SKIP -- frame not found on disk")
            continue

        img = Image.open(frame_path).convert("RGB")
        w, h = img.size

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
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]

        if len(results["boxes"]) == 0:
            print(f"{frame_label:<28} | {object_phrase[:25]:<25} | NO BOX FOUND       | (below threshold)")
            continue

        # Take the highest-confidence box.
        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].tolist()  # [x1, y1, x2, y2] in pixel coords
        score = results["scores"][best_idx].item()

        cx = ((box[0] + box[2]) / 2) / w
        cy = ((box[1] + box[3]) / 2) / h

        gt = resolve_gt(seq_dir, frame_idx, subject_id, BASE_PATH)
        if gt is not None:
            dist = math.sqrt((gt[0] - cx) ** 2 + (gt[1] - cy) ** 2)
            dist_str = f" | dist={dist:.3f}"
        else:
            dist_str = " | (GT lookup not wired up -- see resolve_gt())"

        print(f"{frame_label:<28} | {object_phrase[:25]:<25} | ({cx:.3f}, {cy:.3f})    | {score:.3f}{dist_str}")

    print("\n[NOTE] resolve_gt() is a stub returning None -- wire it up to your existing")
    print("GT-lookup logic (the same one step43d6/d7 already use) to get real ADE")
    print("numbers here. Without it, this run only shows predictions, not accuracy.")


if __name__ == "__main__":
    main()
