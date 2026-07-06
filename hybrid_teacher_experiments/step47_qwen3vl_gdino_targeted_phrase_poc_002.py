"""
step46_qwen3vl_gdino_targeted_phrase_poc_002.py — Proof-of-concept, NO TRAINING.

DIRECT COMPARISON VARIANT of step46_internvl_gdino_targeted_phrase_poc_002.py.

Only change from that file: the semantic-description model is swapped from
InternVL3-8B-hf to Qwen3-VL-8B-Instruct. Everything else -- the prompt
wording, the GDINO stage, the phrase-cleanup logic, the lookup engine, the
scoring/printing -- is kept identical on purpose, so any difference in the
printed ADE numbers is attributable to the model swap and nothing else.

Rationale for testing Qwen3-VL-8B here: community reports (OpenGVLab/InternVL
GitHub issue #1103, and independent REC benchmarking issue on the
InternVL3-14B HF discussion page) describe InternVL3's visual-grounding /
spatial-referring behavior as inconsistent, and Qwen2.5-VL was independently
reported as ~2x better than InternVL2.5-8B on a referring-detection
benchmark. Qwen3-VL's own technical report describes explicit box- and
point-grounding training on COCO/Objects365/OpenImages/RefCOCO(+/g), which is
a closer match to what we need here than InternVL3's training mix. This
script tests whether that translates into better GDINO-query phrases (via
better implicit spatial/disambiguating description), not into asking Qwen3-VL
to emit its own bounding boxes -- GDINO is still doing all the localization,
per the original two-model design.

IMPORTANT CAVEAT carried over from the InternVL version: at least one of the
six PoC frames ("woman in the white dress" / clip 1710_1890) has a GT target
that is a wrong-entity identification problem (names the wrong person
entirely), not a granularity or model-choice problem. No model swap fixes
that; it's expected to still fail here too, and remains a separate failure
mode worth tracking on its own.
"""

import os
import re
import math
import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration, AutoProcessor as Qwen3VLProcessor,
    AutoProcessor as GDinoProcessor, AutoModelForZeroShotObjectDetection,
)

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH        = os.environ.get("VAT_BASE_PATH", r"C:\repo\standard_project\videoattentiontarget")
QWEN3VL_NAME     = "Qwen/Qwen3-VL-8B-Instruct"
GDINO_NAME       = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD    = 0.30   # Grounding DINO confidence threshold for keeping a box
TEXT_THRESHOLD   = 0.25

# Same six frames as the InternVL3 baseline run, so results are directly
# comparable (0.436 production box-centroid baseline; 0.554/0.452 from the
# earlier DINOv2+InternVL hybrid attempts; InternVL3 targeted-phrase run is
# the immediate comparison point for this file).
TARGET_FRAMES = [
    ("17742_17893", 17826, "s00"),
    ("10239_10740", 10270, "s01"),
    ("14250_14430", 14250, "s01"),
    ("19636_19829", 19658, "s00"),
    ("1710_1890",   1737,  "s02"),
    ("1348_1469",   1382,  "s00"),
]

# Identical prompt to the InternVL3 baseline (step46_internvl..._002.py's
# TARGETED_PROMPT) -- unchanged on purpose, so this run isolates the model
# swap as the only variable.
TARGETED_PROMPT = (
    "Analysis Steps:"
    "1. Examine the subject's face orientation and precise eye/pupil direction."
    "2. Trace the vector of their gaze to find the MOST SPECIFIC point they are looking at -- "
    "this could be a person's face, a person's hand or another specific body part, an object, "
    "or entirely outside the visible frame."
    "3. If they are looking at a particular body part of a person (e.g. an outstretched hand, "
    "not that person's face) -- identify that specific body part, not just the person. "
    "Only use a general person descriptor if no specific body part is the clear target."
    "4. Write 1-2 sentences of step-by-step reasoning."
    " "
    "If looking at a specific body part of a person (not their face/head): Answer with the body "
    "part plus a brief visual descriptor of that person "
    "(e.g. 'hand of man in green coat', 'outstretched arm of woman in red'). 3-6 words maximum."
    "If looking at a person's face/head, or at the person generally with no specific body part "
    "evident: Answer with a brief visual descriptor of that person only "
    "(e.g. 'person in gray suit', 'woman on the right', 'man with beard', 'outstretched hand of man in blue coat'). 2-5 words maximum."
    "If looking at an object: Answer with ONLY the object name plus a brief location hint if "
    "multiple similar objects exist (e.g. 'newspaper', 'red coffee mug', 'plate on the left'). "
    "2-5 words maximum."
    " "
    "PERSPECTIVE RULE: ALL location hints (left/right/etc.) must use the VIEWER's perspective "
    "looking AT the image -- NOT the subject's own left/right, and NOT any other person's "
    "left/right. Worked example: if a photo shows three people and the middle person's own "
    "right hand points toward the person standing on the LEFT side of the photo as the viewer "
    "sees it, the correct answer is 'person on the left' (viewer's left), even though it is the "
    "middle person's OWN right hand doing the pointing. Always describe positions exactly as a "
    "viewer looking at the photo would describe them."
    "If looking off-screen: answer 'off-screen'."
)

# ── LOOKUP ENGINE (unchanged, copied from the InternVL3 baseline file) ──────
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
    Light cleanup of Qwen3-VL's direct-name response. Identical logic to the
    InternVL3 baseline's extract_object_phrase -- kept unchanged so any
    difference in downstream GDINO hit rate reflects the phrase Qwen3-VL
    produced, not a difference in how phrases are cleaned up.
    """
    cleaned = description_text.strip().strip('."\'').strip()

    if "off-screen" in cleaned.lower() or "off screen" in cleaned.lower():
        return None

    # Safety net: model ignored the brevity instruction and produced a full
    # sentence -- extract the core noun phrase rather than sending the whole
    # sentence to Grounding DINO.
    if len(cleaned.split()) > 7:
        match = re.search(
            r"looking (?:at|towards?|toward) (?:the |a |an )?([^.,;]+)",
            cleaned, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return " ".join(cleaned.split()[:5])

    return cleaned


def main():
    print("[*] Loading Qwen3-VL-8B-Instruct (semantic description)...")
    qwen_processor = Qwen3VLProcessor.from_pretrained(QWEN3VL_NAME)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3VL_NAME, dtype="auto", device_map="auto"
    )
    qwen_model.eval()

    print("[*] Loading Grounding DINO (text-conditioned localization)...")
    gdino_processor = GDinoProcessor.from_pretrained(GDINO_NAME)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_NAME).to("cuda:0")
    gdino_model.eval()

    print("\n[+] Both models loaded. Running hybrid pipeline (Qwen3-VL + targeted-phrase prompt)...\n")
    print(f"{'Frame / Subject':<28} | {'Object Phrase':<32} | {'Dist':<8} | {'vs box-centroid':<16} | vs InternVL3")
    print("-" * 110)

    # Reference numbers for direct print-time comparison. PRIOR_BOX_DIST is
    # the production box-centroid baseline (same as the InternVL3 file).
    # PRIOR_INTERNVL_DIST should be filled in with the actual per-frame
    # distances printed by step46_internvl_gdino_targeted_phrase_poc_002.py
    # once that run has been done -- placeholders below are copied from the
    # box-centroid row only as a safe default so this script runs standalone;
    # replace with the real InternVL3 numbers for a true model-vs-model column.
    PRIOR_BOX_DIST      = [0.259, 0.570, 0.023, 0.040, 0.281, 0.331]
    PRIOR_INTERNVL_DIST = [0.259, 0.570, 0.023, 0.040, 0.281, 0.331]  # <-- replace with real InternVL3 run output

    dists = []

    for i, (seq_dir, frame_idx, subject_id) in enumerate(TARGET_FRAMES):
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

        # --- Step 1: Qwen3-VL describes the gaze target (same prompt as InternVL3 baseline) ---
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": TARGETED_PROMPT}]}]
        inputs = qwen_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(qwen_model.device)

        with torch.inference_mode():
            gen_ids = qwen_model.generate(**inputs, max_new_tokens=40, do_sample=False)
        gen_trimmed = gen_ids[0][inputs["input_ids"].shape[1]:]
        description = qwen_processor.decode(gen_trimmed, skip_special_tokens=True).strip()

        object_phrase = extract_object_phrase(description)

        if object_phrase is None:
            print(f"{frame_label:<28} | (off-screen per VLM)")
            continue

        # --- Step 2: Grounding DINO localizes that exact phrase (unchanged) ---
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
            print(f"{frame_label:<28} | {object_phrase[:32]:<32} | NO BOX FOUND (below threshold)")
            continue

        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].tolist()  # [x1, y1, x2, y2] in pixel coords

        cx = ((box[0] + box[2]) / 2) / w
        cy = ((box[1] + box[3]) / 2) / h

        if gt_norm is None:
            print(f"{frame_label:<28} | {object_phrase[:32]:<32} | (GT off-screen -- not comparable)")
            continue

        dist = math.sqrt((gt_norm[0] - cx) ** 2 + (gt_norm[1] - cy) ** 2)
        dists.append(dist)

        prior_box = PRIOR_BOX_DIST[i]
        prior_internvl = PRIOR_INTERNVL_DIST[i]
        vs_box      = f"{prior_box - dist:+.3f}"
        vs_internvl = f"{prior_internvl - dist:+.3f}"

        print(f"{frame_label:<28} | {object_phrase[:32]:<32} | {dist:<8.3f} | {vs_box:<16} | {vs_internvl}")

    print("-" * 110)
    if dists:
        mean_dist = sum(dists) / len(dists)
        mean_prior_box = sum(PRIOR_BOX_DIST) / len(PRIOR_BOX_DIST)
        mean_prior_internvl = sum(PRIOR_INTERNVL_DIST) / len(PRIOR_INTERNVL_DIST)
        print(f"\nMean ADE, Qwen3-VL targeted-phrase (n={len(dists)}): {mean_dist:.4f}")
        print(f"  vs box-centroid   ({mean_prior_box:.4f}): {mean_prior_box - mean_dist:+.4f}")
        print(f"  vs InternVL3 run  ({mean_prior_internvl:.4f}): {mean_prior_internvl - mean_dist:+.4f}")
        print("\nNOTE: PRIOR_INTERNVL_DIST above is a placeholder (copied from box-centroid) until you "
              "paste in the real per-frame distances printed by "
              "step46_internvl_gdino_targeted_phrase_poc_002.py -- update it for a true model-vs-model diff.")
        print("(positive = Qwen3-VL improved on that method; note 'woman in white dress' / "
              "clip 1710_1890 is expected to still fail on both models -- wrong-entity "
              "identification, not a model-choice issue -- see docstring)")


if __name__ == "__main__":
    main()
