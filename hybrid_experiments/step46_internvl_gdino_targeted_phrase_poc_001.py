"""
step44_internvl_gdino_targeted_phrase_poc.py — Proof-of-concept, NO TRAINING.

Third localization approach tested today, after:
  - DINOv2+InternVL3 hybrid projector (closed, negative -- separate investigation)
  - GDINO box-centroid alone (ACTIVE in production, step45a)
  - GDINO + RetinaFace face-refinement (closed, negative -- see PoC results
    2026-07-02: mean ADE 0.2507 box-centroid vs 0.3265 face-refined, n=6.
    Two distinct failure modes identified: (1) RetinaFace regresses when
    GT targets a non-face body part -- confirmed directly on the
    "woman in the white dress" frame, where GT is actually the outstretched
    hand of a THIRD person (man in green coat) reaching toward her -- not
    her face, and not even her; (2) crowd contamination -- "highest
    confidence face in the padded crop" has no way to prefer the correct
    person's face over a bystander's in socially dense scenes.)

This PoC tests a DIFFERENT lever: rather than adding a refinement stage
after GDINO, ask InternVL3 to produce a MORE TARGETED phrase up front --
naming the specific body part (hand, arm, etc.) when that's the actual
gaze target, rather than defaulting to a whole-person descriptor -- and
feed that directly to GDINO, same as the original hybrid. No RetinaFace,
no third stage. Also fixes a separate, independently-confirmed bug found
in the RetinaFace PoC run: InternVL3 gave "woman on the left" for a target
actually on the LEFT of the SUBJECT but the RIGHT from the viewer's
perspective -- the previous prompt's abstract "use viewer's perspective"
instruction wasn't sufficient; this version adds a concrete worked example.

IMPORTANT CAVEAT going in: at least one of the six PoC frames ("woman in
the white dress" / clip 1710_1890) has a GT target that InternVL3
misidentifies at the ENTITY level, not just the granularity level -- it
names the wrong person (woman) rather than the correct one (the hand of
the man in green coat reaching toward her). No prompt change to body-part
specificity fixes a wrong-entity identification; that frame is expected to
still fail here, and is a separate, more fundamental failure mode worth
documenting on its own rather than something this test is trying to solve.
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

# <<< this one is the SECOND best so far!
DESCRIBE_PROMPT_001 = (
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

# <<< NEW: targeted-phrase version — body-part specificity + concrete
# perspective example (replaces the abstract-only instruction that failed
# on "woman on the left" in the RetinaFace PoC run)
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
    "(e.g. 'person in gray suit', 'woman on the right', 'man with beard'). 2-5 words maximum."
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

    print("\n[+] Both models loaded. Running hybrid pipeline (targeted-phrase prompt)...\n")
    print(f"{'Frame / Subject':<28} | {'Object Phrase':<32} | {'Dist':<8} | {'vs box-centroid':<16} | vs face-refined")
    print("-" * 110)

    # Reference numbers from today's earlier PoC runs, same 6 frames, for
    # direct print-time comparison (box-centroid baseline; RetinaFace closed
    # result). Indexed to match TARGET_FRAMES order.
    PRIOR_BOX_DIST  = [0.259, 0.570, 0.023, 0.040, 0.281, 0.331]
    PRIOR_FACE_DIST = [0.453, 0.535, 0.145, 0.226, 0.135, 0.465]

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

        # --- Step 1: InternVL3 describes the gaze target (targeted prompt) ---
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": TARGETED_PROMPT}]}]
        inputs = internvl_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(internvl_model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            gen_ids = internvl_model.generate(**inputs, max_new_tokens=40, do_sample=False)
        description = internvl_processor.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
        prior_face = PRIOR_FACE_DIST[i]
        vs_box  = f"{prior_box - dist:+.3f}"
        vs_face = f"{prior_face - dist:+.3f}"

        print(f"{frame_label:<28} | {object_phrase[:32]:<32} | {dist:<8.3f} | {vs_box:<16} | {vs_face}")

    print("-" * 110)
    if dists:
        mean_dist = sum(dists) / len(dists)
        mean_prior_box = sum(PRIOR_BOX_DIST) / len(PRIOR_BOX_DIST)
        mean_prior_face = sum(PRIOR_FACE_DIST) / len(PRIOR_FACE_DIST)
        print(f"\nMean ADE, targeted-phrase (n={len(dists)}): {mean_dist:.4f}")
        print(f"  vs box-centroid  ({mean_prior_box:.4f}): {mean_prior_box - mean_dist:+.4f}")
        print(f"  vs face-refined  ({mean_prior_face:.4f}): {mean_prior_face - mean_dist:+.4f}")
        print("\n(positive = targeted-phrase improved on that method; note 'woman in white dress' / "
              "clip 1710_1890 is expected to still fail -- wrong-entity identification, not a "
              "granularity issue -- see docstring)")


if __name__ == "__main__":
    main()
