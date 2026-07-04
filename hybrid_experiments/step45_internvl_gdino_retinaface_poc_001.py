"""
step44_internvl_gdino_retinaface_poc.py — Proof-of-concept, NO TRAINING.

Extends the InternVL3 + Grounding DINO hybrid (step44_..._006b.py) with a
third stage aimed specifically at the person-target case:

  Stage 1: InternVL3 identifies the gaze target verbally (unchanged).
  Stage 2: Grounding DINO localizes that phrase -> bounding box (unchanged).
  Stage 3 (NEW): crop to the GDINO box (with a small padding margin) and run
    RetinaFace on the crop. If a face is found, refine the predicted
    coordinate to the FACE centroid (mapped back to original image coords)
    instead of the full-body box centroid. If RetinaFace finds nothing
    (side profile, occlusion, back of head, or the crop genuinely isn't a
    person -- e.g. an object crop), silently fall back to the existing
    GDINO box-centroid prediction. Stage 3 runs unconditionally on every
    crop (person or object) rather than gating on phrase content -- it can
    only help or be a no-op, since a failed face search just falls back to
    the box-centroid method already proven to beat raw InternVL3-alone.

MOTIVATION: the step45a production-pipeline TEST_MODE run (2026-07-02)
showed mean ADE 0.329 across 7 person-target frames (Conan, clip
1348_1469, subject s00, same clip reused here as TARGET_FRAMES[5] for a
direct before/after comparison) -- predictions clustered around box-centroid
height (~0.61 normalised y) while GT sat much lower, near head/face level
(~0.92). This matches the previously-identified structural limitation:
bounding-box centroid vs GT gaze point creates an irreducible floor for
person targets. This PoC tests whether refining to a face-crop centroid
closes that gap.

Both models plus RetinaFace are used frozen, at inference only. No
training, no gradients, no checkpointing needed.

FIRST RUN NOTE #1: grounding-dino-base is a dependency already validated in
production (~700MB, one-time HF weight download).

FIRST RUN NOTE #2: retina-face is a NEW, UNTESTED dependency in this
project (TensorFlow-backed face detector). Install with:
    pip install retina-face
The exact return structure (dict keys, box format, BGR vs RGB expectation)
has NOT been verified against your installed version -- this script prints
the raw detection dict on the FIRST successful detection so you can sanity-
check the assumed structure (score / facial_area keys, [x1,y1,x2,y2] box
order) before trusting the numbers. This is the same kind of "new
dependency, unverified API" risk that caused the GDINO threshold/box_threshold
rename issue earlier in this investigation -- check the printed dict before
trusting the refined coordinates.
"""

import os
import re
import math
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoModelForImageTextToText, AutoProcessor as InternVLProcessor,
    AutoProcessor as GDinoProcessor, AutoModelForZeroShotObjectDetection,
)

try:
    from retinaface import RetinaFace
except ImportError:
    raise ImportError(
        "retina-face is not installed. Run: pip install retina-face\n"
        "This is a NEW dependency for this project -- first run will "
        "download detector weights."
    )

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH        = os.environ.get("VAT_BASE_PATH", r"C:\repo\standard_project\videoattentiontarget")
INTERNVL_NAME    = "OpenGVLab/InternVL3-8B-hf"
GDINO_NAME       = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD    = 0.30   # Grounding DINO confidence threshold for keeping a box
TEXT_THRESHOLD   = 0.25
FACE_CROP_PAD    = 0.15   # Pad the GDINO box by this fraction before face search,
                           # so a face near the box edge isn't clipped out.

_printed_raw_face_dict = False  # Flip True after first detection -- see NOTE #2

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

# <<< this one is the best so far!
DESCRIBE_PROMPT = (
    "Analysis Steps:"
    "1. Examine the subject's face orientation and precise eye/pupil direction."
    "2. Check if the vector of their eye gaze points at a person, object or entirely outside the visible image frame boundaries."
    "3. If their eye gaze points at a object or person, identify what object or person they are looking at."
    "4. Write 1-2 sentences of step-by-step reasoning."
    " "
    "If looking at an object: Answer with ONLY the object name plus a brief location hint if multiple similar objects exist "
    "(e.g. 'newspaper', 'red coffee mug', 'plate on the left'). 2-5 words maximum."
    "If looking at a person: Answer with a brief visual descriptor of that person only "
    "(e.g. 'person in gray suit', 'woman on the right', 'man with beard'). 2-5 words maximum."
    "Location hints should be based on the image viewers perspective"
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


def refine_with_face(img, box, w, h):
    """
    Stage 3: crop to the GDINO box (padded), run RetinaFace on the crop.

    Returns (face_x_norm, face_y_norm) in ORIGINAL image normalised coords
    if a face is found, else None (caller falls back to box centroid).

    Runs unconditionally regardless of whether the GDINO target was a
    person or an object -- an object crop simply won't contain a face and
    this returns None, which is a correctness no-op, not a failure mode.
    """
    global _printed_raw_face_dict

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * FACE_CROP_PAD, bh * FACE_CROP_PAD
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = img.crop((cx1, cy1, cx2, cy2))
    # PIL gives RGB; retina-face is built on cv2 conventions (BGR).
    # UNVERIFIED against the installed version -- if refined coordinates
    # look systematically wrong, try dropping the [:, :, ::-1] first.
    crop_bgr = np.array(crop)[:, :, ::-1]

    try:
        faces = RetinaFace.detect_faces(crop_bgr)
    except Exception as e:
        print(f"    [RetinaFace error, falling back to box centroid: {e}]")
        return None

    if not isinstance(faces, dict) or len(faces) == 0:
        return None

    if not _printed_raw_face_dict:
        print(f"\n    [SANITY CHECK] Raw RetinaFace output on first hit:\n    {faces}\n"
              f"    Confirm 'score' and 'facial_area': [x1,y1,x2,y2] keys match this.\n")
        _printed_raw_face_dict = True

    best_face = max(faces.values(), key=lambda f: f["score"])
    fx1, fy1, fx2, fy2 = best_face["facial_area"]

    face_cx = cx1 + (fx1 + fx2) / 2
    face_cy = cy1 + (fy1 + fy2) / 2
    return face_cx / w, face_cy / h



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
    print(f"{'Frame / Subject':<28} | {'Object Phrase':<28} | {'Box dist':<10} | {'Face dist':<10} | Delta")
    print("-" * 100)

    box_dists, face_dists = [], []

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
            print(f"{frame_label:<28} | (off-screen per VLM)")
            continue

        # --- Step 2: Grounding DINO localizes that exact phrase ---
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
            print(f"{frame_label:<28} | {object_phrase[:28]:<28} | NO BOX FOUND (below threshold)")
            continue

        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].tolist()  # [x1, y1, x2, y2] in pixel coords

        cx_box = ((box[0] + box[2]) / 2) / w
        cy_box = ((box[1] + box[3]) / 2) / h

        # --- Step 3 (NEW): refine via face crop ---
        refined = refine_with_face(img, box, w, h)
        cx_face, cy_face = refined if refined is not None else (None, None)

        if gt_norm is None:
            print(f"{frame_label:<28} | {object_phrase[:28]:<28} | (GT off-screen -- not comparable)")
            continue

        dist_box = math.sqrt((gt_norm[0] - cx_box) ** 2 + (gt_norm[1] - cy_box) ** 2)
        box_dists.append(dist_box)

        if cx_face is not None:
            dist_face = math.sqrt((gt_norm[0] - cx_face) ** 2 + (gt_norm[1] - cy_face) ** 2)
            face_dists.append(dist_face)
            delta = dist_box - dist_face
            delta_str = f"{delta:+.3f} {'better' if delta > 0 else 'worse'}"
            face_str = f"{dist_face:.3f}"
        else:
            face_str = "no face"
            delta_str = "n/a"

        print(f"{frame_label:<28} | {object_phrase[:28]:<28} | {dist_box:<10.3f} | {face_str:<10} | {delta_str}")

    print("-" * 100)
    if box_dists:
        print(f"\nMean box-centroid ADE   (n={len(box_dists)}): {sum(box_dists)/len(box_dists):.4f}")
    if face_dists:
        print(f"Mean face-refined ADE   (n={len(face_dists)}): {sum(face_dists)/len(face_dists):.4f}")
        print(f"  (face refinement found a usable face on {len(face_dists)}/{len(box_dists)} frames)")
    else:
        print("Face refinement found no usable faces on any frame -- check FACE_CROP_PAD, "
              "the BGR/RGB conversion note in the docstring, and the sanity-check dict above "
              "if one was printed.")


if __name__ == "__main__":
    main()
