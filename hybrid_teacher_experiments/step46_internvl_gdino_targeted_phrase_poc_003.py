"""
step46_internvl_gdino_targeted_phrase_poc_003.py — Proof-of-concept, NO TRAINING.

Fifth localization/identification lever tested, after:
  - DINOv2+InternVL3 hybrid projector (closed, negative)
  - GDINO box-centroid alone (ACTIVE in production, step45a)
  - GDINO + RetinaFace face-refinement (closed, negative)
  - GDINO + targeted-phrase prompt (poc_001, mean ADE 0.1940 on this 6-frame set)
  - poc_001 + laterality-aware GDINO box selection (poc_002, mean ADE 0.1934)

MOTIVATION: a DIFFERENT failure mode from anything above -- WRONG-ENTITY
identification, where InternVL3 names the wrong person/object entirely
(not just the wrong instance of the right category). Concretely: on clip
1710_1890 ("woman in the white dress"), the actual GT target is the
outstretched hand of a THIRD person (man in green coat) reaching toward
her -- not her, and not her face. No amount of GDINO-side box selection
fixes this, because the correct candidate is never even sent to GDINO --
InternVL3's identification step never mentions it. This has been
documented since poc_001's docstring as a known, separate, more
fundamental failure mode than anything the targeted-phrase or laterality
work addresses.

CHANGE FROM poc_002 (ONLY CHANGE): rather than a single greedy-decoded
identification call, InternVL3 is now sampled N_SAMPLES times (do_sample=
True) for the identification step, and a majority vote across samples
decides the phrase sent forward to GDINO. This is a form of self-
consistency prompting -- if the model's single greedy answer is a
reasoning slip (defaulting to the more "obvious" completion, e.g. a
visible face, over a less obvious but correct one, e.g. someone else's
hand), sampling gives it repeated independent chances to instead surface
the correct answer, and disagreement across samples is itself a usable
confidence signal that step46/step47's audit trail does not currently
capture at all for the identification step (only GDINO's own detection
score is captured today).

Everything else -- TARGETED_PROMPT wording, laterality-aware box selection
from poc_002, the 6 target frames -- is UNCHANGED, so any shift in results
is attributable to sampling alone, not conflated with any other change
(specifically NOT combined with the enumeration-restructuring idea also
discussed -- that is deliberately left for its own separate poc_004, so
each lever's effect stays isolated and attributable, consistent with every
prior step in this investigation).

HONEST EXPECTATION GOING IN: self-consistency sampling helps with
reasoning slips where the correct answer is a plausible completion the
model sometimes produces and sometimes doesn't. It will NOT help if the
model essentially never considers the correct answer across any
reasonable number of samples (e.g. a third person's hand that is easy to
overlook entirely). The clip 1710_1890 case is retained in this PoC
specifically as a stress test of that: if 5 samples never surface "hand of
man in green coat" at all, that is itself an informative (negative)
result, not a bug in this script.
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

# NEW IN poc_003: self-consistency sampling for the identification step.
N_SAMPLES         = 5     # odd number -- avoids exact ties in majority vote
SAMPLE_TEMPERATURE = 0.7  # standard self-consistency default; not tuned
SAMPLE_TOP_P       = 0.9

# Reuse the SAME multi-person object-gaze frames used throughout this
# investigation -- so results are directly comparable against every prior
# ADE figure (0.436 production baseline; 0.554/0.452 DINOv2+InternVL hybrid;
# 0.2507 box-centroid; 0.3265 face-refined; 0.1940 targeted-phrase/poc_001).
TARGET_FRAMES = [
    ("17742_17893", 17826, "s00"),
    ("10239_10740", 10270, "s01"),
    ("14250_14430", 14250, "s01"),
    ("19636_19829", 19658, "s00"),
    ("1710_1890",   1737,  "s02"),
    ("1348_1469",   1382,  "s00"),
]

# Unchanged from poc_001.
TARGETED_PROMPT_001 = (
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

TARGETED_PROMPT = (
    "Identify all the people in the image and what they are doing"
)

# ── LOOKUP ENGINE (copied verbatim from the proven step43d7_prompt_compare.py,
# NOT reinvented -- unchanged from poc_001) ──────────────────────────────────
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
    Unchanged from poc_001. Light cleanup of InternVL3's direct-name
    response. Person descriptors are PRESERVED rather than collapsed to
    bare 'person' -- so Grounding DINO receives 'person in gray suit' or
    'woman on the right' rather than 'person', which is spatially
    ambiguous in multi-person frames.
    """
    cleaned = description_text.strip().strip('."\'').strip()

    if "off-screen" in cleaned.lower() or "off screen" in cleaned.lower():
        return None

    if len(cleaned.split()) > 7:
        match = re.search(
            r"looking (?:at|towards?|toward) (?:the |a |an )?([^.,;]+)",
            cleaned, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return " ".join(cleaned.split()[:5])

    return cleaned


# ── NEW IN poc_003: self-consistency majority vote ───────────────────────
def majority_vote_phrase(phrases):
    """
    phrases: list of object_phrase strings (or None for off-screen), one
    per sample. Returns (winning_phrase_or_None, agreement_fraction).

    Voting is on the EXACT cleaned phrase string, deliberately simple --
    no attempt to cluster near-duplicate phrasings ("woman in white dress"
    vs "woman in a white dress") into the same bucket. That's a real
    limitation worth checking in the printed raw samples below: if
    near-duplicate phrasings are splitting votes that should agree, the
    agreement fraction here will UNDERSTATE true consistency. Flagged
    rather than silently handled, since a fuzzy-match heuristic risks
    merging genuinely different answers.
    """
    if not phrases:
        return None, 0.0
    counts = {}
    for p in phrases:
        counts[p] = counts.get(p, 0) + 1
    winner = max(counts, key=counts.get)
    agreement = counts[winner] / len(phrases)
    return winner, agreement


# ── NEW IN poc_002: laterality-aware box selection ───────────────────────
LATERALITY_PATTERN = re.compile(
    r"^(.*?)\s+(?:on\s+the\s+|at\s+the\s+)?(left|right|center|middle)\s*$",
    re.IGNORECASE
)


def split_phrase_and_laterality(phrase):
    """
    Split 'woman on the left' -> ('woman', 'left'). Returns (phrase, None)
    if no positional qualifier is found, so callers fall back safely to
    poc_001 behaviour.
    """
    match = LATERALITY_PATTERN.match(phrase.strip())
    if match:
        core = match.group(1).strip()
        laterality = match.group(2).lower()
        laterality = "center" if laterality == "middle" else laterality
        if core:  # guard against over-stripping to an empty phrase
            return core, laterality
    return phrase, None


def select_box(boxes, scores, img_w, laterality=None):
    """
    poc_001 behaviour (highest-confidence box) UNLESS InternVL3 gave a
    laterality hint AND GDINO returned more than one candidate box -- in
    which case pick by x-position instead of raw score.
    """
    if laterality is None or len(boxes) == 1:
        best_idx = scores.argmax().item()
        return boxes[best_idx].tolist(), scores[best_idx].item(), False

    centers_x = [((b[0] + b[2]) / 2) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: centers_x[i])

    if laterality == "left":
        chosen = order[0]
    elif laterality == "right":
        chosen = order[-1]
    else:  # "center" -- closest to horizontal midpoint
        mid = img_w / 2
        chosen = min(range(len(boxes)), key=lambda i: abs(centers_x[i] - mid))

    return boxes[chosen].tolist(), scores[chosen].item(), True


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

    print("\n[+] Both models loaded. Running hybrid pipeline (targeted-phrase + laterality-aware selection + self-consistency sampling)...\n")
    print(f"{'Frame / Subject':<28} | {'Object Phrase':<32} | {'Agree':<6} | {'#Boxes':<7} | {'Used Lat.':<9} | {'Dist':<8} | vs poc_002")
    print("-" * 125)

    # Reference numbers from poc_002 (targeted-phrase + laterality-aware
    # selection, greedy decoding), same 6 frames, for direct comparison.
    PRIOR_TARGETED_DIST = [0.263, 0.569, 0.023, 0.040, 0.064, 0.202]  # poc_002's actual per-frame distances

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

        # --- Step 1: InternVL3 describes the gaze target (targeted prompt, unchanged) ---
        # NEW IN poc_003: sample N_SAMPLES times instead of one greedy call,
        # then majority-vote on the resulting phrase. Everything about the
        # prompt itself, and everything downstream (GDINO query, laterality
        # split, box selection), is unchanged from poc_002.
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": TARGETED_PROMPT}]}]
        inputs = internvl_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(internvl_model.device, dtype=torch.bfloat16)

        sample_phrases = []
        sample_descriptions = []
        for _ in range(N_SAMPLES):
            with torch.inference_mode():
                gen_ids = internvl_model.generate(
                    **inputs, max_new_tokens=40, do_sample=True,
                    temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P)
            description = internvl_processor.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            sample_descriptions.append(description)
            sample_phrases.append(extract_object_phrase(description))

        object_phrase, agreement = majority_vote_phrase(sample_phrases)
        description = sample_descriptions[0]  # kept for parity with poc_002's variable name; not otherwise used

        # Diagnostic: print raw samples so disagreement is visible, not hidden
        # behind the majority-vote number alone.
        unique_samples = sorted(set(p if p is not None else "(off-screen)" for p in sample_phrases))
        print(f"    samples: {unique_samples} -> agreement {agreement:.2f}")

        if object_phrase is None:
            print(f"{frame_label:<28} | (off-screen per VLM)")
            continue

        # --- Step 2: Grounding DINO localizes the CORE phrase (NEW: laterality stripped) ---
        core_phrase, laterality = split_phrase_and_laterality(object_phrase)
        gdino_query = core_phrase.lower().strip().rstrip(".") + "."
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

        # --- Step 3: laterality-aware selection (NEW) ---
        box, score, used_laterality = select_box(
            results["boxes"], results["scores"], w, laterality)

        cx = ((box[0] + box[2]) / 2) / w
        cy = ((box[1] + box[3]) / 2) / h

        if gt_norm is None:
            print(f"{frame_label:<28} | {object_phrase[:32]:<32} | (GT off-screen -- not comparable)")
            continue

        dist = math.sqrt((gt_norm[0] - cx) ** 2 + (gt_norm[1] - cy) ** 2)
        dists.append(dist)

        n_boxes = len(results["boxes"])
        lat_flag = f"{laterality}✓" if used_laterality else ("-" if laterality is None else f"{laterality}(unused)")
        vs_prior = f"{PRIOR_TARGETED_DIST[i] - dist:+.3f}" if i < len(PRIOR_TARGETED_DIST) else "n/a"

        print(f"{frame_label:<28} | {object_phrase[:32]:<32} | {agreement:<6.2f} | {n_boxes:<7} | {lat_flag:<9} | {dist:<8.3f} | {vs_prior}")

    print("-" * 125)
    if dists:
        mean_dist = sum(dists) / len(dists)
        print(f"\nMean ADE, targeted-phrase + laterality-aware selection + self-consistency (n={len(dists)}): {mean_dist:.4f}")
        print("(compare against poc_002's mean of 0.1934, and poc_001's 0.1940, on the same 6 frames)")
        print(f"\nSelf-consistency settings: N_SAMPLES={N_SAMPLES}, temperature={SAMPLE_TEMPERATURE}, top_p={SAMPLE_TOP_P}")
        print("Check the per-frame 'samples' printout above, especially for clip 1710_1890 --")
        print("if 'hand of man in green coat' (or similar) never appears across 5 samples, that is")
        print("a documented NEGATIVE result for self-consistency on wrong-entity identification,")
        print("not a bug -- see module docstring.")


if __name__ == "__main__":
    main()
