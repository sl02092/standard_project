"""
step46_hybrid_teacher_pipeline.py — Hybrid Teacher Pipeline (targeted-phrase prompt)
InternVL3-8B-hf (semantic identification, TARGETED-PHRASE prompt) + Grounding
DINO (text-conditioned spatial localization), for object-gaze coordinate
prediction.

RELATIONSHIP TO step45a_hybrid_teacher_pipeline.py:
  This is NOT a replacement -- step45a is still running in full production
  on the HPC as of this script's creation (job 627077, labels_hybrid.jsonl).
  This script uses ENTIRELY SEPARATE output/progress/log filenames so it can
  be queued and run without touching step45a's in-progress output. Once both
  have finished, labels_hybrid.jsonl (step45a, box-centroid + old prompt) and
  labels_hybrid_targeted.jsonl (this script) can be compared directly on the
  same schema.

MOTIVATION (from experimental evidence, step44/step46 local PoC investigation):
  - Local PoC comparison on the same 6 object-gaze frames:
      box-centroid (old prompt):    mean ADE 0.2507
      face-refined (closed, negative): mean ADE 0.3265
      targeted-phrase (this prompt): mean ADE 0.1940
  - Manual frame-by-frame verification against ground truth (pixel-level,
    cross-checked in image editing software) found the targeted-phrase
    outputs were qualitatively BETTER than the raw ADE numbers alone
    suggested -- several "misses" were reasonable inferences on genuinely
    ambiguous frames (motion blur, closed eyes, lost depth cues), not
    hallucinated detail. See conversation log 2026-07-02 for the full
    per-frame verification.
  This pipeline tests whether the targeted-phrase improvement holds at
  production scale, producing labels_hybrid_targeted.jsonl in identical
  schema to labels_hybrid.jsonl / labels_full.jsonl for direct comparison.

ROUTING LOGIC (identical to step45a):
  - use_teacher == False  → GT direct (off-screen frames)
  - use_teacher == True, gaze_type == "social"  → InternVL3-alone
    (social gaze is InternVL3's strength; hybrid adds no value here)
  - use_teacher == True, gaze_type == "object"  → HYBRID (InternVL3 + GDINO),
    using the TARGETED_PROMPT (only change from step45a)
  - GDINO no-box-found or InternVL3 off-screen on object frame → GT fallback

THREE MODES (same as step45a):
  PROMPT_TEST = True  → 10 frames, prints details, no production output
  TEST_MODE   = True  → ~50 frames, end-to-end validation
  TEST_MODE   = False → full 23,096-frame production run

OUTPUT: labels_hybrid_targeted.jsonl (same schema as labels_hybrid.jsonl)
  teacher_version field distinguishes this run:
  "internvl3-8b-v4-prompt"            for social frames (InternVL3-alone, unchanged)
  "internvl3+gdino-base-v2-targeted"  for object frames (targeted-phrase prompt)

CHANGELOG vs step45a_hybrid_teacher_pipeline.py:
  - ONLY CHANGE: IDENTIFY_PROMPT replaced with TARGETED_PROMPT (body-part
    specificity + concrete perspective worked example -- see step46 PoC).
    Everything else (dtype handling, routing, checkpointing, GT fallback
    logic) is untouched and already validated via step45a's own TEST_MODE
    run and ongoing full production run.
  - TEST_MODE = True for this run, same reasoning as step45a's first run:
    this exact prompt has been validated in a 6-frame local PoC only, never
    inside the production pipeline (checkpointing, batch routing, retry
    logic). Confirm clean before flipping to full production.
  - Output/progress/log filenames changed throughout to avoid any collision
    with step45a's in-progress files.
"""

import os
import re
import csv
import json
import math
import time
import torch
import tempfile
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    AutoModelForZeroShotObjectDetection,
)

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

PROMPT_TEST = False
TEST_MODE   = True   # first run with this prompt inside the production
                      # pipeline -- validate end-to-end before full run,
                      # same reasoning as step45a's first run.

BASE_PATH     = os.environ.get("VAT_BASE_PATH",
    "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget")
MANIFEST_PATH = "frame_manifest.csv"
INTERNVL_NAME = "OpenGVLab/InternVL3-8B-hf"
GDINO_NAME    = "IDEA-Research/grounding-dino-base"

TEACHER_VERSION_SOCIAL = "internvl3-8b-v4-prompt"           # social frames unchanged
TEACHER_VERSION_HYBRID = "internvl3+gdino-base-v2-targeted" # object frames: targeted-phrase prompt

# Output files -- ALL DISTINCT from step45a's filenames (labels_hybrid*.jsonl,
# progress_hybrid*.json, pipeline_hybrid*.log) so this can run without
# touching step45a's in-progress production output.
if PROMPT_TEST:
    LABELS_FILE   = "labels_hybrid_targeted_prompt_test.jsonl"
    PROGRESS_FILE = "progress_hybrid_targeted_prompt_test.json"
    LOG_FILE      = "pipeline_hybrid_targeted_prompt_test.log"
elif TEST_MODE:
    LABELS_FILE   = "labels_hybrid_targeted_test.jsonl"
    PROGRESS_FILE = "progress_hybrid_targeted_test.json"
    LOG_FILE      = "pipeline_hybrid_targeted_test.log"
else:
    LABELS_FILE   = "labels_hybrid_targeted.jsonl"
    PROGRESS_FILE = "progress_hybrid_targeted.json"
    LOG_FILE      = "pipeline_hybrid_targeted.log"

# Prompt test / test mode settings (same as step40)
PROMPT_TEST_N     = 10
PROMPT_TEST_SHOWS = ["My Dinner with Andre", "Conan", "Gone with the Wind", "Tartuffe"]
TEST_MAX_FRAMES_PER_SHOW = 17
TEST_SHOWS = ["My Dinner with Andre", "Conan", "Gone with the Wind"]

# Inference settings
MAX_NEW_TOKENS  = 60   # shorter than step40 since hybrid only needs object name
SAVE_EVERY_N    = 10
GT_SAVE_EVERY_N = 100
RETRY_ON_FAIL   = 2
BOX_THRESHOLD   = 0.30
TEXT_THRESHOLD  = 0.25

# ══════════════════════════════════════════════════════════════════════
# ── PROMPTS ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_social_prompt(img_w, img_h, primary_box, secondary_box):
    """Unchanged from step40 — social frames use InternVL3-alone."""
    ax1, ay1, ax2, ay2 = primary_box
    bx1, by1, bx2, by2 = secondary_box
    b_cx = round((bx1 + bx2) / 2 / img_w, 3)
    b_cy = round((by1 + by2) / 2 / img_h, 3)
    na = (round(ax1/img_w,3), round(ay1/img_h,3),
          round(ax2/img_w,3), round(ay2/img_h,3))
    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.

PERSON A (predict gaze): head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})
PERSON B (other person): head box px ({bx1},{by1}) to ({bx2},{by2}), face centre ({b_cx},{b_cy})

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)

Rules:
- If A looks at B's face → use B's face centre: ({b_cx},{b_cy})
- If A looks at an object → estimate object location in normalised coords
- If A looks off-screen → use (-1,-1)
- x,y are normalised 0.0-1.0 (0,0=top-left, 1,1=bottom-right)

After the coordinate, add one sentence explaining why."""


# Targeted-phrase prompt from step46 investigation -- body-part specificity
# + concrete perspective worked example. Replaces IDENTIFY_PROMPT from
# step45a (kept below, commented, for reference/rollback).
IDENTIFY_PROMPT = (
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

# Previous (step45a) identification prompt -- kept for reference/rollback.
# IDENTIFY_PROMPT_V1_STEP45A = (
#     "Analysis Steps:"
#     "1. Examine the subject's face orientation and precise eye/pupil direction."
#     "2. Check if the vector of their eye gaze points at a person, object or entirely outside the visible image frame boundaries."
#     "3. If their eye gaze points at an object or person, identify what object or person they are looking at."
#     "4. Write 1-2 sentences of step-by-step reasoning."
#     " "
#     "If looking at an object: Answer with ONLY the object name plus a brief location hint if multiple similar objects exist "
#     "(e.g. 'newspaper', 'red coffee mug', 'plate on the left'). 2-5 words maximum."
#     "If looking at a person: Answer with a brief visual descriptor of that person only "
#     "(e.g. 'person in gray suit', 'woman on the right', 'man with beard'). 2-5 words maximum."
#     "Location hints should be based on the image viewer's perspective. "
#     "If looking off-screen: answer 'off-screen'."
# )

# ══════════════════════════════════════════════════════════════════════
# ── PARSERS ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def parse_gaze_xy(text):
    """Unchanged from step40."""
    if not text or not text.strip():
        return None, None
    if re.search(r"GAZE_XY:\s*\(-1\s*,\s*-1\)", text):
        return -1.0, -1.0
    match = re.search(
        r"GAZE_XY:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text)
    if match:
        return (max(0.0, min(1.0, float(match.group(1)))),
                max(0.0, min(1.0, float(match.group(2)))))
    matches = re.findall(
        r"\(\s*(0\.[0-9]+|1\.0|0\.0)\s*,\s*(0\.[0-9]+|1\.0|0\.0)\s*\)", text)
    if matches:
        return (max(0.0, min(1.0, float(matches[0][0]))),
                max(0.0, min(1.0, float(matches[0][1]))))
    return None, None


def extract_object_phrase(description_text):
    """
    From step44 investigation -- light cleanup of InternVL3's direct response.
    Preserves person descriptors ('person in gray suit') rather than collapsing
    to bare 'person', which is spatially ambiguous in multi-person frames.
    """
    cleaned = description_text.strip().strip('."\'').strip()
    if "off-screen" in cleaned.lower() or "off screen" in cleaned.lower():
        return None
    if len(cleaned.split()) > 7:
        match = re.search(
            r"looking (?:at|towards?|toward) (?:the |a |an )?([^.,;]+)",
            cleaned, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return " ".join(cleaned.split()[:5])
    return cleaned

# ══════════════════════════════════════════════════════════════════════
# ── PROGRESS / CHECKPOINTING (identical to step40) ────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress):
    dir_name = os.path.dirname(os.path.abspath(PROGRESS_FILE))
    with tempfile.NamedTemporaryFile(
        "w", dir=dir_name, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(progress, tmp)
        tmp_path = tmp.name
    os.replace(tmp_path, PROGRESS_FILE)


def recover_progress_from_jsonl():
    if not os.path.exists(LABELS_FILE):
        return set()
    completed = set()
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pair_id = f"{rec['show']}|{rec['clip']}|{rec['fname']}|{rec['subject']}"
                completed.add(pair_id)
            except (json.JSONDecodeError, KeyError):
                continue
    log(f"Recovered {len(completed)} completed pairs from {LABELS_FILE}")
    return completed


def append_label(record):
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log(msg, also_print=True):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ══════════════════════════════════════════════════════════════════════
# ── MANIFEST LOADING (identical to step40) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    log(f"Manifest loaded: {len(df)} frame-subject pairs")

    if PROMPT_TEST:
        teacher_df = df[(df["use_teacher"] == True) &
                        (df["show"].isin(PROMPT_TEST_SHOWS))]
        unique = teacher_df.drop_duplicates(
            subset=["show","clip","fname"]).head(PROMPT_TEST_N)
        key = teacher_df[["show","clip","fname"]].apply(tuple, axis=1)
        sel_key = unique[["show","clip","fname"]].apply(tuple, axis=1)
        df = teacher_df[key.isin(sel_key)].reset_index(drop=True)
        log(f"PROMPT TEST: {len(df)} frame-subject pairs")
    elif TEST_MODE:
        df = df[df["show"].isin(TEST_SHOWS)]
        sampled = []
        for show in TEST_SHOWS:
            show_df = df[df["show"] == show]
            unique = show_df.drop_duplicates(
                subset=["show","clip","fname"]).head(TEST_MAX_FRAMES_PER_SHOW)
            key = show_df[["show","clip","fname"]].apply(tuple, axis=1)
            sel_key = unique[["show","clip","fname"]].apply(tuple, axis=1)
            sampled.append(show_df[key.isin(sel_key)])
        df = pd.concat(sampled).reset_index(drop=True)
        log(f"TEST MODE: {len(df)} frame-subject pairs")

    df = df.sort_values(["show","clip","subject","fname"]).reset_index(drop=True)
    df["frame_idx"] = df.groupby(["show","clip","subject"]).cumcount()
    return df


def get_other_subjects(df, show, clip, fname, this_subject):
    others = df[(df["show"] == show) & (df["clip"] == clip) &
                (df["fname"] == fname) & (df["subject"] != this_subject)]
    return [(int(r["head_x1"]), int(r["head_y1"]),
             int(r["head_x2"]), int(r["head_y2"]))
            for _, r in others.iterrows()]

# ══════════════════════════════════════════════════════════════════════
# ── INFERENCE ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def run_social_teacher(internvl_model, internvl_processor,
                       img_raw, primary_box, secondary_box, img_w, img_h):
    """InternVL3-alone for social frames. Identical to step40's run_teacher."""
    prompt = build_social_prompt(img_w, img_h, primary_box, secondary_box)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_raw},
        {"type": "text",  "text": prompt},
    ]}]
    inputs = internvl_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(internvl_model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = internvl_model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    response = internvl_processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True)[0]
    pred_x, pred_y = parse_gaze_xy(response)
    return pred_x, pred_y, response


def run_hybrid_object(internvl_model, internvl_processor,
                      gdino_model, gdino_processor,
                      img_raw, img_w, img_h):
    """
    Hybrid pipeline for object-gaze frames:
    Step 1: InternVL3 identifies what the subject is looking at (object name).
    Step 2: Grounding DINO localizes that object and returns a bounding box.
    Step 3: Box centroid → normalized (x, y) coordinate.

    Returns: (pred_x, pred_y, raw_response, object_phrase, gdino_score)
      - pred_x/pred_y: None if either step fails (caller falls back to GT)
      - raw_response: InternVL3's identification text (for audit trail)
      - object_phrase: what was actually sent to Grounding DINO
      - gdino_score: Grounding DINO's detection confidence (or None)
    """
    # Step 1: InternVL3 identification
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_raw},
        {"type": "text",  "text": IDENTIFY_PROMPT},
    ]}]
    inputs = internvl_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(internvl_model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        gen_ids = internvl_model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    raw_response = internvl_processor.decode(
        gen_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()

    object_phrase = extract_object_phrase(raw_response)

    # InternVL3 says off-screen on an object-gaze frame → treat as fallback
    if object_phrase is None:
        return None, None, raw_response, None, None

    # Step 2: Grounding DINO localization
    gdino_query = object_phrase.lower().strip().rstrip(".") + "."
    gdino_inputs = gdino_processor(
        images=img_raw, text=gdino_query, return_tensors="pt").to("cuda:0")

    with torch.inference_mode():
        gdino_out = gdino_model(**gdino_inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        gdino_out, gdino_inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[(img_h, img_w)],
    )[0]

    if len(results["boxes"]) == 0:
        # No box found → caller falls back to GT
        return None, None, raw_response, object_phrase, None

    # Step 3: highest-confidence box centroid → normalized coord
    best_idx = results["scores"].argmax().item()
    box = results["boxes"][best_idx].tolist()
    score = results["scores"][best_idx].item()
    cx = ((box[0] + box[2]) / 2) / img_w
    cy = ((box[1] + box[3]) / 2) / img_h

    return cx, cy, raw_response, object_phrase, score

# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    mode_str = ("PROMPT_TEST" if PROMPT_TEST else
                "TEST_MODE"   if TEST_MODE   else "FULL")
    log(f"Hybrid teacher pipeline starting [{mode_str}]")
    log(f"  InternVL3: {INTERNVL_NAME}")
    log(f"  Grounding DINO: {GDINO_NAME}")
    log(f"  Output: {LABELS_FILE}")

    df = load_manifest()
    df["pair_id"] = (df["show"] + "|" + df["clip"] + "|" +
                     df["fname"].astype(str) + "|" + df["subject"])

    # Resume support
    progress  = load_progress()
    completed = set(progress.get("completed", []))
    if completed:
        log(f"Resuming: {len(completed)} pairs already done")
    df = df[~df["pair_id"].isin(completed)].reset_index(drop=True)
    log(f"Remaining: {len(df)} pairs")

    # Routing: same as step40
    gt_rows      = df[df["use_teacher"] == False]
    teacher_rows = df[df["use_teacher"] == True]
    social_rows  = teacher_rows[teacher_rows["gaze_type"] == "social"]
    object_rows  = teacher_rows[teacher_rows["gaze_type"] == "object"]

    log(f"GT direct: {len(gt_rows)}")
    log(f"Social (InternVL3-alone): {len(social_rows)}")
    log(f"Object (Hybrid InternVL3+GDINO): {len(object_rows)}")

    # ── Load models ───────────────────────────────────────────────────
    log("Loading InternVL3-8B-hf...")
    internvl_processor = AutoProcessor.from_pretrained(
        INTERNVL_NAME, trust_remote_code=True)
    internvl_model = AutoModelForImageTextToText.from_pretrained(
        INTERNVL_NAME, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto")
    internvl_model.eval()
    log("InternVL3 loaded.")

    log("Loading Grounding DINO...")
    from transformers import AutoProcessor as GDinoAutoProcessor
    gdino_processor = GDinoAutoProcessor.from_pretrained(GDINO_NAME)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_NAME).to("cuda:0")
    gdino_model.eval()
    log("Grounding DINO loaded.")

    # ── GT direct labels (identical to step40) ────────────────────────
    log(f"Writing {len(gt_rows)} GT-direct labels...")
    for i, (_, row) in enumerate(gt_rows.iterrows()):
        gx, gy = int(row["gaze_x"]), int(row["gaze_y"])
        try:
            img_raw = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
        except Exception:
            img_w, img_h = 1, 1
        append_label({
            "show": row["show"], "clip": row["clip"],
            "fname": row["fname"], "subject": row["subject"],
            "img_path": row["img_path"], "frame_idx": int(row["frame_idx"]),
            "head_x1": int(row["head_x1"]), "head_y1": int(row["head_y1"]),
            "head_x2": int(row["head_x2"]), "head_y2": int(row["head_y2"]),
            "gaze_type": row["gaze_type"], "label_source": "gt",
            "teacher_version": None,
            "pred_x": gx / img_w, "pred_y": gy / img_h,
            "gt_x":   gx / img_w, "gt_y":   gy / img_h,
            "gt_px_x": gx, "gt_px_y": gy,
            "confidence": "GT", "raw_response": None,
        })
        completed.add(row["pair_id"])
        if (i + 1) % GT_SAVE_EVERY_N == 0:
            progress["completed"] = list(completed)
            save_progress(progress)
    progress["completed"] = list(completed)
    save_progress(progress)
    log(f"GT labels written: {len(gt_rows)}")

    # ── Social frames: InternVL3-alone (same as step40) ───────────────
    log(f"Starting social-frame inference on {len(social_rows)} frames...")
    n_soc_ok, n_soc_fail = 0, 0
    t_start = time.time()

    for i, (_, row) in enumerate(
        tqdm(social_rows.iterrows(), total=len(social_rows), desc="Social")
    ):
        pair_id = row["pair_id"]
        try:
            img_raw      = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
            if img_w == 0 or img_h == 0:
                raise ValueError("Zero-size image")
        except Exception as e:
            log(f"  Image load failed: {row['img_path']}: {e}")
            n_soc_fail += 1
            continue

        primary_box = (int(row["head_x1"]), int(row["head_y1"]),
                       int(row["head_x2"]), int(row["head_y2"]))
        other_boxes = get_other_subjects(
            df, row["show"], row["clip"], row["fname"], row["subject"])

        if not other_boxes:
            # Solo frame fallback (same as step40)
            gt_x_norm = row["gaze_x"] / img_w if row["gaze_x"] != -1 else None
            gt_y_norm = row["gaze_y"] / img_h if row["gaze_y"] != -1 else None
            append_label({
                "show": row["show"], "clip": row["clip"],
                "fname": row["fname"], "subject": row["subject"],
                "img_path": row["img_path"], "frame_idx": int(row["frame_idx"]),
                "head_x1": int(row["head_x1"]), "head_y1": int(row["head_y1"]),
                "head_x2": int(row["head_x2"]), "head_y2": int(row["head_y2"]),
                "gaze_type": row["gaze_type"], "label_source": "gt_fallback_solo",
                "teacher_version": None,
                "pred_x": gt_x_norm, "pred_y": gt_y_norm,
                "gt_x": gt_x_norm, "gt_y": gt_y_norm,
                "gt_px_x": int(row["gaze_x"]), "gt_px_y": int(row["gaze_y"]),
                "confidence": "GT", "raw_response": None,
            })
            completed.add(pair_id)
            continue

        secondary_box = other_boxes[0]
        pred_x, pred_y, response = None, None, ""
        for attempt in range(RETRY_ON_FAIL + 1):
            try:
                pred_x, pred_y, response = run_social_teacher(
                    internvl_model, internvl_processor,
                    img_raw, primary_box, secondary_box, img_w, img_h)
                if pred_x is not None:
                    break
            except Exception as e:
                log(f"  Social inference error attempt {attempt+1}: {e}")
                torch.cuda.empty_cache()

        is_offscreen = (pred_x == -1.0 and pred_y == -1.0)
        gt_x_norm = row["gaze_x"] / img_w if row["gaze_x"] != -1 else None
        gt_y_norm = row["gaze_y"] / img_h if row["gaze_y"] != -1 else None

        if is_offscreen:
            label_source, confidence = "teacher_offscreen", "PARSED"
            final_x, final_y = gt_x_norm, gt_y_norm
        elif pred_x is not None:
            label_source, confidence = "teacher", "PARSED"
            final_x, final_y = pred_x, pred_y
            n_soc_ok += 1
        else:
            label_source, confidence = "gt_fallback", "GT_FALLBACK"
            final_x, final_y = gt_x_norm, gt_y_norm
            n_soc_fail += 1

        append_label({
            "show": row["show"], "clip": row["clip"],
            "fname": row["fname"], "subject": row["subject"],
            "img_path": row["img_path"], "frame_idx": int(row["frame_idx"]),
            "head_x1": int(row["head_x1"]), "head_y1": int(row["head_y1"]),
            "head_x2": int(row["head_x2"]), "head_y2": int(row["head_y2"]),
            "gaze_type": row["gaze_type"], "label_source": label_source,
            "teacher_version": TEACHER_VERSION_SOCIAL,
            "pred_x": final_x, "pred_y": final_y,
            "gt_x": gt_x_norm, "gt_y": gt_y_norm,
            "gt_px_x": int(row["gaze_x"]), "gt_px_y": int(row["gaze_y"]),
            "confidence": confidence, "raw_response": response,
        })
        completed.add(pair_id)

        if (i + 1) % SAVE_EVERY_N == 0:
            progress["completed"] = list(completed)
            save_progress(progress)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_hrs = (len(social_rows) - (i + 1)) / rate / 3600
            tqdm.write(f"  Social [{i+1}/{len(social_rows)}] "
                       f"{1/rate:.1f}s/frame ETA:{eta_hrs:.1f}hrs "
                       f"OK:{n_soc_ok} Fail:{n_soc_fail}")

    progress["completed"] = list(completed)
    save_progress(progress)
    log(f"Social complete: {n_soc_ok} ok, {n_soc_fail} fallback")

    # ── Object frames: Hybrid InternVL3 + Grounding DINO ─────────────
    log(f"Starting hybrid inference on {len(object_rows)} object-gaze frames...")
    n_obj_ok, n_obj_fail = 0, 0
    # Track GDINO-specific failure reasons for the summary log
    n_internvl_offscreen, n_no_box = 0, 0
    t_start = time.time()

    for i, (_, row) in enumerate(
        tqdm(object_rows.iterrows(), total=len(object_rows), desc="Hybrid-Object")
    ):
        pair_id = row["pair_id"]
        try:
            img_raw      = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
            if img_w == 0 or img_h == 0:
                raise ValueError("Zero-size image")
        except Exception as e:
            log(f"  Image load failed: {row['img_path']}: {e}")
            n_obj_fail += 1
            continue

        gt_x_norm = row["gaze_x"] / img_w if row["gaze_x"] != -1 else None
        gt_y_norm = row["gaze_y"] / img_h if row["gaze_y"] != -1 else None

        pred_x, pred_y, response, object_phrase, gdino_score = \
            None, None, "", None, None

        for attempt in range(RETRY_ON_FAIL + 1):
            try:
                pred_x, pred_y, response, object_phrase, gdino_score = \
                    run_hybrid_object(
                        internvl_model, internvl_processor,
                        gdino_model, gdino_processor,
                        img_raw, img_w, img_h)
                if pred_x is not None:
                    break
                # Log specific failure reason on first attempt
                if attempt == 0:
                    if object_phrase is None:
                        n_internvl_offscreen += 1
                    else:
                        n_no_box += 1
            except Exception as e:
                log(f"  Hybrid error attempt {attempt+1}: {e}")
                torch.cuda.empty_cache()

        if pred_x is not None:
            label_source = "teacher_hybrid"
            confidence   = "PARSED"
            final_x, final_y = pred_x, pred_y
            n_obj_ok += 1
        else:
            label_source = "gt_fallback"
            confidence   = "GT_FALLBACK"
            final_x, final_y = gt_x_norm, gt_y_norm
            n_obj_fail += 1

        # Store object_phrase and gdino_score in raw_response field for audit
        audit = {
            "internvl_response": response,
            "object_phrase": object_phrase,
            "gdino_score": gdino_score,
        }

        append_label({
            "show": row["show"], "clip": row["clip"],
            "fname": row["fname"], "subject": row["subject"],
            "img_path": row["img_path"], "frame_idx": int(row["frame_idx"]),
            "head_x1": int(row["head_x1"]), "head_y1": int(row["head_y1"]),
            "head_x2": int(row["head_x2"]), "head_y2": int(row["head_y2"]),
            "gaze_type": row["gaze_type"], "label_source": label_source,
            "teacher_version": TEACHER_VERSION_HYBRID,
            "pred_x": final_x, "pred_y": final_y,
            "gt_x": gt_x_norm, "gt_y": gt_y_norm,
            "gt_px_x": int(row["gaze_x"]), "gt_px_y": int(row["gaze_y"]),
            "confidence": confidence,
            "raw_response": json.dumps(audit),
        })
        completed.add(pair_id)

        if (i + 1) % SAVE_EVERY_N == 0:
            progress["completed"] = list(completed)
            save_progress(progress)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_hrs = (len(object_rows) - (i + 1)) / rate / 3600
            tqdm.write(f"  Hybrid [{i+1}/{len(object_rows)}] "
                       f"{1/rate:.1f}s/frame ETA:{eta_hrs:.1f}hrs "
                       f"OK:{n_obj_ok} Fallback:{n_obj_fail}")

    progress["completed"] = list(completed)
    save_progress(progress)

    elapsed = time.time() - t_start
    log("\n" + "="*60)
    log(f"Pipeline complete [{mode_str}]")
    log("="*60)
    log(f"  GT direct labels:            {len(gt_rows)}")
    log(f"  Social (InternVL3-alone):    {n_soc_ok} ok, {n_soc_fail} fallback")
    log(f"  Object hybrid (success):     {n_obj_ok}")
    log(f"  Object hybrid (fallback GT): {n_obj_fail}")
    log(f"    of which InternVL3 said off-screen: {n_internvl_offscreen}")
    log(f"    of which GDINO found no box:        {n_no_box}")
    log(f"  Labels file: {LABELS_FILE}")
    log(f"  Elapsed: {elapsed/3600:.2f} hrs")

    obj_parse_rate = n_obj_ok / max(len(object_rows), 1) * 100
    log(f"  Object-frame hybrid parse rate: {obj_parse_rate:.1f}%")
    if obj_parse_rate < 60:
        log("  WARNING: low parse rate on object frames -- review BOX_THRESHOLD")


if __name__ == "__main__":
    main()
