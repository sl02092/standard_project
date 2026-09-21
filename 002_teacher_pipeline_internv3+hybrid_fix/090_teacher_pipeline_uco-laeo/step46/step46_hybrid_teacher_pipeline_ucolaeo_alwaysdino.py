"""
step46_hybrid_teacher_pipeline.py — Hybrid Teacher Pipeline (targeted-phrase prompt)
InternVL3-8B-hf (semantic identification, TARGETED-PHRASE prompt) + Grounding
DINO (text-conditioned spatial localization), for object-gaze coordinate
prediction.

THIS FILE (v3 — 2026-07-19): three more fixes, prompted by a second external
code review, on top of v2's resume-desync and off-screen-substitution fixes:
  - FIX: GT-loop image-load fallback was `img_w, img_h = 1, 1` -- worse than
    step40's old 1280x720 guess, since this writes obviously-broken
    "normalized" coordinates (a real pixel value like 450 becomes 450.0).
    Now skips and logs, same approach as step40 v5 / v2's own GT-loop fix.
  - FIX: GDINO was hardcoded to "cuda:0" while InternVL3 uses
    device_map="auto" -- now derives device dynamically, consistent with
    InternVL3. Likely harmless on Eureka2 specifically but removes the
    assumption.
  - FIX: get_other_subjects() now uses a pre-built groupby lookup instead
    of re-scanning the full dataframe per call (same fix as step40 v5).
  All three verified as zero actual occurrences / no measured cost impact
  in the completed production run -- none of this changes any existing
  result from labels_hybrid_targeted.jsonl.
  NOT applied (claim didn't survive checking against real data): stripping
  laterality/positional words before querying GDINO -- this is exactly
  step47's own approach, which regressed vs this script at production scale
  (object ADE 0.2815 vs 0.2767). Left as-is on that evidence.

v2 (2026-07-19): the definitive/canonical version of the frozen
production InternVL3+GDINO teacher (v1, labels_hybrid_targeted.jsonl). Two
fixes applied, prompted by an external code review, both ported from the
identical fixes already applied to step40 v5 and step47 v2:
  - FIX: resume desync -- `completed` is now reconstructed from the JSONL
    directly (recover_progress_from_jsonl(), already present but unused)
    instead of trusting progress.json's separately-checkpointed list.
  - FIX: off-screen substitution in the social-frame path -- now records
    the teacher's actual -1.0,-1.0 claim instead of silently substituting
    GT when they disagree.
  Verified against the completed production run: 0/23,096 records were
  affected by either bug, so NEITHER FIX CHANGES ANY EXISTING RESULT --
  every headline number produced from labels_hybrid_targeted.jsonl (the
  three-way comparison, the confidence-threshold sweep, etc.) is unaffected
  and does not need rerunning because of this. These fixes only matter for
  any future rerun of this pipeline (e.g. on ChildPlay/VACATION).
  Object-hybrid path (run_hybrid_object) was checked and does NOT have the
  off-screen bug -- it already labels GT-substituted failures honestly as
  "gt_fallback"/"GT_FALLBACK", never as a mislabeled "PARSED" success.

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
  "internvl3-8b-v5-noleak"            for social frames (InternVL3-alone, unchanged)
  "internvl3+gdino-base-v3-alwaysdino"  for object frames (targeted-phrase prompt)

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

PROMPT_TEST = os.environ.get("TEACHER_PROMPT_TEST", "0") == "1"
TEST_MODE   = os.environ.get("TEACHER_TEST_MODE", "0") == "1"

BASE_PATH     = os.environ.get("VAT_BASE_PATH",
    "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget")
MANIFEST_PATH = os.environ.get(
    "TEACHER_MANIFEST_PATH",
    "/parallel_scratch/sl02092/standard_project/frame_manifest_ucolaeo.csv",
)
INTERNVL_NAME = "OpenGVLab/InternVL3-8B-hf"
GDINO_NAME    = "IDEA-Research/grounding-dino-base"

# UNCHANGED: same model/prompt as VAT/ChildPlay. Object route is expected
# to see near-zero (possibly literally zero) traffic here -- UCO-LAEO's
# schema has no gaze-at-object concept, every valid target comes from a
# resolved mutual LAEO pair (gaze_type='social' on every manifest row).
TEACHER_VERSION_SOCIAL = "internvl3-8b-v5-noleak"
TEACHER_VERSION_HYBRID = "internvl3+gdino-base-v3-alwaysdino"

_OUT_PREFIX = os.environ.get("TEACHER_OUTPUT_PREFIX", "ucolaeo_hybrid_targeted")
if PROMPT_TEST:
    LABELS_FILE   = f"labels_{_OUT_PREFIX}_prompt_test.jsonl"
    PROGRESS_FILE = f"progress_{_OUT_PREFIX}_prompt_test.json"
    LOG_FILE      = f"pipeline_{_OUT_PREFIX}_prompt_test.log"
elif TEST_MODE:
    LABELS_FILE   = f"labels_{_OUT_PREFIX}_test.jsonl"
    PROGRESS_FILE = f"progress_{_OUT_PREFIX}_test.json"
    LOG_FILE      = f"pipeline_{_OUT_PREFIX}_test.log"
else:
    LABELS_FILE   = f"labels_{_OUT_PREFIX}.jsonl"
    PROGRESS_FILE = f"progress_{_OUT_PREFIX}.json"
    LOG_FILE      = f"pipeline_{_OUT_PREFIX}.log"

# ── RERUN GUARD (patch_teacher_noleak.py) ────────────────────────────
# Every output filename is tagged so this run can NEITHER resume from NOR
# overwrite the frozen production labels. Without this, recover_progress_
# from_jsonl() reads the OLD labels file, concludes every pair is already
# done, and the job exits having written nothing -- silently.
_RERUN_TAG    = os.environ.get("TEACHER_RERUN_TAG", "noleak")
LABELS_FILE   = LABELS_FILE.replace(".jsonl", f"_{_RERUN_TAG}.jsonl")
PROGRESS_FILE = PROGRESS_FILE.replace(".json",  f"_{_RERUN_TAG}.json")
LOG_FILE      = LOG_FILE.replace(".log",        f"_{_RERUN_TAG}.log")


# Same probe set as step40 -- one video per content category, all
# well-populated, none from 'negatives*'.
PROMPT_TEST_N     = 10
_default_test_shows = "got01,mr08,sv03,twd03"
PROMPT_TEST_SHOWS = os.environ.get("TEACHER_PROMPT_TEST_SHOWS", _default_test_shows).split(",")
TEST_MAX_FRAMES_PER_SHOW = 17
TEST_SHOWS = os.environ.get("TEACHER_TEST_SHOWS", _default_test_shows).split(",")

# Inference settings
MAX_NEW_TOKENS  = 60   # shorter than step40 since hybrid only needs object name
SAVE_EVERY_N    = 10
GT_SAVE_EVERY_N = 100
RETRY_ON_FAIL   = 2
# ALWAYS-DINO (patch_teacher_noleak.py). Driving both thresholds to 0 makes
# post_process_grounded_object_detection return its full candidate set, so
# the argmax below always yields a box and the "no box found" GT fallback
# effectively never fires. That is what makes this condition a SINGLE
# generating process -- Section 7.4 item 1's "uniform, but noisier".
# Set HYBRID_ALWAYS_DINO=0 to restore the frozen 0.30/0.25 behaviour.
ALWAYS_DINO     = os.environ.get("HYBRID_ALWAYS_DINO", "1") == "1"
BOX_THRESHOLD   = 0.0 if ALWAYS_DINO else 0.30
TEXT_THRESHOLD  = 0.0 if ALWAYS_DINO else 0.25

# ══════════════════════════════════════════════════════════════════════
# ── PROMPTS ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_social_prompt(img_w, img_h, primary_box, secondary_box):
    """Unchanged from step40 — social frames use InternVL3-alone."""
    ax1, ay1, ax2, ay2 = primary_box

    # NO-LEAK v5 (2026-09-18). The frozen v4 prompt supplied PERSON B's head
    # box AND face centre and told the model to answer with that centre when
    # A looked at B. Section 3.2.3 measured the consequence: the supplied
    # coordinate came back unchanged on 88.1% of social frames and ~84.9% of
    # object frames, where it is wrong -- so the condition was measuring the
    # model's willingness to copy, not its gaze competence.
    #
    # This version withholds the coordinate AND the box it could be trivially
    # derived from, which is Section 7.4 item 2's specified experiment. The
    # LABELLING CONVENTION is deliberately kept -- a social target is still
    # the other person's face centre -- so the output stays comparable with
    # ground truth; only the answer is withheld.
    #
    # secondary_box is still accepted so that every call site is unchanged,
    # but it is deliberately unused. Do not reintroduce it.
    na = (round(ax1/img_w,3), round(ay1/img_h,3),
          round(ax2/img_w,3), round(ay2/img_h,3))

    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.

PERSON A (predict gaze): head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)

Rules:
- Work out where Person A is looking from their head pose and eye direction, then locate that point in the image yourself.
- If A is looking at another person, answer with the centre of THAT person's face.
- If A is looking at an object, answer with the centre of that object.
- If A is looking outside the frame, answer (-1,-1).
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
    df = pd.read_csv(MANIFEST_PATH, dtype={"subject": str})
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


def build_frame_lookup(df):
    """
    v3 FIX (2026-07-19, external-review-prompted, same fix as step40 v5):
    pre-built once, before the loop, instead of get_other_subjects()
    re-scanning the full dataframe with a boolean mask on every call (a
    genuine O(N) scan x N calls = O(N^2) over the whole run). Real observed
    throughput in the completed run (~2.4s/frame at the dual-model object
    stage) doesn't suggest this was the dominant cost -- VLM+DINO inference
    almost certainly dominates -- so this is correctness/hygiene, not a fix
    for an observed slowdown.
    """
    return df.groupby(["show", "clip", "fname"])


def get_other_subjects(frame_lookup, show, clip, fname, this_subject):
    try:
        group = frame_lookup.get_group((show, clip, fname))
    except KeyError:
        return []
    others = group[group["subject"] != this_subject]
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
        images=img_raw, text=gdino_query, return_tensors="pt").to(
        next(gdino_model.parameters()).device)  # v3 FIX: was hardcoded
        # "cuda:0" -- derives the real device from the model instead, so
        # this stays correct wherever gdino_model was actually loaded.

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
    # v2 FIX (2026-07-19, external-review-prompted, ported from step47's
    # identical fix): previously trusted progress.json's "completed" list
    # directly. progress.json is only checkpointed every SAVE_EVERY_N=10
    # frames, while append_label() writes every frame immediately -- a
    # crash inside that window leaves up to 9 frames present in the JSONL
    # but not yet reflected in progress.json, so resuming from progress.json
    # alone would re-run and DUPLICATE those rows in labels_hybrid_targeted.jsonl
    # -- the frozen production teacher file. recover_progress_from_jsonl()
    # already existed in this file (ported from step40) but was never wired
    # into the default resume path -- it now is. Didn't affect this script's
    # actual completed production run (finished cleanly, zero crashes,
    # verified against its slurm log) -- this protects any future rerun.
    completed = recover_progress_from_jsonl()
    progress  = load_progress()
    if completed:
        log(f"Resuming: {len(completed)} pairs already done (reconstructed from {LABELS_FILE})")
    # RESUME FIX (patch_teacher_noleak.py): keep the FULL manifest for the
    # frame lookup. Filtering it before build_frame_lookup() makes any frame
    # whose co-subject finished before the kill look solo on resume, which
    # sends a teacher row down the gt_fallback_solo path and writes ground
    # truth as its label. step40 already does it this way.
    df_all = df
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
    # v3 FIX (2026-07-19, external-review-prompted): previously hardcoded
    # "cuda:0" while InternVL3 above uses device_map="auto" -- inconsistent,
    # and would misbehave on any node where the allocated GPU isn't index 0
    # (e.g. running locally alongside other GPU work, or a shared/multi-job
    # node). Likely harmless on Eureka2 specifically (Slurm typically remaps
    # the allocated GPU to appear as device 0), but this makes both models
    # consistent and removes the assumption.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_NAME).to(device)
    gdino_model.eval()
    log(f"Grounding DINO loaded on {device}.")

    # ── GT direct labels (identical to step40) ────────────────────────
    log(f"Writing {len(gt_rows)} GT-direct labels...")
    n_gt_skipped = 0
    for i, (_, row) in enumerate(gt_rows.iterrows()):
        # v3 FIX (2026-07-19, external-review-prompted): previously defaulted
        # to img_w, img_h = 1, 1 on any load failure and kept going -- this
        # writes obviously-broken "normalized" coordinates (e.g. a real pixel
        # value like 450 gets written as 450.0, wildly outside the expected
        # 0-1 range) rather than a plausible-looking wrong value. Verified
        # zero image-load failures occurred in the completed production run
        # (slurm log has none), so this changes nothing about existing
        # results -- it only protects future runs. Skip and log instead of
        # guessing, same approach as step40 v5's identical fix.
        gx, gy = int(row["gaze_x"]), int(row["gaze_y"])
        try:
            img_raw = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
            if img_w == 0 or img_h == 0:
                raise ValueError(f"Zero-size image: {row['img_path']}")
        except Exception as e:
            log(f"  GT image load failed, skipping: {row['img_path']}: {e}")
            n_gt_skipped += 1
            continue
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
    log(f"GT labels written: {len(gt_rows) - n_gt_skipped} (skipped {n_gt_skipped} due to image load failure)")

    # ── Social frames: InternVL3-alone (same as step40) ───────────────
    log(f"Starting social-frame inference on {len(social_rows)} frames...")
    n_soc_ok, n_soc_fail = 0, 0
    frame_lookup = build_frame_lookup(df_all)  # RESUME FIX: full manifest,
    # not the post-resume remainder -- see the note at the resume filter.
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
            frame_lookup, row["show"], row["clip"], row["fname"], row["subject"])

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
            # v2 FIX (2026-07-19, external-review-prompted, ported from
            # step40 v5): previously wrote gt_x_norm/gt_y_norm here,
            # silently substituting GT's real coordinate whenever the
            # teacher claimed off-screen -- masking exactly the cases
            # where the teacher is wrong (GT says on-screen) and leaking
            # GT into the "teacher" condition's own target, in the frozen
            # production teacher file. Now records what the teacher
            # actually claimed (-1.0, -1.0) regardless of GT agreement;
            # label_source stays "teacher_offscreen" so it's traceable.
            # Verified: 0/23,096 records in the completed production run
            # (labels_hybrid_targeted.jsonl) were affected -- this changes
            # nothing about any existing result, it only protects future
            # runs (ChildPlay/VACATION may hallucinate off-screen at a
            # different rate than VAT did).
            label_source, confidence = "teacher_offscreen", "PARSED"
            final_x, final_y = -1.0, -1.0
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
    n_obj_offscreen_kept = 0   # ALWAYS-DINO: teacher off-screen claims kept, not GT-substituted
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
        elif ALWAYS_DINO and object_phrase is None:
            # InternVL3 claimed off-screen on a frame ground truth calls
            # object-directed. The frozen pipeline substituted GT here.
            # Doing that reintroduces a second generating process into the
            # one condition whose whole purpose is to have only one, so
            # record the teacher's actual claim instead -- the same
            # reasoning as step40 v5's off-screen fix. Counted separately
            # and reported at the end; if this count is large the
            # uniformity claim is weaker than it looks.
            label_source = "teacher_offscreen"
            confidence   = "PARSED"
            final_x, final_y = -1.0, -1.0
            n_obj_offscreen_kept += 1
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
    log(f"  Object teacher off-screen kept (NOT GT): {n_obj_offscreen_kept}")
    log(f"  ALWAYS_DINO={ALWAYS_DINO}  BOX_THRESHOLD={BOX_THRESHOLD}  TEXT_THRESHOLD={TEXT_THRESHOLD}")
    if n_obj_fail:
        log(f"  WARNING: {n_obj_fail} object rows still took GT -- this condition is not a single generating process. Check them before claiming uniformity.")
    log(f"  Labels file: {LABELS_FILE}")
    log(f"  Elapsed: {elapsed/3600:.2f} hrs")

    obj_parse_rate = n_obj_ok / max(len(object_rows), 1) * 100
    log(f"  Object-frame hybrid parse rate: {obj_parse_rate:.1f}%")
    if obj_parse_rate < 60:
        log("  WARNING: low parse rate on object frames -- review BOX_THRESHOLD")


if __name__ == "__main__":
    main()
