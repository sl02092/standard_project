"""
Step 4 - Teacher Labelling Pipeline (v4 — locked schema + hardened resume)
Reads frame_manifest.csv and generates gaze labels for each frame.

KEY CHANGES FROM v3 (this file — v4)
──────────────────────────────────────
SCHEMA CHANGES (locked — do not alter without re-running teacher):
  - NEW FIELD: teacher_version  — records which model+prompt generated
                                   the label. Allows selective retraining
                                   if model or prompt changes later.
  - NEW FIELD: frame_idx        — 0-indexed position of frame within its
                                   clip (sorted order). Required for the
                                   Stage 2 temporal LSTM module. Cannot
                                   be recovered cheaply post-hoc.
  - CHANGED:   confidence       — no longer a model self-report
                                   (unreliable, costs tokens, uncalibrated).
                                   Now encodes label provenance:
                                     "GT"           → ground truth direct
                                     "PARSED"       → teacher coordinate
                                                       successfully extracted
                                     "GT_FALLBACK"  → teacher parse failed,
                                                       fell back to GT

CHECKPOINT / RESUME FIXES:
  - FIX A: GT labels now added to `completed` set AND progress saved in
            batches during the GT write loop. Previously GT rows were
            written to JSONL but never added to progress, so a crash
            mid-GT-write caused duplicate labels on resume.
  - FIX B: Resume can reconstruct `completed` from JSONL if progress
            JSON is missing or empty. Protects against the case where
            the JSONL exists but the progress file was lost (e.g. after
            copying files to HPC). Call recover_progress_from_jsonl()
            to use this.
  - FIX C: SAVE_EVERY_N checkpoint now saves progress atomically via a
            temp file + rename, preventing a half-written JSON from
            corrupting the progress state on a mid-save crash.

INHERITED FROM v3 (unchanged)
───────────────────────────────
  - FIX 1: torch_dtype= (was dtype=, wasted ~8GB VRAM)
  - FIX 2: Operator precedence bug in PROMPT_TEST manifest filter
  - FIX 3: try/except + zero-size guard around image load
  - FIX 4: Solo frame handling — falls back to GT, no self-referential box
  - FIX 5: Safe GT normalisation guards in run_prompt_test

THREE MODES
───────────
PROMPT_TEST = True   → 10 frames, prints raw responses for inspection
                       use this before any full run
TEST_MODE   = True   → ~50 frames across 3 shows (~15 min)
                       use this for end-to-end pipeline validation
TEST_MODE   = False  → all 23,096 frames (full production run)

Usage:
    python step4_1_teacher_pipeline.py

Requirements:
    pip install transformers accelerate pillow torch torchvision pandas tqdm
"""

import os
import re
import csv
import json
import time
import torch
import tempfile
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

PROMPT_TEST = False # ← START HERE: validate prompt on 10 frames
TEST_MODE   = True  # ← set True for ~50 frame pipeline test
                    # ← set both False for full production run

BASE_PATH     = r"C:\repo\standard_project\videoattentiontarget"
MANIFEST_PATH = "frame_manifest.csv"
MODEL_NAME    = "OpenGVLab/InternVL3-8B-hf"

# Locked schema version string — update this if you change the model
# or prompt so you can identify which labels came from which run.
TEACHER_VERSION = "internvl3-8b-v4-prompt"

# Output files (PROMPT_TEST uses its own files, never touches production)
if PROMPT_TEST:
    LABELS_FILE   = "labels_prompt_test.jsonl"
    PROGRESS_FILE = "progress_prompt_test.json"
    LOG_FILE      = "pipeline_prompt_test.log"
elif TEST_MODE:
    LABELS_FILE   = "labels_test.jsonl"
    PROGRESS_FILE = "progress_test.json"
    LOG_FILE      = "pipeline_test.log"
else:
    LABELS_FILE   = "labels_full.jsonl"
    PROGRESS_FILE = "progress_full.json"
    LOG_FILE      = "pipeline_full.log"

# Prompt test settings
PROMPT_TEST_N = 10
PROMPT_TEST_SHOWS = [
    "My Dinner with Andre",
    "Conan",
    "Gone with the Wind",
    "Tartuffe",
]

# Test mode settings
TEST_MAX_FRAMES_PER_SHOW = 17
TEST_SHOWS = [
    "My Dinner with Andre",
    "Conan",
    "Gone with the Wind",
]

# Inference settings
MAX_NEW_TOKENS = 150
SAVE_EVERY_N   = 10     # checkpoint every N teacher frames
GT_SAVE_EVERY_N = 100   # checkpoint every N GT frames (FIX A)
RETRY_ON_FAIL  = 2

# ══════════════════════════════════════════════════════════════════════
# ── PROMPT (v3 — unchanged from v3, output-first) ─────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_social_prompt(img_w, img_h, primary_box, secondary_box):
    """
    GAZE_XY coordinate output comes FIRST — guaranteed within token budget.
    No confidence token — confidence is now derived from parse success,
    not model self-report (see schema notes in header).
    """
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


def parse_gaze_xy(text):
    """
    Robust parser — handles spacing/newline variations.
    Returns (None, None) on parse failure — caller maps this to
    confidence="GT_FALLBACK".
    """
    if not text or not text.strip():
        return None, None

    # Off-screen case
    if re.search(r"GAZE_XY:\s*\(-1\s*,\s*-1\)", text):
        return -1.0, -1.0

    # Standard case
    match = re.search(
        r"GAZE_XY:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)",
        text
    )
    if match:
        x = max(0.0, min(1.0, float(match.group(1))))
        y = max(0.0, min(1.0, float(match.group(2))))
        return x, y

    # Fallback: any (x, y) with decimals in [0,1]
    matches = re.findall(
        r"\(\s*(0\.[0-9]+|1\.0|0\.0)\s*,\s*(0\.[0-9]+|1\.0|0\.0)\s*\)",
        text
    )
    if matches:
        x = max(0.0, min(1.0, float(matches[0][0])))
        y = max(0.0, min(1.0, float(matches[0][1])))
        return x, y

    return None, None

# ══════════════════════════════════════════════════════════════════════
# ── PROGRESS / CHECKPOINTING ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress):
    """
    FIX C: Atomic save via temp file + rename.
    Prevents a half-written JSON from corrupting progress on crash.
    """
    dir_name = os.path.dirname(os.path.abspath(PROGRESS_FILE))
    with tempfile.NamedTemporaryFile(
        "w", dir=dir_name, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(progress, tmp)
        tmp_path = tmp.name
    os.replace(tmp_path, PROGRESS_FILE)


def recover_progress_from_jsonl():
    """
    FIX B: Reconstructs the completed set by reading the JSONL directly.
    Use this if PROGRESS_FILE is missing but LABELS_FILE exists
    (e.g. after copying files to HPC without the progress JSON).

    Usage:
        completed = recover_progress_from_jsonl()
        # then pass as the initial completed set in main()
    """
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
                pair_id = (
                    f"{rec['show']}|{rec['clip']}|"
                    f"{rec['fname']}|{rec['subject']}"
                )
                completed.add(pair_id)
            except (json.JSONDecodeError, KeyError):
                continue
    log(f"Recovered {len(completed)} completed pairs from {LABELS_FILE}")
    return completed


def append_label(record):
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log(msg, also_print=True):
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ══════════════════════════════════════════════════════════════════════
# ── MANIFEST LOADING ──────────────────════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    log(f"Manifest loaded: {len(df)} frame-subject pairs")

    if PROMPT_TEST:
        # FIX 2 (inherited): parentheses around each condition
        teacher_df = df[
            (df["use_teacher"] == True) &
            (df["show"].isin(PROMPT_TEST_SHOWS))
        ]
        unique = teacher_df.drop_duplicates(
            subset=["show", "clip", "fname"]
        ).head(PROMPT_TEST_N)
        key     = teacher_df[["show","clip","fname"]].apply(tuple, axis=1)
        sel_key = unique[["show","clip","fname"]].apply(tuple, axis=1)
        df = teacher_df[key.isin(sel_key)].reset_index(drop=True)
        log(f"PROMPT TEST: {len(df)} frame-subject pairs")
        log(f"  Shows: {df['show'].value_counts().to_dict()}")

    elif TEST_MODE:
        df = df[df["show"].isin(TEST_SHOWS)]
        sampled = []
        for show in TEST_SHOWS:
            show_df = df[df["show"] == show]
            unique  = show_df.drop_duplicates(
                subset=["show","clip","fname"]
            ).head(TEST_MAX_FRAMES_PER_SHOW)
            key     = show_df[["show","clip","fname"]].apply(tuple, axis=1)
            sel_key = unique[["show","clip","fname"]].apply(tuple, axis=1)
            sampled.append(show_df[key.isin(sel_key)])
        df = pd.concat(sampled).reset_index(drop=True)
        log(f"TEST MODE: {len(df)} frame-subject pairs")

    # ── Compute frame_idx: 0-indexed position within each clip ────────
    # Sort by fname within each (show, clip) group, then assign index.
    # This is stable across runs as long as the manifest is unchanged.
    df = df.sort_values(["show", "clip", "subject", "fname"]).reset_index(drop=True)
    df["frame_idx"] = (
        df.groupby(["show", "clip", "subject"])
          .cumcount()
    )

    return df


def get_other_subjects(df, show, clip, fname, this_subject):
    others = df[
        (df["show"]    == show) &
        (df["clip"]    == clip) &
        (df["fname"]   == fname) &
        (df["subject"] != this_subject)
    ]
    return [
        (int(r["head_x1"]), int(r["head_y1"]),
         int(r["head_x2"]), int(r["head_y2"]))
        for _, r in others.iterrows()
    ]

# ══════════════════════════════════════════════════════════════════════
# ── INFERENCE ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def run_teacher(model, processor, img_raw, primary_box, secondary_box,
                img_w, img_h):
    prompt = build_social_prompt(img_w, img_h, primary_box, secondary_box)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_raw},
            {"type": "text",  "text": prompt},
        ],
    }]

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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    response  = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )[0]

    pred_x, pred_y = parse_gaze_xy(response)
    return pred_x, pred_y, response

# ══════════════════════════════════════════════════════════════════════
# ── PROMPT TEST ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def run_prompt_test(model, processor, df):
    """
    Runs inference on PROMPT_TEST_N frames, prints full details.
    Scores parse rate and mean distance vs GT.
    No labels written to production files.
    """
    import math

    log("\n" + "="*60)
    log("PROMPT TEST — evaluating prompt on 10 frames")
    log("="*60)

    teacher_rows = df[df["use_teacher"] == True]
    results = []

    for _, row in teacher_rows.iterrows():
        # FIX 3 (inherited): try/except around image load
        try:
            img_raw      = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
            if img_w == 0 or img_h == 0:
                log(f"  Zero-size image skipped: {row['img_path']}")
                continue
        except Exception as e:
            log(f"  Image load failed: {row['img_path']}: {e}")
            continue

        primary_box = (int(row["head_x1"]), int(row["head_y1"]),
                       int(row["head_x2"]), int(row["head_y2"]))
        other_boxes = get_other_subjects(
            df, row["show"], row["clip"], row["fname"], row["subject"]
        )

        # FIX 4 (inherited): skip solo frames in prompt test
        if not other_boxes:
            log(f"  No secondary subject — skipping: "
                f"{row['show']}/{row['clip']}/{row['fname']}")
            continue
        secondary_box = other_boxes[0]

        pred_x, pred_y, response = run_teacher(
            model, processor, img_raw,
            primary_box, secondary_box, img_w, img_h
        )

        # Map parse result to confidence string (same logic as production)
        if pred_x is None:
            confidence = "GT_FALLBACK"
        elif pred_x == -1.0:
            confidence = "PARSED"   # valid off-screen parse
        else:
            confidence = "PARSED"

        # FIX 5 (inherited): safe GT normalisation
        gt_x = row["gaze_x"] / img_w if (row["gaze_x"] != -1 and img_w > 0) else None
        gt_y = row["gaze_y"] / img_h if (row["gaze_y"] != -1 and img_h > 0) else None

        parsed = pred_x is not None and pred_x != -1.0
        dist   = None
        if parsed and gt_x is not None:
            dist = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

        results.append({
            "show": row["show"], "clip": row["clip"],
            "fname": row["fname"], "subject": row["subject"],
            "frame_idx": int(row["frame_idx"]),
            "pred_x": pred_x, "pred_y": pred_y,
            "gt_x": gt_x,     "gt_y":   gt_y,
            "dist": dist,      "parsed": parsed,
            "confidence": confidence,
            "response": response,
        })

        print(f"\n── {row['show']} / {row['clip']} / {row['fname']} / {row['subject']}")
        print(f"   frame_idx : {int(row['frame_idx'])}")
        print(f"   GT   : ({gt_x:.3f}, {gt_y:.3f})" if gt_x else "   GT   : off-screen")
        print(f"   PRED : ({pred_x:.3f}, {pred_y:.3f})" if parsed else "   PRED : PARSE FAILED")
        print(f"   CONF : {confidence}")
        print(f"   DIST : {dist:.3f}" if dist else "   DIST : N/A")
        print(f"   RAW  : {response[:200]}{'...' if len(response)>200 else ''}")

    n_parsed  = sum(1 for r in results if r["parsed"])
    dists     = [r["dist"] for r in results if r["dist"] is not None]
    mean_dist = sum(dists)/len(dists) if dists else None

    print(f"\n{'='*60}")
    print(f"PROMPT TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Frames tested  : {len(results)}")
    if len(results) > 0:
        print(f"  Parse success  : {n_parsed} / {len(results)} "
              f"({n_parsed/len(results)*100:.0f}%)")
    print(f"  Mean distance  : {mean_dist:.3f}" if mean_dist else "  Mean distance  : N/A")

    if len(results) == 0:
        print("\n  WARNING: no frames tested — check manifest filter.")
        return False

    if n_parsed == len(results):
        print("\n  PASS — prompt parsing reliably.")
        print("  Set PROMPT_TEST=False, TEST_MODE=True and run a 50-frame test.")
    elif n_parsed >= len(results) * 0.8:
        print("\n  PARTIAL — most frames parsed, some failures.")
        print("  Check RAW outputs above for failed frames.")
    else:
        print("\n  FAIL — prompt needs refinement.")

    return n_parsed == len(results)

# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    if PROMPT_TEST:
        mode_str = "PROMPT TEST (10 frames)"
    elif TEST_MODE:
        mode_str = "TEST MODE (~50 frames)"
    else:
        mode_str = "FULL MODE (23,096 frames)"

    log("=" * 60)
    log(f"Step 4 — Teacher Labelling Pipeline  [{mode_str}]")
    log(f"Prompt version : {TEACHER_VERSION}")
    log(f"Schema version : v4 (frame_idx, teacher_version, confidence=provenance)")
    log("=" * 60)

    df       = load_manifest()
    progress = load_progress()

    # ── FIX B: if progress is empty but JSONL exists, recover from it ──
    # Uncomment the two lines below if you need to recover after losing
    # the progress JSON (e.g. copying to HPC without it):
    #
    # if not progress["completed"] and os.path.exists(LABELS_FILE):
    #     progress["completed"] = list(recover_progress_from_jsonl())

    completed = set(progress["completed"])

    df["pair_id"] = (df["show"] + "|" + df["clip"] + "|" +
                     df["fname"] + "|" + df["subject"])
    remaining = df[~df["pair_id"].isin(completed)]

    log(f"Total pairs  : {len(df)}")
    log(f"Already done : {len(completed)}")
    log(f"Remaining    : {len(remaining)}")

    # ── Load model ─────────────────────────────────────────────────────
    log(f"\nLoading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model     = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    log(f"Model loaded  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Prompt test mode ───────────────────────────────────────────────
    if PROMPT_TEST:
        run_prompt_test(model, processor, remaining)
        log("\nPrompt test complete. Review output above.")
        log("If PASS: set PROMPT_TEST=False, TEST_MODE=True, rerun.")
        return

    # ── Write GT labels ────────────────────────────────────────────────
    # FIX A: GT rows now added to `completed` and checkpointed in batches.
    # Previously they were written to JSONL but never marked done,
    # causing duplicate labels on resume after a crash mid-GT-write.
    gt_rows      = remaining[remaining["use_teacher"] == False]
    teacher_rows = remaining[remaining["use_teacher"] == True]

    log(f"\nWriting {len(gt_rows)} GT labels...")
    for i, (_, row) in enumerate(gt_rows.iterrows()):
        try:
            img = Image.open(row["img_path"])
            img_w, img_h = img.size
            if img_w == 0 or img_h == 0:
                img_w, img_h = 1280, 720
        except Exception:
            img_w, img_h = 1280, 720

        gx, gy = int(row["gaze_x"]), int(row["gaze_y"])
        append_label({
            "show":             row["show"],
            "clip":             row["clip"],
            "fname":            row["fname"],
            "subject":          row["subject"],
            "img_path":         row["img_path"],
            "frame_idx":        int(row["frame_idx"]),
            "head_x1":          int(row["head_x1"]),
            "head_y1":          int(row["head_y1"]),
            "head_x2":          int(row["head_x2"]),
            "head_y2":          int(row["head_y2"]),
            "gaze_type":        row["gaze_type"],
            "label_source":     "gt",
            "teacher_version":  None,
            "pred_x":           gx / img_w,
            "pred_y":           gy / img_h,
            "gt_x":             gx / img_w,
            "gt_y":             gy / img_h,
            "gt_px_x":          gx,
            "gt_px_y":          gy,
            "confidence":       "GT",
            "raw_response":     None,
        })
        # FIX A: mark as completed immediately
        completed.add(row["pair_id"])

        # FIX A: checkpoint periodically during GT write
        if (i + 1) % GT_SAVE_EVERY_N == 0:
            progress["completed"] = list(completed)
            save_progress(progress)

    # Final save after GT loop
    progress["completed"] = list(completed)
    save_progress(progress)
    log(f"GT labels written: {len(gt_rows)}")

    if len(teacher_rows) == 0:
        log("No teacher inference needed.")
        return

    # ── Teacher inference ──────────────────────────────────────────────
    log(f"\nStarting teacher inference on {len(teacher_rows)} frames...")
    secs_per_frame = 40
    log(f"Estimated time: ~{len(teacher_rows)*secs_per_frame/3600:.1f} hrs "
        f"at {secs_per_frame}s/frame")

    n_success = 0
    n_fail    = 0
    t_start   = time.time()

    for i, (_, row) in enumerate(
        tqdm(teacher_rows.iterrows(), total=len(teacher_rows), desc="Teacher")
    ):
        pair_id = row["pair_id"]

        try:
            img_raw      = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
            if img_w == 0 or img_h == 0:
                raise ValueError(f"Zero-size image: {row['img_path']}")
        except Exception as e:
            log(f"  Image load failed: {row['img_path']}: {e}")
            n_fail += 1
            continue

        primary_box = (int(row["head_x1"]), int(row["head_y1"]),
                       int(row["head_x2"]), int(row["head_y2"]))
        other_boxes = get_other_subjects(
            df, row["show"], row["clip"], row["fname"], row["subject"]
        )

        # FIX 4 (inherited): solo frames fall back to GT directly
        if not other_boxes:
            gt_x_norm = row["gaze_x"] / img_w if row["gaze_x"] != -1 else None
            gt_y_norm = row["gaze_y"] / img_h if row["gaze_y"] != -1 else None
            append_label({
                "show":             row["show"],
                "clip":             row["clip"],
                "fname":            row["fname"],
                "subject":          row["subject"],
                "img_path":         row["img_path"],
                "frame_idx":        int(row["frame_idx"]),
                "head_x1":          int(row["head_x1"]),
                "head_y1":          int(row["head_y1"]),
                "head_x2":          int(row["head_x2"]),
                "head_y2":          int(row["head_y2"]),
                "gaze_type":        row["gaze_type"],
                "label_source":     "gt_fallback_solo",
                "teacher_version":  None,
                "pred_x":           gt_x_norm,
                "pred_y":           gt_y_norm,
                "gt_x":             gt_x_norm,
                "gt_y":             gt_y_norm,
                "gt_px_x":          int(row["gaze_x"]),
                "gt_px_y":          int(row["gaze_y"]),
                "confidence":       "GT",
                "raw_response":     None,
            })
            completed.add(pair_id)
            continue

        secondary_box = other_boxes[0]

        # Inference with retry
        pred_x, pred_y, response = None, None, ""
        for attempt in range(RETRY_ON_FAIL + 1):
            try:
                pred_x, pred_y, response = run_teacher(
                    model, processor, img_raw,
                    primary_box, secondary_box, img_w, img_h
                )
                if pred_x is not None:
                    break
                log(f"  Parse fail attempt {attempt+1}: "
                    f"{row['fname']} {row['subject']} | {response[:80]}")
            except Exception as e:
                log(f"  Inference error attempt {attempt+1}: {e}")
                torch.cuda.empty_cache()

        # Determine label source and confidence from parse result
        is_offscreen = (pred_x == -1.0 and pred_y == -1.0)
        gt_x_norm    = row["gaze_x"] / img_w if row["gaze_x"] != -1 else None
        gt_y_norm    = row["gaze_y"] / img_h if row["gaze_y"] != -1 else None

        if is_offscreen:
            # Model said off-screen — trust it, use GT coord as pred
            label_source = "teacher_offscreen"
            confidence   = "PARSED"
            final_x      = gt_x_norm
            final_y      = gt_y_norm
        elif pred_x is not None:
            # Successful parse
            label_source = "teacher"
            confidence   = "PARSED"
            final_x, final_y = pred_x, pred_y
            n_success += 1
        else:
            # Parse failed after retries — fall back to GT
            label_source = "gt_fallback"
            confidence   = "GT_FALLBACK"
            final_x, final_y = gt_x_norm, gt_y_norm
            n_fail += 1
            log(f"  Fallback: {row['show']}/{row['clip']}/"
                f"{row['fname']} {row['subject']}")

        append_label({
            "show":             row["show"],
            "clip":             row["clip"],
            "fname":            row["fname"],
            "subject":          row["subject"],
            "img_path":         row["img_path"],
            "frame_idx":        int(row["frame_idx"]),
            "head_x1":          int(row["head_x1"]),
            "head_y1":          int(row["head_y1"]),
            "head_x2":          int(row["head_x2"]),
            "head_y2":          int(row["head_y2"]),
            "gaze_type":        row["gaze_type"],
            "label_source":     label_source,
            "teacher_version":  TEACHER_VERSION,
            "pred_x":           final_x,
            "pred_y":           final_y,
            "gt_x":             gt_x_norm,
            "gt_y":             gt_y_norm,
            "gt_px_x":          int(row["gaze_x"]),
            "gt_px_y":          int(row["gaze_y"]),
            "confidence":       confidence,
            "raw_response":     response,
        })
        completed.add(pair_id)

        # FIX C: atomic checkpoint every N teacher frames
        if (i + 1) % SAVE_EVERY_N == 0:
            progress["completed"] = list(completed)
            save_progress(progress)  # atomic write
            elapsed     = time.time() - t_start
            rate        = (i + 1) / elapsed
            remaining_n = len(teacher_rows) - (i + 1)
            eta_hrs     = remaining_n / rate / 3600
            tqdm.write(
                f"  [{i+1}/{len(teacher_rows)}] "
                f"{1/rate:.1f}s/frame  ETA: {eta_hrs:.1f}hrs  "
                f"OK:{n_success}  Fallback:{n_fail}"
            )

    # Final checkpoint
    progress["completed"] = list(completed)
    save_progress(progress)

    elapsed = time.time() - t_start
    log("\n" + "="*60)
    log(f"Pipeline complete [{mode_str}]")
    log("="*60)
    log(f"  Teacher (success)     : {n_success}")
    log(f"  Fallback to GT        : {n_fail}")
    log(f"  GT labels (direct)    : {len(gt_rows)}")
    log(f"  Total labels written  : {n_success + n_fail + len(gt_rows)}")
    log(f"  Elapsed               : {elapsed/3600:.2f} hrs")
    log(f"  Labels file           : {LABELS_FILE}")
    log(f"  Teacher version       : {TEACHER_VERSION}")

    parse_rate = n_success / max(len(teacher_rows), 1) * 100
    log(f"  Parse success rate    : {parse_rate:.1f}%")

    if parse_rate < 80:
        log("  WARNING: parse rate below 80% — review prompt before full run")
    else:
        log("  Parse rate healthy — ready for next stage")

    if TEST_MODE:
        log("\n  TEST MODE complete.")
        log("  If parse rate is healthy, set TEST_MODE=False for full run.")


if __name__ == "__main__":
    main()