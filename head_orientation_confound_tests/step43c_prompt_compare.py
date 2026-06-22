"""
prompt_compare.py — standalone A/B test for teacher prompt variants.
Does NOT touch step40_teacher_pipeline_004.py, labels_test.jsonl,
or any progress/checkpoint files. Read-only against the manifest/images.

Usage:
    python prompt_compare.py
"""

import re
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH     = r"C:\repo\standard_project\videoattentiontarget"
MANIFEST_PATH = "frame_manifest.csv"
MODEL_NAME    = "OpenGVLab/InternVL3-8B-hf"
MAX_NEW_TOKENS = 150

# Pick exactly the frames you want to compare.
# (show, clip, fname, subject) — subject is the "Person A" being predicted for.
TARGET_FRAMES = [
    # Conan 1348_1469 — Person 1 (s02) view-change left->right
    ("Conan", "1348_1469", "00001358.jpg", "s02"),
    ("Conan", "1348_1469", "00001370.jpg", "s02"),
    ("Conan", "1348_1469", "00001373.jpg", "s02"),
    ("Conan", "1348_1469", "00001376.jpg", "s02"),
    # Conan 1348_1469 — Person 3 (s00) view-change to menu, then to Person 1
    ("Conan", "1348_1469", "00001376.jpg", "s00"),
    ("Conan", "1348_1469", "00001379.jpg", "s00"),
    ("Conan", "1348_1469", "00001382.jpg", "s00"),
    ("Conan", "1348_1469", "00001388.jpg", "s00"),
    ("Conan", "1348_1469", "00001403.jpg", "s00"),
    # Conan 0_300 — head-pose vs camera-gaze confound clip
    # NOTE: fill in real subject IDs once you've checked the manifest —
    # these are placeholders, see CHECK BLOCK below.
    ("Conan", "0_300", "00000009.jpg", "s00"),
    ("Conan", "0_300", "00000175.jpg", "s00"),
]

# ── PROMPT VARIANTS ─────────────────────────────────────────────────────

def build_prompt_v4_current(img_w, img_h, primary_box, secondary_box):
    """Exact copy of build_social_prompt() from step40_teacher_pipeline_004.py — the baseline."""
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


def build_prompt_v5_eyes(img_w, img_h, primary_box, secondary_box):
    """
    Variant 1: explicitly instructs the model to prioritise eye direction
    over head orientation, and adds an explicit "neither" option so the
    model has somewhere to put a non-B, non-face target.
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

IMPORTANT: Base your answer on Person A's EYE DIRECTION, not head orientation.
A person's head can be turned one way while their eyes look another way —
look closely at the eyes/pupils within A's head box before deciding.

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)

Rules:
- If A's eyes are directed at B's face → use B's face centre: ({b_cx},{b_cy})
- If A's eyes are directed at an object or location that is NOT B's face →
  estimate that object/location's coordinates directly, even if A's head
  is turned toward B
- If A looks off-screen → use (-1,-1)
- x,y are normalised 0.0-1.0 (0,0=top-left, 1,1=bottom-right)

After the coordinate, add one sentence explaining why, and state whether
you based this on eye direction or head orientation."""

def build_prompt_v6_multicandidate(img_w, img_h, primary_box, primary_label,
                                     other_boxes_with_labels):
    """
    other_boxes_with_labels: list of (box, spatial_label) tuples for every
    other subject in frame, e.g. [((x1,y1,x2,y2), "the person on the left"), ...]
    """
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w,3), round(ay1/img_h,3),
          round(ax2/img_w,3), round(ay2/img_h,3))

    candidates_text = []
    for (bx1, by1, bx2, by2), label in other_boxes_with_labels:
        cx = round((bx1+bx2)/2/img_w, 3)
        cy = round((by1+by2)/2/img_h, 3)
        candidates_text.append(
            f"  - {label}: head box px ({bx1},{by1})-({bx2},{by2}), face centre ({cx},{cy})"
        )
    candidates_block = "\n".join(candidates_text)

    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.

SUBJECT (predict gaze for): {primary_label}, head box normalised ({na[0]},{na[1]})-({na[2]},{na[3]})

OTHER PEOPLE IN FRAME:
{candidates_block}

Look closely at {primary_label}'s eyes and head together to determine where
they are looking. They may be looking at one of the other people listed
above, OR at something else entirely — an object, the table, off to the
side, or off-screen. Do not assume they are looking at whichever person
they are facing or talking to; check what their eyes are actually doing.

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)

Rules:
- If looking at one of the listed people's faces → use that person's face centre coordinates
- If looking at an object, surface, or location that is NOT a listed face →
  estimate that location's coordinates directly (e.g. a table, menu, prop, or empty space)
- If looking off-screen or eyes not visible → use (-1,-1)
- x,y are normalised 0.0-1.0 (0,0=top-left, 1,1=bottom-right)

After the coordinate, name which option above you chose and why, in one sentence."""

PROMPT_VARIANTS = {
    "v4_current": build_prompt_v4_current,
    "v5_eyes":    build_prompt_v5_eyes,
    "v6_multicandidate":    build_prompt_v6_multicandidate,    
}

# ── PARSING (identical to production) ───────────────────────────────────

def parse_gaze_xy(text):
    if not text or not text.strip():
        return None, None
    if re.search(r"GAZE_XY:\s*\(-1\s*,\s*-1\)", text):
        return -1.0, -1.0
    match = re.search(
        r"GAZE_XY:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text
    )
    if match:
        x = max(0.0, min(1.0, float(match.group(1))))
        y = max(0.0, min(1.0, float(match.group(2))))
        return x, y
    matches = re.findall(
        r"\(\s*(0\.[0-9]+|1\.0|0\.0)\s*,\s*(0\.[0-9]+|1\.0|0\.0)\s*\)", text
    )
    if matches:
        x = max(0.0, min(1.0, float(matches[0][0])))
        y = max(0.0, min(1.0, float(matches[0][1])))
        return x, y
    return None, None

# ── INFERENCE ────────────────────────────────────────────────────────────

def run_inference(model, processor, img_raw, prompt_text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_raw},
            {"type": "text",  "text": prompt_text},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )[0]
    pred_x, pred_y = parse_gaze_xy(response)
    return pred_x, pred_y, response


def get_other_subjects(df, show, clip, fname, this_subject):
    others = df[
        (df["show"] == show) & (df["clip"] == clip) &
        (df["fname"] == fname) & (df["subject"] != this_subject)
    ]
    return [
        (int(r["head_x1"]), int(r["head_y1"]), int(r["head_x2"]), int(r["head_y2"]))
        for _, r in others.iterrows()
    ]


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest loaded: {len(df)} rows")

    # CHECK BLOCK — run this once first if you're unsure of 0_300 subject IDs:
    sample = df[(df["show"] == "Conan") & (df["clip"] == "0_300")]
    print("\n0_300 subjects available:", sorted(sample["subject"].unique()))
    print("0_300 fnames available (first 5):", sorted(sample["fname"].unique())[:5])
    print("Update TARGET_FRAMES above with real subject IDs if needed, then rerun.\n")

    print(f"Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True,
    ).eval()
    print("Model loaded.\n")

    for show, clip, fname, subject in TARGET_FRAMES:
        rows = df[
            (df["show"] == show) & (df["clip"] == clip) &
            (df["fname"] == fname) & (df["subject"] == subject)
        ]
        if rows.empty:
            print(f"SKIP — no manifest row for {show}/{clip}/{fname}/{subject}\n")
            continue
        row = rows.iloc[0]

        try:
            img_raw = Image.open(row["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
        except Exception as e:
            print(f"SKIP — image load failed for {row['img_path']}: {e}\n")
            continue

        primary_box = (int(row["head_x1"]), int(row["head_y1"]),
                       int(row["head_x2"]), int(row["head_y2"]))
        other_boxes = get_other_subjects(df, show, clip, fname, subject)
        if not other_boxes:
            print(f"SKIP — no secondary subject for {show}/{clip}/{fname}/{subject}\n")
            continue
        secondary_box = other_boxes[0]

        # Build the v6 "multi-candidate" inputs: every other subject in this
        # frame, labelled by left/centre/right position rather than A/B.
        all_subjects_this_frame = df[
            (df["show"] == show) & (df["clip"] == clip) & (df["fname"] == fname)
        ]
        sorted_subjects = all_subjects_this_frame.sort_values("head_x1")
        ordered_ids = list(sorted_subjects["subject"])
        n = len(ordered_ids)

        def spatial_label(subj_id):
            idx = ordered_ids.index(subj_id)
            if n <= 2:
                return ["the person on the left", "the person on the right"][idx]
            if idx == 0:
                return "the person on the left"
            if idx == n - 1:
                return "the person on the right"
            return "the person in the centre"

        primary_label = spatial_label(subject)
        other_boxes_with_labels = [
            (
                (int(r["head_x1"]), int(r["head_y1"]), int(r["head_x2"]), int(r["head_y2"])),
                spatial_label(r["subject"]),
            )
            for _, r in all_subjects_this_frame.iterrows()
            if r["subject"] != subject
        ]

        # Each variant needs different arguments — list them here by name.
        PROMPT_ARGS = {
            "v4_current": (img_w, img_h, primary_box, secondary_box),
            "v5_eyes":    (img_w, img_h, primary_box, secondary_box),
            "v6_multicandidate": (img_w, img_h, primary_box, primary_label, other_boxes_with_labels),
        }

        gt_x = row["gaze_x"] / img_w if row["gaze_x"] != -1 else None
        gt_y = row["gaze_y"] / img_h if row["gaze_y"] != -1 else None

        print("=" * 70)
        print(f"{show} / {clip} / {fname} / {subject}")
        print(f"  GT: ({gt_x:.3f}, {gt_y:.3f})" if gt_x is not None else "  GT: off-screen")

        for variant_name, build_fn in PROMPT_VARIANTS.items():
            prompt_text = build_fn(*PROMPT_ARGS[variant_name])
            pred_x, pred_y, response = run_inference(model, processor, img_raw, prompt_text)
            pred_str = f"({pred_x:.3f}, {pred_y:.3f})" if pred_x is not None else "PARSE FAILED"
            print(f"\n  [{variant_name}]")
            print(f"    PRED: {pred_str}")
            print(f"    RAW : {response[:300]}{'...' if len(response) > 300 else ''}")
        print()


if __name__ == "__main__":
    main()