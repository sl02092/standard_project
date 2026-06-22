"""
step43f_prompt_compare_qwen.py — Qwen2.5-VL-7B-Instruct variant of the
prompt comparison harness. Same TARGET_FRAMES, same prompt variants, same
lookup logic as step43e — only the model loading and inference call differ.

Requires transformers >= 4.49.0 for Qwen2.5-VL support.
"""

import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH      = r"C:\repo\standard_project\videoattentiontarget"
MODEL_NAME     = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 200

TARGET_FRAMES = [
    ("13525_13575", 13567, "s00"),
    ("13525_13575", 13570, "s00"),
    ("13525_13575", 13575, "s00"),
    ("2250_2300",  2295, "s00"),
    ("2250_2300",  2300, "s00"),
    ("1650_1775",  1673, "s00"),
    ("1650_1775",  1700, "s00"),
    ("1650_1775",  1728, "s00"),
    ("1650_1775",  1746, "s00"),
    ("1650_1775",  1751, "s00"),
    ("1650_1775",  1770, "s00"),
]

# ── PROMPT VARIANTS — unchanged from step43e ────────────────────────────

def build_prompt_clean_baseline(img_w, img_h, primary_box, primary_label, others_with_labels):
    """An updated baseline that drops Person B completely if the subject is alone."""
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))

    prompt = f"Gaze estimation task. Image: {img_w}x{img_h}px.\n"
    prompt += f"PERSON A (predict gaze): head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})\n"

    if others_with_labels:
        b, lbl = others_with_labels[0]
        b_cx = round((b[0] + b[2]) / 2 / img_w, 3)
        b_cy = round((b[1] + b[3]) / 2 / img_h, 3)
        prompt += f"PERSON B (other person): head box px ({b[0]},{b[1]}) to ({b[2]},{b[3]}), face centre ({b_cx},{b_cy})\n"
    else:
        prompt += "PERSON B (other person): None present in frame.\n"

    prompt += """\nYOUR FIRST LINE MUST BE IN THIS EXACT FORMAT:
GAZE_XY: (x, y)

Rules:
- If A looks at B's face (if present) -> use B's face centre
- If A looks at an object inside the image boundaries -> estimate object location in normalised coords
- If A looks completely off-screen / out of the image boundaries -> use (OFF, OFF) exactly
After the coordinate line, add one sentence explaining why.

Note: if you know the person is looking off-screen, return (OFF, OFF).
If you know the person is looking at an object, spend time determining the position of the
object before providing coordinates. Don't just give mid-screen coordinates — think about
where the object actually is."""
    return prompt


def build_prompt_dynamic_cot(img_w, img_h, primary_box, primary_label, others_with_labels):
    """Forces the model to explicitly perceive the context and think out loud first."""
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))

    prompt = f"Gaze estimation task. Image: {img_w}x{img_h}px.\n"
    prompt += f"SUBJECT (predict gaze for): {primary_label}, head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})\n"

    if others_with_labels:
        prompt += "OTHER PEOPLE IN FRAME:\n"
        for b, lbl in others_with_labels:
            b_cx = round((b[0] + b[2]) / 2 / img_w, 3)
            b_cy = round((b[1] + b[3]) / 2 / img_h, 3)
            prompt += f" - {lbl}: head box px ({b[0]},{b[1]})-({b[2]},{b[3]}), face centre ({b_cx},{b_cy})\n"
    else:
        prompt += "OTHER PEOPLE IN FRAME: None. This person is alone in this scene.\n"

    prompt += """\nAnalysis Steps:
1. Examine the subject's face orientation and precise eye/pupil direction.
2. Check if the vector of their eye gaze points entirely outside the visible image frame boundaries.
3. Write 1-2 sentences of step-by-step reasoning.
4. Output a line specifying if the target is off-screen (Is_Off_Screen: Yes or Is_Off_Screen: No).
5. Conclude your response on the final line with the exact format: GAZE_XY: (x, y)

Note: If Is_Off_Screen is Yes, GAZE_XY must be written as (OFF, OFF).

Note: if you know the person is looking off-screen, return (OFF, OFF).
If you know the person is looking at an object, spend time determining the position of the
object before providing coordinates. Don't just give mid-screen coordinates — think about
where the object actually is."""
    return prompt


def build_prompt_eye_vs_pose_cot(img_w, img_h, primary_box, primary_label, others_with_labels):
    """Explicitly checks for head rotation discrepancies vs where the eyes look."""
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))

    prompt = f"Fine-grained gaze tracking. Image: {img_w}x{img_h}px.\n"
    prompt += f"Target Individual: {primary_label}, head region ({na[0]},{na[1]}) to ({na[2]},{na[3]})\n"

    if others_with_labels:
        prompt += "Other visible people:\n"
        for b, lbl in others_with_labels:
            b_cx = round((b[0] + b[2]) / 2 / img_w, 3)
            b_cy = round((b[1] + b[3]) / 2 / img_h, 3)
            prompt += f" - {lbl}: face centre ({b_cx},{b_cy})\n"
    else:
        prompt += "Other visible people: None.\n"

    prompt += """\nInstructions:
- Disregard the overall angle of the head if the eyeballs are shifted. Isolate the precise glance direction of their pupils.
- Determine if the gaze vector stays inside the image frame or breaks out of the bounds (off-screen).

Format structure:
Reasoning: [Describe eye glance target or direction, explicitly stating if it points off-screen]
Is_Off_Screen: [Write Yes or No]
GAZE_XY: [Write (OFF, OFF) if off-screen, otherwise write the (x, y) coordinates]

Note: if you know the person is looking off-screen, return (OFF, OFF).
If you know the person is looking at an object, spend time determining the position of the
object before providing coordinates. Don't just give mid-screen coordinates — think about
where the object actually is."""
    return prompt


PROMPT_VARIANTS = {
    "v_clean_baseline": build_prompt_clean_baseline,
    "v_dynamic_cot": build_prompt_dynamic_cot,
    "v_eye_pose_cot": build_prompt_eye_vs_pose_cot
}

# ── LOOKUP ENGINE — unchanged from step43e ──────────────────────────────

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


def resolve_context(clip, frame_num, subject):
    ann_dir = find_dir_by_clip_id(os.path.join(BASE_PATH, "annotations"), clip)
    img_dir = find_dir_by_clip_id(os.path.join(BASE_PATH, "images"), clip)
    if not ann_dir or not img_dir:
        return None

    subj_file = f"{subject}.txt"
    primary_data = parse_txt_file_for_frame(os.path.join(ann_dir, subj_file), frame_num)
    if not primary_data:
        return None

    img_path = os.path.join(img_dir, primary_data["fname_actual"])
    if not os.path.exists(img_path):
        return None

    others = []
    for f in os.listdir(ann_dir):
        if f.endswith(".txt") and f != subj_file:
            sib_data = parse_txt_file_for_frame(os.path.join(ann_dir, f), frame_num)
            if sib_data:
                others.append((f.replace(".txt", ""), sib_data))
    return {"img_path": img_path, "primary": primary_data, "others": others}


def parse_gaze_xy(text):
    if not text or not text.strip():
        return None, None

    if (re.search(r"GAZE_XY:\s*\(-1\s*,\s*-1\)", text, re.IGNORECASE) or
        re.search(r"GAZE_XY:\s*\(\s*OFF\s*,\s*OFF\s*\)", text, re.IGNORECASE) or
        re.search(r"Is_Off_Screen:\s*Yes", text, re.IGNORECASE)):
        return -1.0, -1.0

    match = re.search(r"GAZE_XY:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text, re.IGNORECASE)
    if match:
        return max(0.0, min(1.0, float(match.group(1)))), max(0.0, min(1.0, float(match.group(2))))

    return None, None

# ── RUNTIME ENGINE — Qwen2.5-VL loading + inference ─────────────────────

def main():
    print(f"Loading model architecture: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="cuda",
    ).eval()
    pad_token_id = getattr(processor.tokenizer, 'pad_token_id', None) or processor.tokenizer.eos_token_id
    print("Model ready.\n")

    for clip, frame_num, subject in TARGET_FRAMES:
        print("=" * 80)
        print(f"TARGET: Clip {clip} | Frame {frame_num} | Subject {subject}")

        context = resolve_context(clip, frame_num, subject)
        if not context:
            print(f"SKIP — Lookup failure processing frame context.")
            continue

        img_raw = Image.open(context["img_path"]).convert("RGB")
        img_w, img_h = img_raw.size

        p = context["primary"]
        primary_box = (int(p["head_x1"]), int(p["head_y1"]), int(p["head_x2"]), int(p["head_y2"]))

        all_subjects = [(subject, p)] + context["others"]
        all_subjects.sort(key=lambda x: x[1]["head_x1"])
        ordered_ids = [s[0] for s in all_subjects]
        n_total = len(ordered_ids)

        def get_spatial_label(subj_id):
            idx = ordered_ids.index(subj_id)
            if n_total <= 1: return "the only person in frame"
            if n_total == 2: return ["the person on the left", "the person on the right"][idx]
            if idx == 0: return "the person on the left"
            if idx == n_total - 1: return "the person on the right"
            return "the person in the centre"

        primary_label = get_spatial_label(subject)
        other_boxes_with_labels = [((int(s[1]["head_x1"]), int(s[1]["head_y1"]), int(s[1]["head_x2"]), int(s[1]["head_y2"])), get_spatial_label(s[0])) for s in all_subjects if s[0] != subject]

        gt_x = p["gaze_x"] / img_w if p["gaze_x"] != -1 else None
        gt_y = p["gaze_y"] / img_h if p["gaze_y"] != -1 else None
        print(f"  Frame File     : {p['fname_actual']}")
        print(f"  Presence Count : 1 person found" if not other_boxes_with_labels else f"  Presence Count : {len(all_subjects)} people found")
        print(f"  Ground Truth   : ({gt_x:.3f}, {gt_y:.3f})" if gt_x is not None else "  Ground Truth   : off-screen")

        for variant_name, build_fn in PROMPT_VARIANTS.items():
            prompt_text = build_fn(img_w, img_h, primary_box, primary_label, other_boxes_with_labels)
            messages = [{"role": "user", "content": [{"type": "image", "image": img_raw}, {"type": "text", "text": prompt_text}]}]

            # Qwen2.5-VL chat template — same call shape as InternVL3, but
            # Qwen's processor handles its own dynamic image-token expansion,
            # so no manual pixel/patch config needed for single-image input.
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=pad_token_id
                )

            response = processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            pred_x, pred_y = parse_gaze_xy(response)

            if pred_x == -1.0 and pred_y == -1.0:
                coord_str = "off-screen (-1, -1)"
            elif pred_x is not None and pred_y is not None:
                coord_str = f"({pred_x:.3f}, {pred_y:.3f})"
            else:
                coord_str = "PARSING ERROR"

            clean_response = response.replace('\n', ' | ')
            print(f"\n    [{variant_name}] PRED COORDS : {coord_str}")
            print(f"    [{variant_name}] MODEL TEXT  : {clean_response}")


if __name__ == "__main__":
    main()