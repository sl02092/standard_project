"""
prompt_compare_with_explanations.py — Manifest-independent A/B prompt evaluation.
Fixes console token warnings and prints out the full model explanation text.
"""

import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH      = r"C:\repo\standard_project\videoattentiontarget"
MODEL_NAME     = "OpenGVLab/InternVL3-8B-hf"
MAX_NEW_TOKENS = 150

# Target frame evaluation list (handles both 4-digit and 5-digit variants automatically)
TARGET_FRAMES = [
    ("13525_13575", 1367, "s00"),
    ("13525_13575", 1370, "s00"),
    ("13525_13575", 1375, "s00"),
    ("2250_2300",  2295, "s00"),
    ("2250_2300",  2300, "s00"),
    ("1650_1775",  1673, "s00"),
    ("1650_1775",  1700, "s00"),
    ("1650_1775",  1728, "s00"),
    ("1650_1775",  1746, "s00"),
    ("1650_1775",  1751, "s00"),
    ("1650_1775",  1770, "s00"),
]

# ── PROMPT VARIANTS ─────────────────────────────────────────────────────

def build_prompt_v4_current(img_w, img_h, primary_box, secondary_box):
    ax1, ay1, ax2, ay2 = primary_box
    bx1, by1, bx2, by2 = secondary_box
    b_cx = round((bx1 + bx2) / 2 / img_w, 3)
    b_cy = round((by1 + by2) / 2 / img_h, 3)
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))
    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.
PERSON A (predict gaze): head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})
PERSON B (other person): head box px ({bx1},{by1}) to ({bx2},{by2}), face centre ({b_cx},{b_cy})

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)
Rules:
- If A looks at B's face → use B's face centre: ({b_cx},{b_cy})
- If A looks at an object → estimate object location in normalised coords
- If A looks off-screen → use (-1,-1)
After the coordinate, add one sentence explaining why."""

def build_prompt_v5_eyes(img_w, img_h, primary_box, secondary_box):
    ax1, ay1, ax2, ay2 = primary_box
    bx1, by1, bx2, by2 = secondary_box
    b_cx = round((bx1 + bx2) / 2 / img_w, 3)
    b_cy = round((by1 + by2) / 2 / img_h, 3)
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))
    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.
PERSON A (predict gaze): head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})
PERSON B (other person): head box px ({bx1},{by1}) to ({bx2},{by2}), face centre ({b_cx},{b_cy})

IMPORTANT: Base your answer on Person A's EYE DIRECTION, not head orientation.
YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)
After the coordinate, add one sentence explaining why, and state whether you based this on eye direction or head orientation."""

def build_prompt_v6_multicandidate(img_w, img_h, primary_box, primary_label, other_boxes_with_labels):
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))
    candidates_text = [f"  - {lbl}: head box px ({b[0]},{b[1]})-({b[2]},{b[3]}), face centre ({round((b[0]+b[2])/2/img_w,3)},{round((b[1]+b[3])/2/img_h,3)})" for b, lbl in other_boxes_with_labels]
    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.
SUBJECT (predict gaze for): {primary_label}, head box normalised ({na[0]},{na[1]})-({na[2]},{na[3]})
OTHER PEOPLE IN FRAME:
{"\n".join(candidates_text)}

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)
After the coordinate, name which option above you chose and why, in one sentence."""

PROMPT_VARIANTS = {"v4_current": build_prompt_v4_current, "v5_eyes": build_prompt_v5_eyes, "v6_multicandidate": build_prompt_v6_multicandidate}

# ── LOOKUP ENGINE ───────────────────────────────────────────────────────

def find_dir_by_clip_id(root_path, clip_id):
    target = str(clip_id).strip()
    for root, dirs, _ in os.walk(root_path):
        for d in dirs:
            if target in d.strip():
                return os.path.join(root, d)
    return None

def parse_txt_file_for_frame(filepath, frame_num):
    target_idx = int(frame_num)
    target_str = str(frame_num)
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
                
                is_match = (line_idx == target_idx) or (file_idx_str.endswith(target_str[-2:]) and len(target_str) == 4 and len(file_idx_str) == 8)
                
                if is_match:
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
        print(f"  [DEBUG ERROR] Missing directory on disk for Clip: {clip}")
        return None

    subj_file = f"{subject}.txt"
    primary_txt_path = os.path.join(ann_dir, subj_file)
    primary_data = parse_txt_file_for_frame(primary_txt_path, frame_num)
    
    if not primary_data:
        print(f"  [DEBUG ERROR] Frame matching target '{frame_num}' not found inside: {primary_txt_path}")
        return None

    img_path = os.path.join(img_dir, primary_data["fname_actual"])
    if not os.path.exists(img_path):
        print(f"  [DEBUG ERROR] Frame image file missing from disk: {img_path}")
        return None

    others = []
    for f in os.listdir(ann_dir):
        if f.endswith(".txt") and f != subj_file:
            sib_data = parse_txt_file_for_frame(os.path.join(ann_dir, f), frame_num)
            if sib_data:
                others.append((f.replace(".txt", ""), sib_data))

    return {"img_path": img_path, "primary": primary_data, "others": others}

def parse_gaze_xy(text):
    if not text or not text.strip(): return None, None
    if re.search(r"GAZE_XY:\s*\(-1\s*,\s*-1\)", text): return -1.0, -1.0
    match = re.search(r"GAZE_XY:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text)
    if match: return max(0.0, min(1.0, float(match.group(1)))), max(0.0, min(1.0, float(match.group(2))))
    return None, None

# ── RUNTIME ENGINE ───────────────────────────────────────────────────────

def main():
    print(f"Loading model architecture: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda").eval()
    
    # Track down padding token configuration to cleanly eliminate the transformers logging noise
    pad_token_id = getattr(processor.tokenizer, 'pad_token_id', None) or processor.tokenizer.eos_token_id
    print("Model ready.\n")

    for clip, frame_num, subject in TARGET_FRAMES:
        print("=" * 80)
        print(f"TARGET lookup: Clip {clip} | Frame {frame_num} | Subject {subject}")
        
        context = resolve_context(clip, frame_num, subject)
        if not context:
            print(f"SKIP — Lookup failure processing frame context.")
            continue

        try:
            img_raw = Image.open(context["img_path"]).convert("RGB")
            img_w, img_h = img_raw.size
        except Exception as e:
            print(f"SKIP — Unreadable file path: {context['img_path']} ({e})")
            continue

        p = context["primary"]
        primary_box = (int(p["head_x1"]), int(p["head_y1"]), int(p["head_x2"]), int(p["head_y2"]))
        
        all_subjects = [(subject, p)] + context["others"]
        all_subjects.sort(key=lambda x: x[1]["head_x1"])
        ordered_ids = [s[0] for s in all_subjects]
        n_total = len(ordered_ids)

        def get_spatial_label(subj_id):
            idx = ordered_ids.index(subj_id)
            if n_total <= 1: return "the only person"
            if n_total == 2: return ["the person on the left", "the person on the right"][idx]
            if idx == 0: return "the person on the left"
            if idx == n_total - 1: return "the person on the right"
            return "the person in the centre"

        primary_label = get_spatial_label(subject)
        other_boxes_with_labels = [((int(s[1]["head_x1"]), int(s[1]["head_y1"]), int(s[1]["head_x2"]), int(s[1]["head_y2"])), get_spatial_label(s[0])) for s in all_subjects if s[0] != subject]

        secondary_box = primary_box
        if context["others"]:
            s_data = context["others"][0][1]
            secondary_box = (int(s_data["head_x1"]), int(s_data["head_y1"]), int(s_data["head_x2"]), int(s_data["head_y2"]))

        PROMPT_ARGS = {
            "v4_current": (img_w, img_h, primary_box, secondary_box),
            "v5_eyes":    (img_w, img_h, primary_box, secondary_box),
            "v6_multicandidate": (img_w, img_h, primary_box, primary_label, other_boxes_with_labels),
        }

        gt_x = p["gaze_x"] / img_w if p["gaze_x"] != -1 else None
        gt_y = p["gaze_y"] / img_h if p["gaze_y"] != -1 else None
        print(f"  Frame Resolved : {p['fname_actual']}")
        print(f"  Ground Truth   : ({gt_x:.3f}, {gt_y:.3f})" if gt_x is not None else "  Ground Truth   : off-screen")

        for variant_name, build_fn in PROMPT_VARIANTS.items():
            prompt_text = build_fn(*PROMPT_ARGS[variant_name])
            messages = [{"role": "user", "content": [{"type": "image", "image": img_raw}, {"type": "text", "text": prompt_text}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.float16)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS, 
                    do_sample=False,
                    pad_token_id=pad_token_id
                )
            
            response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            pred_x, pred_y = parse_gaze_xy(response)
            
            # Format coordinate string nicely for screen display
            if pred_x == -1.0 and pred_y == -1.0:
                coord_str = "off-screen (-1, -1)"
            elif pred_x is not None and pred_y is not None:
                coord_str = f"({pred_x:.3f}, {pred_y:.3f})"
            else:
                coord_str = "PARSING ERROR"
                
            print(f"\n    [{variant_name}] PRED COORDS : {coord_str}")
            print(f"    [{variant_name}] MODEL TEXT  : {response}")

if __name__ == "__main__":
    main()