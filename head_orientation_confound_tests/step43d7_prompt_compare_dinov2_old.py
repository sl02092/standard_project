"""
step43d7_prompt_compare_dinov2.py — Context-aware A/B prompt evaluation using DINOv2.

Adapted from step43d7_prompt_compare.py (InternVL3-8B-hf) to use DINOv2 as the
vision backbone.

IMPORTANT — DINOv2 is a vision encoder, NOT a generative VLM:
    DINOv2 (facebook/dinov2-large etc.) produces dense image embeddings; it has no
    language decoder and cannot respond to text prompts directly. To use it for gaze
    estimation this script pairs DINOv2 image features with a lightweight generative
    LLM (default: microsoft/Phi-3-mini-4k-instruct) via a simple projection head.
    The image patch tokens are prepended to the text prompt tokens before generation —
    matching the approach used in LLaVA-style pipelines.

    If "DINOv3" becomes publicly available as a standalone model, replace
    DINO_MODEL_NAME with its HuggingFace identifier and verify the feature-extraction
    call in `encode_image_with_dino()` still applies (the ViT interface is stable
    across DINOv2 variants so the change should be drop-in).

    Alternative: if you have a fine-tuned DINOv2-headed gaze regression model,
    replace the `run_inference()` function with a direct regression forward pass and
    skip the LLM entirely — the data-loading and eval logic below is unchanged.

HOW TO POPULATE NEW_MULTIPERSON_OBJECT_FRAMES (same as original):
    Run against your real labels_full.jsonl to find candidates:

        import json
        candidates = []
        with open("labels_full.jsonl") as f:
            for line in f:
                row = json.loads(line)
                if row.get("gaze_type") == "object" and row.get("label_source") == "teacher":
                    candidates.append(row)
        # Then cross-reference against frame_manifest.csv or clip_selected.csv
        # to find which of these clips have n_subjects > 1 (multi-person),
        # and pick a handful of (clip, frame_idx/fname, subject) tuples.

    Fill in NEW_MULTIPERSON_OBJECT_FRAMES below with real (clip, frame_num, subject)
    tuples once identified.
"""

import os
import re
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_PATH      = os.environ.get("VAT_BASE_PATH", r"C:\repo\standard_project\videoattentiontarget")

# DINOv2 vision encoder — swap to "facebook/dinov2-giant" for max capacity,
# or "facebook/dinov2-small" for speed.
DINO_MODEL_NAME = "facebook/dinov2-large"

# Generative LLM that receives the projected DINO tokens + text prompt.
# Phi-3-mini is small and instruction-tuned; swap to any causal LM you prefer.
LLM_MODEL_NAME  = "microsoft/Phi-3-mini-4k-instruct"

MAX_NEW_TOKENS  = 200
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Target frames (identical to original script) ────────────────────────────

ORIGINAL_TARGET_FRAMES = [
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

NEW_MULTIPERSON_OBJECT_FRAMES = [
    # ("CLIP_ID_HERE", FRAME_NUM_HERE, "SUBJECT_HERE"),
]

TARGET_FRAMES = ORIGINAL_TARGET_FRAMES + NEW_MULTIPERSON_OBJECT_FRAMES


# ── PROMPT BUILDERS (unchanged from original — text side is model-agnostic) ─

def build_prompt_clean_baseline(img_w, img_h, primary_box, primary_label, others_with_labels):
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

Note: Ensure the coords match your the description of your analysis text.  If your text identifies the object as off-screen use (OFF, OFF). It is of greatest importance that the coords are correct. 
If you know the person is looking offscreen, return offscreen. if you know the person is looking at an object spend time determining the position of the object before providing coordinates.
Don't just give mid-screen coords, think about where the object actually is, don't just guess"""
    return prompt


def build_prompt_dynamic_cot(img_w, img_h, primary_box, primary_label, others_with_labels):
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

Note: Ensure the coords match your the description of your analysis text.  If your text identifies the object as off-screen use (OFF, OFF). It is of greatest importance that the coords are correct. 
If you know the person is looking offscreen, return offscreen. if you know the person is looking at an object spend time determining the position of the object before providing coordinates.
Don't just give mid-screen coords, think about where the object actually is, don't just guess"""
    return prompt


def build_prompt_eye_vs_pose_cot(img_w, img_h, primary_box, primary_label, others_with_labels):
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

Note: Ensure the coords match your the description of your analysis text.  If your text identifies the object as off-screen use (OFF, OFF). It is of greatest importance that the coords are correct. 
If you know the person is looking offscreen, return offscreen. if you know the person is looking at an object spend time determining the position of the object before providing coordinates.
Don't just give mid-screen coords, think about where the object actually is, don't just guess"""
    return prompt


def build_prompt_dynamic_cot_viewer_perspective(img_w, img_h, primary_box, primary_label, others_with_labels):
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w,3), round(ay1/img_h,3), round(ax2/img_w,3), round(ay2/img_h,3))
    prompt = f"Gaze estimation task. Image: {img_w}x{img_h}px.\n"
    prompt += f"SUBJECT (predict gaze for): {primary_label}, head box ({na[0]},{na[1]}) to ({na[2]},{na[3]})\n"
    prompt += "IMPORTANT: All directions (left/right) in your reasoning must be described from the VIEWER'S perspective (as seen on screen), NOT the subject's own left/right.\n"
    prompt += "IMPORTANT: If the subjects head is rotated towards the left side of the image, say the head is orientated left. If the subjects head is rotated towards the right side of the image, say the head is orientated right.\n"
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
3. Write 1-2 sentences of step-by-step reasoning, using VIEWER-PERSPECTIVE left/right only.
4. Output a line specifying if the target is off-screen (Is_Off_Screen: Yes or Is_Off_Screen: No).
5. Conclude your response on the final line with the exact format: GAZE_XY: (x, y) note here that the coords are in x y order - width then height, this order must be maintained.

Note: If Is_Off_Screen is Yes, GAZE_XY must be written as (OFF, OFF).

IMPORTANT: Ensure the coords match your the description of your analysis text. if you know the person is looking off-screen, return (OFF, OFF).
If you know the person is looking at an object, spend time determining the position of the
object before providing coordinates. Don't just give mid-screen coordinates — think about
where the object actually is - it is usually farther away than you might initially think."""
    return prompt


def build_universal_gaze_prompt(img_w, img_h, primary_box, primary_label, others_with_labels):
    ax1, ay1, ax2, ay2 = primary_box
    na = (round(ax1/img_w, 3), round(ay1/img_h, 3), round(ax2/img_w, 3), round(ay2/img_h, 3))
    prompt = f"TASK: Gaze Tracking and Spatial Grounding. Image Dimensions: {img_w}x{img_h}px.\n"
    prompt += f"TARGET SUBJECT (Predict gaze for): {primary_label} | Head Box (Normalized): [{na[0]}, {na[1]}, {na[2]}, {na[3]}]\n"
    if others_with_labels:
        prompt += "OTHER PEOPLE IN FRAME:\n"
        for b, lbl in others_with_labels:
            nb_x1 = round(b[0] / img_w, 3); nb_y1 = round(b[1] / img_h, 3)
            nb_x2 = round(b[2] / img_w, 3); nb_y2 = round(b[3] / img_h, 3)
            b_cx  = round((nb_x1 + nb_x2) / 2, 3)
            b_cy  = round((nb_y1 + nb_y2) / 2, 3)
            prompt += f" - {lbl}: Head Box [{nb_x1}, {nb_y1}, {nb_x2}, {nb_y2}] | Face Centre: ({b_cx}, {b_cy})\n"
    else:
        prompt += "OTHER PEOPLE IN FRAME: None. Target subject is alone in the scene.\n"
    prompt += """
ANALYSIS PROTOCOL:
1. Focus entirely on the target subject's precise eye/pupil direction. Do not rely on macro head orientation if it points away from the eyes.
2. Formulate a trajectory vector from the eyes. Determine if it cross-references any noted face center or traces completely out of the image boundaries.
3. If looking at an unlisted object, localize the exact spatial region of that object first.

REQUIRED RESPONSE SCHEMA:
Reasoning: [1-2 sentences of step-by-step spatial tracking analysis]
Is_Off_Screen: [Yes or No]
GAZE_XY: (x, y)

CRITICAL COMPLIANCE RULES:
- Output coordinates must be normalized floating points between 0.000 and 1.000.
- If Is_Off_Screen is Yes, GAZE_XY MUST be written exactly as (OFF, OFF). Do not use numbers or alternative strings.
- Never guess a generic center-screen default coordinate like (0.5, 0.5) out of uncertainty.
"""
    return prompt


PROMPT_VARIANTS = {
    "v_clean_baseline":               build_prompt_clean_baseline,
    "v_dynamic_cot":                  build_prompt_dynamic_cot,
    "v_eye_pose_cot":                 build_prompt_eye_vs_pose_cot,
    "v_dynamic_cot_viewer_perspective": build_prompt_dynamic_cot_viewer_perspective,
    "v5_universal":                   build_universal_gaze_prompt,
}


# ── DINO + LLM MODEL WRAPPERS ───────────────────────────────────────────────

class DinoProjectionHead(nn.Module):
    """
    Projects DINOv2 CLS + patch tokens into the LLM's embedding space.

    DINOv2-large outputs 1024-dim features; Phi-3-mini expects 3072-dim
    embeddings. Adjust `llm_hidden_size` if you swap the LLM.

    NOTE: This projection head is randomly initialised here. For meaningful
    gaze-estimation results you need to either:
      (a) fine-tune the whole pipeline end-to-end on a gaze dataset, OR
      (b) replace this class with a trained checkpoint you already have.
    The script as written will produce plausible-format but uncalibrated
    coordinate outputs — useful for verifying the pipeline runs correctly
    before training.
    """
    def __init__(self, dino_hidden_size: int = 1024, llm_hidden_size: int = 3072):
        super().__init__()
        self.proj = nn.Linear(dino_hidden_size, llm_hidden_size)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        # dino_features: (batch, num_tokens, dino_hidden_size)
        return self.proj(dino_features)


def load_models():
    """Load DINOv2 encoder, projection head, and the generative LLM."""
    print(f"Loading DINOv2 encoder : {DINO_MODEL_NAME}")
    dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(
        DINO_MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    print(f"Loading LLM            : {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    ).eval()

    # Determine DINOv2 hidden size from config
    dino_hidden_size = dino_model.config.hidden_size          # 1024 for dinov2-large
    llm_hidden_size  = llm_model.config.hidden_size           # e.g. 3072 for Phi-3-mini

    proj_head = DinoProjectionHead(dino_hidden_size, llm_hidden_size).to(DEVICE)
    proj_head = proj_head.half()  # match fp16

    print("All models loaded.\n")
    return dino_processor, dino_model, llm_tokenizer, llm_model, proj_head


def encode_image_with_dino(image: Image.Image, dino_processor, dino_model) -> torch.Tensor:
    """
    Run image through DINOv2 and return all token embeddings (CLS + patches).
    Returns: (1, num_tokens, hidden_size) float16 tensor on DEVICE.
    """
    inputs = dino_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = dino_model(**inputs)
    # last_hidden_state: (1, 1 + num_patches, hidden_size)
    return outputs.last_hidden_state.half()


def run_inference(image: Image.Image, prompt_text: str,
                  dino_processor, dino_model,
                  llm_tokenizer, llm_model,
                  proj_head) -> str:
    """
    Full forward pass:
      1. Encode image via DINOv2.
      2. Project DINO tokens into LLM embedding space.
      3. Tokenise text prompt and embed via LLM embedding table.
      4. Prepend visual tokens to text token embeddings.
      5. Generate with the LLM using inputs_embeds.
    """
    # 1. Visual features
    dino_feats = encode_image_with_dino(image, dino_processor, dino_model)  # (1, V, D_dino)

    # 2. Project to LLM embedding size
    visual_embeds = proj_head(dino_feats)   # (1, V, D_llm)

    # 3. Text embeddings
    # Phi-3 and most instruction LLMs expect a system+user chat format.
    # We keep it simple: system preamble + user prompt.
    system_msg = "You are a precise gaze-estimation assistant. Analyse the provided image and answer strictly in the requested format."
    formatted  = f"<|system|>\n{system_msg}<|end|>\n<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"

    text_ids = llm_tokenizer(formatted, return_tensors="pt").input_ids.to(DEVICE)
    with torch.inference_mode():
        text_embeds = llm_model.get_input_embeddings()(text_ids)  # (1, T, D_llm)

    # 4. Concatenate: [visual tokens | text tokens]
    combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (1, V+T, D_llm)

    # 5. Generate
    with torch.inference_mode():
        output_ids = llm_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    n_input_tokens = combined_embeds.shape[1]
    # generate() returns token ids starting from position 0; slice from input length
    # Note: when using inputs_embeds, output_ids starts from token 0 of the generation
    response = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Strip the echoed prompt if the model includes it
    if formatted.strip() in response:
        response = response[len(formatted.strip()):].strip()

    return response


# ── LOOKUP ENGINE (unchanged from original) ──────────────────────────────────

def find_dir_by_clip_id(root_path, clip_id):
    target = str(clip_id).strip()
    for root, dirs, _ in os.walk(root_path):
        for d in dirs:
            if d.strip() == target:
                return os.path.join(root, d)
    return None


def parse_txt_file_for_frame(filepath, frame_num):
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
                        "gaze_x":  float(tokens[5]), "gaze_y":  float(tokens[6])
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


# ── RUNTIME ENGINE ───────────────────────────────────────────────────────────

def main():
    if not NEW_MULTIPERSON_OBJECT_FRAMES:
        print("WARNING: NEW_MULTIPERSON_OBJECT_FRAMES is empty — only re-running the")
        print("original step43d6 frame set.\n")

    dino_processor, dino_model, llm_tokenizer, llm_model, proj_head = load_models()

    for clip, frame_num, subject in TARGET_FRAMES:
        print("=" * 80)
        print(f"TARGET: Clip {clip} | Frame {frame_num} | Subject {subject}")

        context = resolve_context(clip, frame_num, subject)
        if not context:
            print("SKIP — Lookup failure processing frame context.")
            continue

        img_raw = Image.open(context["img_path"]).convert("RGB")
        img_w, img_h = img_raw.size

        p = context["primary"]
        primary_box = (int(p["head_x1"]), int(p["head_y1"]), int(p["head_x2"]), int(p["head_y2"]))

        all_subjects = [(subject, p)] + context["others"]
        all_subjects.sort(key=lambda x: x[1]["head_x1"])
        ordered_ids  = [s[0] for s in all_subjects]
        n_total      = len(ordered_ids)

        def get_spatial_label(subj_id):
            idx = ordered_ids.index(subj_id)
            if n_total <= 1: return "the only person in frame"
            if n_total == 2: return ["the person on the left", "the person on the right"][idx]
            if idx == 0:            return "the person on the left"
            if idx == n_total - 1:  return "the person on the right"
            return "the person in the centre"

        primary_label = get_spatial_label(subject)
        other_boxes_with_labels = [
            ((int(s[1]["head_x1"]), int(s[1]["head_y1"]), int(s[1]["head_x2"]), int(s[1]["head_y2"])),
             get_spatial_label(s[0]))
            for s in all_subjects if s[0] != subject
        ]

        gt_x = p["gaze_x"] / img_w if p["gaze_x"] != -1 else None
        gt_y = p["gaze_y"] / img_h if p["gaze_y"] != -1 else None
        is_multiperson = len(other_boxes_with_labels) > 0

        print(f"  Frame File     : {p['fname_actual']}")
        print(f"  Presence Count : {len(all_subjects)} {'people' if is_multiperson else 'person'} found")
        print(f"  Ground Truth   : ({gt_x:.3f}, {gt_y:.3f})" if gt_x is not None else "  Ground Truth   : off-screen")

        for variant_name, build_fn in PROMPT_VARIANTS.items():
            prompt_text = build_fn(img_w, img_h, primary_box, primary_label, other_boxes_with_labels)
            response    = run_inference(
                img_raw, prompt_text,
                dino_processor, dino_model,
                llm_tokenizer, llm_model,
                proj_head,
            )
            pred_x, pred_y = parse_gaze_xy(response)

            if pred_x == -1.0 and pred_y == -1.0:
                coord_str = "off-screen (-1, -1)"
            elif pred_x is not None and pred_y is not None:
                coord_str = f"({pred_x:.3f}, {pred_y:.3f})"
            else:
                coord_str = "PARSING ERROR"

            print(f"\n    [{variant_name}] PRED COORDS : {coord_str}")
            print(f"    [{variant_name}] MODEL TEXT  : {response.replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
