"""
step43d7_prompt_compare_dinov2_internvl38b_003_diag.py — MEMORIZATION DIAGNOSTIC.

Same hybrid DINOv2+InternVL3 eval as step43d7_prompt_compare_dinov2_internvl38b_003.py,
but with two additions, purely to answer one question in <30 min with NO retraining:
is the projector grounding gaze spatially, or has it memorized a constant output keyed
to gaze_type (e.g. always emitting (-1,-1) for off-screen regardless of image content)?

WHY THIS MATTERS: the original 150-sample local test showed V2_Gaze_Follow predicting
EXACTLY (-1.000, -1.000) on every single off-screen-GT frame and NEVER on any on-screen
frame -- zero variance within a group that should show real spatial variation even when
the label is constant (different head poses still produce different gaze vectors).
That pattern is the signature of label-memorization, not grounding.

ADDITIONS (no other logic changed):
  1. TARGET_FRAMES extended with multi-person OBJECT-gaze frames pulled from
     frame_manifest.csv (gaze_type=="object", use_teacher==True, n_subjects>=2) --
     the same diverse set picked out for the step43d7 prompt-comparison work,
     reused here so this run produces directly comparable evidence.
  2. print_variance_report() at the end: groups predictions by GT gaze_type bucket
     and reports the stdev of predicted (x,y) within each bucket. Near-zero stdev
     within a bucket of genuinely different images = memorization red flag.
     Real spatial grounding should show non-trivial within-bucket variance.
"""

import os
import re
import statistics
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, AutoModel

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH       = os.environ.get("VAT_BASE_PATH", r"C:\repo\standard_project\videoattentiontarget")
DINO_MODEL_NAME = "facebook/dinov2-large"
LLM_MODEL_NAME  = "OpenGVLab/InternVL3-8B-hf"
MAX_NEW_TOKENS  = 50

# Original step43d7 off-screen/social frames (kept for direct comparison against
# the original 150-sample console output already reviewed).
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

# NEW: multi-person object-gaze frames, pulled from frame_manifest.csv earlier
# this session (gaze_type=="object", use_teacher==True, n_subjects>=2, deduped
# by clip+subject, spread across shows and group sizes). Added here so the
# memorization check covers the actual production bottleneck category too,
# not just the off-screen frames the original 150-sample run happened to use.
NEW_OBJECT_GAZE_FRAMES = [
    ("17742_17893", 17826, "s00"),
    ("10239_10740", 10270, "s01"),
    ("14250_14430", 14250, "s01"),
    ("19636_19829", 19658, "s00"),
    ("1710_1890",   1737,  "s02"),
    ("1348_1469",   1382,  "s00"),
]

TARGET_FRAMES = ORIGINAL_TARGET_FRAMES + NEW_OBJECT_GAZE_FRAMES

PROMPT_VARIANTS = {
    "V1_Direct": lambda w, h, box, lbl, others: f"Look at the person who is {lbl} at bounding box {box}. Predict their normalized gaze target coordinate as (X, Y) between 0.0 and 1.0.",
    "V2_Gaze_Follow": lambda w, h, box, lbl, others: f"Trace the line of sight vector from the eyes of {lbl}. Provide the coordinate (X, Y) where their gaze lands.",
}

# ── MULTIMODAL PROJECTION ADAPTER LAYER ─────────────────────────────────
class GazeMultimodalProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)  
        )
        
    def forward(self, x):
        return self.projector(x)

# ── DATASET LOOKUP ENGINE ───────────────────────────────────────────────
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

# ── PARSING LOGIC ───────────────────────────────────────────────────────
def parse_gaze_xy(response_str):
    if "off-screen" in response_str.lower():
        return -1.0, -1.0
    match = re.search(r"([\d.]+)\s*,\s*([\d.]+)", response_str)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            pass
    return None, None

# ── HYBRID INFERENCE PIPELINE ───────────────────────────────────────────
def run_custom_hybrid_inference(img_raw, prompt_text, dino_processor, dino_model, llm_tokenizer, llm_model, full_vlm, proj_head):
    # 1. Extract visual features using DINOv2
    inputs_dino = dino_processor(images=img_raw, return_tensors="pt").to(dino_model.device)
    with torch.inference_mode():
        dino_outputs = dino_model(**inputs_dino)
        image_features = dino_outputs.last_hidden_state[:, 1:, :]
        
    # 2. Map visual features into the LLM's embedding space dimension via Projection Head
    projected_vis_embeds = proj_head(image_features.to(torch.bfloat16))
    
    # 3. Process structural context bounds and prompt text
    start_tokens = llm_tokenizer("Image Context: [START_VISION] ", return_tensors="pt").to(llm_model.device)
    end_tokens   = llm_tokenizer(" [END_VISION]\nOperational Prompt: ", return_tensors="pt").to(llm_model.device)
    text_inputs  = llm_tokenizer(prompt_text, return_tensors="pt").to(llm_model.device)
    
    embed_layer = llm_model.get_input_embeddings() if hasattr(llm_model, "get_input_embeddings") else llm_model.model.embed_tokens
    
    start_embeds = embed_layer(start_tokens.input_ids)
    end_embeds   = embed_layer(end_tokens.input_ids)
    text_embeds  = embed_layer(text_inputs.input_ids)
        
    # 4. Concatenate sequence mapping using the Forced Attention Sandwich Layout
    inputs_embeds = torch.cat([
        start_embeds,          # Boundary head text anchor
        projected_vis_embeds,  # Dense visual features
        end_embeds,            # Boundary tail text anchor
        text_embeds            # Instructions prompt
    ], dim=1).to(torch.bfloat16)
    
    # 5. Step-by-step autoregressive generation using the top-level VLM language head
    generated_ids = []
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            outputs = llm_model(inputs_embeds=inputs_embeds)
            hidden_states = outputs.last_hidden_state
            
            # Map the final hidden state token representation through the lm_head
            next_token_logits = full_vlm.lm_head(hidden_states[:, -1, :])
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            generated_ids.append(next_token_id.item())
            
            if next_token_id.item() == llm_tokenizer.eos_token_id:
                break
                
            # Append the newly predicted token to the embedding sequence stream
            next_token_embed = embed_layer(next_token_id.unsqueeze(0))
            inputs_embeds = torch.cat([inputs_embeds, next_token_embed], dim=1)
        
    return llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# ── MEMORIZATION DIAGNOSTIC ──────────────────────────────────────────────
def print_variance_report(results):
    """
    Groups predictions by (variant, gt_bucket) and reports stdev of predicted
    (x, y) within each group, plus how many distinct (x,y) pairs appeared.

    HOW TO READ THIS:
    - These frames are NOT the same image -- different clips, different
      subjects, different scenes. If the model is actually grounding gaze
      spatially, predictions within a bucket should vary noticeably, even
      if most of them land in roughly the same region.
    - If a bucket shows stdev near 0.0 AND only 1 distinct (x,y) pair across
      multiple different images, that's the memorization signature already
      seen in the original 150-sample run (every off-screen-GT frame
      predicting exactly (-1.000, -1.000) under V2_Gaze_Follow).
    - If stdev is meaningfully non-zero and distinct pairs > 1, that's at
      least consistent with the model responding to image content -- it
      doesn't prove correctness, but it rules out the simplest and most
      damaging failure mode (constant-output memorization).
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for variant, bucket, px, py in results:
        if px is None or py is None:
            continue
        groups[(variant, bucket)].append((px, py))

    print("\n" + "=" * 80)
    print("MEMORIZATION / GROUNDING DIAGNOSTIC")
    print("=" * 80)
    print(f"{'Variant':<18} | {'GT Bucket':<10} | {'N':<4} | {'Distinct':<9} | {'StdevX':<8} | {'StdevY':<8} | Verdict")
    print("-" * 80)

    for (variant, bucket), preds in sorted(groups.items()):
        n = len(preds)
        distinct = len(set(preds))
        if n < 2:
            verdict = "too few samples"
            sx = sy = float("nan")
        else:
            xs = [p[0] for p in preds]
            ys = [p[1] for p in preds]
            sx = statistics.stdev(xs)
            sy = statistics.stdev(ys)
            if distinct == 1 and n >= 3:
                verdict = "MEMORIZATION RED FLAG (constant output)"
            elif sx < 0.01 and sy < 0.01:
                verdict = "suspiciously low variance"
            else:
                verdict = "shows variance (not proof of correctness)"
        print(f"{variant:<18} | {bucket:<10} | {n:<4} | {distinct:<9} | {sx:<8.4f} | {sy:<8.4f} | {verdict}")

    print("=" * 80)
    print("NOTE: this checks ONLY whether output varies with image content within")
    print("a GT bucket. It does NOT confirm the predictions are spatially accurate")
    print("-- that's the separate, already-known ADE/distance problem. A model can")
    print("pass this check (real variance) and still be inaccurate. But a model")
    print("that FAILS this check (near-constant output) cannot be trusted at all,")
    print("regardless of how good its loss curve or ADE number looks.")
    print("=" * 80 + "\n")


# ── MAIN EXECUTIVE ──────────────────────────────────────────────────────
def main():
    print(f"[*] Loading Vision Backbone: {DINO_MODEL_NAME}...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")

    print(f"[*] Loading VLM to extract language decoder: {LLM_MODEL_NAME}...")
    full_vlm = AutoModelForImageTextToText.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    llm_model = full_vlm.model.language_model
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    
    dino_dim = dino_model.config.hidden_size      
    llm_dim = llm_model.config.hidden_size        
    
    proj_head = GazeMultimodalProjector(dino_dim, llm_dim).to("cuda:0", dtype=torch.bfloat16)
    
    # Automatically load the trained weights matrix saved from train_step43d7_hybrid_003.py
    weight_path = os.path.join(BASE_PATH, "dino_internvl_projector.pt")
    if os.path.exists(weight_path):
        print(f"[+] Loading trained projection weights matrix from: {weight_path}")
        proj_head.load_state_dict(torch.load(weight_path, map_location="cuda:0"))
    else:
        print(f"[-] WARNING: Trained projection weight file not found at {weight_path}. Running with uninitialized states.")
        
    proj_head.eval().requires_grad_(False)
    dino_model.eval().requires_grad_(False)
    llm_model.eval().requires_grad_(False)

    print("\n[+] Hybrid Engine loaded successfully. Running evaluation...\n")
    print(f"{'Frame / Subject':<28} | {'Variant':<15} | {'GT (Norm)':<18} | {'Pred (Norm)':<18} | {'Output String'}")
    print("-" * 130)

    # Collected for the post-run memorization/variance diagnostic. Each entry:
    # (variant_name, gt_bucket, pred_x, pred_y). gt_bucket is "offscreen" or "onscreen"
    # rather than gaze_type, since gaze_type isn't in primary_data here -- but the
    # original 150-sample run's suspicious pattern was specifically about the
    # offscreen/onscreen split, so that's the bucketing that matters for this check.
    diagnostic_results = []

    for seq_dir, frame_idx, subject_id in TARGET_FRAMES:
        context = resolve_context(seq_dir, frame_idx, subject_id)
        if not context:
            continue
            
        img_raw = Image.open(context["img_path"]).convert("RGB")
        img_w, img_h = img_raw.size
        p = context["primary"]
        
        primary_box = (int(p["head_x1"]), int(p["head_y1"]), int(p["head_x2"]), int(p["head_y2"]))
        
        if p["gaze_x"] != -1 and p["gaze_y"] != -1:
            gt_str = f"({p['gaze_x']/img_w:.3f}, {p['gaze_y']/img_h:.3f})"
        else:
            gt_str = "off-screen"
            
        all_subjects = [(subject_id, p)] + context["others"]
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

        primary_label = get_spatial_label(subject_id)
        other_boxes_with_labels = [
            ((int(s[1]["head_x1"]), int(s[1]["head_y1"]), int(s[1]["head_x2"]), int(s[1]["head_y2"])), get_spatial_label(s[0]))
            for s in all_subjects if s[0] != subject_id
        ]

        frame_label = f"{seq_dir}/{frame_idx} ({subject_id})"

        for variant_name, build_fn in PROMPT_VARIANTS.items():
            prompt_text = build_fn(img_w, img_h, primary_box, primary_label, other_boxes_with_labels)
            
            response = run_custom_hybrid_inference(
                img_raw, prompt_text,
                dino_processor, dino_model,
                llm_tokenizer, llm_model, full_vlm,
                proj_head
            )
            
            pred_x, pred_y = parse_gaze_xy(response)
            pred_str = f"({pred_x:.3f}, {pred_y:.3f})" if pred_x is not None else "Unparsed Text"

            clean_res = response.replace("\n", " ")
            print(f"{frame_label:<28} | {variant_name:<15} | {gt_str:<18} | {pred_str:<18} | {clean_res}")

            gt_bucket = "offscreen" if gt_str == "off-screen" else "onscreen"
            diagnostic_results.append((variant_name, gt_bucket, pred_x, pred_y))

    print_variance_report(diagnostic_results)

if __name__ == "__main__":
    main()