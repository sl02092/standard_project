"""
step43g_prompt_compare_molmo.py — AllenAI Molmo-7B-D variant of the prompt 
comparison harness. Reuses target frames and tracking structures but 
overhauls processing, caching layers, and coordinate extraction to match 
Molmo's native pointing architecture and modern transformers versions.

Includes side-by-side comparison with dataset Ground Truth targets.

Requires:
    pip install transformers bitsandbytes accelerate torchvision pillow
"""

import os
import re
import torch
from PIL import Image
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig

# ── TRANSFORMERS GLOBAL COMPATIBILITY PATCHES ─────────────────────────────

# 1. Patch missing 'all_tied_weights_keys' lookups on custom modeling classes
class TiedWeightsKeysDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return {}
        return instance.__dict__.get("_all_tied_weights_keys_internal", {})
        
    def __set__(self, instance, value):
        instance.__dict__["_all_tied_weights_keys_internal"] = value

torch.nn.Module.all_tied_weights_keys = TiedWeightsKeysDescriptor()

# 2. Patch the tie_weights() keyword argument signature mismatch
orig_finalize = transformers.PreTrainedModel._finalize_model_loading

def patched_finalize(*args, **kwargs):
    for arg in args:
        if hasattr(arg, "tie_weights") and not hasattr(arg, "_patched_gaze_tie_weights"):
            orig_tie_weights = arg.tie_weights
            def safe_tie_weights(*a, **kw):
                try:
                    return orig_tie_weights(*a, **kw)
                except TypeError:
                    return orig_tie_weights()
            arg.tie_weights = safe_tie_weights
            arg._patched_gaze_tie_weights = True
            break
    return orig_finalize(*args, **kwargs)

transformers.PreTrainedModel._finalize_model_loading = patched_finalize

# 3. Patch DynamicCache to support legacy tuple indexing (fixes 'DynamicCache' object is not subscriptable)
try:
    from transformers.cache_utils import DynamicCache
    def dynamic_cache_getitem(self, key):
        if isinstance(key, int):
            # Version check: support newer transformers (v4.47+) layers attribute
            if hasattr(self, "layers"):
                return (self.layers[key].keys, self.layers[key].values)
            # Fallback for older transformers versions
            return (self.key_cache[key], self.value_cache[key])
        raise TypeError(f"DynamicCache indices must be integers, not {type(key).__name__}")
    DynamicCache.__getitem__ = dynamic_cache_getitem
except (ImportError, AttributeError):
    pass
# ───────────────────────────────────────────────────────────────────────

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_PATH      = r"C:\repo\standard_project\videoattentiontarget"
MODEL_NAME     = "allenai/Molmo-7B-D-0924"
MAX_NEW_TOKENS = 150

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

PROMPT_VARIANTS = {
    "V1_Direct_Point": "Point to the exact object or location that the person who is {spatial_label} is looking at.",
    "V2_Gaze_Follow": "Trace the line of sight vector from the eyes of the subject who is {spatial_label}. Point to the target of their gaze.",
    "V3_OffScreen_Rule": "Point to what the person ({spatial_label}) is looking at. If they are looking completely off-screen or out of the frame, reply exactly with the text 'off-screen' and do not provide a point."
}

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
def parse_molmo_gaze_xy(response_str):
    if "off-screen" in response_str.lower():
        return -1.0, -1.0

    match = re.search(r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"', response_str)
    if match:
        try:
            raw_x = float(match.group(1))
            raw_y = float(match.group(2))
            
            # Normalize from Molmo's native 0-100 scale down to standard 0.0-1.0
            norm_x = raw_x / 100.0
            norm_y = raw_y / 100.0
            return norm_x, norm_y
        except ValueError:
            return None, None
            
    return None, None

# ── INITIALIZE MODEL ────────────────────────────────────────────────────
print(f"[*] Initializing {MODEL_NAME} in 4-bit precision for local GPU...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

# ── MODEL INSTANCE-SPECIFIC COMPATIBILITY PATCHES ─────────────────────────

# A. Intercept uninitialized/empty caches during the initial prefill stage
original_prepare = model.prepare_inputs_for_generation

def patched_prepare(input_ids, past_key_values=None, **kwargs):
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
            past_key_values = None
        elif hasattr(past_key_values, "layers") and (len(past_key_values.layers) == 0 or getattr(past_key_values.layers[0], "keys", None) is None):
            past_key_values = None
            
    return original_prepare(input_ids, past_key_values=past_key_values, **kwargs)

model.prepare_inputs_for_generation = patched_prepare


# B. Workaround internal keyword update signatures across library version drifts
original_update_kwargs = model._update_model_kwargs_for_generation

def patched_update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs):
    try:
        return original_update_kwargs(outputs, model_kwargs, **kwargs)
    except Exception:
        if "image_input_idx" in model_kwargs:
            del model_kwargs["image_input_idx"]
        
        model_kwargs["past_key_values"] = getattr(outputs, "past_key_values", None)
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
            
        return model_kwargs

model._update_model_kwargs_for_generation = patched_update_model_kwargs_for_generation

print("[+] Model and runtime engines loaded successfully. Starting loop...\n")

# ── EVALUATION LOOP ─────────────────────────────────────────────────────
# Redesigned header layout to show side-by-side tracking
print(f"{'Frame / Subject':<28} | {'Variant':<18} | {'Ground Truth':<18} | {'Predicted Target':<18} | {'Raw Text Output'}")
print("-" * 130)

for seq_dir, frame_idx, subject_id in TARGET_FRAMES:
    context = resolve_context(seq_dir, frame_idx, subject_id)
    if not context:
        print(f"[-] Lookup failure processing frame context for Clip {seq_dir} | Frame {frame_idx}")
        continue
        
    img = Image.open(context["img_path"]).convert("RGB")
    p = context["primary"]
    
    # Format Dataset Ground Truth string (-1, -1 implies off-screen in target datasets)
    gt_x = p["gaze_x"]
    gt_y = p["gaze_y"]
    if gt_x < 0 or gt_y < 0:
        gt_str = "off-screen (-1, -1)"
    else:
        gt_str = f"({gt_x:.3f}, {gt_y:.3f})"
    
    # Generate screen space layout anchors
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
    frame_label = f"{seq_dir}/{frame_idx} ({subject_id})"

    for var_name, base_prompt in PROMPT_VARIANTS.items():
        prompt_text = base_prompt.format(subject_id=subject_id, spatial_label=primary_label)
        
        inputs = processor.process(images=[img], text=prompt_text)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        with torch.inference_mode():
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=MAX_NEW_TOKENS, 
                    stop_strings=["<|endoftext|>"],
                    use_cache=True,
                    return_legacy_cache=True
                ),
                tokenizer=processor.tokenizer
            )
            
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        pred_x, pred_y = parse_molmo_gaze_xy(response)
        
        if pred_x == -1.0 and pred_y == -1.0:
            coord_str = "off-screen (-1, -1)"
        elif pred_x is not None and pred_y is not None:
            coord_str = f"({pred_x:.3f}, {pred_y:.3f})"
        else:
            coord_str = "PARSING ERROR"
            
        clean_response = response.replace('\n', ' | ')
        print(f"{frame_label:<28} | {var_name:<18} | {gt_str:<18} | {coord_str:<18} | {clean_response}")