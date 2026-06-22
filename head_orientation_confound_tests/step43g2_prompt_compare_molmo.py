import torch
import torch.nn as nn
import os
import re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# --- 1. CRITICAL PATCHES FOR COMPATIBILITY ---
# Patch nn.Module to prevent Accelerate from crashing on 'all_tied_weights_keys'
class PatchedModule(nn.Module):
    @property
    def all_tied_weights_keys(self):
        return getattr(self, "_tied_weights_keys", [])

nn.Module.all_tied_weights_keys = PatchedModule.all_tied_weights_keys

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_PATH      = r"C:\repo\standard_project\videoattentiontarget"
MODEL_NAME     = "allenai/Molmo-7B-D-0924"
MAX_NEW_TOKENS = 50

# Define your frames here
TARGET_FRAMES = [
    ("13525_13575", 13567, "s00"),
    ("1650_1775", 1673, "s00"),
]

# ── HELPERS ─────────────────────────────────────────────────────────────
def resolve_context(clip, frame_num, subject):
    """Update this to match your local file structure."""
    img_dir = os.path.join(BASE_PATH, "images", clip)
    return {
        "img_path": os.path.join(img_dir, f"{frame_num}.jpg"), 
        "primary": {"gaze_x": 423.0, "gaze_y": 639.0} # Placeholder: Replace with real parser
    }

def parse_molmo_gaze_xy(response_str):
    match = re.search(r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"', response_str)
    if match:
        # Molmo returns values 0-100; normalize to 0-1
        return float(match.group(1)) / 100.0, float(match.group(2)) / 100.0
    return None, None

def main():
    # 2. Load Processor and Model
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 3. PATCH: Bypass the incompatible tie_weights method
    model.tie_weights = lambda: None 

    print(f"{'Frame':<15} | {'GT (Norm)':<15} | {'Pred (Norm)':<15} | {'Status'}")
    print("-" * 60)

    for clip, frame, subject in TARGET_FRAMES:
        context = resolve_context(clip, frame, subject)
        if not os.path.exists(context["img_path"]): continue
        
        img = Image.open(context["img_path"]).convert("RGB")
        img_w, img_h = img.size
        p = context["primary"]

        # A. Normalize Ground Truth (Pixel to 0-1)
        if p["gaze_x"] != -1:
            gt_x, gt_y = round(p["gaze_x"] / img_w, 3), round(p["gaze_y"] / img_h, 3)
            gt_str = f"({gt_x:.3f}, {gt_y:.3f})"
        else:
            gt_str = "off-screen"

        # B. Inference
        prompt = "Trace the line of sight vector from the eyes of the subject. Point to the target of their gaze."
        inputs = processor.process(images=[img], text=prompt).to(model.device).unsqueeze(0)
        
        with torch.inference_mode():
            output = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=MAX_NEW_TOKENS), tokenizer=processor.tokenizer)
            response = processor.tokenizer.decode(output[0, inputs['input_ids'].size(1):], skip_special_tokens=True)

        # C. Parse and Compare
        pred_x, pred_y = parse_molmo_gaze_xy(response)
        pred_str = f"({pred_x:.3f}, {pred_y:.3f})" if pred_x is not None else "off-screen"
        
        status = "Match" if gt_str == pred_str else "Diff"
        print(f"{str(frame):<15} | {gt_str:<15} | {pred_str:<15} | {status}")

if __name__ == "__main__":
    main()