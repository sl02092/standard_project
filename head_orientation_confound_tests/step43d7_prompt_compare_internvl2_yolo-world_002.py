import os
import re
import math
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from ultralytics import YOLOWorld

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_PATH = r"C:\repo\standard_project\videoattentiontarget"
VIDEO_VLM_NAME = "OpenGVLab/InternVL2-8B" 
YOLO_WORLD_NAME = "yolov8l-worldv2.pt"
NUM_CONTEXT_FRAMES = 8
FRAME_STRIDE = 2

EVAL_OBJECT_FRAMES = [
    ("13525_13575", 13570, "s00", 0.412, 0.615),
    ("2250_2300",   2295,  "s00", 0.784, 0.312),
]

def load_models():
    print(f"--> Loading Video VLM ({VIDEO_VLM_NAME})...")
    processor = AutoProcessor.from_pretrained(VIDEO_VLM_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(VIDEO_VLM_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        VIDEO_VLM_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).eval().cuda()
    
    print("--> Loading YOLO-World...")
    spatial_engine = YOLOWorld(YOLO_WORLD_NAME).to('cuda')
    return spatial_engine, tokenizer, model, processor

def extract_temporal_sequence(clip_dir, target_frame_num):
    base_images_path = Path(BASE_PATH) / "images"
    found_paths = list(base_images_path.rglob(clip_dir))
    
    if not found_paths:
        raise FileNotFoundError(f"Could not find clip directory '{clip_dir}'")
        
    clip_path = found_paths[0]
    frame_map = {}
    for f in clip_path.glob("*.jpg"):
        try:
            frame_map[int(f.stem)] = f
        except ValueError:
            continue
            
    target_int = int(target_frame_num)
    if target_int not in frame_map:
        raise FileNotFoundError(f"Frame {target_int} not found in {clip_path}")

    sorted_frames = sorted(frame_map.keys())
    target_idx = sorted_frames.index(target_int)
    
    sampled_indices = [sorted_frames[target_idx - (i * FRAME_STRIDE)] 
                       for i in range(NUM_CONTEXT_FRAMES) 
                       if (target_idx - (i * FRAME_STRIDE)) >= 0]
    sampled_indices.reverse()
    
    return [Image.open(frame_map[f_num]).convert("RGB") for f_num in sampled_indices]

def run_video_grounding_inference(spatial_engine, tokenizer, model, processor, frames, anchor_img, subject_desc):
    prompt = f"Focus on {subject_desc}. What object are they looking at? Provide a single noun phrase."
    
    # FIX: Bypass the main processor() call and use the dedicated image processor
    # InternVLProcessor has a 'image_processor' attribute that handles this correctly.
    #pixel_values = processor.image_processor(
    #    images=frames, 
    #    return_tensors='pt'
    #)['pixel_values'].to(torch.bfloat16).cuda()
    # Final fallback if image_processor attribute is missing
    pixel_values = processor.preprocess(images=frames, return_tensors='pt')['pixel_values'].to(torch.bfloat16).cuda()
    
    with torch.inference_mode():
        response = model.chat(tokenizer, pixel_values, f"{''.join(['<img>']*len(frames))}\n{prompt}", 
                              generation_config={"max_new_tokens": 30})
    
    cleaned_class = re.sub(r'[^\w\s-]', '', str(response)).strip().lower() or "object"
    print(f"   [VLM Extraction] Target: '{cleaned_class}'")
    
    # 2. YOLO-World Grounding
    spatial_engine.model.to('cuda')
    results = spatial_engine.predict(source=anchor_img, text=[cleaned_class], verbose=False)[0]
    
    img_w, img_h = anchor_img.size
    best_coords, highest_conf = None, -1.0
    
    for box in results.boxes:
        if box.conf[0].item() > highest_conf:
            highest_conf = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()
            best_coords = (((xyxy[0] + xyxy[2]) / 2.0) / img_w, ((xyxy[1] + xyxy[3]) / 2.0) / img_h)
                
    return cleaned_class, best_coords, highest_conf

def main():
    spatial_engine, tokenizer, model, processor = load_models()
    total_error, valid_evals = 0.0, 0
    
    for clip, frame, sid, gt_x, gt_y in EVAL_OBJECT_FRAMES:
        print(f"\nEvaluating: {clip} | Frame {frame}")
        try:
            frames = extract_temporal_sequence(clip, frame)
            _, coords, _ = run_video_grounding_inference(spatial_engine, tokenizer, model, processor, frames, frames[-1], f"subject {sid}")
            
            if coords:
                err = math.sqrt((coords[0] - gt_x)**2 + (coords[1] - gt_y)**2)
                print(f"   [Result] Error: {err:.4f}")
                total_error += err
            else:
                err = math.sqrt((0.5 - gt_x)**2 + (0.5 - gt_y)**2)
                print(f"   [Result] No boxes, fallback error: {err:.4f}")
                total_error += err
            valid_evals += 1
        except Exception as e:
            print(f"  SKIPPING FRAME: {e}")
            import traceback; traceback.print_exc() # Added to help see if errors persist
            
    print(f"\nMean Error: {total_error / valid_evals:.4f}" if valid_evals > 0 else "No evaluations completed.")

if __name__ == "__main__":
    main()