import re
import math
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from ultralytics import YOLO

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_PATH = r"C:\repo\standard_project\videoattentiontarget"
VIDEO_VLM_NAME = "OpenGVLab/InternVL2-8B"
YOLO_WORLD_NAME = "yolov8l-worldv2.pt"
NUM_CONTEXT_FRAMES = 8
FRAME_STRIDE = 2

# (clip_dir, target_frame, subject_id, gt_x, gt_y)
EVAL_OBJECT_FRAMES = [
    ("13525_13575", 13570, "s00", 0.412, 0.615),
    ("2250_2300",   2295,  "s00", 0.784, 0.312),
]


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
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
    spatial_engine = YOLO(YOLO_WORLD_NAME).to("cuda")

    return spatial_engine, tokenizer, model, processor


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def find_clip_path(clip_dir: str) -> Path:
    base_images_path = Path(BASE_PATH) / "images"
    found_paths = list(base_images_path.rglob(clip_dir))

    if not found_paths:
        raise FileNotFoundError(f"Could not find clip directory '{clip_dir}' under {base_images_path}")

    return found_paths[0]


def extract_temporal_sequence(clip_dir: str, target_frame_num: int):
    clip_path = find_clip_path(clip_dir)

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

    sampled_indices = [
        sorted_frames[target_idx - (i * FRAME_STRIDE)]
        for i in range(NUM_CONTEXT_FRAMES)
        if (target_idx - (i * FRAME_STRIDE)) >= 0
    ]
    sampled_indices.reverse()

    return [Image.open(frame_map[f_num]).convert("RGB") for f_num in sampled_indices]


# ── VIDEO GROUNDING ───────────────────────────────────────────────────────────
def run_video_grounding_inference(
    spatial_engine,
    tokenizer,
    model,
    processor,
    frames,
    anchor_img,
    subject_desc: str
):
    # 1. Build prompt for InternVL2
    prompt = (
        f"You are given a sequence of video frames. "
        f"Identify the object referred to as '{subject_desc}'. "
        f"Return ONLY the object class name."
    )

    # 2. Run InternVL2 using its custom chat interface
    # InternVL2 expects `image=` (can be a list of PIL images), not `images=`
    response = model.chat(
        tokenizer=tokenizer,
        image=frames,
        query=prompt
    )

    # 3. Clean the class name
    cleaned_class = re.sub(r"[^\w\s-]", "", str(response)).strip().lower()
    if not cleaned_class:
        cleaned_class = "object"

    print(f"   [VLM Extraction] Target: '{cleaned_class}'")

    # 4. YOLO-World inference
    spatial_engine.model.to("cuda")

    results = spatial_engine.predict(
        source=anchor_img,
        text=[cleaned_class],
        verbose=False
    )[0]

    img_w, img_h = anchor_img.size
    best_coords, highest_conf = None, -1.0

    for box in results.boxes:
        conf = box.conf[0].item()
        if conf > highest_conf:
            highest_conf = conf
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best_coords = (
                ((x1 + x2) / 2.0) / img_w,
                ((y1 + y2) / 2.0) / img_h
            )

    return cleaned_class, best_coords, highest_conf


# ── EVALUATION UTILITIES ──────────────────────────────────────────────────────
def compute_error(pred_coords, gt_x, gt_y):
    if pred_coords is None:
        pred_x, pred_y = 0.5, 0.5
    else:
        pred_x, pred_y = pred_coords

    return math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    spatial_engine, tokenizer, model, processor = load_models()
    total_error, valid_evals = 0.0, 0

    for clip, frame, sid, gt_x, gt_y in EVAL_OBJECT_FRAMES:
        print(f"\nEvaluating: {clip} | Frame {frame}")
        try:
            frames = extract_temporal_sequence(clip, frame)
            subject_desc = f"subject {sid}"

            _, coords, conf = run_video_grounding_inference(
                spatial_engine,
                tokenizer,
                model,
                processor,
                frames,
                frames[-1],
                subject_desc
            )

            err = compute_error(coords, gt_x, gt_y)
            if coords is not None:
                print(f"   [Result] Conf: {conf:.4f} | Error: {err:.4f}")
            else:
                print(f"   [Result] No boxes, fallback error: {err:.4f}")

            total_error += err
            valid_evals += 1

        except Exception as e:
            print(f"  SKIPPING FRAME: {e}")

    if valid_evals > 0:
        print(f"\nMean Error: {total_error / valid_evals:.4f}")
    else:
        print("No evaluations completed.")


if __name__ == "__main__":
    main()
