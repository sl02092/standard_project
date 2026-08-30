"""
gazelle_zero_shot_vat_eval.py — scores Gaze-LLE's zero-shot (GazeFollow-
only, no VAT fine-tuning) checkpoint against the SAME VAT _TEST
population used everywhere else in this project, for a directly
comparable ADE number.

RUN LOCALLY, in your Gaze-LLE conda environment -- not part of the
HPC/Apptainer pipeline at all, no torch/timm container needed.

CHECKPOINT: use gazelle_dinov2_vitb14.pt (or vitl14) -- the plain
GazeFollow-only checkpoint, NOT the "_inout" variant, which IS
fine-tuned on VideoAttentionTarget and would not be a genuine zero-shot
comparison. Confirmed from the official README's own checkpoint table.

POPULATION: labels_vat_full_TEST.jsonl ALONE is sufficient -- confirmed
directly against this project's own eval_static_v1.py run logs (job
735918) that VAT's three-way join (full/hybrid/mtgs) already has 100%
key alignment (26,250/26,250), so this file's own population already
equals the joined population used for every other VAT result in this
project. No need to load the other two label files at all.

PATH REMAP: img_path in the label file is an HPC absolute path, baked in
at generation time -- same category of issue as ChildPlay/VACATION's
Windows-path problem, just running in the opposite direction (HPC path
needing to become a real local one). Confirmed HPC_PREFIX directly
against a real row in the uploaded file.

OFFSCREEN: excluded, same convention as every other ADE number in this
project -- no real target to compare a prediction against.

GT COORDINATES: uses the label file's own already-normalized gt_x/gt_y
directly, not re-derived from gt_px_x/width -- avoids any risk of a
rounding mismatch with how gt_x was originally computed.

ARGMAX-TO-POINT: the +0.5 grid-cell-center offset matches the confirmed,
working convention from this project's own earlier six_frame_evaluation.py.
"""

import os
import json
import math

import torch
from PIL import Image

from gazelle.model import get_gazelle_model

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

LABELS_PATH = os.environ.get("VAT_LABELS_PATH", r"C:\repo\gazelle\dataset\vat\labels_vat_full_TEST.jsonl")
CHECKPOINT_PATH = os.environ.get("GAZELLE_CHECKPOINT_PATH", r"C:\repo\gazelle\checkpoints\gazelle_dinov2_vitb14.pt")
MODEL_NAME = os.environ.get("GAZELLE_MODEL_NAME", "gazelle_dinov2_vitb14")

# Confirmed directly against a real row in labels_vat_full_TEST.jsonl.
HPC_PREFIX = "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget/images"
# Confirmed directly by you -- NOT guessed (my first guess, .../dataset/vat/images, was wrong).
LOCAL_PREFIX = os.environ.get("VAT_LOCAL_IMAGES_ROOT", r"C:\repo\gazelle\dataset\videoattentiontarget\images")

OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "gazelle_vat_zeroshot_results.json")

HEATMAP_SIZE = 64


def resolve_local_path(hpc_path):
    normalized = hpc_path.replace("\\", "/")
    if not normalized.startswith(HPC_PREFIX):
        raise ValueError(f"img_path '{hpc_path}' doesn't start with expected "
                          f"HPC prefix '{HPC_PREFIX}' -- check HPC_PREFIX is still correct.")
    rel = normalized[len(HPC_PREFIX):].lstrip("/")
    return os.path.join(LOCAL_PREFIX, *rel.split("/"))


def load_population():
    records = []
    n_offscreen = 0
    n_missing = 0
    with open(LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["gaze_type"] == "offscreen":
                n_offscreen += 1
                continue
            local_path = resolve_local_path(row["img_path"])
            if not os.path.isfile(local_path):
                n_missing += 1
                continue
            records.append({
                "img_path": local_path,
                "head_x1": row["head_x1"], "head_y1": row["head_y1"],
                "head_x2": row["head_x2"], "head_y2": row["head_y2"],
                "gt_x": row["gt_x"], "gt_y": row["gt_y"],  # already normalized -- use directly
                "gaze_type": row["gaze_type"],
            })
    print(f"Loaded {len(records)} on-screen records "
          f"(excluded {n_offscreen} offscreen, {n_missing} missing images).")
    if n_missing > 0:
        print(f"  WARNING: {n_missing} missing images -- if this is more than a "
              f"handful, VAT_LOCAL_IMAGES_ROOT is probably wrong, not just a few "
              f"genuinely absent files.")
    return records


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} from {CHECKPOINT_PATH}...")
    model, transform = get_gazelle_model(MODEL_NAME)
    model.load_gazelle_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.to(device)
    model.eval()

    records = load_population()
    if len(records) < 10:
        print("ERROR: too few records loaded -- check VAT_LOCAL_IMAGES_ROOT before trusting this.")
        return

    dists = []
    by_type = {}

    for i, rec in enumerate(records):
        img = Image.open(rec["img_path"]).convert("RGB")
        w, h = img.size

        norm_head = (
            max(0.0, min(1.0, rec["head_x1"] / w)),
            max(0.0, min(1.0, rec["head_y1"] / h)),
            max(0.0, min(1.0, rec["head_x2"] / w)),
            max(0.0, min(1.0, rec["head_y2"] / h)),
        )

        model_input = {
            "images": transform(img).unsqueeze(0).to(device),
            "bboxes": [[norm_head]],
        }
        with torch.no_grad():
            output = model(model_input)
        heatmap = output["heatmap"][0][0]  # [64, 64]

        argmax_idx = torch.argmax(heatmap).item()
        row_idx, col_idx = argmax_idx // HEATMAP_SIZE, argmax_idx % HEATMAP_SIZE
        pred_x = (col_idx + 0.5) / HEATMAP_SIZE
        pred_y = (row_idx + 0.5) / HEATMAP_SIZE

        dist = math.hypot(pred_x - rec["gt_x"], pred_y - rec["gt_y"])
        dists.append(dist)
        by_type.setdefault(rec["gaze_type"], []).append(dist)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(records)} done, running ADE={sum(dists)/len(dists):.4f}")

    pooled_ade = sum(dists) / len(dists)
    by_type_ade = {g: sum(v) / len(v) for g, v in by_type.items()}

    print("\n" + "=" * 60)
    print(f"Gaze-LLE ({MODEL_NAME}, zero-shot) on VAT _TEST")
    print("=" * 60)
    print(f"Pooled ADE: {pooled_ade:.4f}  (n={len(dists)})")
    for g, ade in sorted(by_type_ade.items()):
        print(f"  {g:<10} ADE={ade:.4f}  n={len(by_type[g])}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "model_name": MODEL_NAME, "checkpoint_path": CHECKPOINT_PATH,
            "n_records": len(dists), "pooled_ade": pooled_ade,
            "by_gaze_type": {g: {"ade": v, "n": len(by_type[g])} for g, v in by_type_ade.items()},
        }, f, indent=2)
    print(f"\nFull results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
