"""
gazelle_zero_shot_eval_all.py — runs the same Gaze-LLE zero-shot
evaluation as gazelle_zero_shot_vat_eval.py, but sequentially across
ChildPlay, VACATION, and UCO-LAEO in one process -- one command, walks
away, comes back to three results files. Same checkpoint, same method,
same "already-normalized gt_x/gt_y used directly" convention as the VAT
version -- this is a direct generalization of that script, not a
separate design.

UCO-LAEO has no _TEST suffix (never split, matches every other place
this project treats it that way) -- the ONLY dataset in this list where
that's true.

REQUIRES real local image paths for each dataset -- these are NOT
guessed here (VAT's own local path was guessed wrong once already this
project). Fill in the three *_LOCAL_IMAGES_ROOT variables below before
running, or set them as environment variables.

RUN LOCALLY, in your Gaze-LLE venv -- same as the VAT script.
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

CHECKPOINT_PATH = os.environ.get("GAZELLE_CHECKPOINT_PATH", r"C:\repo\gazelle\checkpoints\gazelle_dinov2_vitb14.pt")
MODEL_NAME = os.environ.get("GAZELLE_MODEL_NAME", "gazelle_dinov2_vitb14")
HEATMAP_SIZE = 64

# ══════════════════════════════════════════════════════════════════════
# FILL IN before running -- local paths, confirmed by you, not guessed.
# Label file paths follow the same convention as the VAT ones you already
# have (adjust if yours live somewhere else). HPC_PREFIX values are
# already confirmed from this project's own established config -- only
# the LOCAL side needs filling in.
# ══════════════════════════════════════════════════════════════════════
DATASETS = {
    "childplay": {
        "labels_path": os.environ.get("CHILDPLAY_LABELS_PATH", r"C:\repo\gazelle\dataset\childplay\labels_childplay_full_TEST.jsonl"),
        "hpc_prefix": "/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze/images",
        "local_prefix": os.environ.get("CHILDPLAY_LOCAL_IMAGES_ROOT", r"FILL_IN_ME"),
    },
    "vacation": {
        "labels_path": os.environ.get("VACATION_LABELS_PATH", r"C:\repo\gazelle\dataset\vacation\labels_vacation_full_TEST.jsonl"),
        "hpc_prefix": "/parallel_scratch/sl02092/standard_project/data/vacation/images",
        "local_prefix": os.environ.get("VACATION_LOCAL_IMAGES_ROOT", r"FILL_IN_ME"),
    },
    "ucolaeo": {
        "labels_path": os.environ.get("UCOLAEO_LABELS_PATH", r"C:\repo\gazelle\dataset\ucolaeo\labels_ucolaeo_full.jsonl"),  # NO _TEST -- never split
        "hpc_prefix": "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames",
        "local_prefix": os.environ.get("UCOLAEO_LOCAL_IMAGES_ROOT", r"FILL_IN_ME"),
    },
}

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")


def resolve_local_path(hpc_path, hpc_prefix, local_prefix):
    normalized = hpc_path.replace("\\", "/")
    if not normalized.startswith(hpc_prefix):
        raise ValueError(f"img_path '{hpc_path}' doesn't start with expected "
                          f"HPC prefix '{hpc_prefix}' -- check this dataset's config.")
    rel = normalized[len(hpc_prefix):].lstrip("/")
    return os.path.join(local_prefix, *rel.split("/"))


def load_population(labels_path, hpc_prefix, local_prefix):
    records = []
    n_offscreen = 0
    n_missing = 0
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["gaze_type"] == "offscreen":
                n_offscreen += 1
                continue
            local_path = resolve_local_path(row["img_path"], hpc_prefix, local_prefix)
            if not os.path.isfile(local_path):
                n_missing += 1
                continue
            records.append({
                "img_path": local_path,
                "head_x1": row["head_x1"], "head_y1": row["head_y1"],
                "head_x2": row["head_x2"], "head_y2": row["head_y2"],
                "gt_x": row["gt_x"], "gt_y": row["gt_y"],
                "gaze_type": row["gaze_type"],
            })
    print(f"  Loaded {len(records)} on-screen records "
          f"(excluded {n_offscreen} offscreen, {n_missing} missing images).")
    if n_missing > 0:
        print(f"  WARNING: {n_missing} missing -- if more than a handful, "
              f"the local_prefix for this dataset is probably wrong.")
    return records


def evaluate_one_dataset(dataset_name, config, model, transform, device):
    print(f"\n{'='*60}\n{dataset_name.upper()}\n{'='*60}")

    if config["local_prefix"] == "FILL_IN_ME":
        print(f"  SKIPPED -- {dataset_name}_LOCAL_IMAGES_ROOT not set.")
        return None
    if not os.path.isfile(config["labels_path"]):
        print(f"  SKIPPED -- labels file not found: {config['labels_path']}")
        return None

    records = load_population(config["labels_path"], config["hpc_prefix"], config["local_prefix"])
    if len(records) < 10:
        print(f"  ERROR: too few records -- check paths before trusting this.")
        return None

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
        heatmap = output["heatmap"][0][0]

        argmax_idx = torch.argmax(heatmap).item()
        row_idx, col_idx = argmax_idx // HEATMAP_SIZE, argmax_idx % HEATMAP_SIZE
        pred_x = (col_idx + 0.5) / HEATMAP_SIZE
        pred_y = (row_idx + 0.5) / HEATMAP_SIZE

        dist = math.hypot(pred_x - rec["gt_x"], pred_y - rec["gt_y"])
        dists.append(dist)
        by_type.setdefault(rec["gaze_type"], []).append(dist)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(records)} done, running ADE={sum(dists)/len(dists):.4f}")

    pooled_ade = sum(dists) / len(dists)
    by_type_ade = {g: sum(v) / len(v) for g, v in by_type.items()}

    print(f"  Pooled ADE: {pooled_ade:.4f}  (n={len(dists)})")
    for g, ade in sorted(by_type_ade.items()):
        print(f"    {g:<10} ADE={ade:.4f}  n={len(by_type[g])}")

    result = {
        "dataset": dataset_name, "model_name": MODEL_NAME,
        "n_records": len(dists), "pooled_ade": pooled_ade,
        "by_gaze_type": {g: {"ade": v, "n": len(by_type[g])} for g, v in by_type_ade.items()},
    }
    out_path = os.path.join(OUTPUT_DIR, f"gazelle_{dataset_name}_zeroshot_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME}...")
    model, transform = get_gazelle_model(MODEL_NAME)
    model.load_gazelle_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.to(device)
    model.eval()

    all_results = {}
    for name, config in DATASETS.items():
        result = evaluate_one_dataset(name, config, model, transform, device)
        if result:
            all_results[name] = result

    print(f"\n{'='*60}\nALL DATASETS SUMMARY\n{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<12} pooled_ADE={r['pooled_ade']:.4f}  n={r['n_records']}")


if __name__ == "__main__":
    main()
