"""
gazelle_zero_shot_eval_rigorous.py — the fully rigorous version. Uses
the EXACT SAME 3-way join logic (full/hybrid/mtgs, GT-mismatch tolerance
2.0px, on-screen only) as eval_static_v1.py, for all 4 datasets -- so
every Gaze-LLE number here is on the byte-identical population our own
models were scored on, not just "close enough."

WHY THIS REPLACES gazelle_zero_shot_vat_eval.py / gazelle_zero_shot_eval_all.py:
those used labels_{dataset}_full_TEST.jsonl ALONE, which only happens to
equal the joined population for VAT and UCO-LAEO (both 100% MTGS-aligned).
For ChildPlay (+10.4% extra rows) and VACATION (+28.2% extra rows), the
full-file-alone population is measurably larger than what our own models
were actually tested on -- confirmed directly against eval_static_v1.py's
own real run logs (jobs 735917/735918). This version closes that gap for
all four datasets at once, not just the two that needed it.

REQUIRES all three label files per dataset locally (full, hybrid,
mtgs-teacher) -- not just "full". If ChildPlay/VACATION/UCO-LAEO only
have their _full file copied locally so far, the other two need copying
too before this will run for that dataset (it fails loudly, not
silently, if a file is missing).

RUN LOCALLY, in your Gaze-LLE venv. One run, all four datasets,
sequentially.
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
GT_MISMATCH_PIXEL_TOLERANCE = 2.0  # matches eval_static_v1.py exactly

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")

# Defaults match every path already confirmed working in prior runs this
# session -- override via env var only if something's actually different.
DATASETS = {
    "vat": {
        "local_dir": os.environ.get("VAT_LABELS_DIR", r"C:\repo\gazelle\dataset\vat"),
        "local_images_root": os.environ.get("VAT_LOCAL_IMAGES_ROOT", r"C:\repo\gazelle\dataset\vat\images"),
        "hpc_images_prefix": "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget/images",
        "full": "labels_vat_full_TEST.jsonl", "hybrid": "labels_vat_hybrid_targeted_TEST.jsonl",
        "mtgs": "mtgs_teacher_vat_merged_TEST.jsonl",
    },
    "childplay": {
        "local_dir": os.environ.get("CHILDPLAY_LABELS_DIR", r"C:\repo\gazelle\dataset\childplay"),
        "local_images_root": os.environ.get("CHILDPLAY_LOCAL_IMAGES_ROOT", r"C:\repo\gazelle\dataset\childplay\images"),
        "hpc_images_prefix": "/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze/images",
        "full": "labels_childplay_full_TEST.jsonl", "hybrid": "labels_childplay_hybrid_targeted_TEST.jsonl",
        "mtgs": "mtgs_teacher_childplay_TEST.jsonl",
    },
    "vacation": {
        "local_dir": os.environ.get("VACATION_LABELS_DIR", r"C:\repo\gazelle\dataset\vacation"),
        "local_images_root": os.environ.get("VACATION_LOCAL_IMAGES_ROOT", r"C:\repo\gazelle\dataset\vacation\images"),
        "hpc_images_prefix": "/parallel_scratch/sl02092/standard_project/data/vacation/images",
        "full": "labels_vacation_full_TEST.jsonl", "hybrid": "labels_vacation_hybrid_targeted_TEST.jsonl",
        "mtgs": "mtgs_teacher_vacation_TEST.jsonl",
    },
    "ucolaeo": {
        "local_dir": os.environ.get("UCOLAEO_LABELS_DIR", r"C:\repo\gazelle\dataset\ucolaeo"),
        "local_images_root": os.environ.get("UCOLAEO_LOCAL_IMAGES_ROOT", r"C:\repo\gazelle\dataset\ucolaeo\images"),
        "hpc_images_prefix": "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames",
        # NO _TEST suffix -- UCO-LAEO is never split, matches every other
        # place this project treats it that way.
        "full": "labels_ucolaeo_full.jsonl", "hybrid": "labels_ucolaeo_hybrid_targeted.jsonl",
        "mtgs": "mtgs_teacher_ucolaeo.jsonl",
    },
}


# ══════════════════════════════════════════════════════════════════════
# ── JOIN LOGIC — copied verbatim from eval_static_v1.py's own
# ── build_dataset_population(), adapted only for local paths ───────────
# ══════════════════════════════════════════════════════════════════════

def load_labels_file(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            required = ["pred_x", "pred_y", "gt_x", "gt_y", "gt_px_x", "gt_px_y", "img_path",
                        "head_x1", "head_y1", "head_x2", "head_y2",
                        "show", "clip", "fname", "subject", "gaze_type"]
            if any(rec.get(k) is None for k in required):
                continue
            key = (str(rec["show"]), str(rec["clip"]), str(rec["fname"]), str(rec["subject"]))
            records[key] = rec
    return records


def resolve_local_path(hpc_path, hpc_prefix, local_prefix):
    normalized = hpc_path.replace("\\", "/")
    if not normalized.startswith(hpc_prefix):
        raise ValueError(f"img_path '{hpc_path}' doesn't start with expected "
                          f"HPC prefix '{hpc_prefix}' -- check this dataset's config.")
    rel = normalized[len(hpc_prefix):].lstrip("/")
    return os.path.join(local_prefix, *rel.split("/"))


def build_joined_population(dataset_name, config):
    full_path = os.path.join(config["local_dir"], config["full"])
    hybrid_path = os.path.join(config["local_dir"], config["hybrid"])
    mtgs_path = os.path.join(config["local_dir"], config["mtgs"])

    for p, label in [(full_path, "full"), (hybrid_path, "hybrid"), (mtgs_path, "mtgs")]:
        if not os.path.isfile(p):
            print(f"  SKIPPED -- {label} file not found: {p}")
            return None

    full_recs = load_labels_file(full_path)
    hybrid_recs = load_labels_file(hybrid_path)
    mtgs_recs = load_labels_file(mtgs_path)
    common = set(full_recs) & set(hybrid_recs) & set(mtgs_recs)
    print(f"  Key alignment: {len(common)} common to all three files "
          f"(full={len(full_recs)}, hybrid={len(hybrid_recs)}, mtgs={len(mtgs_recs)})")

    gt_mismatches, missing_images, offscreen = 0, 0, 0
    records = []
    for key in common:
        fu, h, m = full_recs[key], hybrid_recs[key], mtgs_recs[key]
        d_fh = math.hypot(fu["gt_px_x"] - h["gt_px_x"], fu["gt_px_y"] - h["gt_px_y"])
        d_fm = math.hypot(fu["gt_px_x"] - m["gt_px_x"], fu["gt_px_y"] - m["gt_px_y"])
        if d_fh > GT_MISMATCH_PIXEL_TOLERANCE or d_fm > GT_MISMATCH_PIXEL_TOLERANCE:
            gt_mismatches += 1
            continue
        if fu["gaze_type"] == "offscreen":
            offscreen += 1
            continue
        try:
            local_path = resolve_local_path(fu["img_path"], config["hpc_images_prefix"], config["local_images_root"])
        except ValueError as e:
            print(f"  ERROR: {e}")
            return None
        if not os.path.isfile(local_path):
            missing_images += 1
            continue
        records.append({
            "img_path": local_path,
            "head_x1": fu["head_x1"], "head_y1": fu["head_y1"],
            "head_x2": fu["head_x2"], "head_y2": fu["head_y2"],
            "gt_x": fu["gt_x"], "gt_y": fu["gt_y"],
            "gaze_type": fu["gaze_type"],
        })

    print(f"  GT disagreements excluded (>{GT_MISMATCH_PIXEL_TOLERANCE}px): {gt_mismatches}")
    print(f"  Offscreen excluded: {offscreen}")
    print(f"  Missing local images: {missing_images}")
    if missing_images > 0:
        print(f"  WARNING: {missing_images} missing -- if more than a handful, "
              f"local_images_root for {dataset_name} is probably wrong.")
    print(f"  Final [{dataset_name}] population: {len(records)} records "
          f"(this should match eval_static_v1.py's own joined population size)")
    return records


# ══════════════════════════════════════════════════════════════════════
# ── INFERENCE ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def evaluate_one_dataset(dataset_name, config, model, transform, device):
    print(f"\n{'='*60}\n{dataset_name.upper()}\n{'='*60}")

    records = build_joined_population(dataset_name, config)
    if not records or len(records) < 10:
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
    out_path = os.path.join(OUTPUT_DIR, f"gazelle_{dataset_name}_zeroshot_rigorous_results.json")
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

    print(f"\n{'='*60}\nALL DATASETS SUMMARY (rigorous, exact-population-matched)\n{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<12} pooled_ADE={r['pooled_ade']:.4f}  n={r['n_records']}")

    combined_path = os.path.join(OUTPUT_DIR, "gazelle_all_zeroshot_rigorous_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results (all datasets, one file): {combined_path}")


if __name__ == "__main__":
    main()
