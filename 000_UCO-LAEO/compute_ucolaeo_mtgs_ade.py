"""
compute_ucolaeo_mtgs_ade.py — computes MTGS's actual accuracy on
UCO-LAEO (ADE against real GT) and directly compares it against
InternVL3-alone's own accuracy on the exact same population -- the
comparison this whole export pipeline existed to produce, not yet
actually computed until now (everything checked so far was population
integrity: row counts, duplicates, resume correctness -- never whether
the predictions themselves are any good).

Run against the DEDUPED files (mtgs_teacher_ucolaeo.jsonl,
labels_ucolaeo_full.jsonl) -- both should now have identical 18,726-row
populations. This script explicitly CONFIRMS that alignment rather than
assuming it, and cross-checks GT agreement between the two files as a
free extra consistency check (both trace back to the same underlying
data, so their own gt_x/gt_y should agree exactly for every shared row).
"""

import json
import random

MTGS_PATH = "mtgs_teacher_ucolaeo.jsonl"
INTERNVL3_PATH = "labels_ucolaeo_full.jsonl"


def load_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["show"], r["fname"], r["subject"])
            rows[key] = r
    return rows


def ade(pred_x, pred_y, gt_x, gt_y):
    return ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5


def main():
    mtgs = load_rows(MTGS_PATH)
    internvl3 = load_rows(INTERNVL3_PATH)

    print(f"MTGS rows: {len(mtgs)}, InternVL3 rows: {len(internvl3)}")
    common = set(mtgs) & set(internvl3)
    only_mtgs = set(mtgs) - set(internvl3)
    only_internvl3 = set(internvl3) - set(mtgs)
    print(f"Common keys: {len(common)}, MTGS-only: {len(only_mtgs)}, "
          f"InternVL3-only: {len(only_internvl3)}")
    if only_mtgs or only_internvl3:
        print("  WARNING: populations don't fully match -- investigate "
              "before trusting a direct comparison.")
        print("  Sample MTGS-only:", list(only_mtgs)[:3])
        print("  Sample InternVL3-only:", list(only_internvl3)[:3])

    # GT agreement check -- both files trace back to the same underlying
    # data, so their own gt_x/gt_y should agree exactly for every
    # common key.
    gt_mismatches = []
    for key in common:
        m, i = mtgs[key], internvl3[key]
        if round(m["gt_x"], 4) != round(i["gt_x"], 4) or round(m["gt_y"], 4) != round(i["gt_y"], 4):
            gt_mismatches.append(key)
    print(f"GT mismatches between the two files (should be 0): {len(gt_mismatches)}")
    if gt_mismatches:
        print("  Sample mismatches:", gt_mismatches[:3])

    mtgs_dists = []
    internvl3_dists = []
    for key in common:
        m, i = mtgs[key], internvl3[key]
        mtgs_dists.append(ade(m["pred_x"], m["pred_y"], m["gt_x"], m["gt_y"]))
        if i.get("pred_x") is not None:
            internvl3_dists.append(ade(i["pred_x"], i["pred_y"], i["gt_x"], i["gt_y"]))

    print(f"\nMTGS ADE_norm            : {sum(mtgs_dists)/len(mtgs_dists):.4f}  (n={len(mtgs_dists)})")
    print(f"InternVL3-alone ADE_norm : {sum(internvl3_dists)/len(internvl3_dists):.4f}  (n={len(internvl3_dists)})")

    print("\n--- Spot-check sample (open the real image and confirm by eye) ---")
    random.seed(42)
    sample_keys = random.sample(sorted(common), min(5, len(common)))
    for key in sample_keys:
        m = mtgs[key]
        d = ade(m["pred_x"], m["pred_y"], m["gt_x"], m["gt_y"])
        print(f"  {key}")
        print(f"    img_path={m['img_path']}")
        print(f"    MTGS pred=({m['pred_x']:.4f}, {m['pred_y']:.4f})  "
              f"GT=({m['gt_x']:.4f}, {m['gt_y']:.4f})  dist={d:.4f}")


if __name__ == "__main__":
    main()
