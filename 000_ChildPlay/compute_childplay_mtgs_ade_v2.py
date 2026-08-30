"""
compute_childplay_mtgs_ade_v2.py — adds the gaze-type breakdown to
compute_childplay_mtgs_ade.py's already-clean result (GT agreement was
already fine at the 0.001 threshold here, no resolution-scaling issue
like VACATION's -- ChildPlay's images are much higher resolution).
Tests the same object-gaze-weakness hypothesis that explained VACATION's
large pooled gap: does InternVL3's dramatic pooled ADE here (0.3557
pooled vs MTGS's 0.0797) concentrate in object-gaze specifically, or is
it spread evenly across both categories?
"""

import json

MTGS_PATH = "mtgs_teacher_childplay.jsonl"
INTERNVL3_PATH = "labels_childplay_full.jsonl"


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


def report_ade(label, keys, mtgs, internvl3):
    m_dists = [
        ade(mtgs[k]["pred_x"], mtgs[k]["pred_y"], mtgs[k]["gt_x"], mtgs[k]["gt_y"])
        for k in keys
    ]
    i_dists = [
        ade(internvl3[k]["pred_x"], internvl3[k]["pred_y"], internvl3[k]["gt_x"], internvl3[k]["gt_y"])
        for k in keys if internvl3[k].get("pred_x") is not None
    ]
    print(f"\n{label} (n={len(keys)}):")
    if m_dists:
        print(f"  MTGS ADE_norm            : {sum(m_dists)/len(m_dists):.4f}  (n={len(m_dists)})")
    if i_dists:
        print(f"  InternVL3-alone ADE_norm : {sum(i_dists)/len(i_dists):.4f}  (n={len(i_dists)})")


def main():
    mtgs = load_rows(MTGS_PATH)
    internvl3 = load_rows(INTERNVL3_PATH)
    common = set(mtgs) & set(internvl3)

    # Same None-GT skip as compute_childplay_mtgs_ade.py -- see
    # diagnose_childplay_teacher_none_gt.py for the investigation into
    # WHY these exist (all label_source="teacher", genuinely surprising,
    # not yet fully explained).
    valid_keys = {
        k for k in common
        if mtgs[k].get("gt_x") is not None and internvl3[k].get("gt_x") is not None
    }
    print(f"Common keys: {len(common)}, valid (non-None GT): {len(valid_keys)}")

    report_ade("POOLED", valid_keys, mtgs, internvl3)
    for gtype in ("social", "object"):
        keys_this_type = {k for k in valid_keys if mtgs[k].get("gaze_type") == gtype}
        report_ade(gtype.upper(), keys_this_type, mtgs, internvl3)


if __name__ == "__main__":
    main()
