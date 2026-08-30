"""
compute_vacation_mtgs_ade_v2.py — refinement of
compute_vacation_mtgs_ade.py after two real findings from the first run:

1. GT_TOLERANCE=0.001 was calibrated loosely against UCO-LAEO's own
   ~1280px-wide images. VACATION's images are 640x360 -- lower
   resolution means the SAME half-pixel quantization noise translates
   to a LARGER normalized-space number. The "worst offenders" from the
   first run (dx=0.000781, dy=0.001389, repeated across many frames)
   match 0.5/640=0.00078125 and 0.5/360=0.0013889 almost exactly --
   this is the identical benign rounding phenomenon already confirmed
   for UCO-LAEO, just needing a threshold sized for VACATION's real
   (lower) resolution rather than reused blindly from a different
   dataset's images. Raised to 0.0015 here (safely above 0.5/360).

2. The pooled ADE gap (MTGS 0.0991 vs InternVL3-alone 0.2718) is large
   -- worth testing a real hypothesis rather than accepting at face
   value. InternVL3's own object-gaze weakness is already established
   elsewhere in this project (a zero-shot spatial-reasoning limitation,
   confirmed independently more than once, not dataset-specific).
   VACATION has a substantial object-gaze population (42.4%), unlike
   some other datasets here. If InternVL3's pooled score is being
   dragged up specifically by its object-gaze predictions, a per-type
   breakdown should show a much smaller social-only gap and a much
   larger object-only one.
"""

import json

MTGS_PATH = "mtgs_teacher_vacation.jsonl"
INTERNVL3_PATH = "labels_vacation_full.jsonl"
GT_TOLERANCE = 0.0015


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
    print(f"Common keys: {len(common)}")

    exact_matches = 0
    subpixel_matches = 0
    real_mismatches = []
    skipped_none = set()
    for key in common:
        m, i = mtgs[key], internvl3[key]
        if m.get("gt_x") is None or i.get("gt_x") is None:
            skipped_none.add(key)
            continue
        dx = abs(m["gt_x"] - i["gt_x"])
        dy = abs(m["gt_y"] - i["gt_y"])
        if dx == 0 and dy == 0:
            exact_matches += 1
        elif dx <= GT_TOLERANCE and dy <= GT_TOLERANCE:
            subpixel_matches += 1
        else:
            real_mismatches.append((key, dx, dy))

    real_mismatch_keys = {k for k, _, _ in real_mismatches}
    print(f"GT agreement (tolerance={GT_TOLERANCE}): {exact_matches} exact, "
          f"{subpixel_matches} sub-pixel, {len(real_mismatches)} REAL "
          f"mismatches (should be ~0), {len(skipped_none)} skipped (None GT)")
    if real_mismatches:
        real_mismatches.sort(key=lambda t: -(t[1] + t[2]))
        print("  Worst offenders:")
        for key, dx, dy in real_mismatches[:5]:
            print(f"    {key}: dx={dx:.6f}, dy={dy:.6f}")
    else:
        print("  All mismatches resolved at the corrected threshold -- "
              "confirms this was resolution-scaled rounding noise, same "
              "as UCO-LAEO's, not a new problem.")

    valid_keys = common - skipped_none - real_mismatch_keys

    report_ade("POOLED", valid_keys, mtgs, internvl3)
    for gtype in ("social", "object"):
        keys_this_type = {k for k in valid_keys if mtgs[k].get("gaze_type") == gtype}
        report_ade(gtype.upper(), keys_this_type, mtgs, internvl3)


if __name__ == "__main__":
    main()
