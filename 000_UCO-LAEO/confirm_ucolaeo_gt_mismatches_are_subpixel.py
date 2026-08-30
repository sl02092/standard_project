"""
confirm_ucolaeo_gt_mismatches_are_subpixel.py — follow-up to
diagnose_ucolaeo_gt_mismatch.py. Confirms that ALL 14,203 gt_x/gt_y
mismatches between mtgs_teacher_ucolaeo.jsonl and labels_ucolaeo_full.jsonl
are genuinely sub-pixel rounding noise (MTGS reads the H5's raw float
directly; InternVL3's file routes through an intermediate rounded-to-
integer pixel representation via frame_manifest_ucolaeo.csv), rather
than trusting that conclusion from the 3 example keys alone.

PASS CRITERION: every mismatch's normalized-space difference should be
at most ~0.5 pixel's worth of quantization error for whatever the real
image width/height is. Since UCO-LAEO's images vary in size, this uses
a generous fixed threshold (0.001, comfortably above 0.5/1280≈0.00039
for a typical frame) rather than looking up each image's real
dimensions individually -- if everything passes even this loose bound,
the sub-pixel explanation holds; anything that fails it is a real
candidate worth investigating individually.
"""

import json

MTGS_PATH = "mtgs_teacher_ucolaeo.jsonl"
INTERNVL3_PATH = "labels_ucolaeo_full.jsonl"
THRESHOLD = 0.001


def load_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["show"], r["fname"], r["subject"])
            rows[key] = r
    return rows


def main():
    mtgs = load_rows(MTGS_PATH)
    internvl3 = load_rows(INTERNVL3_PATH)
    common = set(mtgs) & set(internvl3)

    within_threshold = 0
    exceeds_threshold = []
    for key in common:
        m, i = mtgs[key], internvl3[key]
        dx = abs(m["gt_x"] - i["gt_x"])
        dy = abs(m["gt_y"] - i["gt_y"])
        if dx == 0 and dy == 0:
            continue  # exact match, not a mismatch at all
        if dx <= THRESHOLD and dy <= THRESHOLD:
            within_threshold += 1
        else:
            exceeds_threshold.append((key, dx, dy))

    print(f"Mismatches within sub-pixel threshold ({THRESHOLD}): {within_threshold}")
    print(f"Mismatches EXCEEDING threshold (real candidates): {len(exceeds_threshold)}")
    if exceeds_threshold:
        exceeds_threshold.sort(key=lambda t: -(t[1] + t[2]))
        print("\nWorst offenders (largest gt_x/gt_y difference first):")
        for key, dx, dy in exceeds_threshold[:10]:
            print(f"  {key}: dx={dx:.6f}, dy={dy:.6f}")
    else:
        print("\nAll mismatches are within the sub-pixel threshold -- confirms "
              "the rounding-noise explanation holds across the full "
              "population, not just the 3 sample keys already checked.")


if __name__ == "__main__":
    main()
