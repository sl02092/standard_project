"""
diagnose_childplay_teacher_none_gt.py — pulls a few real examples of the
2,117 rows found by compute_childplay_mtgs_ade.py where label_source is
"teacher" (a genuine, successfully-parsed InternVL3 prediction) but the
row's own gt_x/gt_y is None (manifest gaze_x was -1). This combination
is surprising -- a "teacher" label implies the manifest classified the
row as having a real target, but gaze_x=-1 means no real GT was ever
recorded for it. Printed in full (every field) so the actual mechanism
can be read from real data rather than guessed at.
"""

import json

INTERNVL3_PATH = "labels_childplay_full.jsonl"
MTGS_PATH = "mtgs_teacher_childplay.jsonl"
N_EXAMPLES = 5


def main():
    mtgs_gt = {}
    with open(MTGS_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["show"], r["fname"], r["subject"])
            mtgs_gt[key] = (r.get("gt_x"), r.get("gt_y"), r.get("gaze_type"))

    found = 0
    with open(INTERNVL3_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("label_source") != "teacher":
                continue
            if r.get("gt_x") is not None:
                continue
            key = (r["show"], r["fname"], r["subject"])
            if key not in mtgs_gt:
                continue  # not in the "common" population we're diagnosing

            found += 1
            print(f"\n{'='*70}")
            print(f"Example {found}: {key}")
            print(f"{'='*70}")
            print("Full labels_childplay_full.jsonl row:")
            for k, v in r.items():
                print(f"  {k}: {v}")
            m_gt_x, m_gt_y, m_gaze_type = mtgs_gt[key]
            print(f"\nFor comparison, mtgs_teacher_childplay.jsonl's own "
                  f"gt_x/gt_y/gaze_type for this SAME key: "
                  f"({m_gt_x}, {m_gt_y}, {m_gaze_type!r})")

            if found >= N_EXAMPLES:
                return


if __name__ == "__main__":
    main()
