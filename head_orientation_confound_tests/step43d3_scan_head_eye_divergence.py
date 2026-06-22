"""
scan_head_eye_divergence.py — scans VAT annotation files for frames where
GT gaze target is inconsistent with a naive "look where your head points"
heuristic. Flags candidates for the head-orientation-vs-eye-direction
confound check (cf. arXiv 2506.05412).

Does NOT touch step40/step43 pipeline files. Read-only against annotations.
"""

import os
import csv
import math

# ── CONFIG ──────────────────────────────────────────────────────────────
ANNOTATIONS_ROOT = r"C:\repo\standard_project\videoattentiontarget\annotations"

# A frame is flagged if GT lies on the SAME side of the head box centre
# as the head box's own "back" would be — i.e. GT is behind the direction
# the head box's aspect ratio suggests it's facing — OR if GT requires a
# direction sharply inconsistent with a head-box-centre-projected ray.
# We use a simple, conservative heuristic to start: flag any frame where
# GT is FAR from the head box (in image-relative terms) AND on the
# opposite horizontal side from where a same-side "looking at nearest
# face" guess would land. This catches the clearest divergence cases
# without needing actual head-pose/yaw estimation.

MIN_GT_DISTANCE_RATIO = 1.5  # GT must be at least this many head-box-widths away
TOP_N_PER_FILE = 15

def load_annotation_file(filepath):
    """Returns list of dicts: fname, head_box (tuple), gaze (tuple or None)."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or len(line) < 7:
                continue
            try:
                fname = line[0]
                hx1, hy1, hx2, hy2 = map(float, line[1:5])
                gx, gy = float(line[5]), float(line[6])
                gaze = None if (gx == -1 and gy == -1) else (gx, gy)
                rows.append({
                    "fname": fname,
                    "head_box": (hx1, hy1, hx2, hy2),
                    "gaze": gaze,
                })
            except (ValueError, IndexError):
                continue
    return rows


def score_divergence(row):
    """
    Returns a divergence score (higher = more likely head-pose-incompatible),
    or None if GT is off-screen (can't compute divergence against no target).
    """
    if row["gaze"] is None:
        return None

    hx1, hy1, hx2, hy2 = row["head_box"]
    head_cx = (hx1 + hx2) / 2
    head_cy = (hy1 + hy2) / 2
    head_w = hx2 - hx1
    head_h = hy2 - hy1
    head_diag = math.hypot(head_w, head_h)
    if head_diag == 0:
        return None

    gx, gy = row["gaze"]
    dist = math.hypot(gx - head_cx, gy - head_cy)
    dist_ratio = dist / head_diag

    # Heuristic: large distance ratio = GT is far from the head's own
    # location, i.e. the head is looking at something well outside its
    # own immediate area. This alone doesn't prove head/eye divergence
    # (a person CAN turn their head fully toward a far target), but
    # combined with low frame-to-frame head movement (checked separately,
    # see flag_static_head_large_gt_shift below) it's a strong signal.
    return dist_ratio


def scan_file_for_divergence(filepath):
    rows = load_annotation_file(filepath)
    scored = []
    for row in rows:
        d = score_divergence(row)
        if d is not None:
            scored.append((d, row))
    scored.sort(key=lambda x: -x[0])
    return scored[:TOP_N_PER_FILE]


def flag_static_head_large_gt_shift(rows, window=10):
    flagged = []
    for i in range(len(rows) - window):
        window_rows = rows[i:i + window]

        # Check if this window contains an off-screen gap
        has_offscreen_gap = any(r["gaze"] is None for r in window_rows)

        gazes_with_idx = [(j, r["gaze"]) for j, r in enumerate(window_rows) if r["gaze"] is not None]
        if len(gazes_with_idx) < 2:
            continue

        head_centres = [
            ((r["head_box"][0] + r["head_box"][2]) / 2,
             (r["head_box"][1] + r["head_box"][3]) / 2)
            for r in window_rows
        ]
        head_movement = max(
            math.hypot(head_centres[j][0] - head_centres[0][0],
                       head_centres[j][1] - head_centres[0][1])
            for j in range(len(head_centres))
        )

        first_idx, first_gaze = gazes_with_idx[0]
        gaze_movement = max(
            math.hypot(g[0] - first_gaze[0], g[1] - first_gaze[1])
            for _, g in gazes_with_idx
        )

        head_diag = math.hypot(
            window_rows[0]["head_box"][2] - window_rows[0]["head_box"][0],
            window_rows[0]["head_box"][3] - window_rows[0]["head_box"][1],
        )
        if head_diag == 0:
            continue

        if head_movement < 0.3 * head_diag and gaze_movement > 2.0 * head_diag:
            flagged.append({
                "start_fname": window_rows[0]["fname"],
                "end_fname": window_rows[-1]["fname"],
                "head_movement": head_movement,
                "gaze_movement": gaze_movement,
                "head_diag": head_diag,
                "spans_offscreen_gap": has_offscreen_gap,  # NEW
            })
    return flagged

def main():
    print(f"Scanning: {ANNOTATIONS_ROOT}\n")
    all_static_flags = []

    for root, _, files in os.walk(ANNOTATIONS_ROOT):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(root, fname)
            rows = load_annotation_file(filepath)
            if len(rows) < 10:
                continue

            flags = flag_static_head_large_gt_shift(rows)
            for flag in flags:
                flag["clip_path"] = filepath
                all_static_flags.append(flag)

    print(f"Found {len(all_static_flags)} candidate static-head/large-gaze-shift windows\n")
    all_static_flags.sort(key=lambda f: -f["gaze_movement"] / max(f["head_movement"], 1.0))

    for flag in all_static_flags[:30]:
            ratio = flag["gaze_movement"] / max(flag["head_movement"], 1.0)
            gap_note = " [SPANS OFFSCREEN GAP]" if flag["spans_offscreen_gap"] else ""
            print(f"{flag['clip_path']}{gap_note}")
            print(f"  frames: {flag['start_fname']} -> {flag['end_fname']}")
            print(f"  head_movement={flag['head_movement']:.1f}px  "
                f"gaze_movement={flag['gaze_movement']:.1f}px  "
                f"ratio={ratio:.1f}x  head_diag={flag['head_diag']:.1f}px")
            print()


if __name__ == "__main__":
    main()