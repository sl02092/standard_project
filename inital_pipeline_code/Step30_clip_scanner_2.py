"""
Step 3 - VAT Training Clip Scanner (v2)
Scans annotations/train, scores every clip, applies selection criteria,
then applies per-show caps and frame sampling to produce a balanced,
budget-controlled training set definition.

This is the MAIN PROJECT clip selection — the output CSV defines exactly
which frames the teacher labelling pipeline will process.

Outputs:
    clip_summary.csv    — full stats for all 475 clips (for reference)
    clip_selected.csv   — filtered + capped clips with frame budgets
    frame_manifest.csv  — every individual frame selected for processing

Usage:
    python step3_clip_scanner.py

Requirements:
    pip install pandas
"""

import os
import csv
import random
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_PATH     = r"C:\repo\standard_project\videoattentiontarget"
TRAIN_ANN_DIR = os.path.join(BASE_PATH, "annotations", "train")

# ── Clip selection thresholds ──────────────────────────────────────────────────
MIN_FRAMES         = 20      # minimum annotated frames per clip
MIN_SUBJECTS       = 2       # must have at least 2 annotated subjects
MIN_SOCIAL_PCT     = 0.20    # at least 20% social gaze frames
MAX_OFFSCREEN_PCT  = 0.60    # no more than 60% off-screen gaze
MIN_FRAMES_PER_SHOW = 200    # drop shows contributing fewer than this

# ── Frame budget controls ──────────────────────────────────────────────────────
FRAME_SAMPLE_EVERY_N = 3     # use every Nth frame (3 = ~33% of frames)
MAX_FRAMES_PER_SHOW  = 3000  # cap per show to prevent visual bias
TARGET_TOTAL_FRAMES  = 25000 # soft target for total training frames

RANDOM_SEED = 42             # for reproducibility

# ── Output files ───────────────────────────────────────────────────────────────
OUTPUT_ALL      = "clip_summary.csv"
OUTPUT_SELECTED = "clip_selected.csv"
OUTPUT_MANIFEST = "frame_manifest.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_annotations(ann_path):
    frames = []
    with open(ann_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            fname = row[0].strip()
            try:
                x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                gx, gy = int(row[5]), int(row[6])
            except ValueError:
                continue
            gaze = (gx, gy) if gx != -1 and gy != -1 else None
            frames.append({
                "fname": fname,
                "head":  (x1, y1, x2, y2),
                "gaze":  gaze,
            })
    return frames


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def should_include(row):
    if row["n_subjects"] < MIN_SUBJECTS:
        return False, f"only {row['n_subjects']} subject(s)"
    if row["total_ann_frames"] < MIN_FRAMES:
        return False, f"only {row['total_ann_frames']} annotated frames"
    if row["social_pct"] < MIN_SOCIAL_PCT:
        return False, f"social gaze only {row['social_pct']*100:.0f}%"
    if row["offscreen_pct"] > MAX_OFFSCREEN_PCT:
        return False, f"off-screen gaze {row['offscreen_pct']*100:.0f}%"
    return True, "ok"

# ── Per-clip analysis ─────────────────────────────────────────────────────────

def analyse_clip(show, clip, ann_dir):
    clip_path = os.path.join(ann_dir, show, clip)
    ann_files = sorted([f for f in os.listdir(clip_path) if f.endswith(".txt")])
    if not ann_files:
        return None

    subjects = {}
    for fname in ann_files:
        label = fname.replace(".txt", "")
        frames = load_annotations(os.path.join(clip_path, fname))
        subjects[label] = {row["fname"]: row for row in frames}

    n_subjects     = len(subjects)
    subject_labels = list(subjects.keys())

    all_fnames = set()
    for frames in subjects.values():
        all_fnames.update(frames.keys())

    total_frames     = 0
    social_frames    = 0
    object_frames    = 0
    offscreen_frames = 0
    mutual_gaze_frames = 0

    # Per-frame gaze type classification
    frame_types = {}  # fname -> {label -> "social"|"object"|"offscreen"}

    for fname in sorted(all_fnames):
        head_boxes = {
            label: frames[fname]["head"]
            for label, frames in subjects.items()
            if fname in frames
        }
        frame_types[fname] = {}

        for label, frames in subjects.items():
            if fname not in frames:
                continue

            total_frames += 1
            gaze = frames[fname]["gaze"]

            if gaze is None:
                offscreen_frames += 1
                frame_types[fname][label] = "offscreen"
                continue

            is_social = any(
                point_in_box(gaze, box)
                for other_label, box in head_boxes.items()
                if other_label != label
            )

            if is_social:
                social_frames += 1
                frame_types[fname][label] = "social"
            else:
                object_frames += 1
                frame_types[fname][label] = "object"

        # Mutual gaze check
        labels = list(head_boxes.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                la, lb = labels[i], labels[j]
                ga = subjects[la].get(fname, {}).get("gaze") if fname in subjects.get(la, {}) else None
                gb = subjects[lb].get(fname, {}).get("gaze") if fname in subjects.get(lb, {}) else None
                if ga and gb:
                    if point_in_box(ga, head_boxes[lb]) and point_in_box(gb, head_boxes[la]):
                        mutual_gaze_frames += 1

    if total_frames == 0:
        return None

    img_path = os.path.join(BASE_PATH, "images", show, clip)
    n_images = len(os.listdir(img_path)) if os.path.exists(img_path) else 0

    # Sampled frames (every Nth, across sorted unique filenames)
    sorted_fnames = sorted(all_fnames)
    sampled_fnames = sorted_fnames[::FRAME_SAMPLE_EVERY_N]

    return {
        "show":               show,
        "clip":               clip,
        "n_subjects":         n_subjects,
        "subject_files":      "|".join(subject_labels),
        "n_images":           n_images,
        "total_ann_frames":   total_frames,
        "social_frames":      social_frames,
        "object_frames":      object_frames,
        "offscreen_frames":   offscreen_frames,
        "mutual_gaze_frames": mutual_gaze_frames,
        "social_pct":         round(social_frames    / total_frames, 3),
        "object_pct":         round(object_frames    / total_frames, 3),
        "offscreen_pct":      round(offscreen_frames / total_frames, 3),
        "sampled_frames":     len(sampled_fnames),
        "_sampled_fnames":    sampled_fnames,   # internal — not written to CSV
        "_frame_types":       frame_types,       # internal — not written to CSV
        "_subjects":          subjects,          # internal — not written to CSV
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("VAT Training Clip Scanner v2 — Main Project Selection")
    print("=" * 60)
    print(f"\nClip selection thresholds:")
    print(f"  Min subjects         : {MIN_SUBJECTS}")
    print(f"  Min frames per clip  : {MIN_FRAMES}")
    print(f"  Min social %         : {MIN_SOCIAL_PCT*100:.0f}%")
    print(f"  Max off-screen %     : {MAX_OFFSCREEN_PCT*100:.0f}%")
    print(f"  Min frames per show  : {MIN_FRAMES_PER_SHOW}")
    print(f"\nFrame budget controls:")
    print(f"  Sample every N frames: {FRAME_SAMPLE_EVERY_N}")
    print(f"  Max frames per show  : {MAX_FRAMES_PER_SHOW}")
    print(f"  Target total frames  : {TARGET_TOTAL_FRAMES:,}")
    print()

    # ── 1. Scan all clips ──────────────────────────────────────────────────
    shows = sorted(os.listdir(TRAIN_ANN_DIR))
    all_rows = []
    total_clips = 0

    for show in shows:
        show_path = os.path.join(TRAIN_ANN_DIR, show)
        if not os.path.isdir(show_path):
            continue
        clips = sorted(os.listdir(show_path))
        for clip in clips:
            if not os.path.isdir(os.path.join(show_path, clip)):
                continue
            total_clips += 1
            try:
                stats = analyse_clip(show, clip, TRAIN_ANN_DIR)
                if stats:
                    all_rows.append(stats)
            except Exception as e:
                print(f"  ✗ {show}/{clip}: {e}")

    print(f"Scanned {total_clips} clips total\n")

    # ── 2. Apply clip selection criteria ───────────────────────────────────
    for row in all_rows:
        inc, reason = should_include(row)
        row["include"]        = inc
        row["exclude_reason"] = reason

    selected = [r for r in all_rows if r["include"]]

    # ── 3. Apply per-show minimum frame filter ─────────────────────────────
    # Count sampled frames per show before capping
    show_sampled = {}
    for row in selected:
        show_sampled[row["show"]] = show_sampled.get(row["show"], 0) + row["sampled_frames"]

    low_shows = {s for s, n in show_sampled.items() if n < MIN_FRAMES_PER_SHOW}
    for row in selected:
        if row["show"] in low_shows:
            row["include"]        = False
            row["exclude_reason"] = f"show total only {show_sampled[row['show']]} sampled frames"

    selected = [r for r in all_rows if r["include"]]

    # ── 4. Apply per-show frame cap ────────────────────────────────────────
    # Sort selected clips by social_pct descending within each show
    # so we keep the best clips when capping
    selected.sort(key=lambda r: (r["show"], -r["social_pct"]))

    show_frame_counts = {}
    manifest_rows = []

    for row in selected:
        show = row["show"]
        show_frame_counts.setdefault(show, 0)

        remaining_budget = MAX_FRAMES_PER_SHOW - show_frame_counts[show]
        if remaining_budget <= 0:
            row["include"]        = False
            row["exclude_reason"] = "show frame cap reached"
            row["sampled_frames_used"] = 0
            continue

        # Take up to remaining budget from this clip's sampled frames
        fnames_to_use = row["_sampled_fnames"][:remaining_budget]
        row["sampled_frames_used"] = len(fnames_to_use)
        show_frame_counts[show]   += len(fnames_to_use)

        # Build manifest entries for these frames
        for fname in fnames_to_use:
            for label, subject_frames in row["_subjects"].items():
                if fname not in subject_frames:
                    continue
                frame_data = subject_frames[fname]
                gaze_type  = row["_frame_types"].get(fname, {}).get(label, "unknown")
                gaze       = frame_data["gaze"]
                head       = frame_data["head"]

                manifest_rows.append({
                    "show":          show,
                    "clip":          row["clip"],
                    "subject":       label,
                    "fname":         fname,
                    "img_path":      os.path.join(BASE_PATH, "images", show, row["clip"], fname),
                    "ann_path":      os.path.join(TRAIN_ANN_DIR, show, row["clip"], f"{label}.txt"),
                    "head_x1":       head[0],
                    "head_y1":       head[1],
                    "head_x2":       head[2],
                    "head_y2":       head[3],
                    "gaze_x":        gaze[0] if gaze else -1,
                    "gaze_y":        gaze[1] if gaze else -1,
                    "gaze_type":     gaze_type,
                    "use_teacher":   gaze_type in ("social", "object"),
                })

    final_selected = [r for r in all_rows if r["include"]]

    # ── 5. Save outputs ────────────────────────────────────────────────────

    # Full summary CSV (drop internal fields)
    summary_cols = [
        "show", "clip", "n_subjects", "subject_files", "n_images",
        "total_ann_frames", "social_frames", "object_frames",
        "offscreen_frames", "mutual_gaze_frames",
        "social_pct", "object_pct", "offscreen_pct",
        "sampled_frames", "include", "exclude_reason"
    ]
    df_all = pd.DataFrame([
        {k: r[k] for k in summary_cols} for r in all_rows
    ])
    df_all.to_csv(OUTPUT_ALL, index=False)

    # Selected clips CSV
    sel_cols = summary_cols + ["sampled_frames_used"]
    df_sel = pd.DataFrame([
        {k: r.get(k, "") for k in sel_cols} for r in final_selected
    ])
    df_sel.to_csv(OUTPUT_SELECTED, index=False)

    # Frame manifest CSV
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(OUTPUT_MANIFEST, index=False)

    # ── 6. Print summary ───────────────────────────────────────────────────
    total_manifest_frames = len(manifest_rows)
    social_manifest       = sum(1 for r in manifest_rows if r["gaze_type"] == "social")
    object_manifest       = sum(1 for r in manifest_rows if r["gaze_type"] == "object")
    offscreen_manifest    = sum(1 for r in manifest_rows if r["gaze_type"] == "offscreen")
    teacher_frames        = sum(1 for r in manifest_rows if r["use_teacher"])

    print(f"✓ {OUTPUT_ALL:<25} ({len(df_all)} clips)")
    print(f"✓ {OUTPUT_SELECTED:<25} ({len(df_sel)} clips)")
    print(f"✓ {OUTPUT_MANIFEST:<25} ({total_manifest_frames:,} frame-subject pairs)")

    print(f"\n── Final training set ────────────────────────────────────")
    print(f"  Clips selected         : {len(final_selected)}")
    print(f"  Shows represented      : {len(set(r['show'] for r in final_selected))}")
    print(f"  Total frame-subj pairs : {total_manifest_frames:,}")
    print(f"  Social gaze frames     : {social_manifest:,}  ({social_manifest/total_manifest_frames*100:.1f}%)")
    print(f"  Object gaze frames     : {object_manifest:,}  ({object_manifest/total_manifest_frames*100:.1f}%)")
    print(f"  Off-screen frames      : {offscreen_manifest:,}  ({offscreen_manifest/total_manifest_frames*100:.1f}%)")
    print(f"\n  Teacher will label     : {teacher_frames:,} frames  ({teacher_frames/total_manifest_frames*100:.1f}%)")
    print(f"  GT will label          : {total_manifest_frames-teacher_frames:,} frames  ({(total_manifest_frames-teacher_frames)/total_manifest_frames*100:.1f}%)")

    print(f"\n── Per-show frame distribution ───────────────────────────")
    show_summary = {}
    for r in manifest_rows:
        s = r["show"]
        if s not in show_summary:
            show_summary[s] = {"frames": 0, "social": 0}
        show_summary[s]["frames"] += 1
        if r["gaze_type"] == "social":
            show_summary[s]["social"] += 1

    for show, stats in sorted(show_summary.items(), key=lambda x: -x[1]["frames"]):
        pct = stats["social"] / stats["frames"] * 100 if stats["frames"] > 0 else 0
        print(f"  {show:<35} {stats['frames']:>5} frames  {pct:.0f}% social")

    print(f"\n  frame_manifest.csv is the input to Step 4 (teacher labelling pipeline).")
    print(f"  Each row = one frame × one subject to be labelled.")
    print(f"  use_teacher=True  → VLM generates label")
    print(f"  use_teacher=False → GT annotation used directly")
    print("\nDone.")


if __name__ == "__main__":
    main()