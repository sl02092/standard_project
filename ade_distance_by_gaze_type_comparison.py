"""
ade_by_gaze_type_comparison.py

Objective 2 deliverable: mean ADE broken down by gaze_type, compared
across completed teacher-pipeline label files, to pick and freeze the
final production teacher config.

EDIT LABEL_FILES below to point at your actual completed run files.

WHAT THIS COMPUTES, per file, per gaze_type:
  - "Pipeline ADE": mean distance over ALL on-screen frames using whatever
    final (pred_x, pred_y) ended up in the label -- includes GT-fallback
    rows (pred == gt by construction, trivially distance 0). Represents
    REAL end-to-end pipeline accuracy, fallback included -- a robot
    deployment doesn't get to skip the frames the model failed on.
  - "Model-only ADE": same, but EXCLUDING GT-fallback rows -- isolates
    genuine model-generated coordinate accuracy, for comparing raw
    identification+localization quality between label sets.
  - Off-screen frames are reported as a DETECTION ACCURACY, not an ADE --
    distance isn't meaningful for a target that isn't in frame.

IMPORTANT: on/off-screen is determined via gt_px_x/gt_px_y (raw pixel, -1
sentinel), NOT gt_x/gt_y. A real bug was found in a separate investigation
tonight: gt_x/gt_y for off-screen "gt" rows in earlier production runs
were mis-normalized (-1/width) rather than a clean sentinel or null --
using gt_x/gt_y here would silently corrupt this exact check.

DIAGNOSTIC FIRST, ALWAYS: prints unique gaze_type / label_source values
and counts in each file BEFORE computing any statistics. Verify these
match what you expect before trusting the numbers below -- don't assume
the schema, given tonight's frame_manifest.csv saga was exactly that.
"""

import math

import pandas as pd

# EDIT THESE if your downloaded files land somewhere other than the same
# folder as this script. Relative paths assume you've downloaded all three
# .jsonl files locally (only the labels files are needed -- no images, no
# model weights -- so this is cheap to pull down and run without the HPC).
'''
LABEL_FILES = {
    "labels_full (step40, box-centroid, orig. prompt)": "labels_full.jsonl",
    "labels_hybrid (DINOv2 hybrid / box-centroid)": "labels_hybrid.jsonl",
     "labels_hybrid_targeted (targeted-phrase prompt)": "labels_hybrid_targeted.jsonl",
}
'''
LABEL_FILES = {
#    "labels_hybrid_targeted (targeted-phrase prompt)":
#        r"C:\Users\scott\Desktop\dissertation\99_teacher_pipeline_results\InternV3_GDino_Hybrid_Attempt_002_Full\labels_hybrid_targeted.jsonl",
#    "labels_hybrid (DINOv2 hybrid / box-centroid)":
#        r"C:\Users\scott\Desktop\dissertation\99_teacher_pipeline_results\InternV3_Dino_Hybrid_Attempt_001_Full\labels_hybrid.jsonl",
    "labels_hybrid_targeted (targeted-phrase prompt)":
        r"C:\Users\scott\Desktop\dissertation\99_teacher_pipeline_results\InternVL3-8B-hf_Attempt_001_Full\labels_full.jsonl",
}


# Adjust this set if your label_source taxonomy differs -- these are the
# categories where pred == gt by construction (the model didn't produce
# an independent coordinate), so they're excluded from "model-only ADE".
FALLBACK_LABEL_SOURCES = {"gt", "gt_fallback_solo", "gt_fallback", "teacher_offscreen"}

REQUIRED_COLUMNS = {"gaze_type", "label_source", "pred_x", "pred_y",
                     "gt_x", "gt_y", "gt_px_x", "gt_px_y"}


def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def analyze_file(name, path):
    print(f"\n{'='*90}\n{name}\n{path}\n{'='*90}")
    df = pd.read_json(path, lines=True)
    print(f"[+] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"[!] MISSING EXPECTED COLUMNS: {missing} -- schema doesn't "
              f"match what this script assumes. Stopping here rather than "
              f"computing numbers against the wrong fields.")
        return None

    # DIAGNOSTIC FIRST -- verify schema assumptions before trusting anything below.
    print("\n[diagnostic] gaze_type value counts:")
    print(df["gaze_type"].value_counts(dropna=False).to_string())
    print("\n[diagnostic] label_source value counts:")
    print(df["label_source"].value_counts(dropna=False).to_string())

    is_offscreen = (df["gt_px_x"] == -1) | (df["gt_px_y"] == -1)
    on_screen_df = df[~is_offscreen].copy()
    off_screen_df = df[is_offscreen].copy()
    print(f"\n[+] On-screen: {len(on_screen_df)}   Off-screen: {len(off_screen_df)}")

    if len(off_screen_df) > 0:
        # Adjust this condition if your schema signals off-screen
        # differently than gaze_type == "offscreen".
        correctly_flagged = (off_screen_df["gaze_type"] == "offscreen").sum()
        print(f"[+] Off-screen detection: {correctly_flagged}/{len(off_screen_df)} "
              f"correctly flagged as gaze_type == 'offscreen'")

    # Defensive check: any on-screen row unexpectedly missing a real
    # pred_x/pred_y that isn't a fallback? Shouldn't happen per the
    # pipeline's own routing logic, but worth knowing if it does.
    missing_pred = on_screen_df["pred_x"].isna() | on_screen_df["pred_y"].isna()
    if missing_pred.any():
        print(f"[!] {missing_pred.sum()} on-screen rows have missing "
              f"pred_x/pred_y -- excluding from ADE calculation, but worth "
              f"checking why these exist.")
        on_screen_df = on_screen_df[~missing_pred]

    on_screen_df["dist"] = on_screen_df.apply(
        lambda r: euclidean(r["pred_x"], r["pred_y"], r["gt_x"], r["gt_y"]), axis=1)

    is_fallback = on_screen_df["label_source"].isin(FALLBACK_LABEL_SOURCES)
    model_only_df = on_screen_df[~is_fallback]

    print(f"\n[+] On-screen breakdown by gaze_type:")
    print(f"{'gaze_type':<15} {'n_total':<8} {'n_model':<8} {'pipeline_ADE':<14} {'model_only_ADE'}")
    for gtype, group in on_screen_df.groupby("gaze_type"):
        model_group = group[~group["label_source"].isin(FALLBACK_LABEL_SOURCES)]
        pipeline_ade = group["dist"].mean()
        model_ade = model_group["dist"].mean() if len(model_group) > 0 else float("nan")
        print(f"{gtype:<15} {len(group):<8} {len(model_group):<8} "
              f"{pipeline_ade:<14.4f} {model_ade:.4f}")

    overall_pipeline_ade = on_screen_df["dist"].mean()
    overall_model_ade = model_only_df["dist"].mean() if len(model_only_df) > 0 else float("nan")
    print(f"\n[+] OVERALL on-screen pipeline ADE: {overall_pipeline_ade:.4f}  (n={len(on_screen_df)})")
    print(f"[+] OVERALL on-screen model-only ADE: {overall_model_ade:.4f}  (n={len(model_only_df)})")

    return {"name": name, "overall_pipeline_ade": overall_pipeline_ade,
            "overall_model_ade": overall_model_ade, "n_on_screen": len(on_screen_df)}


def main():
    results = []
    for name, path in LABEL_FILES.items():
        result = analyze_file(name, path)
        if result:
            results.append(result)

    print(f"\n\n{'='*90}\nSUMMARY -- overall on-screen ADE across all files\n{'='*90}")
    print(f"{'File':<50} {'Pipeline ADE':<15} {'Model-only ADE'}")
    for r in results:
        print(f"{r['name']:<50} {r['overall_pipeline_ade']:<15.4f} {r['overall_model_ade']:.4f}")

    print("\nFor reference: MTGS (fully-supervised specialist) reports 0.105 "
          "mean L2 distance on the full VAT test set (Table 3b) -- compare "
          "against OVERALL PIPELINE ADE above for the fairest comparison, "
          "since that's the same 'whatever the system actually outputs' "
          "measure MTGS's own number represents.")


if __name__ == "__main__":
    main()
