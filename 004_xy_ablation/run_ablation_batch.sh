#!/bin/bash
# run_ablation_batch.sh — runs the predicted-xy ablation (GRU/Transformer
# only, ResidualGRU/ResidualTransformer excluded by design -- see
# train_temporal_v1.py's own comment on why) across all 4 teacher
# conditions AND all 3 horizons, one model size. TRAINING ONLY -- no
# extraction call anywhere, since this reuses the already-cached
# embeddings from the original frozen-backbone batch. Output goes to a
# completely separate root, no dependency on wherever the original
# temporal_output_* results now live.
#
# Usage:
#   bash run_ablation_batch.sh $PYTHON_BIN <model_label> <feature_cache_root>

set -e

PYTHON_BIN=$1
MODEL_LABEL=$2            # e.g. "vittiny" or "vitsmall" -- output dir naming only
FEATURE_CACHE_ROOT=$3     # e.g. temporal_feature_cache_vittiny -- UNCHANGED, already exists

export VAT_TEMPORAL_MANIFEST_PATH="/parallel_scratch/sl02092/standard_project/temporal_manifest/vat_temporal_manifest.jsonl"
export CHILDPLAY_TEMPORAL_MANIFEST_PATH="/parallel_scratch/sl02092/standard_project/temporal_manifest/childplay_temporal_manifest.jsonl"
export VACATION_TEMPORAL_MANIFEST_PATH="/parallel_scratch/sl02092/standard_project/temporal_manifest/vacation_temporal_manifest.jsonl"

# Raw annotation roots -- needed by each dataset's own baseline/split
# modules (childplay_annotation_utils.py's clips.csv lookup among them).
# FOUND MISSING 2026-08-13: this wrapper was built from scratch and
# didn't carry over the same three exports already fixed in
# run_temporal_batch.sh -- same class of bug, different script.
export VAT_ROOT="/parallel_scratch/sl02092/standard_project/data/videoattentiontarget"
export CHILDPLAY_ROOT="/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze"
export VACATION_LOCAL_ANNOTATIONS_ROOT="/parallel_scratch/sl02092/standard_project/data/vacation/annotation"
# FOUND MISSING 2026-08-15 (diagnosed via the dissertation-writing chat,
# confirmed here directly against train_temporal_v1.py's own code): this
# is a SEPARATE variable from VAT_ROOT above, used only by
# JointTemporalDataset's dimension-resolution step (opens one sample
# image per clip to normalize target coordinates -- runs independently
# of extraction, so "training only, no extraction" doesn't make this
# redundant). Without it, every VAT clip failed that step, was dropped
# per-clip (caught, logged, didn't crash the run), and the ablation
# silently trained/evaluated on ChildPlay+VACATION only -- explains the
# train/val shrinking by different ratios (19% vs 24%): one whole
# dataset missing from both, at a different proportion of each split.
export VAT_IMAGES_BASE_PATH="/parallel_scratch/sl02092/standard_project/data/videoattentiontarget/images"

export TEMPORAL_DATASETS="vat,childplay,vacation"
export INCLUDE_PREDICTED_XY="0"
export FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT

OUTPUT_ROOT="/parallel_scratch/sl02092/standard_project/temporal_output_ablation_${MODEL_LABEL}"
CONDITIONS="gt teacher_hybrid teacher_internvl3_alone mtgs"
HORIZONS="100 300 500"

# ══════════════════════════════════════════════════════════════════════
# UPFRONT VALIDATION -- confirms the embedding cache actually exists for
# each condition before starting training. No extraction fallback here
# on purpose (that step already completed successfully for all 4
# conditions during the original batch) -- a missing cache means
# something else is wrong, worth surfacing immediately, not silently
# re-extracting.
# ══════════════════════════════════════════════════════════════════════
echo "--- Pre-flight checks ---"
FAIL=0
for f in "$VAT_TEMPORAL_MANIFEST_PATH" "$CHILDPLAY_TEMPORAL_MANIFEST_PATH" "$VACATION_TEMPORAL_MANIFEST_PATH"; do
    if [ ! -f "$f" ]; then
        echo "MISSING MANIFEST: $f"
        FAIL=1
    fi
done
for mod in temporal_window_dataset_vat temporal_window_dataset_childplay temporal_window_dataset_vacation \
           compute_last_known_position_baseline_vat compute_last_known_position_baseline_childplay \
           compute_last_known_position_baseline_vacation childplay_annotation_utils vacation_temporal_utils \
           gaze_student_model train_temporal_v1; do
    if [ ! -f "${mod}.py" ]; then
        echo "MISSING MODULE (must sit flat in this directory): ${mod}.py"
        FAIL=1
    fi
done
for CONDITION in $CONDITIONS; do
    for DATASET in vat childplay vacation; do
        if [ ! -f "${FEATURE_CACHE_ROOT}/${DATASET}/${CONDITION}/frame_features_index.jsonl" ]; then
            echo "MISSING CACHED EMBEDDINGS: ${FEATURE_CACHE_ROOT}/${DATASET}/${CONDITION}/ -- expected this to already exist from the original batch"
            FAIL=1
        fi
    done
done
for d in "$VAT_ROOT" "$CHILDPLAY_ROOT" "$VACATION_LOCAL_ANNOTATIONS_ROOT" "$VAT_IMAGES_BASE_PATH"; do
    if [ ! -d "$d" ]; then
        echo "MISSING ANNOTATION ROOT (directory doesn't exist): $d"
        FAIL=1
    fi
done
if [ "$FAIL" = "1" ]; then
    echo "Pre-flight checks FAILED -- fix the above before rerunning. No training attempted."
    exit 1
fi
echo "All manifests, modules, and cached embeddings found -- proceeding."
echo ""

for CONDITION in $CONDITIONS; do
    for HORIZON in $HORIZONS; do
        echo ""
        echo "########################################################################"
        echo "### ABLATION (no predicted-xy): $CONDITION / ${HORIZON}ms  ($(date))"
        echo "########################################################################"
        export STATIC_CONDITION=$CONDITION
        export TARGET_HORIZON_MS=$HORIZON
        export TEMPORAL_OUTPUT_DIR="${OUTPUT_ROOT}/${CONDITION}/horizon_${HORIZON}ms_ablation_no_xy"
        $PYTHON_BIN -u train_temporal_v1.py
    done
done

echo ""
echo "=== Ablation batch complete for ${MODEL_LABEL} ($(date)) ==="
echo "Results in: ${OUTPUT_ROOT}/<condition>/horizon_<X>ms_ablation_no_xy/temporal_comparison.json"
echo ""
echo "To compare against the ORIGINAL (with predicted-xy) result for the same"
echo "condition/horizon/architecture, look at whichever folder you moved the"
echo "original temporal_output_${MODEL_LABEL} results into."
