#!/bin/bash
# run_retrain_on_test_population_full_sweep.sh — full 4-condition x
# 3-horizon version (12 runs) of run_retrain_on_test_population.sh.
# Only two things changed from the focused 2-run version: CONDITIONS and
# looping over HORIZONS instead of a fixed TARGET_HORIZON_MS. Everything
# else (manifests, feature cache, pre-flight checks) identical.
#
# Usage:
#   bash run_retrain_on_test_population_full_sweep.sh $PYTHON_BIN <model_label> <manifest_dir>

set -e

PYTHON_BIN=$1
MODEL_LABEL=$2
MANIFEST_DIR=$3

if [ -z "$MANIFEST_DIR" ]; then
    echo "Usage: bash run_retrain_on_test_population_full_sweep.sh \$PYTHON_BIN <model_label> <manifest_dir>"
    exit 1
fi

export VAT_TEMPORAL_MANIFEST_PATH="${MANIFEST_DIR}/vat_test_from_static_temporal_manifest.jsonl"
export CHILDPLAY_TEMPORAL_MANIFEST_PATH="${MANIFEST_DIR}/childplay_test_from_static_temporal_manifest.jsonl"
export VACATION_TEMPORAL_MANIFEST_PATH="${MANIFEST_DIR}/vacation_test_from_static_temporal_manifest.jsonl"
export TEMPORAL_DATASETS="vat,childplay,vacation"
export FEATURE_CACHE_ROOT="/parallel_scratch/sl02092/standard_project/temporal_feature_cache_from_static_${MODEL_LABEL}"
export VAT_ROOT="/parallel_scratch/sl02092/standard_project/data/videoattentiontarget"
export VAT_IMAGES_BASE_PATH="/parallel_scratch/sl02092/standard_project/data/videoattentiontarget/images"
export CHILDPLAY_ROOT="/parallel_scratch/sl02092/standard_project/data/childplay/ChildPlay-gaze"
export VACATION_LOCAL_ANNOTATIONS_ROOT="/parallel_scratch/sl02092/standard_project/data/vacation/annotation"
export RETRAIN_ON_TEST_POPULATION="1"

echo "--- Pre-flight checks ---"
FAIL=0
for f in "$VAT_TEMPORAL_MANIFEST_PATH" "$CHILDPLAY_TEMPORAL_MANIFEST_PATH" "$VACATION_TEMPORAL_MANIFEST_PATH"; do
    if [ ! -f "$f" ]; then
        echo "MISSING MANIFEST: $f"
        FAIL=1
    fi
done
if [ ! -f "train_temporal_v1_on_test_population.py" ]; then
    echo "MISSING SCRIPT: train_temporal_v1_on_test_population.py (must sit flat in this directory)"
    FAIL=1
fi
for mod in gaze_student_model compute_last_known_position_baseline_childplay \
           compute_last_known_position_baseline_vacation childplay_annotation_utils \
           vacation_temporal_utils; do
    if [ ! -f "${mod}.py" ]; then
        echo "MISSING MODULE (must sit flat in this directory): ${mod}.py"
        FAIL=1
    fi
done
for d in "$VAT_ROOT" "$CHILDPLAY_ROOT" "$VACATION_LOCAL_ANNOTATIONS_ROOT" "$FEATURE_CACHE_ROOT"; do
    if [ ! -d "$d" ]; then
        echo "MISSING DIRECTORY: $d"
        FAIL=1
    fi
done
if [ "$FAIL" = "1" ]; then
    echo "Pre-flight checks FAILED -- fix the above before rerunning. No training attempted."
    exit 1
fi
echo "All manifests, script, modules, and directories found -- proceeding."
echo ""

RESULTS_ROOT="/parallel_scratch/sl02092/standard_project/temporal_retrain_on_test_${MODEL_LABEL}"

CONDITIONS="gt mtgs teacher_hybrid teacher_internvl3_alone"
HORIZONS="100 300 500"

for CONDITION in $CONDITIONS; do
    export STATIC_CONDITION=$CONDITION
    for HORIZON in $HORIZONS; do
        echo ""
        echo "########################################################################"
        echo "### RETRAIN ON TEST POPULATION: condition=$CONDITION horizon=${HORIZON}ms  ($(date))"
        echo "########################################################################"
        export TARGET_HORIZON_MS=$HORIZON
        export TEMPORAL_OUTPUT_DIR="${RESULTS_ROOT}/${CONDITION}/horizon_${HORIZON}ms"
        $PYTHON_BIN -u train_temporal_v1_on_test_population.py
    done
done

echo ""
echo "=== Retrain-on-test-population FULL SWEEP complete for ${MODEL_LABEL} ($(date)) ==="
echo "Results in: ${RESULTS_ROOT}/<condition>/horizon_<horizon>ms/temporal_comparison.json"
