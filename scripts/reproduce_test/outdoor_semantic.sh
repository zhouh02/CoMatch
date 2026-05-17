#!/bin/bash
# outdoor.sh wrapper with semantic diagnostic support
#
# Usage:
#   # Standard evaluation (backward compatible)
#   bash scripts/reproduce_test/outdoor.sh
#
#   # With semantic diagnostic
#   SEMANTIC_DIAGNOSTIC=1 \
#   ONEFORMER_DIR=outputs/oneformer_outdoor_test \
#   SEMANTIC_DIAGNOSTIC_OUTPUT=outputs/semantic_diagnostic_outdoor \
#   SEMANTIC_SCORE_THRESH=0.0 \
#   bash scripts/reproduce_test/outdoor_semantic.sh

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

main_cfg_path="configs/loftr/comatch_full.py"
profiler_name="inference"
n_nodes=1
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1

ckpt_path="weights/comatch_outdoor.ckpt"
data_cfg_path="configs/data/megadepth_test_1500.py"
size="1152"

# Check if semantic diagnostic is enabled
if [ "${SEMANTIC_DIAGNOSTIC:-0}" = "1" ]; then
    echo "=========================================="
    echo "Semantic Diagnostic Mode Enabled"
    echo "=========================================="
    echo "  ONEFORMER_DIR: ${ONEFORMER_DIR:-not set}"
    echo "  OUTPUT_DIR: ${SEMANTIC_DIAGNOSTIC_OUTPUT:-not set}"
    echo "  SCORE_THRESH: ${SEMANTIC_SCORE_THRESH:-0.0}"
    echo "=========================================="

    # Build semantic diagnostic args
    SEMANTIC_ARGS="--semantic-diag"
    if [ -n "${ONEFORMER_DIR}" ]; then
        SEMANTIC_ARGS="${SEMANTIC_ARGS} --oneformer-dir=${ONEFORMER_DIR}"
    fi
    if [ -n "${SEMANTIC_DIAGNOSTIC_OUTPUT}" ]; then
        SEMANTIC_ARGS="${SEMANTIC_ARGS} --semantic-output-dir=${SEMANTIC_DIAGNOSTIC_OUTPUT}"
    fi
    if [ -n "${SEMANTIC_SCORE_THRESH}" ]; then
        SEMANTIC_ARGS="${SEMANTIC_ARGS} --semantic-score-thresh=${SEMANTIC_SCORE_THRESH}"
    fi

    # Use test_semantic.py
    echo "Running with test_semantic.py..."
    python ./test_semantic.py \
        ${data_cfg_path} \
        ${main_cfg_path} \
        --ckpt_path=${ckpt_path} \
        --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
        --batch_size=${batch_size} --num_workers=${torch_num_workers}\
        --profiler_name=${profiler_name} \
        --benchmark \
        --megasize $size \
        --npe \
        --thr 0.1 \
        --ransac_times 5 \
        --deter \
        ${SEMANTIC_ARGS}
else
    # Standard evaluation
    python ./test.py \
        ${data_cfg_path} \
        ${main_cfg_path} \
        --ckpt_path=${ckpt_path} \
        --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
        --batch_size=${batch_size} --num_workers=${torch_num_workers}\
        --profiler_name=${profiler_name} \
        --benchmark \
        --megasize $size \
        --npe \
        --thr 0.1 \
        --ransac_times 5 \
        --deter
fi