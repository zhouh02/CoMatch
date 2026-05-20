#!/bin/bash
# outdoor.sh wrapper with semantic cross-class matching statistics
#
# Usage:
#   # Standard evaluation (no semantic stats)
#   bash scripts/reproduce_test/outdoor.sh
#
#   # With semantic cross-class matching analysis
#   SEMANTIC_CACHE_DIR=outputs/semantic_cache \
#   SEMANTIC_IGNORE_LABELS="255,-1" \
#   SEMANTIC_CONF_THR=0.5 \
#   SEMANTIC_DUMP_NAME=semantic_matches \
#   DUMP_DIR=outputs/comatch_outdoor \
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

# Semantic consistency analysis settings
SEMANTIC_CACHE_DIR="${SEMANTIC_CACHE_DIR:-}"
SEMANTIC_IGNORE_LABELS="${SEMANTIC_IGNORE_LABELS:-255,-1}"
SEMANTIC_CONF_THR="${SEMANTIC_CONF_THR:-}"
SEMANTIC_DUMP_NAME="${SEMANTIC_DUMP_NAME:-semantic_matches}"
DUMP_DIR="${DUMP_DIR:-outputs/comatch_outdoor}"

# Build base args
BASE_ARGS="${data_cfg_path} ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${DUMP_DIR} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark \
    --megasize $size \
    --npe \
    --thr 0.1 \
    --ransac_times 5 \
    --deter"

# Check if semantic cache directory is provided
if [ -n "${SEMANTIC_CACHE_DIR}" ]; then
    echo "=========================================="
    echo "Semantic Cross-Class Matching Analysis"
    echo "=========================================="
    echo "  Cache Dir: ${SEMANTIC_CACHE_DIR}"
    echo "  Ignore Labels: ${SEMANTIC_IGNORE_LABELS}"
    echo "  Conf Thresh: ${SEMANTIC_CONF_THR:-disabled}"
    echo "  Dump Name: ${SEMANTIC_DUMP_NAME}"
    echo "  Dump Dir: ${DUMP_DIR}"
    echo "=========================================="

    # Build semantic args
    SEMANTIC_ARGS="--semantic_cache_dir=${SEMANTIC_CACHE_DIR} \
        --semantic_ignore_labels=${SEMANTIC_IGNORE_LABELS} \
        --semantic_dump_name=${SEMANTIC_DUMP_NAME}"

    if [ -n "${SEMANTIC_CONF_THR}" ]; then
        SEMANTIC_ARGS="${SEMANTIC_ARGS} --semantic_conf_thr=${SEMANTIC_CONF_THR}"
    fi

    echo "Running with semantic analysis..."
    python ./test.py ${BASE_ARGS} ${SEMANTIC_ARGS}
else
    echo "=========================================="
    echo "Standard CoMatch Evaluation (no semantic analysis)"
    echo "=========================================="
    echo "  Dump Dir: ${DUMP_DIR}"
    echo "=========================================="
    echo "To enable semantic analysis, set SEMANTIC_CACHE_DIR"
    echo "=========================================="

    python ./test.py ${BASE_ARGS}
fi