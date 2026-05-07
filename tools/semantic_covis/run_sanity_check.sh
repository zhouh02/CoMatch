#!/bin/bash
#
# Sanity Check Script for Semantic Covisibility Tools
#
# Usage:
#   bash tools/semantic_covis/run_sanity_check.sh
#   bash tools/semantic_covis/run_sanity_check.sh --num-pairs 10 --device cuda
#
# This script tests the entire tools/semantic_covis pipeline without modifying CoMatch.
#

set -eo pipefail  # Exit on error and undefined variables

# =============================================================================
# Python Version Check
# =============================================================================

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo "UNKNOWN")

echo "========================================"
echo "Python Version Check"
echo "========================================"
echo "Current Python: ${PYTHON_VERSION}"
echo "当前项目运行在 Python 3.8，因此 tools/semantic_covis 代码必须兼容 Python 3.8。"
echo ""

# Check if Python version is at least 3.8
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)

if [[ "${PYTHON_MAJOR}" -lt 3 ]] || ([[ "${PYTHON_MAJOR}" -eq 3 ]] && [[ "${PYTHON_MINOR}" -lt 8 ]]); then
    echo "ERROR: Python 3.8 or higher is required. Current version: ${PYTHON_VERSION}"
    exit 1
fi

echo "Python version check passed."
echo ""

# =============================================================================
# Parse Arguments
# =============================================================================

NPZ_ROOT="${NPZ_ROOT:-data/megadepth/index/scene_info_0.1_0.7}"
TRAIN_LIST="${TRAIN_LIST:-data/megadepth/index/trainvaltest_list/train_list.txt}"
IMAGE_ROOT="${IMAGE_ROOT:-data/megadepth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sem_covis_sanity}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-large-patch14-336}"
DEVICE="${DEVICE:-cuda}"
NUM_PAIRS="${NUM_PAIRS:-5}"
SEED="${SEED:-42}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --npz-root) NPZ_ROOT="$2"; shift 2 ;;
        --train-list) TRAIN_LIST="$2"; shift 2 ;;
        --image-root) IMAGE_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num-pairs) NUM_PAIRS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --help) echo "Usage: $0 [--npz-root PATH] [--train-list PATH] [--image-root PATH] [--output-dir PATH] [--model-name NAME] [--device DEVICE] [--num-pairs N] [--seed N]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${OUTPUT_DIR}/sanity_check.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/preprocess_test"

# Function to log messages
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "${LOG_FILE}"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$msg" | tee -a "${LOG_FILE}" >&2
}

# Clear log file
> "${LOG_FILE}"

# =============================================================================
# Main
# =============================================================================

log "========================================"
log "Semantic Covisibility Sanity Check"
log "========================================"

log ""
log "Parameters:"
log "  --npz-root:    ${NPZ_ROOT}"
log "  --train-list:  ${TRAIN_LIST}"
log "  --image-root:  ${IMAGE_ROOT}"
log "  --output-dir:  ${OUTPUT_DIR}"
log "  --model-name:  ${MODEL_NAME}"
log "  --device:      ${DEVICE}"
log "  --num-pairs:   ${NUM_PAIRS}"
log "  --seed:        ${SEED}"
log ""

# =============================================================================
# Step 1: Environment Info
# =============================================================================

log "----------------------------------------"
log "Step 1: Environment Information"
log "----------------------------------------"

log "Current directory: $(pwd)"
log "Python path: $(which python3 || which python)"
log "Python version: ${PYTHON_VERSION}"
log "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
log "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'UNKNOWN')"

# Check transformers
TRANSFORMERS_VERSION=$(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo "NOT INSTALLED")
log "Transformers version: ${TRANSFORMERS_VERSION}"

log ""

# =============================================================================
# Step 2: Syntax Check
# =============================================================================

log "----------------------------------------"
log "Step 2: Python Syntax Check"
log "----------------------------------------"

PY_FILES=(
    "${SCRIPT_DIR}/export_megadepth_pairs.py"
    "${SCRIPT_DIR}/image_preprocess.py"
    "${SCRIPT_DIR}/clip_feature_extractor.py"
)

SYNTAX_OK=true
for py_file in "${PY_FILES[@]}"; do
    if [[ -f "$py_file" ]]; then
        filename=$(basename "$py_file")
        if python3 -m py_compile "$py_file" 2>&1; then
            log "  [OK] ${filename}"
        else
            log_error "  [FAIL] ${filename}"
            SYNTAX_OK=false
        fi
    else
        log_error "  [MISSING] $(basename "$py_file")"
        SYNTAX_OK=false
    fi
done | tee -a "${LOG_FILE}"

if [[ "$SYNTAX_OK" != "true" ]]; then
    log_error "Syntax check failed!"
    exit 1
fi

log "  All Python files passed syntax check."
log ""

# =============================================================================
# Step 3: Export Image Pairs
# =============================================================================

log "----------------------------------------"
log "Step 3: Export Image Pairs"
log "----------------------------------------"

DEBUG_PAIRS="${OUTPUT_DIR}/debug_pairs.txt"

log "Running export_megadepth_pairs.py..."

# Run export script - if it fails, script exits immediately due to set -e
python3 "${SCRIPT_DIR}/export_megadepth_pairs.py" \
    --npz-root "${NPZ_ROOT}" \
    --train-list "${TRAIN_LIST}" \
    --image-root "${IMAGE_ROOT}" \
    --output "${DEBUG_PAIRS}" \
    --num-pairs "${NUM_PAIRS}" \
    --min-overlap 0.1 \
    --shuffle \
    --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

# Check exit status
PIPE_STATUS=(${PIPESTATUS[@]})
EXIT_CODE=${PIPE_STATUS[0]}

if [[ ${EXIT_CODE} -ne 0 ]]; then
    log_error "Export failed with exit code: ${EXIT_CODE}"
    log_error "Check log file for details: ${LOG_FILE}"
    exit 1
fi

log "  Export completed successfully."
log ""

# =============================================================================
# Step 4: Print Debug Pairs
# =============================================================================

log "----------------------------------------"
log "Step 4: Debug Pairs (First 5 lines)"
log "----------------------------------------"

if [[ -f "${DEBUG_PAIRS}" ]]; then
    log "File: ${DEBUG_PAIRS}"
    log "Content:"
    head -n 6 "${DEBUG_PAIRS}" | while IFS= read -r line; do
        log "  $line"
    done
else
    log_error "Debug pairs file not found: ${DEBUG_PAIRS}"
    exit 1
fi

log ""

# =============================================================================
# Step 5: Extract Image Paths from First Pair
# =============================================================================

log "----------------------------------------"
log "Step 5: Extract Image Paths from First Pair"
log "----------------------------------------"

# Skip header line, get first pair
FIRST_PAIR=$(sed -n '2p' "${DEBUG_PAIRS}")

if [[ -z "$FIRST_PAIR" ]]; then
    log_error "No pairs found in debug file!"
    exit 1
fi

# Parse: pair_id scene_id pair_idx idx0 idx1 image0_path image1_path
read -r pair_id scene_id pair_idx idx0 idx1 image0_path image1_path <<< "$FIRST_PAIR"

log "First pair:"
log "  pair_id:       ${pair_id}"
log "  scene_id:      ${scene_id}"
log "  image0_path:   ${image0_path}"
log "  image1_path:   ${image1_path}"

# Construct full paths
FULL_IMAGE0="${IMAGE_ROOT}/${image0_path}"
FULL_IMAGE1="${IMAGE_ROOT}/${image1_path}"

log "Full paths:"
log "  image0: ${FULL_IMAGE0}"
log "  image1: ${FULL_IMAGE1}"

# Check if images exist
if [[ ! -f "$FULL_IMAGE0" ]]; then
    log_error "Image 0 not found: ${FULL_IMAGE0}"
    exit 1
fi

if [[ ! -f "$FULL_IMAGE1" ]]; then
    log_error "Image 1 not found: ${FULL_IMAGE1}"
    exit 1
fi

log ""

# =============================================================================
# Step 6: Test Image Preprocessing
# =============================================================================

log "----------------------------------------"
log "Step 6: Test Image Preprocessing"
log "----------------------------------------"

log "Testing image_preprocess.py with first image..."

python3 "${SCRIPT_DIR}/image_preprocess.py" \
    --image "${FULL_IMAGE0}" \
    --output-dir "${OUTPUT_DIR}/preprocess_test" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
if [[ ${EXIT_CODE} -ne 0 ]]; then
    log_error "Image preprocessing (image 0) failed with exit code: ${EXIT_CODE}"
    exit 1
fi

log "  Image preprocessing (image 0) completed."

log "Testing image_preprocess.py with second image..."

python3 "${SCRIPT_DIR}/image_preprocess.py" \
    --image "${FULL_IMAGE1}" \
    --output-dir "${OUTPUT_DIR}/preprocess_test" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
if [[ ${EXIT_CODE} -ne 0 ]]; then
    log_error "Image preprocessing (image 1) failed with exit code: ${EXIT_CODE}"
    exit 1
fi

log "  Image preprocessing (image 1) completed."
log ""

# =============================================================================
# Step 7: Test CLIP Feature Extractor
# =============================================================================

log "----------------------------------------"
log "Step 7: Test CLIP Feature Extractor"
log "----------------------------------------"

# Helper function to test CLIP extractor with proper exit code handling
test_clip_extractor() {
    local image_path="$1"
    local image_label="$2"

    log "Testing clip_feature_extractor.py with ${image_label}..."

    # Run command, capture both stdout and exit code
    local output
    local exit_code

    output=$("${SCRIPT_DIR}/clip_feature_extractor.py" \
        --image "${image_path}" \
        --model-name "${MODEL_NAME}" \
        --device "${DEVICE}" 2>&1)
    exit_code=$?

    # Print full output to log
    echo "${output}" | tee -a "${LOG_FILE}"

    # Check exit code
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "CLIP feature extraction (${image_label}) failed with exit code: ${exit_code}"
        exit 1
    fi

    # Check for success indicators
    if echo "${output}" | grep -q "Test completed successfully"; then
        log "  ${image_label} extraction succeeded."
    else
        log_error "CLIP output missing 'Test completed successfully'!"
        exit 1
    fi

    # Extract and log key metrics (handles both "(1, 576, 1024)" and "(576, 1024)" formats)
    local patch_shape=$(echo "${output}" | grep "patch_features.shape:" | head -1 | awk '{print $2}')
    local grid_size=$(echo "${output}" | grep "grid_size:" | head -1 | awk '{print $2}')
    local valid_ratio=$(echo "${output}" | grep "valid_ratio:" | head -1 | awk '{print $2}')

    log "  ${image_label} CLIP results:"
    log "    patch_features.shape: ${patch_shape}"
    log "    grid_size: ${grid_size}"
    log "    valid_ratio: ${valid_ratio}"

    # Verify grid_size is (24, 24)
    if [[ "${grid_size}" == "(24,"* ]]; then
        log "    grid_size check: PASS (expected (24, 24))"
    else
        log_error "CLIP grid_size mismatch! Expected (24, 24), got: ${grid_size}"
        exit 1
    fi

    # Verify patch_features shape is valid (accept both formats)
    if [[ -z "${patch_shape}" ]]; then
        log_error "Failed to extract patch_features.shape from CLIP output!"
        exit 1
    fi

    echo "${output}"
}

# Test first image
test_clip_extractor "${FULL_IMAGE0}" "image 0"

log ""

# Test second image
test_clip_extractor "${FULL_IMAGE1}" "image 1"

log ""

# =============================================================================
# Summary
# =============================================================================

log "========================================"
log "SANITY CHECK PASSED"
log "========================================"
log ""
log "Summary:"
log "  - Python syntax: OK"
log "  - Exported pairs: ${NUM_PAIRS}"
log "  - CLIP model: ${MODEL_NAME}"
log "  - CLIP device: ${DEVICE}"
log "  - Image 0 patch features: ${PATCH_SHAPE}"
log "  - Image 0 grid size: ${GRID_SIZE}"
log "  - Image 1 patch features: ${PATCH_SHAPE2}"
log "  - Image 1 grid size: ${GRID_SIZE2}"
log "  - Log file: ${LOG_FILE}"
log ""
log "All tests completed successfully!"

exit 0
