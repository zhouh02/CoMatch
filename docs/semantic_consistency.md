# Semantic Consistency Analysis for CoMatch

This document describes the complete workflow for analyzing semantic cross-class matching in CoMatch evaluations.

## Overview

The semantic consistency analysis pipeline consists of three steps:

1. **Precompute semantic labels** using OneFormer
2. **Validate the semantic cache**
3. **Run CoMatch evaluation with semantic statistics**

## Important: Coordinate System

**Critical Design Decision:**

- CoMatch may resize input images to a longer edge of 832 and pad to 832x832 for inference
- However, the output keypoints `mkpts0_f` / `mkpts1_f` are **already mapped back** to the original MegaDepth image coordinates using `scale0` / `scale1`
- Therefore, semantic label maps must be in **original image resolution** (H_orig x W_orig)
- When checking semantic consistency, use original coordinates directly:

```python
label0 = sem0[int(y0), int(x0)]
label1 = sem1[int(y1), int(x1)]
```

**DO NOT:**
- Use image0/image1 tensor dimensions as semantic label map size
- Map keypoints back to 832/padded dimensions
- Multiply mkpts*_f by scale0/scale1 again

## Step 1: Precompute OneFormer Semantic Cache

### Usage

```bash
python scripts/precompute_oneformer_megadepth_semantics.py \
    --data_cfg configs/data/megadepth_test_1500.py \
    --cache_dir outputs/semantic_cache \
    --model_id shi-labs/oneformer_ade20k_swin_large \
    --device cuda
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_cfg` | MegaDepth data config path | Required |
| `--cache_dir` | Output directory for cache | `outputs/semantic_cache` |
| `--model_id` | Hugging Face model id | `shi-labs/oneformer_ade20k_swin_large` |
| `--device` | Device to run on | `cuda` |
| `--batch_size` | Batch size | 1 |
| `--max_images` | Limit images (debugging) | None |
| `--overwrite` | Overwrite existing files | False |
| `--image_list` | Optional image list file | None |

### Cache Structure

```
outputs/semantic_cache/
├── manifest.json        # Image path -> npz mapping
├── metadata.json        # Model info, id2label mapping
└── labels/
    ├── abc123def456.npz  # Semantic label maps
    └── ...
```

### manifest.json Format

```json
{
  "0015/images/DSC_0583.JPG": "labels/abc123def456.npz"
}
```

Keys are relative paths from the MegaDepth root directory.

### npz Contents

Each npz file contains:

| Key | Type | Description |
|-----|------|-------------|
| `label` | uint16/int32 | H x W semantic class ID map |
| `image_path` | str | Original image path |
| `height` | int | Original image height |
| `width` | int | Original image width |
| `model_id` | str | OneFormer model used |
| `task` | str | "semantic" |

## Step 2: Validate the Cache

After precomputing (or before using), validate the cache:

```bash
python scripts/precompute_oneformer_megadepth_semantics.py \
    --cache_dir outputs/semantic_cache \
    --validate_only
```

This checks:
- All manifest entries have corresponding npz files
- Each npz has a valid `label` array
- Label shapes match stored metadata
- Original images (if accessible) have matching dimensions

## Step 3: Run CoMatch with Semantic Analysis

### Command Line

```bash
python test.py \
    configs/data/megadepth_test_1500.py \
    configs/loftr/comatch_full.py \
    --ckpt_path=weights/comatch_outdoor.ckpt \
    --semantic_cache_dir outputs/semantic_cache \
    --semantic_ignore_labels "255,-1" \
    --semantic_conf_thr 0.5 \
    --semantic_dump_name my_semantic \
    --dump_dir outputs/comatch_results
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--semantic_cache_dir` | Semantic cache directory | Required for semantic analysis |
| `--semantic_ignore_labels` | Comma-separated labels to ignore | None |
| `--semantic_conf_thr` | Minimum confidence threshold | None |
| `--semantic_dump_name` | Base name for output files | `semantic_matches` |

### Output Files

```
outputs/comatch_results/
├── semantic_matches.jsonl       # Per-pair detailed stats
├── semantic_matches_summary.json # Aggregated summary
└── semantic_matches_rank*.jsonl  # Per-rank files (DDP)
```

### semantic_matches_summary.json Format

```json
{
  "num_pairs": 1500,
  "num_skipped_pairs": 10,
  "total_matches": 123456,
  "total_after_conf": 120000,
  "total_in_bounds": 118000,
  "total_valid_semantic": 115000,
  "total_same_semantic": 98000,
  "total_cross_semantic": 17000,
  "micro_cross_semantic_rate": 0.1478,
  "macro_cross_semantic_rate": 0.1523
}
```

## Using Shell Script

For convenience, use the provided shell script:

```bash
# Standard evaluation
bash scripts/reproduce_test/outdoor_semantic.sh

# With semantic analysis
SEMANTIC_CACHE_DIR=outputs/semantic_cache \
SEMANTIC_IGNORE_LABELS="255,-1" \
DUMP_DIR=outputs/comatch_results \
bash scripts/reproduce_test/outdoor_semantic.sh
```

## Environment Variables for outdoor_semantic.sh

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CACHE_DIR` | (empty) | Cache directory (must be set to enable) |
| `SEMANTIC_IGNORE_LABELS` | `255,-1` | Labels to ignore |
| `SEMANTIC_CONF_THR` | (empty) | Confidence threshold |
| `SEMANTIC_DUMP_NAME` | `semantic_matches` | Output file base name |
| `DUMP_DIR` | `outputs/comatch_outdoor` | Dump directory |

## Statistics Explained

| Statistic | Description |
|-----------|-------------|
| `num_matches` | Total matches from CoMatch |
| `num_after_conf` | Matches after confidence filtering |
| `num_in_bounds` | Matches with coordinates in image bounds |
| `num_valid_semantic` | Matches with non-ignored semantic labels |
| `num_same_semantic` | Matches where both points have same label |
| `num_cross_semantic` | Matches where labels differ |
| `cross_semantic_rate` | `num_cross_semantic / num_valid_semantic` |
| `micro_cross_semantic_rate` | Total cross / Total valid (global) |
| `macro_cross_semantic_rate` | Mean of per-pair cross rates |

## Troubleshooting

### "No semantic label found for: ..."

The manifest doesn't contain an entry for the image. Make sure:
1. The semantic cache was computed for all test images
2. The manifest keys match the paths used in CoMatch (check prefix handling)

### "Ambiguous path: ..."

Multiple images in the manifest have the same suffix. Use unique scene/image paths.

### "Cross semantic rate is too high"

This could indicate:
- OneFormer errors on certain classes
- CoMatch matching across class boundaries legitimately
- Label map resolution mismatch

Check that label maps are in original image resolution, not 832x832.

## ScanNet Support

The semantic consistency analysis also supports ScanNet dataset evaluation.

### Precompute OneFormer Semantic Cache for ScanNet

```bash
python scripts/precompute_oneformer_scannet_semantics.py \
    --data_cfg configs/data/scannet_test_1500.py \
    --cache_dir outputs/scannet_semantic_cache \
    --model_id shi-labs/oneformer_ade20k_swin_large \
    --device cuda
```

### Run ScanNet Evaluation with Semantic Analysis

```bash
python test.py \
    configs/data/scannet_test_1500.py \
    configs/loftr/comatch_full.py \
    --ckpt_path=weights/comatch_outdoor.ckpt \
    --semantic_cache_dir outputs/scannet_semantic_cache \
    --semantic_ignore_labels "255,-1" \
    --semantic_dump_name scannet_semantic \
    --dump_dir outputs/comatch_scannet \
    --scannetX 640 \
    --scannetY 480 \
    --thr 0.2
```

### Using Shell Script

```bash
# Standard evaluation
bash scripts/reproduce_test/indoor_semantic.sh

# With semantic analysis
SEMANTIC_CACHE_DIR=outputs/scannet_semantic_cache \
SEMANTIC_IGNORE_LABELS="255,-1" \
DUMP_DIR=outputs/comatch_scannet \
bash scripts/reproduce_test/indoor_semantic.sh
```

### Environment Variables for indoor_semantic.sh

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CACHE_DIR` | (empty) | Cache directory (must be set to enable) |
| `SEMANTIC_IGNORE_LABELS` | `255,-1` | Labels to ignore |
| `SEMANTIC_CONF_THR` | (empty) | Confidence threshold |
| `SEMANTIC_DUMP_NAME` | `semantic_matches` | Output file base name |
| `DUMP_DIR` | `outputs/comatch_full_scannet` | Dump directory |

## ScanNet Analysis & Visualization

### Cross-Class Analysis

Analyze which semantic class combinations most frequently appear in cross-semantic matches:

```bash
python scripts/analyze_scannet_cross_class.py \
    --semantic_matches_jsonl outputs/comatch_full_scannet/semantic_matches.jsonl \
    --semantic_cache_dir outputs/scannet_semantic_cache \
    --output_dir outputs/scannet_cross_class_analysis \
    --top_n 30
```

Output:
- `summary.json` — Aggregate statistics
- `top_cross_pairs.json` — Top pairs by cross-semantic rate

### Visualize Cross-Semantic Matches

Generate side-by-side images with match lines and segmentation overlays:

```bash
python scripts/visualize_scannet_cross_matches.py \
    --data_cfg configs/data/scannet_test_1500.py \
    --main_cfg configs/loftr/comatch_full.py \
    --ckpt_path weights/comatch_outdoor.ckpt \
    --semantic_cache_dir outputs/scannet_semantic_cache \
    --semantic_matches_jsonl outputs/comatch_full_scannet/semantic_matches.jsonl \
    --output_dir outputs/scannet_semantic_vis \
    --top_n 20 \
    --scannetX 640 \
    --scannetY 480
```

Output:
- `matches/` — Original images with red lines (cross-semantic) and green lines (same-semantic)
- `semseg/` — OneFormer segmentation overlay with colored match points
- `index.json` — Metadata for all visualized pairs

### ScanNet Coordinate System

Unlike MegaDepth where mkpts*_f are directly in original coordinates, ScanNet requires an extra mapping step:

1. ScanNet original images have varying resolutions (e.g. 1296x968)
2. `read_scannet_gray()` resizes to (scannetX, scannetY) = (640, 480)
3. CoMatch pads/resizes to 832x832 for inference
4. `mkpts*_f` are mapped back to the resized (640x480) space via `scale0/scale1`
5. OneFormer labels are in **original image resolution**
6. Visualization maps: `x_orig = x_mkpt * (orig_w / 640)`, `y_orig = y_mkpt * (orig_h / 480)`

## Dependencies

- `transformers` (for OneFormer)
- `torch`
- `numpy`
- `PIL` (for image validation)

Without transformers, the CoMatch evaluation still works normally (just without semantic statistics).