#!/usr/bin/env python3
"""
Precompute OneFormer semantic segmentation for MegaDepth test images.

This script runs OneFormer on all unique images from the MegaDepth test set
and saves the semantic label maps in original image resolution.

Usage:
    # Precompute semantic cache
    python scripts/precompute_oneformer_megadepth_semantics.py \
        --data_cfg configs/data/megadepth_test_1500.py \
        --cache_dir outputs/semantic_cache \
        --model_id shi-labs/oneformer_ade20k_swin_large \
        --device cuda

    # Validate existing cache
    python scripts/precompute_oneformer_megadepth_semantics.py \
        --cache_dir outputs/semantic_cache \
        --validate_only

    # Process only specific images
    python scripts/precompute_oneformer_megadepth_semantics.py \
        --image_list path/to/images.txt \
        --cache_dir outputs/semantic_cache

IMPORTANT:
- OneFormer runs on ORIGINAL image resolution (H_orig x W_orig)
- Saved label maps have the same shape as original images
- CoMatch mkpts*_f are in original image coordinates
- Query labels with: label[y, x]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import (
    normalize_semantic_path_key,
    get_path_suffix,
    compute_file_hash,
    validate_semantic_cache,
)


def collect_megadepth_test_images(data_cfg_path: str) -> Tuple[List[str], int, int]:
    """
    Collect unique image paths from MegaDepth test configuration.

    Args:
        data_cfg_path: Path to MegaDepth data config file

    Returns:
        Tuple of (image_paths, num_pairs, num_unique_images)
    """
    from omegaconf import OmegaConf
    from src.datasets.megadepth import MegaDepthDataset
    import torch

    # Load config
    config = OmegaConf.load(data_cfg_path)
    test_cfg = OmegaConf.merge(
        OmegaConf.load("configs/data/base.py"),
        config
    )

    data_root = test_cfg.cfg.DATASET.TEST_DATA_ROOT
    npz_root = test_cfg.cfg.DATASET.TEST_NPZ_ROOT
    scene_list_path = test_cfg.cfg.DATASET.TEST_LIST_PATH

    # Read scene list
    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]

    image_paths = []
    num_pairs = 0

    print(f"Collecting images from {len(npz_names)} scenes...")

    for scene_name in tqdm(npz_names, desc="Scanning scenes"):
        npz_path = os.path.join(npz_root, f"{scene_name}.npz")
        if not os.path.exists(npz_path):
            print(f"Warning: Scene file not found: {npz_path}")
            continue

        # Load scene info without creating full dataset
        scene_data = np.load(npz_path, allow_pickle=True)

        if 'pair_infos' not in scene_data:
            print(f"Warning: No pair_infos in {npz_path}")
            continue

        pair_infos = scene_data['pair_infos']
        num_pairs += len(pair_infos)

        for pair_info in pair_infos:
            idx0, idx1 = pair_info[0]
            img_path_0 = scene_data['image_paths'][idx0]
            img_path_1 = scene_data['image_paths'][idx1]

            # Make absolute paths
            abs_path_0 = os.path.join(data_root, img_path_0)
            abs_path_1 = os.path.join(data_root, img_path_1)

            image_paths.append(abs_path_0)
            image_paths.append(abs_path_1)

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in image_paths:
        # Normalize for deduplication
        norm = normalize_semantic_path_key(p)
        if norm not in seen:
            seen.add(norm)
            unique_paths.append(p)

    return unique_paths, num_pairs, len(unique_paths)


def load_image_list(image_list_path: str) -> List[str]:
    """Load image paths from a text file (one path per line)."""
    with open(image_list_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def run_oneformer_on_image(
    image_path: str,
    model,
    processor,
    device: str,
) -> Tuple[np.ndarray, int, int]:
    """
    Run OneFormer on a single image and return semantic segmentation.

    Args:
        image_path: Path to input image
        model: OneFormerForUniversalSegmentation model
        processor: OneFormerProcessor
        device: Device to run on

    Returns:
        Tuple of (label_map, height, width) where label_map is H x W
    """
    from PIL import Image

    # Load image to get original size
    image = Image.open(image_path).convert("RGB")
    width, height = image.size  # PIL uses (width, height)

    # Process image
    inputs = processor(
        images=image,
        task_inputs=["semantic"],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to original size
    seg = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(height, width)],  # (H, W)
    )[0]

    # Convert to numpy
    label_map = seg.cpu().numpy()

    # Ensure correct shape (H, W)
    assert label_map.shape == (height, width), \
        f"Expected shape {(height, width)}, got {label_map.shape}"

    return label_map, height, width


def save_semantic_label(
    image_path: str,
    label_map: np.ndarray,
    height: int,
    width: int,
    cache_dir: str,
    model_id: str,
    manifest: dict,
) -> str:
    """
    Save semantic label to npz file and update manifest.

    Args:
        image_path: Original image path (used as manifest key)
        label_map: H x W label map
        height: Original image height
        width: Original image width
        cache_dir: Cache directory
        model_id: OneFormer model id
        manifest: Manifest dict to update

    Returns:
        Relative npz path from cache_dir
    """
    cache_dir = Path(cache_dir)
    labels_dir = cache_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename based on image hash
    file_hash = compute_file_hash(image_path)
    npz_filename = f"{file_hash}.npz"
    npz_path = labels_dir / npz_filename

    # Save npz
    np.savez_compressed(
        npz_path,
        label=label_map.astype(np.uint16) if label_map.max() < 65535 else label_map.astype(np.int32),
        image_path=image_path,
        height=height,
        width=width,
        model_id=model_id,
        task="semantic",
    )

    # Create manifest key (relative path from megadepth root)
    # Use normalized path as key
    manifest_key = normalize_semantic_path_key(image_path)

    # Try to make it relative to megadepth root for cleaner keys
    rel_path = manifest_key
    for prefix in ['data/megadepth/', 'data\\megadepth\\']:
        if manifest_key.lower().startswith(prefix.lower()):
            rel_path = manifest_key[len(prefix):]
            break

    # Update manifest
    manifest[rel_path] = f"labels/{npz_filename}"

    return npz_path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute OneFormer semantic segmentation for MegaDepth test images"
    )
    parser.add_argument(
        '--data_cfg',
        type=str,
        help='MegaDepth data config path (e.g., configs/data/megadepth_test_1500.py)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='outputs/semantic_cache',
        help='Output directory for semantic cache'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='shi-labs/oneformer_ade20k_swin_large',
        help='Hugging Face model id for OneFormer'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (default 1)'
    )
    parser.add_argument(
        '--image_list',
        type=str,
        default=None,
        help='Optional path to text file with image paths (one per line)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Limit number of images to process (for debugging)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing npz files'
    )
    parser.add_argument(
        '--validate_only',
        action='store_true',
        help='Only validate existing cache, do not run inference'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of dataloader workers (default 0)'
    )

    args = parser.parse_args()

    # Validate only mode
    if args.validate_only:
        print("=" * 60)
        print("Validating Semantic Cache")
        print("=" * 60)
        results = validate_semantic_cache(args.cache_dir)

        print(f"\nResults:")
        print(f"  Total entries: {results['total_entries']}")
        print(f"  Valid entries: {results['valid_entries']}")
        print(f"  Missing files: {len(results['missing_files'])}")
        print(f"  Bad shapes: {len(results['bad_shapes'])}")
        print(f"  Unreadable images: {len(results['unreadable_images'])}")

        if results['missing_files']:
            print(f"\nMissing files (first 5):")
            for p in results['missing_files'][:5]:
                print(f"  - {p}")

        if results['bad_shapes']:
            print(f"\nBad shapes (first 5):")
            for p, expected, actual in results['bad_shapes'][:5]:
                print(f"  - {p}: expected {expected}, got {actual}")

        if not results['valid']:
            print("\n[FAILED] Cache validation failed")
            sys.exit(1)
        else:
            print("\n[PASSED] Cache validation passed")
            sys.exit(0)

    # --- Normal processing mode ---
    if not args.data_cfg and not args.image_list:
        parser.error("--data_cfg or --image_list is required")

    # Collect image paths
    if args.image_list:
        print(f"Loading image list from: {args.image_list}")
        image_paths = load_image_list(args.image_list)
        num_pairs = len(image_paths) // 2  # Approximate
        num_unique = len(image_paths)
    else:
        print(f"Loading MegaDepth test images from config: {args.data_cfg}")
        image_paths, num_pairs, num_unique = collect_megadepth_test_images(args.data_cfg)

    print(f"\nSummary:")
    print(f"  Num pairs: {num_pairs}")
    print(f"  Num unique images: {num_unique}")

    if args.max_images:
        image_paths = image_paths[:args.max_images]
        print(f"  Limited to first {args.max_images} images")

    if not image_paths:
        print("No images to process")
        sys.exit(0)

    print(f"\nFirst few images:")
    for p in image_paths[:5]:
        print(f"  - {p}")

    # Setup output directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load or create manifest
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"\nLoaded existing manifest with {len(manifest)} entries")
    else:
        manifest = {}
        print("\nCreating new manifest")

    # Load metadata
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_id": args.model_id,
            "task": "semantic",
            "label_space": "ade20k",
        }

    # Load OneFormer
    print("\n" + "=" * 60)
    print(f"Loading OneFormer model: {args.model_id}")
    print("=" * 60)

    try:
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        import torch
    except ImportError as e:
        print(f"\nERROR: transformers not installed: {e}")
        print("Please install: pip install transformers")
        sys.exit(1)

    processor = OneFormerProcessor.from_pretrained(args.model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(args.model_id)
    model = model.to(args.device)
    model.eval()

    # Save model info to metadata
    metadata["model_id"] = args.model_id
    if hasattr(model.config, 'id2label') and model.config.id2label:
        metadata["id2label"] = {str(k): v for k, v in model.config.id2label.items()}

    # Counters
    processed = 0
    skipped = 0
    errors = 0

    print("\n" + "=" * 60)
    print("Processing Images")
    print("=" * 60)

    for img_path in tqdm(image_paths, desc="Running OneFormer"):
        # Check if already processed
        if not args.overwrite:
            norm_key = normalize_semantic_path_key(img_path)
            # Try both absolute and relative keys
            if norm_key in manifest or any(
                normalize_semantic_path_key(k) == norm_key for k in manifest
            ):
                skipped += 1
                continue

        # Check file exists
        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found: {img_path}")
            errors += 1
            continue

        try:
            # Run OneFormer
            label_map, height, width = run_oneformer_on_image(
                img_path, model, processor, args.device
            )

            # Save
            save_semantic_label(
                img_path, label_map, height, width,
                args.cache_dir, args.model_id, manifest
            )

            processed += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            errors += 1

    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save metadata
    metadata["num_images"] = len(manifest)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total in manifest: {len(manifest)}")
    print(f"\nOutput directory: {args.cache_dir}")
    print(f"  - manifest.json")
    print(f"  - metadata.json")
    print(f"  - labels/*.npz")

    if errors > 0:
        print(f"\n[WARNING] {errors} images failed to process")
        sys.exit(1)
    else:
        print("\n[DONE] Semantic cache created successfully")


if __name__ == "__main__":
    main()