#!/usr/bin/env python3
"""
Precompute OneFormer semantic segmentation for MegaDepth test images.

This script runs OneFormer on all unique images from the MegaDepth test set
and saves the semantic label maps in original image resolution.

Usage:
    # Precompute semantic cache (separate inference - default)
    python scripts/precompute_oneformer_megadepth_semantics.py \
        --data_cfg configs/data/megadepth_test_1500.py \
        --cache_dir outputs/semantic_cache \
        --model_id shi-labs/oneformer_ade20k_swin_large \
        --device cuda

    # Precompute semantic cache (joint inference - concatenate pairs)
    python scripts/precompute_oneformer_megadepth_semantics.py \
        --data_cfg configs/data/megadepth_test_1500.py \
        --cache_dir outputs/semantic_cache_joint \
        --model_id shi-labs/oneformer_ade20k_swin_large \
        --device cuda \
        --joint_mode

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

Joint Mode:
- For each pair of images, horizontally concatenate them (padding to same height)
- Run OneFormer on the concatenated image
- Split the result back to individual images
- This ensures semantic consistency between paired images
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import (
    normalize_semantic_path_key,
    get_path_suffix,
    compute_file_hash,
    validate_semantic_cache,
)


def collect_megadepth_pairs(data_cfg_path: str) -> Tuple[List[Tuple[str, str, str]], int]:
    """
    Collect image pairs from MegaDepth test configuration.

    Returns pairs of (abs_path_0, abs_path_1, pair_key) for joint segmentation.
    """
    from src.config.default import get_cfg_defaults
    from src.datasets.megadepth import MegaDepthDataset
    from torch.utils.data import ConcatDataset

    config = get_cfg_defaults()
    config.merge_from_file(data_cfg_path)

    data_root = config.DATASET.TEST_DATA_ROOT
    npz_root = config.DATASET.TEST_NPZ_ROOT
    scene_list_path = config.DATASET.TEST_LIST_PATH

    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    npz_names = [f'{n}.npz' for n in npz_names]

    datasets = []
    for npz_name in tqdm(npz_names, desc="Loading scene datasets"):
        npz_path = os.path.join(npz_root, npz_name)
        if not os.path.exists(npz_path):
            print(f"Warning: Scene file not found: {npz_path}")
            continue
        datasets.append(
            MegaDepthDataset(
                data_root,
                npz_path,
                mode='test',
                min_overlap_score=0.0,
                img_resize=config.DATASET.MGDPT_IMG_RESIZE,
                df=config.DATASET.MGDPT_DF,
                img_padding=config.DATASET.MGDPT_IMG_PAD,
                depth_padding=config.DATASET.MGDPT_DEPTH_PAD,
                fp16=False,
            )
        )

    concat_ds = ConcatDataset(datasets)
    num_pairs = len(concat_ds)

    pair_entries = []

    for idx in tqdm(range(num_pairs), desc="Collecting pairs"):
        cumlen = 0
        for ds in datasets:
            if idx < cumlen + len(ds):
                local_idx = idx - cumlen
                pair_info = ds.pair_infos[local_idx]
                idx0, idx1 = pair_info[0]

                ip_array = ds.scene_info['image_paths']
                rel_path_0 = str(ip_array[idx0])
                rel_path_1 = str(ip_array[idx1])

                abs_path_0 = os.path.join(data_root, rel_path_0)
                abs_path_1 = os.path.join(data_root, rel_path_1)

                pair_key = f"{normalize_semantic_path_key(rel_path_0)}_{normalize_semantic_path_key(rel_path_1)}"
                pair_entries.append((abs_path_0, abs_path_1, pair_key))
                break
            cumlen += len(ds)

    print(f"\nFirst 3 pair entries:")
    for abs0, abs1, key in pair_entries[:3]:
        print(f"  pair_key: {key}")
        print(f"    img0: {abs0}")
        print(f"    img1: {abs1}")

    return pair_entries, num_pairs


def collect_megadepth_test_images(data_cfg_path: str) -> Tuple[List[Tuple[str, str]], int, int]:
    """
    Collect unique image paths from MegaDepth test configuration.

    Uses the actual MegaDepthDataset to get pair_names, guaranteeing
    the same path format that will be used during CoMatch evaluation.

    Args:
        data_cfg_path: Path to MegaDepth data config file

    Returns:
        Tuple of (image_entries, num_pairs, num_unique_images)
        where image_entries is a list of (abs_path, rel_path) tuples.
        rel_path matches the format used in pair_names.
    """
    from src.config.default import get_cfg_defaults
    from src.datasets.megadepth import MegaDepthDataset
    from torch.utils.data import ConcatDataset

    # Load config using yacs (same as test.py)
    config = get_cfg_defaults()
    config.merge_from_file(data_cfg_path)

    data_root = config.DATASET.TEST_DATA_ROOT
    npz_root = config.DATASET.TEST_NPZ_ROOT
    scene_list_path = config.DATASET.TEST_LIST_PATH

    # Read scene list
    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    npz_names = [f'{n}.npz' for n in npz_names]

    # Build datasets (same as MultiSceneDataModule does for test)
    datasets = []
    for npz_name in tqdm(npz_names, desc="Loading scene datasets"):
        npz_path = os.path.join(npz_root, npz_name)
        if not os.path.exists(npz_path):
            print(f"Warning: Scene file not found: {npz_path}")
            continue
        datasets.append(
            MegaDepthDataset(
                data_root,
                npz_path,
                mode='test',
                min_overlap_score=0.0,
                img_resize=config.DATASET.MGDPT_IMG_RESIZE,
                df=config.DATASET.MGDPT_DF,
                img_padding=config.DATASET.MGDPT_IMG_PAD,
                depth_padding=config.DATASET.MGDPT_DEPTH_PAD,
                fp16=False,
            )
        )

    concat_ds = ConcatDataset(datasets)
    num_pairs = len(concat_ds)

    # Extract pair_names from each sample (rel_path format, no data_root prefix)
    image_entries = []
    seen = set()

    for idx in tqdm(range(num_pairs), desc="Collecting image paths"):
        # Find which sub-dataset and local index
        cumlen = 0
        for ds in datasets:
            if idx < cumlen + len(ds):
                local_idx = idx - cumlen
                pair_info = ds.pair_infos[local_idx]
                idx0, idx1 = pair_info[0]

                # Get paths exactly as __getitem__ does
                # scene_info['image_paths'] is a numpy array of strings
                ip_array = ds.scene_info['image_paths']
                rel_path_0 = str(ip_array[idx0])
                rel_path_1 = str(ip_array[idx1])

                abs_path_0 = os.path.join(data_root, rel_path_0)
                abs_path_1 = os.path.join(data_root, rel_path_1)

                for abs_p, rel_p in [(abs_path_0, rel_path_0), (abs_path_1, rel_path_1)]:
                    norm = normalize_semantic_path_key(rel_p)
                    if norm not in seen:
                        seen.add(norm)
                        image_entries.append((abs_p, rel_p))
                break
            cumlen += len(ds)

    # Debug: print first few entries
    print(f"\nFirst 3 image entries (abs_path, rel_path):")
    for abs_p, rel_p in image_entries[:3]:
        print(f"  abs: {abs_p}")
        print(f"  rel: {rel_p}")

    return image_entries, num_pairs, len(image_entries)


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


def run_oneformer_joint(
    image_path_0: str,
    image_path_1: str,
    model,
    processor,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Run OneFormer on a pair of images (horizontally concatenated) and return
    semantic segmentations for both.

    Args:
        image_path_0: Path to first image
        image_path_1: Path to second image
        model: OneFormerForUniversalSegmentation model
        processor: OneFormerProcessor
        device: Device to run on

    Returns:
        Tuple of (label_map_0, label_map_1, h0, w0, h1, w1) where label maps are H x W
    """
    from PIL import Image

    # Load both images
    img0 = Image.open(image_path_0).convert("RGB")
    img1 = Image.open(image_path_1).convert("RGB")

    w0, h0 = img0.size
    w1, h1 = img1.size

    # Pad to same height, then concatenate horizontally
    max_h = max(h0, h1)

    # Create padded images
    pad0 = Image.new("RGB", (w0, max_h))
    pad1 = Image.new("RGB", (w1, max_h))
    pad0.paste(img0, (0, 0))
    pad1.paste(img1, (0, 0))

    # Concatenate horizontally
    joint_img = Image.new("RGB", (w0 + w1, max_h))
    joint_img.paste(pad0, (0, 0))
    joint_img.paste(pad1, (w0, 0))

    # Process joint image
    inputs = processor(
        images=joint_img,
        task_inputs=["semantic"],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to joint image size
    seg = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(max_h, w0 + w1)],
    )[0]

    joint_label = seg.cpu().numpy()  # (max_h, w0+w1)

    # Split back to individual images
    label_map_0 = joint_label[:, :w0][:h0]  # (h0, w0)
    label_map_1 = joint_label[:, w0:][:h1]  # (h1, w1)

    assert label_map_0.shape == (h0, w0), f"Expected {(h0, w0)}, got {label_map_0.shape}"
    assert label_map_1.shape == (h1, w1), f"Expected {(h1, w1)}, got {label_map_1.shape}"

    return label_map_0, label_map_1, h0, w0, h1, w1


def save_semantic_label(
    image_path: str,
    rel_path: str,
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
        image_path: Absolute path to original image (for reading)
        rel_path: Relative path matching pair_names format (for manifest key)
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

    # Use rel_path directly as manifest key (matches pair_names format exactly)
    manifest_key = normalize_semantic_path_key(rel_path)
    manifest[manifest_key] = f"labels/{npz_filename}"

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
        help='Hugging Face model id OR local directory for OneFormer'
    )
    parser.add_argument(
        '--local_files_only',
        action='store_true',
        help='Only load model from local cache (offline mode)'
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
    parser.add_argument(
        '--joint_mode',
        action='store_true',
        help='Process pairs jointly by concatenating images horizontally'
    )
    parser.add_argument(
        '--joint_cache_suffix',
        type=str,
        default='_joint',
        help='Suffix to add to image labels in joint mode to differentiate from separate mode'
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

    # Load config first (needed for both joint and separate modes)
    from src.config.default import get_cfg_defaults
    config = get_cfg_defaults()
    if args.data_cfg:
        config.merge_from_file(args.data_cfg)
    data_root = config.DATASET.TEST_DATA_ROOT

    # Joint mode: collect pairs
    if args.joint_mode:
        print("=" * 60)
        print("Joint Mode: Processing pairs together")
        print("=" * 60)
        pair_entries, num_pairs = collect_megadepth_pairs(args.data_cfg)

        print(f"\nSummary:")
        print(f"  Num pairs: {num_pairs}")

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

        # Track processed pairs and images
        joint_pairs_processed = defaultdict(list)  # image_key -> list of pair_keys

        # Load OneFormer
        print("\n" + "=" * 60)
        print(f"Loading OneFormer model: {args.model_id}")
        print("=" * 60)

        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        model_path = args.model_id
        processor = OneFormerProcessor.from_pretrained(model_path, args.local_files_only)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_path, args.local_files_only)
        model = model.to(args.device)
        model.eval()

        # Counters
        processed = 0
        skipped = 0
        errors = 0

        print("\n" + "=" * 60)
        print("Processing Pairs (Joint Mode)")
        print("=" * 60)

        for abs_path_0, abs_path_1, pair_key in tqdm(pair_entries, desc="Processing pairs"):
            # Normalize keys for check
            if data_root in abs_path_0:
                norm_key_0 = normalize_semantic_path_key(abs_path_0.replace(data_root, '').lstrip('/'))
                norm_key_1 = normalize_semantic_path_key(abs_path_1.replace(data_root, '').lstrip('/'))
            else:
                norm_key_0 = normalize_semantic_path_key(abs_path_0.split('/')[-2] + '/' + abs_path_0.split('/')[-1])
                norm_key_1 = normalize_semantic_path_key(abs_path_1.split('/')[-2] + '/' + abs_path_1.split('/')[-1])

            # Check if already processed
            if not args.overwrite:
                # Check if both labels are already in manifest
                key_0_in_manifest = any(normalize_semantic_path_key(k) == norm_key_0 for k in manifest)
                key_1_in_manifest = any(normalize_semantic_path_key(k) == norm_key_1 for k in manifest)
                if key_0_in_manifest and key_1_in_manifest:
                    skipped += 1
                    continue

            # Check files exist
            if not os.path.exists(abs_path_0) or not os.path.exists(abs_path_1):
                print(f"\nWarning: Image not found: {abs_path_0} or {abs_path_1}")
                errors += 1
                continue

            try:
                # Run joint inference
                label_map_0, label_map_1, h0, w0, h1, w1 = run_oneformer_joint(
                    abs_path_0, abs_path_1, model, processor, args.device
                )

                # Extract rel_path from abs_path using data_root
                if data_root in abs_path_0:
                    rel_path_0 = abs_path_0.replace(data_root, '').lstrip('/')
                    rel_path_1 = abs_path_1.replace(data_root, '').lstrip('/')
                else:
                    rel_path_0 = abs_path_0.split('/')[-2] + '/' + abs_path_0.split('/')[-1]
                    rel_path_1 = abs_path_1.split('/')[-2] + '/' + abs_path_1.split('/')[-1]

                # Save both labels
                save_semantic_label(
                    abs_path_0, rel_path_0, label_map_0, h0, w0,
                    args.cache_dir, args.model_id, manifest
                )
                save_semantic_label(
                    abs_path_1, rel_path_1, label_map_1, h1, w1,
                    args.cache_dir, args.model_id, manifest
                )

                processed += 1

            except Exception as e:
                print(f"\nError processing pair {pair_key}: {e}")
                errors += 1

        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save metadata
        metadata_path = cache_dir / "metadata.json"
        metadata = {
            "model_id": args.model_id,
            "task": "semantic",
            "label_space": "ade20k",
            "mode": "joint",
            "num_pairs": len(pair_entries),
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("Summary (Joint Mode)")
        print("=" * 60)
        print(f"  Processed pairs: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Total images in manifest: {len(manifest)}")
        print(f"\nOutput directory: {args.cache_dir}")

        if errors > 0:
            print(f"\n[WARNING] {errors} pairs failed to process")
            sys.exit(1)
        else:
            print("\n[DONE] Joint semantic cache created successfully")
            sys.exit(0)

    # Separate mode (original behavior)
    if args.image_list:
        print(f"Loading image list from: {args.image_list}")
        raw_paths = load_image_list(args.image_list)
        # For image_list mode, use path as both abs and rel
        image_entries = [(p, p) for p in raw_paths]
        num_pairs = len(raw_paths) // 2  # Approximate
        num_unique = len(raw_paths)
    else:
        print(f"Loading MegaDepth test images from config: {args.data_cfg}")
        image_entries, num_pairs, num_unique = collect_megadepth_test_images(args.data_cfg)

    print(f"\nSummary:")
    print(f"  Num pairs: {num_pairs}")
    print(f"  Num unique images: {num_unique}")

    if args.max_images:
        image_entries = image_entries[:args.max_images]
        print(f"  Limited to first {args.max_images} images")

    if not image_entries:
        print("No images to process")
        sys.exit(0)

    print(f"\nFirst few images:")
    for abs_p, rel_p in image_entries[:5]:
        print(f"  - abs: {abs_p}")
        print(f"    rel: {rel_p}")

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
    except ImportError as e:
        print(f"\nERROR: transformers not installed: {e}")
        print("Please install: pip install transformers")
        sys.exit(1)

    # from_pretrained supports both HuggingFace hub IDs and local directories
    # For local models, pass the local path as model_id
    model_path = args.model_id
    if os.path.isdir(model_path):
        print(f"Loading model from local directory: {model_path}")
    processor = OneFormerProcessor.from_pretrained(model_path, args.local_files_only)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_path, args.local_files_only)
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

    for abs_path, rel_path in tqdm(image_entries, desc="Running OneFormer"):
        # Check if already processed (by rel_path key)
        norm_key = normalize_semantic_path_key(rel_path)
        if not args.overwrite:
            if norm_key in manifest or any(
                normalize_semantic_path_key(k) == norm_key for k in manifest
            ):
                skipped += 1
                continue

        # Check file exists
        if not os.path.exists(abs_path):
            print(f"\nWarning: Image not found: {abs_path}")
            errors += 1
            continue

        try:
            # Run OneFormer on the actual image file
            label_map, height, width = run_oneformer_on_image(
                abs_path, model, processor, args.device
            )

            # Save with rel_path as manifest key (matches pair_names format)
            save_semantic_label(
                abs_path, rel_path, label_map, height, width,
                args.cache_dir, args.model_id, manifest
            )

            processed += 1

        except Exception as e:
            print(f"\nError processing {abs_path}: {e}")
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