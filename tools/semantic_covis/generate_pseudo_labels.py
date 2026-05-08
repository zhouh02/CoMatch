#!/usr/bin/env python3
"""
Generate CLIP Semantic Covisibility Pseudo-labels for CoMatch.

Reads image pair list from export_megadepth_pairs.py output,
extracts CLIP patch tokens, computes semantic covisibility,
and saves per-pair npz files.

Usage:
    python tools/semantic_covis/generate_pseudo_labels.py \
        --pair-list outputs/sem_covis_sanity/debug_pairs.txt \
        --image-root data/megadepth/train \
        --output-dir outputs/sem_covis_labels_debug \
        --model-name /ssd-data3/zh2025/CoMatch/models/clip-vit-large-patch14-336 \
        --device cuda \
        --max-pairs 5 \
        --save-debug
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Add tools/semantic_covis to import path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from clip_feature_extractor import CLIPFeatureExtractor, resize_heatmap_to_coarse
from covisibility_estimator import compute_semantic_covisibility


def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(
        description="Generate CLIP semantic covisibility pseudo-labels"
    )
    parser.add_argument(
        "--pair-list", type=str,
        default="outputs/sem_covis_sanity/debug_pairs.txt",
        help="Path to pair list txt file",
    )
    parser.add_argument(
        "--image-root", type=str,
        default="data/megadepth/train",
        help="MegaDepth image root directory",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/sem_covis_labels_debug",
        help="Output directory for npz files",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="/ssd-data3/zh2025/CoMatch/models/clip-vit-large-patch14-336",
        help="CLIP model name or local path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument("--long-edge", type=int, default=832)
    parser.add_argument("--df", type=int, default=8)
    parser.add_argument("--coarse-scale", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--q-low", type=float, default=40)
    parser.add_argument("--q-high", type=float, default=85)
    parser.add_argument(
        "--mode", type=str, default="v1", choices=["v1", "v2"],
        help="Covisibility estimator mode: v1 (original) or v2 (enhanced)",
    )
    parser.add_argument(
        "--specificity-gamma", type=float, default=1.5,
        help="Gamma for specificity power (v2 only, default: 1.5)",
    )
    parser.add_argument("--k-margin", type=int, default=10)
    parser.add_argument("--q-margin-low", type=float, default=30)
    parser.add_argument("--q-margin-high", type=float, default=80)
    parser.add_argument("--max-pairs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing npz files")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save additional debug fields (clip-grid level)")
    return parser.parse_args()


def read_pair_list(pair_list_path):
    # type: (str) -> List[Dict[str, Any]]
    """Read pair list from txt file, skip header line.

    Returns list of dicts with keys:
        pair_id, scene_id, pair_idx, idx0, idx1, image0_path, image1_path
    """
    pairs = []
    with open(pair_list_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # Skip header
        if i == 0 and line.startswith("pair_id"):
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        pairs.append({
            "pair_id": parts[0],
            "scene_id": parts[1],
            "pair_idx": int(parts[2]),
            "idx0": int(parts[3]),
            "idx1": int(parts[4]),
            "image0_path": parts[5],
            "image1_path": parts[6],
        })

    return pairs


def downsample_mask(mask, target_h, target_w):
    # type: (np.ndarray, int, int) -> np.ndarray
    """Downsample a boolean mask to target size using nearest interpolation.

    Args:
        mask: Boolean mask [H, W]
        target_h: Target height
        target_w: Target width

    Returns:
        Boolean mask [target_h, target_w]
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_uint8, (target_w, target_h),
                              interpolation=cv2.INTER_NEAREST)
    return mask_resized > 127


def resize_to_coarse(tensor_2d, target_h, target_w):
    # type: (torch.Tensor, int, int) -> np.ndarray
    """Resize a 2D tensor [H, W] to coarse grid [target_h, target_w].

    Returns numpy array.
    """
    arr = tensor_2d.cpu().numpy().astype(np.float32)
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def process_pair(
    pair_info,       # type: Dict[str, Any]
    image_root,      # type: str
    output_dir,      # type: Path
    extractor,       # type: CLIPFeatureExtractor
    coarse_scale,    # type: int
    mode,            # type: str
    topk,            # type: int
    temperature,     # type: float
    q_low,           # type: float
    q_high,          # type: float
    specificity_gamma,  # type: float
    k_margin,        # type: int
    q_margin_low,    # type: float
    q_margin_high,   # type: float
    save_debug,      # type: bool
    overwrite,       # type: bool
):
    # type: (...) -> Dict[str, Any]
    """Process a single image pair and save npz.

    Returns stats dict.
    """
    pair_id = pair_info["pair_id"]
    npz_path = output_dir / "{}.npz".format(pair_id)

    # Skip if exists
    if npz_path.exists() and not overwrite:
        return {"pair_id": pair_id, "status": "skipped", "error": ""}

    # Construct full image paths
    img0 = os.path.join(image_root, pair_info["image0_path"])
    img1 = os.path.join(image_root, pair_info["image1_path"])

    if not os.path.isfile(img0):
        return {"pair_id": pair_id, "status": "failed",
                "error": "image0 not found: {}".format(img0)}
    if not os.path.isfile(img1):
        return {"pair_id": pair_id, "status": "failed",
                "error": "image1 not found: {}".format(img1)}

    # Extract CLIP features
    result0 = extractor.extract(img0, return_dict=True, return_numpy=False)
    result1 = extractor.extract(img1, return_dict=True, return_numpy=False)

    pf0 = result0["patch_features"]  # [1, N, D] or [N, D]
    pf1 = result1["patch_features"]
    grid0 = result0["grid_size"]  # (Gh, Gw)
    grid1 = result1["grid_size"]
    pp0 = result0["preprocess"]
    pp1 = result1["preprocess"]

    # Squeeze batch dim if needed
    if pf0.ndim == 3 and pf0.shape[0] == 1:
        pf0 = pf0.squeeze(0)
    if pf1.ndim == 3 and pf1.shape[0] == 1:
        pf1 = pf1.squeeze(0)

    # Compute semantic covisibility on CLIP grid
    covis_result = compute_semantic_covisibility(
        pf0, pf1, grid0, grid1,
        mode=mode,
        topk=topk,
        temperature=temperature,
        q_low=q_low,
        q_high=q_high,
        specificity_gamma=specificity_gamma,
        k_margin=k_margin,
        q_margin_low=q_margin_low,
        q_margin_high=q_margin_high,
    )

    # Resize to CoMatch coarse grid
    pad_size = pp0["pad_size"]
    ch = pad_size // coarse_scale
    cw = pad_size // coarse_scale

    y_sem0_coarse = resize_to_coarse(covis_result["y_sem0_clip"], ch, cw)
    y_sem1_coarse = resize_to_coarse(covis_result["y_sem1_clip"], ch, cw)
    conf0_coarse = resize_to_coarse(covis_result["conf0_clip"], ch, cw)
    conf1_coarse = resize_to_coarse(covis_result["conf1_clip"], ch, cw)

    # Downsample valid_mask to coarse grid
    valid_mask0 = pp0["valid_mask"]
    valid_mask1 = pp1["valid_mask"]
    if isinstance(valid_mask0, torch.Tensor):
        valid_mask0 = valid_mask0.cpu().numpy()
    if isinstance(valid_mask1, torch.Tensor):
        valid_mask1 = valid_mask1.cpu().numpy()

    coarse_valid0 = downsample_mask(valid_mask0, ch, cw)
    coarse_valid1 = downsample_mask(valid_mask1, ch, cw)

    # Zero out padding regions
    y_sem0_coarse[~coarse_valid0] = 0.0
    y_sem1_coarse[~coarse_valid1] = 0.0
    conf0_coarse[~coarse_valid0] = 0.0
    conf1_coarse[~coarse_valid1] = 0.0

    # Compute scale arrays
    scale0 = pp0["scale"]
    scale1 = pp1["scale"]
    if isinstance(scale0, torch.Tensor):
        scale0 = scale0.cpu().numpy()
    if isinstance(scale1, torch.Tensor):
        scale1 = scale1.cpu().numpy()

    # Build npz data
    npz_data = {
        "pair_id": pair_id,
        "scene_id": pair_info["scene_id"],
        "pair_idx": pair_info["pair_idx"],
        "idx0": pair_info["idx0"],
        "idx1": pair_info["idx1"],
        "image0_rel_path": pair_info["image0_path"],
        "image1_rel_path": pair_info["image1_path"],
        "image0_abs_path": os.path.abspath(img0),
        "image1_abs_path": os.path.abspath(img1),
        "original_hw0": np.array(pp0["original_hw"], dtype=np.int32),
        "original_hw1": np.array(pp1["original_hw"], dtype=np.int32),
        "resized_hw0": np.array(pp0["resized_hw"], dtype=np.int32),
        "resized_hw1": np.array(pp1["resized_hw"], dtype=np.int32),
        "valid_hw0": np.array(pp0["valid_hw"], dtype=np.int32),
        "valid_hw1": np.array(pp1["valid_hw"], dtype=np.int32),
        "pad_size0": np.array(pp0["pad_size"], dtype=np.int32),
        "pad_size1": np.array(pp1["pad_size"], dtype=np.int32),
        "coarse_hw0": np.array([ch, cw], dtype=np.int32),
        "coarse_hw1": np.array([ch, cw], dtype=np.int32),
        "clip_grid0": np.array(grid0, dtype=np.int32),
        "clip_grid1": np.array(grid1, dtype=np.int32),
        "scale0": scale0.astype(np.float32),
        "scale1": scale1.astype(np.float32),
        "y_sem0": y_sem0_coarse.astype(np.float16),
        "y_sem1": y_sem1_coarse.astype(np.float16),
        "conf0": conf0_coarse.astype(np.float16),
        "conf1": conf1_coarse.astype(np.float16),
    }

    # Debug fields
    if save_debug:
        for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip",
                     "existence0_clip", "existence1_clip",
                     "specificity0_clip", "specificity1_clip",
                     "raw0_clip", "raw1_clip"]:
            npz_data[key] = covis_result[key].cpu().numpy().astype(np.float16)

    # Save npz
    np.savez(str(npz_path), **npz_data)

    # Compute stats
    stats = {
        "pair_id": pair_id,
        "y_sem0_mean": float(y_sem0_coarse.mean()),
        "y_sem1_mean": float(y_sem1_coarse.mean()),
        "y_sem0_max": float(y_sem0_coarse.max()),
        "y_sem1_max": float(y_sem1_coarse.max()),
        "conf0_mean": float(conf0_coarse.mean()),
        "conf1_mean": float(conf1_coarse.mean()),
        "conf0_max": float(conf0_coarse.max()),
        "conf1_max": float(conf1_coarse.max()),
        "high_ratio0": float((y_sem0_coarse > 0.6).mean()),
        "high_ratio1": float((y_sem1_coarse > 0.6).mean()),
        "status": "ok",
        "error": "",
    }

    return stats


def main():
    # type: () -> int
    args = parse_args()

    print("=" * 60)
    print("CLIP Semantic Covisibility Pseudo-label Generator")
    print("=" * 60)
    print("  pair-list:    {}".format(args.pair_list))
    print("  image-root:   {}".format(args.image_root))
    print("  output-dir:   {}".format(args.output_dir))
    print("  model-name:   {}".format(args.model_name))
    print("  device:       {}".format(args.device))
    print("  long-edge:    {}".format(args.long_edge))
    print("  coarse-scale: {}".format(args.coarse_scale))
    print("  topk:         {}".format(args.topk))
    print("  temperature:  {}".format(args.temperature))
    print("  q-low:        {}".format(args.q_low))
    print("  q-high:       {}".format(args.q_high))
    print("  mode:         {}".format(args.mode))
    print("  gamma:        {}".format(args.specificity_gamma))
    print("  k-margin:     {}".format(args.k_margin))
    print("  max-pairs:    {}".format(args.max_pairs if args.max_pairs > 0 else "all"))
    print("  overwrite:    {}".format(args.overwrite))
    print("  save-debug:   {}".format(args.save_debug))
    print("=" * 60)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read pair list
    pairs = read_pair_list(args.pair_list)
    print("\nLoaded {} pairs from {}".format(len(pairs), args.pair_list))

    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
        print("Limited to {} pairs".format(len(pairs)))

    # Load CLIP model once
    print("\nLoading CLIP model...")
    extractor = CLIPFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        long_edge=args.long_edge,
    )

    # Process pairs
    stats_list = []  # type: List[Dict[str, Any]]
    processed = 0
    skipped = 0
    failed = 0

    # Open output files
    jsonl_path = output_dir / "index.jsonl"
    failed_path = output_dir / "failed.txt"
    csv_path = output_dir / "stats.csv"

    jsonl_file = open(str(jsonl_path), "w")
    failed_file = open(str(failed_path), "w")

    csv_columns = [
        "pair_id", "y_sem0_mean", "y_sem1_mean",
        "y_sem0_max", "y_sem1_max",
        "conf0_mean", "conf1_mean",
        "conf0_max", "conf1_max",
        "high_ratio0", "high_ratio1",
        "status", "error",
    ]
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    print("\nProcessing pairs...")
    for i, pair_info in enumerate(pairs):
        pair_id = pair_info["pair_id"]
        print("\n[{}/{}] Processing pair: {}".format(i + 1, len(pairs), pair_id))

        try:
            stats = process_pair(
                pair_info=pair_info,
                image_root=args.image_root,
                output_dir=output_dir,
                extractor=extractor,
                coarse_scale=args.coarse_scale,
                mode=args.mode,
                topk=args.topk,
                temperature=args.temperature,
                q_low=args.q_low,
                q_high=args.q_high,
                specificity_gamma=args.specificity_gamma,
                k_margin=args.k_margin,
                q_margin_low=args.q_margin_low,
                q_margin_high=args.q_margin_high,
                save_debug=args.save_debug,
                overwrite=args.overwrite,
            )
        except Exception as e:
            stats = {
                "pair_id": pair_id,
                "y_sem0_mean": 0, "y_sem1_mean": 0,
                "y_sem0_max": 0, "y_sem1_max": 0,
                "conf0_mean": 0, "conf1_mean": 0,
                "conf0_max": 0, "conf1_max": 0,
                "high_ratio0": 0, "high_ratio1": 0,
                "status": "failed",
                "error": str(e),
            }
            print("  ERROR: {}".format(str(e)))

        stats_list.append(stats)

        # Write to files
        jsonl_file.write(json.dumps(stats) + "\n")
        csv_writer.writerow(stats)

        if stats["status"] == "ok":
            processed += 1
            print("  y_sem0: mean={:.4f}, max={:.4f}".format(
                stats["y_sem0_mean"], stats["y_sem0_max"]))
            print("  y_sem1: mean={:.4f}, max={:.4f}".format(
                stats["y_sem1_mean"], stats["y_sem1_max"]))
            print("  conf0:  mean={:.4f}, max={:.4f}".format(
                stats["conf0_mean"], stats["conf0_max"]))
            print("  high_ratio0: {:.4f}".format(stats["high_ratio0"]))
        elif stats["status"] == "skipped":
            skipped += 1
            print("  SKIPPED (npz already exists)")
        else:
            failed += 1
            failed_file.write("{}\t{}\n".format(pair_id, stats["error"]))
            print("  FAILED: {}".format(stats["error"]))

        # Flush files periodically
        if (i + 1) % 10 == 0:
            jsonl_file.flush()
            csv_file.flush()
            failed_file.flush()

    # Cleanup
    jsonl_file.close()
    csv_file.close()
    failed_file.close()
    extractor.close()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  Total pairs:   {}".format(len(pairs)))
    print("  Processed:     {}".format(processed))
    print("  Skipped:       {}".format(skipped))
    print("  Failed:        {}".format(failed))
    print("  Output dir:    {}".format(output_dir))
    print("  Stats CSV:     {}".format(csv_path))
    print("  Index JSONL:   {}".format(jsonl_path))
    print("  Failed log:    {}".format(failed_path))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
