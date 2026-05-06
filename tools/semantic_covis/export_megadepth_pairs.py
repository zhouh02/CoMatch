#!/usr/bin/env python3
"""
Export MegaDepth image pairs from scene_info npz files for CLIP pseudo-label generation.

Usage:
    python tools/semantic_covis/export_megadepth_pairs.py \
        --npz-root data/megadepth/index/scene_info_0.1_0.7 \
        --train-list data/megadepth/index/trainvaltest_list/train_list.txt \
        --image-root data/megadepth \
        --output tools/semantic_covis/debug_pairs.txt \
        --num-pairs 50 \
        --min-overlap 0.1 \
        --seed 42
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def stable_hash(*args) -> str:
    """Generate a stable hash from multiple string inputs.

    Args:
        *args: Variable number of strings to hash.

    Returns:
        First 16 characters of SHA1 hash of joined inputs.
    """
    combined = "".join(str(a) for a in args)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export MegaDepth image pairs from scene_info npz files"
    )
    parser.add_argument(
        "--npz-root",
        type=str,
        required=True,
        help="MegaDepth scene_info npz root directory",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        required=True,
        help="Path to train_list.txt containing scene names",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="MegaDepth image root directory (root_dir in CoMatch)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tools/semantic_covis/debug_pairs.txt",
        help="Output txt file path (default: tools/semantic_covis/debug_pairs.txt)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50,
        help="Number of pairs to export (-1 for all, default: 50)",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.1,
        help="Minimum overlap score threshold (default: 0.1)",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=-1,
        help="Limit number of scenes to read (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle pairs before exporting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0)",
    )
    return parser.parse_args()


def load_scene_list(train_list_path: str) -> List[str]:
    """Load scene names from train_list.txt.

    Args:
        train_list_path: Path to train_list.txt

    Returns:
        List of scene names (without .npz extension)
    """
    with open(train_list_path, "r") as f:
        scenes = [line.strip().split()[0] for line in f if line.strip()]
    return scenes


def load_scene_pairs(
    npz_path: str,
    min_overlap: float = 0.1,
) -> Tuple[str, List, List[float]]:
    """Load pair_infos from a scene_info npz file.

    Args:
        npz_path: Path to the .npz file
        min_overlap: Minimum overlap score threshold

    Returns:
        Tuple of (scene_id, list of (idx0, idx1) tuples, list of overlap scores)
    """
    scene_info = np.load(npz_path, allow_pickle=True)
    pair_infos = scene_info["pair_infos"]

    scene_id = Path(npz_path).stem
    pairs = []
    scores = []

    for pair_info in pair_infos:
        (idx0, idx1), overlap_score, _ = pair_info
        if overlap_score >= min_overlap:
            pairs.append((idx0, idx1))
            scores.append(float(overlap_score))

    return scene_id, pairs, scores


def export_pairs(
    npz_root: str,
    train_list_path: str,
    image_root: str,
    output_path: str,
    num_pairs: int = -1,
    min_overlap: float = 0.1,
    scene_limit: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> Dict:
    """Export MegaDepth pairs to txt and jsonl files.

    Args:
        npz_root: Root directory containing scene_info npz files
        train_list_path: Path to train_list.txt
        image_root: MegaDepth image root directory
        output_path: Output txt file path
        num_pairs: Number of pairs to export (-1 for all)
        min_overlap: Minimum overlap score threshold
        scene_limit: Limit number of scenes (-1 for all)
        shuffle: Whether to shuffle pairs
        seed: Random seed

    Returns:
        Dictionary with export statistics
    """
    npz_root = Path(npz_root)
    image_root = Path(image_root)
    output_path = Path(output_path)

    # Load scene list
    scene_names = load_scene_list(train_list_path)
    if scene_limit > 0:
        scene_names = scene_names[:scene_limit]

    print("Loaded {} scenes from train_list.txt".format(len(scene_names)))
    if scene_limit > 0:
        print("  (limited to first {} scenes)".format(scene_limit))

    # Collect all pairs from all scenes
    all_pairs = []  # List of dicts with pair info
    skipped = 0

    print("\nReading scene_info npz files...")
    for scene_name in tqdm(scene_names, desc="Loading scenes"):
        npz_path = npz_root / "{}.npz".format(scene_name)
        if not npz_path.exists():
            tqdm.write("Warning: {} not found, skipping".format(npz_path))
            continue

        try:
            scene_info = np.load(npz_path, allow_pickle=True)
            image_paths = scene_info["image_paths"]
            pair_infos = scene_info["pair_infos"]

            for pair_idx, pair_info in enumerate(pair_infos):
                (idx0, idx1), overlap_score, _ = pair_info

                if overlap_score < min_overlap:
                    continue

                img_path0 = image_paths[idx0]
                img_path1 = image_paths[idx1]

                full_img0 = image_root / img_path0
                full_img1 = image_root / img_path1

                # Check if images exist
                if not full_img0.exists():
                    print("Warning: Image not found: {}".format(full_img0))
                    skipped += 1
                    continue
                if not full_img1.exists():
                    print("Warning: Image not found: {}".format(full_img1))
                    skipped += 1
                    continue

                pair_id = stable_hash(scene_name, pair_idx, img_path0, img_path1)

                all_pairs.append(
                    {
                        "pair_id": pair_id,
                        "scene_id": scene_name,
                        "pair_idx": pair_idx,
                        "idx0": int(idx0),
                        "idx1": int(idx1),
                        "image0_path": str(img_path0),
                        "image1_path": str(img_path1),
                        "overlap_score": float(overlap_score),
                    }
                )
        except Exception as e:
            tqdm.write("Error loading {}: {}".format(npz_path, e))
            continue

    total_pairs = len(all_pairs)
    print("\nTotal scenes processed: {}".format(len(scene_names)))
    print("Total pairs found: {}".format(total_pairs))
    print("Skipped (images not found): {}".format(skipped))

    # Shuffle if requested
    if shuffle:
        import random

        random.seed(seed)
        random.shuffle(all_pairs)
        print("Shuffled with seed={}".format(seed))

    # Limit number of pairs
    if num_pairs > 0 and len(all_pairs) > num_pairs:
        all_pairs = all_pairs[:num_pairs]
        print("Limited to {} pairs".format(num_pairs))

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write txt file
    print("\nWriting to {}...".format(output_path))
    with open(output_path, "w") as f:
        # Header
        f.write(
            "pair_id scene_id pair_idx idx0 idx1 image0_path image1_path\n"
        )
        for p in all_pairs:
            f.write(
                "{} {} {} {} {} {} {}\n".format(
                    p["pair_id"], p["scene_id"], p["pair_idx"],
                    p["idx0"], p["idx1"], p["image0_path"], p["image1_path"]
                )
            )

    # Write jsonl file
    jsonl_path = output_path.with_suffix(".jsonl")
    print("Writing to {}...".format(jsonl_path))
    with open(jsonl_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    # Print statistics
    print("\n" + "=" * 60)
    print("Export Statistics:")
    print("=" * 60)
    print("  Total scenes:        {}".format(len(scene_names)))
    print("  Total pairs found:  {}".format(total_pairs))
    print("  Successfully exported: {}".format(len(all_pairs)))
    print("  Skipped (not found): {}".format(skipped))
    if shuffle:
        print("  Shuffled:            Yes (seed={})".format(seed))
    print("=" * 60)

    # Print first 5 examples
    print("\nFirst 5 examples (txt format):")
    print("-" * 60)
    for p in all_pairs[:5]:
        print(
            "{} {} {} {} {} {} {}".format(
                p["pair_id"], p["scene_id"], p["pair_idx"],
                p["idx0"], p["idx1"], p["image0_path"], p["image1_path"]
            )
        )
    print("-" * 60)

    print("\nFirst 5 examples (jsonl format):")
    print("-" * 60)
    for p in all_pairs[:5]:
        print(json.dumps(p))
    print("-" * 60)

    return {
        "num_scenes": len(scene_names),
        "total_pairs": total_pairs,
        "exported_pairs": len(all_pairs),
        "skipped": skipped,
        "output_txt": str(output_path),
        "output_jsonl": str(jsonl_path),
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("MegaDepth Pair Exporter for CLIP Pseudo-Labels")
    print("=" * 60)
    print("  npz-root:    {}".format(args.npz_root))
    print("  train-list:  {}".format(args.train_list))
    print("  image-root:  {}".format(args.image_root))
    print("  output:      {}".format(args.output))
    num_pairs_str = args.num_pairs if args.num_pairs > 0 else "all"
    print("  num-pairs:   {}".format(num_pairs_str))
    print("  min-overlap: {}".format(args.min_overlap))
    scene_limit_str = args.scene_limit if args.scene_limit > 0 else "all"
    print("  scene-limit: {}".format(scene_limit_str))
    print("  shuffle:     {}".format(args.shuffle))
    print("  seed:        {}".format(args.seed))
    print("=" * 60)

    stats = export_pairs(
        npz_root=args.npz_root,
        train_list_path=args.train_list,
        image_root=args.image_root,
        output_path=args.output,
        num_pairs=args.num_pairs,
        min_overlap=args.min_overlap,
        scene_limit=args.scene_limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
