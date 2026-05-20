#!/usr/bin/env python3
"""
Debug script to verify coordinate system mapping between CoMatch and OneFormer.

This script helps identify if there's a resolution mismatch issue between
CoMatch output coordinates and OneFormer segmentation.

Usage:
    python tools/semantic_covis/debug_resolution_mismatch.py \
        --oneformer-dir outputs/oneformer_outdoor_test \
        --test-pair "Undistorted_SfM/0022/images/427154679_de14c315f4_o.jpg" \
                   "Undistorted_SfM/0022/images/427189698_c61ab5d7db_o.jpg"

    # Or run with CoMatch dump:
    python tools/semantic_covis/debug_resolution_mismatch.py \
        --oneformer-dir outputs/oneformer_outdoor_test \
        --comatch-dump outputs/comatch_outdoor_test
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.semantic_covis.semantic_match_diagnostic import (
    OneFormerSegStore,
    image_path_to_safe_stem,
)


def debug_resolution_mismatch(oneformer_dir, image_paths=None, comatch_dump_dir=None):
    """Debug resolution mismatch between CoMatch and OneFormer."""
    print("=" * 60)
    print("Resolution Mismatch Debug Tool")
    print("=" * 60)

    seg_store = OneFormerSegStore(oneformer_dir, verbose=False)

    if image_paths:
        for img_path in image_paths:
            print(f"\n--- Image: {img_path} ---")
            safe_stem = image_path_to_safe_stem(img_path)
            npz_path = seg_store.npz_dir / f"{safe_stem}.npz"
            json_path = seg_store.json_dir / f"{safe_stem}.json"

            print(f"  Safe stem: {safe_stem}")
            print(f"  Expected npz: {npz_path}")
            print(f"  NPZ exists: {npz_path.exists()}")

            if json_path and json_path.exists():
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                print(f"  Seg dimensions: {meta.get('height')}x{meta.get('width')}")
                print(f"  Original dimensions: {meta.get('orig_h')}x{meta.get('orig_w')}")
                print(f"  Processed dimensions: {meta.get('processed_hw')}")

            # Get from seg_store
            seg = seg_store.get(img_path)
            if seg:
                print(f"  Seg from store: {seg['height']}x{seg['width']}")
            else:
                print("  No segmentation found!")

    # Show typical CoMatch settings for reference
    print("\n" + "=" * 60)
    print("CoMatch Expected Settings for Megadepth Test:")
    print("=" * 60)
    print("  megasize=1152 (from test_semantic.py example)")
    print("  CoMatch input: resize(longer_edge=1152) + padding to square")
    print("  This means if original is 1920x1080:")
    print("    - Resize: 1920x1080 -> 1152x648 (pad to 1152x1152)")
    print("    - CoMatch hw0_i = (1152, 1152)")
    print()
    print("  megasize=832 (from megadepth_test_1500.py)")
    print("  CoMatch input: resize(longer_edge=832) + padding to square")
    print("  If original is 1920x1080:")
    print("    - Resize: 1920x1080 -> 832x468 (pad to 832x832)")
    print("    - CoMatch hw0_i = (832, 832)")

    # Check if there are any size variations in the dataset
    print("\n" + "=" * 60)
    print("Analyzing OneFormer Output Size Distribution:")
    print("=" * 60)

    sizes = []
    for npz_file in sorted(seg_store.npz_dir.glob("*.npz"))[:20]:  # Sample first 20
        data = np.load(npz_file, allow_pickle=True)
        h = int(data['height'])
        w = int(data['width'])
        sizes.append((h, w))

    if sizes:
        unique_sizes = list(set(sizes))
        print(f"  Found {len(seg_store._npz_index)} total images")
        print(f"  Sampled {len(sizes)} images, {len(unique_sizes)} unique sizes")
        for s in unique_sizes[:10]:
            count = sizes.count(s)
            print(f"    {s[0]}x{s[1]}: {count} images")

    # Check if all Megadepth images have same original size
    print("\n" + "=" * 60)
    print("Checking Original Size Distribution (from JSON):")
    print("=" * 60)

    orig_sizes = []
    for json_file in sorted(seg_store.json_dir.glob("*.json"))[:20]:
        with open(json_file, 'r') as f:
            meta = json.load(f)
        orig_sizes.append((meta.get('orig_h', 0), meta.get('orig_w', 0)))

    if orig_sizes:
        unique_orig = list(set(orig_sizes))
        print(f"  Sampled {len(orig_sizes)} images, {len(unique_orig)} unique original sizes")
        for s in unique_orig[:10]:
            count = orig_sizes.count(s)
            print(f"    {s[0]}x{s[1]}: {count} images")

    return seg_store


def simulate_coordinate_mapping(seg_store, image_path, comatch_hw=(832, 832)):
    """Simulate the coordinate mapping process."""
    print("\n" + "=" * 60)
    print("Simulating Coordinate Mapping:")
    print("=" * 60)

    seg = seg_store.get(image_path)
    if not seg:
        print(f"  No segmentation found for {image_path}")
        return

    seg_hw = (seg['height'], seg['width'])
    match_hw = comatch_hw  # CoMatch input size

    print(f"  Image: {image_path}")
    print(f"  CoMatch input size (match_hw): {match_hw}")
    print(f"  OneFormer seg size (seg_hw): {seg_hw}")
    print(f"  Sizes match: {match_hw == seg_hw}")

    if match_hw != seg_hw:
        # Calculate scaling
        scale_x = seg_hw[1] / match_hw[1]  # seg_w / match_w
        scale_y = seg_hw[0] / match_hw[0]  # seg_h / match_h
        print(f"  Required scale: ({scale_x:.4f}, {scale_y:.4f})")
        print(f"  This mapping assumes: CoMatch coords are in CoMatch's {match_hw} space")
        print(f"  But CoMatch mkpts0_c coordinates are actually in ORIGINAL image space!")
        print()
        print("  *** POTENTIAL MISMATCH DETECTED ***")
        print(f"  If CoMatch outputs coords in original image space (e.g. 1920x1080)")
        print(f"  and OneFormer seg is in {seg_hw} space,")
        print(f"  then mapping directly from CoMatch coords using ({match_hw} -> {seg_hw})")
        print(f"  would give WRONG results!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug resolution mismatch")
    parser.add_argument('--oneformer-dir', type=str, required=True)
    parser.add_argument('--test-pair', nargs=2, type=str, help='Two image paths to check')
    parser.add_argument('--comatch-dump', type=str, help='CoMatch dump directory')
    parser.add_argument('--comatch-hw', nargs=2, type=int, default=[832, 832],
                        help='CoMatch input size (H W)')

    args = parser.parse_args()

    seg_store = debug_resolution_mismatch(
        args.oneformer_dir,
        image_paths=args.test_pair
    )

    if args.test_pair:
        simulate_coordinate_mapping(
            seg_store,
            args.test_pair[0],
            comatch_hw=tuple(args.comatch_hw)
        )