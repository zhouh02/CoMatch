#!/usr/bin/env python3
"""
Visualize image pairs in original Megadepth resolution with center alignment and padding.

This script:
1. Reads image pair list from JSONL
2. Loads images at original resolution
3. Aligns them center-to-center with padding for size mismatch
4. Visualizes with size info
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize image pairs in original size')
    parser.add_argument('--pair-list', type=str, required=True,
                        help='JSONL file with image pairs')
    parser.add_argument('--image-root', type=str, required=True,
                        help='Root directory for images')
    parser.add_argument('--output-dir', type=str, default='outputs/pair_vis',
                        help='Output directory')
    parser.add_argument('--max-pairs', type=int, default=10,
                        help='Max number of pairs to visualize')
    parser.add_argument('--padding-color', type=int, nargs=3, default=[0, 0, 0],
                        help='RGB color for padding (default: black)')
    return parser.parse_args()


def load_image(image_path, image_root=None):
    """Load image from path, handling list-wrapped paths."""
    path_str = str(image_path).strip()

    # Handle list-wrapped paths like "['path.jpg']"
    if path_str.startswith("[") and path_str.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(path_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                path_str = str(parsed[0])
        except Exception:
            pass

    path = Path(path_str)
    if not path.is_absolute() and image_root:
        path = Path(image_root) / path

    img = cv2.imread(str(path))
    if img is None:
        return None, path_str
    return img, str(path)


def center_align_and_pad(img1, img2, pad_color=(0, 0, 0)):
    """
    Center-align two images and pad the smaller one.

    Args:
        img1: First image [H, W, C]
        img2: Second image [H, W, C]
        pad_color: RGB color for padding

    Returns:
        Canvas with both images center-aligned
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Target canvas size: max of both dimensions
    max_h = max(h1, h2)
    max_w = max(w1, w2)

    # Create canvas with padding color
    canvas = np.full((max_h, max_w, 3), pad_color, dtype=np.uint8)

    # Calculate offset for center alignment
    offset_y1 = (max_h - h1) // 2
    offset_x1 = (max_w - w1) // 2
    offset_y2 = (max_h - h2) // 2
    offset_x2 = (max_w - w2) // 2

    # Place images
    canvas[offset_y1:offset_y1+h1, offset_x1:offset_x1+w1] = img1
    canvas[offset_y2:offset_y2+h2, offset_x2:offset_x2+w2] = img2

    return canvas


def visualize_pair(img1, img2, info, output_path):
    """Visualize a single pair with size information."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create side-by-side canvas with center alignment
    pad_color = info.get('padding_color', (0, 0, 0))
    canvas = center_align_and_pad(img1, img2, pad_color)

    # Add info bar
    info_bar_h = 60
    info_bar = np.zeros((info_bar_h, canvas.shape[1], 3), dtype=np.uint8)

    # Text: image sizes
    text1 = f"Image0: {w1}x{h1}"
    text2 = f"Image1: {w2}x{h2}"
    text3 = f"Canvas: {canvas.shape[1]}x{canvas.shape[0]}"

    cv2.putText(info_bar, text1, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(info_bar, text2, (400, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(info_bar, text3, (800, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Combine
    result = np.vstack([info_bar, canvas])

    # Add divider line between images
    divider_x = canvas.shape[1] // 2
    cv2.line(result, (divider_x, info_bar_h), (divider_x, result.shape[0]),
             (128, 128, 128), 2)

    # Save
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")

    return canvas.shape


def load_pair_list(pair_list_path):
    """Load pairs from JSONL file."""
    pairs = []
    with open(pair_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                entry = json.loads(line)
                pairs.append(entry)
            except json.JSONDecodeError:
                continue
    return pairs


def main():
    args = parse_args()

    # Load pair list
    if not Path(args.pair_list).exists():
        print(f"ERROR: Pair list not found: {args.pair_list}")
        return 1

    pairs = load_pair_list(args.pair_list)
    print(f"Loaded {len(pairs)} pairs from {args.pair_list}")

    # Limit pairs
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"Limited to {args.max_pairs} pairs")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    success_count = 0
    for i, pair in enumerate(pairs):
        img0_path = pair.get('image0', '')
        img1_path = pair.get('image1', '')

        if not img0_path or not img1_path:
            print(f"  Skip pair {i}: missing path")
            continue

        # Load images
        img0, img0_abs = load_image(img0_path, args.image_root)
        img1, img1_abs = load_image(img1_path, args.image_root)

        if img0 is None:
            print(f"  Skip pair {i}: cannot load {img0_abs}")
            continue
        if img1 is None:
            print(f"  Skip pair {i}: cannot load {img1_abs}")
            continue

        print(f"\nProcessing pair {i}:")
        print(f"  Image0: {img0.shape[1]}x{img0.shape[0]} - {Path(img0_abs).name}")
        print(f"  Image1: {img1.shape[1]}x{img1.shape[0]} - {Path(img1_abs).name}")

        # Check size match
        size_match = (img0.shape == img1.shape)
        print(f"  Size match: {'TRUE' if size_match else 'FALSE'}")

        # Visualize
        info = {
            'pair_idx': i,
            'image0': img0_abs,
            'image1': img1_abs,
            'padding_color': args.padding_color,
        }

        output_path = output_dir / f"pair_{i:04d}_original_size.png"
        canvas_size = visualize_pair(img0, img1, info, output_path)
        print(f"  Canvas: {canvas_size[1]}x{canvas_size[0]}")

        success_count += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(pairs)} pairs visualized")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)