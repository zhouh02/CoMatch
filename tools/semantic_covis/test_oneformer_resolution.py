#!/usr/bin/env python3
"""
Test script to verify OneFormer inference keeps original image resolution.

This script checks:
1. Original image size (from cv2.imread)
2. After OneFormer preprocessing (processor returns)
3. OneFormer output segmentation size
4. Whether any resizing occurred
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Test OneFormer resolution behavior')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model-dir', type=str, default=None, help='Local model path')
    parser.add_argument('--meta-dir', type=str, default=None, help='Metadata directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--target-size', type=int, default=None, help='Optional resize target')
    return parser.parse_args()


def load_oneformer(model_dir=None, meta_dir=None, device='cuda'):
    """Load OneFormer model and processor."""
    print("Loading OneFormer...")

    kwargs = {"local_files_only": True} if model_dir else {}

    processor_kwargs = {}
    if meta_dir:
        processor_kwargs["repo_path"] = str(meta_dir)
        processor_kwargs["class_info_file"] = "ade20k_panoptic.json"

    if model_dir:
        processor = OneFormerProcessor.from_pretrained(model_dir, **processor_kwargs)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_dir, **kwargs)
    else:
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large", **processor_kwargs)
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large", **kwargs)

    model = model.to(device)
    model.eval()

    return processor, model


def main():
    args = parse_args()

    print("=" * 60)
    print("OneFormer Resolution Test")
    print("=" * 60)

    # Step 1: Load original image
    print("\n[Step 1] Loading original image...")
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot load image: {args.image}")
        return 1

    orig_h, orig_w = img.shape[:2]
    print(f"  Original image: {orig_w}x{orig_h}")
    print(f"  Shape: {img.shape}")

    # Step 2: Check preprocessing
    print("\n[Step 2] OneFormer preprocessing...")

    processor, model = load_oneformer(args.model_dir, args.meta_dir, args.device)

    # Check processor configuration
    print(f"  Processor type: {type(processor).__name__}")

    # Preprocess manually to see what happens
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_image = Image.fromarray(image)

    print(f"  PIL image size: {pil_image.size}")  # (w, h)

    # Check if processor will resize
    print(f"  Target size param: {args.target_size}")

    # Run processor to see intermediate sizes
    inputs = processor(
        images=pil_image,
        task_inputs=["semantic"],
        return_tensors="pt"
    )

    print(f"  Pixel values shape: {inputs['pixel_values'].shape}")  # [1, 3, H, W]
    pixel_h = inputs['pixel_values'].shape[2]
    pixel_w = inputs['pixel_values'].shape[3]
    print(f"  Preprocessed size: {pixel_w}x{pixel_h}")

    # Step 3: Run inference
    print("\n[Step 3] Running OneFormer inference...")

    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"  Output keys: {outputs.keys()}")

    # Step 4: Post-process
    print("\n[Step 4] Post-processing...")

    # Determine target size for post-processing
    if args.target_size:
        # Use user-specified target size
        if max(orig_w, orig_h) > args.target_size:
            ratio = args.target_size / max(orig_w, orig_h)
            post_h = int(orig_h * ratio)
            post_w = int(orig_w * ratio)
        else:
            post_h, post_w = orig_h, orig_w
    else:
        # No target size: use processed (pre-resized) image size
        post_h, post_w = pil_image.size[1], pil_image.size[0]

    target_size = (post_h, post_w)  # (height, width)
    print(f"  Post-process target size: {target_size[1]}x{target_size[0]}")

    try:
        seg_map = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[target_size],
        )[0]
    except AttributeError:
        seg_map = processor.image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[target_size],
        )[0]

    print(f"  Output seg_map shape: {seg_map.shape}")  # [H, W]
    output_h, output_w = seg_map.shape

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Original:    {orig_w}x{orig_h}")
    print(f"  Preprocessed: {pixel_w}x{pixel_h}")
    print(f"  Output:      {output_w}x{output_h}")
    print(f"  Target size param: {args.target_size}")

    # Determine if resizing occurred
    if output_w == orig_w and output_h == orig_h:
        print(f"\n  ✓ NO RESIZING: Output matches original size")
        return 0
    else:
        print(f"\n  ✗ RESIZING DETECTED!")
        print(f"    - Output is {output_w/orig_w:.2f}x of original width")
        print(f"    - Output is {output_h/orig_h:.2f}x of original height")

        # Check if it's due to target_size
        if args.target_size:
            print(f"    - Cause: --target-size={args.target_size} parameter")
        else:
            print(f"    - Cause: OneFormer internal preprocessing")
        return 1


if __name__ == '__main__':
    exit(main() or 0)