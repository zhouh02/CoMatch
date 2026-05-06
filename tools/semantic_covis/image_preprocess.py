#!/usr/bin/env python3
"""
Image Preprocessing for CLIP Pseudo-label Generation.

Reproduces CoMatch's image resize/pad logic but preserves RGB color.
Designed to work with CLIPFeatureExtractor.

Reference: src/utils/dataset.py
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# Core Preprocessing Functions (CoMatch style)
# =============================================================================


def get_resized_wh(w: int, h: int, resize: int | None = None) -> tuple[int, int]:
    """Resize image so that the longer edge equals `resize`.

    Identical to src/utils/dataset.py::get_resized_wh()

    Args:
        w: Original width
        h: Original height
        resize: Target size for longer edge (None for no resize)

    Returns:
        (new_width, new_height)

    Example:
        >>> get_resized_wh(1920, 1080, 832)
        (832, 468)  # 1920/832 = 4.5, 1080/832 = 1.3, so height becomes 832/4.5=468
    """
    if resize is not None:
        scale = resize / max(h, w)
        w_new = int(round(w * scale))
        h_new = int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w: int, h: int, df: int | None = None) -> tuple[int, int]:
    """Round dimensions down to be divisible by `df`.

    Identical to src/utils/dataset.py::get_divisible_wh()

    Args:
        w: Current width
        h: Current height
        df: Divisibility factor (None for no change)

    Returns:
        (rounded_width, rounded_height)

    Example:
        >>> get_divisible_wh(832, 468, 8)
        (832, 464)  # 468 // 8 * 8 = 464
    """
    if df is not None:
        w_new = int(w // df * df)
        h_new = int(h // df * df)
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(
    inp: np.ndarray, pad_size: int, ret_mask: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Zero-pad image to square shape at bottom-right.

    Identical to src/utils/dataset.py::pad_bottom_right()

    Args:
        inp: Input array (H, W) or (H, W, C)
        pad_size: Target size for both dimensions
        ret_mask: If True, return validity mask

    Returns:
        (padded_image, mask)
        - padded_image: Zero-padded array
        - mask: Boolean mask where valid regions are True (if ret_mask=True)

    Example:
        >>> arr = np.zeros((480, 640, 3))
        >>> padded, mask = pad_bottom_right(arr, 640)
        >>> mask.shape
        (640, 640)
        >>> mask[479, 639]
        True
        >>> mask[480, 0]
        False
    """
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])

    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        mask = None
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: inp.shape[0], : inp.shape[1]] = True

    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[0], : inp.shape[1]] = inp
        mask = None
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : inp.shape[0], : inp.shape[1]] = True

    else:
        raise NotImplementedError(f"Unsupported input ndim: {inp.ndim}")

    return padded, mask


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read an image as RGB numpy array.

    Args:
        path: Path to the image file

    Returns:
        RGB image as numpy array (H, W, 3) in range [0, 255], dtype=uint8

    Supported formats:
        - Local files (cv2.imread)
        - S3 paths (s3://bucket/path)
    """
    path = str(path)

    if path.startswith("s3://"):
        try:
            from src.utils.dataset import load_array_from_s3

            data = load_array_from_s3(path, client=None, cv_type=cv2.IMREAD_COLOR)
            # load_array_from_s3 returns BGR, convert to RGB
            image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        except ImportError:
            raise RuntimeError(
                f"Cannot load S3 image: {path}. "
                "Make sure S3 client is configured."
            )
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# =============================================================================
# Main Preprocessing Function
# =============================================================================


def read_rgb_for_clip_and_comatch(
    image_path: str | Path,
    long_edge: int = 832,
    df: int = 8,
    pad_to_square: bool = True,
    return_pil: bool = False,
) -> dict:
    """Read and preprocess an RGB image using CoMatch style.

    This function replicates CoMatch's image preprocessing logic but
    preserves RGB color for CLIP feature extraction.

    Processing pipeline:
        1. Read image as RGB
        2. Resize so that longer edge = long_edge
        3. Round dimensions down to be divisible by df
        4. Zero-pad to square (if pad_to_square=True)

    Args:
        image_path: Path to the input image
        long_edge: Target size for longer edge (default: 832)
        df: Round dimensions down to this divisor (default: 8)
        pad_to_square: If True, pad to square shape (default: True)
        return_pil: If True, return PIL.Image instead of numpy array

    Returns:
        Dictionary with keys:
            - image: RGB image, shape (H, W, 3) or PIL.Image
            - resized_hw: (H, W) after resize (before padding)
            - valid_hw: (H_valid, W_valid) = resized_hw (before padding)
            - scale: np.array([w/w_new, h/h_new]) scale factor from original to resized
            - valid_mask: np.ndarray [H_pad, W_pad], True for valid pixels
            - original_hw: (H_orig, W_orig) original image dimensions
            - pad_size: Final padded size (if pad_to_square=True) or resized size

    Example:
        >>> result = read_rgb_for_clip_and_comatch("test.jpg")
        >>> result['image'].shape
        (832, 832, 3)  # Padded to square
        >>> result['valid_mask'].shape
        (832, 832)
        >>> result['valid_mask'][:468, :832].all()  # Valid region
        True
        >>> result['valid_mask'][468:, :].all()  # Padded region
        False
    """
    # Step 0: Read image
    image = read_rgb_image(image_path)
    h_orig, w_orig = image.shape[:2]

    # Step 1: Resize to long_edge
    w_resized, h_resized = get_resized_wh(w_orig, h_orig, long_edge)

    # Step 2: Round down to divisible by df
    w_valid, h_valid = get_divisible_wh(w_resized, h_resized, df)

    # Resize image
    image = cv2.resize(image, (w_valid, h_valid), interpolation=cv2.INTER_LINEAR)

    # Compute scale: original -> resized_valid
    # scale[0] = w_orig / w_valid, scale[1] = h_orig / h_valid
    scale = np.array([w_orig / w_valid, h_orig / h_valid], dtype=np.float32)

    # Step 3: Pad to square
    if pad_to_square:
        pad_size = max(h_valid, w_valid)
        image, valid_mask = pad_bottom_right(image, pad_size, ret_mask=True)
    else:
        pad_size = max(h_valid, w_valid)
        valid_mask = np.ones((h_valid, w_valid), dtype=bool)

    # Convert to PIL if requested
    if return_pil:
        image = Image.fromarray(image.astype(np.uint8))

    return {
        "image": image,
        "resized_hw": (h_valid, w_valid),  # After resize, before padding
        "valid_hw": (h_valid, w_valid),  # Alias for consistency
        "scale": scale,
        "valid_mask": valid_mask,
        "original_hw": (h_orig, w_orig),
        "pad_size": pad_size,
    }


# =============================================================================
# Convenience Functions
# =============================================================================


def apply_valid_mask_to_heatmap(
    heatmap: np.ndarray,
    valid_mask: np.ndarray,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Apply valid_mask to a heatmap, setting masked regions to pad_value.

    Args:
        heatmap: Heatmap array (H, W) or (B, H, W)
        valid_mask: Valid pixel mask (H, W), True = valid
        pad_value: Value to set for invalid (padded) regions

    Returns:
        Heatmap with masked regions set to pad_value
    """
    heatmap = heatmap.copy()
    if heatmap.ndim == 2:
        heatmap[~valid_mask] = pad_value
    elif heatmap.ndim == 3:
        for i in range(heatmap.shape[0]):
            heatmap[i, ~valid_mask] = pad_value
    return heatmap


def resize_heatmap_to_grid(
    heatmap: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize a heatmap from source grid to target grid.

    Args:
        heatmap: Heatmap array (H, W) or (B, H, W)
        source_hw: (H, W) source grid size
        target_hw: (H, W) target grid size
        interpolation: cv2 interpolation mode

    Returns:
        Resized heatmap
    """
    h_src, w_src = source_hw
    h_tgt, w_tgt = target_hw

    if heatmap.ndim == 2:
        return cv2.resize(heatmap, (w_tgt, h_tgt), interpolation=interpolation)
    elif heatmap.ndim == 3:
        resized = []
        for i in range(heatmap.shape[0]):
            resized.append(
                cv2.resize(heatmap[i], (w_tgt, h_tgt), interpolation=interpolation)
            )
        return np.stack(resized, axis=0)
    else:
        raise ValueError(f"Invalid heatmap ndim: {heatmap.ndim}")


# =============================================================================
# Main Test
# =============================================================================


def visualize_preprocessing(
    image_path: str,
    output_dir: str = "tools/semantic_covis/test_output",
) -> None:
    """Test and visualize image preprocessing.

    Args:
        image_path: Path to input image
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {image_path}")

    # Preprocess
    result = read_rgb_for_clip_and_comatch(
        image_path,
        long_edge=832,
        df=8,
        pad_to_square=True,
    )

    # Print info
    print("\nPreprocessing result:")
    print(f"  Original size: {result['original_hw']}")
    print(f"  Resized size:  {result['resized_hw']}")
    print(f"  Valid size:    {result['valid_hw']}")
    print(f"  Pad size:      {result['pad_size']}")
    print(f"  Scale:         {result['scale']}")
    print(f"  Image shape:   {result['image'].shape}")
    print(f"  Mask shape:    {result['valid_mask'].shape}")

    # Save visualizations
    base_name = Path(image_path).stem

    # Save padded RGB image
    rgb_path = output_dir / f"{base_name}_rgb_padded.png"
    if isinstance(result["image"], np.ndarray):
        Image.fromarray(result["image"].astype(np.uint8)).save(rgb_path)
    else:
        result["image"].save(rgb_path)
    print(f"\nSaved RGB: {rgb_path}")

    # Save valid mask
    mask = result["valid_mask"].astype(np.uint8) * 255
    mask_path = output_dir / f"{base_name}_valid_mask.png"
    Image.fromarray(mask).save(mask_path)
    print(f"Saved mask: {mask_path}")

    # Create side-by-side visualization
    rgb = np.array(result["image"])
    # Blend RGB with mask visualization
    mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_vis[result["valid_mask"]] = [0, 255, 0]  # Green for valid
    mask_vis[~result["valid_mask"]] = [255, 0, 0]  # Red for padding

    blended = cv2.addWeighted(rgb, 0.7, mask_vis, 0.3, 0)
    blend_path = output_dir / f"{base_name}_overlay.png"
    Image.fromarray(blended).save(blend_path)
    print(f"Saved overlay: {blend_path}")

    # Save metadata as JSON
    import json

    meta_path = output_dir / f"{base_name}_metadata.json"
    meta = {
        "original_hw": result["original_hw"],
        "resized_hw": result["resized_hw"],
        "valid_hw": result["valid_hw"],
        "scale": result["scale"].tolist(),
        "pad_size": result["pad_size"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test image preprocessing for CLIP pseudo-label generation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tools/semantic_covis/test_output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=832,
        help="Long edge resize target (default: 832)",
    )
    parser.add_argument(
        "--df",
        type=int,
        default=8,
        help="Divisibility factor (default: 8)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Image Preprocessing Test")
    print("=" * 60)
    print(f"  Long edge:  {args.long_edge}")
    print(f"  Divisible by: {args.df}")
    print("=" * 60)

    # If no image provided, create a synthetic test
    if args.image is None or not os.path.exists(args.image):
        print("\nNo valid image provided, creating synthetic test image...")

        # Create a test image with known dimensions
        h, w = 1080, 1920  # 16:9 aspect ratio
        test_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Add some pattern to make it interesting
        for i in range(0, h, 60):
            test_image[i : i + 30, :, 0] = 200  # Red stripes
        for j in range(0, w, 80):
            test_image[:, j : j + 40, 1] = 150  # Green stripes
        test_image[:, :, 2] = 100  # Blue channel

        # Save synthetic image for testing
        synth_dir = Path(args.output_dir)
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_path = synth_dir / "synthetic_test.png"
        Image.fromarray(test_image).save(synth_path)
        print(f"Created synthetic image: {synth_path}")

        args.image = str(synth_path)

    # Run visualization
    visualize_preprocessing(args.image, args.output_dir)

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
