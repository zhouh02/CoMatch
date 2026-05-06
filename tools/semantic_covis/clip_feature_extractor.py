#!/usr/bin/env python3
"""
CLIP Feature Extractor for Semantic Covisibility.

Extracts CLIP patch tokens from images for pseudo-label generation.
Designed to work with image_preprocess.py for CoMatch-style preprocessing.

Usage:
    from clip_feature_extractor import CLIPFeatureExtractor

    extractor = CLIPFeatureExtractor()
    result = extractor.extract("path/to/image.jpg")

    # Or legacy tuple return:
    patch_features, grid_size = extractor.extract("path/to/image.jpg", return_dict=False)

Grid sizes for common CLIP models:
    - clip-vit-large-patch14-336:  24 x 24 (336 / 14)
    - clip-vit-base-patch16:      14 x 14 (224 / 16)
    - clip-vit-base-patch32:       7 x  7 (224 / 32)

Preprocessing pipeline:
    1. read_rgb_for_clip_and_comatch(): CoMatch resize/pad to 832x832 RGB
    2. CLIP processor: resize to model input (336 or 224)
    3. CLIP ViT: extract patch tokens (exclude CLS)
    4. L2 normalize
    5. Output: [N, D] patch features
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Check dependencies
_required_packages = ["transformers", "torch", "PIL", "tqdm"]
_missing_packages = []
for pkg in _required_packages:
    try:
        __import__(pkg)
    except ImportError:
        _missing_packages.append(pkg)

if _missing_packages:
    print(
        "Error: Missing required packages. Please install them with:\n"
        "  pip install {}\n"
        "Full installation:\n"
        "  pip install transformers pillow tqdm matplotlib opencv-python torch".format(
            " ".join(_missing_packages)
        ),
        file=sys.stderr,
    )
    sys.exit(1)

from transformers import CLIPImageProcessor, CLIPVisionModel
from tqdm import tqdm

# Import CoMatch-style preprocessing from sibling module
from image_preprocess import read_rgb_for_clip_and_comatch


# =============================================================================
# CLIP Feature Extractor
# =============================================================================


class CLIPFeatureExtractor:
    """Extract CLIP patch tokens from images.

    This extractor processes images through CoMatch-style preprocessing
    before feeding them to CLIP for feature extraction.

    Attributes:
        model_name: HuggingFace model name or local path
        device: 'cuda' or 'cpu'
        model: CLIPVisionModel instance
        processor: CLIPImageProcessor instance
        dtype: torch dtype for model inference
        grid_size: (Gh, Gw) number of patches
        patch_size: size of each patch in pixels
        image_size: model input image size
    """

    # Supported models and their grid sizes
    KNOWN_MODELS = {
        "openai/clip-vit-large-patch14-336": {"grid": (24, 24), "patch": 14, "image": 336},
        "openai/clip-vit-base-patch16": {"grid": (14, 14), "patch": 16, "image": 224},
        "openai/clip-vit-base-patch32": {"grid": (7, 7), "patch": 32, "image": 224},
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        long_edge: int = 832,
    ):
        """Initialize CLIP feature extractor.

        Args:
            model_name: HuggingFace model name or local path to CLIP model
            device: Device for inference ('cuda', 'cpu', or None for auto)
            dtype: Data type for model inference
            long_edge: CoMatch-style resize target (default: 832)
        """
        self.model_name = model_name
        self.dtype = dtype
        self.long_edge = long_edge

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("Loading CLIP model: {}".format(model_name))
        print("Device: {}".format(self.device))
        print("CoMatch long_edge: {}".format(self.long_edge))

        # Load model
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        # Load processor
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        # Configure processor to prevent unwanted transforms
        # This ensures no random crop or other spatial perturbations
        if hasattr(self.processor, "do_resize"):
            self.processor.do_resize = True
        if hasattr(self.processor, "do_normalize"):
            self.processor.do_normalize = True
        # Note: processor will resize to model's image_size (336 or 224)
        # This is expected and correct - we're just maintaining spatial layout

        # Determine patch grid size from model config
        self._parse_grid_size()

        print("CLIP input image size: {}".format(self.image_size))
        print("CLIP patch size: {}".format(self.patch_size))
        print("CLIP patch grid: {}".format(self.grid_size))

    def _parse_grid_size(self) -> None:
        """Parse patch grid size from model configuration."""
        config = self.model.config

        # Get image size
        if hasattr(config, "image_size"):
            image_size = config.image_size
            if isinstance(image_size, (list, tuple)):
                image_size = image_size[0]
        elif hasattr(config, "vision_config"):
            image_size = config.vision_config.get("image_size", 224)
        else:
            image_size = 224

        # Get patch size
        if hasattr(config, "patch_size"):
            patch_size = config.patch_size
        elif hasattr(config, "vision_config"):
            patch_size = config.vision_config.get("patch_size", 16)
        else:
            patch_size = 16

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)

    @torch.no_grad()
    def extract(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        return_dict: bool = True,
        return_numpy: bool = True,
    ) -> Union[Dict[str, Any], Tuple[np.ndarray, Tuple[int, int]]]:
        """Extract CLIP patch tokens from an image.

        Preprocessing pipeline:
            1. Read and preprocess image using CoMatch style (resize 832, pad square)
            2. CLIP processor resizes to model input (336 or 224) - spatial layout preserved
            3. Extract patch tokens (exclude CLS token at index 0)
            4. L2 normalize

        Args:
            image: Input as numpy array (H, W, 3), file path, or PIL Image
            return_dict: If True, return dict with features and preprocessing info
            return_numpy: If True, return numpy arrays; else torch.Tensors

        Returns:
            If return_dict=True:
                dict with keys:
                    - patch_features: [N, D] tensor, L2-normalized patch features
                    - grid_size: (Gh, Gw) CLIP patch grid size
                    - preprocess: dict with preprocessing metadata:
                        - original_hw: (H, W) original image size
                        - resized_hw: (H, W) after resize (before padding)
                        - valid_hw: (H, W) = resized_hw
                        - pad_size: final square padded size
                        - scale: [w_scale, h_scale] from original to resized
                        - valid_mask: [pad_H, pad_W] bool mask

            If return_dict=False (legacy):
                (patch_features, grid_size) tuple
        """
        # Step 1: CoMatch-style preprocessing (resize 832, pad to square)
        if isinstance(image, (str, Path)):
            preprocess_result = read_rgb_for_clip_and_comatch(
                str(image),
                long_edge=self.long_edge,
                df=8,
                pad_to_square=True,
                return_pil=False,
            )
        elif isinstance(image, Image.Image):
            # Save PIL image temporarily
            tmp_path = Path("/tmp") / "clip_temp_{}.png".format(id(image))
            image.save(str(tmp_path))
            preprocess_result = read_rgb_for_clip_and_comatch(
                str(tmp_path),
                long_edge=self.long_edge,
                df=8,
                pad_to_square=True,
                return_pil=False,
            )
            tmp_path.unlink(missing_ok=True)
        else:
            # numpy array - convert to PIL then preprocess
            pil_image = Image.fromarray(image.astype(np.uint8))
            tmp_path = Path("/tmp") / "clip_temp_{}.png".format(id(image))
            pil_image.save(str(tmp_path))
            preprocess_result = read_rgb_for_clip_and_comatch(
                str(tmp_path),
                long_edge=self.long_edge,
                df=8,
                pad_to_square=True,
                return_pil=False,
            )
            tmp_path.unlink(missing_ok=True)

        # Step 2: CLIP processor - resize to model input size
        # This is a simple resize, spatial layout is preserved
        image_pil = Image.fromarray(preprocess_result["image"].astype(np.uint8))
        inputs = self.processor(images=image_pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)

        # Step 3: Forward through CLIP vision model
        outputs = self.model(pixel_values)

        # Step 4: Extract patch tokens (exclude CLS token at index 0)
        # last_hidden_state: [batch, seq_len, hidden_dim]
        # seq_len = 1 + num_patches, CLS is at index 0
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        # Step 5: Reshape to [1, Gh, Gw, D] and L2 normalize
        gh, gw = self.grid_size
        hidden_dim = patch_tokens.shape[-1]
        patch_features = patch_tokens.reshape(1, gh, gw, hidden_dim)

        # L2 normalize
        patch_features = patch_features / (
            torch.norm(patch_features, p=2, dim=-1, keepdim=True) + 1e-8
        )

        # Reshape to [N, D]
        patch_features = patch_features.reshape(1, gh * gw, hidden_dim)

        # Convert to numpy if requested
        if return_numpy:
            patch_features_np = patch_features.cpu().numpy()
            valid_mask_np = preprocess_result["valid_mask"]
            scale_np = preprocess_result["scale"]
        else:
            patch_features_np = patch_features
            valid_mask_np = torch.from_numpy(preprocess_result["valid_mask"])
            scale_np = torch.from_numpy(preprocess_result["scale"])

        # Build result
        if return_dict:
            return {
                "patch_features": patch_features_np,
                "grid_size": self.grid_size,
                "preprocess": {
                    "original_hw": preprocess_result["original_hw"],
                    "resized_hw": preprocess_result["resized_hw"],
                    "valid_hw": preprocess_result["valid_hw"],
                    "pad_size": preprocess_result["pad_size"],
                    "scale": scale_np,
                    "valid_mask": valid_mask_np,
                },
            }
        else:
            # Legacy tuple return for backward compatibility
            return patch_features_np, self.grid_size

    @torch.no_grad()
    def extract_batch(
        self,
        images: List[Union[np.ndarray, str, Path, Image.Image]],
        batch_size: int = 8,
        return_dict: bool = False,
    ) -> Union[Dict[str, Any], Tuple[np.ndarray, List[Tuple[int, int]]]]:
        """Extract CLIP features from a batch of images.

        Args:
            images: List of input images
            batch_size: Batch size for processing
            return_dict: If True, return dict with features and preprocessing info

        Returns:
            If return_dict=True:
                dict with:
                    - patch_features: [B, N, D] stacked features
                    - grid_sizes: list of (Gh, Gw) per image
                    - preprocess: list of preprocess dicts

            If return_dict=False:
                (all_features, grid_sizes) tuple
        """
        all_features = []
        all_preprocess = []
        grid_sizes = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting CLIP features"):
            batch_images = images[i : i + batch_size]
            batch_results = []

            for img in batch_images:
                result = self.extract(img, return_dict=True, return_numpy=True)
                batch_results.append(result)
                all_features.append(result["patch_features"][0])  # [N, D]
                all_preprocess.append(result["preprocess"])
                grid_sizes.append(result["grid_size"])

        all_features = np.stack(all_features, axis=0)  # [B, N, D]

        if return_dict:
            return {
                "patch_features": all_features,
                "grid_size": self.grid_size,  # Same for all
                "grid_sizes": grid_sizes,
                "preprocess": all_preprocess,
            }
        else:
            return all_features, grid_sizes

    def close(self):
        """Clean up model resources."""
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Utility Functions
# =============================================================================


def resize_heatmap_to_coarse(
    heatmap: np.ndarray,
    source_grid: Tuple[int, int],
    target_grid: Tuple[int, int] = (104, 104),
) -> np.ndarray:
    """Resize a heatmap from CLIP grid to CoMatch coarse grid.

    Args:
        heatmap: Input heatmap [Gh, Gw] or [B, Gh, Gw]
        source_grid: (Gh, Gw) source grid size (CLIP grid)
        target_grid: (Gh, Gw) target grid size (default: CoMatch coarse 104x104)

    Returns:
        Resized heatmap
    """
    import cv2

    gh, gw = source_grid
    th, tw = target_grid

    if heatmap.ndim == 2:
        return cv2.resize(heatmap, (tw, th), interpolation=cv2.INTER_LINEAR)
    elif heatmap.ndim == 3:
        resized = []
        for i in range(heatmap.shape[0]):
            resized.append(
                cv2.resize(heatmap[i], (tw, th), interpolation=cv2.INTER_LINEAR)
            )
        return np.stack(resized, axis=0)
    else:
        raise ValueError("Invalid heatmap ndim: {}".format(heatmap.ndim))


def apply_valid_mask(
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


# =============================================================================
# Main Test
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CLIP Feature Extractor")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP model name or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=832,
        help="CoMatch-style resize long edge (default: 832)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CLIP Feature Extractor Test")
    print("=" * 60)
    print("  Model:     {}".format(args.model_name))
    print("  Device:    {}".format(args.device or "auto"))
    print("  Long edge: {}".format(args.long_edge))
    print("=" * 60)

    # Initialize extractor
    extractor = CLIPFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        long_edge=args.long_edge,
    )

    # Prepare test image
    if args.image and os.path.exists(args.image):
        print("\nUsing test image: {}".format(args.image))
        image = args.image
    else:
        # Create synthetic test image with known dimensions
        print("\nNo valid image provided, creating synthetic 1920x1080 test image...")
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Add checkerboard pattern for visual verification
        for i in range(0, 1080, 60):
            for j in range(0, 1920, 80):
                if (i // 60 + j // 80) % 2 == 0:
                    image[i : i + 60, j : j + 80] = [200, 150, 100]

    # Extract features
    print("\nExtracting features...")
    result = extractor.extract(image, return_dict=True, return_numpy=True)

    # Print results
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)

    pf = result["patch_features"]
    pp = result["preprocess"]

    print("  patch_features.shape: {}".format(pf.shape))
    print("    - N (num patches):  {}".format(pf.shape[0]))
    print("    - D (feature dim):  {}".format(pf.shape[1]))
    print("  grid_size:            {}".format(result["grid_size"]))

    print("\n  Preprocessing info:")
    print("    - original_hw:     {}".format(pp["original_hw"]))
    print("    - resized_hw:       {}".format(pp["resized_hw"]))
    print("    - pad_size:         {}".format(pp["pad_size"]))
    print("    - scale:            {}".format(pp["scale"]))

    valid_mask = pp["valid_mask"]
    valid_ratio = valid_mask.sum() / valid_mask.size
    print("\n  valid_mask:")
    print("    - shape:            {}".format(valid_mask.shape))
    print("    - valid_ratio:      {:.4f} ({} / {})".format(
        valid_ratio, valid_mask.sum(), valid_mask.size
    ))

    # Also test legacy tuple return
    print("\n" + "-" * 60)
    print("Legacy tuple return test:")
    print("-" * 60)
    pf_tuple, gs_tuple = extractor.extract(image, return_dict=False)
    print("  patch_features.shape: {}".format(pf_tuple.shape))
    print("  grid_size:            {}".format(gs_tuple))

    # Test heatmap resize utility
    print("\n" + "-" * 60)
    print("Heatmap resize to CoMatch coarse (104x104):")
    print("-" * 60)
    dummy_heatmap = np.random.rand(*result["grid_size"]).astype(np.float32)
    resized = resize_heatmap_to_coarse(dummy_heatmap, result["grid_size"])
    print("  Source: {} -> Target: {}".format(dummy_heatmap.shape, resized.shape))

    # Cleanup
    extractor.close()

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
