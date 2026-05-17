#!/usr/bin/env python3
"""
OneFormer Semantic Segmentation Exporter.

This script runs OneFormer on a list of images and exports semantic segmentation
results for semantic covisibility diagnostics.

Usage:
    # From image list
    python tools/semantic_covis/export_oneformer_dir.py \
        --image-list outputs/semantic_diagnostic_outdoor/full_test_image_list.txt \
        --image-root /path/to/megadepth \
        --output-dir outputs/oneformer_outdoor_test \
        --model-dir /ssd-data3/zh2025/HFModel/oneformer_ade20k_swin_large \
        --meta-dir /ssd-data3/zh2025/HFModel/oneformer_demo \
        --device cuda

    # Single image
    python tools/semantic_covis/export_oneformer_dir.py \
        --image-dir /path/to/images \
        --output-dir outputs/oneformer_outdoor_test \
        --model-dir /ssd-data3/zh2025/HFModel/oneformer_ade20k_swin_large \
        --meta-dir /ssd-data3/zh2025/HFModel/oneformer_demo \
        --device cuda

Output structure:
    outputs/oneformer_outdoor_test/
    ├── npz/                    # semantic segmentation arrays
    ├── json/                   # metadata with id2label
    ├── vis_panoptic_overlay/  # visualization
    ├── vis_semantic_label_overlay/
    ├── label_id_map.json
    └── summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Check dependencies
_required = ["torch", "transformers", "numpy", "PIL", "cv2"]
_missing = []
for pkg in _required:
    try:
        __import__(pkg)
    except ImportError:
        _missing.append(pkg)

if _missing:
    print(f"Error: Missing packages: {_missing}", file=sys.stderr)
    print("Install with: pip install " + " ".join(_missing), file=sys.stderr)
    sys.exit(1)

import numpy as np
import torch
from PIL import Image

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


# =============================================================================
# OneFormer Segmentation
# =============================================================================

class OneFormerSegmentation:
    """OneFormer semantic segmentation wrapper."""

    def __init__(
        self,
        model_dir: str = None,
        meta_dir: str = None,
        device: str = None,
        model_name: str = "shi-labs/oneformer_ade20k_swin_large",
    ):
        """Initialize OneFormer model.

        Args:
            model_dir: Local path to model (priority over model_name)
            meta_dir: Local path to metadata (id2label.json)
            device: 'cuda' or 'cpu'
            model_name: HuggingFace model name (fallback if model_dir not found)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading OneFormer model...")
        print(f"  Model dir: {model_dir or 'from hub'}")
        print(f"  Meta dir:   {meta_dir}")
        print(f"  Device:     {self.device}")

        # Load processor
        if model_dir:
            self.processor = OneFormerProcessor.from_pretrained(model_dir)
        else:
            self.processor = OneFormerProcessor.from_pretrained(model_name)

        # Load model
        if model_dir:
            self.model = OneFormerForUniversalSegmentation.from_pretrained(model_dir)
        else:
            self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load id2label
        if meta_dir and os.path.exists(meta_dir):
            id2label_path = os.path.join(meta_dir, "id2label.json")
            if os.path.exists(id2label_path):
                with open(id2label_path, 'r') as f:
                    self.id2label = json.load(f)
            else:
                self.id2label = self.processor.id2label
        else:
            self.id2label = self.processor.id2label

        print(f"  Loaded {len(self.id2label)} semantic classes")

    @torch.no_grad()
    def segment_image(self, image_path: str, target_size: int = None):
        """Run OneFormer on a single image.

        Args:
            image_path: Path to input image
            target_size: Optional resize target (longer edge), None for no resize

        Returns:
            dict with keys:
                - semantic_label: [H, W] int array of class IDs
                - segment_score: [H, W] float array of confidence
                - height: original height
                - width: original width
                - processed_hw: (H, W) after preprocessing
                - image_path: original image path
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Optional resize
        if target_size:
            if max(orig_w, orig_h) > target_size:
                ratio = target_size / max(orig_w, orig_h)
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
                image = image.resize((new_w, new_h), Image.LANCZOS)

        # Process
        inputs = self.processor(
            images=image,
            task_inputs=[{"task": "semantic"}],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward
        outputs = self.model(**inputs)

        # Post-process
        result = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]  # PIL uses (w, h), we need (h, w)
        )[0]

        # Get semantic map
        semantic_label = result.cpu().numpy().astype(np.int32)

        # Create segment score map (use class probabilities as confidence)
        class_scores = torch.softmax(outputs.logits[0], dim=0)
        max_scores, _ = class_scores.max(dim=0)
        segment_score = max_scores.cpu().numpy().astype(np.float32)

        return {
            'semantic_label': semantic_label,
            'segment_score': segment_score,
            'height': semantic_label.shape[0],
            'width': semantic_label.shape[1],
            'processed_hw': semantic_label.shape[:2],
            'orig_hw': (orig_h, orig_w),
            'image_path': image_path,
        }


# =============================================================================
# Main Export Function
# =============================================================================

def export_oneformer_dir(
    image_list=None,
    image_dir=None,
    image_root=None,
    output_dir=None,
    model_dir=None,
    meta_dir=None,
    device=None,
    model_name="shi-labs/oneformer_ade20k_swin_large",
    target_size=None,
    save_vis=False,
    save_label_vis=False,
    draw_labels=True,
    skip_existing=True,
):
    """Export OneFormer segmentation for a directory of images.

    Args:
        image_list: Path to text file with one image path per line
        image_dir: Directory containing images (alternative to image_list)
        image_root: Root directory for relative paths in image_list
        output_dir: Output directory for npz/json files
        model_dir: Local path to OneFormer model
        meta_dir: Local path to metadata
        device: 'cuda' or 'cpu'
        model_name: HuggingFace model name (fallback)
        target_size: Optional resize target
        save_vis: Save visualization overlays
        save_label_vis: Save semantic label overlay
        draw_labels: Draw class labels on visualization
        skip_existing: Skip images that already have npz output
    """
    # Determine output structure
    output_dir = Path(output_dir)
    npz_dir = output_dir / "npz"
    json_dir = output_dir / "json"
    vis_dir = output_dir / "vis_panoptic_overlay"
    label_vis_dir = output_dir / "vis_semantic_label_overlay"

    npz_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_paths = []

    if image_list:
        with open(image_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle absolute and relative paths
                    if os.path.isabs(line):
                        image_paths.append(line)
                    elif image_root:
                        image_paths.append(str(Path(image_root) / line))
                    else:
                        image_paths.append(line)
    elif image_dir:
        image_dir = Path(image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_paths.extend(image_dir.glob(ext))
        image_paths = sorted(image_paths)
    else:
        raise ValueError("Must specify either --image-list or --image-dir")

    print(f"Found {len(image_paths)} images to process")

    # Initialize OneFormer
    segmentor = OneFormerSegmentation(
        model_dir=model_dir,
        meta_dir=meta_dir,
        device=device,
        model_name=model_name,
    )

    # Process images
    processed = 0
    skipped = 0
    failed = 0
    missing = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        if not os.path.exists(img_path):
            missing.append(img_path)
            print(f"WARNING: Image not found: {img_path}")
            continue

        # Determine output paths
        img_stem = Path(img_path).stem

        # Try to get relative stem
        if image_root and str(img_path).startswith(str(image_root)):
            rel_path = Path(img_path).relative_to(image_root)
            rel_stem = str(rel_path.with_suffix(''))
        elif image_list and os.path.isabs(img_path):
            # Use stem with some structure
            rel_stem = Path(img_path).stem
        else:
            rel_stem = img_stem

        # npz_path = npz_dir / f"{rel_stem.replace('/', '_').replace('\\\\', '_')}.npz"
        safe_stem = rel_stem.replace("/", "_").replace("\\", "_")
        npz_path = npz_dir / f"{safe_stem}.npz"
        json_path = json_dir / f"{rel_stem.replace('/', '_').replace('\\\\', '_')}.json"

        # Skip existing
        if skip_existing and npz_path.exists() and json_path.exists():
            skipped += 1
            continue

        try:
            # Run segmentation
            result = segmentor.segment_image(img_path, target_size=target_size)

            # Save npz
            np.savez(
                npz_path,
                panoptic_seg=result['semantic_label'],
                semantic_label=result['semantic_label'],
                segment_score=result['segment_score'],
                height=result['height'],
                width=result['width'],
            )

            # Save json
            json_data = {
                'image_path': img_path,
                'relative_stem': rel_stem,
                'height': result['height'],
                'width': result['width'],
                'orig_hw': result['orig_hw'],
                'processed_hw': result['processed_hw'],
                'id2label': segmentor.id2label,
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f)

            processed += 1

        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            failed += 1

    # Save label map
    label_map_path = output_dir / "label_id_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(segmentor.id2label, f, indent=2)

    # Save summary
    summary = {
        'total_images': len(image_paths),
        'processed': processed,
        'skipped': skip_existing,
        'failed': failed,
        'missing': len(missing),
        'output_dir': str(output_dir),
        'id2label': segmentor.id2label,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Missing:  {len(missing)}")
    if missing:
        print(f"\nMissing images (sample):")
        for m in missing[:10]:
            print(f"  - {m}")

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export OneFormer semantic segmentation for images"
    )
    parser.add_argument(
        '--image-list',
        type=str,
        help='Path to text file with image paths (one per line)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images (alternative to --image-list)'
    )
    parser.add_argument(
        '--image-root',
        type=str,
        help='Root directory for relative paths in image-list'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for npz/json files'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Local path to OneFormer model'
    )
    parser.add_argument(
        '--meta-dir',
        type=str,
        help='Local path to metadata (id2label.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='shi-labs/oneformer_ade20k_swin_large',
        help='HuggingFace model name (fallback)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=None,
        help='Optional resize target (longer edge)'
    )
    parser.add_argument(
        '--save-vis',
        action='store_true',
        help='Save visualization overlays'
    )
    parser.add_argument(
        '--save-label-vis',
        action='store_true',
        help='Save semantic label overlay'
    )
    parser.add_argument(
        '--draw-labels',
        action='store_true',
        default=False,
        help='Draw class labels on visualization'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip images that already have npz output'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing outputs'
    )

    args = parser.parse_args()

    # Validate
    if not args.image_list and not args.image_dir:
        parser.error("Must specify either --image-list or --image-dir")

    if args.overwrite:
        args.skip_existing = False

    # Run export
    summary = export_oneformer_dir(
        image_list=args.image_list,
        image_dir=args.image_dir,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        meta_dir=args.meta_dir,
        device=args.device,
        model_name=args.model_name,
        target_size=args.target_size,
        save_vis=args.save_vis,
        save_label_vis=args.save_label_vis,
        draw_labels=args.draw_labels,
        skip_existing=args.skip_existing,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)