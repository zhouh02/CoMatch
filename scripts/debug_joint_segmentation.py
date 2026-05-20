#!/usr/bin/env python3
"""
Compare per-image vs joint-pair OneFormer segmentation.

For each pair, generates:
  1. *_separate.png — Two images segmented independently (current approach)
  2. *_joint.png    — Two images concatenated, segmented as one, then split back

This reveals whether "cross-semantic" matches are actually segmentation
inconsistency artifacts.

Usage:
    python scripts/debug_joint_segmentation.py \
        --semantic_cache_dir outputs/semantic_cache \
        --model_id /ssd-data3/zh2025/HFModel/oneformer_ade20k_swin_large \
        --data_root data/megadepth/test \
        --output_dir outputs/segmentation_compare \
        --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import normalize_semantic_path_key


# The 5 pairs to compare
PAIRS = [
    {
        "pair_id": "pair1_5025_1469",
        "image0": "Undistorted_SfM/0022/images/50254895_ce589a7280_o.jpg",
        "image1": "Undistorted_SfM/0022/images/1469603948_0052cdbe5d_o.jpg",
        "cross_rate": 0.4820,
    },
    {
        "pair_id": "pair2_4418_3910",
        "image0": "Undistorted_SfM/0022/images/441847585_4066c9b34b_o.jpg",
        "image1": "Undistorted_SfM/0022/images/391017899_2a67c43613_o.jpg",
        "cross_rate": 0.4154,
    },
    {
        "pair_id": "pair3_4512_4121",
        "image0": "Undistorted_SfM/0022/images/451231185_24b5b1b13c_o.jpg",
        "image1": "Undistorted_SfM/0022/images/412171384_3a7a94fa71_o.jpg",
        "cross_rate": 0.2826,
    },
    {
        "pair_id": "pair4_2696_3103",
        "image0": "Undistorted_SfM/0015/images/2696052764_9a2716f136_o.jpg",
        "image1": "Undistorted_SfM/0015/images/3103619104_4b7b5d04de_o.jpg",
        "cross_rate": 0.2166,
    },
    {
        "pair_id": "pair5_2696_2761",
        "image0": "Undistorted_SfM/0015/images/2696052764_9a2716f136_o.jpg",
        "image1": "Undistorted_SfM/0015/images/2761597789_e2faa53b_o.jpg",
        "cross_rate": 0.2139,
    },
]


def generate_colormap(n=200):
    """Generate deterministic colormap."""
    np.random.seed(42)
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        hue = int(180 * i / n)
        sat = 180 + (i % 3) * 25
        val = 180 + (i % 2) * 40
        hsv = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cmap[i] = bgr[0, 0]
    np.random.seed(None)
    return cmap


def label_to_color(label_map, colormap):
    """Convert label map to RGB color image."""
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label_id in np.unique(label_map):
        if 0 <= label_id < len(colormap):
            color[label_map == label_id] = colormap[label_id]
    return color


def run_oneformer(image_rgb, processor, model, device):
    """Run OneFormer on a PIL RGB image, return label map."""
    pil_img = Image.fromarray(image_rgb)
    width, height = pil_img.size

    inputs = processor(
        images=pil_img,
        task_inputs=["semantic"],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(height, width)]
    )[0]

    label_map = seg.cpu().numpy()
    assert label_map.shape == (height, width), \
        f"Shape mismatch: {label_map.shape} vs ({height}, {width})"
    return label_map


def make_overlay(img_rgb, label_map, colormap, alpha=0.45):
    """Blend original image with segmentation colors."""
    color_map = label_to_color(label_map, colormap)
    return cv2.addWeighted(img_rgb, 1 - alpha, color_map, alpha, 0)


def pad_to_same_size(img0, img1):
    """Pad both images to the same size (max of both dims), center-aligned."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    max_h = max(h0, h1)
    max_w = max(w0, w1)

    canvas0 = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas1 = np.zeros((max_h, max_w, 3), dtype=np.uint8)

    canvas0[(max_h - h0) // 2 : (max_h - h0) // 2 + h0,
             (max_w - w0) // 2 : (max_w - w0) // 2 + w0] = img0
    canvas1[(max_h - h1) // 2 : (max_h - h1) // 2 + h1,
             (max_w - w1) // 2 : (max_w - w1) // 2 + w1] = img1

    return canvas0, canvas1, max_h, max_w


def compute_cross_stats(mkpts0, mkpts1, sem0, sem1, h0, w0, h1, w1, ignore_labels):
    """Compute cross-semantic stats for a pair."""
    cross = 0
    valid = 0
    same = 0
    cross_labels = defaultdict(int)

    if len(mkpts0) == 0:
        return 0, 0, 0, {}

    x0 = np.rint(mkpts0[:, 0]).astype(int)
    y0 = np.rint(mkpts0[:, 1]).astype(int)
    x1 = np.rint(mkpts1[:, 0]).astype(int)
    y1 = np.rint(mkpts1[:, 1]).astype(int)

    for i in range(len(mkpts0)):
        if not (0 <= x0[i] < w0 and 0 <= y0[i] < h0):
            continue
        if not (0 <= x1[i] < w1 and 0 <= y1[i] < h1):
            continue

        l0 = int(sem0[y0[i], x0[i]])
        l1 = int(sem1[y1[i], x1[i]])

        if ignore_labels and (l0 in ignore_labels or l1 in ignore_labels):
            continue

        valid += 1
        if l0 == l1:
            same += 1
        else:
            cross += 1
            cross_labels[f"{l0}->{l1}"] += 1

    return cross, same, valid, dict(cross_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_cache_dir", type=str, default="outputs/semantic_cache")
    parser.add_argument("--model_id", type=str,
                        default="/ssd-data3/zh2025/HFModel/oneformer_ade20k_swin_large")
    parser.add_argument("--data_root", type=str, default="data/megadepth/test")
    parser.add_argument("--output_dir", type=str, default="outputs/segmentation_compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    args = parser.parse_args()

    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from src.utils.semantic_consistency import SemanticLabelCache

    # Load model
    print("Loading OneFormer...")
    processor = OneFormerProcessor.from_pretrained(args.model_id, local_files_only=True)
    model = OneFormerForUniversalSegmentation.from_pretrained(args.model_id, local_files_only=True)
    model = model.to(args.device).eval()

    # Load existing cache for separate-inference comparison
    print("Loading semantic cache...")
    sem_cache = SemanticLabelCache(args.semantic_cache_dir)

    colormap = generate_colormap(200)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CoMatch for matching
    print("Loading CoMatch...")
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config
    from src.loftr import LoFTR

    config = get_cfg_defaults()
    config.merge_from_file("configs/loftr/comatch_full.py")
    config.merge_from_file("configs/data/megadepth_test_1500.py")
    config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE]
    _config = lower_config(config)
    loftr_cfg = lower_config(_config["loftr"])

    # Find checkpoint
    ckpt_path = "weights/comatch_outdoor.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}, skipping match analysis")
        matcher = None
    else:
        matcher = LoFTR(config=loftr_cfg)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        matcher.load_state_dict(state_dict, strict=True)
        matcher = matcher.to(args.device).eval()

    # Summary table
    summary = []

    for pair_info in PAIRS:
        pair_id = pair_info["pair_id"]
        rel0 = pair_info["image0"]
        rel1 = pair_info["image1"]

        abs0 = os.path.join(args.data_root, rel0)
        abs1 = os.path.join(args.data_root, rel1)

        print(f"\n{'='*70}")
        print(f"Processing: {pair_id}")
        print(f"  image0: {rel0}")
        print(f"  image1: {rel1}")

        # Load original images
        img0_bgr = cv2.imread(abs0)
        img1_bgr = cv2.imread(abs1)
        if img0_bgr is None or img1_bgr is None:
            print("  ERROR: Cannot load images")
            continue

        img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img0_rgb.shape[:2]
        h1, w1 = img1_rgb.shape[:2]
        print(f"  sizes: {w0}x{h0} | {w1}x{h1}")

        # ---- Method A: Separate inference (current) ----
        print("  Running separate inference...")
        sem0_sep = run_oneformer(img0_rgb, processor, model, args.device)
        sem1_sep = run_oneformer(img1_rgb, processor, model, args.device)

        # Load cached labels for comparison
        try:
            sem0_cache = sem_cache.get_label(rel0)
            sem1_cache = sem_cache.get_label(rel1)
            cache_available = True
        except Exception:
            sem0_cache = sem0_sep
            sem1_cache = sem1_sep
            cache_available = False

        # ---- Method B: Joint inference (concatenate horizontally) ----
        print("  Running joint inference (horizontal concat)...")
        # Pad to same height, then concatenate horizontally
        max_h = max(h0, h1)
        pad0 = np.zeros((max_h, w0, 3), dtype=np.uint8)
        pad1 = np.zeros((max_h, w1, 3), dtype=np.uint8)
        pad0[:h0, :w0] = img0_rgb
        pad1[:h1, :w1] = img1_rgb

        joint_img = np.concatenate([pad0, pad1], axis=1)  # (max_h, w0+w1, 3)
        joint_label = run_oneformer(joint_img, processor, model, args.device)

        # Split back
        sem0_joint = joint_label[:, :w0][:h0]
        sem1_joint = joint_label[:, w0:][:h1]

        # ---- Run CoMatch matching for match coordinates ----
        mkpts0 = mkpts1 = None
        if matcher is not None:
            from src.datasets.megadepth import MegaDepthDataset
            from src.utils.dataset import read_megadepth_gray

            # Load as CoMatch does (resize + pad)
            image0, mask0, scale0 = read_megadepth_gray(abs0, 832, 8, True, None)
            image1, mask1, scale1 = read_megadepth_gray(abs1, 832, 8, True, None)

            batch = {
                'image0': image0[None],  # (1,1,H,W)
                'image1': image1[None],
                'scale0': scale0[None],
                'scale1': scale1[None],
            }

            with torch.no_grad():
                with torch.autocast(enabled=False, device_type="cuda"):
                    matcher(batch)

            mkpts0 = batch["mkpts0_f"].cpu().numpy()  # original coords
            mkpts1 = batch["mkpts1_f"].cpu().numpy()
            print(f"  Matches: {len(mkpts0)}")

        # ---- Compute cross-semantic stats for both methods ----
        ignore_labels = {255, -1}

        if mkpts0 is not None and len(mkpts0) > 0:
            cross_sep, same_sep, valid_sep, labels_sep = compute_cross_stats(
                mkpts0, mkpts1, sem0_cache, sem1_cache, h0, w0, h1, w1, ignore_labels)
            cross_joint, same_joint, valid_joint, labels_joint = compute_cross_stats(
                mkpts0, mkpts1, sem0_joint, sem1_joint, h0, w0, h1, w1, ignore_labels)

            rate_sep = cross_sep / valid_sep if valid_sep > 0 else float("nan")
            rate_joint = cross_joint / valid_joint if valid_joint > 0 else float("nan")

            print(f"  Separate: {cross_sep}/{valid_sep} cross ({rate_sep:.4f})")
            print(f"  Joint:    {cross_joint}/{valid_joint} cross ({rate_joint:.4f})")
            print(f"  Top cross labels (separate): {dict(sorted(labels_sep.items(), key=lambda x:-x[1])[:5])}")
            print(f"  Top cross labels (joint):    {dict(sorted(labels_joint.items(), key=lambda x:-x[1])[:5])}")

            summary.append({
                "pair_id": pair_id,
                "num_matches": len(mkpts0),
                "separate": {"cross": cross_sep, "valid": valid_sep, "rate": rate_sep, "top_labels": dict(sorted(labels_sep.items(), key=lambda x:-x[1])[:5])},
                "joint": {"cross": cross_joint, "valid": valid_joint, "rate": rate_joint, "top_labels": dict(sorted(labels_joint.items(), key=lambda x:-x[1])[:5])},
            })

        # ---- Generate visualizations ----
        # 1. Separate segmentation overlay
        ov0_sep = make_overlay(img0_rgb, sem0_sep, colormap, args.overlay_alpha)
        ov1_sep = make_overlay(img1_rgb, sem1_sep, colormap, args.overlay_alpha)

        # 2. Joint segmentation overlay
        ov0_joint = make_overlay(img0_rgb, sem0_joint, colormap, args.overlay_alpha)
        ov1_joint = make_overlay(img1_rgb, sem1_joint, colormap, args.overlay_alpha)

        # 3. Difference map (where labels differ between methods)
        diff0 = np.where(sem0_sep != sem0_joint, 255, 0).astype(np.uint8)
        diff1 = np.where(sem1_sep != sem1_joint, 255, 0).astype(np.uint8)
        diff0_color = np.zeros((*diff0.shape, 3), dtype=np.uint8)
        diff1_color = np.zeros((*diff1.shape, 3), dtype=np.uint8)
        diff0_color[diff0 > 0] = (255, 255, 0)  # Yellow = label changed
        diff1_color[diff1 > 0] = (255, 255, 0)
        diff0_vis = cv2.addWeighted(img0_rgb, 0.7, diff0_color, 0.3, 0)
        diff1_vis = cv2.addWeighted(img1_rgb, 0.7, diff1_color, 0.3, 0)

        # Helper: create side-by-side with info text
        def save_comparison(ov0, ov1, label, suffix):
            gap = 4
            max_h_loc = max(ov0.shape[0], ov1.shape[0])
            # Pad to same height
            p0 = np.zeros((max_h_loc, ov0.shape[1], 3), dtype=np.uint8)
            p1 = np.zeros((max_h_loc, ov1.shape[1], 3), dtype=np.uint8)
            p0[:ov0.shape[0]] = ov0
            p1[:ov1.shape[0]] = ov1
            combined = np.concatenate([p0, np.full((max_h_loc, gap, 3), 64, dtype=np.uint8), p1], axis=1)

            # Info bar
            bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, f"{pair_id} | {label} | {w0}x{h0} | {w1}x{h1}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            result = np.vstack([combined, bar])
            cv2.imwrite(str(output_dir / f"{pair_id}_{suffix}.png"),
                        cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        save_comparison(ov0_sep, ov1_sep, "Separate inference", "separate")
        save_comparison(ov0_joint, ov1_joint, "Joint inference (concat)", "joint")
        save_comparison(diff0_vis, diff1_vis, "Label diff (yellow=changed)", "diff")

        print(f"  Saved: {pair_id}_separate.png, {pair_id}_joint.png, {pair_id}_diff.png")

    # Save summary
    summary_path = output_dir / "comparison_summary.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Pair':<35} {'Separate Rate':>15} {'Joint Rate':>12} {'Reduction':>10}")
    print("-" * 70)
    for s in summary:
        sep = s["separate"]["rate"]
        joint = s["joint"]["rate"]
        reduction = (sep - joint) / sep * 100 if sep > 0 else 0
        print(f"{s['pair_id']:<35} {sep:>15.4f} {joint:>12.4f} {reduction:>9.1f}%")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
