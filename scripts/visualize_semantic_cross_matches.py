#!/usr/bin/env python3
"""
Visualize cross-semantic matches on original MegaDepth images.

Reads the per-pair semantic stats JSONL, picks the top-N pairs by cross_semantic_rate,
re-runs CoMatch matching on those pairs, and draws red lines for cross-semantic matches
on original-resolution images.

Usage:
    python scripts/visualize_semantic_cross_matches.py \
        --data_cfg configs/data/megadepth_test_1500.py \
        --main_cfg configs/loftr/comatch_full.py \
        --ckpt_path weights/comatch_outdoor.ckpt \
        --semantic_cache_dir outputs/semantic_cache \
        --semantic_matches_jsonl outputs/comatch_outdoor/semantic_matches.jsonl \
        --output_dir outputs/semantic_vis \
        --top_n 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import (
    SemanticLabelCache,
    compute_semantic_match_stats,
    normalize_semantic_path_key,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cross-semantic matches on original MegaDepth images"
    )
    parser.add_argument("--data_cfg", type=str, required=True)
    parser.add_argument("--main_cfg", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--semantic_cache_dir", type=str, required=True)
    parser.add_argument("--semantic_matches_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/semantic_vis")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--semantic_ignore_labels", type=str, default="255,-1")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_original_image(image_path):
    """Load original RGB image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_image_to_size(img, target_h, target_w, pad_color=(0, 0, 0)):
    """Pad image to target size, centering it."""
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y_off = (target_h - h) // 2
    x_off = (target_w - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img
    return canvas, (x_off, y_off)


def find_pair_in_dataset(datasets, target_name0, target_name1):
    """Find a specific pair in the dataset by pair_names."""
    target0 = normalize_semantic_path_key(target_name0)
    target1 = normalize_semantic_path_key(target_name1)

    for ds in datasets:
        for local_idx in range(len(ds)):
            pair_info = ds.pair_infos[local_idx]
            idx0, idx1 = pair_info[0]
            ip = ds.scene_info["image_paths"]
            p0 = normalize_semantic_path_key(str(ip[idx0]))
            p1 = normalize_semantic_path_key(str(ip[idx1]))

            if (p0 == target0 and p1 == target1) or (p0 == target1 and p1 == target0):
                return ds, local_idx
    return None, None


def draw_cross_semantic_vis(
    img0_rgb, img1_rgb,
    mkpts0, mkpts1, sem0, sem1,
    ignore_labels, pair_info, output_path,
):
    """
    Draw cross-semantic matches as red lines on original images.

    Args:
        img0_rgb, img1_rgb: Original RGB images (H x W x 3)
        mkpts0, mkpts1: Nx2 match coordinates in original image space [x, y]
        sem0, sem1: HxW semantic label maps
        ignore_labels: Set of label IDs to ignore
        pair_info: Dict with pair metadata
        output_path: Where to save PNG
    """
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    # Pad images to same size (center-aligned)
    max_h = max(h0, h1)
    max_w = max(w0, w1)

    img0_padded, (off_x0, off_y0) = pad_image_to_size(img0_rgb, max_h, max_w)
    img1_padded, (off_x1, off_y1) = pad_image_to_size(img1_rgb, max_h, max_w)

    # Create side-by-side canvas
    gap = 4  # pixels gap between images
    canvas_h = max_h + 50  # extra space for info bar
    canvas_w = max_w * 2 + gap

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place images
    canvas[0:max_h, 0:max_w] = img0_padded
    canvas[0:max_h, max_w + gap : max_w * 2 + gap] = img1_padded

    # Add gray divider
    canvas[:, max_w : max_w + gap] = 64

    # Find cross-semantic matches
    cross_pts0 = []
    cross_pts1 = []
    total_valid = 0
    total_same = 0

    if len(mkpts0) > 0:
        x0_int = np.rint(mkpts0[:, 0]).astype(int)
        y0_int = np.rint(mkpts0[:, 1]).astype(int)
        x1_int = np.rint(mkpts1[:, 0]).astype(int)
        y1_int = np.rint(mkpts1[:, 1]).astype(int)

        for i in range(len(mkpts0)):
            # Bounds check
            if not (0 <= x0_int[i] < w0 and 0 <= y0_int[i] < h0):
                continue
            if not (0 <= x1_int[i] < w1 and 0 <= y1_int[i] < h1):
                continue

            label0 = sem0[y0_int[i], x0_int[i]]
            label1 = sem1[y1_int[i], x1_int[i]]

            if ignore_labels and (label0 in ignore_labels or label1 in ignore_labels):
                continue

            total_valid += 1
            if label0 != label1:
                cross_pts0.append((x0_int[i], y0_int[i]))
                cross_pts1.append((x1_int[i], y1_int[i]))
            else:
                total_same += 1

    # Draw cross-semantic match lines (red, semi-transparent)
    for (px0, py0), (px1, py1) in zip(cross_pts0, cross_pts1):
        # Map to canvas coordinates
        canvas_x0 = px0 + off_x0
        canvas_y0 = py0 + off_y0
        canvas_x1 = px1 + off_x1 + max_w + gap
        canvas_y1 = py1 + off_y1

        # Draw line
        cv2.line(
            canvas,
            (canvas_x0, canvas_y0),
            (canvas_x1, canvas_y1),
            color=(255, 60, 60),  # Red
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Draw endpoints as small circles
    for (px0, py0) in cross_pts0:
        cx = px0 + off_x0
        cy = py0 + off_y0
        cv2.circle(canvas, (cx, cy), 2, (255, 60, 60), -1, cv2.LINE_AA)

    for (px1, py1) in cross_pts1:
        cx = px1 + off_x1 + max_w + gap
        cy = py1 + off_y1
        cv2.circle(canvas, (cx, cy), 2, (255, 60, 60), -1, cv2.LINE_AA)

    # Info bar
    info_y = max_h + 5
    cross_rate = pair_info.get("cross_semantic_rate", 0)
    if isinstance(cross_rate, float):
        cross_rate_str = f"{cross_rate:.4f}"
    else:
        cross_rate_str = str(cross_rate)

    texts = [
        f"Pair: {pair_info.get('pair_id', 'unknown')}",
        f"Cross-semantic: {len(cross_pts0)}/{total_valid} ({cross_rate_str})",
        f"Image0: {w0}x{h0}  Image1: {w1}x{h1}",
    ]
    for i, txt in enumerate(texts):
        cv2.putText(
            canvas,
            txt,
            (10 + i * 500, info_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Save
    # Convert RGB back to BGR for cv2.imwrite
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), canvas_bgr)

    return len(cross_pts0), total_valid


def main():
    args = parse_args()

    # Load per-pair stats
    print(f"Loading semantic matches from: {args.semantic_matches_jsonl}")
    all_stats = load_jsonl(args.semantic_matches_jsonl)
    print(f"Total pairs: {len(all_stats)}")

    # Filter out skipped pairs and sort by cross_semantic_rate descending
    valid_stats = [s for s in all_stats if not s.get("skipped", False)]
    valid_stats.sort(key=lambda x: x.get("cross_semantic_rate", 0), reverse=True)

    top_pairs = valid_stats[: args.top_n]
    print(f"\nTop {len(top_pairs)} pairs by cross_semantic_rate:")
    for i, s in enumerate(top_pairs):
        print(
            f"  {i + 1}. {s.get('pair_id', '?')}: "
            f"rate={s.get('cross_semantic_rate', 0):.4f}, "
            f"cross={s.get('num_cross_semantic', 0)}/{s.get('num_valid_semantic', 0)}"
        )

    # Load config
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg)
    config.merge_from_file(args.data_cfg)

    # Set eval parameters
    config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE]

    _config = lower_config(config)
    loftr_cfg = lower_config(_config["loftr"])

    # Build datasets
    from src.datasets.megadepth import MegaDepthDataset

    data_root = config.DATASET.TEST_DATA_ROOT
    if args.image_root:
        data_root = args.image_root
    npz_root = config.DATASET.TEST_NPZ_ROOT
    scene_list_path = config.DATASET.TEST_LIST_PATH

    with open(scene_list_path, "r") as f:
        npz_names = [name.split()[0] for name in f.readlines()]

    datasets = []
    for scene_name in tqdm(npz_names, desc="Loading scene datasets"):
        npz_path = os.path.join(npz_root, f"{scene_name}.npz")
        if not os.path.exists(npz_path):
            continue
        datasets.append(
            MegaDepthDataset(
                data_root,
                npz_path,
                mode="test",
                min_overlap_score=0.0,
                img_resize=config.DATASET.MGDPT_IMG_RESIZE,
                df=config.DATASET.MGDPT_DF,
                img_padding=config.DATASET.MGDPT_IMG_PAD,
                depth_padding=config.DATASET.MGDPT_DEPTH_PAD,
                fp16=False,
            )
        )

    # Load matcher
    print("\nLoading CoMatch matcher...")
    from src.loftr import LoFTR

    matcher = LoFTR(config=loftr_cfg)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)
    matcher = matcher.to(args.device)
    matcher.eval()

    # Load semantic cache
    print("Loading semantic cache...")
    sem_cache = SemanticLabelCache(args.semantic_cache_dir)

    ignore_labels = set()
    if args.semantic_ignore_labels:
        ignore_labels = set(int(x) for x in args.semantic_ignore_labels.split(","))

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each top pair
    print(f"\nVisualizing {len(top_pairs)} pairs...")
    success = 0

    for i, pair_stat in enumerate(top_pairs):
        name0 = pair_stat.get("image0", "")
        name1 = pair_stat.get("image1", "")
        pair_id = pair_stat.get("pair_id", f"pair_{i}")

        if not name0 or not name1:
            print(f"  [{i + 1}] Skipped: missing image paths")
            continue

        # Find pair in dataset
        ds, local_idx = find_pair_in_dataset(datasets, name0, name1)
        if ds is None:
            print(f"  [{i + 1}] Skipped: pair not found in dataset: {name0} / {name1}")
            continue

        # Get sample
        sample = ds[local_idx]

        # Prepare batch (add batch dimension)
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(args.device)
            else:
                batch[k] = v

        # Run matcher
        with torch.no_grad():
            with torch.autocast(enabled=False, device_type="cuda"):
                matcher(batch)

        # Get match coordinates (original image space)
        mkpts0 = batch["mkpts0_f"].cpu().numpy()  # Nx2 [x, y]
        mkpts1 = batch["mkpts1_f"].cpu().numpy()

        # Load original RGB images
        abs_path0 = os.path.join(data_root, name0)
        abs_path1 = os.path.join(data_root, name1)
        img0_rgb = load_original_image(abs_path0)
        img1_rgb = load_original_image(abs_path1)

        if img0_rgb is None or img1_rgb is None:
            print(f"  [{i + 1}] Skipped: cannot load images")
            continue

        # Load semantic labels
        try:
            sem0 = sem_cache.get_label(name0)
            sem1 = sem_cache.get_label(name1)
        except Exception as e:
            print(f"  [{i + 1}] Skipped: semantic cache error: {e}")
            continue

        # Draw visualization
        output_path = output_dir / f"{i + 1:02d}_{pair_id}.png"
        n_cross, n_valid = draw_cross_semantic_vis(
            img0_rgb,
            img1_rgb,
            mkpts0,
            mkpts1,
            sem0,
            sem1,
            ignore_labels,
            pair_stat,
            output_path,
        )

        print(
            f"  [{i + 1}] {pair_id}: {n_cross} cross-semantic / {n_valid} valid "
            f"-> {output_path.name}"
        )
        success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(top_pairs)} pairs visualized")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
