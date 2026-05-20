#!/usr/bin/env python3
"""
Visualize cross-semantic matches on original MegaDepth images.

Generates two types of images per pair:
  1. *_matches.png  — Original images with red lines for cross-semantic matches
  2. *_semseg.png   — Original images with OneFormer segmentation overlay

Supports per-scene sampling to avoid concentrating on a single scene.

Usage:
    python scripts/visualize_semantic_cross_matches.py \
        --data_cfg configs/data/megadepth_test_1500.py \
        --main_cfg configs/loftr/comatch_full.py \
        --ckpt_path weights/comatch_outdoor.ckpt \
        --semantic_cache_dir outputs/semantic_cache \
        --semantic_matches_jsonl outputs/comatch_outdoor/semantic_matches.jsonl \
        --output_dir outputs/semantic_vis \
        --top_n 20 \
        --max_per_scene 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import (
    SemanticLabelCache,
    normalize_semantic_path_key,
)

# ADE20K colormap (150 classes) — deterministic colors for each label id
# Generated with a fixed seed for reproducibility
def _generate_ade20k_colormap(n=150):
    """Generate a deterministic colormap for ADE20K labels."""
    np.random.seed(42)
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        # Use HSV with evenly spaced hues for good visual separation
        hue = int(180 * i / n)
        sat = 200 + (i % 2) * 55  # alternate saturation
        val = 180 + (i % 3) * 25
        hsv = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cmap[i] = bgr[0, 0]
    np.random.seed(None)
    return cmap

ADE20K_COLORMAP = _generate_ade20k_colormap(150)


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
    parser.add_argument("--top_n", type=int, default=20, help="Total pairs to visualize")
    parser.add_argument("--max_per_scene", type=int, default=3,
                        help="Max pairs per scene (0 = unlimited)")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--semantic_ignore_labels", type=str, default="255,-1")
    parser.add_argument("--overlay_alpha", type=float, default=0.4,
                        help="Transparency for segmentation overlay (0=invisible, 1=opaque)")
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


def extract_scene_id(image_path):
    """Extract scene ID from image path like 'Undistorted_SfM/0022/images/...' -> '0022'."""
    parts = image_path.replace('\\', '/').split('/')
    for i, p in enumerate(parts):
        if p == 'Undistorted_SfM' and i + 1 < len(parts):
            return parts[i + 1]
    return 'unknown'


def select_diverse_pairs(stats, top_n, max_per_scene):
    """Select top pairs with per-scene diversity."""
    if max_per_scene <= 0:
        return stats[:top_n]

    scene_counts = defaultdict(int)
    selected = []
    for s in stats:
        if s.get("skipped", False):
            continue
        scene = extract_scene_id(s.get("image0", ""))
        if scene_counts[scene] < max_per_scene:
            selected.append(s)
            scene_counts[scene] += 1
            if len(selected) >= top_n:
                break
    return selected


def label_to_color(label_map, colormap):
    """Convert HxW label map to HxWx3 color image."""
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label_id in range(len(colormap)):
        mask = label_map == label_id
        if mask.any():
            color[mask] = colormap[label_id]
    return color


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


def make_info_bar(canvas_w, texts, bar_h=50):
    """Create an info bar with text labels."""
    bar = np.zeros((bar_h, canvas_w, 3), dtype=np.uint8)
    for i, txt in enumerate(texts):
        cv2.putText(
            bar, txt,
            (10 + i * 600, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return bar


def draw_matches_vis(img0_rgb, img1_rgb, mkpts0, mkpts1, sem0, sem1,
                     ignore_labels, pair_info, output_path):
    """Draw cross-semantic matches as red lines on original images."""
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    max_h = max(h0, h1)
    max_w = max(w0, w1)

    img0_padded, (off_x0, off_y0) = pad_image_to_size(img0_rgb, max_h, max_w)
    img1_padded, (off_x1, off_y1) = pad_image_to_size(img1_rgb, max_h, max_w)

    gap = 4
    canvas_w = max_w * 2 + gap
    canvas = np.zeros((max_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :max_w] = img0_padded
    canvas[:, max_w + gap:] = img1_padded
    canvas[:, max_w:max_w + gap] = 64

    cross_pts0, cross_pts1 = [], []
    total_valid = 0

    if len(mkpts0) > 0:
        x0 = np.rint(mkpts0[:, 0]).astype(int)
        y0 = np.rint(mkpts0[:, 1]).astype(int)
        x1 = np.rint(mkpts1[:, 0]).astype(int)
        y1 = np.rint(mkpts1[:, 1]).astype(int)

        for i in range(len(mkpts0)):
            if not (0 <= x0[i] < w0 and 0 <= y0[i] < h0):
                continue
            if not (0 <= x1[i] < w1 and 0 <= y1[i] < h1):
                continue

            l0 = sem0[y0[i], x0[i]]
            l1 = sem1[y1[i], x1[i]]

            if ignore_labels and (l0 in ignore_labels or l1 in ignore_labels):
                continue

            total_valid += 1
            if l0 != l1:
                cross_pts0.append((x0[i], y0[i]))
                cross_pts1.append((x1[i], y1[i]))

    # Draw lines and dots
    for (px0, py0), (px1, py1) in zip(cross_pts0, cross_pts1):
        cx0 = px0 + off_x0
        cy0 = py0 + off_y0
        cx1 = px1 + off_x1 + max_w + gap
        cy1 = py1 + off_y1
        cv2.line(canvas, (cx0, cy0), (cx1, cy1), (255, 60, 60), 1, cv2.LINE_AA)

    for (px0, py0) in cross_pts0:
        cv2.circle(canvas, (px0 + off_x0, py0 + off_y0), 2, (255, 60, 60), -1, cv2.LINE_AA)
    for (px1, py1) in cross_pts1:
        cv2.circle(canvas, (px1 + off_x1 + max_w + gap, py1 + off_y1), 2, (255, 60, 60), -1, cv2.LINE_AA)

    # Info bar
    cross_rate = pair_info.get("cross_semantic_rate", 0)
    rate_str = f"{cross_rate:.4f}" if isinstance(cross_rate, float) else str(cross_rate)
    texts = [
        f"Pair: {pair_info.get('pair_id', '?')}",
        f"Cross: {len(cross_pts0)}/{total_valid} ({rate_str})",
        f"Size: {w0}x{h0} | {w1}x{h1}",
    ]
    info_bar = make_info_bar(canvas_w, texts)
    result = np.vstack([canvas, info_bar])

    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return len(cross_pts0), total_valid


def draw_semseg_vis(img0_rgb, img1_rgb, sem0, sem1, mkpts0, mkpts1,
                    sem0_color, sem1_color, ignore_labels, pair_info,
                    output_path, overlay_alpha):
    """Draw OneFormer segmentation overlay with cross-semantic match points highlighted."""
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    # Blend original image with segmentation color
    overlay0 = cv2.addWeighted(img0_rgb, 1 - overlay_alpha, sem0_color[:h0, :w0], overlay_alpha, 0)
    overlay1 = cv2.addWeighted(img1_rgb, 1 - overlay_alpha, sem1_color[:h1, :w1], overlay_alpha, 0)

    max_h = max(h0, h1)
    max_w = max(w0, w1)

    img0_pad, (off_x0, off_y0) = pad_image_to_size(overlay0, max_h, max_w)
    img1_pad, (off_x1, off_y1) = pad_image_to_size(overlay1, max_h, max_w)

    gap = 4
    canvas_w = max_w * 2 + gap
    canvas = np.zeros((max_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :max_w] = img0_pad
    canvas[:, max_w + gap:] = img1_pad
    canvas[:, max_w:max_w + gap] = 64

    # Draw all matches: green=same, red=cross, yellow=ignored
    if len(mkpts0) > 0:
        x0 = np.rint(mkpts0[:, 0]).astype(int)
        y0 = np.rint(mkpts0[:, 1]).astype(int)
        x1 = np.rint(mkpts1[:, 0]).astype(int)
        y1 = np.rint(mkpts1[:, 1]).astype(int)

        for i in range(len(mkpts0)):
            in0 = 0 <= x0[i] < w0 and 0 <= y0[i] < h0
            in1 = 0 <= x1[i] < w1 and 0 <= y1[i] < h1
            if not (in0 and in1):
                continue

            l0 = sem0[y0[i], x0[i]]
            l1 = sem1[y1[i], x1[i]]

            if ignore_labels and (l0 in ignore_labels or l1 in ignore_labels):
                color = (200, 200, 60)  # yellow for ignored
            elif l0 == l1:
                color = (60, 220, 60)   # green for same
            else:
                color = (255, 60, 60)   # red for cross

            # Draw on left image
            cx0 = x0[i] + off_x0
            cy0 = y0[i] + off_y0
            cv2.circle(canvas, (cx0, cy0), 2, color, -1, cv2.LINE_AA)

            # Draw on right image
            cx1 = x1[i] + off_x1 + max_w + gap
            cy1 = y1[i] + off_y1
            cv2.circle(canvas, (cx1, cy1), 2, color, -1, cv2.LINE_AA)

    # Info bar with legend
    cross_rate = pair_info.get("cross_semantic_rate", 0)
    rate_str = f"{cross_rate:.4f}" if isinstance(cross_rate, float) else str(cross_rate)
    texts = [
        f"SemSeg: {pair_info.get('pair_id', '?')}",
        f"Rate: {rate_str}",
        f"Green=Same  Red=Cross  Yellow=Ignored",
    ]
    info_bar = make_info_bar(canvas_w, texts)
    result = np.vstack([canvas, info_bar])

    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()

    # Load per-pair stats
    print(f"Loading semantic matches from: {args.semantic_matches_jsonl}")
    all_stats = load_jsonl(args.semantic_matches_jsonl)
    print(f"Total pairs: {len(all_stats)}")

    # Filter skipped, sort by cross_semantic_rate descending
    valid_stats = [s for s in all_stats if not s.get("skipped", False)]
    valid_stats.sort(key=lambda x: x.get("cross_semantic_rate", 0), reverse=True)

    # Select diverse pairs across scenes
    top_pairs = select_diverse_pairs(valid_stats, args.top_n, args.max_per_scene)

    # Print scene distribution
    scene_counts = defaultdict(int)
    for s in top_pairs:
        scene_counts[extract_scene_id(s.get("image0", ""))] += 1
    print(f"\nSelected {len(top_pairs)} pairs from {len(scene_counts)} scenes:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"  Scene {scene}: {count} pairs")
    print()
    for i, s in enumerate(top_pairs):
        scene = extract_scene_id(s.get("image0", ""))
        print(
            f"  {i + 1}. [{scene}] {s.get('pair_id', '?')}: "
            f"rate={s.get('cross_semantic_rate', 0):.4f}, "
            f"cross={s.get('num_cross_semantic', 0)}/{s.get('num_valid_semantic', 0)}"
        )

    # Load config
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg)
    config.merge_from_file(args.data_cfg)
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
                data_root, npz_path,
                mode="test", min_overlap_score=0.0,
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

    # Load id2label from metadata if available
    id2label = {}
    metadata_path = Path(args.semantic_cache_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        id2label = metadata.get("id2label", {})
        print(f"Loaded id2label with {len(id2label)} entries")

    ignore_labels = set()
    if args.semantic_ignore_labels:
        ignore_labels = set(int(x) for x in args.semantic_ignore_labels.split(","))

    # Output dirs
    output_dir = Path(args.output_dir)
    matches_dir = output_dir / "matches"
    semseg_dir = output_dir / "semseg"
    matches_dir.mkdir(parents=True, exist_ok=True)
    semseg_dir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    print(f"\nVisualizing {len(top_pairs)} pairs...")
    success = 0

    for i, pair_stat in enumerate(top_pairs):
        name0 = pair_stat.get("image0", "")
        name1 = pair_stat.get("image1", "")
        pair_id = pair_stat.get("pair_id", f"pair_{i}")

        if not name0 or not name1:
            print(f"  [{i + 1}] Skipped: missing image paths")
            continue

        ds, local_idx = find_pair_in_dataset(datasets, name0, name1)
        if ds is None:
            print(f"  [{i + 1}] Skipped: pair not found: {name0} / {name1}")
            continue

        # Prepare batch
        sample = ds[local_idx]
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

        mkpts0 = batch["mkpts0_f"].cpu().numpy()
        mkpts1 = batch["mkpts1_f"].cpu().numpy()

        # Load images
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

        # Generate segmentation color overlays
        max_label = max(int(sem0.max()), int(sem1.max())) + 1
        colormap = ADE20K_COLORMAP
        if max_label > len(colormap):
            colormap = _generate_ade20k_colormap(max_label)

        sem0_color = label_to_color(sem0, colormap)
        sem1_color = label_to_color(sem1, colormap)

        base_name = f"{i + 1:02d}_{pair_id}"

        # 1. Matches visualization (red lines for cross-semantic)
        matches_path = matches_dir / f"{base_name}_matches.png"
        n_cross, n_valid = draw_matches_vis(
            img0_rgb, img1_rgb, mkpts0, mkpts1, sem0, sem1,
            ignore_labels, pair_stat, matches_path,
        )

        # 2. Segmentation overlay visualization
        semseg_path = semseg_dir / f"{base_name}_semseg.png"
        draw_semseg_vis(
            img0_rgb, img1_rgb, sem0, sem1, mkpts0, mkpts1,
            sem0_color, sem1_color, ignore_labels, pair_stat,
            semseg_path, args.overlay_alpha,
        )

        print(
            f"  [{i + 1}] {pair_id}: {n_cross} cross/{n_valid} valid "
            f"-> {base_name}_matches.png, {base_name}_semseg.png"
        )
        success += 1

    # Write index
    index_path = output_dir / "index.json"
    index = {
        "total_selected": len(top_pairs),
        "total_visualized": success,
        "scene_distribution": dict(scene_counts),
        "overlay_alpha": args.overlay_alpha,
        "ignore_labels": list(ignore_labels),
        "pairs": [
            {
                "idx": i + 1,
                "pair_id": s.get("pair_id", "?"),
                "scene": extract_scene_id(s.get("image0", "")),
                "cross_semantic_rate": s.get("cross_semantic_rate", 0),
                "num_cross_semantic": s.get("num_cross_semantic", 0),
                "num_valid_semantic": s.get("num_valid_semantic", 0),
                "matches_file": f"matches/{i + 1:02d}_{s.get('pair_id', '?')}_matches.png",
                "semseg_file": f"semseg/{i + 1:02d}_{s.get('pair_id', '?')}_semseg.png",
            }
            for i, s in enumerate(top_pairs)
        ],
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, default=_convert_numpy)

    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(top_pairs)} pairs visualized")
    print(f"Output:")
    print(f"  Matches (red lines): {matches_dir}/")
    print(f"  SemSeg  (overlay):   {semseg_dir}/")
    print(f"  Index:               {index_path}")
    print(f"{'=' * 60}")


def _convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
