#!/usr/bin/env python3
"""
Visualize cross-semantic matches on original ScanNet images.

IMPORTANT - ScanNet coordinate system:
  - ScanNet original images have varying resolutions (e.g. 1296x968, 640x480)
  - read_scannet_gray() resizes to (scannetX, scannetY) = (640, 480)
  - CoMatch then pads/resizes to 832x832 for inference
  - mkpts*_f are mapped back to the resized (640x480) space via scale0/scale1
  - To overlay on original images, we must map: mkpt_orig = mkpt_f * (orig_dim / 480_or_640)
  - OneFormer labels are in original image resolution

Usage:
    python scripts/visualize_scannet_cross_matches.py \
        --data_cfg configs/data/scannet_test_1500.py \
        --main_cfg configs/loftr/comatch_full.py \
        --ckpt_path weights/comatch_outdoor.ckpt \
        --semantic_cache_dir outputs/scannet_semantic_cache \
        --semantic_matches_jsonl outputs/comatch_full_scannet/semantic_matches.jsonl \
        --output_dir outputs/scannet_semantic_vis \
        --top_n 20
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

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


def _generate_ade20k_colormap(n=150):
    np.random.seed(42)
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        hue = int(180 * i / n)
        sat = 200 + (i % 2) * 55
        val = 180 + (i % 3) * 25
        hsv = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cmap[i] = bgr[0, 0]
    np.random.seed(None)
    return cmap


ADE20K_COLORMAP = _generate_ade20k_colormap(150)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cross-semantic matches on original ScanNet images"
    )
    parser.add_argument("--data_cfg", type=str, required=True)
    parser.add_argument("--main_cfg", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--semantic_cache_dir", type=str, required=True)
    parser.add_argument("--semantic_matches_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/scannet_semantic_vis")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--max_per_scene", type=int, default=0,
                        help="Max pairs per scene (0 = unlimited)")
    parser.add_argument("--scannetX", type=int, default=640)
    parser.add_argument("--scannetY", type=int, default=480)
    parser.add_argument("--semantic_ignore_labels", type=str, default="255,-1")
    parser.add_argument("--overlay_alpha", type=float, default=0.4)
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
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_image_to_size(img, target_h, target_w, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y_off = (target_h - h) // 2
    x_off = (target_w - w) // 2
    canvas[y_off: y_off + h, x_off: x_off + w] = img
    return canvas, (x_off, y_off)


def extract_scene_id(image_path):
    parts = image_path.replace("\\", "/").split("/")
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def select_diverse_pairs(stats, top_n, max_per_scene):
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
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label_id in range(len(colormap)):
        mask = label_map == label_id
        if mask.any():
            color[mask] = colormap[label_id]
    return color


def make_info_bar(canvas_w, texts, bar_h=50):
    bar = np.zeros((bar_h, canvas_w, 3), dtype=np.uint8)
    for i, txt in enumerate(texts):
        cv2.putText(
            bar, txt,
            (10 + i * 500, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
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
    same_pts0, same_pts1 = [], []
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
            else:
                same_pts0.append((x0[i], y0[i]))
                same_pts1.append((x1[i], y1[i]))

    # Draw same-semantic matches (green, thinner)
    for (px0, py0), (px1, py1) in zip(same_pts0, same_pts1):
        cx0 = px0 + off_x0
        cy0 = py0 + off_y0
        cx1 = px1 + off_x1 + max_w + gap
        cy1 = py1 + off_y1
        cv2.line(canvas, (cx0, cy0), (cx1, cy1), (60, 200, 60), 1, cv2.LINE_AA)

    # Draw cross-semantic matches (red)
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

    cross_rate = pair_info.get("cross_semantic_rate", 0)
    rate_str = f"{cross_rate:.4f}" if isinstance(cross_rate, float) else str(cross_rate)
    texts = [
        f"Pair: {pair_info.get('pair_id', '?')}",
        f"Cross: {len(cross_pts0)}/{total_valid} ({rate_str})",
        f"Orig: {w0}x{h0} | {w1}x{h1}  Green=Same Red=Cross",
    ]
    info_bar = make_info_bar(canvas_w, texts)
    result = np.vstack([canvas, info_bar])

    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return len(cross_pts0), len(same_pts0), total_valid


def draw_semseg_vis(img0_rgb, img1_rgb, sem0, sem1, mkpts0, mkpts1,
                    sem0_color, sem1_color, ignore_labels, pair_info,
                    output_path, overlay_alpha, id2label):
    """Draw segmentation overlay with match points."""
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

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
                color = (200, 200, 60)
            elif l0 == l1:
                color = (60, 220, 60)
            else:
                color = (255, 60, 60)

            cx0 = x0[i] + off_x0
            cy0 = y0[i] + off_y0
            cv2.circle(canvas, (cx0, cy0), 2, color, -1, cv2.LINE_AA)

            cx1 = x1[i] + off_x1 + max_w + gap
            cy1 = y1[i] + off_y1
            cv2.circle(canvas, (cx1, cy1), 2, color, -1, cv2.LINE_AA)

    cross_rate = pair_info.get("cross_semantic_rate", 0)
    rate_str = f"{cross_rate:.4f}" if isinstance(cross_rate, float) else str(cross_rate)
    texts = [
        f"SemSeg: {pair_info.get('pair_id', '?')}",
        f"Rate: {rate_str}",
        f"Green=Same  Red=Cross",
    ]
    info_bar = make_info_bar(canvas_w, texts)
    result = np.vstack([canvas, info_bar])

    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def find_pair_in_scannet_dataset(datasets, target_name0, target_name1):
    """Find a specific pair in ScanNet datasets by pair_names."""
    target0 = normalize_semantic_path_key(target_name0)
    target1 = normalize_semantic_path_key(target_name1)

    for ds in datasets:
        for local_idx in range(len(ds)):
            data = ds[local_idx]
            p0 = normalize_semantic_path_key(str(data['pair_names'][0]))
            p1 = normalize_semantic_path_key(str(data['pair_names'][1]))

            if (p0 == target0 and p1 == target1) or (p0 == target1 and p1 == target0):
                return ds, local_idx
    return None, None


def main():
    args = parse_args()

    print(f"Loading semantic matches from: {args.semantic_matches_jsonl}")
    all_stats = load_jsonl(args.semantic_matches_jsonl)
    valid_stats = [s for s in all_stats if not s.get("skipped", False)]
    valid_stats.sort(key=lambda x: x.get("cross_semantic_rate", 0), reverse=True)

    top_pairs = select_diverse_pairs(valid_stats, args.top_n, args.max_per_scene)

    scene_counts = defaultdict(int)
    for s in top_pairs:
        scene_counts[extract_scene_id(s.get("image0", ""))] += 1
    print(f"\nSelected {len(top_pairs)} pairs from {len(scene_counts)} scenes:")
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

    _config = lower_config(config)
    loftr_cfg = lower_config(_config["loftr"])

    # Build ScanNet datasets
    from src.datasets.scannet import ScanNetDataset

    data_root = config.DATASET.TEST_DATA_ROOT
    npz_root = config.DATASET.TEST_NPZ_ROOT
    scene_list_path = config.DATASET.TEST_LIST_PATH
    intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

    with open(scene_list_path, "r") as f:
        scene_info_list = [line.strip().split() for line in f.readlines()]

    datasets = []
    for info in tqdm(scene_info_list, desc="Loading scene datasets"):
        scene_npz_name = info[0]
        if not scene_npz_name.endswith(".npz"):
            scene_npz_name = f"{scene_npz_name}.npz"
        npz_path = os.path.join(npz_root, scene_npz_name)
        if not os.path.exists(npz_path):
            continue
        datasets.append(
            ScanNetDataset(
                data_root, npz_path,
                intrinsic_path,
                mode="test", min_overlap_score=0.0,
                img_resize=(args.scannetX, args.scannetY),
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

    id2label = {}
    metadata_path = Path(args.semantic_cache_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        id2label = metadata.get("id2label", {})
        id2label = {int(k): v for k, v in id2label.items()}

    ignore_labels = set()
    if args.semantic_ignore_labels:
        ignore_labels = set(int(x) for x in args.semantic_ignore_labels.split(","))

    # Output dirs
    output_dir = Path(args.output_dir)
    matches_dir = output_dir / "matches"
    semseg_dir = output_dir / "semseg"
    matches_dir.mkdir(parents=True, exist_ok=True)
    semseg_dir.mkdir(parents=True, exist_ok=True)

    # ScanNet resize dimensions (what CoMatch sees)
    scannet_w = args.scannetX  # 640
    scannet_h = args.scannetY  # 480

    print(f"\nScanNet resize: {scannet_w}x{scannet_h}")
    print(f"mkpts*_f are in {scannet_w}x{scannet_h} space, will map to original resolution")
    print(f"\nVisualizing {len(top_pairs)} pairs...")

    success = 0
    vis_index = []

    for i, pair_stat in enumerate(top_pairs):
        name0 = pair_stat.get("image0", "")
        name1 = pair_stat.get("image1", "")
        pair_id = pair_stat.get("pair_id", f"pair_{i}")

        if not name0 or not name1:
            print(f"  [{i + 1}] Skipped: missing image paths")
            continue

        ds, local_idx = find_pair_in_scannet_dataset(datasets, name0, name1)
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

        # mkpts*_f are in resized (scannet_w x scannet_h) space
        mkpts0_resized = batch["mkpts0_f"].cpu().numpy()
        mkpts1_resized = batch["mkpts1_f"].cpu().numpy()

        # Load original images
        abs_path0 = os.path.join(data_root, name0)
        abs_path1 = os.path.join(data_root, name1)
        img0_rgb = load_original_image(abs_path0)
        img1_rgb = load_original_image(abs_path1)

        if img0_rgb is None or img1_rgb is None:
            print(f"  [{i + 1}] Skipped: cannot load images")
            continue

        orig_h0, orig_w0 = img0_rgb.shape[:2]
        orig_h1, orig_w1 = img1_rgb.shape[:2]

        # Map mkpts from resized space to original image space
        # x_orig = x_mkpt * (orig_w / scannet_w)
        # y_orig = y_mkpt * (orig_h / scannet_h)
        scale_x0 = orig_w0 / scannet_w
        scale_y0 = orig_h0 / scannet_h
        scale_x1 = orig_w1 / scannet_w
        scale_y1 = orig_h1 / scannet_h

        mkpts0 = mkpts0_resized.copy()
        mkpts0[:, 0] *= scale_x0
        mkpts0[:, 1] *= scale_y0

        mkpts1 = mkpts1_resized.copy()
        mkpts1[:, 0] *= scale_x1
        mkpts1[:, 1] *= scale_y1

        # Load semantic labels (already in original resolution)
        try:
            sem0 = sem_cache.get_label(name0)
            sem1 = sem_cache.get_label(name1)
        except Exception as e:
            print(f"  [{i + 1}] Skipped: semantic cache error: {e}")
            continue

        # Sanity check
        assert sem0.shape == (orig_h0, orig_w0), \
            f"sem0 shape {sem0.shape} != img0 shape ({orig_h0}, {orig_w0})"
        assert sem1.shape == (orig_h1, orig_w1), \
            f"sem1 shape {sem1.shape} != img1 shape ({orig_h1}, {orig_w1})"

        # Generate segmentation color overlays
        max_label = max(int(sem0.max()), int(sem1.max())) + 1
        colormap = ADE20K_COLORMAP
        if max_label > len(colormap):
            colormap = _generate_ade20k_colormap(max_label)

        sem0_color = label_to_color(sem0, colormap)
        sem1_color = label_to_color(sem1, colormap)

        # Make pair_id filesystem-safe
        safe_pair_id = pair_id.replace("/", "_").replace("\\", "_")
        base_name = f"{i + 1:02d}_{safe_pair_id}"

        # 1. Matches visualization
        matches_path = matches_dir / f"{base_name}_matches.png"
        n_cross, n_same, n_valid = draw_matches_vis(
            img0_rgb, img1_rgb, mkpts0, mkpts1, sem0, sem1,
            ignore_labels, pair_stat, matches_path,
        )

        # 2. Segmentation overlay visualization
        semseg_path = semseg_dir / f"{base_name}_semseg.png"
        draw_semseg_vis(
            img0_rgb, img1_rgb, sem0, sem1, mkpts0, mkpts1,
            sem0_color, sem1_color, ignore_labels, pair_stat,
            semseg_path, args.overlay_alpha, id2label,
        )

        scene = extract_scene_id(name0)
        print(
            f"  [{i + 1}] [{scene}] {Path(name0).stem}<->{Path(name1).stem}: "
            f"cross={n_cross}/{n_valid} same={n_same} "
            f"orig={orig_w0}x{orig_h0}|{orig_w1}x{orig_h1} "
            f"-> {base_name}_matches.png, {base_name}_semseg.png"
        )

        vis_index.append({
            "idx": i + 1,
            "pair_id": pair_id,
            "scene": scene,
            "cross_semantic_rate": pair_stat.get("cross_semantic_rate", 0),
            "n_cross": n_cross,
            "n_same": n_same,
            "n_valid": n_valid,
            "orig_size_0": f"{orig_w0}x{orig_h0}",
            "orig_size_1": f"{orig_w1}x{orig_h1}",
            "resized_space": f"{scannet_w}x{scannet_h}",
            "matches_file": f"matches/{base_name}_matches.png",
            "semseg_file": f"semseg/{base_name}_semseg.png",
        })
        success += 1

    # Write index
    index_path = output_dir / "index.json"
    index = {
        "total_selected": len(top_pairs),
        "total_visualized": success,
        "scene_distribution": dict(scene_counts),
        "coordinate_info": {
            "mkpts_space": f"{scannet_w}x{scannet_h} (ScanNet resized)",
            "mapped_to": "original image resolution",
            "mapping": "x_orig = x_mkpt * (orig_w / scannet_w), y_orig = y_mkpt * (orig_h / scannet_h)",
        },
        "overlay_alpha": args.overlay_alpha,
        "ignore_labels": list(ignore_labels),
        "pairs": vis_index,
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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
