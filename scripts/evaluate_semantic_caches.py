#!/usr/bin/env python3
"""
Evaluate cross-semantic match rates using separate vs joint segmentation caches.

This script compares how cross-semantic match rates change when using:
1. Separate segmentation (current cache)
2. Joint segmentation (new --joint_mode cache)

Usage:
    python scripts/evaluate_semantic_caches.py \
        --separate_cache outputs/semantic_cache \
        --joint_cache outputs/semantic_cache_joint \
        --data_cfg configs/data/megadepth_test_1500.py \
        --model_id /ssd-data3/zh2025/HFModel/oneformer_ade20k_swin_large \
        --device cuda
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import SemanticLabelCache, normalize_semantic_path_key


def load_megadepth_pairs(data_cfg_path: str):
    """Load all pairs from MegaDepth test set."""
    from src.config.default import get_cfg_defaults
    from src.datasets.megadepth import MegaDepthDataset
    from torch.utils.data import ConcatDataset

    config = get_cfg_defaults()
    config.merge_from_file(data_cfg_path)

    data_root = config.DATASET.TEST_DATA_ROOT
    npz_root = config.DATASET.TEST_NPZ_ROOT
    scene_list_path = config.DATASET.TEST_LIST_PATH

    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    npz_names = [f'{n}.npz' for n in npz_names]

    datasets = []
    for npz_name in npz_names:
        npz_path = os.path.join(npz_root, npz_name)
        if not os.path.exists(npz_path):
            continue
        datasets.append(
            MegaDepthDataset(
                data_root,
                npz_path,
                mode='test',
                min_overlap_score=0.0,
                img_resize=config.DATASET.MGDPT_IMG_RESIZE,
                df=config.DATASET.MGDPT_DF,
                img_padding=config.DATASET.MGDPT_IMG_PAD,
                depth_padding=config.DATASET.MGDPT_DEPTH_PAD,
                fp16=False,
            )
        )

    concat_ds = ConcatDataset(datasets)
    pairs = []

    for idx in range(len(concat_ds)):
        cumlen = 0
        for ds in datasets:
            if idx < cumlen + len(ds):
                local_idx = idx - cumlen
                pair_info = ds.pair_infos[local_idx]
                idx0, idx1 = pair_info[0]

                ip_array = ds.scene_info['image_paths']
                rel_path_0 = str(ip_array[idx0])
                rel_path_1 = str(ip_array[idx1])

                abs_path_0 = os.path.join(data_root, rel_path_0)
                abs_path_1 = os.path.join(data_root, rel_path_1)

                pairs.append({
                    'pair_id': idx,
                    'rel_path_0': rel_path_0,
                    'rel_path_1': rel_path_1,
                    'abs_path_0': abs_path_0,
                    'abs_path_1': abs_path_1,
                })
                break
            cumlen += len(ds)

    return pairs, data_root


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
    parser.add_argument('--separate_cache', type=str, required=True)
    parser.add_argument('--joint_cache', type=str, required=True)
    parser.add_argument('--data_cfg', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/cache_comparison.json')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Limit number of pairs to evaluate (for debugging)')
    args = parser.parse_args()

    print("=" * 60)
    print("Loading Semantic Caches")
    print("=" * 60)

    cache_sep = SemanticLabelCache(args.separate_cache)
    cache_joint = SemanticLabelCache(args.joint_cache)

    print(f"  Separate cache: {args.separate_cache}")
    print(f"  Joint cache: {args.joint_cache}")

    # Check availability
    sep_entries = len(cache_sep._manifest) if cache_sep._manifest else 0
    joint_entries = len(cache_joint._manifest) if cache_joint._manifest else 0
    print(f"  Separate entries: {sep_entries}")
    print(f"  Joint entries: {joint_entries}")

    print("\n" + "=" * 60)
    print("Loading CoMatch")
    print("=" * 60)

    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config
    from src.loftr import LoFTR
    from src.utils.dataset import read_megadepth_gray

    config = get_cfg_defaults()
    config.merge_from_file("configs/loftr/comatch_full.py")
    config.merge_from_file(args.data_cfg)
    config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE]
    _config = lower_config(config)
    loftr_cfg = lower_config(_config["loftr"])

    ckpt_path = "weights/comatch_outdoor.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    matcher = LoFTR(config=loftr_cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)
    matcher = matcher.to(args.device).eval()

    print("\n" + "=" * 60)
    print("Loading MegaDepth Pairs")
    print("=" * 60)

    pairs, data_root = load_megadepth_pairs(args.data_cfg)
    print(f"  Total pairs: {len(pairs)}")

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"  Limited to: {len(pairs)}")

    print("\n" + "=" * 60)
    print("Evaluating Pairs")
    print("=" * 60)

    ignore_labels = {255, -1}
    results = []

    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            print(f"  Processing pair {i}/{len(pairs)}...")

        rel0 = pair['rel_path_0']
        rel1 = pair['rel_path_1']
        abs0 = pair['abs_path_0']
        abs1 = pair['abs_path_1']

        # Get semantic labels from both caches
        try:
            sem0_sep = cache_sep.get_label(rel0)
            sem1_sep = cache_sep.get_label(rel1)
        except Exception as e:
            print(f"  Warning: Missing in separate cache: {rel0} or {rel1}")
            continue

        try:
            sem0_joint = cache_joint.get_label(rel0)
            sem1_joint = cache_joint.get_label(rel1)
        except Exception as e:
            print(f"  Warning: Missing in joint cache: {rel0} or {rel1}")
            continue

        h0, w0 = sem0_sep.shape[:2]
        h1, w1 = sem1_sep.shape[:2]

        # Run CoMatch
        image0, mask0, scale0 = read_megadepth_gray(abs0, 832, 8, True, None)
        image1, mask1, scale1 = read_megadepth_gray(abs1, 832, 8, True, None)

        batch = {
            'image0': image0[None].float().to(args.device),
            'image1': image1[None].float().to(args.device),
            'scale0': scale0[None].float().to(args.device),
            'scale1': scale1[None].float().to(args.device),
        }

        with torch.no_grad():
            with torch.autocast(enabled=False, device_type="cuda"):
                matcher(batch)

        mkpts0 = batch["mkpts0_f"].cpu().numpy()
        mkpts1 = batch["mkpts1_f"].cpu().numpy()

        if len(mkpts0) == 0:
            continue

        # Compute cross stats for both caches
        cross_sep, same_sep, valid_sep, labels_sep = compute_cross_stats(
            mkpts0, mkpts1, sem0_sep, sem1_sep, h0, w0, h1, w1, ignore_labels)
        cross_joint, same_joint, valid_joint, labels_joint = compute_cross_stats(
            mkpts0, mkpts1, sem0_joint, sem1_joint, h0, w0, h1, w1, ignore_labels)

        rate_sep = cross_sep / valid_sep if valid_sep > 0 else float("nan")
        rate_joint = cross_joint / valid_joint if valid_joint > 0 else float("nan")

        results.append({
            'pair_id': pair['pair_id'],
            'num_matches': len(mkpts0),
            'separate': {
                'cross': int(cross_sep),
                'valid': int(valid_sep),
                'rate': float(rate_sep),
            },
            'joint': {
                'cross': int(cross_joint),
                'valid': int(valid_joint),
                'rate': float(rate_joint),
            },
            'reduction': float((rate_sep - rate_joint) / rate_sep * 100) if rate_sep > 0 else 0.0,
        })

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    total_pairs = len(results)
    total_matches = sum(r['num_matches'] for r in results)
    total_cross_sep = sum(r['separate']['cross'] for r in results)
    total_cross_joint = sum(r['joint']['cross'] for r in results)
    total_valid = sum(r['separate']['valid'] for r in results)

    overall_sep_rate = total_cross_sep / total_valid if total_valid > 0 else 0
    overall_joint_rate = total_cross_joint / total_valid if total_valid > 0 else 0

    # Count improvement cases
    improved = sum(1 for r in results if r['reduction'] > 0)
    degraded = sum(1 for r in results if r['reduction'] < 0)
    unchanged = sum(1 for r in results if r['reduction'] == 0)

    print(f"\nOverall Statistics:")
    print(f"  Pairs evaluated: {total_pairs}")
    print(f"  Total matches: {total_matches}")
    print(f"  Valid matches: {total_valid}")
    print(f"  Separate cross rate: {overall_sep_rate:.4f} ({total_cross_sep}/{total_valid})")
    print(f"  Joint cross rate: {overall_joint_rate:.4f} ({total_cross_joint}/{total_valid})")
    print(f"  Absolute reduction: {overall_sep_rate - overall_joint_rate:.4f}")
    print(f"  Relative reduction: {(overall_sep_rate - overall_joint_rate) / overall_sep_rate * 100:.2f}%")

    print(f"\nPer-pair improvement:")
    print(f"  Improved: {improved} ({improved/total_pairs*100:.1f}%)")
    print(f"  Degraded: {degraded} ({degraded/total_pairs*100:.1f}%)")
    print(f"  Unchanged: {unchanged} ({unchanged/total_pairs*100:.1f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'total_pairs': total_pairs,
        'total_matches': total_matches,
        'total_valid': total_valid,
        'overall_separate_rate': float(overall_sep_rate),
        'overall_joint_rate': float(overall_joint_rate),
        'absolute_reduction': float(overall_sep_rate - overall_joint_rate),
        'relative_reduction': float((overall_sep_rate - overall_joint_rate) / overall_sep_rate * 100) if overall_sep_rate > 0 else 0,
        'improved_pairs': improved,
        'degraded_pairs': degraded,
        'unchanged_pairs': unchanged,
        'per_pair_results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print top improved/degraded pairs
    sorted_by_improvement = sorted(results, key=lambda x: -x['reduction'])
    print(f"\nTop 5 most improved pairs:")
    for r in sorted_by_improvement[:5]:
        print(f"  Pair {r['pair_id']}: {r['separate']['rate']:.4f} -> {r['joint']['rate']:.4f} ({r['reduction']:.1f}% improvement)")

    print(f"\nTop 5 most degraded pairs:")
    for r in sorted_by_improvement[-5:]:
        if r['reduction'] < 0:
            print(f"  Pair {r['pair_id']}: {r['separate']['rate']:.4f} -> {r['joint']['rate']:.4f} ({r['reduction']:.1f}% worse)")


if __name__ == "__main__":
    main()