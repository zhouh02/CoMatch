#!/usr/bin/env python3
"""
Dump Outdoor Test Images - Export full test image set for semantic diagnostics.

This script statically parses the MegaDepth test configuration to export
the complete candidate image set, NOT just images from a single eval batch.

Usage:
    python tools/semantic_covis/dump_outdoor_test_images.py --out-dir outputs/semantic_diagnostic_outdoor

The script exports:
    - full_test_image_list.txt: one image path per line (relative to image_root)
    - full_test_pair_meta.jsonl: one pair per line with metadata
    - test_dataset_source_report.json: dataset configuration summary
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_scene_list(scene_list_path):
    """Parse scene list file to get npz names."""
    with open(scene_list_path, 'r') as f:
        return [name.strip().split()[0] for name in f.readlines()]


def load_test_config():
    """Load test configuration from megadepth_test_1500.py"""
    # Read the config file to extract paths
    config_path = PROJECT_ROOT / "configs" / "data" / "megadepth_test_1500.py"
    with open(config_path, 'r') as f:
        content = f.read()

    # Parse paths from config
    test_base_path = "assets/megadepth_test_1500_scene_info"
    test_data_root = "data/megadepth/test"
    test_npz_root = f"{test_base_path}"
    test_list_path = f"{test_base_path}/megadepth_test_1500.txt"

    return {
        'test_base_path': test_base_path,
        'test_data_root': test_data_root,
        'test_npz_root': test_npz_root,
        'test_list_path': test_list_path,
    }


def get_image_root_from_ckpt(ckpt_path=None):
    """
    Determine the actual image root for MegaDepth.

    The actual image path structure is:
        {data_root}/phoenix/S6/zlsa/...

    We need to find where the images actually are.
    """
    # For MegaDepth, the images are stored at a different path than test_data_root
    # The npz files contain paths like 'Undistorted_SfM/0022/images/xxx.jpg'
    # We need to determine the actual image root

    # Common MegaDepth paths
    possible_roots = [
        PROJECT_ROOT / "data" / "megadepth" / "test",
        PROJECT_ROOT / "data" / "megadepth" / "images",
        Path("D:/data/megadepth"),
        Path("E:/data/megadepth"),
        Path("/ssd-data3/zh2025/megadepth"),
    ]

    for root in possible_roots:
        if root.exists():
            # Check if 'phoenix' subdirectory exists
            phoenix = root / "phoenix"
            if phoenix.exists():
                return str(root)

    # Default to test data root
    return str(PROJECT_ROOT / test_data_root)


def extract_images_from_npz(npz_path, scene_id):
    """
    Extract all image paths from an npz file.

    Returns:
        - valid_image_paths: list of relative paths to images
        - pair_infos: list of (idx0, idx1, overlap_score) tuples
        - total_indices: all indices that appear in pairs
    """
    data = dict(npz_path.share_vars()) if hasattr(npz_path, 'share_vars') else np.load(npz_path, allow_pickle=True)

    image_paths = data['image_paths']
    pair_infos = data['pair_infos']

    # Filter valid image paths (non-None)
    valid_paths = []
    valid_indices = set()
    for i, path in enumerate(image_paths):
        if path is not None:
            valid_paths.append((i, path))
            valid_indices.add(i)

    # Collect all indices used in pairs
    all_pair_indices = set()
    for pair_info in pair_infos:
        idx0, idx1 = pair_info[0]
        all_pair_indices.add(idx0)
        all_pair_indices.add(idx1)

    # Get unique images used in pairs
    images_in_pairs = []
    for idx in sorted(all_pair_indices):
        if idx in valid_indices:
            rel_path = image_paths[idx]
            images_in_pairs.append(rel_path)

    # Get all valid images (from valid_indices)
    all_valid_images = [image_paths[i] for i in sorted(valid_indices)]

    return {
        'all_valid_images': all_valid_images,
        'images_in_pairs': images_in_pairs,
        'all_pair_indices': all_pair_indices,
        'num_pairs': len(pair_infos),
        'pair_infos_sample': [(tuple(p[0]), float(p[1])) for p in pair_infos[:3]],
    }


def main():
    parser = argparse.ArgumentParser(description="Dump outdoor test images for semantic diagnostics")
    parser.add_argument('--out-dir', type=str, default='outputs/semantic_diagnostic_outdoor',
                        help='Output directory for the dumped images')
    parser.add_argument('--npz-dir', type=str, default=None,
                        help='Override npz directory path')
    parser.add_argument('--image-root', type=str, default=None,
                        help='Override image root path')
    parser.add_argument('--scene-list', type=str, default=None,
                        help='Override scene list path')
    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent.parent.parent
    config = load_test_config()

    test_base_path = project_root / config['test_base_path']
    test_list_path = project_root / config['test_list_path']

    if args.npz_dir:
        npz_dir = Path(args.npz_dir)
    else:
        npz_dir = test_base_path

    if args.scene_list:
        scene_list_path = Path(args.scene_list)
    else:
        scene_list_path = test_list_path

    print("=" * 60)
    print("Dump Outdoor Test Images")
    print("=" * 60)
    print(f"  npz_dir:     {npz_dir}")
    print(f"  scene_list:  {scene_list_path}")
    print(f"  output_dir:   {args.out_dir}")

    # Parse scene list
    scene_names = parse_scene_list(scene_list_path)
    print(f"\n[1] Parsed {len(scene_names)} scenes from {scene_list_path.name}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images and pairs
    all_images = set()
    all_pairs = []
    pair_count = 0
    scene_reports = []

    for scene_name in scene_names:
        npz_file = npz_dir / f"{scene_name}.npz"

        if not npz_file.exists():
            print(f"  WARNING: {npz_file} not found, skipping")
            continue

        # Load npz
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"  ERROR: Failed to load {npz_file}: {e}")
            continue

        image_paths = data['image_paths']
        pair_infos = data['pair_infos']

        # Get valid images (non-None paths)
        valid_indices = set()
        valid_images = {}
        for i, path in enumerate(image_paths):
            # Handle numpy bytes and None
            if path is None or (isinstance(path, (bytes, np.bytes_)) and len(path) == 0):
                continue
            if isinstance(path, np.bytes_):
                path = path.decode('utf-8')
            elif not isinstance(path, str):
                continue
            valid_indices.add(i)
            valid_images[i] = path
            all_images.add(path)

        # Extract pair information
        scene_pair_count = 0
        for pair_info in pair_infos:
            idx0 = int(pair_info[0][0])
            idx1 = int(pair_info[0][1])
            overlap_score = float(pair_info[1])

            if idx0 not in valid_indices or idx1 not in valid_indices:
                continue

            img0 = valid_images[idx0]
            img1 = valid_images[idx1]

            pair_id = f"{scene_name}_{pair_count}"
            all_pairs.append({
                'pair_id': pair_id,
                'scene_id': scene_name,
                'image0': img0,
                'image1': img1,
                'overlap_score': overlap_score,
                'idx0': int(idx0),
                'idx1': int(idx1),
                'source_split_file': str(scene_list_path),
            })

            scene_pair_count += 1
            pair_count += 1

        scene_reports.append({
            'scene_name': scene_name,
            'num_images': len(valid_indices),
            'num_pairs': scene_pair_count,
            'npz_path': str(npz_file),
        })

        print(f"  Scene {scene_name}: {len(valid_indices)} images, {scene_pair_count} pairs")

    print(f"\n[2] Summary:")
    print(f"  Total scenes:         {len(scene_reports)}")
    print(f"  Total pairs:           {pair_count}")
    print(f"  Total unique images:  {len(all_images)}")

    # Write outputs
    print(f"\n[3] Writing output files...")

    # 1. full_test_image_list.txt
    image_list_path = out_dir / "full_test_image_list.txt"
    with open(image_list_path, 'w') as f:
        for img in sorted(all_images):
            f.write(img + '\n')
    print(f"  Wrote {image_list_path} ({len(all_images)} images)")

    # 2. full_test_pair_meta.jsonl
    pair_meta_path = out_dir / "full_test_pair_meta.jsonl"
    with open(pair_meta_path, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')
    print(f"  Wrote {pair_meta_path} ({len(all_pairs)} pairs)")

    # 3. test_dataset_source_report.json
    report = {
        'dataset_name': 'MegaDepth',
        'data_source': 'MegaDepth',
        'test_data_root': config['test_data_root'],
        'npz_root': str(npz_dir),
        'split_files': [str(scene_list_path)],
        'pair_files': [str(npz_dir / f"{s}.npz") for s in scene_names],
        'num_scenes': len(scene_reports),
        'num_pairs': pair_count,
        'num_unique_images': len(all_images),
        'has_random_sampling': False,
        'random_sampling_notes': (
            "No random sampling in test mode. "
            "All pairs from npz files are deterministic. "
            "DistributedSampler only shuffles order, not the set of pairs."
        ),
        'path_format': 'relative_to_data_root',
        'image_paths_in_npz': 'relative paths like Undistorted_SfM/0022/images/xxx.jpg',
        'outdoor_sh_command': (
            "python ./test.py configs/data/megadepth_test_1500.py configs/loftr/comatch_full.py "
            "--ckpt_path=weights/comatch_outdoor.ckpt --megasize 1152 --thr 0.1 --ransac_times 5 --deter"
        ),
        'eval_entry': 'test.py with PL_LoFTR.test_step',
        'config_files': [
            'configs/data/megadepth_test_1500.py',
            'configs/loftr/comatch_full.py',
        ],
        'scene_reports': scene_reports,
        'mkpts_coordinate_note': (
            "mkpts0_f / mkpts1_f are in resized input image coordinates (not original). "
            "scale0/scale1 from dataset describe resize ratio. "
            "OneFormer segmentation must match the resized input dimensions."
        ),
    }

    report_path = out_dir / "test_dataset_source_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_path}")

    print(f"\n[4] Done!")
    print(f"\nNext steps:")
    print(f"  1. Run OneFormer preprocessing:")
    print(f"     python tools/semantic_covis/export_oneformer_dir.py \\")
    print(f"       --image-list {image_list_path} \\")
    print(f"       --image-root <actual_megadepth_root> \\")
    print(f"       --output-dir outputs/oneformer_outdoor_test \\")
    print(f"       --model-dir /ssd-data3/zh2025/HFModel/oneformer_ade20k_swin_large \\")
    print(f"       --meta-dir /ssd-data3/zh2025/HFModel/oneformer_demo")
    print(f"\n  2. Run outdoor evaluation with semantic diagnostic:")
    print(f"     SEMANTIC_DIAGNOSTIC=1 \\")
    print(f"     ONEFORMER_DIR=outputs/oneformer_outdoor_test \\")
    print(f"     SEMANTIC_DIAGNOSTIC_OUTPUT=outputs/semantic_diagnostic_outdoor \\")
    print(f"     bash scripts/reproduce_test/outdoor.sh")

    return 0


if __name__ == '__main__':
    import numpy as np
    sys.exit(main())