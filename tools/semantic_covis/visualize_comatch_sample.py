#!/usr/bin/env python3
"""
Quick CoMatch visualization on sample image pairs.
Fully reuses test.py data loading logic.

Usage:
    cd /ssd-data3/zh2025/CoMatch
    python tools/semantic_covis/visualize_comatch_sample.py \
        --image0 assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg \
        --image1 assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg \
        --ckpt_path weights/comatch_outdoor.ckpt \
        --megasize 1152 \
        --output_dir outputs/sample_vis
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.lightning.lightning_loftr import PL_LoFTR
from src.datasets.megadepth import MegaDepthDataset
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image0', type=str, required=True)
    parser.add_argument('--image1', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='weights/comatch_outdoor.ckpt')
    parser.add_argument('--megasize', type=int, default=1152)
    parser.add_argument('--thr', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='outputs/sample_vis')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # === Config (same as test.py) ===
    config = get_cfg_defaults()
    config.merge_from_file('configs/loftr/comatch_full.py')
    config.merge_from_file('configs/data/megadepth_test_1500.py')
    config.LOFTR.MATCH_COARSE.THR = args.thr
    config.DATASET.MGDPT_IMG_RESIZE = args.megasize
    config.LOFTR.COARSE.NPE = [832, 832, args.megasize, args.megasize]
    config.LOFTR.HALF = False
    config.LOFTR.MP = False
    config.DATASET.FP16 = False
    logger.info("Config done")

    # === Model ===
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path)
    model = model.cuda().eval()
    model.warmup = True
    logger.info("Model done")

    # === Create temp dataset with our images ===
    temp_dir = tempfile.mkdtemp()
    temp_npz_dir = os.path.join(temp_dir, 'scenes')
    os.makedirs(temp_npz_dir)

    img0_name = os.path.basename(args.image0)
    img1_name = os.path.basename(args.image1)

    scene_data = {
        'image_paths': [img0_name, img1_name],
        'intrinsics': [np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)],
        'poses': [np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)],
        'depth_paths': ['', ''],
        'pair_infos': np.array([([0, 1], 1.0, [])]),
    }
    scene_name = 'temp_scene'
    np.savez(os.path.join(temp_npz_dir, f'{scene_name}.npz'), **scene_data)
    with open(os.path.join(temp_dir, 'scene_list.txt'), 'w') as f:
        f.write(scene_name + '\n')

    test_root = os.path.dirname(args.image0)
    config.DATASET.TEST_DATA_ROOT = test_root
    config.DATASET.TEST_NPZ_ROOT = temp_npz_dir
    config.DATASET.TEST_LIST_PATH = os.path.join(temp_dir, 'scene_list.txt')

    dataset = MegaDepthDataset(
        root_dir=test_root,
        npz_path=os.path.join(temp_npz_dir, f'{scene_name}.npz'),
        mode='test',
        min_overlap_score=0.0,
        img_resize=config.DATASET.MGDPT_IMG_RESIZE,
        df=8,
        img_padding=False,
        depth_padding=False,
    )

    sample = dataset[0]
    batch = {
        'image0': sample['image0'].unsqueeze(0).cuda(),
        'image1': sample['image1'].unsqueeze(0).cuda(),
        'scale0': sample['scale0'].unsqueeze(0).cuda(),
        'scale1': sample['scale1'].unsqueeze(0).cuda(),
    }

    proc0_h, proc0_w = sample['image0'].shape[1], sample['image0'].shape[2]
    proc1_h, proc1_w = sample['image1'].shape[1], sample['image1'].shape[2]
    scale0 = sample['scale0'].numpy()
    scale1 = sample['scale1'].numpy()

    print(f"\n=== Sizes ===")
    print(f"Image0: processed={proc0_w}x{proc0_h}, scale0={scale0.tolist()}")
    print(f"Image1: processed={proc1_w}x{proc1_h}, scale1={scale1.tolist()}")

    del dataset

    # === Run model ===
    with torch.no_grad():
        model.matcher(batch)

    mkpts0_f = batch['mkpts0_f'].cpu().numpy()
    mkpts1_f = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    hw0_i = batch['hw0_i']
    hw1_i = batch['hw1_i']

    print(f"\n=== Model output ===")
    print(f"hw0_i={list(hw0_i)}, hw1_i={list(hw1_i)}")
    print(f"Matches: {len(mkpts0_f)}")
    print(f"mkpts0_f x: [{mkpts0_f[:,0].min():.1f}, {mkpts0_f[:,0].max():.1f}]")
    print(f"mkpts0_f y: [{mkpts0_f[:,1].min():.1f}, {mkpts0_f[:,1].max():.1f}]")

    # Divide by scale -> resized coords
    mkpts0_proc = mkpts0_f / scale0[[1, 0]]
    mkpts1_proc = mkpts1_f / scale1[[1, 0]]

    print(f"\n=== After / scale (proc coords) ===")
    print(f"mkpts0_proc x: [{mkpts0_proc[:,0].min():.1f}, {mkpts0_proc[:,0].max():.1f}] vs proc0_w={proc0_w}")
    print(f"mkpts0_proc y: [{mkpts0_proc[:,1].min():.1f}, {mkpts0_proc[:,1].max():.1f}] vs proc0_h={proc0_h}")

    print(f"\n=== Coordinate check ===")
    if mkpts0_f[:,0].max() <= proc0_w and mkpts0_f[:,1].max() <= proc0_h:
        print(">>> mkpts0_f is in PROCESSED image space")
    if mkpts0_f[:,0].max() <= hw0_i[1] and mkpts0_f[:,1].max() <= hw0_i[0]:
        print(">>> mkpts0_f is in hw0_i space")

    # === Visualize ===
    img0_raw = cv2.imread(args.image0, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
    orig0_h, orig0_w = img0_raw.shape
    orig1_h, orig1_w = img1_raw.shape

    img0_proc_np = (sample['image0'][0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img1_proc_np = (sample['image1'][0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    if img0_proc_np.ndim == 3 and img0_proc_np.shape[2] == 1:
        img0_proc_np = img0_proc_np.squeeze(2)
    if img1_proc_np.ndim == 3 and img1_proc_np.shape[2] == 1:
        img1_proc_np = img1_proc_np.squeeze(2)
    img0_proc_color = cv2.cvtColor(img0_proc_np, cv2.COLOR_GRAY2BGR)
    img1_proc_color = cv2.cvtColor(img1_proc_np, cv2.COLOR_GRAY2BGR)

    idx = np.arange(len(mkpts0_proc))
    if len(idx) > 500:
        idx = np.sort(np.random.choice(len(idx), 500, replace=False))

    # Vis on processed images
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(img0_proc_color, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img1_proc_color, cv2.COLOR_BGR2RGB))
    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    plt.tight_layout(pad=1)

    pts0 = mkpts0_proc[idx]
    pts1 = mkpts1_proc[idx]
    fig.canvas.draw()
    tf = fig.transFigure.inverted()
    fkpts0 = tf.transform(axes[0].transData.transform(pts0))
    fkpts1 = tf.transform(axes[1].transData.transform(pts1))
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
        transform=fig.transFigure, c='lime', linewidth=0.5) for i in range(len(pts0))]
    axes[0].scatter(pts0[:, 0], pts0[:, 1], c='lime', s=2)
    axes[1].scatter(pts1[:, 0], pts1[:, 1], c='lime', s=2)
    fig.suptitle(f'Processed ({proc0_w}x{proc0_h} | {proc1_w}x{proc1_h}), n={len(mkpts0_f)}', fontsize=12)
    plt.savefig(f'{args.output_dir}/matches_processed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output_dir}/matches_processed.png")

    # Vis on original images
    img0_orig_color = cv2.cvtColor(img0_raw, cv2.COLOR_GRAY2BGR)
    img1_orig_color = cv2.cvtColor(img1_raw, cv2.COLOR_GRAY2BGR)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(img0_orig_color, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img1_orig_color, cv2.COLOR_BGR2RGB))
    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    plt.tight_layout(pad=1)

    pts0_o = mkpts0_f[idx]
    pts1_o = mkpts1_f[idx]
    fig.canvas.draw()
    tf = fig.transFigure.inverted()
    fkpts0 = tf.transform(axes[0].transData.transform(pts0_o))
    fkpts1 = tf.transform(axes[1].transData.transform(pts1_o))
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
        transform=fig.transFigure, c='lime', linewidth=0.5) for i in range(len(pts0_o))]
    axes[0].scatter(pts0_o[:, 0], pts0_o[:, 1], c='lime', s=2)
    axes[1].scatter(pts1_o[:, 0], pts1_o[:, 1], c='lime', s=2)
    fig.suptitle(f'Original ({orig0_w}x{orig0_h} | {orig1_w}x{orig1_h}), n={len(mkpts0_f)}', fontsize=12)
    plt.savefig(f'{args.output_dir}/matches_original.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output_dir}/matches_original.png")

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
