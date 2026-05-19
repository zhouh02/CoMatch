#!/usr/bin/env python3
"""
Quick CoMatch visualization on sample image pairs.

This script runs CoMatch on a pair of images and visualizes the matching results,
showing the coordinate system at each stage so you can verify correctness.

Usage:
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
import matplotlib.lines as mlines

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.utils.dataset import read_megadepth_gray, get_resized_wh, get_divisible_wh


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

    # Load config
    config = get_cfg_defaults()
    config.merge_from_file('configs/loftr/comatch_full.py')
    config.LOFTR.MATCH_COARSE.THR = args.thr
    config.LOFTR.HALF = False
    config.LOFTR.MP = False

    # Load model
    model = LoFTR(config=config)
    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()
    print("Model loaded.")

    # Load and preprocess images (same as MegaDepth dataset)
    img0_raw = cv2.imread(args.image0, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
    orig0_h, orig0_w = img0_raw.shape
    orig1_h, orig1_w = img1_raw.shape
    print(f"\n=== Original image sizes ===")
    print(f"Image0: {orig0_w}x{orig0_h}")
    print(f"Image1: {orig1_w}x{orig1_h}")

    # Process same as read_megadepth_gray
    resize = args.megasize
    df = 8

    # Image 0
    w0, h0 = img0_raw.shape[1], img0_raw.shape[0]
    w0_new, h0_new = get_resized_wh(w0, h0, resize)
    w0_new, h0_new = get_divisible_wh(w0_new, h0_new, df)
    img0_resized = cv2.resize(img0_raw, (w0_new, h0_new))
    scale0 = torch.tensor([w0 / w0_new, h0 / h0_new], dtype=torch.float)

    # Image 1
    w1, h1 = img1_raw.shape[1], img1_raw.shape[0]
    w1_new, h1_new = get_resized_wh(w1, h1, resize)
    w1_new, h1_new = get_divisible_wh(w1_new, h1_new, df)
    img1_resized = cv2.resize(img1_raw, (w1_new, h1_new))
    scale1 = torch.tensor([w1 / w1_new, h1 / h1_new], dtype=torch.float)

    print(f"\n=== After resize (long edge={resize}, df={df}) ===")
    print(f"Image0: {w0_new}x{h0_new}, scale0={scale0.tolist()}")
    print(f"Image1: {w1_new}x{h1_new}, scale1={scale1.tolist()}")

    # Build batch (no padding for test mode)
    image0 = torch.from_numpy(img0_resized).float()[None] / 255  # (1, H, W)
    image1 = torch.from_numpy(img1_resized).float()[None] / 255

    data = {
        'image0': image0.unsqueeze(0).cuda(),  # (1, 1, H, W)
        'image1': image1.unsqueeze(0).cuda(),
        'scale0': scale0.unsqueeze(0).cuda(),  # (1, 2)
        'scale1': scale1.unsqueeze(0).cuda(),
    }

    # Run model
    with torch.no_grad():
        model(data)

    # Extract results
    hw0_i = data['hw0_i']  # (H, W) input image size
    hw1_i = data['hw1_i']
    mkpts0_f = data['mkpts0_f'].cpu().numpy()  # (M, 2) in scaled coordinates
    mkpts1_f = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()

    print(f"\n=== Model output ===")
    print(f"hw0_i: {hw0_i}")
    print(f"hw1_i: {hw1_i}")
    print(f"Num matches: {len(mkpts0_f)}")
    print(f"mkpts0_f x range: [{mkpts0_f[:,0].min():.1f}, {mkpts0_f[:,0].max():.1f}]")
    print(f"mkpts0_f y range: [{mkpts0_f[:,1].min():.1f}, {mkpts0_f[:,1].max():.1f}]")
    print(f"mkpts1_f x range: [{mkpts1_f[:,0].min():.1f}, {mkpts1_f[:,0].max():.1f}]")
    print(f"mkpts1_f y range: [{mkpts1_f[:,1].min():.1f}, {mkpts1_f[:,1].max():.1f}]")

    # Convert mkpts back to resized image coordinates (divide by scale)
    # This is what plotting.py does: kpts0 = kpts0 / scale0[[1,0]]
    mkpts0_resized = mkpts0_f / scale0.numpy()[[1, 0]]  # divide by [scale_h, scale_w]
    mkpts1_resized = mkpts1_f / scale1.numpy()[[1, 0]]

    print(f"\n=== After dividing by scale (resized image coords) ===")
    print(f"scale0: {scale0.tolist()}, scale0[[1,0]]: {scale0.numpy()[[1,0]].tolist()}")
    print(f"mkpts0_resized x range: [{mkpts0_resized[:,0].min():.1f}, {mkpts0_resized[:,0].max():.1f}]")
    print(f"mkpts0_resized y range: [{mkpts0_resized[:,1].min():.1f}, {mkpts0_resized[:,1].max():.1f}]")
    print(f"Expected range: x<=w0_new={w0_new}, y<=h0_new={h0_new}")

    # ============ Visualization 1: On resized images ============
    img0_color = cv2.cvtColor(img0_resized, cv2.COLOR_GRAY2BGR)
    img1_color = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    plt.tight_layout(pad=1)

    # Sample matches for clarity
    max_draw = 500
    if len(mkpts0_resized) > max_draw:
        idx = np.random.choice(len(mkpts0_resized), max_draw, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(len(mkpts0_resized))

    pts0 = mkpts0_resized[idx]
    pts1 = mkpts1_resized[idx]

    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(axes[0].transData.transform(pts0))
    fkpts1 = transFigure.transform(axes[1].transData.transform(pts1))
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]),
        (fkpts0[i, 1], fkpts1[i, 1]),
        transform=fig.transFigure, c='lime', linewidth=0.5)
        for i in range(len(pts0))]
    axes[0].scatter(pts0[:, 0], pts0[:, 1], c='lime', s=2)
    axes[1].scatter(pts1[:, 0], pts1[:, 1], c='lime', s=2)
    fig.suptitle(f"CoMatch on resized images ({w0_new}x{h0_new} | {w1_new}x{h1_new}), "
                 f"matches={len(mkpts0_f)}", fontsize=12)
    plt.savefig(f"{args.output_dir}/matches_resized.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output_dir}/matches_resized.png")

    # ============ Visualization 2: On original images ============
    img0_orig_color = cv2.cvtColor(img0_raw, cv2.COLOR_GRAY2BGR)
    img1_orig_color = cv2.cvtColor(img1_raw, cv2.COLOR_GRAY2BGR)

    # mkpts0_f are already in original scale (multiplied by scale0)
    # So they should map directly onto original images
    print(f"\n=== On original images ===")
    print(f"mkpts0_f x range: [{mkpts0_f[:,0].min():.1f}, {mkpts0_f[:,0].max():.1f}] vs orig_w={orig0_w}")
    print(f"mkpts0_f y range: [{mkpts0_f[:,1].min():.1f}, {mkpts0_f[:,1].max():.1f}] vs orig_h={orig0_h}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(img0_orig_color, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img1_orig_color, cv2.COLOR_BGR2RGB))
    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    plt.tight_layout(pad=1)

    pts0_orig = mkpts0_f[idx]
    pts1_orig = mkpts1_f[idx]

    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(axes[0].transData.transform(pts0_orig))
    fkpts1 = transFigure.transform(axes[1].transData.transform(pts1_orig))
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]),
        (fkpts0[i, 1], fkpts1[i, 1]),
        transform=fig.transFigure, c='lime', linewidth=0.5)
        for i in range(len(pts0_orig))]
    axes[0].scatter(pts0_orig[:, 0], pts0_orig[:, 1], c='lime', s=2)
    axes[1].scatter(pts1_orig[:, 0], pts1_orig[:, 1], c='lime', s=2)
    fig.suptitle(f"CoMatch on ORIGINAL images ({orig0_w}x{orig0_h} | {orig1_w}x{orig1_h}), "
                 f"matches={len(mkpts0_f)}", fontsize=12)
    plt.savefig(f"{args.output_dir}/matches_original.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output_dir}/matches_original.png")

    print(f"\n=== Summary ===")
    print(f"mkpts0_f coordinates are in ORIGINAL image space (multiplied by scale0)")
    print(f"  scale0 = [orig_w/processed_w, orig_h/processed_h] = {scale0.tolist()}")
    print(f"  So mkpts0_f x_max={mkpts0_f[:,0].max():.1f} should be <= orig_w={orig0_w}")
    print(f"  And mkpts0_f y_max={mkpts0_f[:,1].max():.1f} should be <= orig_h={orig0_h}")


if __name__ == '__main__':
    main()
