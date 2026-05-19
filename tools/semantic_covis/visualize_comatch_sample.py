#!/usr/bin/env python3
"""
Quick CoMatch visualization on sample image pairs.

Usage (on server):
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.loftr import LoFTR
from src.utils.dataset import get_resized_wh, get_divisible_wh


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

    # Config (same as test.py / PL_LoFTR)
    config = get_cfg_defaults()
    config.merge_from_file('configs/loftr/comatch_full.py')
    config.merge_from_file('configs/data/megadepth_test_1500.py')
    config.LOFTR.MATCH_COARSE.THR = args.thr
    config.LOFTR.HALF = False
    config.LOFTR.MP = False
    if args.megasize is not None:
        config.DATASET.MGDPT_IMG_RESIZE = args.megasize
    # NPE config (same as outdoor.sh --npe)
    config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE]
    _config = lower_config(config)
    loftr_cfg = lower_config(_config['loftr'])

    # Model
    model = LoFTR(config=loftr_cfg)
    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()
    print("Model loaded.")

    # Load images
    img0_raw = cv2.imread(args.image0, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
    orig0_h, orig0_w = img0_raw.shape
    orig1_h, orig1_w = img1_raw.shape
    print(f"\n=== Original image sizes ===")
    print(f"Image0: {orig0_w}x{orig0_h}")
    print(f"Image1: {orig1_w}x{orig1_h}")

    # Resize (same as dataset, no padding)
    resize = args.megasize
    df = 8

    w0_new, h0_new = get_divisible_wh(*get_resized_wh(orig0_w, orig0_h, resize), df)
    img0_r = cv2.resize(img0_raw, (w0_new, h0_new))
    scale0 = torch.tensor([orig0_w / w0_new, orig0_h / h0_new], dtype=torch.float)

    w1_new, h1_new = get_divisible_wh(*get_resized_wh(orig1_w, orig1_h, resize), df)
    img1_r = cv2.resize(img1_raw, (w1_new, h1_new))
    scale1 = torch.tensor([orig1_w / w1_new, orig1_h / h1_new], dtype=torch.float)

    print(f"\n=== After resize (long_edge={resize}, df={df}) ===")
    print(f"Image0: {w0_new}x{h0_new}, scale0=[{scale0[0]:.4f}, {scale0[1]:.4f}]")
    print(f"Image1: {w1_new}x{h1_new}, scale1=[{scale1[0]:.4f}, {scale1[1]:.4f}]")

    # Build batch
    image0 = torch.from_numpy(img0_r).float()[None][None].cuda() / 255  # (1,1,H,W)
    image1 = torch.from_numpy(img1_r).float()[None][None].cuda() / 255

    data = {
        'image0': image0,
        'image1': image1,
        'scale0': scale0.unsqueeze(0).cuda(),
        'scale1': scale1.unsqueeze(0).cuda(),
    }

    # Run model
    with torch.no_grad():
        model(data)

    hw0_i = data['hw0_i']
    hw1_i = data['hw1_i']
    mkpts0_f = data['mkpts0_f'].cpu().numpy()
    mkpts1_f = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()

    print(f"\n=== Model output ===")
    print(f"hw0_i={list(hw0_i)}, hw1_i={list(hw1_i)}")
    print(f"Matches: {len(mkpts0_f)}")
    print(f"mkpts0_f x: [{mkpts0_f[:,0].min():.1f}, {mkpts0_f[:,0].max():.1f}]")
    print(f"mkpts0_f y: [{mkpts0_f[:,1].min():.1f}, {mkpts0_f[:,1].max():.1f}]")
    print(f"mkpts1_f x: [{mkpts1_f[:,0].min():.1f}, {mkpts1_f[:,0].max():.1f}]")
    print(f"mkpts1_f y: [{mkpts1_f[:,1].min():.1f}, {mkpts1_f[:,1].max():.1f}]")

    # Divide by scale -> resized image coordinates
    mkpts0_resized = mkpts0_f / scale0.numpy()[[1, 0]]
    mkpts1_resized = mkpts1_f / scale1.numpy()[[1, 0]]

    print(f"\n=== After / scale (resized coords) ===")
    print(f"mkpts0_resized x: [{mkpts0_resized[:,0].min():.1f}, {mkpts0_resized[:,0].max():.1f}] vs w0_new={w0_new}")
    print(f"mkpts0_resized y: [{mkpts0_resized[:,1].min():.1f}, {mkpts0_resized[:,1].max():.1f}] vs h0_new={h0_new}")

    # Conclusion
    if mkpts0_f[:,0].max() <= w0_new and mkpts0_f[:,1].max() <= h0_new:
        print(f"\n>>> CONCLUSION: mkpts0_f is in RESIZED image space ({w0_new}x{h0_new})")
    elif mkpts0_f[:,0].max() <= orig0_w and mkpts0_f[:,1].max() <= orig0_h:
        print(f"\n>>> CONCLUSION: mkpts0_f is in ORIGINAL image space ({orig0_w}x{orig0_h})")
    else:
        print(f"\n>>> CONCLUSION: mkpts0_f is in SCALED space (multiplied by scale0)")

    # === Visualization 1: On resized images (divide by scale) ===
    idx = np.arange(len(mkpts0_resized))
    if len(idx) > 500:
        idx = np.sort(np.random.choice(len(idx), 500, replace=False))

    img0_vis = cv2.cvtColor(img0_r, cv2.COLOR_GRAY2BGR)
    img1_vis = cv2.cvtColor(img1_r, cv2.COLOR_GRAY2BGR)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(img0_vis, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    plt.tight_layout(pad=1)

    pts0 = mkpts0_resized[idx]
    pts1 = mkpts1_resized[idx]
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(axes[0].transData.transform(pts0))
    fkpts1 = transFigure.transform(axes[1].transData.transform(pts1))
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
        transform=fig.transFigure, c='lime', linewidth=0.5) for i in range(len(pts0))]
    axes[0].scatter(pts0[:, 0], pts0[:, 1], c='lime', s=2)
    axes[1].scatter(pts1[:, 0], pts1[:, 1], c='lime', s=2)
    fig.suptitle(f'On resized images ({w0_new}x{h0_new}), matches={len(mkpts0_f)}', fontsize=12)
    plt.savefig(f'{args.output_dir}/matches_resized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output_dir}/matches_resized.png")

    # === Visualization 2: On original images (use mkpts0_f directly) ===
    img0_orig = cv2.cvtColor(img0_raw, cv2.COLOR_GRAY2BGR)
    img1_orig = cv2.cvtColor(img1_raw, cv2.COLOR_GRAY2BGR)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(img0_orig, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB))
    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    plt.tight_layout(pad=1)

    pts0_o = mkpts0_f[idx]
    pts1_o = mkpts1_f[idx]
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(axes[0].transData.transform(pts0_o))
    fkpts1 = transFigure.transform(axes[1].transData.transform(pts1_o))
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
        transform=fig.transFigure, c='lime', linewidth=0.5) for i in range(len(pts0_o))]
    axes[0].scatter(pts0_o[:, 0], pts0_o[:, 1], c='lime', s=2)
    axes[1].scatter(pts1_o[:, 0], pts1_o[:, 1], c='lime', s=2)
    fig.suptitle(f'On original images ({orig0_w}x{orig0_h}), matches={len(mkpts0_f)}', fontsize=12)
    plt.savefig(f'{args.output_dir}/matches_original.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output_dir}/matches_original.png")


if __name__ == '__main__':
    main()
