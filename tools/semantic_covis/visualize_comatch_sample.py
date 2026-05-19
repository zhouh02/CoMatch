#!/usr/bin/env python3
"""
CoMatch two-image matching visualization.

Loads two images, runs CoMatch inference, and produces a side-by-side
match visualization with confidence-colored lines.

Usage:
    cd /path/to/CoMatch
    python tools/semantic_covis/visualize_comatch_sample.py \
        --image0 assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg \
        --image1 assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg \
        --ckpt_path weights/comatch_outdoor.ckpt \
        --output_path outputs/sample_vis/match.png
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.config.default import get_cfg_defaults
from src.utils.dataset import read_megadepth_gray
from src.utils.plotting import make_matching_figure, dynamic_alpha, error_colormap
from src.lightning.lightning_loftr import PL_LoFTR
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='CoMatch two-image match visualization')
    parser.add_argument('--image0', type=str, required=True, help='Path to first image')
    parser.add_argument('--image1', type=str, required=True, help='Path to second image')
    parser.add_argument('--ckpt_path', type=str, default='weights/comatch_outdoor.ckpt')
    parser.add_argument('--megasize', type=int, default=1152, help='Resize longer edge')
    parser.add_argument('--thr', type=float, default=0.1, help='Coarse match confidence threshold')
    parser.add_argument('--max_matches', type=int, default=300, help='Max matches to draw')
    parser.add_argument('--output_path', type=str, default='outputs/sample_vis/match.png')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--no_npe', action='store_true', help='Disable NPE positional encoding')
    return parser.parse_args()


def load_and_preprocess(image_path, resize, df=8):
    """Load image, resize, and return tensor + metadata."""
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    img_tensor, _, scale = read_megadepth_gray(image_path, resize=resize, df=df)
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    return img_tensor, scale, img_raw, h, w


def run_inference(model, image0, image1, scale0, scale1):
    """Run CoMatch and return match keypoints + confidence."""
    batch = {
        'image0': image0.unsqueeze(0).cuda(),
        'image1': image1.unsqueeze(0).cuda(),
        'scale0': scale0.unsqueeze(0).cuda(),
        'scale1': scale1.unsqueeze(0).cuda(),
    }

    with torch.no_grad():
        model.matcher(batch)

    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # original-image coords
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    return mkpts0, mkpts1, mconf


def visualize(img0, img1, mkpts0, mkpts1, mconf, scale0, scale1,
              proc0_hw, proc1_hw, max_matches=300, output_path='match.png', dpi=150):
    """Draw matches on processed images and save."""
    scale0_np = scale0.numpy()
    scale1_np = scale1.numpy()

    # Convert mkpts from original coords to processed-image coords
    mkpts0_proc = mkpts0 / scale0_np[[1, 0]]
    mkpts1_proc = mkpts1 / scale1_np[[1, 0]]

    # Subsample if too many
    n = len(mkpts0_proc)
    if n > max_matches:
        idx = np.sort(np.random.choice(n, max_matches, replace=False))
        mkpts0_proc = mkpts0_proc[idx]
        mkpts1_proc = mkpts1_proc[idx]
        mconf = mconf[idx]

    # Color by confidence (green=high, red=low)
    alpha = dynamic_alpha(n)
    color = error_colormap(1 - mconf, 0.5, alpha=alpha)

    # Prepare processed images as numpy (from tensor)
    # Use the raw images resized to processed size for better visualization
    img0_vis = cv2.resize(img0, (proc0_hw[1], proc0_hw[0]))
    img1_vis = cv2.resize(img1, (proc1_hw[1], proc1_hw[0]))

    text = [
        f'#Matches: {n}',
        f'Image0: {proc0_hw[1]}x{proc0_hw[0]}',
        f'Image1: {proc1_hw[1]}x{proc1_hw[0]}',
    ]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    make_matching_figure(
        img0_vis, img1_vis,
        mkpts0_proc, mkpts1_proc, color,
        text=text, dpi=dpi, path=output_path,
    )
    logger.info(f'Saved: {output_path}  ({n} matches)')


def main():
    args = parse_args()

    # --- Config ---
    config = get_cfg_defaults()
    config.merge_from_file('configs/loftr/comatch_full.py')
    config.merge_from_file('configs/data/megadepth_test_1500.py')
    config.LOFTR.MATCH_COARSE.THR = args.thr
    config.DATASET.MGDPT_IMG_RESIZE = args.megasize
    if not args.no_npe:
        config.LOFTR.COARSE.NPE = [832, 832, args.megasize, args.megasize]
    else:
        config.LOFTR.COARSE.NPE = [832, 832, 832, 832]
    config.LOFTR.HALF = False
    config.LOFTR.MP = False
    config.DATASET.FP16 = False
    logger.info('Config loaded')

    # --- Model ---
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path)
    model.cuda().eval()
    logger.info(f'Model loaded from {args.ckpt_path}')

    # --- Images ---
    img0_tensor, scale0, img0_raw, h0, w0 = load_and_preprocess(args.image0, args.megasize)
    img1_tensor, scale1, img1_raw, h1, w1 = load_and_preprocess(args.image1, args.megasize)
    logger.info(f'Image0: {img0_raw.shape[1]}x{img0_raw.shape[0]} -> {w0}x{h0}')
    logger.info(f'Image1: {img1_raw.shape[1]}x{img1_raw.shape[0]} -> {w1}x{h1}')

    # --- Inference ---
    mkpts0, mkpts1, mconf = run_inference(model, img0_tensor, img1_tensor, scale0, scale1)
    logger.info(f'Matches found: {len(mkpts0)}')

    # --- Visualize ---
    visualize(
        img0_raw, img1_raw, mkpts0, mkpts1, mconf,
        scale0, scale1,
        proc0_hw=(h0, w0), proc1_hw=(h1, w1),
        max_matches=args.max_matches,
        output_path=args.output_path,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
