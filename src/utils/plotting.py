import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic', use_original_size=False, image_root=None):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    # Get processed images (resize+padding)
    img0_proc = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1_proc = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)

    # mkpts0_f/mkpts1_f are in ORIGINAL image coordinate space
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    # for megadepth, mkpts are in original image space
    if 'scale0' in data:
        # kpts are already in original image coords from fine_matching
        # scale0 = [orig_w/processed_w, orig_h/processed_h]
        # So original coords = kpts (no conversion needed)
        pass

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)

    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # If use_original_size is requested but image_root not provided, fall back to processed
    if use_original_size and image_root:
        import cv2
        import os
        from pathlib import Path

        # Get original image paths from batch
        pair_names = data.get('pair_names', [])
        if isinstance(pair_names, (list, tuple)) and len(pair_names) >= 2:
            img0_rel = str(pair_names[0])
            img1_rel = str(pair_names[1])

            # Handle list-wrapped paths
            if img0_rel.startswith('[') or img0_rel.startswith('('):
                import ast
                try:
                    img0_rel = str(ast.literal_eval(img0_rel)[0])
                    img1_rel = str(ast.literal_eval(img1_rel)[0])
                except:
                    pass

            # Load original images
            img0_path = Path(image_root) / img0_rel if image_root else Path(img0_rel)
            img1_path = Path(image_root) / img1_rel if image_root else Path(img1_rel)

            if img0_path.exists() and img1_path.exists():
                img0_orig = cv2.imread(str(img0_path), cv2.IMREAD_GRAYSCALE)
                img1_orig = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
                if img0_orig is not None and img1_orig is not None:
                    text.append(f'Original size: {img0_orig.shape[1]}x{img0_orig.shape[0]} | {img1_orig.shape[1]}x{img1_orig.shape[0]}')
                    figure = make_matching_figure(img0_orig, img1_orig, kpts0, kpts1, color, text=text)
                    return figure

    # Fallback: visualize on processed images (kpts in original coords -> need to convert back)
    # kpts are in original coords, but we want to show on processed images
    scale0 = data['scale0'][b_id].cpu().numpy()
    scale1 = data['scale1'][b_id].cpu().numpy()
    kpts0_proc = kpts0 / scale0[[1, 0]]  # convert original -> processed
    kpts1_proc = kpts1 / scale1[[1, 0]]

    # Resize processed images to common size for display
    h0, w0 = img0_proc.shape
    h1, w1 = img1_proc.shape
    max_h = max(h0, h1)
    img0_display = cv2.resize(img0_proc, (w0, max_h)) if h0 < max_h else img0_proc
    img1_display = cv2.resize(img1_proc, (w1, max_h)) if h1 < max_h else img1_proc

    figure = make_matching_figure(img0_display, img1_display, kpts0_proc, kpts1_proc, color, text=text)
    return figure

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()

def make_matching_figures(data, config, mode='evaluation', use_original_size=False, image_root=None):
    """ Make matching figures for a batch.

    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
        use_original_size (bool): if True, visualize on original image sizes
        image_root (str): root directory for loading original images
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA,
                use_original_size=use_original_size,
                image_root=image_root)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)