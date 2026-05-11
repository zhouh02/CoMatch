#!/usr/bin/env python3
"""
CLIP-CSLS Semantic Co-visibility Pseudo-label Estimator.

Implements the clip_csls_v1 method:
    - CSLS (Cross-domain Similarity through Softmax) debiased similarity
    - Top-k existence score (soft, not max)
    - Soft mutual consistency (not hard MNN)
    - Per-image quantile normalization
    - Optional spatial smoothing
    - Optional background prompt penalty

Usage:
    from clip_csls_sem_covis import compute_clip_semantic_covisibility_label

    result = compute_clip_semantic_covisibility_label(
        F0, F1, grid_hw0, grid_hw1,
        csls_k=20, topk_ratio=0.05, topk_min=5,
        tau=0.07, smooth_kernel=3,
        return_debug=True,
    )
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# =============================================================================
# Core helper functions
# =============================================================================


def l2_normalize(x, dim=-1, eps=1e-6):
    # type: (torch.Tensor, int, float) -> torch.Tensor
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def compute_csls_similarity(F0, F1, csls_k=20):
    # type: (torch.Tensor, torch.Tensor, int) -> Tuple[torch.Tensor, torch.Tensor]
    """Compute CSLS debiased similarity.

    Args:
        F0: [N0, C] patch features.
        F1: [N1, C] patch features.
        csls_k: k for top-k mean in CSLS hubness correction.

    Returns:
        (S_csls, S_raw) both [N0, N1].
    """
    F0 = l2_normalize(F0, dim=-1)
    F1 = l2_normalize(F1, dim=-1)

    S = F0 @ F1.T  # [N0, N1]

    N0, N1 = S.shape

    k0 = min(csls_k, N1)
    k1 = min(csls_k, N0)

    # r0[i] = mean of top-k similarities in row i
    r0 = S.topk(k=k0, dim=1).values.mean(dim=1)  # [N0]
    # r1[j] = mean of top-k similarities in column j
    r1 = S.topk(k=k1, dim=0).values.mean(dim=0)  # [N1]

    S_csls = 2.0 * S - r0.unsqueeze(1) - r1.unsqueeze(0)  # [N0, N1]

    return S_csls, S


def compute_existence_score(S_csls, topk_ratio=0.05, topk_min=5):
    # type: (torch.Tensor, float, int) -> torch.Tensor
    """Compute soft semantic existence score per source patch.

    For each row i in S_csls, take the mean of top-k values in that row.

    Args:
        S_csls: [N0, N1] similarity matrix.
        topk_ratio: fraction of target patches for top-k.
        topk_min: minimum k.

    Returns:
        E0: [N0] existence score.
    """
    N_target = S_csls.shape[1]
    k = max(topk_min, int(N_target * topk_ratio))
    k = min(k, N_target)
    k = max(k, 1)

    topk_vals = S_csls.topk(k=k, dim=1).values  # [N0, k]
    return topk_vals.mean(dim=1)  # [N0]


def compute_soft_mutual_consistency(S_csls, tau=0.07):
    # type: (torch.Tensor, float) -> torch.Tensor
    """Compute soft mutual consistency.

    P01 = softmax(S_csls / tau, dim=1)
    P10 = softmax(S_csls.T / tau, dim=1)
    C0[i] = sum_j P01[i,j] * P10[j,i]

    Args:
        S_csls: [N0, N1].
        tau: softmax temperature.

    Returns:
        C0: [N0].
    """
    P01 = F.softmax(S_csls / tau, dim=1)  # [N0, N1]
    P10 = F.softmax(S_csls.T / tau, dim=1)  # [N1, N0]

    # C0[i] = sum_j P01[i,j] * P10[j,i]
    C0 = (P01 * P10.T).sum(dim=1)  # [N0]
    return C0


def quantile_normalize(x, q_low=0.05, q_high=0.95, eps=1e-6):
    # type: (torch.Tensor, float, float, float) -> torch.Tensor
    """Per-image quantile normalization to [0, 1]."""
    flat = x.flatten().float()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    return torch.clamp((x.float() - lo) / (hi - lo + eps), 0.0, 1.0)


def smooth_score_map(score, grid_hw, kernel_size=3):
    # type: (torch.Tensor, Tuple[int, int], int) -> torch.Tensor
    """Smooth a score map via avg_pool2d.

    Args:
        score: [N] score vector.
        grid_hw: (Hc, Wc).
        kernel_size: pooling kernel. 1 = no smoothing.

    Returns:
        score_map: [Hc, Wc].
    """
    Hc, Wc = grid_hw
    score_map = score.view(1, 1, Hc, Wc)

    if kernel_size <= 1:
        return score_map.squeeze(0).squeeze(0)

    padding = kernel_size // 2
    score_map = F.avg_pool2d(score_map, kernel_size=kernel_size,
                              stride=1, padding=padding)
    return score_map.squeeze(0).squeeze(0)  # [Hc, Wc]


# =============================================================================
# Background prompt penalty
# =============================================================================

_DEFAULT_BG_PROMPTS = [
    "sky", "cloud", "water", "lake", "sea", "river",
    "grass", "road", "wall", "floor", "ceiling",
    "plain background", "repetitive texture", "tree leaves",
]


def compute_background_prompt_score(
    patch_features,
    text_features,
    prompts=None,
    q_low=0.05,
    q_high=0.95,
):
    # type: (torch.Tensor, torch.Tensor, list, float, float) -> Optional[torch.Tensor]
    """Compute background score from CLIP text features.

    Args:
        patch_features: [N, C] L2-normalized vision patch features.
        text_features: [T, C] L2-normalized text features.
        prompts: list of prompt strings (unused at runtime, for reference).
        q_low, q_high: quantile normalization bounds.

    Returns:
        bg_score: [N] background score in [0, 1], or None if dims mismatch.
    """
    patch_dim = patch_features.shape[-1]
    text_dim = text_features.shape[-1]

    if patch_dim != text_dim:
        warnings.warn(
            "background prompt skipped because patch dim != text dim: "
            "patch dim = {}, text dim = {}".format(patch_dim, text_dim)
        )
        return None

    pf = l2_normalize(patch_features, dim=-1)
    tf = l2_normalize(text_features, dim=-1)

    sim = pf @ tf.T  # [N, T]
    bg_score = sim.max(dim=1).values  # [N]
    bg_score = quantile_normalize(bg_score, q_low, q_high)
    return bg_score


# =============================================================================
# Main: compute_clip_semantic_covisibility_label
# =============================================================================


def compute_clip_semantic_covisibility_label(
    F0,                          # type: torch.Tensor
    F1,                          # type: torch.Tensor
    grid_hw0,                    # type: Tuple[int, int]
    grid_hw1,                    # type: Tuple[int, int]
    csls_k=20,                   # type: int
    topk_ratio=0.05,             # type: float
    topk_min=5,                  # type: int
    tau=0.07,                    # type: float
    smooth_kernel=3,             # type: int
    q_low=0.05,                  # type: float
    q_high=0.95,                 # type: float
    bg_score0=None,              # type: Optional[torch.Tensor]
    bg_score1=None,              # type: Optional[torch.Tensor]
    bg_weight=0.5,               # type: float
    normalize_components=True,   # type: bool
    return_debug=False,          # type: bool
):
    # type: (...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]
    """Compute CLIP-CSLS semantic co-visibility pseudo-labels.

    Args:
        F0: [N0, C] CLIP patch features for image 0.
        F1: [N1, C] CLIP patch features for image 1.
        grid_hw0: (H0, W0) spatial grid for image 0.
        grid_hw1: (H1, W1) spatial grid for image 1.
        csls_k: k for CSLS hubness correction.
        topk_ratio: fraction of target patches for existence score.
        topk_min: minimum k for existence score.
        tau: softmax temperature for soft mutual consistency.
        smooth_kernel: avg_pool kernel size for spatial smoothing.
        q_low: lower quantile for normalization.
        q_high: upper quantile for normalization.
        bg_score0: optional [N0] background prompt score for image 0.
        bg_score1: optional [N1] background prompt score for image 1.
        bg_weight: weight for background penalty.
        normalize_components: if True use 0.3+0.7 weighting; else 0.5+0.5.
        return_debug: if True return extra debug dict.

    Returns:
        (Y0_map, Y1_map) each [H, W] in [0, 1].
        If return_debug, also returns debug dict.
    """
    # Shape asserts
    N0 = F0.shape[0]
    N1 = F1.shape[0]
    H0, W0 = grid_hw0
    H1, W1 = grid_hw1
    assert N0 == H0 * W0, (
        "F0.shape[0]={} != grid_hw0 area {}*{}={}".format(N0, H0, W0, H0 * W0)
    )
    assert N1 == H1 * W1, (
        "F1.shape[0]={} != grid_hw1 area {}*{}={}".format(N1, H1, W1, H1 * W1)
    )

    # CSLS similarity
    S_csls, S_raw = compute_csls_similarity(F0, F1, csls_k)

    # Existence scores
    E0 = compute_existence_score(S_csls, topk_ratio, topk_min)     # [N0]
    E1 = compute_existence_score(S_csls.T, topk_ratio, topk_min)   # [N1]

    # Soft mutual consistency
    C0 = compute_soft_mutual_consistency(S_csls, tau)   # [N0]
    C1 = compute_soft_mutual_consistency(S_csls.T, tau)  # [N1]

    # Combine existence and consistency
    if normalize_components:
        E0 = quantile_normalize(E0, q_low, q_high)
        E1 = quantile_normalize(E1, q_low, q_high)
        C0 = quantile_normalize(C0, q_low, q_high)
        C1 = quantile_normalize(C1, q_low, q_high)
        Y0 = E0 * (0.3 + 0.7 * C0)
        Y1 = E1 * (0.3 + 0.7 * C1)
    else:
        Y0 = E0 * (0.5 + 0.5 * C0)
        Y1 = E1 * (0.5 + 0.5 * C1)

    # Background penalty
    if bg_score0 is not None:
        Y0 = Y0 * (1.0 - bg_weight * bg_score0)
    if bg_score1 is not None:
        Y1 = Y1 * (1.0 - bg_weight * bg_score1)

    # Save pre-smooth values for debug
    Y0_before_smooth = Y0.clone()
    Y1_before_smooth = Y1.clone()

    # Reshape to spatial grid + smooth
    Y0_map = smooth_score_map(Y0, grid_hw0, smooth_kernel)
    Y1_map = smooth_score_map(Y1, grid_hw1, smooth_kernel)

    # Save pre-final-norm for debug
    Y0_before_norm = Y0_map.clone()
    Y1_before_norm = Y1_map.clone()

    # Final quantile normalize to [0, 1]
    Y0_map = quantile_normalize(Y0_map, q_low, q_high)
    Y1_map = quantile_normalize(Y1_map, q_low, q_high)

    if return_debug:
        debug = {
            "S_raw": S_raw,
            "S_csls": S_csls,
            "E0": E0,
            "E1": E1,
            "C0": C0,
            "C1": C1,
            "Y0_before_smooth": Y0_before_smooth,
            "Y1_before_smooth": Y1_before_smooth,
            "Y0_before_norm": Y0_before_norm,
            "Y1_before_norm": Y1_before_norm,
        }
        return Y0_map, Y1_map, debug

    return Y0_map, Y1_map


# =============================================================================
# Unit Tests
# =============================================================================


def _check_tensor(name, t, expected_shape, device="cpu"):
    # type: (str, torch.Tensor, Tuple[int, ...], str) -> None
    assert t.shape == expected_shape, (
        "{}: expected shape {}, got {}".format(name, expected_shape, list(t.shape))
    )
    assert t.device.type == device, (
        "{}: expected device {}, got {}".format(name, device, t.device)
    )
    assert torch.isfinite(t).all(), "{}: contains NaN or Inf".format(name)


def run_tests():
    # type: () -> None
    """Run unit tests for clip_csls_sem_covis module."""
    print("=" * 60)
    print("clip_csls_sem_covis Unit Tests")
    print("=" * 60)

    all_pass = True

    # ------------------------------------------------------------------
    # Test 1: Basic forward pass
    # ------------------------------------------------------------------
    print("\n[Test 1] Basic: [576, 1024] + 24x24 grid")
    torch.manual_seed(42)
    F0 = torch.randn(576, 1024)
    F1 = torch.randn(576, 1024)
    F0 = l2_normalize(F0)
    F1 = l2_normalize(F1)

    Y0, Y1, dbg = compute_clip_semantic_covisibility_label(
        F0, F1, (24, 24), (24, 24), return_debug=True,
    )

    _check_tensor("Y0", Y0, (24, 24))
    _check_tensor("Y1", Y1, (24, 24))
    assert (Y0 >= 0).all() and (Y0 <= 1).all(), "Y0 out of [0,1]"
    assert (Y1 >= 0).all() and (Y1 <= 1).all(), "Y1 out of [0,1]"

    for key in ["S_raw", "S_csls", "E0", "E1", "C0", "C1"]:
        assert torch.isfinite(dbg[key]).all(), "{} not finite".format(key)

    print("  Y0 range: [{:.4f}, {:.4f}]".format(float(Y0.min()), float(Y0.max())))
    print("  Y1 range: [{:.4f}, {:.4f}]".format(float(Y1.min()), float(Y1.max())))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 2: Asymmetric grids
    # ------------------------------------------------------------------
    print("\n[Test 2] Asymmetric: F0 [480, 256] + 20x24, F1 [360, 256] + 15x24")
    torch.manual_seed(42)
    F0a = l2_normalize(torch.randn(480, 256))
    F1a = l2_normalize(torch.randn(360, 256))

    Y0a, Y1a = compute_clip_semantic_covisibility_label(
        F0a, F1a, (20, 24), (15, 24),
    )

    _check_tensor("Y0a", Y0a, (20, 24))
    _check_tensor("Y1a", Y1a, (15, 24))
    assert (Y0a >= 0).all() and (Y0a <= 1).all()
    assert (Y1a >= 0).all() and (Y1a <= 1).all()
    print("  Y0a shape: {}".format(list(Y0a.shape)))
    print("  Y1a shape: {}".format(list(Y1a.shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 3: Small patches (edge case)
    # ------------------------------------------------------------------
    print("\n[Test 3] Small: [49, 128] + 7x7")
    torch.manual_seed(42)
    F0s = l2_normalize(torch.randn(49, 128))
    F1s = l2_normalize(torch.randn(49, 128))

    Y0s, Y1s = compute_clip_semantic_covisibility_label(
        F0s, F1s, (7, 7), (7, 7), csls_k=5, topk_min=2,
    )

    _check_tensor("Y0s", Y0s, (7, 7))
    _check_tensor("Y1s", Y1s, (7, 7))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 4: Background penalty
    # ------------------------------------------------------------------
    print("\n[Test 4] Background penalty")
    torch.manual_seed(42)
    F0b = l2_normalize(torch.randn(576, 1024))
    F1b = l2_normalize(torch.randn(576, 1024))
    bg0 = torch.rand(576)
    bg1 = torch.rand(576)

    Y0b, Y1b = compute_clip_semantic_covisibility_label(
        F0b, F1b, (24, 24), (24, 24),
        bg_score0=bg0, bg_score1=bg1,
    )
    assert (Y0b >= 0).all() and (Y0b <= 1).all()
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 5: normalize_components=False
    # ------------------------------------------------------------------
    print("\n[Test 5] normalize_components=False")
    Y0n, Y1n = compute_clip_semantic_covisibility_label(
        F0, F1, (24, 24), (24, 24), normalize_components=False,
    )
    assert (Y0n >= 0).all() and (Y0n <= 1).all()
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 6: smooth_kernel=1 (no smoothing)
    # ------------------------------------------------------------------
    print("\n[Test 6] smooth_kernel=1")
    Y0k, Y1k = compute_clip_semantic_covisibility_label(
        F0, F1, (24, 24), (24, 24), smooth_kernel=1,
    )
    assert Y0k.shape == (24, 24)
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 7: Shape mismatch assert
    # ------------------------------------------------------------------
    print("\n[Test 7] Shape mismatch assertion")
    try:
        F0m = l2_normalize(torch.randn(576, 1024))
        F1m = l2_normalize(torch.randn(576, 1024))
        compute_clip_semantic_covisibility_label(F0m, F1m, (20, 20), (24, 24))
        print("  [FAIL] Should have raised AssertionError")
        all_pass = False
    except AssertionError:
        print("  [PASS] Correctly raised AssertionError")

    # ------------------------------------------------------------------
    # Test 8: CSLS similarity helper
    # ------------------------------------------------------------------
    print("\n[Test 8] compute_csls_similarity")
    torch.manual_seed(42)
    A = l2_normalize(torch.randn(50, 64))
    B = l2_normalize(torch.randn(40, 64))
    S_csls, S_raw = compute_csls_similarity(A, B, csls_k=10)
    assert S_csls.shape == (50, 40)
    assert S_raw.shape == (50, 40)
    print("  S_csls shape: {}".format(list(S_csls.shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 9: GPU
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        print("\n[Test 9] GPU test")
        F0g = l2_normalize(torch.randn(576, 1024)).cuda()
        F1g = l2_normalize(torch.randn(576, 1024)).cuda()
        Y0g, Y1g = compute_clip_semantic_covisibility_label(
            F0g, F1g, (24, 24), (24, 24),
        )
        _check_tensor("Y0g", Y0g, (24, 24), device="cuda")
        print("  [PASS]")
    else:
        print("\n[Test 9] GPU: SKIPPED (no CUDA)")

    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests passed")
    else:
        print("Some tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
