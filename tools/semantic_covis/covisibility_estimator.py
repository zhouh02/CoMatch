#!/usr/bin/env python3
"""
Semantic Covisibility Pseudo-label Estimator.

Computes CLIP-based semantic covisibility pseudo-labels from
normalized patch tokens of two images.

Usage:
    python tools/semantic_covis/covisibility_estimator.py

Pipeline:
    Z0, Z1 (L2-normalized patch tokens)
    -> S01 = Z0 @ Z1.T                    (similarity matrix)
    -> existence  = top-k row/col mean     (semantic existence)
    -> specificity = 1 - entropy/log(N)    (matching specificity)
    -> raw = existence * (0.5 + 0.5*spec)  (raw score)
    -> y_sem = percentile normalize         (pseudo-label)
    -> conf = spec * |y_sem - 0.5| * 2     (confidence)
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _ensure_2d(Z):
    # type: (torch.Tensor) -> torch.Tensor
    """Squeeze [1, N, D] -> [N, D]. Pass through [N, D] unchanged."""
    if Z.ndim == 3 and Z.shape[0] == 1:
        return Z.squeeze(0)
    if Z.ndim != 2:
        raise ValueError(
            "Expected 2D [N, D] or 3D [1, N, D], got shape {}".format(
                list(Z.shape)
            )
        )
    return Z


def compute_semantic_covisibility(
    Z0,              # type: torch.Tensor
    Z1,              # type: torch.Tensor
    grid0,           # type: Tuple[int, int]
    grid1,           # type: Tuple[int, int]
    topk=5,          # type: int
    temperature=0.07,    # type: float
    q_low=40,        # type: float
    q_high=85,       # type: float
    eps=1e-8,        # type: float
):
    # type: (...) -> Dict[str, torch.Tensor]
    """Compute semantic covisibility pseudo-labels from CLIP patch tokens.

    Args:
        Z0: Patch tokens for image 0, shape [N0, D] or [1, N0, D].
        Z1: Patch tokens for image 1, shape [N1, D] or [1, N1, D].
        grid0: Spatial grid size for image 0, (Gh0, Gw0).
        grid1: Spatial grid size for image 1, (Gh1, Gw1).
        topk: Number of top matches for existence score (default: 5).
        temperature: Softmax temperature for specificity (default: 0.07).
        q_low: Lower percentile for normalization (default: 40).
        q_high: Upper percentile for normalization (default: 85).
        eps: Small constant for numerical stability.

    Returns:
        Dict with keys (all tensors on same device as input):
            y_sem0_clip:      [Gh0, Gw0] pseudo-label in [0, 1]
            y_sem1_clip:      [Gh1, Gw1] pseudo-label in [0, 1]
            conf0_clip:       [Gh0, Gw0] confidence in [0, 1]
            conf1_clip:       [Gh1, Gw1] confidence in [0, 1]
            existence0_clip:  [Gh0, Gw0] existence score
            existence1_clip:  [Gh1, Gw1] existence score
            specificity0_clip: [Gh0, Gw0] specificity score
            specificity1_clip: [Gh1, Gw1] specificity score
            raw0_clip:        [Gh0, Gw0] raw score before normalization
            raw1_clip:        [Gh1, Gw1] raw score before normalization
    """
    # ------------------------------------------------------------------
    # 0. Prepare tensors
    # ------------------------------------------------------------------
    Z0 = _ensure_2d(Z0)  # [N0, D]
    Z1 = _ensure_2d(Z1)  # [N1, D]

    N0, D = Z0.shape
    N1, _ = Z1.shape
    Gh0, Gw0 = grid0
    Gh1, Gw1 = grid1

    assert N0 == Gh0 * Gw0, "Z0 length {} != grid0 area {}".format(N0, Gh0 * Gw0)
    assert N1 == Gh1 * Gw1, "Z1 length {} != grid1 area {}".format(N1, Gh1 * Gw1)

    device = Z0.device
    dtype = Z0.dtype

    # ------------------------------------------------------------------
    # 1. L2 normalize (idempotent if already normalized)
    # ------------------------------------------------------------------
    Z0 = F.normalize(Z0, p=2, dim=-1)
    Z1 = F.normalize(Z1, p=2, dim=-1)

    # ------------------------------------------------------------------
    # 2. Similarity matrix
    # ------------------------------------------------------------------
    S01 = Z0 @ Z1.T  # [N0, N1]

    # ------------------------------------------------------------------
    # 3. Semantic existence: row/col top-k mean
    # ------------------------------------------------------------------
    k = min(topk, N1)
    topk_vals_0, _ = S01.topk(k, dim=1)  # [N0, k]
    existence0 = topk_vals_0.mean(dim=1)  # [N0]

    k1 = min(topk, N0)
    topk_vals_1, _ = S01.topk(k1, dim=0)  # [k1, N1]
    existence1 = topk_vals_1.mean(dim=0)  # [N1]

    # ------------------------------------------------------------------
    # 4. Specificity: 1 - entropy / log(N)
    # ------------------------------------------------------------------
    log_N1 = math.log(float(N1))
    log_N0 = math.log(float(N0))

    # Specificity for image 0: softmax over columns (dim=1)
    P01 = F.softmax(S01 / temperature, dim=1)  # [N0, N1]
    entropy0 = -(P01 * (P01 + eps).log()).sum(dim=1)  # [N0]
    specificity0 = 1.0 - entropy0 / log_N1  # [N0]

    # Specificity for image 1: softmax over rows (dim=0) via S01.T
    P10 = F.softmax(S01.T / temperature, dim=1)  # [N1, N0]
    entropy1 = -(P10 * (P10 + eps).log()).sum(dim=1)  # [N1]
    specificity1 = 1.0 - entropy1 / log_N0  # [N1]

    # ------------------------------------------------------------------
    # 5. Raw score: existence * (0.5 + 0.5 * specificity)
    # ------------------------------------------------------------------
    raw0 = existence0 * (0.5 + 0.5 * specificity0)  # [N0]
    raw1 = existence1 * (0.5 + 0.5 * specificity1)  # [N1]

    # ------------------------------------------------------------------
    # 6. Percentile normalize: y_sem in [0, 1]
    # ------------------------------------------------------------------
    def percentile_normalize(raw):
        # type: (torch.Tensor) -> torch.Tensor
        """Normalize raw scores using percentile clipping."""
        flat = raw.flatten().float()
        q_lo_val = torch.quantile(flat, q_low / 100.0)
        q_hi_val = torch.quantile(flat, q_high / 100.0)
        normalized = (raw - q_lo_val) / (q_hi_val - q_lo_val + eps)
        return torch.clamp(normalized, 0.0, 1.0)

    y_sem0 = percentile_normalize(raw0)  # [N0]
    y_sem1 = percentile_normalize(raw1)  # [N1]

    # ------------------------------------------------------------------
    # 7. Confidence: spec * |y_sem - 0.5| * 2
    # ------------------------------------------------------------------
    conf0 = specificity0 * (y_sem0 - 0.5).abs() * 2.0  # [N0]
    conf1 = specificity1 * (y_sem1 - 0.5).abs() * 2.0  # [N1]
    conf0 = torch.clamp(conf0, 0.0, 1.0)
    conf1 = torch.clamp(conf1, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 8. Reshape to spatial grid
    # ------------------------------------------------------------------
    result = {
        "y_sem0_clip": y_sem0.view(Gh0, Gw0),
        "y_sem1_clip": y_sem1.view(Gh1, Gw1),
        "conf0_clip": conf0.view(Gh0, Gw0),
        "conf1_clip": conf1.view(Gh1, Gw1),
        "existence0_clip": existence0.view(Gh0, Gw0),
        "existence1_clip": existence1.view(Gh1, Gw1),
        "specificity0_clip": specificity0.view(Gh0, Gw0),
        "specificity1_clip": specificity1.view(Gh1, Gw1),
        "raw0_clip": raw0.view(Gh0, Gw0),
        "raw1_clip": raw1.view(Gh1, Gw1),
    }

    return result


# =============================================================================
# Unit Tests
# =============================================================================


def _check_tensor(name, t, expected_shape, device="cpu"):
    # type: (str, torch.Tensor, Tuple[int, int], str) -> None
    """Assert tensor has correct shape, is finite, and in valid range."""
    assert t.shape == expected_shape, \
        "{}: expected shape {}, got {}".format(name, expected_shape, list(t.shape))
    assert t.device.type == device, \
        "{}: expected device {}, got {}".format(name, device, t.device)
    assert torch.isfinite(t).all(), \
        "{}: contains NaN or Inf".format(name)


def run_tests():
    # type: () -> None
    """Run all unit tests for covisibility estimator."""
    print("=" * 60)
    print("Covisibility Estimator Unit Tests")
    print("=" * 60)

    all_pass = True

    # ------------------------------------------------------------------
    # Test 1: [576, 1024] + 24x24 grid (CLIP-L/14@336)
    # ------------------------------------------------------------------
    print("\n[Test 1] CLIP-L/14@336: [576, 1024] + 24x24 grid")
    torch.manual_seed(42)
    Z0 = torch.randn(576, 1024)
    Z1 = torch.randn(576, 1024)
    Z0 = F.normalize(Z0, p=2, dim=-1)
    Z1 = F.normalize(Z1, p=2, dim=-1)

    result = compute_semantic_covisibility(Z0, Z1, (24, 24), (24, 24))

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result[key], (24, 24))
        assert (result[key] >= 0).all() and (result[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    for key in ["existence0_clip", "existence1_clip",
                "specificity0_clip", "specificity1_clip",
                "raw0_clip", "raw1_clip"]:
        _check_tensor(key, result[key], (24, 24))
        assert torch.isfinite(result[key]).all(), \
            "{}: contains NaN/Inf".format(key)

    print("  y_sem0_clip shape: {}".format(list(result["y_sem0_clip"].shape)))
    print("  y_sem0_clip range: [{:.4f}, {:.4f}]".format(
        float(result["y_sem0_clip"].min()), float(result["y_sem0_clip"].max())))
    print("  conf0_clip range:  [{:.4f}, {:.4f}]".format(
        float(result["conf0_clip"].min()), float(result["conf0_clip"].max())))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 2: [1, 576, 1024] + 24x24 grid (with batch dim)
    # ------------------------------------------------------------------
    print("\n[Test 2] CLIP-L/14@336 with batch dim: [1, 576, 1024] + 24x24 grid")
    torch.manual_seed(42)
    Z0_b = torch.randn(1, 576, 1024)
    Z1_b = torch.randn(1, 576, 1024)
    Z0_b = F.normalize(Z0_b, p=2, dim=-1)
    Z1_b = F.normalize(Z1_b, p=2, dim=-1)

    result2 = compute_semantic_covisibility(Z0_b, Z1_b, (24, 24), (24, 24))

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result2[key], (24, 24))
        assert (result2[key] >= 0).all() and (result2[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    print("  y_sem0_clip shape: {}".format(list(result2["y_sem0_clip"].shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 3: [196, 768] + 14x14 grid (CLIP-B/16)
    # ------------------------------------------------------------------
    print("\n[Test 3] CLIP-B/16: [196, 768] + 14x14 grid")
    torch.manual_seed(42)
    Z0_s = torch.randn(196, 768)
    Z1_s = torch.randn(196, 768)
    Z0_s = F.normalize(Z0_s, p=2, dim=-1)
    Z1_s = F.normalize(Z1_s, p=2, dim=-1)

    result3 = compute_semantic_covisibility(Z0_s, Z1_s, (14, 14), (14, 14))

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result3[key], (14, 14))
        assert (result3[key] >= 0).all() and (result3[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    print("  y_sem0_clip shape: {}".format(list(result3["y_sem0_clip"].shape)))
    print("  conf0_clip range:  [{:.4f}, {:.4f}]".format(
        float(result3["conf0_clip"].min()), float(result3["conf0_clip"].max())))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 4: Asymmetric grids
    # ------------------------------------------------------------------
    print("\n[Test 4] Asymmetric grids: Z0 [480, 256] + 20x24, Z1 [360, 256] + 15x24")
    torch.manual_seed(42)
    Z0_a = F.normalize(torch.randn(480, 256), p=2, dim=-1)
    Z1_a = F.normalize(torch.randn(360, 256), p=2, dim=-1)

    result4 = compute_semantic_covisibility(Z0_a, Z1_a, (20, 24), (15, 24))

    _check_tensor("y_sem0_clip", result4["y_sem0_clip"], (20, 24))
    _check_tensor("y_sem1_clip", result4["y_sem1_clip"], (15, 24))
    _check_tensor("conf0_clip", result4["conf0_clip"], (20, 24))
    _check_tensor("conf1_clip", result4["conf1_clip"], (15, 24))
    print("  y_sem0_clip shape: {}".format(list(result4["y_sem0_clip"].shape)))
    print("  y_sem1_clip shape: {}".format(list(result4["y_sem1_clip"].shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 5: GPU (if available)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        print("\n[Test 5] GPU test: [1, 576, 1024] + 24x24 on CUDA")
        torch.manual_seed(42)
        Z0_gpu = F.normalize(torch.randn(1, 576, 1024), p=2, dim=-1).cuda()
        Z1_gpu = F.normalize(torch.randn(1, 576, 1024), p=2, dim=-1).cuda()

        result5 = compute_semantic_covisibility(Z0_gpu, Z1_gpu, (24, 24), (24, 24))

        for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
            _check_tensor(key, result5[key], (24, 24), device="cuda")
            assert (result5[key] >= 0).all() and (result5[key] <= 1).all()

        print("  y_sem0_clip device: {}".format(result5["y_sem0_clip"].device))
        print("  [PASS]")
    else:
        print("\n[Test 5] GPU: SKIPPED (no CUDA)")

    # ------------------------------------------------------------------
    # Test 6: Grid mismatch assertion
    # ------------------------------------------------------------------
    print("\n[Test 6] Grid mismatch assertion")
    try:
        Z0_m = F.normalize(torch.randn(576, 1024), p=2, dim=-1)
        Z1_m = F.normalize(torch.randn(576, 1024), p=2, dim=-1)
        compute_semantic_covisibility(Z0_m, Z1_m, (20, 20), (24, 24))
        print("  [FAIL] Should have raised AssertionError")
        all_pass = False
    except AssertionError:
        print("  [PASS] Correctly raised AssertionError")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests passed")
        print("  y_sem0_clip shape: {}".format(list(result["y_sem0_clip"].shape)))
        print("  conf0_clip range:  [{:.4f}, {:.4f}]".format(
            float(result["conf0_clip"].min()), float(result["conf0_clip"].max())))
    else:
        print("Some tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
