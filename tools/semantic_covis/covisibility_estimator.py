#!/usr/bin/env python3
"""
Semantic Covisibility Pseudo-label Estimator.

Supports two modes:
    v1: Original formula (compatibility)
    v2: Enhanced formula with specificity_gamma, mutual nearest neighbor, margin gate

Usage:
    python tools/semantic_covis/covisibility_estimator.py
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


def _percentile_normalize(raw, q_low, q_high, eps):
    # type: (torch.Tensor, float, float, float) -> torch.Tensor
    """Normalize raw scores using percentile clipping."""
    flat = raw.flatten().float()
    q_lo_val = torch.quantile(flat, q_low / 100.0)
    q_hi_val = torch.quantile(flat, q_high / 100.0)
    normalized = (raw - q_lo_val) / (q_hi_val - q_lo_val + eps)
    return torch.clamp(normalized, 0.0, 1.0)


def _margin_gate(S01, N0, N1, k_margin, q30, q80, eps):
    # type: (torch.Tensor, int, int, int, float, float, float) -> Tuple[torch.Tensor, torch.Tensor]
    """Compute margin gate for both directions.

    Args:
        S01: Similarity matrix [N0, N1]
        N0, N1: Number of patches
        k_margin: k for top-k margin computation (default 10)
        q30, q80: Percentiles for normalization
        eps: Small constant

    Returns:
        (margin_gate0, margin_gate1) both in [0, 1]
    """
    k0 = min(k_margin, N1)
    k1 = min(k_margin, N0)

    # Margin for image 0 patches
    top_vals0, _ = S01.topk(k=k0, dim=1)  # [N0, k0]
    margin0 = top_vals0[:, 0] - top_vals0[:, 1:].mean(dim=1)  # [N0]
    margin_gate0 = _percentile_normalize(margin0, q30, q80, eps)

    # Margin for image 1 patches
    top_vals1, _ = S01.T.topk(k=k1, dim=1)  # [N1, k1]
    margin1 = top_vals1[:, 0] - top_vals1[:, 1:].mean(dim=1)  # [N1]
    margin_gate1 = _percentile_normalize(margin1, q30, q80, eps)

    return margin_gate0, margin_gate1


def _mutual_nearest_consistency(S01, N0, N1):
    # type: (torch.Tensor, int, int) -> Tuple[torch.Tensor, torch.Tensor]
    """Compute mutual nearest neighbor consistency scores.

    Args:
        S01: Similarity matrix [N0, N1]
        N0, N1: Number of patches

    Returns:
        (mutual0, mutual1) both in [0, 1]
    """
    # Best match for each patch in image 0
    best01 = S01.argmax(dim=1)  # [N0]
    # Best match for each patch in image 1
    best10 = S01.argmax(dim=0)  # [N1]

    # Mutual: patch 0's best is 1, and 1's best is 0
    mutual0 = torch.zeros(N0, dtype=torch.float32, device=S01.device)
    mutual1 = torch.zeros(N1, dtype=torch.float32, device=S01.device)

    for i in range(N0):
        j = best01[i].item()
        if best10[j].item() == i:
            mutual0[i] = 1.0

    for j in range(N1):
        i = best10[j].item()
        if best01[i].item() == j:
            mutual1[j] = 1.0

    return mutual0, mutual1


def compute_semantic_covisibility(
    Z0,              # type: torch.Tensor
    Z1,              # type: torch.Tensor
    grid0,           # type: Tuple[int, int]
    grid1,           # type: Tuple[int, int]
    mode="v1",      # type: str
    topk=5,          # type: int
    temperature=0.07,    # type: float
    q_low=40,        # type: float
    q_high=85,       # type: float
    specificity_gamma=1.5,  # type: float
    k_margin=10,     # type: int
    q_margin_low=30, # type: float
    q_margin_high=80, # type: float
    eps=1e-8,       # type: float
):
    # type: (...) -> Dict[str, torch.Tensor]
    """Compute semantic covisibility pseudo-labels from CLIP patch tokens.

    Args:
        Z0: Patch tokens for image 0, shape [N0, D] or [1, N0, D].
        Z1: Patch tokens for image 1, shape [N1, D] or [1, N1, D].
        grid0: Spatial grid size for image 0, (Gh0, Gw0).
        grid1: Spatial grid size for image 1, (Gh1, Gw1).
        mode: "v1" (original) or "v2" (enhanced).
        topk: Number of top matches for existence score (default: 5).
        temperature: Softmax temperature for specificity (default: 0.07).
        q_low: Lower percentile for y_sem normalization (default: 40).
        q_high: Upper percentile for y_sem normalization (default: 85).
        specificity_gamma: Gamma for specificity power (v2 only, default: 1.5).
        k_margin: k for margin computation (v2 only, default: 10).
        q_margin_low: Lower percentile for margin gate (v2 only, default: 30).
        q_margin_high: Upper percentile for margin gate (v2 only, default: 80).
        eps: Small constant for numerical stability.

    Returns:
        Dict with keys (all tensors on same device as input):
            y_sem0_clip, y_sem1_clip: [Gh0, Gw0], [Gh1, Gw1] pseudo-label in [0, 1]
            conf0_clip, conf1_clip: [Gh0, Gw0], [Gh1, Gw1] confidence in [0, 1]
            existence0_clip, existence1_clip: [Gh0, Gw0], [Gh1, Gw1]
            specificity0_clip, specificity1_clip: [Gh0, Gw0], [Gh1, Gw1]
            raw0_clip, raw1_clip: [Gh0, Gw0], [Gh1, Gw1]
            margin_gate0_clip, margin_gate1_clip: [Gh0, Gw0], [Gh1, Gw1] (v2 only)
            mutual0_clip, mutual1_clip: [Gh0, Gw0], [Gh1, Gw1] (v2 only)
    """
    # Prepare tensors
    Z0 = _ensure_2d(Z0)  # [N0, D]
    Z1 = _ensure_2d(Z1)  # [N1, D]

    N0, D = Z0.shape
    N1, _ = Z1.shape
    Gh0, Gw0 = grid0
    Gh1, Gw1 = grid1

    assert N0 == Gh0 * Gw0, "Z0 length {} != grid0 area {}".format(N0, Gh0 * Gw0)
    assert N1 == Gh1 * Gw1, "Z1 length {} != grid1 area {}".format(N1, Gh1 * Gw1)

    # L2 normalize (idempotent if already normalized)
    Z0 = F.normalize(Z0, p=2, dim=-1)
    Z1 = F.normalize(Z1, p=2, dim=-1)

    # Similarity matrix
    S01 = Z0 @ Z1.T  # [N0, N1]

    # Semantic existence: row/col top-k mean
    k = min(topk, N1)
    topk_vals_0, _ = S01.topk(k, dim=1)  # [N0, k]
    existence0 = topk_vals_0.mean(dim=1)  # [N0]

    k1 = min(topk, N0)
    topk_vals_1, _ = S01.topk(k1, dim=0)  # [k1, N1]
    existence1 = topk_vals_1.mean(dim=0)  # [N1]

    # Specificity: 1 - entropy / log(N)
    log_N1 = math.log(float(N1))
    log_N0 = math.log(float(N0))

    P01 = F.softmax(S01 / temperature, dim=1)  # [N0, N1]
    entropy0 = -(P01 * (P01 + eps).log()).sum(dim=1)  # [N0]
    specificity0 = 1.0 - entropy0 / log_N1  # [N0]

    P10 = F.softmax(S01.T / temperature, dim=1)  # [N1, N0]
    entropy1 = -(P10 * (P10 + eps).log()).sum(dim=1)  # [N1]
    specificity1 = 1.0 - entropy1 / log_N0  # [N1]

    if mode == "v1":
        # V1: Original formula
        raw0 = existence0 * (0.5 + 0.5 * specificity0)  # [N0]
        raw1 = existence1 * (0.5 + 0.5 * specificity1)  # [N1]
        margin_gate0 = None
        margin_gate1 = None
        mutual0 = None
        mutual1 = None

    elif mode == "v2":
        # V2: Enhanced formula
        # Step 1: raw = existence * specificity^gamma
        raw0 = existence0 * (specificity0.pow(specificity_gamma))  # [N0]
        raw1 = existence1 * (specificity1.pow(specificity_gamma))  # [N1]

        # Step 2: Mutual nearest neighbor consistency
        mutual0, mutual1 = _mutual_nearest_consistency(S01, N0, N1)
        raw0 = raw0 * (0.5 + 0.5 * mutual0)
        raw1 = raw1 * (0.5 + 0.5 * mutual1)

        # Step 3: Margin gate
        margin_gate0, margin_gate1 = _margin_gate(
            S01, N0, N1, k_margin, q_margin_low, q_margin_high, eps
        )
        raw0 = raw0 * (0.5 + 0.5 * margin_gate0)
        raw1 = raw1 * (0.5 + 0.5 * margin_gate1)

    else:
        raise ValueError("Unknown mode: {}. Use 'v1' or 'v2'.".format(mode))

    # Percentile normalize: y_sem in [0, 1]
    y_sem0 = _percentile_normalize(raw0, q_low, q_high, eps)  # [N0]
    y_sem1 = _percentile_normalize(raw1, q_low, q_high, eps)  # [N1]

    # Confidence
    if mode == "v1":
        conf0 = specificity0 * (y_sem0 - 0.5).abs() * 2.0  # [N0]
        conf1 = specificity1 * (y_sem1 - 0.5).abs() * 2.0  # [N1]
    else:  # v2
        conf0 = (
            specificity0.pow(specificity_gamma)
            * margin_gate0
            * (y_sem0 - 0.5).abs()
            * 2.0
        )  # [N0]
        conf1 = (
            specificity1.pow(specificity_gamma)
            * margin_gate1
            * (y_sem1 - 0.5).abs()
            * 2.0
        )  # [N1]

    conf0 = torch.clamp(conf0, 0.0, 1.0)
    conf1 = torch.clamp(conf1, 0.0, 1.0)

    # Reshape to spatial grid
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

    # v2 only fields
    if mode == "v2":
        result["margin_gate0_clip"] = margin_gate0.view(Gh0, Gw0)
        result["margin_gate1_clip"] = margin_gate1.view(Gh1, Gw1)
        result["mutual0_clip"] = mutual0.view(Gh0, Gw0)
        result["mutual1_clip"] = mutual1.view(Gh1, Gw1)

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
    # Test v1: [576, 1024] + 24x24 grid
    # ------------------------------------------------------------------
    print("\n[Test v1-1] V1: CLIP-L/14@336: [576, 1024] + 24x24 grid")
    torch.manual_seed(42)
    Z0 = torch.randn(576, 1024)
    Z1 = torch.randn(576, 1024)
    Z0 = F.normalize(Z0, p=2, dim=-1)
    Z1 = F.normalize(Z1, p=2, dim=-1)

    result = compute_semantic_covisibility(Z0, Z1, (24, 24), (24, 24), mode="v1")

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result[key], (24, 24))
        assert (result[key] >= 0).all() and (result[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    print("  y_sem0_clip shape: {}".format(list(result["y_sem0_clip"].shape)))
    print("  y_sem0_clip range: [{:.4f}, {:.4f}]".format(
        float(result["y_sem0_clip"].min()), float(result["y_sem0_clip"].max())))
    print("  conf0_clip range:  [{:.4f}, {:.4f}]".format(
        float(result["conf0_clip"].min()), float(result["conf0_clip"].max())))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test v1 with batch dim
    # ------------------------------------------------------------------
    print("\n[Test v1-2] V1 with batch dim: [1, 576, 1024] + 24x24")
    torch.manual_seed(42)
    Z0_b = torch.randn(1, 576, 1024)
    Z1_b = torch.randn(1, 576, 1024)
    Z0_b = F.normalize(Z0_b, p=2, dim=-1)
    Z1_b = F.normalize(Z1_b, p=2, dim=-1)

    result2 = compute_semantic_covisibility(Z0_b, Z1_b, (24, 24), (24, 24), mode="v1")

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result2[key], (24, 24))
        assert (result2[key] >= 0).all() and (result2[key] <= 1).all()

    print("  y_sem0_clip shape: {}".format(list(result2["y_sem0_clip"].shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test v2: [576, 1024] + 24x24 grid
    # ------------------------------------------------------------------
    print("\n[Test v2-1] V2: CLIP-L/14@336: [576, 1024] + 24x24 grid")
    torch.manual_seed(42)
    Z0 = torch.randn(576, 1024)
    Z1 = torch.randn(576, 1024)
    Z0 = F.normalize(Z0, p=2, dim=-1)
    Z1 = F.normalize(Z1, p=2, dim=-1)

    result_v2 = compute_semantic_covisibility(
        Z0, Z1, (24, 24), (24, 24),
        mode="v2",
        specificity_gamma=1.5,
        k_margin=10,
        q_margin_low=30,
        q_margin_high=80,
    )

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result_v2[key], (24, 24))
        assert (result_v2[key] >= 0).all() and (result_v2[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    for key in ["margin_gate0_clip", "margin_gate1_clip",
                "mutual0_clip", "mutual1_clip"]:
        _check_tensor(key, result_v2[key], (24, 24))
        assert (result_v2[key] >= 0).all() and (result_v2[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    print("  y_sem0_clip shape: {}".format(list(result_v2["y_sem0_clip"].shape)))
    print("  y_sem0_clip range: [{:.4f}, {:.4f}]".format(
        float(result_v2["y_sem0_clip"].min()), float(result_v2["y_sem0_clip"].max())))
    print("  conf0_clip range:  [{:.4f}, {:.4f}]".format(
        float(result_v2["conf0_clip"].min()), float(result_v2["conf0_clip"].max())))
    print("  margin_gate0 range: [{:.4f}, {:.4f}]".format(
        float(result_v2["margin_gate0_clip"].min()),
        float(result_v2["margin_gate0_clip"].max())))
    print("  mutual0 mean: {:.4f}".format(float(result_v2["mutual0_clip"].mean())))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test v2 with batch dim
    # ------------------------------------------------------------------
    print("\n[Test v2-2] V2 with batch dim: [1, 576, 1024] + 24x24")
    torch.manual_seed(42)
    Z0_b = torch.randn(1, 576, 1024)
    Z1_b = torch.randn(1, 576, 1024)
    Z0_b = F.normalize(Z0_b, p=2, dim=-1)
    Z1_b = F.normalize(Z1_b, p=2, dim=-1)

    result_v2b = compute_semantic_covisibility(
        Z0_b, Z1_b, (24, 24), (24, 24), mode="v2"
    )

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result_v2b[key], (24, 24))
        assert (result_v2b[key] >= 0).all() and (result_v2b[key] <= 1).all()

    print("  y_sem0_clip shape: {}".format(list(result_v2b["y_sem0_clip"].shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test v2: CLIP-B/16 [196, 768] + 14x14
    # ------------------------------------------------------------------
    print("\n[Test v2-3] V2: CLIP-B/16: [196, 768] + 14x14 grid")
    torch.manual_seed(42)
    Z0_s = torch.randn(196, 768)
    Z1_s = torch.randn(196, 768)
    Z0_s = F.normalize(Z0_s, p=2, dim=-1)
    Z1_s = F.normalize(Z1_s, p=2, dim=-1)

    result3 = compute_semantic_covisibility(Z0_s, Z1_s, (14, 14), (14, 14), mode="v2")

    for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
        _check_tensor(key, result3[key], (14, 14))
        assert (result3[key] >= 0).all() and (result3[key] <= 1).all(), \
            "{}: values out of [0, 1]".format(key)

    print("  y_sem0_clip shape: {}".format(list(result3["y_sem0_clip"].shape)))
    print("  conf0_clip range:  [{:.4f}, {:.4f}]".format(
        float(result3["conf0_clip"].min()), float(result3["conf0_clip"].max())))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test v2: Asymmetric grids
    # ------------------------------------------------------------------
    print("\n[Test v2-4] V2: Asymmetric grids: Z0 [480, 256] + 20x24, Z1 [360, 256] + 15x24")
    torch.manual_seed(42)
    Z0_a = F.normalize(torch.randn(480, 256), p=2, dim=-1)
    Z1_a = F.normalize(torch.randn(360, 256), p=2, dim=-1)

    result4 = compute_semantic_covisibility(Z0_a, Z1_a, (20, 24), (15, 24), mode="v2")

    _check_tensor("y_sem0_clip", result4["y_sem0_clip"], (20, 24))
    _check_tensor("y_sem1_clip", result4["y_sem1_clip"], (15, 24))
    _check_tensor("conf0_clip", result4["conf0_clip"], (20, 24))
    _check_tensor("conf1_clip", result4["conf1_clip"], (15, 24))
    print("  y_sem0_clip shape: {}".format(list(result4["y_sem0_clip"].shape)))
    print("  y_sem1_clip shape: {}".format(list(result4["y_sem1_clip"].shape)))
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test v2: GPU
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        print("\n[Test v2-5] V2 GPU: [1, 576, 1024] + 24x24 on CUDA")
        torch.manual_seed(42)
        Z0_gpu = F.normalize(torch.randn(1, 576, 1024), p=2, dim=-1).cuda()
        Z1_gpu = F.normalize(torch.randn(1, 576, 1024), p=2, dim=-1).cuda()

        result5 = compute_semantic_covisibility(
            Z0_gpu, Z1_gpu, (24, 24), (24, 24), mode="v2"
        )

        for key in ["y_sem0_clip", "y_sem1_clip", "conf0_clip", "conf1_clip"]:
            _check_tensor(key, result5[key], (24, 24), device="cuda")
            assert (result5[key] >= 0).all() and (result5[key] <= 1).all()

        print("  y_sem0_clip device: {}".format(result5["y_sem0_clip"].device))
        print("  [PASS]")
    else:
        print("\n[Test v2-5] V2 GPU: SKIPPED (no CUDA)")

    # ------------------------------------------------------------------
    # Test v2: Grid mismatch assertion
    # ------------------------------------------------------------------
    print("\n[Test v2-6] V2 Grid mismatch assertion")
    try:
        Z0_m = F.normalize(torch.randn(576, 1024), p=2, dim=-1)
        Z1_m = F.normalize(torch.randn(576, 1024), p=2, dim=-1)
        compute_semantic_covisibility(Z0_m, Z1_m, (20, 20), (24, 24), mode="v2")
        print("  [FAIL] Should have raised AssertionError")
        all_pass = False
    except AssertionError:
        print("  [PASS] Correctly raised AssertionError")

    # ------------------------------------------------------------------
    # Test v2: Invalid mode
    # ------------------------------------------------------------------
    print("\n[Test v2-7] V2 Invalid mode assertion")
    try:
        Z0_e = F.normalize(torch.randn(576, 1024), p=2, dim=-1)
        Z1_e = F.normalize(torch.randn(576, 1024), p=2, dim=-1)
        compute_semantic_covisibility(Z0_e, Z1_e, (24, 24), (24, 24), mode="v3")
        print("  [FAIL] Should have raised ValueError")
        all_pass = False
    except ValueError:
        print("  [PASS] Correctly raised ValueError")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests passed")
        print("  v1 y_sem0 range:  [{:.4f}, {:.4f}]".format(
            float(result["y_sem0_clip"].min()), float(result["y_sem0_clip"].max())))
        print("  v2 y_sem0 range:  [{:.4f}, {:.4f}]".format(
            float(result_v2["y_sem0_clip"].min()), float(result_v2["y_sem0_clip"].max())))
        print("  v2 conf0 range:  [{:.4f}, {:.4f}]".format(
            float(result_v2["conf0_clip"].min()), float(result_v2["conf0_clip"].max())))
    else:
        print("Some tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
