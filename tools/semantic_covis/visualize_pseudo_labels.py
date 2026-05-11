#!/usr/bin/env python3
"""
Visualize CLIP Semantic Covisibility Pseudo-labels.

Reads npz files from generate_pseudo_labels.py output and overlays
y_sem/conf heatmaps on CoMatch-style RGB images.

Usage:
    python tools/semantic_covis/visualize_pseudo_labels.py \
        --label-dir outputs/sem_covis_labels_debug \
        --output-dir outputs/sem_covis_vis_debug \
        --max-vis 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Add tools/semantic_covis to import path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from image_preprocess import read_rgb_for_clip_and_comatch


def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(
        description="Visualize CLIP semantic covisibility pseudo-labels"
    )
    parser.add_argument(
        "--label-dir", type=str,
        default="outputs/sem_covis_labels_debug",
        help="Directory containing npz files from generate_pseudo_labels.py",
    )
    parser.add_argument(
        "--index", type=str, default=None,
        help="Path to index.jsonl (default: label-dir/index.jsonl)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/sem_covis_vis_debug",
        help="Output directory for visualization images",
    )
    parser.add_argument("--max-vis", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--show-debug", action="store_true",
                        help="Show CSLS debug fields (E0/E1, C0/C1) if available")
    return parser.parse_args()


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    # type: (np.ndarray, int) -> np.ndarray
    """Convert a [0, 1] heatmap to RGB using OpenCV colormap.

    Args:
        heatmap: Float32 array [H, W] in range [0, 1]
        colormap: OpenCV colormap constant

    Returns:
        Uint8 RGB array [H, W, 3]
    """
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(rgb, heatmap, alpha, valid_mask=None):
    # type: (np.ndarray, np.ndarray, float, np.ndarray) -> np.ndarray
    """Overlay a heatmap on an RGB image.

    Args:
        rgb: RGB image [H, W, 3], uint8
        heatmap: Float32 heatmap [H, W] in range [0, 1]
        alpha: Overlay transparency (0=only image, 1=only heatmap)
        valid_mask: Boolean mask [H, W], True=valid region

    Returns:
        Blended RGB image [H, W, 3], uint8
    """
    heatmap_rgb = apply_colormap(heatmap)
    blended = cv2.addWeighted(rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

    # Set padding region to dark gray
    if valid_mask is not None:
        blended[~valid_mask] = [40, 40, 40]

    return blended


def resize_heatmap(heatmap, target_h, target_w):
    # type: (np.ndarray, int, int) -> np.ndarray
    """Resize heatmap from coarse to target size.

    Args:
        heatmap: Float array [H, W]
        target_h: Target height
        target_w: Target width

    Returns:
        Resized float array [target_h, target_w]
    """
    return cv2.resize(heatmap.astype(np.float32), (target_w, target_h),
                      interpolation=cv2.INTER_LINEAR)


def add_text_annotations(img, pair_id, stats):
    # type: (np.ndarray, str, Dict[str, Any]) -> np.ndarray
    """Add text annotations to top-left corner of image.

    Args:
        img: RGB image [H, W, 3]
        pair_id: Pair identifier
        stats: Dict with stat keys

    Returns:
        Annotated image
    """
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    y_offset = 20
    line_height = 18

    lines = [
        pair_id[:24],
        "y_m={:.3f} y_x={:.3f}".format(
            stats.get("y_mean", 0), stats.get("y_max", 0)),
        "c_m={:.3f} c_x={:.3f}".format(
            stats.get("conf_mean", 0), stats.get("conf_max", 0)),
        "e_m={:.3f} e_x={:.3f}".format(
            stats.get("eff_mean", 0), stats.get("eff_max", 0)),
        "e_hi={:.3f} hi={:.3f}".format(
            stats.get("eff_high_ratio", 0), stats.get("high_ratio", 0)),
    ]

    for i, line in enumerate(lines):
        y = y_offset + i * line_height
        # Get text size for background
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(img, (4, y - th - 2), (8 + tw, y + 4), bg_color, -1)
        cv2.putText(img, line, (6, y), font, font_scale, color, thickness,
                    cv2.LINE_AA)

    return img


def _compute_stats(heatmap, mask):
    # type: (np.ndarray, np.ndarray) -> Dict[str, float]
    """Compute summary stats for a heatmap within a valid region."""
    if mask is not None:
        valid = heatmap[mask]
    else:
        valid = heatmap.flatten()
    if len(valid) == 0:
        return {"y_mean": 0, "y_max": 0}
    return {
        "y_mean": float(valid.mean()),
        "y_max": float(valid.max()),
    }


def _compute_effective_stats(effective, mask):
    # type: (np.ndarray, np.ndarray) -> Dict[str, float]
    """Compute effective = y_sem * conf stats."""
    if mask is not None:
        valid = effective[mask]
    else:
        valid = effective.flatten()
    if len(valid) == 0:
        return {"eff_mean": 0, "eff_max": 0, "eff_high_ratio": 0}
    return {
        "eff_mean": float(valid.mean()),
        "eff_max": float(valid.max()),
        "eff_high_ratio": float((valid > 0.3).mean()),
    }


def create_canvas(
    rgb0,          # type: np.ndarray
    rgb1,          # type: np.ndarray
    y_sem0_full,   # type: np.ndarray
    y_sem1_full,   # type: np.ndarray
    conf0_full,    # type: np.ndarray
    conf1_full,    # type: np.ndarray
    valid_mask0,   # type: np.ndarray
    valid_mask1,   # type: np.ndarray
    pair_id,       # type: str
    alpha,         # type: float
):
    # type: (...) -> np.ndarray
    """Create a 2x4 canvas for visualization.

    Layout:
        Row 0: [RGB0] [y_sem0 overlay] [conf0 overlay] [effective0 overlay]
        Row 1: [RGB1] [y_sem1 overlay] [conf1 overlay] [effective1 overlay]

    Returns:
        Canvas image [2*H, 4*W, 3]
    """
    # Compute effective maps
    eff0_full = y_sem0_full * conf0_full
    eff1_full = y_sem1_full * conf1_full

    # Compute stats
    y_stats0 = _compute_stats(y_sem0_full, valid_mask0)
    y_stats1 = _compute_stats(y_sem1_full, valid_mask1)
    c_stats0 = _compute_stats(conf0_full, valid_mask0)
    c_stats1 = _compute_stats(conf1_full, valid_mask1)
    e_stats0 = _compute_effective_stats(eff0_full, valid_mask0)
    e_stats1 = _compute_effective_stats(eff1_full, valid_mask1)

    # Merge stats for annotation
    def merge_stats(*dicts):
        # type: (*Dict[str, float]) -> Dict[str, float]
        out = {}  # type: Dict[str, float]
        for d in dicts:
            out.update(d)
        return out

    anno_y0 = merge_stats(y_stats0, c_stats0, e_stats0,
                          {"high_ratio": float((y_sem0_full[valid_mask0] > 0.6).mean()) if valid_mask0.any() else 0})
    anno_y1 = merge_stats(y_stats1, c_stats1, e_stats1,
                          {"high_ratio": float((y_sem1_full[valid_mask1] > 0.6).mean()) if valid_mask1.any() else 0})

    # Build panels
    panels = [
        # Row 0: image 0
        add_text_annotations(rgb0, pair_id + " img0", anno_y0),
        add_text_annotations(
            overlay_heatmap(rgb0, y_sem0_full, alpha, valid_mask0),
            "y_sem0", anno_y0),
        add_text_annotations(
            overlay_heatmap(rgb0, conf0_full, alpha, valid_mask0),
            "conf0", anno_y0),
        add_text_annotations(
            overlay_heatmap(rgb0, eff0_full, alpha, valid_mask0),
            "eff0=y*c", anno_y0),
        # Row 1: image 1
        add_text_annotations(rgb1, pair_id + " img1", anno_y1),
        add_text_annotations(
            overlay_heatmap(rgb1, y_sem1_full, alpha, valid_mask1),
            "y_sem1", anno_y1),
        add_text_annotations(
            overlay_heatmap(rgb1, conf1_full, alpha, valid_mask1),
            "conf1", anno_y1),
        add_text_annotations(
            overlay_heatmap(rgb1, eff1_full, alpha, valid_mask1),
            "eff1=y*c", anno_y1),
    ]

    # Arrange into 2x4 grid
    row0 = np.concatenate([panels[0], panels[1], panels[2], panels[3]], axis=1)
    row1 = np.concatenate([panels[4], panels[5], panels[6], panels[7]], axis=1)
    canvas = np.concatenate([row0, row1], axis=0)

    return canvas


def process_pair_vis(
    npz_path,      # type: Path
    output_dir,    # type: Path
    alpha,         # type: float
    show_debug=False,  # type: bool
):
    # type: (...) -> Dict[str, Any]
    """Process one pair and generate visualization canvas.

    Returns dict with vis info or error.
    """
    # Load npz
    data = np.load(str(npz_path), allow_pickle=True)

    pair_id = str(data["pair_id"])
    img0_path = str(data["image0_abs_path"])
    img1_path = str(data["image1_abs_path"])

    # Check images exist
    if not os.path.isfile(img0_path):
        return {"pair_id": pair_id, "status": "failed",
                "error": "image0 not found: {}".format(img0_path)}
    if not os.path.isfile(img1_path):
        return {"pair_id": pair_id, "status": "failed",
                "error": "image1 not found: {}".format(img1_path)}

    # Load and preprocess images
    pp0 = read_rgb_for_clip_and_comatch(img0_path, long_edge=832, df=8,
                                         pad_to_square=True, return_pil=False)
    pp1 = read_rgb_for_clip_and_comatch(img1_path, long_edge=832, df=8,
                                         pad_to_square=True, return_pil=False)

    rgb0 = pp0["image"].astype(np.uint8)  # [832, 832, 3]
    rgb1 = pp1["image"].astype(np.uint8)
    valid_mask0 = pp0["valid_mask"]  # [832, 832]
    valid_mask1 = pp1["valid_mask"]

    h, w = rgb0.shape[:2]  # 832, 832

    # Read pseudo-labels
    y_sem0 = data["y_sem0"].astype(np.float32)  # [104, 104]
    y_sem1 = data["y_sem1"].astype(np.float32)
    conf0 = data["conf0"].astype(np.float32)
    conf1 = data["conf1"].astype(np.float32)

    # Resize heatmaps from 104x104 to 832x832
    y_sem0_full = resize_heatmap(y_sem0, h, w)
    y_sem1_full = resize_heatmap(y_sem1, h, w)
    conf0_full = resize_heatmap(conf0, h, w)
    conf1_full = resize_heatmap(conf1, h, w)

    # Create canvas
    canvas = create_canvas(
        rgb0, rgb1,
        y_sem0_full, y_sem1_full,
        conf0_full, conf1_full,
        valid_mask0, valid_mask1,
        pair_id, alpha,
    )

    # Save canvas
    canvas_path = output_dir / "{}_canvas.jpg".format(pair_id)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(canvas_path), canvas_bgr)

    # Debug panels for clip_csls_v1
    debug_paths = []  # type: list
    if show_debug:
        csls_debug_keys = [
            ("csls_E0", "csls_E1", "E_existence"),
            ("csls_C0", "csls_C1", "C_consistency"),
        ]
        for key0, key1, label in csls_debug_keys:
            if key0 in data and key1 in data:
                d0 = data[key0].astype(np.float32)
                d1 = data[key1].astype(np.float32)
                # These are 1D at clip-grid level; reshape if needed
                clip_grid0 = tuple(data["clip_grid0"].astype(int))
                clip_grid1 = tuple(data["clip_grid1"].astype(int))
                if d0.ndim == 1 and d0.shape[0] == clip_grid0[0] * clip_grid0[1]:
                    d0 = d0.reshape(clip_grid0)
                if d1.ndim == 1 and d1.shape[0] == clip_grid1[0] * clip_grid1[1]:
                    d1 = d1.reshape(clip_grid1)
                d0_full = resize_heatmap(d0, h, w)
                d1_full = resize_heatmap(d1, h, w)
                dbg_canvas = np.concatenate([
                    overlay_heatmap(rgb0, d0_full, alpha, valid_mask0),
                    overlay_heatmap(rgb1, d1_full, alpha, valid_mask1),
                ], axis=1)
                dbg_path = output_dir / "{}_debug_{}.jpg".format(pair_id, label)
                cv2.imwrite(str(dbg_path), cv2.cvtColor(dbg_canvas, cv2.COLOR_RGB2BGR))
                debug_paths.append(str(dbg_path))

    # Compute effective maps
    eff0_full = y_sem0_full * conf0_full
    eff1_full = y_sem1_full * conf1_full

    # Compute stats
    vm0 = valid_mask0
    vm1 = valid_mask1
    stats = {
        "pair_id": pair_id,
        "y_sem0_mean": float(y_sem0_full[vm0].mean()) if vm0.any() else 0,
        "y_sem1_mean": float(y_sem1_full[vm1].mean()) if vm1.any() else 0,
        "y_sem0_max": float(y_sem0_full[vm0].max()) if vm0.any() else 0,
        "y_sem1_max": float(y_sem1_full[vm1].max()) if vm1.any() else 0,
        "conf0_mean": float(conf0_full[vm0].mean()) if vm0.any() else 0,
        "conf1_mean": float(conf1_full[vm1].mean()) if vm1.any() else 0,
        "high_ratio0": float((y_sem0_full[vm0] > 0.6).mean()) if vm0.any() else 0,
        "high_ratio1": float((y_sem1_full[vm1] > 0.6).mean()) if vm1.any() else 0,
        "eff0_mean": float(eff0_full[vm0].mean()) if vm0.any() else 0,
        "eff1_mean": float(eff1_full[vm1].mean()) if vm1.any() else 0,
        "eff0_max": float(eff0_full[vm0].max()) if vm0.any() else 0,
        "eff1_max": float(eff1_full[vm1].max()) if vm1.any() else 0,
        "eff0_high_ratio": float((eff0_full[vm0] > 0.3).mean()) if vm0.any() else 0,
        "eff1_high_ratio": float((eff1_full[vm1] > 0.3).mean()) if vm1.any() else 0,
        "canvas_path": str(canvas_path),
        "status": "ok",
        "error": "",
        "debug_paths": debug_paths,
    }

    return stats


def main():
    # type: () -> int
    args = parse_args()

    print("=" * 60)
    print("Pseudo-label Visualization")
    print("=" * 60)
    print("  label-dir:  {}".format(args.label_dir))
    print("  output-dir: {}".format(args.output_dir))
    print("  max-vis:    {}".format(args.max_vis))
    print("  alpha:      {}".format(args.alpha))
    print("  show-debug: {}".format(args.show_debug))
    print("=" * 60)

    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine index path
    if args.index is not None:
        index_path = Path(args.index)
    else:
        index_path = label_dir / "index.jsonl"

    # Read index
    entries = []  # type: List[Dict[str, Any]]
    with open(str(index_path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print("\nLoaded {} entries from {}".format(len(entries), index_path))

    # Limit
    if args.max_vis > 0:
        entries = entries[:args.max_vis]
        print("Limited to {} entries".format(len(entries)))

    # Process each pair
    vis_results = []  # type: List[Dict[str, Any]]
    ok_count = 0
    fail_count = 0

    for i, entry in enumerate(entries):
        pair_id = entry.get("pair_id", "unknown")
        npz_path = label_dir / "{}.npz".format(pair_id)

        print("\n[{}/{}] Processing: {}".format(i + 1, len(entries), pair_id))

        if not npz_path.exists():
            print("  SKIP: npz not found: {}".format(npz_path))
            vis_results.append({
                "pair_id": pair_id, "status": "failed",
                "error": "npz not found", "canvas_path": "",
            })
            fail_count += 1
            continue

        try:
            result = process_pair_vis(npz_path, output_dir, args.alpha,
                                      show_debug=args.show_debug)
            vis_results.append(result)

            if result["status"] == "ok":
                ok_count += 1
                print("  OK -> {}".format(result["canvas_path"]))
                print("    y_sem0: mean={:.3f}, max={:.3f}".format(
                    result["y_sem0_mean"], result["y_sem0_max"]))
                print("    conf0:  mean={:.3f}".format(result["conf0_mean"]))
                print("    high0:  {:.3f}".format(result["high_ratio0"]))
                print("    eff0:   mean={:.3f}, max={:.3f}, high={:.3f}".format(
                    result["eff0_mean"], result["eff0_max"],
                    result["eff0_high_ratio"]))
            else:
                fail_count += 1
                print("  FAILED: {}".format(result["error"]))

        except Exception as e:
            vis_results.append({
                "pair_id": pair_id, "status": "failed",
                "error": str(e), "canvas_path": "",
            })
            fail_count += 1
            print("  ERROR: {}".format(str(e)))

    # Write summary.md
    summary_path = output_dir / "summary.md"
    with open(str(summary_path), "w") as f:
        f.write("# Pseudo-label Visualization Summary\n\n")
        f.write("Generated from: {}\n\n".format(args.label_dir))
        f.write("| # | pair_id | y_sem0_mean | y_sem0_max | conf0_mean | high_ratio0 | eff0_mean | eff0_max | eff0_high_ratio | canvas |\n")
        f.write("|---|---------|-------------|------------|------------|-------------|-----------|----------|-----------------|--------|\n")
        for i, r in enumerate(vis_results):
            if r["status"] == "ok":
                f.write("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | [canvas]({}) |\n".format(
                    i + 1, r["pair_id"][:16],
                    r["y_sem0_mean"], r["y_sem0_max"],
                    r["conf0_mean"], r["high_ratio0"],
                    r["eff0_mean"], r["eff0_max"], r["eff0_high_ratio"],
                    r["canvas_path"].replace("\\", "/"),
                ))
            else:
                f.write("| {} | {} | - | - | - | - | - | - | - | FAILED: {} |\n".format(
                    i + 1, r["pair_id"][:16], r.get("error", "unknown")))
        f.write("\n**Total: {} ok, {} failed**\n".format(ok_count, fail_count))

    # Final summary
    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print("  Total:   {}".format(len(entries)))
    print("  OK:      {}".format(ok_count))
    print("  Failed:  {}".format(fail_count))
    print("  Output:  {}".format(output_dir))
    print("  Summary: {}".format(summary_path))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
