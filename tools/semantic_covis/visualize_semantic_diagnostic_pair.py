#!/usr/bin/env python3
"""
Visualize semantic diagnostic results for selected pairs or all pairs.

This script creates visualization images showing:
1. Left-right image concatenation with match lines
2. Color-coded by semantic consistency (green/red/orange)
3. OneFormer segmentation overlay
4. Statistics and pair info

Usage (target pairs):
    python tools/semantic_covis/visualize_semantic_diagnostic_pair.py \
        --per-match-jsonl outputs/semantic_diagnostic_outdoor_vis/per_match_semantic_diagnostic.jsonl \
        --pair-list outputs/semantic_diagnostic_outdoor/top_coarse_error_pairs.jsonl \
        --image-root /ssd-data3/zh2025/datasets/MegaDepth \
        --oneformer-dir outputs/oneformer_outdoor_test \
        --output-dir outputs/semantic_diagnostic_visualizations \
        --max-lines 500 \
        --overlay-alpha 0.45

Usage (all pairs - no pair-list needed):
    python tools/semantic_covis/visualize_semantic_diagnostic_pair.py \
        --per-match-jsonl outputs/semantic_diagnostic_outdoor_vis/per_match_semantic_diagnostic.jsonl \
        --image-root /ssd-data3/zh2025/datasets/MegaDepth \
        --oneformer-dir outputs/oneformer_outdoor_test \
        --output-dir outputs/semantic_diagnostic_visualizations_full \
        --max-lines 500 \
        --overlay-alpha 0.45 \
        --max-pairs 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Color Definitions
# =============================================================================

# Color palette for semantic labels
LABEL_COLORS = [
    (230, 25, 33), (60, 180, 75), (255, 225, 25), (0, 130, 200), (145, 30, 180),
    (70, 240, 240), (240, 32, 160), (0, 0, 255), (255, 255, 255), (0, 0, 0),
    (128, 128, 128), (100, 100, 100), (200, 200, 200), (255, 200, 150),
    (150, 100, 50), (100, 50, 0), (50, 150, 200), (200, 50, 50),
    (50, 200, 50), (150, 200, 100), (100, 150, 200), (200, 100, 150),
    (50, 100, 200), (150, 50, 100), (100, 200, 150), (200, 150, 100),
]

# Match line colors
COLOR_COARSE_CONSISTENT = (0, 200, 0)       # Green
COLOR_COARSE_ERROR = (0, 0, 255)             # Red
COLOR_FINE_ERROR_COARSE_OK = (0, 180, 255)   # Orange (fine error but coarse correct)


# =============================================================================
# ADE20K Label Mapping
# =============================================================================

DEFAULT_ID2LABEL = {
    0: "wall", 1: "building, edifice", 2: "sky", 3: "floor, flooring",
    4: "tree", 5: "ceiling", 6: "road, route", 7: "grass", 8: "person",
    9: "earth, ground", 10: "mountain, mount", 11: "plant", 12: "car",
    13: "panel", 14: "screen door", 15: "bridge", 16: "bookcase",
    17: "blind", 18: "door", 19: "table", 20: "chair", 21: "cabinet",
    22: "lamp", 23: "climbing", 24: "escalator", 25: "grandstand",
    26: "stage", 27: "railroad", 28: "sidewalk, pavement", 29: "swimming pool",
    30: "wall", 31: "railing", 32: "lamp", 33: "bridge", 34: "bookcase",
    35: "television", 36: "computer", 37: "bed", 38: "couch",
    39: "chair", 40: "curtain", 41: "lamp", 42: "counter", 43: "shelf",
    44: "sink", 45: "stairs", 46: "shower", 47: "refrigerator",
    48: " oven", 49: "toilet", 50: "boiler", 51: "fireplace",
    52: "fridge", 53: "washing machine", 54: "mirror", 55: "whiteboard",
    56: "skyscraper", 57: "house", 58: "tower", 59: "tent",
    60: "rock", 61: "stairs", 62: "fence", 63: "road", 64: "sidewalk",
    65: "parking", 66: "curb", 67: "crosswalk", 68: "floor", 69: "playground",
    70: "football field", 71: "pitch", 72: "track", 73: "court",
    74: "base", 75: "building", 76: "bridge", 77: "fence", 78: "road",
    79: "earth", 80: "hill", 81: "mountain", 82: "rock", 83: "tree",
    84: "forest", 85: "sand", 86: "field", 87: "gravel", 88: "pavement",
    89: "sidewalk", 90: "earth", 91: "grass", 92: "plant", 93: "flower",
    94: "rock", 95: "gravel", 96: "tree", 97: "bush", 98: "hill",
    99: "mountain", 100: "sky", 101: "grass", 102: "tree", 103: "mountain",
    104: "sky", 105: "building", 106: "house", 107: "tent", 108: "fence",
    109: "road", 110: "sidewalk", 111: "floor", 112: "sky", 113: "grass",
    114: "tree", 115: "mountain", 116: "water", 117: "person", 118: "car",
    119: "bicycle", 120: "motorbike", 121: "bus", 122: "truck", 123: "train",
    124: "boat", 125: "sky", 126: "grass", 127: "tree", 128: "mountain",
    129: "water", 130: "building", 131: "house", 132: "tent", 133: "fence",
    134: "road", 135: "sidewalk", 136: "floor", 137: "sky", 138: "grass",
    139: "tree", 140: "mountain", 141: "water", 142: "person", 143: "car",
    144: "bicycle", 145: "motorbike", 146: "bus", 147: "truck", 148: "train",
    149: "boat", 150: "sky", 151: "grass", 152: "tree", 153: "mountain",
    154: "water", 155: "building", 156: "house", 157: "tent", 158: "fence",
    159: "road", 160: "sidewalk", 161: "floor", 162: "sky", 163: "grass",
    164: "tree", 165: "mountain", 166: "water", 167: "person", 168: "car",
    169: "bicycle", 170: "motorbike", 171: "bus", 172: "truck", 173: "train",
    174: "boat", 175: "sky", 176: "grass", 177: "tree", 178: "mountain",
    179: "water", 180: "building", 181: "house", 182: "tent", 183: "fence",
    184: "road", 185: "sidewalk", 186: "floor", 187: "sky", 188: "grass",
    189: "tree", 190: "mountain", 191: "water", 192: "person", 193: "car",
    194: "bicycle", 195: "motorbike", 196: "bus", 197: "truck", 198: "train",
    199: "boat", 200: "sky", 201: "grass", 202: "tree", 203: "mountain",
    204: "water", 205: "building", 206: "house", 207: "tent", 208: "fence",
    209: "road", 210: "sidewalk", 211: "floor", 212: "sky", 213: "grass",
    214: "tree", 215: "mountain", 216: "water", 217: "person", 218: "car",
    219: "bicycle", 220: "motorbike", 221: "bus", 222: "truck", 223: "train",
    224: "boat", 225: "sky", 226: "grass", 227: "tree", 228: "mountain",
    229: "water", 230: "building", 231: "house", 232: "tent", 233: "fence",
    234: "road", 235: "sidewalk", 236: "floor", 237: "sky", 238: "grass",
    239: "tree", 240: "mountain", 241: "water", 242: "person", 243: "car",
    244: "bicycle", 245: "motorbike", 246: "bus", 247: "truck", 248: "train",
    249: "boat", 250: "sky", 251: "grass", 252: "tree", 253: "mountain",
    254: "water", 255: "void",
}


# =============================================================================
# Utility Functions
# =============================================================================

def load_label_map(oneformer_dir: Path) -> Dict[int, str]:
    """Load id2label mapping from OneFormer output directory."""
    sources = [
        oneformer_dir / "label_id_map.json",
        oneformer_dir / "id2label.json",
        oneformer_dir / "ade20k_panoptic.json",
    ]

    for path in sources:
        if not path.exists():
            continue
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            return parse_label_map(raw)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

    # Fallback to default
    return DEFAULT_ID2LABEL.copy()


def parse_label_map(raw) -> Dict[int, str]:
    """Parse various label map formats."""
    if isinstance(raw, dict):
        # Format 1: {"0": "building", "1": "road"}
        if all(isinstance(v, str) for v in raw.values()):
            return {int(k): v for k, v in raw.items()}

        # Format 2: {"0": {"name": "building"}, ...}
        result = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                if 'name' in v:
                    result[int(k)] = v['name']
                elif 'label' in v:
                    result[int(k)] = v['label']
        if result:
            return result

    return {}


def load_oneformer_segmentation(image_path: str, oneformer_dir: Path) -> Optional[np.ndarray]:
    """Load OneFormer segmentation for an image.

    Args:
        image_path: Relative or absolute path to image
        oneformer_dir: Path to OneFormer output directory

    Returns:
        [H, W] numpy array of label IDs, or None if not found
    """
    npz_dir = oneformer_dir / "npz"

    # Handle list-wrapped paths like "['path.jpg']"
    path_str = str(image_path).strip()
    if path_str.startswith("[") and path_str.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(path_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                path_str = str(parsed[0])
        except Exception:
            pass

    # Build safe stem from image path
    p = Path(path_str)
    rel_stem = p.with_suffix("")
    safe_stem = rel_stem.as_posix().replace("/", "_").replace("\\", "_")

    npz_path = npz_dir / f"{safe_stem}.npz"

    # Try alternative if not found
    if not npz_path.exists():
        # Try with different path combinations
        for stem_attempt in [p.stem, rel_stem.name]:
            alt_path = npz_dir / f"{stem_attempt}.npz"
            if alt_path.exists():
                npz_path = alt_path
                break

    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        # Find the label array - try different keys
        for key in ['semantic_label', 'label_map', 'segmentation', 'panoptic_seg', 'pred']:
            if key in data:
                return data[key]
        # If only one array, use it
        if len(data.files) == 1:
            return data[data.files[0]]
        return None
    except Exception as e:
        print(f"Failed to load {npz_path}: {e}")
        return None


def load_image(image_path: str, image_root: str = None) -> Optional[np.ndarray]:
    """Load an image as numpy array."""
    if not HAS_CV2:
        return None

    # Handle list-wrapped paths like "['path.jpg']"
    path_str = str(image_path).strip()
    if path_str.startswith("[") and path_str.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(path_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                path_str = str(parsed[0])
        except Exception:
            pass

    path = Path(path_str)
    if not path.is_absolute() and image_root:
        path = Path(image_root) / path

    if not path.exists():
        return None

    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def blend_overlay(base_img: np.ndarray, seg_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend segmentation overlay onto base image."""
    if seg_map is None or base_img is None:
        return base_img

    h, w = base_img.shape[:2]
    seg_h, seg_w = seg_map.shape[:2]

    # Resize seg_map to match base image
    if seg_h != h or seg_w != w:
        seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create colored overlay
    overlay = np.zeros_like(base_img)

    unique_labels = np.unique(seg_map)
    for label in unique_labels:
        if label < 0 or label >= len(LABEL_COLORS):
            color = (128, 128, 128)  # gray for unknown
        else:
            color = LABEL_COLORS[label % len(LABEL_COLORS)]

        mask = seg_map == label
        overlay[mask] = color

    # Blend
    result = cv2.addWeighted(base_img, 1.0, overlay, alpha, 0)
    return result


# =============================================================================
# Match Line Visualization
# =============================================================================

def draw_match_lines(
    img_left: np.ndarray,
    img_right: np.ndarray,
    matches: List[Dict],
    max_lines: int = 500,
    error_only: bool = False,
    seg0_hw: Tuple[int, int] = None,
    seg1_hw: Tuple[int, int] = None,
) -> np.ndarray:
    """Draw match lines on concatenated left-right image.

    Coordinates (seg_x0/y0/x1/y1) are in segmentation space.
    Images are resized to match segmentation dimensions before drawing,
    so match lines align correctly.

    Args:
        img_left: [H, W, 3] left image (original resolution)
        img_right: [H, W, 3] right image (original resolution)
        matches: List of match dicts with seg_x0, seg_y0, seg_x1, seg_y1
        max_lines: Maximum number of lines to draw
        error_only: If True, only draw coarse error lines
        seg0_hw: Segmentation size (H, W) for left image
        seg1_hw: Segmentation size (H, W) for right image

    Returns:
        Concatenated image with match lines drawn
    """
    if not HAS_CV2:
        return np.zeros((max(img_left.shape[0], img_right.shape[0]),
                        img_left.shape[1] + img_right.shape[1], 3), dtype=np.uint8)

    # Resize images to segmentation dimensions so coordinates align directly
    if seg0_hw is not None and seg0_hw[0] > 0 and seg0_hw[1] > 0:
        img_left = cv2.resize(img_left, (seg0_hw[1], seg0_hw[0]))
    if seg1_hw is not None and seg1_hw[0] > 0 and seg1_hw[1] > 0:
        img_right = cv2.resize(img_right, (seg1_hw[1], seg1_hw[0]))

    # Ensure both images have same height for concatenation
    h = max(img_left.shape[0], img_right.shape[0])
    if img_left.shape[0] != h:
        img_left = cv2.resize(img_left, (img_left.shape[1], h))
    if img_right.shape[0] != h:
        img_right = cv2.resize(img_right, (img_right.shape[1], h))

    w_left = img_left.shape[1]
    w_right = img_right.shape[1]

    # Compute scale from seg-space to canvas-space (needed if heights were adjusted)
    scale_y_left = h / seg0_hw[0] if seg0_hw else 1.0
    scale_y_right = h / seg1_hw[0] if seg1_hw else 1.0

    # Concatenate
    canvas = np.zeros((h, w_left + w_right, 3), dtype=np.uint8)
    canvas[:h, :w_left] = img_left
    canvas[:h, w_left:] = img_right

    # Filter matches
    if error_only:
        filtered = [m for m in matches if not m.get('coarse_consistent', True)]
    else:
        filtered = matches

    if not filtered:
        return canvas

    # Separate by type
    coarse_error = [m for m in filtered if not m.get('coarse_consistent', True)]
    fine_error_coarse_ok = [m for m in filtered
                           if m.get('fine_consistent', True) is False
                           and m.get('coarse_consistent', True) is True]
    coarse_ok = [m for m in filtered if m.get('coarse_consistent', True)]

    # Sample: 70% error, 30% consistent
    if not error_only and max_lines < len(filtered):
        n_error = int(max_lines * 0.7)
        n_consistent = max_lines - n_error

        sampled = []
        if len(coarse_error) > 0:
            n = min(n_error // 2, len(coarse_error))
            indices = np.random.choice(len(coarse_error), n, replace=False)
            sampled.extend([coarse_error[i] for i in indices])
        if len(fine_error_coarse_ok) > 0:
            n = min(n_error - len(sampled), len(fine_error_coarse_ok))
            indices = np.random.choice(len(fine_error_coarse_ok), n, replace=False)
            sampled.extend([fine_error_coarse_ok[i] for i in indices])
        if len(coarse_ok) > 0 and len(sampled) < max_lines:
            n = min(n_consistent, len(coarse_ok))
            indices = np.random.choice(len(coarse_ok), n, replace=False)
            sampled.extend([coarse_ok[i] for i in indices])

        filtered = sampled
    elif max_lines < len(filtered):
        indices = np.random.choice(len(filtered), max_lines, replace=False)
        filtered = [filtered[i] for i in indices]

    # Draw lines using seg coordinates directly
    for match in filtered:
        x0 = match.get('seg_x0')
        y0 = match.get('seg_y0')
        x1 = match.get('seg_x1')
        y1 = match.get('seg_y1')

        if x0 is None or y0 is None or x1 is None or y1 is None:
            continue

        # Scale y to match canvas height (if heights were adjusted for concatenation)
        y0_draw = y0 * scale_y_left
        y1_draw = y1 * scale_y_right
        x0_draw = x0
        x1_draw = x1

        # Determine color
        if not match.get('coarse_consistent', True):
            color = COLOR_COARSE_ERROR  # Red
        elif not match.get('fine_consistent', True):
            color = COLOR_FINE_ERROR_COARSE_OK  # Orange
        else:
            color = COLOR_COARSE_CONSISTENT  # Green

        pt1 = (int(x0_draw), int(y0_draw))
        pt2 = (int(x1_draw) + w_left, int(y1_draw))

        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 2, color, -1)
        cv2.circle(canvas, pt2, 2, color, -1)

    return canvas


def add_title_bar(img: np.ndarray, title: str, font_scale: float = 0.6) -> np.ndarray:
    """Add a title bar to the top of the image."""
    if not HAS_CV2:
        return img

    h, w = img.shape[:2]
    bar_height = int(25 * font_scale)

    # Create title bar
    title_bar = np.ones((bar_height, w, 3), dtype=np.uint8) * 30

    # Add text
    cv2.putText(title_bar, title, (5, bar_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Concatenate
    result = np.vstack([title_bar, img])
    return result


# =============================================================================
# Main Visualization Logic
# =============================================================================

def load_per_match_data(per_match_jsonl: Path) -> List[Dict]:
    """Load per-match diagnostic data from JSONL."""
    records = []
    if not per_match_jsonl.exists():
        return records

    with open(per_match_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def load_pair_list(pair_list_jsonl: Path) -> List[Dict]:
    """Load target pair list from JSONL."""
    pairs = []
    if not pair_list_jsonl.exists():
        return pairs

    with open(pair_list_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pairs


def group_matches_by_pair(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group per-match records by pair_key."""
    grouped = defaultdict(list)
    for rec in records:
        pair_key = rec.get('pair_key', '')
        if pair_key:
            grouped[pair_key].append(rec)
    return grouped


def build_pair_stats(matches: List[Dict]) -> Dict:
    """Compute statistics for a pair's matches."""
    total = len(matches)
    if total == 0:
        return {
            'num_matches': 0,
            'fine_consistent_count': 0,
            'fine_error_count': 0,
            'fine_error_rate': 0.0,
            'coarse_consistent_count': 0,
            'coarse_error_count': 0,
            'coarse_error_rate': 0.0,
        }

    fine_consistent = sum(1 for m in matches if m.get('fine_consistent', False))
    fine_error = total - fine_consistent

    coarse_consistent = sum(1 for m in matches if m.get('coarse_consistent', False))
    coarse_error = total - coarse_consistent

    return {
        'num_matches': total,
        'fine_consistent_count': fine_consistent,
        'fine_error_count': fine_error,
        'fine_error_rate': fine_error / total,
        'coarse_consistent_count': coarse_consistent,
        'coarse_error_count': coarse_error,
        'coarse_error_rate': coarse_error / total,
    }


def visualize_pair(
    pair_info: Dict,
    matches: List[Dict],
    image_root: str,
    oneformer_dir: Path,
    output_dir: Path,
    max_lines: int = 500,
    overlay_alpha: float = 0.45,
    rank_subdir: str = "",
) -> bool:
    """Visualize a single pair.

    Args:
        pair_info: Dict with pair info from pair list
        matches: List of match records for this pair
        image_root: Root directory for images
        oneformer_dir: Path to OneFormer output
        output_dir: Output directory
        max_lines: Max lines to draw
        overlay_alpha: Overlay transparency
        rank_subdir: Subdirectory name for rank-specific output

    Returns:
        True if successful, False otherwise
    """
    image0 = pair_info.get('image0', '')
    image1 = pair_info.get('image1', '')

    if not image0 or not image1:
        return False

    # Create output subdirectory
    rank_str = pair_info.get('rank', 0)
    if rank_subdir:
        pair_dir = output_dir / rank_subdir / f"pair_{rank_str:03d}"
    else:
        pair_dir = output_dir / f"pair_{rank_str:03d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    img0 = load_image(image0, image_root)
    img1 = load_image(image1, image_root)

    if img0 is None:
        print(f"Failed to load image: {image0}")
        return False
    if img1 is None:
        print(f"Failed to load image: {image1}")
        return False

    # Get image dimensions
    img0_h, img0_w = img0.shape[:2]
    img1_h, img1_w = img1.shape[:2]

    # Load segmentation
    seg0 = load_oneformer_segmentation(image0, oneformer_dir)
    seg1 = load_oneformer_segmentation(image1, oneformer_dir)

    # Get segmentation dimensions
    seg0_h, seg0_w = seg0.shape[:2] if seg0 is not None else (img0_h, img0_w)
    seg1_h, seg1_w = seg1.shape[:2] if seg1 is not None else (img1_h, img1_w)

    print(f"  Pair {rank_str}: img0={img0_w}x{img0_h} seg0={seg0_w}x{seg0_h} | img1={img1_w}x{img1_h} seg1={seg1_w}x{seg1_h}")

    # Build stats
    stats = build_pair_stats(matches)

    # ========== matches_all.png ==========
    # Resize images to seg dimensions so coordinates align, then draw match lines
    matches_canvas = draw_match_lines(
        img0, img1, matches,
        max_lines=max_lines,
        error_only=False,
        seg0_hw=(seg0_h, seg0_w),
        seg1_hw=(seg1_h, seg1_w),
    )

    title = f"Pair {rank_str}: {Path(image0).name} <-> {Path(image1).name}"
    title += f" | matches={stats['num_matches']} | fine_err={stats['fine_error_rate']:.2f} | coarse_err={stats['coarse_error_rate']:.2f}"
    matches_canvas = add_title_bar(matches_canvas, title)

    cv2.imwrite(str(pair_dir / "matches_all.png"), matches_canvas)

    # ========== matches_error_only.png ==========
    error_canvas = draw_match_lines(
        img0, img1, matches,
        max_lines=max_lines,
        error_only=True,
        seg0_hw=(seg0_h, seg0_w),
        seg1_hw=(seg1_h, seg1_w),
    )

    title_err = f"Coarse Error Matches Only (n={stats['coarse_error_count']})"
    error_canvas = add_title_bar(error_canvas, title_err)

    cv2.imwrite(str(pair_dir / "matches_error_only.png"), error_canvas)

    # ========== segmentation_overlay.png ==========
    overlay0 = blend_overlay(img0, seg0, alpha=overlay_alpha)
    overlay1 = blend_overlay(img1, seg1, alpha=overlay_alpha)

    h = max(overlay0.shape[0], overlay1.shape[0])
    if overlay0.shape[0] != h:
        overlay0 = cv2.resize(overlay0, (overlay0.shape[1], h))
    if overlay1.shape[0] != h:
        overlay1 = cv2.resize(overlay1, (overlay1.shape[1], h))

    seg_canvas = np.hstack([overlay0, overlay1])
    seg_title = f"Segmentation Overlay: {Path(image0).name} | {Path(image1).name}"
    seg_canvas = add_title_bar(seg_canvas, seg_title)

    cv2.imwrite(str(pair_dir / "segmentation_overlay.png"), seg_canvas)

    # ========== pair_info.json ==========
    pair_info_out = {
        'rank': pair_info.get('rank'),
        'image0': image0,
        'image1': image1,
        'pair_key': pair_info.get('pair_key', ''),
        'img0_size': [img0_h, img0_w],
        'img1_size': [img1_h, img1_w],
        'seg0_size': [seg0_h, seg0_w],
        'seg1_size': [seg1_h, seg1_w],
        **stats,
    }

    with open(pair_dir / "pair_info.json", 'w') as f:
        json.dump(pair_info_out, f, indent=2)

    return True


def build_pairs_from_matches(records: List[Dict]) -> List[Dict]:
    """Build pair list from per-match records.

    When no explicit pair-list is provided, extract unique pairs from
    per-match JSONL. Each unique pair_key becomes one entry.
    Assigns sequential rank numbers since target-pair files may not have rank.
    """
    pair_keys_seen = {}
    for rec in records:
        pair_key = rec.get('pair_key', '')
        if not pair_key:
            continue
        if pair_key not in pair_keys_seen:
            # Handle list-wrapped paths
            img0 = rec.get('image0', '')
            img1 = rec.get('image1', '')
            if isinstance(img0, str) and img0.startswith("[") and img0.endswith("]"):
                import ast
                try:
                    parsed = ast.literal_eval(img0)
                    if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                        img0 = str(parsed[0])
                except Exception:
                    pass
            if isinstance(img1, str) and img1.startswith("[") and img1.endswith("]"):
                import ast
                try:
                    parsed = ast.literal_eval(img1)
                    if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                        img1 = str(parsed[0])
                except Exception:
                    pass
            pair_keys_seen[pair_key] = {
                'pair_key': pair_key,
                'image0': img0,
                'image1': img1,
                'rank': len(pair_keys_seen),
            }

    return list(pair_keys_seen.values())


def main():
    parser = argparse.ArgumentParser(
        description="Visualize semantic diagnostic results for selected pairs or all pairs"
    )
    parser.add_argument(
        '--per-match-jsonl',
        type=str,
        required=True,
        help='Path to per_match_semantic_diagnostic.jsonl'
    )
    parser.add_argument(
        '--pair-list',
        type=str,
        default=None,
        required=False,
        help='Path to top_coarse_error_pairs.jsonl (optional, builds from per-match if not provided)'
    )
    parser.add_argument(
        '--image-root',
        type=str,
        default='',
        help='Root directory for images'
    )
    parser.add_argument(
        '--oneformer-dir',
        type=str,
        required=True,
        help='Path to OneFormer output directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=500,
        help='Maximum number of match lines to draw'
    )
    parser.add_argument(
        '--error-only',
        type=str,
        default='false',
        choices=['true', 'false'],
        help='Only show error lines'
    )
    parser.add_argument(
        '--overlay-alpha',
        type=float,
        default=0.45,
        help='Segmentation overlay alpha'
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=0,
        help='Maximum number of pairs to visualize (0 = all)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=False,
        help='Skip pairs that already have visualization output'
    )

    args = parser.parse_args()

    per_match_jsonl = Path(args.per_match_jsonl)
    pair_list_jsonl = Path(args.pair_list) if args.pair_list else None
    oneformer_dir = Path(args.oneformer_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-match data from: {per_match_jsonl}")
    records = load_per_match_data(per_match_jsonl)
    print(f"  Loaded {len(records)} match records")

    # Build pair list: from explicit file, or from per-match data
    if pair_list_jsonl and pair_list_jsonl.exists():
        print(f"Loading pair list from: {pair_list_jsonl}")
        pairs = load_pair_list(pair_list_jsonl)
        print(f"  Loaded {len(pairs)} pairs from explicit list")
    else:
        print("No explicit pair-list provided, building from per-match data...")
        pairs = build_pairs_from_matches(records)
        print(f"  Built {len(pairs)} unique pairs from per-match data")

    # Group matches by pair_key
    grouped = group_matches_by_pair(records)
    print(f"  Grouped into {len(grouped)} unique pairs")

    # Limit number of pairs if requested
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        print(f"  Limiting to first {args.max_pairs} pairs")
        pairs = pairs[:args.max_pairs]

    # Process each pair
    success_count = 0
    skipped_count = 0
    for pair in pairs:
        pair_key = pair.get('pair_key', '')
        if not pair_key:
            continue

        matches = grouped.get(pair_key, [])
        if not matches:
            print(f"  No match data for pair: {pair_key[:80]}...")
            continue

        # Check skip-existing flag
        if args.skip_existing:
            rank_str = pair.get('rank', 0)
            # Must match the path used in visualize_pair
            if rank_subdir:
                pair_dir = output_dir / rank_subdir / f"pair_{rank_str:03d}"
            else:
                pair_dir = output_dir / f"pair_{rank_str:03d}"
            if pair_dir.exists() and (pair_dir / "pair_info.json").exists():
                skipped_count += 1
                continue

        print(f"  Processing pair {pair.get('rank')}: {Path(pair.get('image0', '')).name}")

        # Use rank from pair info
        rank_subdir = f"rank_{pair.get('rank', 0):03d}"

        success = visualize_pair(
            pair, matches, args.image_root, oneformer_dir,
            output_dir, args.max_lines, args.overlay_alpha,
            rank_subdir=rank_subdir,
        )
        if success:
            success_count += 1

    print(f"\nVisualization complete!")
    print(f"  Processed {success_count} pairs")
    if skipped_count > 0:
        print(f"  Skipped (existing) {skipped_count} pairs")
    print(f"  Total pairs in list: {len(pairs)}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    sys.exit(main() or 0)