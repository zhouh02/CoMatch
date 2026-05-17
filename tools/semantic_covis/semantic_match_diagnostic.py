#!/usr/bin/env python3
"""
Semantic Match Diagnostic - Analyze semantic consistency of matches.

This module provides tools for diagnosing semantic consistency in image matching
results using OneFormer segmentation output. It does NOT depend on torch/transformers.

Usage:
    from semantic_match_diagnostic import (
        OneFormerSegStore,
        sample_semantic_at_points,
        diagnose_matches_semantic,
    )

Key functions:
    - OneFormerSegStore: Load and access OneFormer segmentation results
    - sample_semantic_at_points: Sample semantic labels at match points
    - map_points_to_segmentation_coords: Handle coordinate system mapping
    - diagnose_matches_semantic: Compute fine/coarse consistency metrics
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from collections import Counter, defaultdict


# =============================================================================
# ADE20K Coarse Label Mapping
# =============================================================================

# Default ADE20K coarse grouping for semantic consistency diagnosis
DEFAULT_COARSE_GROUPS = {
    'building_like': [
        'building', 'wall', 'tower', 'bridge', 'house', 'skyscraper',
        'column', 'fence', 'rail', 'railing', 'window', 'windowpane',
        'door', 'arcade', 'bannister', 'stairway', 'stair', 'escalator',
        'awning', 'balcony', 'base', 'ceiling', 'cornice', 'counter',
        'curb', 'dome', 'elevator', 'fireplace', 'floor', 'floor_mat',
        'grandstand', 'minibar', 'minifont', 'partition', 'pedestal',
        'platform', 'pool', 'ramp', 'river', 'road', 'roof', 'sand',
        'screen', 'shelf', 'shop', 'sink', 'sofa', 'stage', 'stand',
        'street', 'table', 'terrace', 'toilet', 'towel', 'tub',
        'vanity', 'video', 'wall', 'washer', 'wine', 'blind', 'board',
        'bookcase', 'cabinet', 'case', 'chair', 'chest', 'climbing',
    ],
    'sky_like': [
        'sky',
    ],
    'ground_like': [
        'road, route', 'sidewalk, pavement', 'floor', 'earth, ground',
        'path', 'field', 'sand', 'runway', 'land', 'terrain', 'grass',
        'plant', 'tree', 'bush', 'hill', 'mountain', 'rock', 'stone',
        'gravel', 'pavement', 'sidewalk', 'curb', 'crosswalk', 'lane',
        'parking', 'rail track', 'railroad', 'ground', 'dirt', 'mud',
    ],
    'vegetation_like': [
        'tree', 'grass', 'plant', 'flower', 'palm', 'bush', 'shrub',
        'fern', 'moss', 'weed', 'hedge', 'vine', 'lawn', 'meadow',
        'forest', 'wood', 'branch', 'trunk', 'leaf', 'pollen',
    ],
    'water_like': [
        'water', 'sea', 'river', 'lake', 'pool', 'pond', 'ocean',
        'wave', 'fountain', 'ice', 'snow', 'glacier', 'waterfall',
    ],
    'vehicle_like': [
        'car', 'bus', 'truck', 'bicycle', 'boat', 'motorbike', 'van',
        'taxi', 'train', 'tram', 'subway', 'airplane', 'helicopter',
        'ship', 'cart', 'wagon', 'trailer', 'motorbike', 'scooter',
    ],
    'human_like': [
        'person', 'people', 'man', 'woman', 'child', 'boy', 'girl',
        'human', 'pedestrian', 'rider', 'surfer', 'skater',
    ],
    'furniture_like': [
        'bed', 'chair', 'couch', 'table', 'desk', 'cabinet', 'shelf',
        'mirror', 'window', 'door', 'lamp', 'light', 'screen', 'board',
        'bench', 'bookcase', 'counter', 'curtain', 'pillow', 'sink',
        'stairs', 'step', 'stool', 'rack', 'tray', 'trolley',
    ],
    'food_like': [
        'food', 'fruit', 'vegetable', 'meat', 'fish', 'bread', 'cake',
        'pizza', 'salad', 'sandwich', 'dessert', 'drink', 'bottle',
        'cup', 'glass', 'plate', 'bowl', 'spoon', 'fork', 'knife',
    ],
    'electronics_like': [
        'tv', 'monitor', 'screen', 'laptop', 'computer', 'keyboard',
        'mouse', 'phone', 'camera', 'printer', 'speaker', 'lamp',
    ],
    'textile_like': [
        'curtain', 'pillow', 'blanket', 'sheet', 'towel', 'rug',
        'carpet', 'mat', 'cushion', 'upholstery',
    ],
}


def normalize_label_name(name: str) -> str:
    """Normalize label name for comparison.

    Args:
        name: Label name (may contain commas, trailing spaces, etc.)

    Returns:
        Normalized label name (lowercase, stripped)
    """
    if name is None:
        return ''
    return str(name).lower().strip()


def build_coarse_mapping(coarse_groups: Dict[str, List[str]] = None) -> Dict[str, str]:
    """Build fine-to-coarse label mapping.

    Args:
        coarse_groups: Dict mapping coarse group name to list of fine label names.
                      If None, uses DEFAULT_COARSE_GROUPS.

    Returns:
        Dict mapping normalized fine label to coarse group name.
        Unmapped labels get 'other:<original_label>' format.
    """
    if coarse_groups is None:
        coarse_groups = DEFAULT_COARSE_GROUPS

    mapping = {}
    for coarse_name, fine_labels in coarse_groups.items():
        for label in fine_labels:
            normalized = normalize_label_name(label)
            mapping[normalized] = coarse_name

    return mapping


def get_coarse_group(fine_label_id: int, id2label: Dict, coarse_mapping: Dict[str, str]) -> str:
    """Map fine label ID to coarse group.

    Args:
        fine_label_id: OneFormer label ID
        id2label: Dict mapping label ID to label name
        coarse_mapping: Dict mapping normalized label name to coarse group

    Returns:
        Coarse group name, or 'other:<original_label>' for unmapped labels
    """
    label_name = id2label.get(fine_label_id, str(fine_label_id))
    normalized = normalize_label_name(label_name)

    if normalized in coarse_mapping:
        return coarse_mapping[normalized]
    else:
        return f'other:{label_name}'


# =============================================================================
# OneFormer Segmentation Store
# =============================================================================

class OneFormerSegStore:
    """Store for accessing OneFormer segmentation results.

    This class loads pre-computed OneFormer segmentation outputs (npz/json)
    and provides methods to look up segmentation data for images.

    Attributes:
        oneformer_dir: Root directory containing OneFormer outputs
        image_root: Root directory for resolving relative image paths
        npz_dir: Directory containing npz segmentation arrays
        json_dir: Directory containing metadata JSON files
        id2label: Dict mapping label IDs to label names
        label_map_path: Path to label_id_map.json
    """

    def __init__(
        self,
        oneformer_dir: str,
        image_root: str = None,
        verbose: bool = True,
    ):
        """Initialize OneFormer segmentation store.

        Args:
            oneformer_dir: Directory containing OneFormer outputs (npz/, json/)
            image_root: Root directory for resolving relative paths
            verbose: Print warnings for missing files or ambiguous matches
        """
        self.oneformer_dir = Path(oneformer_dir)
        self.image_root = Path(image_root) if image_root else None
        self.verbose = verbose

        # Set up directories
        self.npz_dir = self.oneformer_dir / "npz"
        self.json_dir = self.oneformer_dir / "json"

        if not self.npz_dir.exists():
            raise ValueError(f"npz directory not found: {self.npz_dir}")
        if not self.json_dir.exists():
            raise ValueError(f"json directory not found: {self.json_dir}")

        # Load label map
        label_map_path = self.oneformer_dir / "label_id_map.json"
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                self.id2label = json.load(f)
        else:
            # Try to load from first JSON file
            json_files = list(self.json_dir.glob("*.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    self.id2label = data.get('id2label', {})
            else:
                self.id2label = {}
                if self.verbose:
                    print(f"WARNING: No label_id_map.json found, id2label may be incomplete")

        # Build index for fast lookup
        self._build_index()

    def _build_index(self):
        """Build index for fast image-to-npz/json lookup."""
        self._npz_index = {}  # path stem -> npz path
        self._json_index = {}  # path stem -> json path

        # Index npz files
        for npz_file in self.npz_dir.glob("*.npz"):
            stem = npz_file.stem
            self._npz_index[stem] = npz_file
            # Also store with path separators normalized
            normalized = stem.replace('/', '_').replace('\\', '_')
            if normalized != stem:
                self._npz_index[normalized] = npz_file

        # Index json files
        for json_file in self.json_dir.glob("*.json"):
            stem = json_file.stem
            self._json_index[stem] = json_file
            normalized = stem.replace('/', '_').replace('\\', '_')
            if normalized != stem:
                self._json_index[normalized] = json_file

    def _resolve_path(self, image_path: str) -> Optional[Path]:
        """Resolve image path to actual file location.

        Args:
            image_path: Image path (may be relative or absolute)

        Returns:
            Resolved Path object, or None if not found
        """
        path = Path(image_path)

        # Already absolute
        if path.is_absolute():
            return path if path.exists() else None

        # Try relative to image_root
        if self.image_root:
            abs_path = self.image_root / path
            if abs_path.exists():
                return abs_path

        # Try as-is (relative to current directory)
        if path.exists():
            return path

        return None

    def _find_segmentation(self, image_path: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Find corresponding npz and json files for an image.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (npz_path, json_path) or (None, None) if not found
        """
        path = Path(image_path)

        # Try different stem formats
        stems_to_try = []

        # 1. Original relative path (for MegaDepth style: Undistorted_SfM/0022/images/xxx.jpg)
        if self.image_root:
            try:
                rel_path = path.relative_to(self.image_root)
                stems_to_try.append(str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_'))
                stems_to_try.append(str(rel_path.stem))
            except ValueError:
                pass

        # 2. Original path without extension
        stems_to_try.append(str(path.with_suffix('')).replace('/', '_').replace('\\', '_'))
        stems_to_try.append(path.stem)

        # 3. Basename without extension
        stems_to_try.append(path.name.replace(path.suffix, ''))
        stems_to_try.append(path.stem.replace('/', '_').replace('\\', '_'))

        # Try each stem
        for stem in stems_to_try:
            if stem in self._npz_index:
                return self._npz_index[stem], self._json_index.get(stem)

        # Fallback: basename search
        basename = path.name.replace(path.suffix, '')
        matches = [s for s in self._npz_index.keys() if basename in s]

        if len(matches) > 1 and self.verbose:
            print(f"WARNING: Multiple matches for '{basename}': {matches}")
        elif len(matches) == 1:
            return self._npz_index[matches[0]], self._json_index.get(matches[0])

        if self.verbose:
            print(f"WARNING: No segmentation found for: {image_path}")

        return None, None

    def get(self, image_path: str) -> Optional[Dict]:
        """Get segmentation data for an image.

        Args:
            image_path: Path to the image

        Returns:
            Dict with keys:
                - semantic_label: [H, W] int array of class IDs
                - segment_score: [H, W] float array of confidence
                - height: segmentation height
                - width: segmentation width
                - id2label: dict mapping label IDs to names
                - npz_path: path to source npz
                - json_path: path to source json
                - image_path: original image path (resolved)
            Returns None if not found.
        """
        npz_path, json_path = self._find_segmentation(image_path)

        if npz_path is None:
            return None

        # Load npz
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            if self.verbose:
                print(f"ERROR loading {npz_path}: {e}")
            return None

        result = {
            'semantic_label': data['semantic_label'],
            'segment_score': data['segment_score'],
            'height': int(data['height']),
            'width': int(data['width']),
            'id2label': self.id2label,
            'npz_path': str(npz_path),
            'json_path': str(json_path) if json_path else None,
        }

        return result

    def has(self, image_path: str) -> bool:
        """Check if segmentation exists for an image."""
        npz_path, _ = self._find_segmentation(image_path)
        return npz_path is not None

    def get_coarse_mapping(self) -> Dict[str, str]:
        """Get the default coarse label mapping."""
        return build_coarse_mapping()


# =============================================================================
# Coordinate Mapping
# =============================================================================

def map_points_to_segmentation_coords(
    points_xy: np.ndarray,
    match_hw: Tuple[int, int],
    seg_hw: Tuple[int, int],
    scale: np.ndarray = None,
) -> np.ndarray:
    """Map match points to segmentation coordinate system.

    This handles the case where matches are in a different resolution than
    the segmentation output (e.g., matches in resized image coords,
    segmentation in different resolution).

    Args:
        points_xy: Match points in [N, 2] format (x, y coordinates)
        match_hw: (H, W) of the match coordinate system
        seg_hw: (H, W) of the segmentation coordinate system
        scale: Optional [N, 2] scale factors (sx, sy) for point-wise scaling.
               If provided, used instead of ratio-based scaling.

    Returns:
        Mapped points in segmentation coordinates [M, 2]

    Note:
        Coordinate system handling:
        - points_xy are in xy format (width, height order)
        - h/w are in standard (height, width) order
        - Conversion: (x, y) -> (y, x) for indexing into (H, W) arrays
    """
    points_xy = np.asarray(points_xy)
    if points_xy.ndim == 1:
        points_xy = points_xy.reshape(1, -1)

    # Convert to (y, x) for array indexing
    points_yx = points_xy[:, ::-1]  # [N, 2] now (y, x)

    match_h, match_w = match_hw
    seg_h, seg_w = seg_hw

    # If dimensions match, no mapping needed
    if match_hw == seg_hw:
        return points_xy

    # Use scale if provided
    if scale is not None and len(scale) == len(points_xy):
        scale = np.asarray(scale)
        mapped_yx = points_yx * scale[:, ::-1]  # scale is (sx, sy), apply to (y, x)
        mapped_xy = mapped_yx[:, ::-1]  # back to (x, y)
        return mapped_xy

    # Otherwise, scale by ratio
    if match_h == 0 or match_w == 0:
        print(f"WARNING: match_hw has zero dimension: {match_hw}")
        return points_xy

    scale_y = seg_h / match_h
    scale_x = seg_w / match_w

    mapped_yx = points_yx * np.array([scale_y, scale_x])
    mapped_xy = mapped_yx[:, ::-1]  # back to (x, y)

    return mapped_xy


def sample_semantic_at_points(
    semantic_label: np.ndarray,
    segment_score: np.ndarray,
    points_xy: np.ndarray,
    score_thresh: float = 0.0,
    boundary_radius: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample semantic labels at given points.

    Args:
        semantic_label: [H, W] int array of label IDs
        segment_score: [H, W] float array of confidence scores
        points_xy: [N, 2] array of (x, y) coordinates to sample
        score_thresh: Minimum segment score to consider valid
        boundary_radius: Distance from edge to consider as boundary (0 = no boundary check)

    Returns:
        Tuple of (label_ids, valid_mask)
        - label_ids: [N] int array of sampled label IDs (negative = invalid)
        - valid_mask: [N] bool array indicating which samples are valid
    """
    h, w = semantic_label.shape

    # Convert to integer and clamp to valid range
    points_xy = np.asarray(points_xy)
    if points_xy.ndim == 1:
        points_xy = points_xy.reshape(1, -1)

    xs = np.round(points_xy[:, 0]).astype(int)
    ys = np.round(points_xy[:, 1]).astype(int)

    # Clamp to image bounds
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    # Sample
    label_ids = semantic_label[ys, xs]
    scores = segment_score[ys, xs]

    # Build valid mask
    valid = np.ones(len(points_xy), dtype=bool)

    # Invalid if label < 0
    invalid_label = label_ids < 0
    valid &= ~invalid_label

    # Invalid if score below threshold
    invalid_score = scores < score_thresh
    valid &= ~invalid_score

    # Boundary check
    if boundary_radius > 0:
        at_edge = (xs <= boundary_radius) | (xs >= w - boundary_radius - 1) | \
                  (ys <= boundary_radius) | (ys >= h - boundary_radius - 1)
        valid &= ~at_edge

    # Mark invalid as -1
    label_ids[~valid] = -1

    return label_ids, valid


# =============================================================================
# Semantic Match Diagnosis
# =============================================================================

def diagnose_matches_semantic(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    mconf: np.ndarray,
    seg0: Dict,
    seg1: Dict,
    id2label: Dict,
    coarse_mapping: Dict[str, str] = None,
    score_thresh: float = 0.0,
    thresholds: Tuple[float, ...] = (0.2, 0.5, 0.8),
    boundary_radius: int = 0,
    save_all_matches: bool = False,
) -> Dict:
    """Diagnose semantic consistency of matches.

    Args:
        mkpts0: [M, 2] match points in image 0 (xy coordinates)
        mkpts1: [M, 2] match points in image 1 (xy coordinates)
        mconf: [M] match confidence scores
        seg0: Segmentation dict for image 0 (from OneFormerSegStore.get)
        seg1: Segmentation dict for image 1
        id2label: Dict mapping label ID to label name
        coarse_mapping: Dict mapping normalized label name to coarse group.
                       If None, uses DEFAULT_COARSE_GROUPS.
        score_thresh: Minimum segment score for valid sample
        thresholds: Tuple of confidence thresholds for stratified analysis
        boundary_radius: Distance from edge to mark as boundary (0 = disabled)
        save_all_matches: If True, include per-match details (can be large)

    Returns:
        Dict with diagnostic results:
            - num_matches: total number of matches
            - num_valid: number of matches with valid segmentation samples
            - fine_consistent: count of matches with same fine label
            - fine_mismatch: count of matches with different fine label
            - fine_consistency_rate: fine_consistent / num_valid
            - fine_mismatch_rate: fine_mismatch / num_valid
            - coarse_consistent: count of matches with same coarse group
            - coarse_mismatch: count of matches with different coarse group
            - coarse_consistency_rate: coarse_consistent / num_valid
            - coarse_mismatch_rate: coarse_mismatch / num_valid
            - high_conf_stats: dict with stats per confidence threshold
            - top_fine_mismatch_pairs: list of most common fine label mismatches
            - top_coarse_mismatch_pairs: list of most common coarse mismatches
            - per_match (optional): per-match diagnostic data
    """
    if coarse_mapping is None:
        coarse_mapping = build_coarse_mapping()

    # Sample semantic labels at match points
    # Note: mkpts are in segmentation coordinate space (after resize/pad)
    labels0, valid0 = sample_semantic_at_points(
        seg0['semantic_label'],
        seg0['segment_score'],
        mkpts0,
        score_thresh=score_thresh,
        boundary_radius=boundary_radius,
    )
    labels1, valid1 = sample_semantic_at_points(
        seg1['semantic_label'],
        seg1['segment_score'],
        mkpts1,
        score_thresh=score_thresh,
        boundary_radius=boundary_radius,
    )

    # Combined valid mask
    valid_mask = valid0 & valid1

    num_matches = len(mkpts0)
    num_valid = valid_mask.sum()

    # Fine label consistency
    fine_match = labels0 == labels1
    fine_consistent = (fine_match & valid_mask).sum()
    fine_mismatch = num_valid - fine_consistent

    # Coarse label consistency
    coarse0 = np.array([get_coarse_group(l, id2label, coarse_mapping) for l in labels0])
    coarse1 = np.array([get_coarse_group(l, id2label, coarse_mapping) for l in labels1])
    coarse_match = coarse0 == coarse1
    coarse_consistent = (coarse_match & valid_mask).sum()
    coarse_mismatch = num_valid - coarse_consistent

    # Compute rates
    fine_consistency_rate = fine_consistent / num_valid if num_valid > 0 else 0.0
    fine_mismatch_rate = fine_mismatch / num_valid if num_valid > 0 else 0.0
    coarse_consistency_rate = coarse_consistent / num_valid if num_valid > 0 else 0.0
    coarse_mismatch_rate = coarse_mismatch / num_valid if num_valid > 0 else 0.0

    # Stratified analysis by confidence
    high_conf_stats = {}
    for thr in thresholds:
        high_conf_mask = valid_mask & (mconf >= thr)
        high_conf_count = high_conf_mask.sum()

        if high_conf_count > 0:
            hc_fine_consistent = (fine_match & high_conf_mask).sum()
            hc_coarse_consistent = (coarse_match & high_conf_mask).sum()

            high_conf_stats[f"{thr:.1f}"] = {
                'count': int(high_conf_count),
                'fine_consistent': int(hc_fine_consistent),
                'coarse_consistent': int(hc_coarse_consistent),
                'fine_consistency_rate': hc_fine_consistent / high_conf_count,
                'coarse_consistency_rate': hc_coarse_consistent / high_conf_count,
            }

    # Top mismatch pairs (fine label)
    mismatch_indices = np.where(valid_mask & ~fine_match)[0]
    fine_mismatch_pairs = Counter()
    for idx in mismatch_indices:
        label0_name = id2label.get(labels0[idx], str(labels0[idx]))
        label1_name = id2label.get(labels1[idx], str(labels1[idx]))
        pair = (label0_name, label1_name)
        fine_mismatch_pairs[pair] += 1

    top_fine_mismatch_pairs = [
        {'label0': p[0], 'label1': p[1], 'count': c}
        for p, c in fine_mismatch_pairs.most_common(10)
    ]

    # Top mismatch pairs (coarse)
    coarse_mismatch_indices = np.where(valid_mask & ~coarse_match)[0]
    coarse_mismatch_pairs = Counter()
    for idx in coarse_mismatch_indices:
        pair = (coarse0[idx], coarse1[idx])
        coarse_mismatch_pairs[pair] += 1

    top_coarse_mismatch_pairs = [
        {'coarse0': p[0], 'coarse1': p[1], 'count': c}
        for p, c in coarse_mismatch_pairs.most_common(10)
    ]

    # Build result
    result = {
        'num_matches': num_matches,
        'num_valid': int(num_valid),
        'fine_consistent': int(fine_consistent),
        'fine_mismatch': int(fine_mismatch),
        'fine_consistency_rate': float(fine_consistency_rate),
        'fine_mismatch_rate': float(fine_mismatch_rate),
        'coarse_consistent': int(coarse_consistent),
        'coarse_mismatch': int(coarse_mismatch),
        'coarse_consistency_rate': float(coarse_consistency_rate),
        'coarse_mismatch_rate': float(coarse_mismatch_rate),
        'high_conf_stats': high_conf_stats,
        'top_fine_mismatch_pairs': top_fine_mismatch_pairs,
        'top_coarse_mismatch_pairs': top_coarse_mismatch_pairs,
    }

    # Optionally save per-match details
    if save_all_matches:
        per_match = []
        for i in range(num_matches):
            entry = {
                'idx': i,
                'conf': float(mconf[i]) if i < len(mconf) else None,
                'pt0': mkpts0[i].tolist() if i < len(mkpts0) else None,
                'pt1': mkpts1[i].tolist() if i < len(mkpts1) else None,
                'valid': bool(valid_mask[i]) if i < len(valid_mask) else False,
                'label0': int(labels0[i]) if i < len(labels0) else None,
                'label1': int(labels1[i]) if i < len(labels1) else None,
                'fine_match': bool(fine_match[i]) if i < len(fine_match) else None,
                'coarse0': str(coarse0[i]) if i < len(coarse0) else None,
                'coarse1': str(coarse1[i]) if i < len(coarse1) else None,
                'coarse_match': bool(coarse_match[i]) if i < len(coarse_match) else None,
            }
            per_match.append(entry)
        result['per_match'] = per_match

    return result


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def diagnose_batch(
    batch: Dict,
    seg_store: OneFormerSegStore,
    coarse_mapping: Dict[str, str] = None,
    score_thresh: float = 0.0,
    thresholds: Tuple[float, ...] = (0.2, 0.5, 0.8),
    boundary_radius: int = 0,
    save_all_matches: bool = False,
    save_per_match: bool = None,
) -> List[Dict]:
    """Diagnose semantic consistency for a batch of matches.

    Args:
        batch: Dict from CoMatch evaluation with keys:
            - mkpts0_f: [M, 2] match points in image 0
            - mkpts1_f: [M, 2] match points in image 1
            - mconf: [M] match confidence
            - image0_path / image1_path or pair_names
            - scale0 / scale1 (optional, for coordinate mapping)
            - hw0_i / hw1_i (optional, for coordinate mapping)
        seg_store: OneFormerSegStore instance
        coarse_mapping: Coarse label mapping
        score_thresh: Minimum segment score
        thresholds: Confidence thresholds for stratified analysis
        boundary_radius: Boundary radius for validity check
        save_all_matches: If True, save per-match details (preferred name)
        save_per_match: Deprecated alias of save_all_matches (kept for
            backward compatibility)

    Returns:
        List of diagnostic results, one per pair in the batch
    """
    # Back-compat: accept either save_all_matches or save_per_match
    if save_per_match is not None:
        save_all_matches = bool(save_per_match) or save_all_matches

    results = []

    # Extract match data
    mkpts0 = batch.get('mkpts0_f')
    mkpts1 = batch.get('mkpts1_f')
    mconf = batch.get('mconf')

    if mkpts0 is None or mkpts1 is None:
        return results

    # Handle batch dimension
    if mkpts0.ndim == 3:
        mkpts0 = mkpts0[0]  # [M, 2]
        mkpts1 = mkpts1[0]
        mconf = mconf[0] if mconf is not None else np.ones(len(mkpts0))

    # Get image paths
    image0_path = None
    image1_path = None

    if 'image0_path' in batch:
        image0_path = batch['image0_path']
        image1_path = batch['image1_path']
    elif 'pair_names' in batch:
        pair_names = batch['pair_names']
        if isinstance(pair_names, (list, tuple)) and len(pair_names) >= 2:
            image0_path = pair_names[0]
            image1_path = pair_names[1]

    if image0_path is None or image1_path is None:
        print("WARNING: Could not determine image paths from batch")
        return results

    # Get segmentation data
    seg0 = seg_store.get(image0_path) if isinstance(image0_path, str) else None
    seg1 = seg_store.get(image1_path) if isinstance(image1_path, str) else None

    if seg0 is None or seg1 is None:
        return [{
            'num_matches': len(mkpts0),
            'num_valid': 0,
            'fine_consistent': 0,
            'fine_mismatch': 0,
            'fine_consistency_rate': 0.0,
            'fine_mismatch_rate': 0.0,
            'coarse_consistent': 0,
            'coarse_mismatch': 0,
            'coarse_consistency_rate': 0.0,
            'coarse_mismatch_rate': 0.0,
            'missing_segmentation': True,
            'image0': image0_path,
            'image1': image1_path,
        }]

    # Determine coordinate mapping
    # mkpts0_f / mkpts1_f are in the processed image coordinates
    # Need to check if they match seg0 / seg1 dimensions

    seg_hw = (seg0['height'], seg0['width'])
    match_hw = None

    # Try to get match dimensions from batch
    if 'hw0_i' in batch:
        hw0 = batch['hw0_i']
        if isinstance(hw0, (list, tuple)):
            match_hw = (hw0[0], hw0[1])
        elif hasattr(hw0, '__len__') and len(hw0) >= 2:
            match_hw = (int(hw0[0]), int(hw0[1]))

    if match_hw is None:
        # Estimate from image size or use seg dimensions
        match_hw = seg_hw

    # Map points if needed
    if match_hw != seg_hw:
        mkpts0_mapped = map_points_to_segmentation_coords(
            mkpts0, match_hw, seg_hw
        )
        mkpts1_mapped = map_points_to_segmentation_coords(
            mkpts1, match_hw, seg_hw
        )
    else:
        mkpts0_mapped = mkpts0
        mkpts1_mapped = mkpts1

    # Run diagnosis
    result = diagnose_matches_semantic(
        mkpts0_mapped,
        mkpts1_mapped,
        mconf,
        seg0,
        seg1,
        seg0['id2label'],
        coarse_mapping=coarse_mapping,
        score_thresh=score_thresh,
        thresholds=thresholds,
        boundary_radius=boundary_radius,
        save_all_matches=save_all_matches,
    )

    result['image0'] = image0_path if isinstance(image0_path, str) else str(image0_path)
    result['image1'] = image1_path if isinstance(image1_path, str) else str(image1_path)

    return [result]


# =============================================================================
# Summary Aggregation
# =============================================================================

def aggregate_diagnostics(pair_results: List[Dict]) -> Dict:
    """Aggregate per-pair diagnostic results into global summary.

    Args:
        pair_results: List of diagnostic results from diagnose_batch

    Returns:
        Dict with aggregated statistics
    """
    if not pair_results:
        return {
            'num_pairs': 0,
            'num_pairs_with_valid_semantic': 0,
            'num_pairs_missing_segmentation': 0,
            'total_matches': 0,
            'total_valid_matches': 0,
            'global_fine_consistency_rate': 0.0,
            'global_fine_mismatch_rate': 0.0,
            'global_coarse_consistency_rate': 0.0,
            'global_coarse_mismatch_rate': 0.0,
        }

    total_pairs = len(pair_results)
    missing_count = sum(1 for r in pair_results if r.get('missing_segmentation', False))
    valid_pairs = total_pairs - missing_count

    total_matches = sum(r.get('num_matches', 0) for r in pair_results)
    total_valid = sum(r.get('num_valid', 0) for r in pair_results)

    # Aggregate consistency
    total_fine_consistent = sum(r.get('fine_consistent', 0) for r in pair_results)
    total_fine_mismatch = sum(r.get('fine_mismatch', 0) for r in pair_results)
    total_coarse_consistent = sum(r.get('coarse_consistent', 0) for r in pair_results)
    total_coarse_mismatch = sum(r.get('coarse_mismatch', 0) for r in pair_results)

    # Aggregate high-conf stats
    all_thresholds = set()
    for r in pair_results:
        all_thresholds.update(r.get('high_conf_stats', {}).keys())

    global_high_conf = {}
    for thr in sorted(all_thresholds):
        count = sum(r.get('high_conf_stats', {}).get(thr, {}).get('count', 0) for r in pair_results)
        fine = sum(r.get('high_conf_stats', {}).get(thr, {}).get('fine_consistent', 0) for r in pair_results)
        coarse = sum(r.get('high_conf_stats', {}).get(thr, {}).get('coarse_consistent', 0) for r in pair_results)

        global_high_conf[thr] = {
            'count': count,
            'fine_consistent': fine,
            'coarse_consistent': coarse,
            'fine_consistency_rate': fine / count if count > 0 else 0.0,
            'coarse_consistency_rate': coarse / count if count > 0 else 0.0,
        }

    # Aggregate top mismatch pairs
    from collections import Counter

    all_fine_mismatches = Counter()
    all_coarse_mismatches = Counter()

    for r in pair_results:
        for pair in r.get('top_fine_mismatch_pairs', []):
            key = (pair['label0'], pair['label1'])
            all_fine_mismatches[key] += pair['count']
        for pair in r.get('top_coarse_mismatch_pairs', []):
            key = (pair['coarse0'], pair['coarse1'])
            all_coarse_mismatches[key] += pair['count']

    global_top_fine_mismatch = [
        {'label0': p[0], 'label1': p[1], 'count': c}
        for p, c in all_fine_mismatches.most_common(20)
    ]

    global_top_coarse_mismatch = [
        {'coarse0': p[0], 'coarse1': p[1], 'count': c}
        for p, c in all_coarse_mismatches.most_common(20)
    ]

    return {
        'num_pairs': total_pairs,
        'num_pairs_with_valid_semantic': valid_pairs,
        'num_pairs_missing_segmentation': missing_count,
        'total_matches': total_matches,
        'total_valid_matches': total_valid,
        'global_fine_consistency_rate': total_fine_consistent / total_valid if total_valid > 0 else 0.0,
        'global_fine_mismatch_rate': total_fine_mismatch / total_valid if total_valid > 0 else 0.0,
        'global_coarse_consistency_rate': total_coarse_consistent / total_valid if total_valid > 0 else 0.0,
        'global_coarse_mismatch_rate': total_coarse_mismatch / total_valid if total_valid > 0 else 0.0,
        'global_high_conf_stats': global_high_conf,
        'global_top_fine_mismatch_pairs': global_top_fine_mismatch,
        'global_top_coarse_mismatch_pairs': global_top_coarse_mismatch,
    }


# =============================================================================
# CLI for Standalone Testing
# =============================================================================

def main():
    """Standalone test / demo of the diagnostic module."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test semantic match diagnostic")
    parser.add_argument('--oneformer-dir', type=str, required=True)
    parser.add_argument('--image0', type=str, help='Test image 0 path')
    parser.add_argument('--image1', type=str, help='Test image 1 path')
    parser.add_argument('--mkpts0', type=str, help='Path to mkpts0 .npy file')
    parser.add_argument('--mkpts1', type=str, help='Path to mkpts1 .npy file')
    parser.add_argument('--mconf', type=str, help='Path to mconf .npy file')

    args = parser.parse_args()

    print("=" * 60)
    print("Semantic Match Diagnostic Test")
    print("=" * 60)

    # Initialize store
    seg_store = OneFormerSegStore(args.oneformer_dir, verbose=True)

    if args.image0 and args.image1:
        print(f"\nLooking up segmentations...")
        print(f"  Image 0: {args.image0}")
        print(f"  Image 1: {args.image1}")

        seg0 = seg_store.get(args.image0)
        seg1 = seg_store.get(args.image1)

        if seg0 is None:
            print(f"  WARNING: No segmentation found for {args.image0}")
        else:
            print(f"  Seg 0: {seg0['height']}x{seg0['width']}")

        if seg1 is None:
            print(f"  WARNING: No segmentation found for {args.image1}")
        else:
            print(f"  Seg 1: {seg1['height']}x{seg1['width']}")

    print("\nAvailable labels:")
    for lid, lname in list(seg_store.id2label.items())[:20]:
        print(f"  {lid}: {lname}")
    if len(seg_store.id2label) > 20:
        print(f"  ... and {len(seg_store.id2label) - 20} more")

    print("\n" + "=" * 60)
    print("Done")


if __name__ == '__main__':
    sys.exit(main() or 0)