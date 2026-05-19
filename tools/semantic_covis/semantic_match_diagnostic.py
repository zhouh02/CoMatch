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
import torch

from collections import Counter, defaultdict


# =============================================================================
# Path Normalization
# =============================================================================


def normalize_rel_image_path(x):
    """Normalize an image path that may be wrapped in a list / repr string.

    CoMatch batches with batch_size=1 often store paths as
    ``('path.jpg',)`` or ``['path.jpg']`` which, when str()'d, become
    ``"('path.jpg',)"`` or ``"['path.jpg']"``.  This function unwraps
    them back to a plain string.

    Args:
        x: image path - str, list, tuple, numpy array, or repr-wrapped str

    Returns:
        Clean relative path string, e.g. ``"Undistorted_SfM/0022/images/xxx.jpg"``
    """
    import ast as _ast

    # Tensor / numpy array -> list
    if hasattr(x, "tolist") and not isinstance(x, str):
        x = x.tolist()

    # list / tuple -> first element
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        x = x[0]
        # recurse in case it's nested
        if isinstance(x, (list, tuple)):
            return normalize_rel_image_path(x)

    if not isinstance(x, str):
        x = str(x)

    s = x.strip()

    # "['xxx.jpg']" or "('xxx.jpg',)" -> unwrap
    if s.startswith("[") or s.startswith("("):
        try:
            parsed = _ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                s = str(parsed[0]).strip()
        except Exception:
            pass

    # Strip any remaining quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]

    return s


def image_path_to_safe_stem(image_path: str) -> str:
    """Convert an image path to the safe_stem used in OneFormer output filenames.

    This must match the logic in export_oneformer_dir.py:
        rel_stem = Path(rel_path).with_suffix("")
        safe_stem = rel_stem.as_posix().replace("/", "_").replace("\\", "_")

    For example:
        "Undistorted_SfM/0022/images/427154679_de14c315f4_o.jpg"
        -> "Undistorted_SfM_0022_images_427154679_de14c315f4_o"
    """
    p = Path(image_path)
    rel_stem = p.with_suffix("")
    # Use as_posix to get forward slashes on all platforms
    safe_stem = rel_stem.as_posix().replace("/", "_").replace("\\", "_")
    return safe_stem


# =============================================================================
# ADE20K Coarse Label Mapping
# =============================================================================

# Default ADE20K coarse grouping for semantic consistency diagnosis
DEFAULT_COARSE_GROUPS = {
    # ADE20K label names are normalized via normalize_label_name():
    #   "building, edifice" -> "building"
    #   "road, route"       -> "road"
    #   "sidewalk, pavement"-> "sidewalk"
    #   "earth, ground"     -> "earth"
    #   "swimming pool"     -> "swimming pool"
    # So COARSE_GROUPS entries should use the normalized primary class names.
    'building_like': [
        'building', 'wall', 'tower', 'bridge', 'house', 'skyscraper',
        'column', 'fence', 'railing', 'windowpane', 'door',
        'arcade', 'bannister', 'stairway', 'stair', 'escalator',
        'awning', 'balcony', 'ceiling', 'cornice', 'dome', 'elevator',
        'fireplace', 'grandstand', 'minibar', 'partition', 'pedestal',
        'platform', 'ramp', 'roof', 'screen', 'shelf', 'shop', 'sink',
        'sofa', 'stage', 'stand', 'street', 'table', 'terrace', 'toilet',
        'vanity', 'blind', 'board', 'bookcase', 'cabinet', 'case',
        'chest', 'climbing', 'counter', 'curtain', 'lamp', 'light',
        'bench', 'stairs', 'step', 'stool', 'rack', 'tray', 'trolley',
        'bed', 'couch', 'desk', 'mirror', 'computer', 'laptop',
        'tv', 'phone', 'camera', 'printer', 'speaker', 'keyboard',
        'food', 'fruit', 'vegetable', 'meat', 'fish', 'bread', 'cake',
        'pizza', 'salad', 'sandwich', 'dessert', 'drink', 'bottle',
        'cup', 'glass', 'plate', 'bowl', 'spoon', 'fork', 'knife',
        'pillow', 'blanket', 'sheet', 'rug', 'carpet', 'mat', 'cushion',
        'upholstery', 'banner', 'flag', 'flagpole', 'book', 'clock',
        'bottle', 'vase', 'basket', 'pot', 'jar', 'hen', 'frisbee',
        'skateboard', 'surfboard', 'snowboard', 'sports car', 'ambulance',
        'minivan', 'taxi', 'train', 'tram', 'subway', 'helicopter',
        'cart', 'wagon', 'trailer', 'scooter', 'skier',
    ],
    'sky_like': [
        'sky',
    ],
    'ground_like': [
        'road', 'sidewalk', 'floor', 'earth', 'path', 'field', 'sand',
        'runway', 'land', 'terrain', 'gravel', 'pavement', 'curb',
        'crosswalk', 'lane', 'parking', 'rail track', 'railroad',
        'ground', 'dirt', 'mud', 'stairs', 'step',
    ],
    'vegetation_like': [
        'tree', 'grass', 'plant', 'flower', 'palm', 'bush', 'shrub',
        'fern', 'moss', 'weed', 'hedge', 'vine', 'lawn', 'meadow',
        'forest', 'wood', 'branch', 'trunk', 'leaf', 'hill', 'mountain',
        'rock', 'stone', 'bathtub', 'pool', 'swimming pool',
    ],
    'water_like': [
        'water', 'sea', 'river', 'lake', 'pond', 'ocean',
        'wave', 'fountain', 'ice', 'snow', 'glacier', 'waterfall',
    ],
    'vehicle_like': [
        'car', 'bus', 'truck', 'bicycle', 'boat', 'motorbike', 'van',
        'taxi', 'train', 'tram', 'subway', 'airplane', 'helicopter',
        'ship', 'cart', 'wagon', 'trailer', 'scooter', 'sports car',
        'ambulance', 'minivan', 'skateboard', 'surfboard', 'snowboard',
    ],
    'human_like': [
        'person', 'people', 'man', 'woman', 'child', 'boy', 'girl',
        'human', 'pedestrian', 'rider', 'surfer', 'skater', 'skier',
    ],
}


def normalize_label_name(name: str) -> str:
    """Normalize an ADE20K label name for coarse-group matching.

    Handles:
    - trailing spaces / uppercase: "Building ", "CEILING" -> "building"
    - comma-separated synonyms: "building, edifice" -> "building"
    - multi-word labels: "crosswalk" (no comma, keep as-is)
    - hyphens and apostrophes: removed

    Args:
        name: Label name from id2label

    Returns:
        Normalized primary label name (lowercase, comma-split, stripped)
    """
    if name is None:
        return ''
    s = str(name).lower().strip()
    # ADE20K format: "building, edifice", "road, route", "earth, ground"
    if ',' in s:
        s = s.split(',')[0].strip()
    # Remove leading/trailing hyphens within word (e.g. "cross-walk" -> "cross walk")
    # but keep internal hyphens as spaces for multi-word matching
    # "swimming-pool" -> "swimming pool" so we can match "swimming pool" -> "pool"
    s = s.replace('-', ' ')
    return s


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
    raw_name = id2label.get(fine_label_id, str(fine_label_id))
    normalized = normalize_label_name(raw_name)

    if normalized in coarse_mapping:
        return coarse_mapping[normalized]
    else:
        return f'other:{raw_name}'


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
        # json dir is optional - npz is the hard requirement
        if not self.json_dir.exists():
            self.json_dir = None

        # Load id2label from multiple possible sources / formats
        self.id2label = self._load_id2label()

        # Build index for fast lookup
        self._build_index()

    def _load_id2label(self) -> Dict[int, str]:
        """Load id2label from multiple possible sources and formats.

        Supports:
          1. label_id_map.json: {"0": "building", "1": "road", ...} or
                                {"0": {"name": "building"}, "1": {"name": "road"}}
          2. OneFormer per-image JSON: {"id2label": {...}, ...}
          3. ade20k_panoptic.json: {"0": {"isthing": ..., "name": "..."}, ...}
          4. categories: [{"id": ..., "name": ...}, ...]

        Returns:
            Dict mapping int label ID to label name string.
        """
        sources = []

        # Source 1: label_id_map.json at oneformer_dir root
        label_map_path = self.oneformer_dir / "label_id_map.json"
        if label_map_path.exists():
            sources.append(('label_id_map.json', label_map_path))

        # Source 2: metadata JSON at oneformer_dir root (e.g. from --meta-dir copy)
        for name in ("id2label.json", "ade20k_id2label.json", "ade20k_panoptic.json"):
            p = self.oneformer_dir / name
            if p.exists():
                sources.append((name, p))

        # Source 3: first per-image json in json_dir
        if self.json_dir and self.json_dir.exists():
            json_files = sorted(self.json_dir.glob("*.json"))
            if json_files:
                sources.append(('first json_dir file', json_files[0]))

        # Source 4: meta_dir passed during construction (stored in oneformer_dir if different)
        for name in ("id2label.json", "ade20k_id2label.json", "ade20k_panoptic.json",
                     "ade20k_semantic.json"):
            p = self.oneformer_dir / name
            if p.exists() and (label_map_path, p) not in sources:
                sources.append((name, p))

        for source_name, path in sources:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"  Failed to load {path}: {e}")
                continue

            id2label = self._parse_id2label(raw)
            if id2label:
                if self.verbose:
                    sample = list(id2label.items())[:3]
                    print(f"  Loaded id2label from {source_name} ({len(id2label)} labels)")
                    print(f"    Sample: {sample}")
                return id2label

        if self.verbose:
            print(f"  WARNING: Could not load id2label from any source")
        return {}

    @staticmethod
    def _parse_id2label(raw) -> Optional[Dict[int, str]]:
        """Parse raw JSON into id2label dict.

        Handles:
          - {"0": "building", "1": "road"}  (plain strings)
          - {"0": {"name": "building"}, "1": {"name": "road"}}  (ADE20K panoptic format)
          - {"0": {"isthing": ..., "name": "..."}, ...}  (OneFormer demo format)
          - [{"id": 0, "name": "building"}, ...]  (categories list)
          - {"id2label": {"0": "building", ...}}  (wrapped)

        Returns:
            Dict mapping int label ID to label name string, or None if unrecognized.
        """
        if not raw:
            return None

        # Unwrap if top-level key is "id2label"
        if isinstance(raw, dict) and 'id2label' in raw:
            raw = raw['id2label']

        # Case 1: plain strings {"0": "building", "1": "road"}
        if all(isinstance(v, str) for v in raw.values()):
            return {int(k): v for k, v in raw.items()}

        # Case 2: categories list [{"id": 0, "name": "building"}, ...]
        if isinstance(raw, list):
            result = {}
            for item in raw:
                if isinstance(item, dict) and 'id' in item and 'name' in item:
                    result[int(item['id'])] = item['name']
            if result:
                return result

        # Case 3: ADE20K panoptic / OneFormer demo format
        # {"0": {"isthing": ..., "name": "..."}, ...}
        # or {"0": {"name": "..."}, ...}
        if isinstance(raw, dict):
            result = {}
            for k, v in raw.items():
                if isinstance(v, dict) and 'name' in v:
                    result[int(k)] = v['name']
                elif isinstance(v, dict) and 'label' in v:
                    result[int(k)] = v['label']
            if result:
                return result

        return None

    def _build_index(self):
        """Build index for fast image-to-npz/json lookup.

        The index maps safe_stem (as produced by export_oneformer_dir.py)
        to the actual npz/json file paths.
        """
        self._npz_index = {}  # safe_stem -> npz path
        self._json_index = {}  # safe_stem -> json path

        # Index npz files
        for npz_file in self.npz_dir.glob("*.npz"):
            self._npz_index[npz_file.stem] = npz_file

        # Index json files (optional dir)
        if self.json_dir and self.json_dir.exists():
            for json_file in self.json_dir.glob("*.json"):
                self._json_index[json_file.stem] = json_file

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

        The lookup strategy mirrors export_oneformer_dir.py's naming:
            safe_stem = Path(rel_path).with_suffix("").as_posix()
                           .replace("/", "_").replace("\\", "_")
            npz_path = oneformer_dir / "npz" / f"{safe_stem}.npz"

        Args:
            image_path: Path to the image (may be relative or absolute)

        Returns:
            Tuple of (npz_path, json_path) or (None, None) if not found
        """
        # Normalize the input path first
        clean_path = normalize_rel_image_path(image_path)

        # Strategy 1: direct safe_stem lookup (matches export_oneformer_dir.py)
        safe_stem = image_path_to_safe_stem(clean_path)
        npz_path = self._npz_index.get(safe_stem)
        if npz_path is not None:
            json_path = self._json_index.get(safe_stem)
            return npz_path, json_path

        # Strategy 2: if image_root is set, try stripping it to get relative path
        if self.image_root:
            try:
                rel = Path(clean_path).relative_to(self.image_root)
                safe_stem2 = image_path_to_safe_stem(str(rel))
                npz_path = self._npz_index.get(safe_stem2)
                if npz_path is not None:
                    return npz_path, self._json_index.get(safe_stem2)
            except ValueError:
                pass

        # Strategy 3: basename fallback
        basename = Path(clean_path).stem
        matches = [s for s in self._npz_index.keys() if s.endswith(basename)]
        if len(matches) == 1:
            return self._npz_index[matches[0]], self._json_index.get(matches[0])
        elif len(matches) > 1 and self.verbose:
            print(f"WARNING: Multiple npz matches for basename '{basename}': {matches}")

        if self.verbose:
            expected = self.npz_dir / f"{safe_stem}.npz"
            print(f"WARNING: No segmentation found for: {clean_path}")
            print(f"  Expected: {expected}")
            print(f"  Exists: {expected.exists()}")

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
    num_out_of_bounds = int(num_matches - num_valid)

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

    # Compute rates (denominator = num_valid semantic matches)
    fine_consistency_rate = fine_consistent / num_valid if num_valid > 0 else 0.0
    fine_error_rate = fine_mismatch / num_valid if num_valid > 0 else 0.0
    coarse_consistency_rate = coarse_consistent / num_valid if num_valid > 0 else 0.0
    coarse_error_rate = coarse_mismatch / num_valid if num_valid > 0 else 0.0

    # Diagnostic: fine error but coarse correct (proves coarse mapping works)
    fine_error_but_coarse_correct = int(((~fine_match) & coarse_match & valid_mask).sum())
    fine_error_coarse_correct_rate = fine_error_but_coarse_correct / num_valid if num_valid > 0 else 0.0

    # Count unknown coarse labels
    unknown_counts = Counter()
    seen_labels = Counter()
    for lbl_id in labels0[valid_mask]:
        lbl_name = id2label.get(lbl_id, str(lbl_id))
        seen_labels[lbl_name] += 1
        if get_coarse_group(lbl_id, id2label, coarse_mapping).startswith('other:'):
            unknown_counts[normalize_label_name(lbl_name)] += 1
    for lbl_id in labels1[valid_mask]:
        lbl_name = id2label.get(lbl_id, str(lbl_id))
        seen_labels[lbl_name] += 1
        if get_coarse_group(lbl_id, id2label, coarse_mapping).startswith('other:'):
            unknown_counts[normalize_label_name(lbl_name)] += 1

    num_known_coarse_labels = len(seen_labels) - len(unknown_counts)
    num_unknown_coarse_labels = len(unknown_counts)
    unknown_label_top20 = [
        {'label': l, 'count': c} for l, c in unknown_counts.most_common(20)
    ]

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

    # Fine label mismatch pairs
    fine_mismatch_indices = np.where(valid_mask & ~fine_match)[0]
    fine_mismatch_pairs = Counter()
    for idx in fine_mismatch_indices:
        l0 = id2label.get(labels0[idx], str(labels0[idx]))
        l1 = id2label.get(labels1[idx], str(labels1[idx]))
        fine_mismatch_pairs[(l0, l1)] += 1
    top_fine_mismatch_pairs = [
        {'label0': p[0], 'label1': p[1], 'count': c}
        for p, c in fine_mismatch_pairs.most_common(10)
    ]

    # Coarse mismatch pairs
    coarse_mismatch_indices = np.where(valid_mask & ~coarse_match)[0]
    coarse_mismatch_pairs = Counter()
    coarse_error_pairs = Counter()
    for idx in coarse_mismatch_indices:
        cg0, cg1 = coarse0[idx], coarse1[idx]
        coarse_mismatch_pairs[(cg0, cg1)] += 1
        l0 = id2label.get(labels0[idx], str(labels0[idx]))
        l1 = id2label.get(labels1[idx], str(labels1[idx]))
        coarse_error_pairs[(cg0, cg1, l0, l1)] += 1
    top_coarse_mismatch_pairs = [
        {'coarse0': p[0], 'coarse1': p[1], 'count': c}
        for p, c in coarse_mismatch_pairs.most_common(10)
    ]

    # Build result
    result = {
        'num_pred_matches': num_matches,
        'num_valid_semantic_matches': int(num_valid),
        'num_out_of_bounds': num_out_of_bounds,
        'missing_segmentation': False,
        'fine_consistent_count': int(fine_consistent),
        'fine_error_count': int(fine_mismatch),
        'fine_consistency_rate': float(fine_consistency_rate),
        'fine_error_rate': float(fine_error_rate),
        'coarse_consistent_count': int(coarse_consistent),
        'coarse_error_count': int(coarse_mismatch),
        'coarse_consistency_rate': float(coarse_consistency_rate),
        'coarse_error_rate': float(coarse_error_rate),
        # Diagnostic fields (prove coarse mapping is working)
        'num_fine_error_but_coarse_correct': fine_error_but_coarse_correct,
        'fine_error_coarse_correct_rate': float(fine_error_coarse_correct_rate),
        'num_known_coarse_labels': num_known_coarse_labels,
        'num_unknown_coarse_labels': num_unknown_coarse_labels,
        'unknown_label_top20': unknown_label_top20,
        'high_conf_stats': high_conf_stats,
        'top_fine_mismatch_pairs': top_fine_mismatch_pairs,
        'top_coarse_mismatch_pairs': top_coarse_mismatch_pairs,
        'coarse_error_pair_top20': [
            {'coarse0': p[0], 'coarse1': p[1], 'fine0': p[2], 'fine1': p[3], 'count': c}
            for p, c in coarse_error_pairs.most_common(20)
        ],
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
    target_pairs: Dict[str, dict] = None,
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
        target_pairs: Dict mapping canonical pair_key -> {image0, image1, ...}
                     If provided, only pairs in this dict will have per-match saved.
                     Other pairs get per-pair stats only.

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

    # Convert to numpy if needed
    if hasattr(mkpts0, 'cpu'):
        mkpts0 = mkpts0.detach().cpu().numpy()
    if hasattr(mkpts1, 'cpu'):
        mkpts1 = mkpts1.detach().cpu().numpy()
    if mconf is not None and hasattr(mconf, 'cpu'):
        mconf = mconf.detach().cpu().numpy()

    # Handle batch dimension
    if mkpts0.ndim == 3:
        mkpts0 = mkpts0[0]  # [M, 2]
        mkpts1 = mkpts1[0]
        mconf = mconf[0] if mconf is not None else np.ones(len(mkpts0))

    # Ensure mconf exists
    if mconf is None:
        mconf = np.ones(len(mkpts0))

    # Get image paths (normalize from batch format)
    image0_path = None
    image1_path = None

    if 'pair_names' in batch:
        pair_names = batch['pair_names']
        if isinstance(pair_names, (list, tuple)) and len(pair_names) >= 2:
            image0_path = pair_names[0]
            image1_path = pair_names[1]
    elif 'image0_path' in batch:
        image0_path = batch['image0_path']
        image1_path = batch['image1_path']

    # Normalize paths (unwrap list/repr wrappers)
    if image0_path is not None:
        image0_path = normalize_rel_image_path(image0_path)
    if image1_path is not None:
        image1_path = normalize_rel_image_path(image1_path)

    # Build canonical pair key for target pair matching
    pair_key = None
    is_target_pair = False
    if image0_path and image1_path:
        norm0, norm1 = sorted([image0_path, image1_path])
        pair_key = f"{norm0}|||{norm1}"
        if target_pairs and pair_key in target_pairs:
            is_target_pair = True

    if image0_path is None or image1_path is None:
        print("WARNING: Could not determine image paths from batch")
        return results

    # Helper to build expected npz path for debug output
    def _expected_npz(img_path):
        safe = image_path_to_safe_stem(img_path)
        return str(seg_store.npz_dir / f"{safe}.npz")

    # Get segmentation data
    seg0 = seg_store.get(image0_path)
    seg1 = seg_store.get(image1_path)

    if seg0 is None or seg1 is None:
        exp0 = _expected_npz(image0_path)
        exp1 = _expected_npz(image1_path)
        return [{
            'num_pred_matches': len(mkpts0),
            'num_valid_semantic_matches': 0,
            'num_out_of_bounds': 0,
            'missing_segmentation': True,
            'fine_consistent_count': 0,
            'fine_error_count': 0,
            'fine_consistency_rate': 0.0,
            'fine_error_rate': 0.0,
            'coarse_consistent_count': 0,
            'coarse_error_count': 0,
            'coarse_consistency_rate': 0.0,
            'coarse_error_rate': 0.0,
            'image0': image0_path,
            'image1': image1_path,
            'image0_rel_path': image0_path,
            'image1_rel_path': image1_path,
            'expected_seg0_path': exp0,
            'expected_seg1_path': exp1,
            'exists_seg0': os.path.exists(exp0),
            'exists_seg1': os.path.exists(exp1),
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
    # Only save per-match if explicitly requested AND (this is a target pair OR no target_pairs specified)
    # When target_pairs is None/empty, save per-match for all pairs
    should_save_per_match = save_all_matches and (is_target_pair or target_pairs is None or len(target_pairs) == 0)

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
        save_all_matches=should_save_per_match,
    )

    result['image0'] = image0_path
    result['image1'] = image1_path
    result['pair_key'] = pair_key
    result['missing_segmentation'] = False

    # Save coordinate system information for visualization
    # mkpts0_f/mkpts1_f are in processed image coords (batch['hw0_i'])
    # seg0/seg1 are in OneFormer segmentation coords
    result['hw0_i'] = list(match_hw)  # CoMatch input image size [H, W]
    result['hw1_i'] = list(match_hw)  # CoMatch input image size for image1
    result['seg_hw0'] = list(seg_hw)  # OneFormer segmentation size [H, W]
    result['seg_hw1'] = (seg1['height'], seg1['width'])  # OneFormer segmentation size for image1

    # Also save original image size from segmentation if available
    result['orig_hw0'] = seg0.get('orig_hw', list(seg_hw))
    result['orig_hw1'] = seg1.get('orig_hw', (seg1['height'], seg1['width']))

    # Save whether coordinate mapping was needed
    result['coord_mapped'] = (match_hw != seg_hw)
    if match_hw != seg_hw:
        result['coord_scale'] = {
            'scale0_x': seg_hw[1] / match_hw[1],  # seg_w / match_w
            'scale0_y': seg_hw[0] / match_hw[0],  # seg_h / match_h
            'scale1_x': seg1['width'] / match_hw[1] if match_hw[1] != 0 else 1.0,
            'scale1_y': seg1['height'] / match_hw[0] if match_hw[0] != 0 else 1.0,
        }

    # Save scale0/scale1 from batch if available (original image / processed image ratio)
    # scale0/scale1 are [scale_w, scale_h] = [orig_w/processed_w, orig_h/processed_h]
    if 'scale0' in batch:
        scale0 = batch['scale0']
        if torch.is_tensor(scale0):
            scale0 = scale0.cpu().numpy().tolist()
        result['batch_scale0'] = scale0
    if 'scale1' in batch:
        scale1 = batch['scale1']
        if torch.is_tensor(scale1):
            scale1 = scale1.cpu().numpy().tolist()
        result['batch_scale1'] = scale1

    # If this is a target pair and we need enhanced per-match data, add seg coordinates
    if should_save_per_match and 'per_match' in result:
        # Add segmentation-mapped coordinates to per_match entries
        for entry in result['per_match']:
            idx = entry['idx']
            if idx < len(mkpts0_mapped):
                entry['seg_x0'] = float(mkpts0_mapped[idx, 0])
                entry['seg_y0'] = float(mkpts0_mapped[idx, 1])
                entry['seg_x1'] = float(mkpts1_mapped[idx, 0])
                entry['seg_y1'] = float(mkpts1_mapped[idx, 1])
            # Add label names for convenience
            label_id0 = entry.get('label0')
            label_id1 = entry.get('label1')
            if label_id0 is not None and label_id0 >= 0:
                entry['label0_name'] = seg0['id2label'].get(label_id0, str(label_id0))
            if label_id1 is not None and label_id1 >= 0:
                entry['label1_name'] = seg1['id2label'].get(label_id1, str(label_id1))
            # Add fine/coarse consistency as booleans for visualization
            entry['fine_consistent'] = bool(entry.get('fine_match', False))
            entry['coarse_consistent'] = bool(entry.get('coarse_match', False))

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
            'num_pairs_total': 0,
            'num_pairs_evaluated': 0,
            'num_unique_images': 0,
            'num_pred_matches_total': 0,
            'num_valid_semantic_matches': 0,
            'num_out_of_bounds_matches': 0,
            'num_missing_segmentation_pairs': 0,
            'num_pairs_without_matches': 0,
            'fine_consistent_matches': 0,
            'fine_error_matches': 0,
            'fine_consistency_rate': 0.0,
            'fine_error_rate': 0.0,
            'coarse_consistent_matches': 0,
            'coarse_error_matches': 0,
            'coarse_consistency_rate': 0.0,
            'coarse_error_rate': 0.0,
            'main_error_definition': 'coarse_error',
            'main_semantic_error_matches': 0,
            'main_semantic_error_rate': 0.0,
        }

    total_pairs = len(pair_results)
    missing_count = sum(1 for r in pair_results if r.get('missing_segmentation', False))
    valid_pairs = total_pairs - missing_count

    # Count unique images
    all_images = set()
    for r in pair_results:
        if 'image0' in r:
            all_images.add(r['image0'])
        if 'image1' in r:
            all_images.add(r['image1'])
    num_unique_images = len(all_images)

    # Aggregate match counts (back-compat: try both old and new field names)
    total_pred = sum(r.get('num_pred_matches', r.get('num_matches', 0)) for r in pair_results)
    total_valid = sum(
        r.get('num_valid_semantic_matches', r.get('num_valid', 0))
        for r in pair_results
    )
    total_oob = sum(r.get('num_out_of_bounds', 0) for r in pair_results)
    no_match_count = sum(
        1 for r in pair_results
        if r.get('num_pred_matches', r.get('num_matches', 0)) == 0
    )

    # Aggregate consistency
    total_fine_consistent = sum(
        r.get('fine_consistent_count', r.get('fine_consistent', 0))
        for r in pair_results
    )
    total_fine_error = sum(
        r.get('fine_error_count', r.get('fine_mismatch', 0))
        for r in pair_results
    )
    total_coarse_consistent = sum(
        r.get('coarse_consistent_count', r.get('coarse_consistent', 0))
        for r in pair_results
    )
    total_coarse_error = sum(
        r.get('coarse_error_count', r.get('coarse_mismatch', 0))
        for r in pair_results
    )

    # Rates (denominator = num_valid_semantic_matches)
    fine_consistency_rate = total_fine_consistent / total_valid if total_valid > 0 else 0.0
    fine_error_rate = total_fine_error / total_valid if total_valid > 0 else 0.0
    coarse_consistency_rate = total_coarse_consistent / total_valid if total_valid > 0 else 0.0
    coarse_error_rate = total_coarse_error / total_valid if total_valid > 0 else 0.0

    # Aggregate diagnostic fields for coarse mapping verification
    total_fine_error_but_coarse_correct = sum(
        r.get('num_fine_error_but_coarse_correct', 0) for r in pair_results
    )
    # Weighted average of fine_error_coarse_correct_rate
    weighted_fecc = sum(
        r.get('fine_error_coarse_correct_rate', 0.0) * r.get('num_valid_semantic_matches', 0)
        for r in pair_results
    )
    global_fine_error_coarse_correct_rate = (
        weighted_fecc / total_valid if total_valid > 0 else 0.0
    )

    # Aggregate unknown labels across all pairs
    from collections import Counter as _Counter
    global_unknown_counts = _Counter()
    for r in pair_results:
        for entry in r.get('unknown_label_top20', []):
            global_unknown_counts[entry['label']] += entry['count']
    global_unknown_label_top20 = [
        {'label': l, 'count': c} for l, c in global_unknown_counts.most_common(20)
    ]

    # Take max of per-pair known/unknown counts (they're per-dataset estimates)
    global_num_known = max((r.get('num_known_coarse_labels', 0) for r in pair_results), default=0)
    global_num_unknown = max((r.get('num_unknown_coarse_labels', 0) for r in pair_results), default=0)

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
    all_fine_mismatches = _Counter()
    all_coarse_mismatches = _Counter()
    all_coarse_error_pairs = _Counter()

    for r in pair_results:
        for pair in r.get('top_fine_mismatch_pairs', []):
            key = (pair['label0'], pair['label1'])
            all_fine_mismatches[key] += pair['count']
        for pair in r.get('top_coarse_mismatch_pairs', []):
            key = (pair['coarse0'], pair['coarse1'])
            all_coarse_mismatches[key] += pair['count']
        for pair in r.get('coarse_error_pair_top20', []):
            key = (pair['coarse0'], pair['coarse1'], pair['fine0'], pair['fine1'])
            all_coarse_error_pairs[key] += pair['count']

    global_top_fine_mismatch = [
        {'label0': p[0], 'label1': p[1], 'count': c}
        for p, c in all_fine_mismatches.most_common(20)
    ]

    global_top_coarse_mismatch = [
        {'coarse0': p[0], 'coarse1': p[1], 'count': c}
        for p, c in all_coarse_mismatches.most_common(20)
    ]

    global_coarse_error_pair_top20 = [
        {'coarse0': p[0], 'coarse1': p[1], 'fine0': p[2], 'fine1': p[3], 'count': c}
        for p, c in all_coarse_error_pairs.most_common(20)
    ]

    return {
        'num_pairs_total': total_pairs,
        'num_pairs_evaluated': valid_pairs,
        'num_unique_images': num_unique_images,
        'num_pred_matches_total': total_pred,
        'num_valid_semantic_matches': total_valid,
        'num_out_of_bounds_matches': total_oob,
        'num_missing_segmentation_pairs': missing_count,
        'num_pairs_without_matches': no_match_count,
        'fine_consistent_matches': total_fine_consistent,
        'fine_error_matches': total_fine_error,
        'fine_consistency_rate': fine_consistency_rate,
        'fine_error_rate': fine_error_rate,
        'coarse_consistent_matches': total_coarse_consistent,
        'coarse_error_matches': total_coarse_error,
        'coarse_consistency_rate': coarse_consistency_rate,
        'coarse_error_rate': coarse_error_rate,
        'main_error_definition': 'coarse_error',
        'main_semantic_error_matches': total_coarse_error,
        'main_semantic_error_rate': coarse_error_rate,
        # Coarse mapping diagnostic fields
        'num_fine_error_but_coarse_correct': total_fine_error_but_coarse_correct,
        'fine_error_coarse_correct_rate': global_fine_error_coarse_correct_rate,
        'num_known_coarse_labels': global_num_known,
        'num_unknown_coarse_labels': global_num_unknown,
        'unknown_label_top20': global_unknown_label_top20,
        'global_high_conf_stats': global_high_conf,
        'global_top_fine_mismatch_pairs': global_top_fine_mismatch,
        'global_top_coarse_mismatch_pairs': global_top_coarse_mismatch,
        'global_coarse_error_pair_top20': global_coarse_error_pair_top20,
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