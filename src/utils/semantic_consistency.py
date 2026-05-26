"""
Semantic consistency statistics for CoMatch evaluation.

IMPORTANT NOTE:
----------------
MegaDepth:
  CoMatch maps mkpts0_f / mkpts1_f back to original image coordinates via scale0/scale1.
  Semantic label maps are in original image resolution (H_orig x W_orig).
  Direct lookup is correct: label = sem[int(y), int(x)]

ScanNet:
  CoMatch maps mkpts0_f / mkpts1_f back to the RESIZED space (scannetX x scannetY),
  NOT the original image resolution. The scale0/scale1 are [1,1] for default 640x480.
  Semantic label maps are in original image resolution (H_orig x W_orig).
  You MUST pass mkpts_space_size=(640, 480) to compute_semantic_match_stats,
  which will scale coordinates: x_orig = x_mkpt * (orig_w / 640)

DO NOT:
- Use image0/image1 tensor dimensions as semantic label map size
- Map keypoints back to 832/padded dimensions
- Multiply mkpts*_f by scale0/scale1 again
- For ScanNet: do NOT use mkpts directly on original-resolution sem maps without scaling
"""

import json
import os
import hashlib
from pathlib import Path
import numpy as np


def normalize_semantic_path_key(path: str) -> str:
    """
    Normalize a path to be used as a semantic cache manifest key.

    - Converts Windows backslashes to forward slashes
    - Removes leading ./
    - Strips trailing slashes

    Args:
        path: Image path (may be absolute or relative, Windows or Linux style)

    Returns:
        Normalized path with forward slashes and no leading ./
    """
    if path is None:
        return None
    normalized = str(path).replace('\\', '/').strip()
    if normalized.startswith('./'):
        normalized = normalized[2:]
    # Remove trailing slash
    normalized = normalized.rstrip('/')
    return normalized


def get_path_suffix(path: str, n_components: int = 2) -> str:
    """
    Get the last N components of a normalized path.

    Used for suffix matching when manifest keys and query paths
    may have different prefixes.

    Args:
        path: Normalized path
        n_components: Number of trailing components to return

    Returns:
        Path suffix like "scene_name/image.jpg"
    """
    parts = path.replace('\\', '/').split('/')
    if len(parts) >= n_components:
        return '/'.join(parts[-n_components:])
    return path


def compute_file_hash(file_path: str, n_bytes: int = 8192) -> str:
    """
    Compute a short hash of a file for naming npz files.

    Uses first N bytes of file + file size.

    Args:
        file_path: Path to file
        n_bytes: Number of bytes to read for hash

    Returns:
        12-character hex string
    """
    h = hashlib.sha1()
    with open(file_path, 'rb') as f:
        h.update(f.read(n_bytes))
    h.update(str(os.path.getsize(file_path)).encode())
    return h.hexdigest()[:12]


class SemanticLabelCache:
    """
    Cache for semantic segmentation label maps.

    Loads label maps from pre-computed npz files based on image paths.

    Args:
        cache_dir: Directory containing manifest.json and npz files
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._memory_cache = {}
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load manifest.json mapping image paths to npz files."""
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        normalized_manifest = {}
        for img_path, npz_path in manifest.items():
            # Normalize path separators and remove leading ./
            normalized_img = normalize_semantic_path_key(img_path)
            # npz path may be relative to cache_dir or already a full relative path
            if not npz_path.startswith('semantic_cache'):
                # Assume it's relative to cache_dir
                npz_full = str(self.cache_dir / npz_path)
            else:
                npz_full = str(self.cache_dir / npz_path)
            normalized_manifest[normalized_img] = npz_full

        return normalized_manifest

    def _find_npz_path(self, image_path: str) -> str:
        """
        Find npz path for an image path.

        Supports both absolute and relative paths from batch data.
        Uses suffix matching for flexibility when paths differ in prefix.

        Args:
            image_path: Image path (may be absolute or relative)

        Returns:
            Path to npz file

        Raises:
            FileNotFoundError: If no matching npz found
        """
        normalized = normalize_semantic_path_key(image_path)

        # Direct match
        if normalized in self._manifest:
            return self._manifest[normalized]

        # Try suffix matching
        suffix = get_path_suffix(normalized, n_components=2)

        candidates = []
        for img_key, npz_path in self._manifest.items():
            key_suffix = get_path_suffix(img_key, n_components=2)
            if suffix == key_suffix:
                candidates.append(npz_path)

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous path: {image_path}\n"
                f"Multiple matches found: {candidates}\n"
                f"Manifest keys: {list(self._manifest.keys())[:5]}..."
            )

        raise FileNotFoundError(
            f"No semantic label found for: {image_path}\n"
            f"Available entries: {list(self._manifest.keys())[:5]}..."
        )

    def get_label(self, image_path: str) -> np.ndarray:
        """
        Load semantic label map for an image.

        Args:
            image_path: Path to the image

        Returns:
            label: H x W numpy array of label IDs
        """
        # Check memory cache
        if image_path in self._memory_cache:
            return self._memory_cache[image_path]

        # Find npz path
        npz_path = self._find_npz_path(image_path)

        # Load npz
        data = np.load(npz_path)
        label = data['label']

        # Cache in memory
        self._memory_cache[image_path] = label

        return label


def compute_semantic_match_stats(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    sem0: np.ndarray,
    sem1: np.ndarray,
    conf: np.ndarray = None,
    ignore_labels: set = None,
    conf_thr: float = None,
    mkpts_space_size: tuple = None,
) -> dict:
    """
    Compute semantic consistency statistics for matches.

    Args:
        mkpts0: Nx2 array, match coordinates [x, y]
        mkpts1: Nx2 array, match coordinates [x, y]
        sem0: H0 x W0 label map for image 0 (original resolution)
        sem1: H1 x W1 label map for image 1 (original resolution)
        conf: N confidence scores (optional)
        ignore_labels: Set of label IDs to ignore (optional)
        conf_thr: Minimum confidence threshold (optional)
        mkpts_space_size: (W, H) tuple indicating the coordinate space of mkpts.
            If provided and different from sem shape, mkpts will be scaled
            to match sem resolution. For ScanNet, this is typically (640, 480).
            For MegaDepth, mkpts are already in original resolution, so omit this.

    Returns:
        Dictionary with statistics:
        - num_matches: Total matches
        - num_after_conf: Matches after confidence filtering
        - num_in_bounds: Matches with in-bounds coordinates
        - num_valid_semantic: Matches with valid (non-ignored) labels
        - num_same_semantic: Matches with same semantic label
        - num_cross_semantic: Matches with different semantic labels
        - cross_semantic_rate: ratio of cross-semantic to valid semantic
    """
    # Scale mkpts to sem resolution if mkpts_space_size is provided
    if mkpts_space_size is not None:
        sp_w, sp_h = mkpts_space_size
        sem0_h, sem0_w = sem0.shape
        sem1_h, sem1_w = sem1.shape
        if sp_w != sem0_w or sp_h != sem0_h:
            mkpts0 = mkpts0.copy()
            mkpts0[:, 0] = mkpts0[:, 0] * (sem0_w / sp_w)
            mkpts0[:, 1] = mkpts0[:, 1] * (sem0_h / sp_h)
        if sp_w != sem1_w or sp_h != sem1_h:
            mkpts1 = mkpts1.copy()
            mkpts1[:, 0] = mkpts1[:, 0] * (sem1_w / sp_w)
            mkpts1[:, 1] = mkpts1[:, 1] * (sem1_h / sp_h)

    N = len(mkpts0)
    num_matches = N

    # Filter by confidence
    if conf_thr is not None and conf is not None:
        valid_conf = conf >= conf_thr
        mkpts0 = mkpts0[valid_conf]
        mkpts1 = mkpts1[valid_conf]
        if conf is not None:
            conf = conf[valid_conf]
        num_after_conf = len(mkpts0)
    else:
        num_after_conf = N

    # Convert to integer coordinates
    x0_int = np.rint(mkpts0[:, 0]).astype(int)
    y0_int = np.rint(mkpts0[:, 1]).astype(int)
    x1_int = np.rint(mkpts1[:, 0]).astype(int)
    y1_int = np.rint(mkpts1[:, 1]).astype(int)

    # Boundary check
    in_bounds0 = (x0_int >= 0) & (x0_int < sem0.shape[1]) & (y0_int >= 0) & (y0_int < sem0.shape[0])
    in_bounds1 = (x1_int >= 0) & (x1_int < sem1.shape[1]) & (y1_int >= 0) & (y1_int < sem1.shape[0])
    in_bounds = in_bounds0 & in_bounds1

    num_in_bounds = np.sum(in_bounds)

    # Get labels for in-bounds points (use [y, x] indexing)
    labels0 = sem0[y0_int[in_bounds], x0_int[in_bounds]]
    labels1 = sem1[y1_int[in_bounds], x1_int[in_bounds]]

    # Filter ignored labels
    if ignore_labels:
        valid_label0 = ~np.isin(labels0, list(ignore_labels))
        valid_label1 = ~np.isin(labels1, list(ignore_labels))
        valid_semantic = valid_label0 & valid_label1
    else:
        valid_semantic = np.ones(len(labels0), dtype=bool)

    num_valid_semantic = np.sum(valid_semantic)

    # Compare labels
    same_semantic = labels0[valid_semantic] == labels1[valid_semantic]
    num_same_semantic = np.sum(same_semantic)
    num_cross_semantic = num_valid_semantic - num_same_semantic

    # Cross semantic rate
    if num_valid_semantic > 0:
        cross_semantic_rate = num_cross_semantic / num_valid_semantic
    else:
        cross_semantic_rate = float("nan")

    return {
        "num_matches": num_matches,
        "num_after_conf": num_after_conf,
        "num_in_bounds": num_in_bounds,
        "num_valid_semantic": num_valid_semantic,
        "num_same_semantic": num_same_semantic,
        "num_cross_semantic": num_cross_semantic,
        "cross_semantic_rate": cross_semantic_rate,
    }


def validate_semantic_cache(cache_dir: str) -> dict:
    """
    Validate the semantic cache directory.

    Checks:
    - manifest.json exists
    - All npz files exist
    - Each npz has 'label' array with correct shape
    - If image file is accessible, label shape matches original image

    Args:
        cache_dir: Path to semantic cache directory

    Returns:
        Dictionary with validation results:
        - valid: bool, True if all checks pass
        - total_entries: int
        - valid_entries: int
        - missing_files: list of missing npz paths
        - bad_shapes: list of (path, expected_shape, actual_shape)
        - unreadable_images: list of (path, error)
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    results = {
        "valid": True,
        "total_entries": 0,
        "valid_entries": 0,
        "missing_files": [],
        "bad_shapes": [],
        "unreadable_images": [],
    }

    # Check manifest exists
    if not manifest_path.exists():
        results["valid"] = False
        results["error"] = f"Manifest not found: {manifest_path}"
        return results

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    results["total_entries"] = len(manifest)

    # Validate each entry
    for img_path, npz_rel_path in manifest.items():
        npz_path = cache_dir / npz_rel_path

        # Check npz exists
        if not npz_path.exists():
            results["missing_files"].append(str(npz_path))
            results["valid"] = False
            continue

        try:
            data = np.load(npz_path)

            # Check label exists
            if 'label' not in data:
                results["bad_shapes"].append((str(npz_path), "no 'label' key", None))
                results["valid"] = False
                continue

            label = data['label']

            # Check label is 2D
            if label.ndim != 2:
                results["bad_shapes"].append((str(npz_path), "2D array", f"{label.ndim}D"))
                results["valid"] = False
                continue

            # If height/width metadata exists, check shape
            if 'height' in data and 'width' in data:
                expected_h = int(np.ravel(data['height'])[0]) if hasattr(data['height'], '__iter__') else int(data['height'])
                expected_w = int(np.ravel(data['width'])[0]) if hasattr(data['width'], '__iter__') else int(data['width'])
                if label.shape != (expected_h, expected_w):
                    results["bad_shapes"].append(
                        (str(npz_path), f"({expected_h}, {expected_w})", label.shape)
                    )
                    results["valid"] = False
                    continue

            # If original image exists, check shape using PIL if available
            stored_image_path = data.get('image_path')
            if stored_image_path and os.path.exists(stored_image_path):
                try:
                    from PIL import Image
                    img = Image.open(stored_image_path)
                    orig_w, orig_h = img.size  # PIL gives (width, height)
                    if label.shape != (orig_h, orig_w):
                        results["bad_shapes"].append(
                            (str(npz_path), f"({orig_h}, {orig_w}) from image", label.shape)
                        )
                        results["valid"] = False
                        continue
                except Exception as e:
                    results["unreadable_images"].append((str(npz_path), str(e)))

            results["valid_entries"] += 1

        except Exception as e:
            results["bad_shapes"].append((str(npz_path), "read error", str(e)))
            results["valid"] = False

    return results