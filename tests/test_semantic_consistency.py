"""Tests for semantic_consistency.py"""

import json
import numpy as np
import os
import pytest
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.semantic_consistency import (
    compute_semantic_match_stats,
    normalize_semantic_path_key,
    get_path_suffix,
    compute_file_hash,
    validate_semantic_cache,
    SemanticLabelCache,
)


class TestComputeSemanticMatchStats:
    """Test compute_semantic_match_stats function."""

    def test_basic_same_semantic(self):
        """Test basic case with same semantic labels."""
        # 10x10 label maps
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem1 = np.zeros((10, 10), dtype=np.int32)

        # Set different regions with label 1
        sem0[2:5, 2:5] = 1
        sem1[2:5, 2:5] = 1

        # One match within same label region (all at label 1)
        mkpts0 = np.array([[3.0, 3.0]])  # x, y
        mkpts1 = np.array([[3.0, 3.0]])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 1
        assert result['num_after_conf'] == 1
        assert result['num_in_bounds'] == 1
        assert result['num_valid_semantic'] == 1
        assert result['num_same_semantic'] == 1
        assert result['num_cross_semantic'] == 0
        assert result['cross_semantic_rate'] == 0.0

    def test_cross_semantic(self):
        """Test cross-semantic matching."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem0[2:5, 2:5] = 1  # Region with label 1

        sem1 = np.zeros((10, 10), dtype=np.int32)
        sem1[2:5, 2:5] = 2  # Region with label 2

        # Match across different labels
        mkpts0 = np.array([[3.0, 3.0]])  # x, y in label 1 region
        mkpts1 = np.array([[3.0, 3.0]])  # x, y in label 2 region

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 1
        assert result['num_in_bounds'] == 1
        assert result['num_valid_semantic'] == 1
        assert result['num_same_semantic'] == 0
        assert result['num_cross_semantic'] == 1
        assert result['cross_semantic_rate'] == 1.0

    def test_out_of_bounds(self):
        """Test out-of-bounds coordinate handling."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem1 = np.zeros((10, 10), dtype=np.int32)

        # Match with coordinates outside image bounds
        # Only the first point is in bounds, the rest have x>9 or y<0
        mkpts0 = np.array([[3.0, 3.0], [15.0, 3.0], [3.0, -1.0], [15.0, -1.0]])
        mkpts1 = np.array([[3.0, 3.0], [15.0, 3.0], [3.0, -1.0], [15.0, -1.0]])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 4
        # Only first point (x=3, y=3) is within [0,10) for both images
        assert result['num_in_bounds'] == 1
        assert result['num_valid_semantic'] == 1

    def test_ignore_labels(self):
        """Test ignore_labels functionality."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem1 = np.zeros((10, 10), dtype=np.int32)

        # Set label 255 in specific regions
        sem0[2, 2] = 255  # Should be ignored
        sem0[3, 3] = 1   # Should be counted
        sem1[2, 2] = 255  # Should be ignored
        sem1[3, 3] = 2   # Different label but should still be valid

        # Match points
        mkpts0 = np.array([[2.0, 2.0], [3.0, 3.0]])
        mkpts1 = np.array([[2.0, 2.0], [3.0, 3.0]])

        result = compute_semantic_match_stats(
            mkpts0, mkpts1, sem0, sem1,
            ignore_labels={255}
        )

        # First match is ignored (label 255)
        assert result['num_matches'] == 2
        assert result['num_in_bounds'] == 2
        assert result['num_valid_semantic'] == 1  # Only second match counts
        assert result['num_same_semantic'] == 0
        assert result['num_cross_semantic'] == 1

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem1 = np.zeros((10, 10), dtype=np.int32)

        mkpts0 = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        mkpts1 = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        conf = np.array([0.9, 0.5, 0.3])

        result = compute_semantic_match_stats(
            mkpts0, mkpts1, sem0, sem1,
            conf=conf,
            conf_thr=0.5
        )

        assert result['num_matches'] == 3
        assert result['num_after_conf'] == 2  # Only 2 above 0.5 threshold

    def test_empty_matches(self):
        """Test with no matches."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem1 = np.zeros((10, 10), dtype=np.int32)

        mkpts0 = np.empty((0, 2))
        mkpts1 = np.empty((0, 2))

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 0
        assert result['num_after_conf'] == 0
        assert result['num_in_bounds'] == 0
        assert result['num_valid_semantic'] == 0
        assert np.isnan(result['cross_semantic_rate'])

    def test_no_valid_semantic(self):
        """Test when all matches have ignored labels."""
        sem0 = np.full((10, 10), 255, dtype=np.int32)
        sem1 = np.full((10, 10), 255, dtype=np.int32)

        mkpts0 = np.array([[3.0, 3.0], [4.0, 4.0]])
        mkpts1 = np.array([[3.0, 3.0], [4.0, 4.0]])

        result = compute_semantic_match_stats(
            mkpts0, mkpts1, sem0, sem1,
            ignore_labels={255}
        )

        assert result['num_matches'] == 2
        assert result['num_in_bounds'] == 2
        assert result['num_valid_semantic'] == 0
        assert np.isnan(result['cross_semantic_rate'])

    def test_coordinate_ordering(self):
        """Verify that coordinates are [x, y] and indexing is [y, x]."""
        sem0 = np.zeros((100, 200), dtype=np.int32)
        sem1 = np.zeros((100, 200), dtype=np.int32)

        # Set a specific label at (x=50, y=30)
        sem0[30, 50] = 42
        sem1[30, 50] = 42  # Both images have same label

        # Match point at (x=50, y=30) on both images
        mkpts0 = np.array([[50.0, 30.0]])  # [x, y]
        mkpts1 = np.array([[50.0, 30.0]])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        # Should find same label (42) at the matched location
        assert result['num_matches'] == 1
        assert result['num_valid_semantic'] == 1
        # Both should get label 42
        assert result['num_same_semantic'] == 1

    def test_multiple_matches_mixed(self):
        """Test with multiple matches of different types."""
        sem0 = np.zeros((20, 20), dtype=np.int32)
        sem1 = np.zeros((20, 20), dtype=np.int32)

        # Define regions
        sem0[0:5, 0:5] = 1   # Label 1
        sem0[5:10, 0:5] = 2  # Label 2
        sem1[0:5, 0:5] = 1   # Label 1
        sem1[5:10, 0:5] = 3  # Label 3 (cross semantic)

        # Create matches: same semantic, cross semantic, out of bounds
        mkpts0 = np.array([
            [2.0, 2.0],   # Same semantic (both label 1)
            [7.0, 2.0],   # Cross semantic (label 2 vs 3)
            [7.0, 7.0],   # Out of bounds (y=7 not in region)
        ])
        mkpts1 = np.array([
            [2.0, 2.0],
            [7.0, 7.0],   # y=7 is in bounds but sem1 has no label there
            [7.0, 7.0],
        ])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 3
        # Check in_bounds: all x in [0,20), some y issues
        assert result['num_in_bounds'] == 3  # All coordinates are in [0,20)

    def test_rounding_coordinates(self):
        """Test that coordinates are properly rounded."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem0[3, 2] = 5  # Set label 5 at rounded position

        sem1 = np.zeros((10, 10), dtype=np.int32)
        sem1[3, 2] = 5

        # Coordinates with .5 that should round to 2 or 3
        mkpts0 = np.array([[2.4, 3.1]])  # Rounds to (2, 3)
        mkpts1 = np.array([[2.4, 3.1]])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 1
        assert result['num_in_bounds'] == 1
        assert result['num_valid_semantic'] == 1
        # Both should get label 5
        assert result['num_same_semantic'] == 1


class TestEdgeCases:
    """Test edge cases."""

    def test_different_shape_labelmaps(self):
        """Test with differently sized label maps."""
        sem0 = np.zeros((10, 10), dtype=np.int32)
        sem1 = np.zeros((15, 20), dtype=np.int32)

        sem0[5, 5] = 1
        sem1[5, 5] = 1

        mkpts0 = np.array([[5.0, 5.0]])
        mkpts1 = np.array([[5.0, 5.0]])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        assert result['num_matches'] == 1
        # Point is in bounds for both (10x10 and 15x20)
        assert result['num_in_bounds'] == 1

    def test_float_coordinates(self):
        """Test with float coordinate values."""
        sem0 = np.zeros((100, 100), dtype=np.int32)
        sem1 = np.zeros((100, 100), dtype=np.int32)

        mkpts0 = np.array([[50.7, 49.3]])
        mkpts1 = np.array([[50.7, 49.3]])

        result = compute_semantic_match_stats(mkpts0, mkpts1, sem0, sem1)

        # Should round to (51, 49) which is still in bounds
        assert result['num_matches'] == 1
        assert result['num_in_bounds'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestPathNormalization:
    """Test path normalization utilities."""

    def test_normalize_windows_paths(self):
        """Test Windows backslash conversion."""
        path = r"data\megadepth\test\images\scene_01\img1.jpg"
        normalized = normalize_semantic_path_key(path)
        assert normalized == "data/megadepth/test/images/scene_01/img1.jpg"
        assert "\\" not in normalized

    def test_normalize_strip_leading_dotslash(self):
        """Test stripping leading ./"""
        path = "./relative/path/image.jpg"
        normalized = normalize_semantic_path_key(path)
        assert not normalized.startswith("./")
        assert normalized == "relative/path/image.jpg"

    def test_normalize_trailing_slash(self):
        """Test stripping trailing slashes."""
        path = "path/to/dir///"
        normalized = normalize_semantic_path_key(path)
        assert not normalized.endswith("/")

    def test_get_path_suffix_two_components(self):
        """Test getting last 2 components of path."""
        path = "data/megadepth/test/images/scene_01/img1.jpg"
        suffix = get_path_suffix(path, n_components=2)
        assert suffix == "scene_01/img1.jpg"

    def test_get_path_suffix_one_component(self):
        """Test getting last component."""
        path = "image.jpg"
        suffix = get_path_suffix(path, n_components=1)
        assert suffix == "image.jpg"


class TestSemanticLabelCache:
    """Test SemanticLabelCache class."""

    def setup_method(self):
        """Create temp cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_npz(self, path: Path, height: int, width: int):
        """Create a test npz file."""
        np.savez_compressed(
            path,
            label=np.zeros((height, width), dtype=np.uint16),
            image_path="dummy",
            height=height,
            width=width,
        )

    def test_cache_with_labels_prefix(self):
        """Test manifest with labels/xxx.npz format."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create test npz
        npz_path = labels_dir / "abc123.npz"
        self._create_test_npz(npz_path, 100, 200)

        # Create manifest
        manifest = {"relative/path/image.jpg": "labels/abc123.npz"}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Load cache
        cache = SemanticLabelCache(str(self.cache_dir))

        # Should be able to find the npz
        result_path = cache._find_npz_path("relative/path/image.jpg")
        assert result_path == str(npz_path)

    def test_cache_with_absolute_path_npz(self):
        """Test manifest with absolute path npz values."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create test npz
        npz_path = labels_dir / "def456.npz"
        self._create_test_npz(npz_path, 100, 200)

        # Create manifest with absolute path
        manifest = {"relative/path/image.jpg": str(npz_path)}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Load cache
        cache = SemanticLabelCache(str(self.cache_dir))

        result_path = cache._find_npz_path("relative/path/image.jpg")
        assert result_path == str(npz_path)

    def test_cache_suffix_matching(self):
        """Test suffix matching when paths differ in prefix."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create test npz
        npz_path = labels_dir / "ghi789.npz"
        self._create_test_npz(npz_path, 100, 200)

        # Manifest with relative path
        manifest = {"scene/img.jpg": "labels/ghi789.npz"}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Load cache
        cache = SemanticLabelCache(str(self.cache_dir))

        # Query with different prefix should still match via suffix
        result_path = cache._find_npz_path("data/megadepth/scene/img.jpg")
        assert result_path == str(npz_path)

    def test_cache_ambiguous_path(self):
        """Test that ambiguous paths raise error."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create two npz with same suffix
        npz1 = labels_dir / "npz1.npz"
        npz2 = labels_dir / "npz2.npz"
        self._create_test_npz(npz1, 100, 200)
        self._create_test_npz(npz2, 100, 200)

        # Create manifest with same suffix for both
        manifest = {
            "scene1/sub/img.jpg": "labels/npz1.npz",
            "scene2/sub/img.jpg": "labels/npz2.npz",
        }
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Load cache
        cache = SemanticLabelCache(str(self.cache_dir))

        # Query with path that has same suffix as both entries should raise error
        with pytest.raises(FileNotFoundError) as exc_info:
            cache._find_npz_path("data/megadepth/scene1/sub/img.jpg")

        assert "Ambiguous" in str(exc_info.value)


class TestValidateSemanticCache:
    """Test cache validation function."""

    def setup_method(self):
        """Create temp cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_valid_cache(self):
        """Test validation of valid cache."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create valid npz
        npz_path = labels_dir / "valid.npz"
        label = np.zeros((100, 200), dtype=np.uint16)
        np.savez_compressed(
            npz_path,
            label=label,
            image_path="dummy",
            height=100,
            width=200,
        )

        # Create manifest
        manifest = {"test/image.jpg": "labels/valid.npz"}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Validate
        results = validate_semantic_cache(str(self.cache_dir))

        assert results['valid'] is True
        assert results['total_entries'] == 1
        assert results['valid_entries'] == 1
        assert len(results['missing_files']) == 0
        assert len(results['bad_shapes']) == 0

    def test_validate_missing_npz(self):
        """Test validation detects missing npz files."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create manifest with non-existent npz
        manifest = {"test/image.jpg": "labels/missing.npz"}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Validate
        results = validate_semantic_cache(str(self.cache_dir))

        assert results['valid'] is False
        assert len(results['missing_files']) == 1
        assert "missing.npz" in results['missing_files'][0]

    def test_validate_bad_shape(self):
        """Test validation detects shape mismatches."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create npz with wrong shape metadata
        npz_path = labels_dir / "wrong_shape.npz"
        np.savez_compressed(
            npz_path,
            label=np.zeros((100, 200), dtype=np.uint16),
            image_path="dummy",
            height=50,   # Wrong!
            width=100,   # Wrong!
        )

        # Create manifest
        manifest = {"test/image.jpg": "labels/wrong_shape.npz"}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Validate
        results = validate_semantic_cache(str(self.cache_dir))

        assert results['valid'] is False
        assert len(results['bad_shapes']) == 1

    def test_validate_missing_label(self):
        """Test validation detects npz without label."""
        labels_dir = self.cache_dir / "labels"
        labels_dir.mkdir()

        # Create npz without label
        npz_path = labels_dir / "no_label.npz"
        np.savez_compressed(
            npz_path,
            image_path="dummy",
            height=100,
            width=200,
        )

        # Create manifest
        manifest = {"test/image.jpg": "labels/no_label.npz"}
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        # Validate
        results = validate_semantic_cache(str(self.cache_dir))

        assert results['valid'] is False
        assert len(results['bad_shapes']) == 1
        assert "no 'label' key" in results['bad_shapes'][0][1]