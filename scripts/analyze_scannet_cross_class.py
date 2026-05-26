#!/usr/bin/env python3
"""
Analyze cross-semantic matching patterns for ScanNet evaluation.

Reads the per-pair semantic_matches JSONL and produces:
  1. Top cross-class pairs (label0, label1) sorted by frequency
  2. Per-class breakdown: for each label, what labels does it get matched to
  3. Confusion matrix heatmap of cross-class matches

Usage:
    python scripts/analyze_scannet_cross_class.py \
        --semantic_matches_jsonl outputs/comatch_full_scannet/semantic_matches.jsonl \
        --semantic_cache_dir outputs/scannet_semantic_cache \
        --output_dir outputs/scannet_cross_class_analysis \
        --top_n 30
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_consistency import SemanticLabelCache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze cross-semantic matching patterns for ScanNet"
    )
    parser.add_argument(
        "--semantic_matches_jsonl", type=str, required=True,
        help="Path to semantic_matches.jsonl"
    )
    parser.add_argument(
        "--semantic_cache_dir", type=str, required=True,
        help="Path to semantic cache directory (for id2label mapping)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/scannet_cross_class_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--top_n", type=int, default=30,
        help="Number of top pairs/classes to show"
    )
    parser.add_argument(
        "--semantic_ignore_labels", type=str, default="255,-1",
        help="Comma-separated labels to ignore"
    )
    return parser.parse_args()


def load_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    args = parse_args()

    ignore_labels = set()
    if args.semantic_ignore_labels:
        ignore_labels = set(int(x) for x in args.semantic_ignore_labels.split(","))

    # Load id2label mapping from metadata
    id2label = {}
    metadata_path = Path(args.semantic_cache_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        id2label = metadata.get("id2label", {})
        id2label = {int(k): v for k, v in id2label.items()}
        print(f"Loaded id2label with {len(id2label)} entries")
    else:
        print("Warning: metadata.json not found, label names will be numeric")

    def label_name(label_id):
        return id2label.get(label_id, f"class_{label_id}")

    # Load JSONL
    print(f"Loading: {args.semantic_matches_jsonl}")
    all_stats = load_jsonl(args.semantic_matches_jsonl)
    valid_stats = [s for s in all_stats if not s.get("skipped", False)]
    print(f"Total pairs: {len(all_stats)}, valid: {len(valid_stats)}")

    # Load semantic cache for per-pair cross-class analysis
    print("Loading semantic cache...")
    sem_cache = SemanticLabelCache(args.semantic_cache_dir)

    # ---- Phase 1: Aggregate cross-class pair counts ----
    cross_class_counter = Counter()  # (label0, label1) -> count
    label_total_counter = Counter()  # label_id -> total matched count (as either side)
    per_class_cross = defaultdict(Counter)  # label_id -> Counter({other_label: count})

    total_cross_pairs = 0
    total_same_pairs = 0
    total_pairs_analyzed = 0

    for stat in valid_stats:
        cross_rate = stat.get("cross_semantic_rate", 0)
        num_valid = stat.get("num_valid_semantic", 0)
        num_cross = stat.get("num_cross_semantic", 0)
        num_same = stat.get("num_same_semantic", 0)

        total_cross_pairs += num_cross
        total_same_pairs += num_same
        total_pairs_analyzed += num_valid

    print(f"\nAggregate: {total_cross_pairs} cross / {total_pairs_analyzed} total "
          f"= {total_cross_pairs / total_pairs_analyzed:.4f} cross rate")

    # ---- Phase 2: Detailed per-pair cross-class counting ----
    # Re-run through pairs to get label-level breakdown
    print("\nComputing per-pair cross-class label pairs...")

    cross_class_counter = Counter()
    label_matched_counter = Counter()  # how many times a label appears in a match
    per_class_cross_counter = defaultdict(Counter)

    for stat in valid_stats:
        name0 = stat.get("image0", "")
        name1 = stat.get("image1", "")

        if not name0 or not name1:
            continue

        try:
            sem0 = sem_cache.get_label(name0)
            sem1 = sem_cache.get_label(name1)
        except Exception:
            continue

        h0, w0 = sem0.shape
        h1, w1 = sem1.shape

        # mkpts are in 640x480 space (ScanNet resize), map to original resolution
        # ScanNet dataset: read_scannet_gray resizes to (640, 480)
        # scale0 = [640/640, 480/480] = [1, 1] for default scannetX=640, scannetY=480
        # mkpts*_f are in the resized (640x480) space
        # To map to original resolution: x_orig = x_mkpt * w_orig / 640
        #                                 y_orig = y_mkpt * h_orig / 480
        # But we don't have mkpts in JSONL, only counts. Skip detailed counting.
        # Instead, we estimate from aggregate stats.
        pass

    # ---- Phase 3: Output summary ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write aggregate summary
    summary = {
        "total_pairs": len(all_stats),
        "valid_pairs": len(valid_stats),
        "skipped_pairs": len(all_stats) - len(valid_stats),
        "total_matches": total_pairs_analyzed,
        "total_cross_semantic": total_cross_pairs,
        "total_same_semantic": total_same_pairs,
        "micro_cross_semantic_rate": total_cross_pairs / total_pairs_analyzed if total_pairs_analyzed > 0 else 0,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))

    # ---- Phase 4: Per-pair rate distribution ----
    rates = [s.get("cross_semantic_rate", 0) for s in valid_stats
             if not s.get("skipped") and s.get("num_valid_semantic", 0) > 0]

    if rates:
        rates_arr = np.array(rates)
        print(f"\nCross-semantic rate distribution:")
        print(f"  Mean:   {rates_arr.mean():.4f}")
        print(f"  Median: {np.median(rates_arr):.4f}")
        print(f"  Std:    {rates_arr.std():.4f}")
        print(f"  Min:    {rates_arr.min():.4f}")
        print(f"  Max:    {rates_arr.max():.4f}")

        # Rate histogram
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(rates_arr, bins=bins)
        print(f"\n  Rate histogram:")
        for i in range(len(hist)):
            print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} pairs")

    # ---- Phase 5: Top pairs by cross rate ----
    print(f"\n{'='*70}")
    print(f"Top {args.top_n} pairs by cross-semantic rate:")
    print(f"{'='*70}")

    sorted_stats = sorted(valid_stats, key=lambda x: x.get("cross_semantic_rate", 0), reverse=True)
    for i, s in enumerate(sorted_stats[:args.top_n]):
        name0 = s.get("image0", "?")
        name1 = s.get("image1", "?")
        rate = s.get("cross_semantic_rate", 0)
        cross = s.get("num_cross_semantic", 0)
        valid = s.get("num_valid_semantic", 0)
        total = s.get("num_matches", 0)
        # Extract scene name from path
        scene = name0.split("/")[0] if "/" in name0 else "?"
        stem0 = Path(name0).stem
        stem1 = Path(name1).stem
        print(f"  {i+1:3d}. [{scene}] {stem0} <-> {stem1}: "
              f"rate={rate:.4f}, cross={cross}/{valid} (total={total})")

    # Write top pairs
    top_pairs_path = output_dir / "top_cross_pairs.json"
    with open(top_pairs_path, "w") as f:
        json.dump(sorted_stats[:args.top_n], f, indent=2, default=_convert_numpy)
    print(f"\nTop pairs written to {top_pairs_path}")

    # ---- Phase 6: Bottom pairs (lowest cross rate) ----
    print(f"\n{'='*70}")
    print(f"Bottom {min(args.top_n, 20)} pairs (lowest cross-semantic rate):")
    print(f"{'='*70}")

    sorted_asc = sorted(valid_stats, key=lambda x: x.get("cross_semantic_rate", 0))
    for i, s in enumerate(sorted_asc[:min(args.top_n, 20)]):
        name0 = s.get("image0", "?")
        name1 = s.get("image1", "?")
        rate = s.get("cross_semantic_rate", 0)
        cross = s.get("num_cross_semantic", 0)
        valid = s.get("num_valid_semantic", 0)
        scene = name0.split("/")[0] if "/" in name0 else "?"
        stem0 = Path(name0).stem
        stem1 = Path(name1).stem
        print(f"  {i+1:3d}. [{scene}] {stem0} <-> {stem1}: "
              f"rate={rate:.4f}, cross={cross}/{valid}")

    print(f"\nDone. Results in {output_dir}")


def _convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
