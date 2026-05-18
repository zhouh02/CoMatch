#!/usr/bin/env python3
"""
Select top-K pairs from semantic diagnostic CSV for targeted per-match saving.

Usage:
    python tools/semantic_covis/select_semantic_diagnostic_pairs.py \
        --csv outputs/semantic_diagnostic_outdoor/per_pair_semantic_diagnostic.csv \
        --sort-by coarse_error_rate \
        --topk 20 \
        --output outputs/semantic_diagnostic_outdoor/top_coarse_error_pairs.jsonl

The script reads per_pair_semantic_diagnostic.csv, sorts by the specified column,
and outputs top-K pairs to a JSONL file.  This file can then be passed to
test_semantic.py via --semantic-target-pair-list so that only those pairs
get per-match details saved.

Output format (one JSON object per line):
    {
      "rank": 1,
      "image0": "Undistorted_SfM/0015/images/2959733121_2a804e7d1c_o.jpg",
      "image1": "Undistorted_SfM/0015/images/389635223_1213f76125_o.jpg",
      "num_valid_semantic_matches": 4232,
      "fine_error_rate": 0.7741,
      "coarse_error_rate": 0.7564
    }
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path normalisation (same logic as semantic_match_diagnostic.py)
# --------------------------------------------------------------------------- #

def normalize_rel_image_path(x):
    """Normalise an image path that may be wrapped in a list / repr string."""
    import ast as _ast

    if hasattr(x, "tolist") and not isinstance(x, str):
        x = x.tolist()

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        x = x[0]
        if isinstance(x, (list, tuple)):
            return normalize_rel_image_path(x)

    if not isinstance(x, str):
        x = str(x)

    s = x.strip()
    if s.startswith("[") or s.startswith("("):
        try:
            parsed = _ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                s = str(parsed[0]).strip()
        except Exception:
            pass

    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]

    return s


def canonical_pair_key(image0: str, image1: str) -> str:
    """Create a direction-independent pair key from two image paths."""
    norm0 = normalize_rel_image_path(image0)
    norm1 = normalize_rel_image_path(image1)
    a, b = sorted([norm0, norm1])
    return f"{a}|||{b}"


# --------------------------------------------------------------------------- #
# CSV column name mapping (handles old / new field names)
# --------------------------------------------------------------------------- #

COLUMNS = [
    'image0', 'image1',
    'num_pred_matches', 'num_valid_semantic_matches',
    'fine_consistent_count', 'fine_error_count',
    'fine_consistency_rate', 'fine_error_rate',
    'coarse_consistent_count', 'coarse_error_count',
    'coarse_consistency_rate', 'coarse_error_rate',
    'missing_segmentation',
]


def load_csv(path: Path):
    """Load semantic diagnostic CSV and return list of dicts."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for col in COLUMNS:
                val = raw.get(col) or raw.get(col.replace('_', ''), '')
                if col in ('fine_consistency_rate', 'fine_error_rate',
                          'coarse_consistency_rate', 'coarse_error_rate',
                          'num_valid_semantic_matches', 'num_pred_matches',
                          'fine_consistent_count', 'fine_error_count',
                          'coarse_consistent_count', 'coarse_error_count'):
                    try:
                        row[col] = float(val) if val else 0.0
                    except (ValueError, TypeError):
                        row[col] = 0.0
                else:
                    row[col] = val.strip() if val else ''
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Select top-K pairs from semantic diagnostic CSV"
    )
    parser.add_argument(
        '--csv', type=str, required=True,
        help='Path to per_pair_semantic_diagnostic.csv'
    )
    parser.add_argument(
        '--sort-by', type=str, default='coarse_error_rate',
        choices=['coarse_error_rate', 'fine_error_rate',
                'num_valid_semantic_matches', 'coarse_error_count'],
        help='Column to sort by'
    )
    parser.add_argument(
        '--topk', type=int, default=20,
        help='Number of top pairs to select'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output JSONL path'
    )
    parser.add_argument(
        '--min-valid', type=int, default=10,
        help='Minimum num_valid_semantic_matches to be considered'
    )
    parser.add_argument(
        '--descending', action='store_true', default=True,
        help='Sort descending (highest error first)'
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    # Filter by minimum valid matches
    filtered = [r for r in rows if r['num_valid_semantic_matches'] >= args.min_valid]
    print(f"  {len(filtered)} pairs with >= {args.min_valid} valid matches")

    # Sort
    sort_col = args.sort_by
    filtered.sort(key=lambda r: r.get(sort_col, 0.0), reverse=args.descending)

    # Top-K
    topk = filtered[:args.topk]

    # Normalise image paths and write JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys = set()
    written = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for rank, row in enumerate(topk, start=1):
            image0 = normalize_rel_image_path(row['image0'])
            image1 = normalize_rel_image_path(row['image1'])
            key = canonical_pair_key(image0, image1)

            # Skip duplicates (A-B == B-A)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            entry = {
                'rank': rank,
                'pair_key': key,
                'image0': image0,
                'image1': image1,
                'num_valid_semantic_matches': row['num_valid_semantic_matches'],
                'fine_error_rate': row['fine_error_rate'],
                'coarse_error_rate': row['coarse_error_rate'],
                'fine_consistent_count': row['fine_consistent_count'],
                'fine_error_count': row['fine_error_count'],
                'coarse_consistent_count': row['coarse_consistent_count'],
                'coarse_error_count': row['coarse_error_count'],
            }
            f.write(json.dumps(entry) + '\n')
            written += 1

    print(f"Wrote {written} pairs to {out_path}")
    print(f"\nFirst 3 pairs:")
    for i, line in enumerate(open(out_path)):
        if i >= 3:
            break
        entry = json.loads(line)
        print(f"  [{entry['rank']}] coarse_error_rate={entry['coarse_error_rate']:.4f} "
              f"image0={entry['image0'][-60:]}")

    print(f"\nNext step:")
    print(f"  python test_semantic.py ... --semantic-save-per-match "
          f"--semantic-target-pair-list {out_path}")


if __name__ == '__main__':
    sys.exit(main() or 0)