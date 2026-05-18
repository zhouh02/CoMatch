#!/usr/bin/env python3
"""
Test script for CoMatch with optional semantic diagnostic.

This script extends the standard test.py with semantic covisibility diagnostics.
It is backward compatible - running without SEMANTIC_DIAGNOSTIC=1 gives identical
results to the original test.py.

Usage:
    # Standard evaluation (backward compatible)
    bash scripts/reproduce_test/outdoor.sh

    # With semantic diagnostic
    SEMANTIC_DIAGNOSTIC=1 \
    ONEFORMER_DIR=outputs/oneformer_outdoor_test \
    SEMANTIC_DIAGNOSTIC_OUTPUT=outputs/semantic_diagnostic_outdoor \
    SEMANTIC_SCORE_THRESH=0.0 \
    SEMANTIC_SAVE_PER_MATCH=0 \
    python test_semantic.py \
        configs/data/megadepth_test_1500.py \
        configs/loftr/comatch_full.py \
        --ckpt_path=weights/comatch_outdoor.ckpt \
        --megasize 1152 --thr 0.1 --ransac_times 5 --deter

Environment Variables:
    SEMANTIC_DIAGNOSTIC: 0 or 1 (default: 0, disabled)
    ONEFORMER_DIR: Path to OneFormer preprocessing output
    SEMANTIC_DIAGNOSTIC_OUTPUT: Output directory for semantic diagnostic results
    SEMANTIC_SCORE_THRESH: Minimum segment score for valid sample (default: 0.0)
    SEMANTIC_SAVE_PER_MATCH: Save per-match details (0 or 1, default: 0)
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger
from loguru import logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR

from src.utils.misc import flattenList
from src.utils.comm import gather
from src.utils.metrics import aggregate_metrics

import torch

# =============================================================================
# Semantic Diagnostic Import (optional)
# =============================================================================

SEMANTIC_DIAGNOSTIC_ENABLED = os.environ.get('SEMANTIC_DIAGNOSTIC', '0') == '1'
_semantic_module = None


def _get_semantic_module():
    """Lazy load semantic diagnostic module to avoid hard dependency."""
    global _semantic_module
    if _semantic_module is None:
        # Only import if semantic diagnostic is enabled
        if SEMANTIC_DIAGNOSTIC_ENABLED:
            try:
                from tools.semantic_covis.semantic_match_diagnostic import (
                    OneFormerSegStore,
                    diagnose_batch,
                    aggregate_diagnostics,
                )
                _semantic_module = {
                    'OneFormerSegStore': OneFormerSegStore,
                    'diagnose_batch': diagnose_batch,
                    'aggregate_diagnostics': aggregate_diagnostics,
                }
            except ImportError as e:
                logger.error(f"Failed to import semantic diagnostic module: {e}")
                raise
    return _semantic_module


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--ckpt_path', type=str, default="weights/indoor_ds.ckpt", help='path to the checkpoint')
    parser.add_argument('--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument('--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--thr', type=float, default=None, help='modify the coarse-level matching threshold.')
    parser.add_argument('--pixel_thr', type=float, default=None, help='modify the RANSAC threshold.')
    parser.add_argument('--ransac', type=str, default=None, help='modify the RANSAC method')
    parser.add_argument('--scannetX', type=int, default=None, help='ScanNet resize X')
    parser.add_argument('--scannetY', type=int, default=None, help='ScanNet resize Y')
    parser.add_argument('--megasize', type=int, default=None, help='MegaDepth resize')
    parser.add_argument('--npe', action='store_true', default=False, help='')
    parser.add_argument('--fp32', action='store_true', default=False, help='pure32')
    parser.add_argument('--ransac_times', type=int, default=None, help='repeat ransac multiple times for more robust evaluation')
    parser.add_argument('--rmbd', type=int, default=None, help='remove border matches')
    parser.add_argument('--deter', action='store_true', default=False, help='use deterministic mode for testing')
    parser.add_argument('--half', action='store_true', default=False, help='pure16')
    parser.add_argument('--flash', action='store_true', default=False, help='flash')

    # Semantic diagnostic options
    parser.add_argument('--semantic-diag', action='store_true', default=SEMANTIC_DIAGNOSTIC_ENABLED,
                        help='Enable semantic diagnostic (overrides env var)')
    parser.add_argument('--oneformer-dir', type=str, default=os.environ.get('ONEFORMER_DIR'),
                        help='Path to OneFormer preprocessing output')
    parser.add_argument('--semantic-output-dir', type=str, default=os.environ.get('SEMANTIC_DIAGNOSTIC_OUTPUT'),
                        help='Output directory for semantic diagnostic results')
    parser.add_argument('--semantic-score-thresh', type=float, default=0.0,
                        help='Minimum segment score for valid sample')
    parser.add_argument('--semantic-save-per-match', action='store_true', default=False,
                        help='Save per-match details (can be large)')
    parser.add_argument('--semantic-target-pair-list', type=str, default=None,
                        help='Path to JSONL with target pairs. Only these pairs will have per-match saved.')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


class PL_LoFTR_Semantic(PL_LoFTR):
    """Extended PL_LoFTR with semantic diagnostic capability."""

    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None,
                 semantic_diag=False, oneformer_dir=None, semantic_output_dir=None,
                 semantic_score_thresh=0.0, semantic_save_per_match=False,
                 semantic_target_pair_list=None):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)

        self.semantic_diag = semantic_diag
        self.semantic_output_dir = semantic_output_dir
        self.semantic_score_thresh = semantic_score_thresh
        self.semantic_save_per_match = semantic_save_per_match
        self.semantic_target_pair_list = semantic_target_pair_list

        # Load target pairs if provided
        self._target_pairs = None
        if semantic_target_pair_list and os.path.exists(semantic_target_pair_list):
            self._load_target_pairs(semantic_target_pair_list)

        # Initialize semantic store lazily
        self._seg_store = None
        self._semantic_module = None

        if self.semantic_diag:
            if not oneformer_dir:
                raise ValueError("SEMANTIC_DIAGNOSTIC=1 requires ONEFORMER_DIR to be set")
            if not os.path.exists(oneformer_dir):
                raise ValueError(f"OneFormer directory not found: {oneformer_dir}")

            self._semantic_module = _get_semantic_module()
            self._seg_store = self._semantic_module['OneFormerSegStore'](
                oneformer_dir, verbose=True
            )

            # Prepare output directory
            if self.semantic_output_dir:
                os.makedirs(self.semantic_output_dir, exist_ok=True)

            self._pair_diagnostics = []

            loguru_logger.info(f"Semantic diagnostic enabled:")
            loguru_logger.info(f"  OneFormer dir: {oneformer_dir}")
            loguru_logger.info(f"  Output dir: {semantic_output_dir or 'None'}")
            loguru_logger.info(f"  Score thresh: {semantic_score_thresh}")
            if self._target_pairs:
                loguru_logger.info(f"  Target pairs: {len(self._target_pairs)} loaded from {semantic_target_pair_list}")

    def _load_target_pairs(self, target_pair_list_path: str) -> None:
        """Load target pairs from JSONL file for selective per-match saving."""
        import json
        self._target_pairs = {}
        try:
            with open(target_pair_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # Build canonical pair key matching diagnose_batch format:
                    # sorted(image0, image1) joined by |||
                    norm0 = entry.get('image0', '')
                    norm1 = entry.get('image1', '')
                    a, b = sorted([norm0, norm1])
                    key = f"{a}|||{b}"
                    self._target_pairs[key] = entry
            loguru_logger.info(f"Loaded {len(self._target_pairs)} target pairs")
        except Exception as e:
            loguru_logger.warning(f"Failed to load target pairs: {e}")
            self._target_pairs = None

    def test_step(self, batch, batch_idx):
        # Run standard matcher
        if not self.warmup:
            if self.config.LOFTR.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True
            torch.cuda.synchronize()

        if self.config.LOFTR.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.start_event.record()
                self.matcher(batch, True)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # Semantic diagnostic
        if self.semantic_diag:
            try:
                from tools.semantic_covis.semantic_match_diagnostic import diagnose_batch

                diag_results = diagnose_batch(
                    batch,
                    self._seg_store,
                    score_thresh=self.semantic_score_thresh,
                    save_all_matches=self.semantic_save_per_match,
                    target_pairs=self._target_pairs,
                )

                # Add pair info to results
                for diag in diag_results:
                    if 'pair_names' in batch:
                        pn = batch['pair_names']
                        if isinstance(pn, (list, tuple)):
                            diag['image0'] = str(pn[0])
                            diag['image1'] = str(pn[1])
                        elif isinstance(pn, torch.Tensor):
                            diag['image0'] = str(pn[0].item())
                            diag['image1'] = str(pn[1].item())

                self._pair_diagnostics.extend(diag_results)

            except Exception as e:
                loguru_logger.warning(f"Semantic diagnostic failed for batch {batch_idx}: {e}")

        return ret_dict

    def test_epoch_end(self, outputs):
        # Standard metrics aggregation
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.trainer.global_rank == 0:
            print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))

        # Semantic diagnostic output: each rank writes its own file, then rank 0 merges
        if self.semantic_diag:
            self._write_semantic_diagnostics()

    def _write_semantic_diagnostics(self):
        """Write semantic diagnostic results per-rank, then merge on rank 0."""
        if not self.semantic_output_dir:
            return

        from tools.semantic_covis.semantic_match_diagnostic import aggregate_diagnostics

        try:
            rank = self.trainer.global_rank
        except Exception:
            rank = 0

        out_dir = Path(self.semantic_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Each rank writes its own JSONL ---
        rank_jsonl = out_dir / f"per_pair_semantic_diagnostic_rank{rank}.jsonl"
        rank_per_match_jsonl = out_dir / f"per_match_semantic_diagnostic_rank{rank}.jsonl"

        # Write per-match JSONL first (before per_match is deleted from dicts)
        with open(rank_per_match_jsonl, 'w') as f:
            for diag in self._pair_diagnostics:
                if 'per_match' in diag:
                    for pm in diag['per_match']:
                        record = {
                            'pair_key': diag.get('pair_key', ''),
                            'image0': diag.get('image0', ''),
                            'image1': diag.get('image1', ''),
                            'match_idx': pm.get('idx', 0),
                            'x0': pm.get('pt0', [None, None])[0] if pm.get('pt0') else None,
                            'y0': pm.get('pt0', [None, None])[1] if pm.get('pt0') else None,
                            'x1': pm.get('pt1', [None, None])[0] if pm.get('pt1') else None,
                            'y1': pm.get('pt1', [None, None])[1] if pm.get('pt1') else None,
                            'seg_x0': pm.get('seg_x0'),
                            'seg_y0': pm.get('seg_y0'),
                            'seg_x1': pm.get('seg_x1'),
                            'seg_y1': pm.get('seg_y1'),
                            'score': pm.get('conf'),
                            'label_id0': pm.get('label0'),
                            'label_id1': pm.get('label1'),
                            'label0': pm.get('label0_name', ''),
                            'label1': pm.get('label1_name', ''),
                            'coarse0': pm.get('coarse0', ''),
                            'coarse1': pm.get('coarse1', ''),
                            'fine_consistent': pm.get('fine_consistent', False),
                            'coarse_consistent': pm.get('coarse_consistent', False),
                        }
                        f.write(json.dumps(record) + '\n')

        # Write per-pair JSONL (strips per_match to avoid large files)
        with open(rank_jsonl, 'w') as f:
            for diag in self._pair_diagnostics:
                if 'per_match' in diag:
                    del diag['per_match']
                f.write(json.dumps(diag) + '\n')

        loguru_logger.info(f"[rank {rank}] Wrote {len(self._pair_diagnostics)} pairs to {rank_jsonl}")
        loguru_logger.info(f"[rank {rank}] Wrote per-match details to {rank_per_match_jsonl}")

        # --- Rank 0 merges all ranks ---
        if rank != 0:
            return

        # Warn if multi-GPU semantic diagnostic is used
        try:
            world_size = self.trainer.world_size
        except Exception:
            world_size = 1

        if world_size > 1:
            loguru_logger.warning(
                f"Semantic diagnostic with DDP ({world_size} GPUs): "
                f"merging rank JSONL files. If ranks are on separate nodes, "
                f"you may need to manually merge the per_pair_semantic_diagnostic_rank*.jsonl files."
            )

        # Collect all pairs from all rank files
        all_pairs = []
        for r in range(world_size):
            rjsonl = out_dir / f"per_pair_semantic_diagnostic_rank{r}.jsonl"
            if not rjsonl.exists():
                loguru_logger.warning(f"Missing rank file: {rjsonl}")
                continue
            with open(rjsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_pairs.append(json.loads(line))

        # Write merged JSONL
        merged_jsonl = out_dir / "per_pair_semantic_diagnostic.jsonl"
        with open(merged_jsonl, 'w') as f:
            for diag in all_pairs:
                f.write(json.dumps(diag) + '\n')

        # --- Merge per-match JSONL files ---
        merged_per_match_jsonl = out_dir / "per_match_semantic_diagnostic.jsonl"
        all_per_match = []
        for r in range(world_size):
            rjsonl = out_dir / f"per_match_semantic_diagnostic_rank{r}.jsonl"
            if not rjsonl.exists():
                continue
            with open(rjsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_per_match.append(json.loads(line))
        with open(merged_per_match_jsonl, 'w') as f:
            for rec in all_per_match:
                f.write(json.dumps(rec) + '\n')
        loguru_logger.info(f"Merged {len(all_per_match)} per-match records to {merged_per_match_jsonl}")

        # Write CSV
        csv_fields = [
            'image0', 'image1',
            'num_pred_matches', 'num_valid_semantic_matches', 'num_out_of_bounds',
            'missing_segmentation',
            'fine_consistent_count', 'fine_error_count',
            'fine_consistency_rate', 'fine_error_rate',
            'coarse_consistent_count', 'coarse_error_count',
            'coarse_consistency_rate', 'coarse_error_rate',
        ]
        csv_path = out_dir / "per_pair_semantic_diagnostic.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
            writer.writeheader()
            for diag in all_pairs:
                row = {}
                for field in csv_fields:
                    val = diag.get(field, '')
                    if isinstance(val, float):
                        row[field] = f"{val:.4f}"
                    else:
                        row[field] = val
                writer.writerow(row)

        # Aggregate and write summary
        summary = aggregate_diagnostics(all_pairs)
        summary_path = out_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Log key metrics
        logger.info(f"Semantic diagnostic results written to {self.semantic_output_dir}")
        logger.info(f"  Pairs: {summary['num_pairs_total']} total, "
                     f"{summary['num_pairs_evaluated']} evaluated, "
                     f"{summary['num_missing_segmentation_pairs']} missing seg, "
                     f"{summary['num_unique_images']} unique images")
        logger.info(f"  Matches: {summary['num_pred_matches_total']} predicted, "
                     f"{summary['num_valid_semantic_matches']} valid semantic")
        logger.info(f"  Fine:  consistency={summary['fine_consistency_rate']:.4f}, "
                     f"error={summary['fine_error_rate']:.4f}")
        logger.info(f"  Coarse: consistency={summary['coarse_consistency_rate']:.4f}, "
                     f"error={summary['coarse_error_rate']:.4f}")
        logger.info(f"  MAIN semantic error rate (coarse_error): "
                     f"{summary['main_semantic_error_rate']:.4f}")


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


if __name__ == '__main__':
    args = parse_args()
    pprint.pprint(vars(args))

    # Determine semantic diagnostic settings
    semantic_diag = args.semantic_diag
    oneformer_dir = args.oneformer_dir
    semantic_output_dir = args.semantic_output_dir
    semantic_score_thresh = args.semantic_score_thresh
    semantic_save_per_match = args.semantic_save_per_match
    semantic_target_pair_list = args.semantic_target_pair_list

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.deter:
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(config.TRAINER.SEED)

    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    if args.scannetX is not None and args.scannetY is not None:
        config.DATASET.SCAN_IMG_RESIZEX = args.scannetX
        config.DATASET.SCAN_IMG_RESIZEY = args.scannetY
    if args.megasize is not None:
        config.DATASET.MGDPT_IMG_RESIZE = args.megasize

    if args.npe:
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            if config.DATASET.NPE_NAME == 'megadepth':
                config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE]
            elif config.DATASET.NPE_NAME == 'scannet':
                config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.SCAN_IMG_RESIZEX, config.DATASET.SCAN_IMG_RESIZEX]
    else:
        config.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    print(config.LOFTR.COARSE.NPE)

    if args.ransac_times is not None:
        config.LOFTR.EVAL_TIMES = args.ransac_times

    if args.rmbd is not None:
        config.LOFTR.MATCH_COARSE.BORDER_RM = args.rmbd

    if args.pixel_thr is not None:
        config.TRAINER.RANSAC_PIXEL_THR = args.pixel_thr

    if args.ransac is not None:
        config.TRAINER.POSE_ESTIMATION_METHOD = args.ransac
        if args.ransac == 'LO-RANSAC' and config.TRAINER.RANSAC_PIXEL_THR == 0.5:
            config.TRAINER.RANSAC_PIXEL_THR = 2.0

    if args.fp32:
        config.LOFTR.MP = False

    if args.half:
        config.LOFTR.HALF = True
        config.DATASET.FP16 = True
    else:
        config.LOFTR.HALF = False
        config.DATASET.FP16 = False

    if args.flash:
        config.LOFTR.COARSE.NO_FLASH = False

    loguru_logger.info(f"Args and config initialized!")

    # lightning module - use extended version if semantic diagnostic enabled
    profiler = build_profiler(args.profiler_name)

    if semantic_diag:
        model = PL_LoFTR_Semantic(
            config,
            pretrained_ckpt=args.ckpt_path,
            profiler=profiler,
            dump_dir=args.dump_dir,
            semantic_diag=semantic_diag,
            oneformer_dir=oneformer_dir,
            semantic_output_dir=semantic_output_dir,
            semantic_score_thresh=semantic_score_thresh,
            semantic_save_per_match=semantic_save_per_match,
            semantic_target_pair_list=semantic_target_pair_list,
        )
    else:
        model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)

    loguru_logger.info(f"LoFTR-lightning initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)