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
from pathlib import Path
from collections import defaultdict

import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR

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

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


class PL_LoFTR_Semantic(PL_LoFTR):
    """Extended PL_LoFTR with semantic diagnostic capability."""

    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None,
                 semantic_diag=False, oneformer_dir=None, semantic_output_dir=None,
                 semantic_score_thresh=0.0, semantic_save_per_match=False):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)

        self.semantic_diag = semantic_diag
        self.semantic_output_dir = semantic_output_dir
        self.semantic_score_thresh = semantic_score_thresh
        self.semantic_save_per_match = semantic_save_per_match

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

        # Semantic diagnostic output
        if self.semantic_diag and self.trainer.global_rank == 0:
            self._write_semantic_diagnostics()

    def _write_semantic_diagnostics(self):
        """Write semantic diagnostic results to output directory."""
        if not self._pair_diagnostics:
            return

        if not self.semantic_output_dir:
            return

        from tools.semantic_covis.semantic_match_diagnostic import aggregate_diagnostics

        # Write per-pair results
        pair_jsonl_path = Path(self.semantic_output_dir) / "per_pair_semantic_diagnostic.jsonl"
        with open(pair_jsonl_path, 'w') as f:
            for diag in self._pair_diagnostics:
                # Remove per_match data if not requested
                if not self.semantic_save_per_match and 'per_match' in diag:
                    del diag['per_match']
                f.write(json.dumps(diag) + '\n')

        # Write CSV summary
        pair_csv_path = Path(self.semantic_output_dir) / "per_pair_semantic_diagnostic.csv"
        import csv
        with open(pair_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pair_id', 'image0', 'image1', 'num_matches', 'num_valid',
                'fine_consistency_rate', 'coarse_consistency_rate',
                'fine_mismatch_rate', 'coarse_mismatch_rate'
            ])
            writer.writeheader()
            for diag in self._pair_diagnostics:
                row = {
                    'pair_id': diag.get('pair_id', ''),
                    'image0': diag.get('image0', ''),
                    'image1': diag.get('image1', ''),
                    'num_matches': diag.get('num_matches', 0),
                    'num_valid': diag.get('num_valid', 0),
                    'fine_consistency_rate': f"{diag.get('fine_consistency_rate', 0):.4f}",
                    'coarse_consistency_rate': f"{diag.get('coarse_consistency_rate', 0):.4f}",
                    'fine_mismatch_rate': f"{diag.get('fine_mismatch_rate', 0):.4f}",
                    'coarse_mismatch_rate': f"{diag.get('coarse_mismatch_rate', 0):.4f}",
                }
                writer.writerow(row)

        # Write global summary
        summary = aggregate_diagnostics(self._pair_diagnostics)
        summary_path = Path(self.semantic_output_dir) / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Semantic diagnostic results written to {self.semantic_output_dir}")
        logger.info(f"  Fine consistency rate: {summary.get('global_fine_consistency_rate', 0):.4f}")
        logger.info(f"  Coarse consistency rate: {summary.get('global_coarse_consistency_rate', 0):.4f}")


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