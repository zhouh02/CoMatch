
from collections import defaultdict
import json
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss_epipolar import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.semantic_consistency import SemanticLabelCache, compute_semantic_match_stats

from torch.profiler import profile

class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None,
                 semantic_cache_dir=None, semantic_ignore_labels=None,
                 semantic_conf_thr=None, semantic_dump_name=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'], profiler=self.profiler)
        self.loss = LoFTRLoss(_config)
   
        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.warmup = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.ms_1 = 0

        # Semantic consistency analysis
        self.semantic_cache_dir = semantic_cache_dir
        self.semantic_ignore_labels = None
        if semantic_ignore_labels:
            try:
                self.semantic_ignore_labels = set(int(x) for x in semantic_ignore_labels.split(','))
            except ValueError:
                logger.warning(f"Invalid semantic_ignore_labels format: {semantic_ignore_labels}")
        self.semantic_conf_thr = semantic_conf_thr if semantic_conf_thr and semantic_conf_thr > 0 else None
        self.semantic_dump_name = semantic_dump_name or 'semantic_matches'
        self._semantic_cache = None
        self._pair_semantic_stats = []

        if self.semantic_cache_dir:
            try:
                self._semantic_cache = SemanticLabelCache(self.semantic_cache_dir)
                logger.info(f"Semantic label cache initialized from: {semantic_cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic cache: {e}")
                self.semantic_cache_dir = None

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_fine(batch, self.config, self.logger)
            
        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.loss(batch)
    
    def _compute_metrics(self, batch):
        
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [(batch['epi_errs'].reshape(-1,1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'num_matches': [batch['mconf'].shape[0]], # batch size = 1 only
            }
        ret_dict = {'metrics': metrics}

        return ret_dict, rel_pair_names

       
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # self.logger.experiment.add_scalar(
            #         f'coarse_temp', self.matcher.coarse_matching.temperature.clone().detach().cpu().data, self.global_step)
            # self.logger.experiment.add_scalar(
            #         f'fine_temp', self.matcher.fine_matching.local_regress_temperature.clone().detach().cpu().data, self.global_step)
  
        torch.set_printoptions(precision=50)
        # print(f"step {self.global_step}: {batch['loss']}\n")
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def on_validation_epoch_start(self):
        self.matcher.fine_matching.validate = True

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)
            
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        self.matcher.fine_matching.validate = False
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        
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
                self.matcher(batch,True)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # Semantic consistency analysis
        if self.semantic_cache_dir and self._semantic_cache:
            self._compute_semantic_stats(batch)

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.trainer.global_rank == 0:
            print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))

        # Aggregate and write semantic statistics
        if self.semantic_cache_dir and self._semantic_cache:
            self._aggregate_semantic_stats()

    def _compute_semantic_stats(self, batch):
        """Compute semantic consistency statistics for a batch."""
        try:
            mkpts0_f = batch['mkpts0_f'].cpu().numpy()
            mkpts1_f = batch['mkpts1_f'].cpu().numpy()
            mconf = batch.get('mconf')
            mconf = mconf.cpu().numpy() if mconf is not None else None
            m_bids = batch.get('m_bids')
            m_bids = m_bids.cpu().numpy() if m_bids is not None else None
            pair_names = batch.get('pair_names')

            if pair_names is None:
                return

            bs = batch['image0'].size(0)

            for b in range(bs):
                # Extract matches for this pair
                if m_bids is not None:
                    mask = m_bids == b
                    pair_mkpts0 = mkpts0_f[mask]
                    pair_mkpts1 = mkpts1_f[mask]
                    pair_conf = mconf[mask] if mconf is not None else None
                else:
                    if bs == 1:
                        pair_mkpts0 = mkpts0_f
                        pair_mkpts1 = mkpts1_f
                        pair_conf = mconf
                    else:
                        logger.warning(
                            f"Batch size > 1 but no m_bids in batch. "
                            f"Skipping semantic stats for batch."
                        )
                        return

                # Get pair names
                if isinstance(pair_names, (list, tuple)):
                    if isinstance(pair_names[0], (list, tuple)):
                        # pair_names = [(name0, name1), ...]
                        name0, name1 = pair_names[b]
                    else:
                        # pair_names might be two separate lists or other structure
                        name0, name1 = pair_names[0], pair_names[1]
                else:
                    name0, name1 = str(pair_names[0]), str(pair_names[1])

                pair_id = f"{Path(name0).stem}_{Path(name1).stem}"

                # Compute stats for this pair
                stat = self._compute_pair_semantic_stats(
                    pair_id, name0, name1,
                    pair_mkpts0, pair_mkpts1, pair_conf
                )
                self._pair_semantic_stats.append(stat)

        except Exception as e:
            logger.warning(f"Failed to compute semantic stats: {e}")

    def _compute_pair_semantic_stats(self, pair_id, name0, name1, mkpts0, mkpts1, conf):
        """Compute semantic stats for a single pair."""
        try:
            sem0 = self._semantic_cache.get_label(name0)
            sem1 = self._semantic_cache.get_label(name1)

            stats = compute_semantic_match_stats(
                mkpts0, mkpts1, sem0, sem1,
                conf=conf,
                ignore_labels=self.semantic_ignore_labels,
                conf_thr=self.semantic_conf_thr
            )

            return {
                "pair_id": pair_id,
                "image0": str(name0),
                "image1": str(name1),
                "num_matches": stats["num_matches"],
                "num_after_conf": stats["num_after_conf"],
                "num_in_bounds": stats["num_in_bounds"],
                "num_valid_semantic": stats["num_valid_semantic"],
                "num_same_semantic": stats["num_same_semantic"],
                "num_cross_semantic": stats["num_cross_semantic"],
                "cross_semantic_rate": stats["cross_semantic_rate"],
                "skipped": False,
                "skip_reason": None
            }
        except FileNotFoundError as e:
            return {
                "pair_id": pair_id,
                "image0": str(name0),
                "image1": str(name1),
                "skipped": True,
                "skip_reason": str(e)
            }
        except Exception as e:
            return {
                "pair_id": pair_id,
                "image0": str(name0),
                "image1": str(name1),
                "skipped": True,
                "skip_reason": f"Error: {e}"
            }

    def _aggregate_semantic_stats(self):
        """Aggregate semantic stats from all ranks and write output files."""
        try:
            rank = self.trainer.global_rank
        except Exception:
            rank = 0

        try:
            world_size = self.trainer.world_size
        except Exception:
            world_size = 1

        # Write per-rank JSONL first
        if self._pair_semantic_stats:
            rank_jsonl = Path(self.dump_dir or '.') / f"{self.semantic_dump_name}_rank{rank}.jsonl"
            rank_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(rank_jsonl, 'w') as f:
                for stat in self._pair_semantic_stats:
                    f.write(json.dumps(stat) + '\n')
            logger.info(f"[rank {rank}] Wrote {len(self._pair_semantic_stats)} pairs to {rank_jsonl}")

        # Only rank 0 aggregates and writes summary
        if rank != 0:
            return

        # Gather all stats from all ranks
        all_stats = []
        for r in range(world_size):
            r_jsonl = Path(self.dump_dir or '.') / f"{self.semantic_dump_name}_rank{r}.jsonl"
            if not r_jsonl.exists():
                continue
            with open(r_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_stats.append(json.loads(line))

        if not all_stats:
            return

        # Write merged JSONL
        merged_jsonl = Path(self.dump_dir or '.') / f"{self.semantic_dump_name}.jsonl"
        with open(merged_jsonl, 'w') as f:
            for stat in all_stats:
                f.write(json.dumps(stat) + '\n')

        # Compute summary
        num_pairs = len(all_stats)
        num_skipped = sum(1 for s in all_stats if s.get('skipped', False))
        total_matches = sum(s.get('num_matches', 0) for s in all_stats)
        total_after_conf = sum(s.get('num_after_conf', 0) for s in all_stats)
        total_in_bounds = sum(s.get('num_in_bounds', 0) for s in all_stats)
        total_valid_semantic = sum(s.get('num_valid_semantic', 0) for s in all_stats)
        total_same_semantic = sum(s.get('num_same_semantic', 0) for s in all_stats)
        total_cross_semantic = sum(s.get('num_cross_semantic', 0) for s in all_stats)

        # Micro cross semantic rate
        if total_valid_semantic > 0:
            micro_cross_semantic_rate = total_cross_semantic / total_valid_semantic
        else:
            micro_cross_semantic_rate = float("nan")

        # Macro cross semantic rate (mean of non-skipped, non-nan rates)
        valid_rates = []
        for s in all_stats:
            if not s.get('skipped', False):
                rate = s.get('cross_semantic_rate')
                if rate is not None and not (isinstance(rate, float) and np.isnan(rate)):
                    valid_rates.append(rate)
        macro_cross_semantic_rate = float(np.mean(valid_rates)) if valid_rates else float("nan")

        summary = {
            "num_pairs": num_pairs,
            "num_skipped_pairs": num_skipped,
            "total_matches": total_matches,
            "total_after_conf": total_after_conf,
            "total_in_bounds": total_in_bounds,
            "total_valid_semantic": total_valid_semantic,
            "total_same_semantic": total_same_semantic,
            "total_cross_semantic": total_cross_semantic,
            "micro_cross_semantic_rate": micro_cross_semantic_rate,
            "macro_cross_semantic_rate": macro_cross_semantic_rate,
        }

        # Write summary
        summary_path = Path(self.dump_dir or '.') / f"{self.semantic_dump_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Semantic consistency summary written to {summary_path}")
        logger.info(f"  Pairs: {num_pairs} total, {num_skipped} skipped")
        logger.info(f"  Matches: {total_matches} total, {total_valid_semantic} valid semantic")
        logger.info(f"  Cross semantic: {total_cross_semantic} ({micro_cross_semantic_rate:.4f} micro rate)")


    