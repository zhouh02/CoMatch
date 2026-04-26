import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .loftr_module import LocalFeatureTransformer, FinePreprocess, LocalFeatureTransformer_loftr
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching_epipolar import FineMatching
from ..utils.misc import detect_NaN

from loguru import logger


class CovisibilityInject(nn.Module):
    """Residual inject: fuse upsampled 1/16 transformed features into 1/8 raw features,
    gated by upsampled covisibility scores."""

    def __init__(self, c8_dim, c16_dim):
        super().__init__()
        self.proj16 = nn.Conv2d(c16_dim, c8_dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feat8, feat16, covi16):
        """
        Args:
            feat8: [B, C8, H8, W8] raw 1/8 features
            feat16: [B, C16, H16, W16] transformed 1/16 features from DCAT16
            covi16: [B, 1, H16, W16] covisibility/matchability scores from DCAT16
        Returns:
            feat8_fused: [B, C8, H8, W8]
            covi8: [B, 1, H8, W8]
        """
        feat16_up = F.interpolate(feat16, size=feat8.shape[-2:], mode='bilinear', align_corners=False)
        covi8 = F.interpolate(covi16, size=feat8.shape[-2:], mode='bilinear', align_corners=False)
        injected = self.proj16(feat16_up)
        feat8_fused = feat8 + self.alpha * covi8 * injected
        return feat8_fused, covi8


class LoFTR(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler
        self.use_dcat16_inject = config.get('use_dcat16_inject', False)

        # Modules
        self.backbone = build_backbone(config)
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config)
        self.loftr_fine = LocalFeatureTransformer_loftr(config["fine"])

        # 1/16 DCAT branch (experiment)
        logger.info(f"[DCAT16 Inject] USE_DCAT16_INJECT={self.use_dcat16_inject}")
        if self.use_dcat16_inject:
            d_model_16 = config['coarse16']['d_model']
            d_model_8 = config['coarse']['d_model']
            # Build a config copy where 'coarse' points to 'coarse16' settings
            config16 = copy.deepcopy(config)
            config16['coarse'] = config['coarse16']
            # Inherit npe from original coarse if None (avoids RoPEPositionEncodingSine assert failure)
            if config['coarse16'].get('npe', None) is None:
                config16['coarse'] = copy.deepcopy(config['coarse16'])
                config16['coarse']['npe'] = config['coarse']['npe']
            self.dcat16 = LocalFeatureTransformer(config16)
            self.inject16to8 = CovisibilityInject(d_model_8, d_model_16)


    def forward(self, data, timing=False):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            ret_dict = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            feats_c = ret_dict['feats_c']
            data.update({
                'feats_x2': ret_dict['feats_x2'],
                'feats_x1': ret_dict['feats_x1'],
            })
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            # 1/16 features
            if self.use_dcat16_inject:
                (feat_c16_0, feat_c16_1) = ret_dict['feats_c16'].split(data['bs'])
        else:  # handle different input shapes
            ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
            feat_c0 = ret_dict0['feats_c']
            feat_c1 = ret_dict1['feats_c']
            data.update({
                'feats_x2_0': ret_dict0['feats_x2'],
                'feats_x1_0': ret_dict0['feats_x1'],
                'feats_x2_1': ret_dict1['feats_x2'],
                'feats_x1_1': ret_dict1['feats_x1'],
            })
            # 1/16 features
            if self.use_dcat16_inject:
                feat_c16_0 = ret_dict0['feats_c16']
                feat_c16_1 = ret_dict1['feats_c16']


        mul = self.config['resolution'][0] // self.config['resolution'][1]
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul] ,
            'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]
        })
            

        # 2. coarse-level loftr module
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']

        # 2b. 1/16 DCAT branch + inject (experiment)
        if self.use_dcat16_inject:
            # Build 1/16 masks: downsample original masks to feat_c16 spatial size
            mask16_0, mask16_1 = None, None
            if mask_c0 is not None:
                if mask_c0.dim() == 3:  # [B, H, W] -> downsample to 1/16
                    mask16_0 = F.interpolate(
                        mask_c0.float().unsqueeze(1), size=feat_c16_0.shape[-2:],
                        mode='nearest').squeeze(1).bool()
                    mask16_1 = F.interpolate(
                        mask_c1.float().unsqueeze(1), size=feat_c16_1.shape[-2:],
                        mode='nearest').squeeze(1).bool()
                else:  # [B, H*W] -> reshape to [B,H8,W8] then downsample
                    H8, W8 = data['hw0_c']
                    mask16_0 = mask_c0.view(data['bs'], H8, W8).float()
                    mask16_0 = F.interpolate(
                        mask16_0.unsqueeze(1), size=feat_c16_0.shape[-2:],
                        mode='nearest').squeeze(1).bool()
                    mask16_1 = mask_c1.view(data['bs'], data['hw1_c'][0], data['hw1_c'][1]).float()
                    mask16_1 = F.interpolate(
                        mask16_1.unsqueeze(1), size=feat_c16_1.shape[-2:],
                        mode='nearest').squeeze(1).bool()
                # Debug only: enable this log when checking 1/16 mask/feature shapes.
                if self.use_dcat16_inject and not getattr(self, "_dcat16_shape_logged", False):
                    logger.info(f"[DCAT16 Inject] feat_c0.shape={feat_c0.shape}, "
                                f"feat_c16_0.shape={feat_c16_0.shape}, "
                                f"mask_c0.shape={mask_c0.shape}, "
                                f"mask16_0.shape={mask16_0.shape}")
                    self._dcat16_shape_logged = True
            feat_c16_t0, feat_c16_t1, matchability16_list0, matchability16_list1 = self.dcat16(
                feat_c16_0, feat_c16_1, mask16_0, mask16_1)
            # Take last matchability score from each image as covisibility
            covi16_0 = matchability16_list0[-1]  # [B, 1, H16, W16]
            covi16_1 = matchability16_list1[-1]  # [B, 1, H16, W16]
            # Inject into 1/8 raw features
            feat_c0, covi8_0 = self.inject16to8(feat_c0, feat_c16_t0, covi16_0)
            feat_c1, covi8_1 = self.inject16to8(feat_c1, feat_c16_t1, covi16_1)
            # Store 1/16 matchability scores for potential loss use
            data.update({
                'matchability_score_list0_16': matchability16_list0,
                'matchability_score_list1_16': matchability16_list1,
            })



        feat_c0, feat_c1, matchability_score_list0, matchability_score_list1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        data.update({
            'matchability_score_list0': matchability_score_list0,
            'matchability_score_list1': matchability_score_list1,
        })

       
        
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        
        # detect NaN during mixed precision training
        if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
            detect_NaN(feat_c0, feat_c1)
        
        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, 
                                mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0, 
                                mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1
                                )

        # prevent fp16 overflow during mixed precision training
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                        [feat_c0, feat_c1])

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, data)
        
        # detect NaN during mixed precision training
        if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
            detect_NaN(feat_f0_unfold, feat_f1_unfold)
        
        del feat_c0, feat_c1, mask_c0, mask_c1

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level            
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)