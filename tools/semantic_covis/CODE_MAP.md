# CoMatch 项目代码地图

> 生成时间: 2026-05-06
> 目的: 语义共视性扩展 - 第一阶段 (CLIP 离线伪标签生成)

---

## 1. 项目训练入口

### 1.1 核心入口文件

| 文件路径 | 说明 |
|---------|------|
| `train.py` | 主训练脚本 |
| `test.py` | 主测试脚本 |
| `src/lightning/lightning_loftr.py` | Lightning Module (`PL_LoFTR`) |
| `src/lightning/data.py` | DataModule (`MultiSceneDataModule`) |

### 1.2 配置文件

| 文件路径 | 说明 |
|---------|------|
| `configs/loftr/comatch_full.py` | 主模型配置 |
| `configs/data/megadepth_trainval_832.py` | MegaDepth 训练数据配置 |
| `configs/data/megadepth_test_1500.py` | MegaDepth 测试配置 |
| `configs/data/scannet_test_1500.py` | ScanNet 测试配置 |
| `src/config/default.py` | 默认配置 |

### 1.3 训练启动示例

```bash
# 训练
python -u ./train.py \
    configs/data/megadepth_trainval_832.py \
    configs/loftr/comatch_full.py \
    --exp_name=your_exp_name \
    --gpus=4 --num_nodes=1 --accelerator="ddp" \
    --batch_size=3 --num_workers=4 \
    --max_epochs=30

# 测试
python ./test.py \
    configs/data/megadepth_test_1500.py \
    configs/loftr/comatch_full.py \
    --ckpt_path=weights/comatch_outdoor.ckpt \
    --gpus=1 --num_nodes=1 --accelerator="ddp"
```

---

## 2. 数据读取逻辑

### 2.1 Dataset / DataLoader 类

| 文件路径 | 类名/函数名 | 说明 |
|---------|------------|------|
| `src/datasets/megadepth.py` | `MegaDepthDataset` | MegaDepth 数据集类 |
| `src/datasets/scannet.py` | `ScanNetDataset` | ScanNet 数据集类 |
| `src/utils/dataset.py` | `read_megadepth_gray()` | MegaDepth 图像读取+预处理 |
| `src/utils/dataset.py` | `read_scannet_gray()` | ScanNet 图像读取 |
| `src/lightning/data.py` | `MultiSceneDataModule` | PL DataModule，整合所有 Dataset |

### 2.2 image0 / image1 读取位置

**MegaDepth** (`src/datasets/megadepth.py:71-84`):
```python
def __getitem__(self, idx):
    (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

    img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
    img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])

    image0, mask0, scale0 = read_megadepth_gray(...)
    image1, mask1, scale1 = read_megadepth_gray(...)
```

**ScanNet** (`src/datasets/scannet.py:73-86`):
```python
img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
image0 = read_scannet_gray(img_name0, resize=self.img_resize, augment_fn=None)
image1 = read_scannet_gray(img_name1, resize=self.img_resize, augment_fn=None)
```

### 2.3 Image Pair 唯一标识构造

**文件**: `src/datasets/megadepth.py:122`

```python
'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1])
```

- pair_names 是两个图像路径组成的 tuple
- 唯一标识 = `'scene_001/images/001.jpg#scene_001/images/002.jpg'` (用 `#` 连接)

### 2.4 MegaDepth Train Pair 列表 / NPZ 数据加载

**配置文件** (`configs/data/megadepth_trainval_832.py`):
```python
TRAIN_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list.txt"
```

**NPZ 加载位置** (`src/datasets/megadepth.py:42-52`):
```python
self.scene_info = np.load(npz_path, allow_pickle=True)
self.pair_infos = self.scene_info['pair_infos'].copy()
del self.scene_info['pair_infos']
self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]
```

**NPZ 文件包含**:
- `pair_infos`: 图像对索引和 overlap score
- `image_paths`: 图像路径列表
- `depth_paths`: 深度图路径列表
- `poses`: 相机位姿 (4x4)
- `intrinsics`: 相机内参

### 2.5 图像预处理 (Resize / Padding / Crop / Long Edge 832)

**核心函数**: `src/utils/dataset.py`

| 函数 | 位置 | 功能 |
|-----|------|------|
| `get_resized_wh()` | 55-61 | 长边 resize (如 832) |
| `get_divisible_wh()` | 64-69 | 向下取整到 8 的倍数 |
| `pad_bottom_right()` | 72-89 | 零填充到正方形 |
| `read_megadepth_gray()` | 94-126 | 完整预处理流程 |

**关键参数** (`configs/data/megadepth_trainval_832.py`):
```python
cfg.DATASET.MGDPT_IMG_RESIZE = 832  # long edge
cfg.DATASET.MGDPT_DF = 8            # divisible by 8
cfg.DATASET.MGDPT_IMG_PAD = True     # zero-pad to square
```

**预处理流程**:
```
原图 (如 1920x1080)
    ↓
get_resized_wh() → 长边 resize 到 832
    ↓
get_divisible_wh() → 向下取整到 8 的倍数 → (832, 468)
    ↓
pad_bottom_right() → 零填充到正方形 → (832, 832)
    ↓
输出: image [1, 832, 832], mask [832, 832], scale [w/w_new, h/h_new]
```

### 2.6 Coarse Feature 尺度 Hc/Wc

**配置** (`src/config/default.py:8`):
```python
_CN.LOFTR.RESOLUTION = (8, 1)
```

**计算** (`src/lightning/data.py:77`):
```python
self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125 = 1/8
```

**结论**:
- **Hc = H_img / 8, Wc = W_img / 8** (固定比例)
- 例如: 输入 832x832 图像 → Coarse feature = **104x104**

---

## 3. CoMatch 模型结构

### 3.1 文件位置总览

| 模块 | 文件路径 |
|-----|---------|
| 主模型 | `src/loftr/loftr.py` |
| Backbone | `src/loftr/backbone/resnet.py` |
| Transformer | `src/loftr/loftr_module/transformer.py` |
| Coarse Matching | `src/loftr/utils/coarse_matching.py` |
| Fine Matching | `src/loftr/utils/fine_matching_epipolar.py` |
| Loss | `src/losses/loftr_loss_epipolar.py` |
| 几何工具 | `src/loftr/utils/geometry.py` |
| 可视化 | `src/utils/plotting.py` |

### 3.2 Backbone

**文件**: `src/loftr/backbone/resnet.py`

| 类名 | 说明 |
|-----|------|
| `BasicBlock` | ResNet 基本残差块 |
| `ResNet_8_1_align` | 主干网络类 |

**输出**:
- `feat_x2`: 1/2 分辨率特征 (H/2, W/2)
- `feat_x1`: 1/8 分辨率特征 (H/8, W/8)

### 3.3 DCAT / AG_RoPE Transformer

**注意**: CoMatch 使用 **AG_RoPE_EncoderLayer** 而非 DCAT

**文件**: `src/loftr/loftr_module/transformer.py`

| 类名 | 说明 |
|-----|------|
| `LocalFeatureTransformer` | Local Feature Transformer (8 层) |
| `AG_RoPE_EncoderLayer` | Attention-guided with RoPE 的 Encoder Layer |

### 3.4 Covisibility Score 预测位置

**文件**: `src/loftr/loftr_module/transformer.py`

**定义** (第 133-136 行):
```python
self.matchability_predictor = nn.ModuleList([nn.Sequential(
    nn.Conv2d(config['d_model'], config['d_model'], kernel_size=3, padding=1, bias=False, groups=config['d_model']),
    nn.ReLU(inplace=True),
    nn.Conv2d(config['d_model'], 1, kernel_size=1, stride=1, bias=True)) for _ in range(len(self.layer_names)//2 - 1)])
```

**预测调用** (第 183-184 行):
```python
matchability_score0 = torch.sigmoid(self.matchability_predictor[i//2](feat0))
matchability_score1 = torch.sigmoid(self.matchability_predictor[i//2](feat1))
```

**存储** (第 166 行):
```python
matchability_score_list0, matchability_score_list1 = [], []
```

**输出张量**:
- `matchability_score`: shape `[B, 1, Hc, Wc]` (H/8, W/8)
- `matchability_score_list0`, `matchability_score_list1`: 列表，每个元素对应一层

### 3.5 CGTC 使用 Covisibility Score 的位置

**CGTC = Cross-guided Transformer with Covisibility**

**文件**: `src/loftr/loftr_module/transformer.py`

**AG_RoPE_EncoderLayer.forward()** 中的使用:

| 位置 | 功能 |
|-----|------|
| 第 78-79 行 | Source 加权 (unfold 后求和) |
| 第 80 行 | Query 加权 (`aggregate` 操作) |
| 第 92-93 行 | Value 加权 |

```python
# A. Query 加权
query, source = self.norm1(self.aggregate(x * x_matchability_score).permute(0,2,3,1))

# B. Source 加权
source_unfold = torch.sum(source_unfold * source_matchability_score_unfold.unsqueeze(1), dim=2)

# C. Value 加权
if source_matchability_score != None:
    value = value * pooled_source_matchability_score
```

### 3.6 CAA 使用 Covisibility Score 的位置

**CAA = Covisibility-guided Attention Aggregation**

CAA 与 CGTC 共用同一机制，在 `LocalFeatureTransformer.forward()` (第 145-227 行) 中：

```python
# Cross Attention 中使用
feat0 = layer(feat0, feat1, mask0, mask1, matchability_score0, matchability_score1, name)
feat1 = layer(feat1, feat0, mask1, mask0, matchability_score1, matchability_score0, name)
```

### 3.7 Coarse Matching / Dual-Softmax

**文件**: `src/loftr/utils/coarse_matching.py`

| 类名 | 方法 | 说明 |
|-----|------|------|
| `CoarseMatching` | `forward()` | 计算相似度矩阵 |
| `CoarseMatching` | `get_coarse_match()` | 提取粗匹配结果 |

**Dual-Softmax 实现** (第 121-124 行):
```python
if self.skip_softmax:
    sim_matrix = sim_matrix
else:
    sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
```

**关键张量**:
- `sim_matrix`: shape `[B, N, M]` (N=Hc*Wc, M=Hc*Wc)
- `conf_matrix`: shape `[B, Hc, Wc]`
- `mkpts0_c`, `mkpts1_c`: 粗匹配关键点
- `mconf`: 匹配置信度

### 3.8 Loss 汇总位置

**文件**: `src/losses/loftr_loss_epipolar.py`

**类名**: `LoFTRLoss`

| 方法 | 功能 |
|-----|------|
| `class_focal_loss()` | Matchability focal loss |
| `compute_coarse_loss()` | Coarse-level 匹配损失 |
| `compute_fine_loss()` | Fine-level 像素级损失 |
| `_compute_local_loss_epipolar()` | 子像素级极线损失 |
| `_compute_local_loss_l2()` | 子像素级 L2 损失 |

**Loss 汇总** (第 202-293 行):
```python
loss = loss_c * coarse_weight      # Coarse matching loss
    + loss_f * fine_weight          # Fine matching loss
    + loss_l * local_weight         # Subpixel loss
    + matchablity_loss * 0.25       # Matchability loss
```

---

## 4. 可视化工具现状

### 4.1 已有工具

**文件**: `src/utils/plotting.py`

| 函数 | 功能 |
|-----|------|
| `make_matching_figure()` | 绘制图像对匹配结果 |
| `_make_evaluation_figure()` | 评估结果可视化 |
| `make_matching_figures()` | 批量生成匹配图 |
| `dynamic_alpha()` | 动态调整透明度 |
| `error_colormap()` | 误差颜色映射 |

### 4.2 中间结果存储

| 文件 | 存储内容 |
|-----|---------|
| `src/loftr/loftr.py` | feat_c0, feat_c1, matchability_score_list |
| `src/loftr/utils/coarse_matching.py` | conf_matrix, mkpts0_c, mkpts1_c |
| `src/loftr/utils/fine_matching_epipolar.py` | heatmap0, heatmap1, mkpts0_f |

### 4.3 Covisibility Heatmap

**文件**: `src/loftr/utils/geometry.py`

| 函数 | 功能 |
|-----|------|
| `warp_kpts()` | 计算共可见性掩码 |
| `warp_kpts_ada()` | 自适应变形 (带深度一致性检查) |

**输出**:
- `covisible_mask0`, `covisible_mask1`: 共可见性掩码
- `consistent_mask0`, `consistent_mask1`: 深度一致性掩码

### 4.4 训练可视化

**文件**: `src/lightning/lightning_loftr.py`

- `validation_step()`: 使用 `make_matching_figures()` 生成可视化
- 记录到 TensorBoard

### 4.5 缺失的工具

- 没有独立的 `tools/` 可视化脚本
- 没有保存特征到文件的工具 (`np.save`/`torch.save`)
- 没有保存 covisibility heatmap 的独立工具
- `dump_dir` 功能已定义但未实现

---

## 5. 最小改动文件清单

### 5.1 第一阶段: CLIP 离线伪标签生成

**新增文件** (不需要改动现有代码):

| 文件路径 | 说明 |
|---------|------|
| `tools/semantic_covis/generate_pseudo_labels.py` | 主脚本 |
| `tools/semantic_covis/clip_feature_extractor.py` | CLIP 特征提取 |
| `tools/semantic_covis/covisibility_estimator.py` | 共视性估计 |
| `tools/semantic_covis/npz_to_pseudo_labels.py` | NPZ 到伪标签转换 |

### 5.2 第二阶段: 接入训练 (预估)

**需要改动的文件**:

| 文件路径 | 改动内容 |
|---------|---------|
| `src/utils/dataset.py` | 添加加载伪标签的函数 |
| `src/datasets/megadepth.py` | 在 `__getitem__` 中加载伪标签 |
| `src/loftr/loftr_module/transformer.py` | 在 CGTC/CAA 中添加 semantic covisibility |
| `src/config/default.py` | 添加 semantic covisibility 配置 |
| `configs/loftr/comatch_full.py` | 添加新配置项 |

---

## 6. 离线伪标签脚本设计建议

### 6.1 输入

- MegaDepth train pair NPZ 文件 (与训练相同)
- CLIP 模型 (如 openai/clip-vit-base-patch32)

### 6.2 输出 

伪标签 NPZ 文件，包含:

| 字段 | Shape | 说明 |
|-----|-------|------|
| `scene_id` | str | 场景 ID |
| `pair_infos` | list | 图像对信息 (沿用原格式) |
| `semantic_covis_0` | [N, Hc, Wc] | image0 的语义共视性 |
| `semantic_covis_1` | [N, Hc, Wc] | image1 的语义共视性 |
| `clip_features_0` | [N, Hc, Wc, D] | image0 的 CLIP 特征 |
| `clip_features_1` | [N, Hc, Wc, D] | image1 的 CLIP 特征 |

### 6.3 读取 Image Pair 的方式

```python
# 方案 1: 直接复用 Dataset 逻辑
from src.datasets.megadepth import MegaDepthDataset
from src.utils.dataset import read_megadepth_gray

# 方案 2: 从 NPZ 读取
import numpy as np
scene_info = np.load(npz_path, allow_pickle=True)
image_paths = scene_info['image_paths']
poses = scene_info['poses']
intrinsics = scene_info['intrinsics']

# 读取图像
for idx0, idx1 in pair_indices:
    img_path = image_paths[idx0]
    image = read_megadepth_gray(img_path, resize=832, df=8, padding=True)
```

### 6.4 关键尺寸

| 参数 | 值 |
|-----|---|
| Long edge resize | 832 |
| Divisible by | 8 |
| Coarse feature size | H/8 x W/8 |
| 例如 832x832 输入 | 104x104 coarse |

---

## 7. 关键张量总结

| 张量名 | Shape | 位置 |
|-------|-------|------|
| `image0`, `image1` | [B, 1, H, W] | Dataset output |
| `feat_c0`, `feat_c1` | [B, 256, Hc, Wc] | Backbone output |
| `matchability_score` | [B, 1, Hc, Wc] | matchability_predictor |
| `matchability_score_list` | List[[B,1,Hc,Wc]] | LocalFeatureTransformer |
| `sim_matrix` | [B, N, M] | CoarseMatching |
| `conf_matrix` | [B, Hc, Wc] | CoarseMatching |
| `covisible_mask` | [B, Hc, Wc] | geometry.warp_kpts |
| `spv_matchability_map` | [B, 1, Hc, Wc] | supervision.get_scale_gt_matrix5 |

---

## 8. 参考配置文件

```python
# configs/data/megadepth_trainval_832.py 关键参数
cfg.DATASET.TRAIN_NPZ_ROOT = "data/megadepth/index/scene_info_0.1_0.7"
cfg.DATASET.TRAIN_LIST_PATH = "data/megadepth/index/trainvaltest_list/train_list.txt"
cfg.DATASET.MGDPT_IMG_RESIZE = 832
cfg.DATASET.MGDPT_DF = 8
cfg.DATASET.MGDPT_IMG_PAD = True

# src/config/default.py 关键参数
cfg.LOFTR.RESOLUTION = (8, 1)  # coarse scale = 1/8
```
