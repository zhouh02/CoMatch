# CoMatch dev/exp1 — Coarse Matching 之前数据流详细报告

> 撰写日期：2026-04-26
> 分支：dev/exp1
> 实验主题：1/16 DCAT + covisibility-guided inject 到 1/8 coarse feature

---

## 一、涉及文件与函数

### 1. 训练入口与 Lightning 封装

| 文件 | 类 / 函数 | 负责什么 | 原始 / 新增 |
|------|-----------|---------|-------------|
| `src/lightning/lightning_loftr.py` | `PL_LoFTR` | 封装 LoFTR matcher，调用 `self.matcher(batch)`；调用 `compute_supervision_coarse`；计算 loss | 原始 |
| `src/loftr/utils/supervision.py` | `compute_supervision_coarse` / `spvs_coarse` | 根据深度图 + 位姿 warp grid 生成 `spv_matchability_map0/1`（共视 GT）和 `conf_matrix_gt`，**原始图像尺度上**生成 mask | 原始 |
| `src/datasets/megadepth.py` | `read_megadepth_gray` | 读取图像和 mask，`mask0/mask1` 形状为 `[B, H_pad, W_pad]`，是 **原始图像尺寸** | 原始 |

### 2. 主模型

| 文件 | 类 / 函数 | 负责什么 | 原始 / 新增 |
|------|-----------|---------|-------------|
| `src/loftr/loftr.py` | `LoFTR.__init__` | 构建所有子模块；`use_dcat16_inject=False` 时不实例化 dcat16/inject16to8 | 新增 `use_dcat16_inject` 分支 |
| `src/loftr/loftr.py` | `LoFTR.forward` | 主数据流编排（见下方详述） | 大幅新增 |
| `src/loftr/loftr.py` | `CovisibilityInject` | 1/16 → 1/8 residual inject 模块 | **新增** |
| `src/loftr/loftr.py` | `load_state_dict` | 加载 checkpoint，兼容 `matcher.` 前缀 | 原始 |

### 3. 骨干网络

| 文件 | 类 / 函数 | 负责什么 | 原始 / 新增 |
|------|-----------|---------|-------------|
| `src/loftr/backbone/resnet.py` | `ResNet_8_1_align` | stem → layer1(1/2) → layer2(1/4) → layer3(1/8) → **layer4(1/16)** | **新增 layer4 + feats_c16** |
| `src/loftr/backbone/resnet.py` | `BasicBlock` | 2 层 conv3x3 + BN + residual | 原始 |

### 4. Transformer / DCAT

| 文件 | 类 / 函数 | 负责什么 | 原始 / 新增 |
|------|-----------|---------|-------------|
| `src/loftr/loftr_module/transformer.py` | `LocalFeatureTransformer` | 8 层 self/cross attention + matchability prediction，**同时充当原始 DCAT 和实验中的 dcat16** | 原始（无改动） |
| `src/loftr/loftr_module/transformer.py` | `AG_RoPE_EncoderLayer` | 带 RoPE 的聚合 attention 层 | 原始 |
| `src/loftr/loftr_module/transformer.py` | `RoPEPositionEncodingSine` | 2D 旋转位置编码 | 原始 |

### 5. 粗匹配

| 文件 | 类 / 函数 | 负责什么 | 原始 / 新增 |
|------|-----------|---------|-------------|
| `src/loftr/utils/coarse_matching.py` | `CoarseMatching` | dual-softmax + MNN 粗匹配，接收 `[N, HW, C]` flatten 特征 | 原始（无改动） |

---

## 二、当前完整数据流

### 流程图

```
image0 / image1  [B, 1, H, W]
    │
    ▼
backbone (ResNet_8_1_align)
    ├─ stem: conv7x7 stride2 + BN + ReLU
    ├─ layer1: stride1 → 1/2
    ├─ layer2: stride2 → 1/4
    ├─ layer3: stride2 → 1/8    ← feats_c / feats_c0 / feats_c1
    │                              feats_x2 (1/4) / feats_x1 (1/2)
    └─ layer4: stride2 → 1/16   ← feats_c16 / feat_c16_0 / feat_c16_1 (★新增★)
    │
    ▼
ret_dict = {
    'feats_c':    [B, 256, H/8,  W/8 ]   ← feats_c0/1 split 自此
    'feats_c16':  [B, 256, H/16, W/16]   ← feat_c16_0/1 split 自此  (★新增★)
    'feats_x2':   [B, 128, H/4,  W/4 ]   ← 1/4 fine feature
    'feats_x1':   [B, 64,  H/2,  W/2 ]   ← 1/2 fine feature
}
    │
    ▼
构造 mask_c0 / mask_c1
    来源：data['mask0'] / data['mask1']
    原始形状：[B, H_orig, W_orig]  ← 原始图像尺寸
    在 LoFTR.forward 中：不做 resize，直接从 data 中取出
    │
    ▼
★新增★  构造 mask16_0 / mask16_1
    若 mask_c0.dim() == 3（[B, H_orig, W_orig]）：
        mask16_0 = nearest_interpolate(mask_c0.float, size=feat_c16_0.shape[-2:])
    若 mask_c0.dim() == 2（[B, H*W]，已被 flatten）：
        mask16_0 = nearest_interpolate(
            mask_c0.view([B, H8, W8]).float,
            size=feat_c16_0.shape[-2:])
    结果：mask16_0: [B, H/16, W/16]，bool
    │
    ▼
★新增★  dcat16 (LocalFeatureTransformer，config=COARSE16)
    输入：
        feat_c16_0:  [B, 256, H/16, W/16]
        feat_c16_1:  [B, 256, H/16, W/16]
        mask16_0:    [B, H/16, W/16]  (bool)
        mask16_1:    [B, H/16, W/16]  (bool)
    输出：
        feat_c16_t0: [B, 256, H/16, W/16]  ← transformed feature
        feat_c16_t1: [B, 256, H/16, W/16]
        matchability16_list0: list of [B, 1, H/16, W/16] × 3
        matchability16_list1: list of [B, 1, H/16, W/16] × 3
    │
    ▼
★新增★  取 covi16（第 3 个 cross 层后的 matchability 作为共视分数）
    covi16_0 = matchability16_list0[-1]  ← [B, 1, H/16, W/16]
    covi16_1 = matchability16_list1[-1]  ← [B, 1, H/16, W/16]
    │
    ▼
★新增★  inject16to8 (CovisibilityInject)
    feat_c0_fused, covi8_0 = inject16to8(feat_c0_raw, feat_c16_t0, covi16_0)
    feat_c1_fused, covi8_1 = inject16to8(feat_c1_raw, feat_c16_t1, covi16_1)
    feat_c0 / feat_c1 原地被替换为 fused 版本
    │
    ▼
原始 loftr_coarse (LocalFeatureTransformer，config=COARSE)
    输入：
        feat_c0_fused:  [B, 256, H/8, W/8]  ← 已不再是 raw
        feat_c1_fused:  [B, 256, H/8, W/8]
        mask_c0:        [B, H_orig, W_orig]  ← ★仍然是原始图像尺度★
        mask_c1:        [B, H_orig, W_orig]
    输出：
        feat_c0:        [B, 256, H/8, W/8]  ← transformed
        feat_c1:        [B, 256, H/8, W/8]
        matchability_score_list0/1
    │
    ▼
rearrange feat_c0/feat_c1: [N,C,H,W] → [N,HW,C]
    feat_c0: [B, H/8*W/8, 256]
    feat_c1: [B, H/8*W/8, 256]
    │
    ▼
data['hw0_c'] / data['hw1_c']  ← feat_c0.shape[2:] = [H/8, W/8]  ★始终 1/8，不受 1/16 分支影响★
    │
    ▼
coarse_matching (CoarseMatching)
    输入：
        feat_c0: [B, HW/8*8, 256]   ← **从未看到 feat_c16**
        feat_c1: [B, HW/8*8, 256]
        mask_c0: [B, H/8*W/8] (flatten 后)
        mask_c1: [B, H/8*W/8]
        data['hw0_c'] / data['hw1_c']: [H/8, W/8]
```

### 关键结论

- **当前不是"1/16 DCAT 替代 1/8 DCAT"**，而是"1/16 DCAT inject + 原始 1/8 loftr_coarse"串联
- **coarse matching 始终只看到 1/8 特征**，不知道 1/16 分支存在
- **data['hw0_c'] / data['hw1_c'] 始终保持 1/8 尺度**，不随 inject 而变
- **新增的 feat_c16_0/1 只在 inject 阶段使用**，inject 后不再保留引用

---

## 三、所有关键变量 Shape

| 变量名 | Shape | 尺度 | 来源 | 用途 |
|--------|-------|------|------|------|
| `image0` / `image1` | `[B, 1, H, W]` | 原图 | dataset | backbone 输入 |
| `data['mask0']` / `data['mask1']` | `[B, H_orig, W_orig]` | 原图 | dataset，原始图像 padding mask | loftr_coarse 的 mask 下采样来源 |
| `feats_c` (backbone 拼接后) | `[2B, 256, H/8, W/8]` | 1/8 | backbone layer3 | split 为 feat_c0/1 |
| `feats_c0` / `feats_c1` | `[B, 256, H/8, W/8]` | 1/8 | backbone split | **inject 输入（原始）** / loftr_coarse 输入 |
| `feats_c16` (backbone 拼接后) | `[2B, 256, H/16, W/16]` | 1/16 | backbone layer4 (★新增★) | split 为 feat_c16_0/1 |
| `feat_c16_0` / `feat_c16_1` | `[B, 256, H/16, W/16]` | 1/16 | backbone split | dcat16 输入 |
| `feats_x2` | `[2B, 128, H/4, W/4]` | 1/4 | backbone layer2 | FinePreprocess FPN |
| `feats_x1` | `[2B, 64, H/2, W/2]` | 1/2 | backbone layer1 | FinePreprocess FPN |
| `mask_c0` / `mask_c1` | `[B, H_orig, W_orig]` | 原图 | `data['mask0'] / data['mask1']` | 传给 loftr_coarse 的 mask（不变） |
| `mask16_0` / `mask16_1` | `[B, H/16, W/16]` | 1/16 | `mask_c0` nearest downsample (★新增★) | dcat16 的 mask |
| `feat_c16_t0` / `feat_c16_t1` | `[B, 256, H/16, W/16]` | 1/16 | dcat16 输出 | inject 输入 |
| `matchability16_list0` / `list1` | `list of [B, 1, H/16, W/16]` (3个) | 1/16 | dcat16 输出 | 取 `[-1]` 作为 covi16 |
| `covi16_0` / `covi16_1` | `[B, 1, H/16, W/16]` | 1/16 | `matchability16_list0[-1]` | inject 的 soft gate |
| `covi8_0` / `covi8_1` | `[B, 1, H/8, W/8]` | 1/8 | inject 输出 | 训练可记录，**不传入 loss** |
| `feat_c0_fused` / `feat_c1_fused` | `[B, 256, H/8, W/8]` | 1/8 | inject 输出 | 原地替换 feat_c0/1，传给 loftr_coarse |
| `loftr_coarse` 输出 `feat_c0/feat_c1` | `[B, 256, H/8, W/8]` | 1/8 | loftr_coarse | rearrange 后传入 coarse_matching |
| rearrange 后 `feat_c0/feat_c1` | `[B, H/8*W/8, 256]` | 1/8 flatten | loftr.py forward | **coarse_matching 的真正输入** |
| `matchability_score_list0/1` | `list of [B, 1, H/8, W/8]` (3个) | 1/8 | loftr_coarse 输出 | **传入 loss**（不是 matchability16_list） |
| `data['hw0_c'] / data['hw1_c']` | `[H/8, W/8]` | 1/8 | `feat_c0.shape[2:]` | coarse_matching 中算 scale |
| `data['hw0_f'] / data['hw1_f']` | `[H/4, W/4]` | 1/4 | `feat_c0.shape[2:] * mul` | FinePreprocess stride |
| `matchability_score_list0_16` / `_16` | `list of [B, 1, H/16, W/16]` | 1/16 | data.update | **不传入 loss**，仅记录 |

---

## 四、Mask 逻辑

### 4.1 原始 mask 是怎么来的

`data['mask0'] / data['mask1']` 来自 `megadepth.py` 的 `read_megadepth_gray`：

```python
# megadepth.py
image0, mask0, scale0 = read_megadepth_gray(...)
data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
```

`mask0` 是**原始图像尺寸**（如 640×480 之类的 resize 后尺寸）的 padding mask。`'0'` 表示 padding 区域（应被忽略），`'1'` 表示有效像素。

在 `spvs_coarse` 中：
```python
if 'mask0' in data:
    grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
```
mask 作用于 **原始图像尺度**（`grid_pt0_i` 乘以 `scale` 后再 mask），不直接作用于 coarse feature。

### 4.2 mask_c0 / mask_c1 是原始图像尺度还是 1/8 尺度？

**原始图像尺度**：`data['mask0']` / `data['mask1']` 直接从 dataset 传入，未经 resize。

在 `LoFTR.forward` 中：
```python
mask_c0, mask_c1 = data['mask0'], data['mask1']  # [B, H_orig, W_orig]
```

它们被传给 `loftr_coarse`。在 `AG_RoPE_EncoderLayer` 中，mask 会经过 `max_pool` 下采样到当前层的尺度。

### 4.3 mask16_0 / mask16_1 是如何下采样的

```python
if mask_c0.dim() == 3:  # [B, H_orig, W_orig]
    mask16_0 = F.interpolate(
        mask_c0.float().unsqueeze(1),   # → [B, 1, H_orig, W_orig]
        size=feat_c16_0.shape[-2:],        # → [B, 1, H/16, W/16]
        mode='nearest'
    ).squeeze(1).bool()
```

两种分支：
- `mask_c0.dim() == 3`：`[B, H_orig, W_orig]` 直接 nearest 下采样到 1/16 尺寸
- `mask_c0.dim() == 2`：`[B, H*W]` flatten 形式，先 reshape 到 `[B, H8, W8]` 再下采样

### 4.4 nearest 下采样的合理性

**存在边界风险**：nearest 下采样对 mask 是"最邻近复制"，如果 padding 区域和有效区域边界不在 2 的倍数上，nearest 可能错误地将部分有效区域标记为 padding 或反之。

更安全的做法是用 **mean pooling** 或 **max pooling**：
- max pooling：只要感受野内有有效像素就保留 1
- mean pooling：感受野内有效像素占比低于阈值则标记为 0

但当前用 nearest，主要考虑是 mask 是 binary 的（0/1），nearest 可以保持锐利的边界——只要图像尺寸是 16 的倍数，边界就对齐良好。

### 4.5 原始 loftr_coarse 是否仍用 1/8 mask？

**是**：`mask_c0 / mask_c1` 仍然是 `[B, H_orig, W_orig]`（原始图像尺度），不变。

`loftr_coarse` 内部会通过 `max_pool` 将 mask 下采样到当前特征尺度（1/8），用于在 attention 中 mask 掉 padding 区域。

### 4.6 data['hw0_c'] / data['hw1_c'] 是否被污染？

**没有被污染**：在 `LoFTR.forward` 中：

```python
data.update({
    'hw0_c': feat_c0.shape[2:],  # ← feat_c0 此时还是 1/8
    'hw1_c': feat_c1.shape[2:],  # ← 在 inject 之后才替换，但这里用的是引用前的值
    ...
})
```

这段代码在 `mask_c0 = data['mask0']` 之后、`dcat16` 之前执行。`data['hw0_c']` 基于 `feat_c0.shape[2:]`（此时仍是 1/8 raw），所以始终是 1/8 尺度。

---

## 五、DCAT16 分支细节

### 5.1 dcat16 是不是独立 LocalFeatureTransformer 实例？

**是**：`dcat16` 是 `LocalFeatureTransformer(config16)` 的独立实例，与 `loftr_coarse` 完全独立。

```python
config16 = copy.deepcopy(config)
config16['coarse'] = config['coarse16']  # 使其读取 COARSE16 配置
self.dcat16 = LocalFeatureTransformer(config16)
```

### 5.2 是否复用原始 transformer 类？

**是**：`LocalFeatureTransformer` 类本身没有任何改动。dcat16 使用相同的类，但通过 `config16['coarse']` 指向 `COARSE16` 配置，获得了不同的参数。

### 5.3 COARSE16 配置参数

| 参数 | COARSE (原始) | COARSE16 (实验) |
|------|--------------|-----------------|
| `d_model` | 256 | 256 |
| `nhead` | 8 | 8 |
| `layer_names` | `['self','cross']*4` (8层) | `['self','cross']*4` (8层) |
| `agg_size0` | 4 | **2** |
| `agg_size1` | 4 | **2** |
| `no_flash` | True | True |
| `rope` | True | True |
| `npe` | None | None（从 COARSE 继承） |

### 5.4 agg_size=2 在 1/16 上的物理意义

原始 1/8 特征上 `agg_size=4`：每个 token 聚合 4×4=16 个像素，输出 1 个 token。
在 1/16 特征上 `agg_size=2`：每个 token 聚合 2×2=4 个像素。

换算到原图：
- 原始：4×8=32 像素的局部区域聚合为 1 个 coarse token
- 1/16 分支：2×16=32 像素的局部区域聚合为 1 个 1/16 token

**感受野相同（32px）**，但 1/16 分支在更低分辨率上计算，注意力头能看到更大的空间范围。

### 5.5 matchability_score 输出机制

在 8 层 `['self','cross']*4` 配置下：
- `matchability_predictor` 有 `8//2 - 1 = 3` 个
- 输出时机：`i=3,5,7` 三个 cross 层之后各输出一次
- `matchability16_list0/1` 长度为 3
- 最后一个（`i=7`）被取为 `covi16_0/1`

### 5.6 matchability16_list 是否进入 loss？

**不进入**：loss 中使用的 `matchability_score_list0/1` 来自 `loftr_coarse`（1/8）的输出，与 `matchability16_list0/1` 完全独立。

`matchability16_list0/1` 被存入 `data['matchability_score_list0_16'] / `data['matchability_score_list1_16']`，**loss 不读取这些 key**。

**这是当前实验设计**，没有显式 1/16 covisibility loss。dcat16 的监督信号完全依赖：
1. inject 后的 feat_c0_fused 通过 loftr_coarse → coarse_matching → loss 的间接梯度
2. alpha 可学习参数从 loss 反向传播

### 5.7 当前 dcat16 与原始 1/8 loftr_coarse 的关系

**串联，不是替代**：

```
backbone → 1/16 feat ──→ dcat16 ──→ inject ──→ fused 1/8 feat ──→ loftr_coarse ──→ coarse_matching
backbone → 1/8 feat ───────────────────────────────────────────────────→ loftr_coarse
```

- dcat16 在 inject 阶段处理 1/16 特征，输出 transformed 1/16 特征和 matchability
- inject 将 1/16 transformed feature 上采样到 1/8，与 raw 1/8 feature 融合
- 融合后的 1/8 feature 进入**原始的** loftr_coarse（不是替代！）
- loftr_coarse 仍然完整执行 8 层 self/cross attention

---

## 六、Inject Layer 细节

### 6.1 输入输出

```python
class CovisibilityInject(nn.Module):
    def __init__(self, c8_dim, c16_dim):  # c8_dim=256, c16_dim=256
        self.proj16 = nn.Conv2d(c16_dim, c8_dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))  # 初始化为 0.0
```

| | Shape | 描述 |
|--|-------|------|
| 输入 `feat8` | `[B, 256, H/8, W/8]` | raw 1/8 features |
| 输入 `feat16` | `[B, 256, H/16, W/16]` | transformed 1/16 features (dcat16 输出) |
| 输入 `covi16` | `[B, 1, H/16, W/16]` | matchability score (covi16_0/1) |
| 输出 `feat8_fused` | `[B, 256, H/8, W/8]` | fused feature |
| 输出 `covi8` | `[B, 1, H/8, W/8]` | 上采样后的 covisibility |

### 6.2 公式

```
feat16_up   = Bilinear_Interpolate(feat16_t,   size=(H/8, W/8))
covi8       = Bilinear_Interpolate(covi16,     size=(H/8, W/8))
injected    = Conv1x1(feat16_up)               # 256→256
feat8_fused = feat8_raw + alpha * covi8 * injected
```

- `feat8_raw`：原始 1/8 特征，**通过 residual 完整保留**
- `alpha`：**可学习标量**，初始化为 0.0（训练初期等价于原始 F8）
- `covi8`：作为 **soft gate**（0~1 之间的 matchability），不做 hard mask
- **无额外 normalize**：不改变特征分布，只是 residual addition

### 6.3 与建议公式的对比

用户建议的公式：
```python
feat8_fused = feat8_raw + alpha * C8_from16 * Proj(T8_from16)
```

实际实现完全一致：
- `C8_from16` = `covi8`（bilinear 上采样后的 covisibility）
- `Proj` = `Conv1x1` (proj16)
- `T8_from16` = `feat16_up`（bilinear 上采样后的 transformed 1/16 feature）
- `alpha` = `nn.Parameter(torch.zeros(1))`

---

## 七、与原始 CoMatch 的区别

### 7.1 原始 CoMatch 数据流

```
image → backbone(1/8 feats_c) → loftr_coarse → coarse_matching
                                     ↑
                               mask: 原图尺度
```

### 7.2 当前实验版数据流

```
image → backbone(1/8 feats_c + 1/16 feats_c16)
                     ↓              ↓
              feats_c0/1     feat_c16_0/1
                     ↓              ↓
              mask_c0/1      mask16_0/1 (nearest downsample)
                     ↓              ↓
              loftr_coarse ← dcat16 → inject → loftr_coarse
                     ↓
              coarse_matching
```

### 7.3 对比表

| 项目 | 原始 CoMatch | 当前实验版 |
|------|-------------|-----------|
| coarse_matching 输入 shape | `[B, HW/8*8, 256]` | `[B, HW/8*8, 256]`（完全一致） | |
| coarse_matching 内部逻辑 | 不变 | **不变** |
| fine_matching / loss | 不变 | **不变** |
| data['hw0_c'] / data['hw1_c'] | 1/8 | 1/8（不变） |
| backbone 参数量 | layer1~3 | layer1~4（新增 ~2 层 BasicBlock） |
| transformer 参数量 | loftr_coarse | loftr_coarse + dcat16（新增 1 套完整 LocalFeatureTransformer） |
| inject 参数量 | 无 | Conv1x1(256→256) + alpha |
| checkpoint 兼容性 | — | **strict=False** 加载，旧 ckpt 会缺少新增参数 |

---

## 八、潜在风险与需要验证的点

### 8.1 梯度流验证

**风险**：dcat16 / inject16to8 是否真的有梯度？

分析：
- `inject` 输出 `feat8_fused` 被传给 `loftr_coarse`
- `loftr_coarse` 输出 `feat_c0` 被传给 `coarse_matching`
- `coarse_matching` 输出 `conf_matrix` 被 loss 使用
- loss → coarse_matching → loftr_coarse → inject → **dcat16** 的梯度路径是完整的

验证方法：
```python
for name, param in model.matcher.named_parameters():
    if param.requires_grad:
        print(f"GRAD: {name}, shape={param.shape}")
# 确认 dcat16.* / inject16to8.* / backbone.layer4.* 都在列表中
```

### 8.2 alpha 初始化为 0 的影响

**风险**：alpha=0 意味着训练初期 dcat16 的输出完全被忽略，梯度可能很弱。

分析：
- alpha=0 → inject 输出 = feat8_raw → 前向传播等价于原始 CoMatch
- 但 dcat16 本身仍有梯度（通过 loftr_coarse → loss 传回）
- alpha 在第一个 epoch 内会开始更新

建议：记录 alpha 的训练曲线，确认它是否快速收敛到非零值。

### 8.3 unused_parameters

**风险**：`find_unused_parameters=True` 只是绕过 DDP 报错，不意味着没有 unused 参数。

分析：
- 如果 `USE_DCAT16_INJECT=True` 但 dcat16 的输出从未被使用，参数就不会更新
- 当前 dcat16 输出被 inject 使用 → 梯度应该流通
- 但 dcat16 的 matchability predictor 输出存入 `data['*_16']` 而 loss 不读取 → **matchability predictor 本身可能有弱梯度**

验证方法：
```python
# 在第一个 step 后检查 dcat16 参数的 grad
for name, param in matcher.dcat16.named_parameters():
    if param.grad is None:
        print(f"NO GRAD: {name}")
```

### 8.4 1/16 matchability 无显式 loss

**这是当前实验设计**，不是 bug。但需要注意：
- 如果 inject 的梯度信号太弱（alpha 接近 0），dcat16 实际上不会被有效监督
- 可能的改进方向：后续加入 1/16 covisibility loss

### 8.5 nearest mask 下采样的边界风险

**存在边界风险**：
- 如果原图 H/W 不是 16 的倍数，nearest 下采样后的 mask 可能偏移
- padding 区域和有效区域的边界可能不对齐

**缓解**：确保数据集图像 resize 到 16 的倍数（或在 dataset 中做 padding 对齐）

### 8.6 USE_DCAT16_INJECT=False 下的额外开销

当 `USE_DCAT16_INJECT=False` 时：
- `backbone.layer4` 仍然存在并参与 forward（虽然输出未被使用）
- `backbone.layer4_outconv` 仍然存在
- 额外 ~5% 参数量（但 layer4 输出在 forward 后无引用，可能被优化掉）

### 8.7 计算量

- dcat16：输入 [B, 256, H/16, W/16]，约 `4×` 面积比 1/8 层更小
- inject：额外 Conv1x1 和 interpolate
- 总体额外计算量约 **15-25%**（估算）

### 8.8 指标归因

如果指标提升，需要确认：
1. 提升来自 1/16 高层语义信息注入，还是
2. 来自 dcat16 提供了额外的 attention 层（实际加深了网络），还是
3. 来自 alpha 可学习的自适应融合

建议消融：
- `USE_DCAT16_INJECT=True` vs `USE_DCAT16_INJECT=False`（baseline）
- alpha 固定为 0（仅 dcat16 + 无 inject）
- 仅用 dcat16 预测 covisibility，不做 inject（只改 loss 信号）

### 8.9 如果指标下降

可能原因：
1. inject 的 bilinear 上采样破坏了 1/8 特征的精度
2. alpha 初始化为 0 导致训练初期 dcat16 未被有效利用
3. nearest mask 下采样引入了错误的 padding 信号
4. dcat16 的 agg_size=2 感受野太小，学不到有用的共视先验

---

## 九、建议的验证实验

### 最小验证清单（必须做）

1. **USE_DCAT16_INJECT=False baseline**
   - 确认原始 CoMatch 训练正常
   - 记录 val AUC@5/10/20、loss 曲线、训练速度

2. **USE_DCAT16_INJECT=True 当前版**
   - 对比 val AUC@5/10/20、loss 曲线、训练速度/显存
   - 记录首个 step 日志中的 shape 确认

3. **梯度检查**（第一个 epoch 第 1 个 step）
   ```python
   # 在 trainer 或 model forward 后
   for name, param in matcher.dcat16.named_parameters():
       assert param.grad is not None, f"No grad: {name}"
   for name, param in matcher.inject16to8.named_parameters():
       assert param.grad is not None, f"No grad: {name}"
   print("All dcat16/inject params have gradients")
   ```

4. **alpha 训练曲线**
   - tensorboard 记录 `inject16to8.alpha`
   - 观察 alpha 在前 10 个 epoch 是否快速上升

5. **dcat16 参数 grad 检查**
   - 检查 `matchability_predictor` 层是否有 grad

### 进阶验证（可选）

6. **alpha 初始化对比**：0 vs 0.01
7. **mask 下采样方式对比**：nearest vs max_pool
8. **agg_size 对比**：agg=2 vs agg=4（更小感受野 vs 与 coarse 相同）
9. **是否加 1/16 covisibility loss**：在 loss 中读取 `matchability_score_list0_16`，加 focal loss
10. **是否减少 1/8 loftr_coarse 层数**：测试 dcat16 能否替代部分 1/8 transformer

---

## 十、报告维护记录

| 日期 | 修改人 | 修改内容 |
|------|--------|---------|
| 2026-04-26 | Claude | 初版：完成代码梳理，输出完整报告 |
