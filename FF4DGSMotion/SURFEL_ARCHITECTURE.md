# SURFEL-based 4DGS 架构文档

## 概述

本文档描述了重构后的 `Trellis4DGS4DCanonical` 模型，采用 **SURFEL（表面元素）** 的方法来表示 3D 高斯。相比原始方法，新架构具有以下优势：

1. **几何感知**：直接从点云的局部几何（法线、曲率）中提取高斯参数
2. **参数高效**：Head 只需预测颜色和不透明度，几何由 SURFEL 提供
3. **全局一致性**：使用 Weighted FPS 确保高斯分布均匀且有置信度加权
4. **可解释性**：每个高斯对应一个表面元素，具有清晰的几何含义

---

## 核心流程

### 1. SURFEL 提取 (SurfelExtractor)

**输入**：`points_3d [T,N,3]` - 多时间帧的点云

**处理**：
- 时序汇合：将所有时间帧的点合并为 `[T*N, 3]`
- 局部 PCA：对每个点的 K-近邻邻域进行主成分分析
  - 计算协方差矩阵 `Cov = (1/k) * X^T @ X`
  - 特征分解得到特征值 `λ₁ ≥ λ₂ ≥ λ₃` 和特征向量
  
**输出**：
- `μ_j [N_surfel, 3]`：SURFEL 中心（PCA 均值）
- `R_j [N_surfel, 3]`：主法线方向（最小特征值对应的特征向量）
- `s_j [N_surfel, 1]`：局部半径（最大特征值的平方根）
- `confidence [N_surfel, 1]`：置信度 = `1 - (λ_min / λ_max)`
  - 值越接近 1，表面越平坦，SURFEL 越可靠

**关键参数**：
```python
surfel_k_neighbors: int = 16  # K-近邻数量
use_surfel_confidence: bool = True  # 是否使用置信度加权
```

---

### 2. Weighted FPS (WeightedFPS)

**输入**：
- `points [M, 3]`：SURFEL 中心
- `weights [M, 1]`：置信度
- `num_samples: int`：目标采样数 K

**处理**：
- 初始化：根据置信度分布随机选择第一个点
- 迭代选择 K-1 次：
  - 计算每个未选点到已选点的最小距离
  - 综合距离和置信度：`score = min_dist * weight`
  - 选择得分最高的点

**输出**：
- `indices [K]`：选中的点的索引
- `selected_points [K, 3]`：选中的 SURFEL 中心

**效果**：
- 30k SURFEL → 5k 高斯（可配置）
- 高置信度的点优先被选中
- 点之间保持足够的空间距离

**关键参数**：
```python
target_num_gaussians: int = 5000  # 目标高斯数量
```

---

### 3. Multi-view Feature Aggregation (PerGaussianAggregator)

**输入**：
- `μ_j [M, 3]`：选中的 SURFEL 中心
- `feat_2d [T, V, H', W', C]`：多视角 2D 特征图
- `camera_poses [T, V, 4, 4]`：相机位姿
- `camera_intrinsics [T, V, 3, 3]`：相机内参
- `surfel_normal [M, 3]`：SURFEL 法线（可选）
- `surfel_radius [M, 1]`：SURFEL 半径（可选）

**处理**：
1. 对每个 SURFEL 中心 `μ_j`，投影到所有 `(t, v)` 视角
2. 双线性采样对应的 2D 特征
3. 加入 SURFEL 几何信息（法线、半径）
4. 加入时间和视角位置编码
5. 通过 Transformer 聚合所有视角的特征

**输出**：
- `g_j [M, C]`：canonical feature（C=256 by default）

**关键参数**：
```python
feat_agg_dim: int = 256  # 特征维度
feat_agg_layers: int = 2  # Transformer 层数
feat_agg_heads: int = 4  # 注意力头数
time_emb_dim: int = 32  # 时间编码维度
view_emb_dim: int = 32  # 视角编码维度
```

---

### 4. Gaussian Head (GaussianHead) - SURFEL 版本

**输入**：
- `g_j [M, C]`：canonical feature
- `surfel_scale [M, 3]`：SURFEL 半径（扩展为各向异性）
- `surfel_rot [M, 3, 3]`：SURFEL 旋转矩阵（从法线构造）

**处理**：
- MLP backbone：`g_j → h [M, hidden_dim]`
- 必需输出：
  - `c_j = sigmoid(fc_color(h))`：RGB 颜色 [M, 3]
  - `o_j = sigmoid(fc_opac(h))`：不透明度 [M, 1]
- 可选微调（需启用）：
  - `Δs_j = tanh(fc_scale_delta(h)) * 0.2`：尺度微调 [M, 3]
  - `ΔR_j = fc_rot_delta(h) * 0.1`：旋转微调 [M, 6]

**输出**：
```python
{
    'color': [M, 3],      # 颜色
    'opacity': [M, 1],    # 不透明度
    'scale': [M, 3],      # 最终尺度（SURFEL 或微调后）
    'rot': [M, 3, 3],     # 最终旋转（SURFEL 或微调后）
    'scale_delta': [M, 3],  # (可选) 尺度微调
    'rot_delta': [M, 6],    # (可选) 旋转微调
}
```

**关键参数**：
```python
gaussian_head_hidden: int = 256  # MLP 隐层维度
use_scale_refine: bool = False  # 是否启用尺度微调
use_rot_refine: bool = False    # 是否启用旋转微调
```

**设计理念**：
- ✅ Head 只预测**外观**（颜色、不透明度）
- ✅ **几何**（尺度、旋转）直接来自 SURFEL
- ✅ 可选的微调允许模型进行小幅调整
- ✅ 减少参数量，提高训练稳定性

---

### 5. 旋转矩阵构造 (_build_rotation_from_normal)

从 SURFEL 法线 `R_j` 构造旋转矩阵：

```
Z 轴 = normalize(法线)
X 轴 = normalize(ref - (ref·Z)Z)  # 投影到垂直平面
Y 轴 = Z × X  # 叉积
Rot = [X | Y | Z]  # [M, 3, 3]
```

其中 `ref` 的选择：
- 如果 `|Z.z| < 0.9`：使用 `ref = [0, 0, 1]`
- 否则：使用 `ref = [1, 0, 0]`

---

### 6. Motion Head (TimeWarpMotionHead)

**输入**：
- `z_g [M, motion_dim]`：运动特征（从 `g_j` 投影）
- `T`：时间帧数
- `t_ids [T]`：时间 ID
- `xyz [M, 3]`：canonical 中心
- `scale [M, 3]`：canonical 尺度
- `color [M, 3]`：canonical 颜色
- `alpha [M, 1]`：canonical 不透明度

**处理**：
- 时间位置编码：`t_emb [T, time_emb_dim]`
- 融合：`z_tm = gate(t_emb) ⊗ z_m`
- 预测偏移：`Δxyz, Δlog_s, Δc, Δσ`
- 应用到 canonical 参数

**输出**：
```python
xyz_t [T, M, 3]      # per-frame 中心
scale_t [T, M, 3]    # per-frame 尺度
color_t [T, M, 3]    # per-frame 颜色
alpha_t [T, M, 1]    # per-frame 不透明度
dxyz_t [T, M, 3]     # 动态偏移
```

---

## 完整数据流

```
points_3d [T,N,3]
    ↓
[SurfelExtractor]
    ↓
μ_j [N_surfel,3], R_j [N_surfel,3], s_j [N_surfel,1], conf [N_surfel,1]
    ↓
[WeightedFPS]  (30k → 5k)
    ↓
μ_j [M,3], R_j [M,3], s_j [M,1]
    ↓
[PerGaussianAggregator]
    ↓ (with feat_2d, camera_poses, camera_intrinsics)
    ↓
g_j [M,C]
    ↓
[GaussianHead]
    ↓ (with surfel_scale, surfel_rot)
    ↓
c_j [M,3], o_j [M,1], (Δs_j, ΔR_j)
    ↓
[TimeWarpMotionHead]
    ↓ (with z_g, time_ids)
    ↓
xyz_t [T,M,3], scale_t [T,M,3], color_t [T,M,3], alpha_t [T,M,1]
```

---

## 配置示例

```yaml
model:
  # SURFEL 参数
  surfel_k_neighbors: 16
  use_surfel_confidence: true
  target_num_gaussians: 5000
  
  # Feature Aggregator 参数
  feat_agg_dim: 256
  feat_agg_layers: 2
  feat_agg_heads: 4
  time_emb_dim: 32
  view_emb_dim: 32
  
  # Gaussian Head 参数
  gaussian_head_hidden: 256
  use_scale_refine: false  # 可选启用
  use_rot_refine: false    # 可选启用
  
  # Motion Head 参数
  motion_dim: 128
  
  # World space 参数
  aabb_margin: 0.05
```

---

## 缓存机制

模型使用 `_world_cache` 缓存以下数据，避免重复计算：

```python
{
    'aabb': [2, 3],              # 世界 AABB
    'surfel_mu': [N_surfel, 3],  # SURFEL 中心
    'surfel_normal': [N_surfel, 3],  # SURFEL 法线
    'surfel_radius': [N_surfel, 1],  # SURFEL 半径
    'surfel_confidence': [N_surfel, 1],  # 置信度
    'selected_indices': [M],     # FPS 选中的索引
}
```

调用 `reset_world_cache()` 清除缓存。

---

## 优势总结

| 方面 | 原始方法 | SURFEL 方法 |
|------|---------|-----------|
| 几何表示 | 纯学习 | 几何感知 |
| Head 输出 | 6 个参数 | 2 个参数 |
| 参数量 | 较多 | 较少 |
| 可解释性 | 低 | 高 |
| 初始化 | 随机 | 几何驱动 |
| 微调灵活性 | 完全 | 可选 |

---

## 使用示例

```python
import torch
from models.trellis_4dgs_canonical4d import Trellis4DGS4DCanonical

device = torch.device('cuda')
model = Trellis4DGS4DCanonical(
    surfel_k_neighbors=16,
    use_surfel_confidence=True,
    target_num_gaussians=5000,
    feat_agg_dim=256,
    use_scale_refine=False,
    use_rot_refine=False,
).to(device)

# 前向传播
output = model(
    points_3d=points_3d,  # [T, N, 3]
    feat_2d=feat_2d,      # [T, V, H', W', C]
    camera_poses=camera_poses,  # [T, V, 4, 4]
    camera_intrinsics=camera_intrinsics,  # [T, V, 3, 3]
    time_ids=time_ids,    # [T]
)

# 输出包含：
# - mu_t: [T, M, 3] per-frame 高斯中心
# - scale_t: [T, M, 3] per-frame 尺度
# - color_t: [T, M, 3] per-frame 颜色
# - alpha_t: [T, M, 1] per-frame 不透明度
# - surfel_mu: [M, 3] canonical SURFEL 中心
# - surfel_normal: [M, 3] canonical SURFEL 法线
# - surfel_radius: [M, 1] canonical SURFEL 半径
```

---

## 注意事项

1. **SURFEL 提取的计算成本**：K-近邻搜索和 PCA 在大点云上可能较慢
   - 考虑在 CPU 上预计算或使用 FAISS 加速
   
2. **Weighted FPS 的贪心性**：贪心算法不保证全局最优
   - 对于关键应用，考虑使用更高级的采样方法

3. **法线方向的歧义**：PCA 得到的法线可能有 ±180° 的歧义
   - 当前实现未处理此问题，可在需要时添加一致性检查

4. **微调参数的影响**：启用 `use_scale_refine` 或 `use_rot_refine` 会增加参数量
   - 建议在基础模型收敛后再考虑启用

---

## 扩展方向

1. **更复杂的 SURFEL 表示**：使用椭球体而非球体
2. **动态 SURFEL**：允许 SURFEL 参数随时间变化
3. **层级 SURFEL**：多尺度表示
4. **约束优化**：在训练中加入几何一致性约束


