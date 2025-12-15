# SURFEL 4DGS - 快速参考指南

## 📋 核心改动总结

### 原始流程 (Voxel-based)
```
points_3d → PointDownsampler (Voxel Grid) → mu → PerGaussianAggregator → g → GaussianHead (预测所有参数)
```

### 新流程 (SURFEL-based)
```
points_3d 
  ↓
SurfelExtractor (局部 PCA)
  ↓ μ_j, R_j, s_j, confidence
  ↓
WeightedFPS (30k→5k)
  ↓ 选中的 SURFEL
  ↓
PerGaussianAggregator (加入几何信息)
  ↓ g_j
  ↓
GaussianHead (只预测 c_j, o_j)
  ↓ color, opacity
  ↓
TimeWarpMotionHead
  ↓ per-frame 动态
```

---

## 🔧 关键类和方法

### 1. SurfelExtractor
```python
from models.trellis_4dgs_canonical4d import SurfelExtractor

extractor = SurfelExtractor(
    k_neighbors=16,  # K-近邻数量
    use_confidence_weighting=True,  # 使用置信度
)

surfel_data = extractor(points_3d)  # [T,N,3]
# 返回：
# - mu: [N_surfel, 3]
# - normal: [N_surfel, 3]
# - radius: [N_surfel, 1]
# - confidence: [N_surfel, 1]
```

**关键参数**：
- `k_neighbors`：K-近邻数量，影响 PCA 的稳定性
  - 值越大，法线越平滑但细节丢失
  - 推荐：8-32
- `use_confidence_weighting`：是否计算置信度
  - `confidence = 1 - (λ_min / λ_max)`
  - 用于 FPS 的加权采样

---

### 2. WeightedFPS
```python
from models.trellis_4dgs_canonical4d import WeightedFPS

fps = WeightedFPS()

indices, selected_points = fps.forward(
    points=surfel_mu,  # [M, 3]
    weights=surfel_confidence,  # [M, 1]
    num_samples=5000,  # 目标采样数
)
# 返回：
# - indices: [K] 选中点的索引
# - selected_points: [K, 3] 选中的点
```

**工作原理**：
1. 根据权重随机选择第一个点
2. 迭代 K-1 次：
   - 计算未选点到已选点的最小距离
   - `score = min_distance * weight`
   - 选择得分最高的点

---

### 3. PerGaussianAggregator (改进版)
```python
# 新增参数
g = aggregator(
    mu=mu,  # [M, 3]
    feat_2d=feat_2d,  # [T, V, H', W', C]
    camera_poses=camera_poses,  # [T, V, 4, 4]
    camera_intrinsics=camera_intrinsics,  # [T, V, 3, 3]
    time_ids=time_ids,  # [T]
    surfel_normal=surfel_normal,  # [M, 3] 新增
    surfel_radius=surfel_radius,  # [M, 1] 新增
)
```

**改进**：
- 在采样 2D 特征后，拼接 SURFEL 几何信息
- 法线投影到相机坐标系
- 半径直接拼接

---

### 4. GaussianHead (SURFEL 版本)
```python
# 初始化
head = GaussianHead(
    in_dim=256,
    hidden_dim=256,
    use_scale_refine=False,  # 可选：启用尺度微调
    use_rot_refine=False,    # 可选：启用旋转微调
)

# 前向传播
params = head(
    g=g,  # [M, C]
    surfel_scale=surfel_radius.expand(-1, 3),  # [M, 3]
    surfel_rot=surfel_rot,  # [M, 3, 3]
)

# 输出
{
    'color': [M, 3],      # ✅ Head 预测
    'opacity': [M, 1],    # ✅ Head 预测
    'scale': [M, 3],      # ❌ 来自 SURFEL（或微调）
    'rot': [M, 3, 3],     # ❌ 来自 SURFEL（或微调）
    'scale_delta': [M, 3],  # (可选) 微调量
    'rot_delta': [M, 6],    # (可选) 微调量
}
```

**设计**：
- ✅ Head 只预测 **2 个参数**：颜色、不透明度
- ❌ 几何参数（尺度、旋转）来自 SURFEL
- 🔧 可选微调允许小幅调整

---

### 5. Trellis4DGS4DCanonical (主模型)
```python
from models.trellis_4dgs_canonical4d import Trellis4DGS4DCanonical

model = Trellis4DGS4DCanonical(
    # SURFEL 参数
    surfel_k_neighbors=16,
    use_surfel_confidence=True,
    target_num_gaussians=5000,  # 30k → 5k
    
    # Feature Aggregator
    feat_agg_dim=256,
    feat_agg_layers=2,
    feat_agg_heads=4,
    
    # Gaussian Head
    gaussian_head_hidden=256,
    use_scale_refine=False,
    use_rot_refine=False,
    
    # Motion Head
    motion_dim=128,
).to(device)

# 前向传播
output = model(
    points_3d=points_3d,  # [T, N, 3]
    feat_2d=feat_2d,  # [T, V, H', W', C]
    camera_poses=camera_poses,  # [T, V, 4, 4]
    camera_intrinsics=camera_intrinsics,  # [T, V, 3, 3]
    time_ids=time_ids,  # [T]
)

# 输出
{
    'mu_t': [T, M, 3],        # per-frame 中心
    'scale_t': [T, M, 3],     # per-frame 尺度
    'color_t': [T, M, 3],     # per-frame 颜色
    'alpha_t': [T, M, 1],     # per-frame 不透明度
    'dxyz_t': [T, M, 3],      # 动态偏移
    'world_aabb': [2, 3],     # 世界 AABB
    'surfel_mu': [M, 3],      # canonical SURFEL 中心
    'surfel_normal': [M, 3],  # canonical SURFEL 法线
    'surfel_radius': [M, 1],  # canonical SURFEL 半径
}
```

---

## 📊 参数对比

| 参数 | 原始方法 | SURFEL 方法 | 说明 |
|------|---------|-----------|------|
| voxel_size | 0.02 | - | 已移除 |
| use_kmeans_refine | ✓ | - | 已移除 |
| adaptive_voxel | ✓ | - | 已移除 |
| target_num_gaussians | ✓ | ✓ | 保留，用于 FPS |
| surfel_k_neighbors | - | 16 | 新增 |
| use_surfel_confidence | - | ✓ | 新增 |
| use_scale_refine | - | ✓ | 新增，可选 |
| use_rot_refine | - | ✓ | 新增，可选 |

---

## 🎯 使用建议

### 1. 基础配置（推荐）
```python
model = Trellis4DGS4DCanonical(
    surfel_k_neighbors=16,
    use_surfel_confidence=True,
    target_num_gaussians=5000,
    feat_agg_dim=256,
    gaussian_head_hidden=256,
    use_scale_refine=False,  # 不启用微调
    use_rot_refine=False,
    motion_dim=128,
)
```

### 2. 高精度配置（更多参数）
```python
model = Trellis4DGS4DCanonical(
    surfel_k_neighbors=16,
    use_surfel_confidence=True,
    target_num_gaussians=5000,
    feat_agg_dim=512,  # 更大的特征维度
    feat_agg_layers=3,  # 更多 Transformer 层
    gaussian_head_hidden=512,
    use_scale_refine=True,  # 启用尺度微调
    use_rot_refine=True,    # 启用旋转微调
    motion_dim=256,
)
```

### 3. 快速推理配置（更少参数）
```python
model = Trellis4DGS4DCanonical(
    surfel_k_neighbors=8,  # 更少的邻域
    use_surfel_confidence=True,
    target_num_gaussians=2000,  # 更少的高斯
    feat_agg_dim=128,  # 更小的特征维度
    feat_agg_layers=1,
    gaussian_head_hidden=128,
    use_scale_refine=False,
    use_rot_refine=False,
    motion_dim=64,
)
```

---

## 🔍 调试技巧

### 检查 SURFEL 质量
```python
# 在 forward 后访问缓存
surfel_mu = model._world_cache['surfel_mu']  # [M, 3]
surfel_normal = model._world_cache['surfel_normal']  # [M, 3]
surfel_radius = model._world_cache['surfel_radius']  # [M, 1]
surfel_confidence = model._world_cache['surfel_confidence']  # [M, 1]

# 统计信息
print(f"SURFEL 数量: {surfel_mu.shape[0]}")
print(f"平均半径: {surfel_radius.mean():.4f}")
print(f"平均置信度: {surfel_confidence.mean():.4f}")
```

### 清除缓存
```python
# 如果需要重新计算 SURFEL（例如输入点云变化）
model.reset_world_cache()
```

### 可视化 SURFEL
```python
import numpy as np

# 导出为 PLY 格式
surfel_mu_np = surfel_mu.cpu().numpy()
surfel_normal_np = surfel_normal.cpu().numpy()
surfel_radius_np = surfel_radius.cpu().numpy()

# 可用于 CloudCompare 或其他 3D 可视化工具
```

---

## ⚠️ 常见问题

### Q1: 为什么 SURFEL 数量很多（30k）？
**A**: 这是从所有时间帧的点云提取的。Weighted FPS 会将其下采样到目标数量（默认 5k）。

### Q2: 法线方向有歧义吗？
**A**: 是的，PCA 得到的法线可能有 ±180° 的歧义。当前实现未处理此问题。可在需要时添加一致性检查。

### Q3: 为什么不直接使用 SURFEL 的旋转矩阵？
**A**: SURFEL 只提供法线（1 个自由度），不足以确定完整的旋转矩阵（3 个自由度）。当前实现使用法线作为 Z 轴，自动构造 X、Y 轴。

### Q4: 启用 `use_scale_refine` 会显著增加参数量吗？
**A**: 是的，会增加一个 `[hidden_dim, 3]` 的线性层。对于 `hidden_dim=256`，增加 768 个参数。

### Q5: 如何调整高斯数量？
**A**: 修改 `target_num_gaussians` 参数。例如：
```python
model = Trellis4DGS4DCanonical(target_num_gaussians=10000)  # 10k 高斯
```

---

## 📈 性能指标

| 指标 | 原始方法 | SURFEL 方法 |
|------|---------|-----------|
| Head 参数数量 | 较多 | 较少 |
| 初始化质量 | 随机 | 几何驱动 |
| 推理速度 | 快 | 稍慢（SURFEL 提取） |
| 渲染质量 | 中等 | 更好（几何感知） |
| 可解释性 | 低 | 高 |

---

## 🚀 下一步

1. **训练**：使用新的 SURFEL 模型进行训练
2. **评估**：对比原始方法的性能
3. **优化**：根据需要调整参数
4. **扩展**：考虑更复杂的 SURFEL 表示

---

## 📚 相关文件

- `trellis_4dgs_canonical4d.py`：主模型实现
- `SURFEL_ARCHITECTURE.md`：详细架构文档
- `SURFEL_QUICK_REFERENCE.md`：本文件

---

**最后更新**：2025-12-09


