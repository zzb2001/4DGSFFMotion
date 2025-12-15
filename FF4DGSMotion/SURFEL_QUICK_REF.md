# SURFEL 4DGS - 快速参考

## 核心改动

### 原始流程
```
points_3d → PointDownsampler → mu → PerGaussianAggregator → g → GaussianHead
```

### 新流程 (SURFEL)
```
points_3d → SurfelExtractor → μ_j, R_j, s_j, confidence
         → WeightedFPS (30k→5k)
         → PerGaussianAggregator (加入几何)
         → GaussianHead (只预测 c_j, o_j)
         → TimeWarpMotionHead
```

---

## 关键类

### SurfelExtractor
- 输入：`points_3d [T,N,3]`
- 输出：`mu, normal, radius, confidence`
- 方法：K-近邻 + 局部 PCA

### WeightedFPS
- 输入：`points [M,3], weights [M,1], num_samples`
- 输出：`indices [K], selected_points [K,3]`
- 方法：加权最远点采样

### GaussianHead (改进版)
- 输入：`g [M,C], surfel_scale, surfel_rot`
- 输出：`color, opacity, (scale, rot)`
- **只预测颜色和不透明度**

---

## 参数变化

| 参数 | 旧 | 新 | 说明 |
|------|----|----|------|
| voxel_size | ✓ | ✗ | 移除 |
| use_kmeans_refine | ✓ | ✗ | 移除 |
| target_num_gaussians | ✓ | ✓ | 保留 |
| surfel_k_neighbors | ✗ | ✓ | 新增 |
| use_surfel_confidence | ✗ | ✓ | 新增 |
| use_scale_refine | ✗ | ✓ | 新增（可选） |
| use_rot_refine | ✗ | ✓ | 新增（可选） |

---

## 使用示例

```python
from models.trellis_4dgs_canonical4d import Trellis4DGS4DCanonical

model = Trellis4DGS4DCanonical(
    surfel_k_neighbors=16,
    use_surfel_confidence=True,
    target_num_gaussians=5000,
    feat_agg_dim=256,
    use_scale_refine=False,
    use_rot_refine=False,
).to(device)

output = model(
    points_3d=points_3d,
    feat_2d=feat_2d,
    camera_poses=camera_poses,
    camera_intrinsics=camera_intrinsics,
    time_ids=time_ids,
)
```

---

## 输出

```python
{
    'mu_t': [T, M, 3],        # per-frame 中心
    'scale_t': [T, M, 3],     # per-frame 尺度
    'color_t': [T, M, 3],     # per-frame 颜色
    'alpha_t': [T, M, 1],     # per-frame 不透明度
    'dxyz_t': [T, M, 3],      # 动态偏移
    'surfel_mu': [M, 3],      # canonical SURFEL 中心
    'surfel_normal': [M, 3],  # canonical SURFEL 法线
    'surfel_radius': [M, 1],  # canonical SURFEL 半径
}
```

---

## 配置建议

### 基础（推荐）
```python
surfel_k_neighbors=16
use_surfel_confidence=True
target_num_gaussians=5000
feat_agg_dim=256
use_scale_refine=False
use_rot_refine=False
```

### 高精度
```python
surfel_k_neighbors=16
use_surfel_confidence=True
target_num_gaussians=5000
feat_agg_dim=512
feat_agg_layers=3
use_scale_refine=True
use_rot_refine=True
```

### 快速推理
```python
surfel_k_neighbors=8
use_surfel_confidence=True
target_num_gaussians=2000
feat_agg_dim=128
feat_agg_layers=1
use_scale_refine=False
use_rot_refine=False
```

---

## 调试

```python
# 访问 SURFEL 缓存
surfel_mu = model._world_cache['surfel_mu']
surfel_normal = model._world_cache['surfel_normal']
surfel_radius = model._world_cache['surfel_radius']
surfel_confidence = model._world_cache['surfel_confidence']

# 清除缓存
model.reset_world_cache()
```

---

## 优势

✅ 几何感知的初始化
✅ 减少 Head 参数
✅ 更好的可解释性
✅ 可选的微调灵活性


