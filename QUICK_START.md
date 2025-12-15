# FF4DGSMotion 快速开始指南

## 项目概述

FF4DGSMotion 是一个轻量级的 4D 高斯溅射模型。

**核心特点：**
- 无 Trellis 依赖 - 轻量级实现
- 三模块架构 - 易于理解和扩展
- 自适应体素化 - 自动适应场景尺度
- Transformer 特征聚合 - 多视角融合
- 时间动态模型 - 支持动态场景

---

## 安装

### 依赖项

```bash
pip install torch torchvision
pip install pyyaml numpy
pip install plyfile
pip install safetensors
```

---

## 快速使用

### 推理示例

```python
import torch
from FF4DGSMotion.models.trellis_4dgs_canonical4d import Trellis4DGS4DCanonical

# 初始化模型
model = Trellis4DGS4DCanonical(
    voxel_size=0.02,
    feat_agg_dim=256,
    feat_agg_layers=2,
    feat_agg_heads=4,
    motion_dim=128,
).cuda()

# 准备输入
points_3d = torch.randn(6, 10000, 3).cuda()      # [T, N, 3]
feat_2d = torch.randn(6, 4, 27, 36, 256).cuda()  # [T, V, H', W', C]
camera_poses = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(6, 4, 4, 4).cuda()
camera_intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(6, 4, 3, 3).cuda()
time_ids = torch.arange(6).cuda()

# 前向传播
with torch.no_grad():
    output = model(
        points_3d=points_3d,
        feat_2d=feat_2d,
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
        time_ids=time_ids,
    )

# 输出
mu_t = output['mu_t']           # [T, M, 3]
scale_t = output['scale_t']     # [T, M, 3]
color_t = output['color_t']     # [T, M, 3]
alpha_t = output['alpha_t']     # [T, M, 1]
```

### 配置文件

```yaml
model:
  voxel_size: 0.02
  use_kmeans_refine: true
  adaptive_voxel: true
  target_num_gaussians: 5000
  
  feat_agg_dim: 256
  feat_agg_layers: 2
  feat_agg_heads: 4
  time_emb_dim: 32
  
  gaussian_head_hidden: 256
  motion_dim: 128
  aabb_margin: 0.05
```

---

## 常见问题

### 高斯数量太多？

```yaml
model:
  target_num_gaussians: 2000  # 减少
  voxel_size: 0.05            # 增加
```

### 特征聚合不充分？

```yaml
model:
  feat_agg_layers: 4          # 增加层数
  feat_agg_heads: 8           # 增加头数
```

### 时间动态不平滑？

```yaml
model:
  motion_dim: 256
  time_emb_dim: 64
```

---

## 详细文档

- 改进说明：`REFACTORING_IMPROVEMENTS.md`
- 推理脚本：`step2_inference_4DGSFFMotion.py`
- 训练脚本：`step2_train_4DGSFFMotion.py`

---

**最后更新：2025-12-09**
