# 修改后的脚本使用指南

## 快速开始

### 1. 验证修改（推荐首先执行）

```bash
# 运行参考实现 test_render.py
python test_render.py --config configs/anchorwarp_4dgs.yaml --index 0

# 检查输出
ls -la gsplat_test_output/
# 应该看到：test_render_out.png
```

### 2. 运行推理脚本

```bash
# 基础推理
python step2_inference_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --checkpoint <path_to_checkpoint> \
    --output_dir results_inference

# 查看结果
ls -la results_inference/rendered_images/
ls -la results_inference/save_gaussians/
```

**常用参数**：
```bash
--config              配置文件路径（必需）
--checkpoint          模型检查点路径（可选）
--hf_dir              HuggingFace 权重目录
--output_dir          输出目录
--batch_size          批次大小
--save_images         保存渲染图像（默认 True）
--save_gaussians      保存高斯参数（默认 True）
```

### 3. 运行训练脚本

```bash
# 基础训练
python step2_train_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --output_dir results_train

# 从检查点恢复
python step2_train_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --resume results_train/latest.pth \
    --output_dir results_train
```

**常用参数**：
```bash
--config              配置文件路径（必需）
--hf_dir              HuggingFace 权重目录
--resume              恢复的检查点路径
--output_dir          输出目录
--batch_size          验证批次大小
--freeze_flow         冻结流模块
--freeze_decoder      冻结解码器模块
--freeze_motion       冻结运动头
```

## 详细说明

### 推理工作流

```
1. 加载配置和数据集
   ↓
2. 初始化模型
   ├─ 从 HuggingFace 加载权重（可选）
   └─ 或从检查点加载
   ↓
3. 对每个批次：
   ├─ 准备批次数据
   ├─ 模型前向传播
   ├─ 逐视角渲染
   ├─ 计算指标
   └─ 保存结果
   ↓
4. 输出统计信息
```

### 训练工作流

```
1. 加载配置和数据集
   ↓
2. 初始化模型和优化器
   ├─ 从 HuggingFace 加载权重（可选）
   └─ 设置冻结参数
   ↓
3. 对每个 epoch：
   ├─ 训练阶段：
   │  ├─ 逐批次训练
   │  ├─ 逐视角渲染
   │  ├─ 计算损失
   │  └─ 反向传播
   │
   ├─ 验证阶段：
   │  ├─ 逐批次验证
   │  ├─ 逐视角渲染
   │  └─ 计算指标
   │
   └─ 保存检查点
   ↓
4. 输出最终统计
```

## 输出说明

### 推理输出

```
results_inference/
├── rendered_images/
│   ├── batch_00000_pred_gt_grid.png
│   ├── batch_00001_pred_gt_grid.png
│   └── ...
├── save_gaussians/
│   ├── time_00000_gaussians.ply
│   ├── time_00001_gaussians.ply
│   └── ...
└── metrics/
    ├── batch_00000.json
    ├── batch_00001.json
    └── ...
```

**文件说明**：
- `batch_*_pred_gt_grid.png`：预测和真实图像的网格
- `time_*_gaussians.ply`：高斯参数（可用 3D 查看器打开）
- `batch_*.json`：指标（SSIM、L1 等）

### 训练输出

```
results_train/YYYYMMDD_HHMMSS/
├── config.yaml                    # 训练配置副本
├── latest.pth                     # 最新检查点
├── best.pth                       # 最佳检查点
├── tensorboard_logs/              # TensorBoard 日志
│   └── events.out.tfevents.*
├── epoch_images/
│   ├── epoch_00000_pred_gt_grid.png
│   ├── epoch_00001_pred_gt_grid.png
│   └── ...
└── epoch_gaussians/
    ├── epoch_00000_time_00000_gaussians.ply
    ├── epoch_00001_time_00000_gaussians.ply
    └── ...
```

**文件说明**：
- `config.yaml`：训练配置（用于复现）
- `latest.pth`：最新模型（用于恢复训练）
- `best.pth`：验证损失最低的模型
- `tensorboard_logs/`：用于 TensorBoard 可视化
- `epoch_images/`：每个 epoch 的渲染结果
- `epoch_gaussians/`：每个 epoch 的高斯参数

## 监控训练

### 使用 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir results_train/YYYYMMDD_HHMMSS/tensorboard_logs/

# 在浏览器中打开
# http://localhost:6006
```

**可视化内容**：
- 训练损失曲线
- 验证损失曲线
- 各项损失分量
- 渲染图像

### 查看日志

```bash
# 实时查看训练输出
tail -f results_train/YYYYMMDD_HHMMSS/tensorboard_logs/events.out.tfevents.*

# 或使用 grep 查找特定信息
grep "Epoch" results_train/YYYYMMDD_HHMMSS/tensorboard_logs/*
```

## 常见问题

### Q1：推理时出现 "shape mismatch" 错误
**A**：检查以下几点：
1. 高斯数量是否过多？
2. 图像分辨率是否正确？
3. 相机参数是否有效？

### Q2：训练时 GPU 内存不足
**A**：尝试以下方案：
1. 减少批次大小
2. 减少高斯数量
3. 减少视角数量
4. 使用混合精度训练

### Q3：渲染结果全黑
**A**：检查以下几点：
1. 颜色值是否在 [0,1]？
2. 不透明度是否过小？
3. 相机矩阵是否正确？
4. 背景颜色设置是否正确？

### Q4：如何修改背景颜色？
**A**：在脚本中找到以下行并修改：
```python
bg_color = torch.ones(3, device=device)  # 白色
# 改为：
bg_color = torch.zeros(3, device=device)  # 黑色
# 或：
bg_color = torch.tensor([0.5, 0.5, 0.5], device=device)  # 灰色
```

### Q5：如何调整相机参数？
**A**：在脚本中找到以下行并修改：
```python
cam = IntrinsicsCamera(
    K=K_np, R=R, T=t_vec,
    width=int(W_t), height=int(H_t),
    znear=0.01,      # 修改近平面
    zfar=100.0,      # 修改远平面
)
```

## 性能优化建议

### 1. 加速推理
```bash
# 减少高斯数量
python step2_inference_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --checkpoint <path> \
    --output_dir results_fast

# 在配置文件中修改：
# model:
#   target_num_gaussians: 2000  # 减少高斯数量
```

### 2. 加速训练
```bash
# 使用混合精度（如果支持）
# 在配置文件中修改：
# model:
#   use_fp16: true

# 增加学习率
# training:
#   learning_rate: 0.001  # 增加学习率
```

### 3. 节省内存
```bash
# 减少批次大小
python step2_train_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --batch_size 2 \
    --output_dir results_memory_efficient
```

## 调试技巧

### 1. 启用详细日志
```python
# 在脚本开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 保存中间结果
```python
# 在渲染后添加
import torch
torch.save(rendered_images, 'debug_rendered.pt')
torch.save(mu_t, 'debug_mu.pt')
```

### 3. 可视化高斯
```bash
# 使用 3D 查看器打开 PLY 文件
# 推荐工具：CloudCompare, MeshLab, Blender
open results_inference/save_gaussians/time_00000_gaussians.ply
```

## 高级用法

### 1. 自定义数据集
```python
# 修改 prepare_batch 函数以支持自定义数据格式
# 参考 step2_train_4DGSFFMotion.py 中的 prepare_batch 函数
```

### 2. 自定义损失函数
```python
# 在配置文件中修改 loss_weights
# training:
#   loss_weights:
#     photo_l1: 1.0
#     ssim: 0.1
#     silhouette: 0.0
#     chamfer: 0.0
```

### 3. 自定义模型架构
```python
# 修改 Trellis4DGS4DCanonical 的初始化参数
# 参考脚本中的 model_config 部分
```

## 故障恢复

### 1. 恢复中断的训练
```bash
# 从最新检查点恢复
python step2_train_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --resume results_train/YYYYMMDD_HHMMSS/latest.pth \
    --output_dir results_train
```

### 2. 使用最佳模型进行推理
```bash
# 使用验证损失最低的模型
python step2_inference_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --checkpoint results_train/YYYYMMDD_HHMMSS/best.pth \
    --output_dir results_inference_best
```

### 3. 清理临时文件
```bash
# 删除旧的训练结果
rm -rf results_train/YYYYMMDD_HHMMSS/tensorboard_logs/
rm -rf results_train/YYYYMMDD_HHMMSS/epoch_images/
```

## 参考资源

- `test_render.py` - 基础渲染示例
- `configs/anchorwarp_4dgs.yaml` - 配置文件示例
- `RENDERING_QUICK_REFERENCE.md` - 快速参考
- `RENDERING_REFACTOR_SUMMARY.md` - 技术细节
- `MODIFICATION_SUMMARY_CN.md` - 中文说明

## 获取帮助

1. 查看脚本中的注释
2. 参考相关文档
3. 检查配置文件
4. 运行 test_render.py 进行对比
5. 查看输出日志和错误信息

---

**最后更新**：2024-12-12
**版本**：1.0



