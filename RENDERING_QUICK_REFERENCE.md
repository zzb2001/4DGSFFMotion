# 渲染流程快速参考

## 修改概览

### 两个文件的修改内容
| 文件 | 修改内容 | 影响范围 |
|------|--------|--------|
| `step2_inference_4DGSFFMotion.py` | 推理时的渲染管线 | inference() 函数 |
| `step2_train_4DGSFFMotion.py` | 训练和验证时的渲染管线 | train_epoch() 和 validate() 函数 |

## 核心改动对比

### 旧流程（render_one_frame_simple_gs）
```
输入：mu, scale, color, alpha, camera_poses, camera_intrinsics, H, W, gt_images
  ↓
单一函数调用（内部处理所有视角）
  ↓
输出：render_pack["color"] [V,H,W,3]
```

### 新流程（IntrinsicsCamera + render_gs）
```
输入：mu, scale, color, alpha, camera_poses, camera_intrinsics, H, W
  ↓
逐视角循环：
  ├─ c2w → w2c (矩阵求逆)
  ├─ 创建 IntrinsicsCamera(K, R, T)
  ├─ 构建 gs_attrs 字典
  └─ render_gs() → [3,H,W]
  ↓
堆叠视角 [V,3,H,W] → 转置 [V,H,W,3]
  ↓
输出：[V,H,W,3]
```

## 关键代码片段

### 1. 导入（两个文件都需要）
```python
from FF4DGSMotion.camera.camera import IntrinsicsCamera
from FF4DGSMotion.diff_renderer.gaussian import render_gs
```

### 2. 相机矩阵转换
```python
# 输入：c2w [4,4]
c2w = camera_poses_t[vi].detach().cpu().numpy()
w2c = np.linalg.inv(c2w)
R = w2c[:3, :3].astype(np.float32)
t_vec = w2c[:3, 3].astype(np.float32)
```

### 3. 创建相机对象
```python
cam = IntrinsicsCamera(
    K=K_np,              # [3,3] numpy array
    R=R,                 # [3,3] numpy array
    T=t_vec,             # [3] numpy array
    width=int(W_t),      # 图像宽度
    height=int(H_t),     # 图像高度
    znear=0.01,          # 近平面
    zfar=100.0,          # 远平面
)
```

### 4. 高斯属性字典
```python
gs_attrs = {
    'mu': mu_frame,                                           # [M,3]
    'scale': scale_frame,                                     # [M,3]
    'color': color_frame,                                     # [M,3]
    'opacity': alpha_frame.squeeze(-1) if alpha_frame.dim() > 1 else alpha_frame,  # [M]
}
```

### 5. 渲染调用
```python
res_v = render_gs(
    camera=cam,
    bg_color=bg_color,           # [3] torch tensor
    gs=gs_attrs,
    target_image=None,           # 不需要 GT 图像
    sh_degree=0,                 # 球谐度数
    scaling_modifier=1.0,        # 尺度修饰符
)
img_v = res_v["color"]  # [3,H,W]
```

### 6. 视角堆叠
```python
# 收集所有视角的渲染结果
imgs_t = []  # 每个元素是 [3,H,W]
for vi in range(V):
    # ... 渲染代码 ...
    imgs_t.append(res_v["color"])

# 堆叠为 [V,3,H,W]
imgs_t_stacked = torch.stack(imgs_t, dim=0)

# 转换为 [V,H,W,3]（用于后续损失计算）
imgs_t_hwc = imgs_t_stacked.permute(0, 2, 3, 1).contiguous()
```

## 数据格式检查清单

- [ ] `camera_poses_t` 是 c2w 矩阵 [V,4,4]
- [ ] `camera_intrinsics_t` 是 [V,3,3] 内参矩阵
- [ ] `mu_frame` 是 [M,3] 世界坐标
- [ ] `scale_frame` 是 [M,3] 或 [M]
- [ ] `color_frame` 是 [M,3] 且值在 [0,1]
- [ ] `alpha_frame` 是 [M] 或 [M,1]
- [ ] `bg_color` 是 [3] torch tensor
- [ ] K 矩阵转换为 numpy float32
- [ ] 所有张量在同一设备上

## 常见问题排查

### 问题 1：形状不匹配错误
**症状**：`RuntimeError: shape mismatch`
**检查**：
- alpha_frame 是否需要 squeeze？
- 输出是否正确转换为 [V,H,W,3]？

### 问题 2：设备不匹配
**症状**：`RuntimeError: expected all tensors to be on the same device`
**检查**：
- bg_color 是否在正确的设备上？
- 高斯属性是否都在 GPU 上？

### 问题 3：渲染结果全黑
**症状**：输出图像全为 0
**检查**：
- color_frame 值是否在 [0,1]？
- opacity 是否过小？
- 相机矩阵是否正确？

### 问题 4：内存溢出
**症状**：`CUDA out of memory`
**解决**：
- 减少高斯数量（M）
- 减少视角数量（V）
- 考虑批量渲染优化

## 性能优化建议

1. **批量渲染**：可考虑一次性渲染多个视角（如果内存允许）
2. **缓存相机**：如果相机参数不变，可缓存 IntrinsicsCamera 对象
3. **异步处理**：可在 CPU 上进行矩阵转换，GPU 上进行渲染
4. **混合精度**：考虑使用 float16 加速（需验证数值稳定性）

## 验证步骤

### 1. 单帧测试
```python
# 运行 test_render.py 验证基础功能
python test_render.py --config configs/anchorwarp_4dgs.yaml --index 0
```

### 2. 推理测试
```python
# 运行修改后的推理脚本
python step2_inference_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --checkpoint <model_path> \
    --output_dir results_test
```

### 3. 训练测试
```python
# 运行修改后的训练脚本
python step2_train_4DGSFFMotion.py \
    --config configs/anchorwarp_4dgs.yaml \
    --output_dir results_test
```

### 4. 输出验证
- 检查 `results_test/rendered_images/` 中的图像
- 验证图像质量和颜色
- 检查损失值的合理性
- 对比 test_render.py 的输出

## 回滚方案

如果需要回滚到原始版本：
```bash
git checkout HEAD -- step2_inference_4DGSFFMotion.py step2_train_4DGSFFMotion.py
```

## 相关文件

- `test_render.py`：参考实现（已验证可用）
- `FF4DGSMotion/camera/camera.py`：IntrinsicsCamera 类定义
- `FF4DGSMotion/diff_renderer/gaussian.py`：render_gs 函数定义
- `RENDERING_REFACTOR_SUMMARY.md`：详细改动说明



