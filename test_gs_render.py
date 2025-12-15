#!/usr/bin/env python3
"""
基于 test_render.py 的思路重构：
- 使用数据集中"原始点云"的相机参数（c2w 与 K），不再手写 NDC 投影矩阵
- 世界→相机→像素的直观管线交给 IntrinsicsCamera 完成（内部会生成兼容渲染器的投影矩阵）
- 用原始点云初始化高斯（不做跨视角平均），然后用每个视角各自的 c2w 与 K 渲染
- 流程：render_gs -> fast_forward -> render_gs
- 支持加载 GT 图像作为 target_image 传递给 render_gs

用法：
  python test_gs_render.py --config configs/anchorwarp_4dgs.yaml --index 0
"""
import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model.AnchorWarp4DGS.data.dataset import VoxelFF4DGSDataset
from model.AnchorWarp4DGS.models.simple_gaussian import SimpleGaussianModel
from model.AnchorWarp4DGS.models._utils import Struct
from model.AnchorWarp4DGS.camera.camera import IntrinsicsCamera
from model.AnchorWarp4DGS.diff_renderer.gaussian import render_gs


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def custom_collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def subsample_points(mu: torch.Tensor, max_gs: int) -> torch.Tensor:
    M = mu.shape[0]
    if max_gs is None or M <= max_gs:
        return torch.arange(M, device=mu.device)
    idx = torch.randperm(M, device=mu.device)[:max_gs]
    return idx


@torch.no_grad()
def compute_knn_scale(mu: torch.Tensor, anchors: int = 5000, alpha: float = 0.5) -> torch.Tensor:
    """在世界坐标中用近邻间距估计一个基础尺度。"""
    M = mu.shape[0]
    S = min(anchors, M)
    idx = torch.randperm(M, device=mu.device)[:S]
    anc = mu[idx]
    dmins = []
    step = 32768
    for i in range(0, M, step):
        d = torch.cdist(mu[i:i+step], anc)
        dmins.append(d.min(dim=1).values)
    dmin = torch.cat(dmins, dim=0)
    scale = alpha * dmin.clamp_min(1e-6)
    return scale


@torch.no_grad()
def adjust_scale_to_pixels(mu: torch.Tensor, scale_world: torch.Tensor, c2w: torch.Tensor, K: torch.Tensor,
                           r_target: float = 1.0, r_min: float = 0.5, r_max: float = 3.0) -> torch.Tensor:
    """把世界尺度近似换算到像素半径，保持每个高斯屏幕半径接近 r_target 像素。"""
    # 世界点 -> 相机系
    w2c = torch.inverse(c2w)
    ones = torch.ones(mu.shape[0], 1, device=mu.device, dtype=mu.dtype)
    Xw_h = torch.cat([mu, ones], dim=1)  # [M,4]
    Xc = (w2c @ Xw_h.t()).t()[:, :3]
    z = Xc[:, 2].clamp_min(1e-6)
    fx = K[0, 0]
    r_px = (fx * scale_world) / z
    r_des = torch.full_like(r_px, r_target).clamp(min=r_min, max=r_max)
    fac = (r_des / r_px.clamp_min(1e-6)).clamp(0.25, 4.0)
    return (scale_world * fac).clamp_min(1e-4)


def build_intrinsics_from_config(cfg: dict, H_t: int, W_t: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    data_cfg = cfg.get('data', {})
    K_cfg = data_cfg.get('K_orig', None)
    H0, W0 = data_cfg.get('orig_size', (H_t, W_t))
    sx = float(W_t) / float(W0)
    sy = float(H_t) / float(H0)
    if K_cfg is not None:
        K_orig = torch.tensor(K_cfg, device=device, dtype=dtype)
        S = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], device=device, dtype=dtype)
        K_new = (S @ K_orig)
    else:
        f_orig = float(data_cfg.get('f_orig', max(H0, W0) * 1.2))
        cx0, cy0 = W0 / 2.0, H0 / 2.0
        fx = f_orig * sx
        fy = f_orig * sy
        cx_new = cx0 * sx
        cy_new = cy0 * sy
        K_new = torch.tensor([[fx, 0.0, cx_new], [0.0, fy, cy_new], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    return K_new


def load_gt_images(batch, data_cfg, H_t, W_t, V, H, W, device):
    """
    尝试从多个来源加载 GT 图像：
    1. point_cloud_ply 中的 colors
    2. image_dir 中的图像文件
    返回 [V,3,H,W] 的 tensor，或 None
    """
    gt_images = None
    imgs_dir=data_cfg['ex4dgs_dir']
    try:
        from PIL import Image
        camera_names = ['cam00', 'cam06', 'cam11', 'cam16']
        time_idx = batch.get('time_idx', None)
        if time_idx is not None:
            time_idx = time_idx[0].item() if isinstance(time_idx, torch.Tensor) else time_idx
            time_str = f"{time_idx:06d}"
            gt_images_list = []
            all_found = True
            for cam_name in camera_names:
                img_path = os.path.join(imgs_dir, cam_name, f"{time_str}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((W_t, H_t), Image.Resampling.LANCZOS)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    gt_images_list.append(img_array)
                else:
                    all_found = False
                    break
            
            if all_found and len(gt_images_list) == V:
                gt_images = torch.from_numpy(np.stack(gt_images_list, axis=0)).float().to(device)
                # 转换为 CHW 格式：[V,3,H,W]
                gt_images = gt_images.permute(0, 3, 1, 2).contiguous()
                print(f"  ✓ Loaded GT images from image_dir: shape={tuple(gt_images.shape)}")
                return gt_images
    except Exception as e:
        print(f"  Warning: Failed to load GT images from image_dir: {e}")
    
    print(f"  ⚠ No GT images loaded (target_image will be None)")
    return None


def main():
    parser = argparse.ArgumentParser(description='使用原始点云初始化高斯并按原始相机参数渲染（render_gs -> fast_forward -> render_gs）')
    parser.add_argument('--config', type=str, default='configs/anchorwarp_4dgs.yaml', help='Path to config yaml')
    parser.add_argument('--index', type=int, default=0, help='Dataset sample index to fetch')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_gs', type=int, default=50000, help='Max gaussians to render')
    parser.add_argument('--scale_mult', type=float, default=1.0, help='Scale multiplier (pixels)')
    parser.add_argument('--out_dir', type=str, default='gsplat_test_output', help='Output directory')
    parser.add_argument('--near', type=float, default=0.01)
    parser.add_argument('--far', type=float, default=100.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    # Dataset
    data_cfg = cfg.get('data', {})
    dataset = VoxelFF4DGSDataset(
        pi3_results_dir=data_cfg['pi3_results_dir'],
        seganymo_dir=data_cfg['seganymo_dir'],
        reloc3r_dir=data_cfg['reloc3r_dir'],
        image_size=tuple(data_cfg['image_size']),
        image_dir=data_cfg.get('image_dir', None),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )

    # Fetch the requested batch
    batch = None
    for i, b in enumerate(loader):
        if i == args.index:
            batch = b
            break
    if batch is None:
        raise IndexError(f"Index {args.index} out of range for dataset of size {len(dataset)}")

    H_t, W_t = tuple(data_cfg['image_size'])

    # Get tensors (原始世界坐标的点云 + 原始 c2w)
    points = batch.get('points', None)          # [B,V,H,W,3] 世界坐标（与 camera_poses 一致）
    c2w_all = batch.get('camera_poses', None)   # [B,V,4,4]  c2w
    K_all = batch.get('camera_intrinsics', None)  # [B,V,3,3] (可能不存在)

    if points is None or c2w_all is None:
        raise RuntimeError("Dataset batch must contain 'points' and 'camera_poses'.")

    # 仅取 batch 内第 1 个样本
    pts_b = points[0].to(device)    # [V,H,W,3]
    c2w_b = c2w_all[0].to(device)   # [V,4,4]

    if K_all is not None:
        K_b = K_all[0].to(device)   # [V,3,3]
    else:
        # 用 config 的原始内参缩放到目标分辨率
        K_one = build_intrinsics_from_config(cfg, H_t, W_t, c2w_b.dtype, device)
        V = c2w_b.shape[0]
        K_b = K_one.view(1, 3, 3).expand(V, 3, 3).contiguous()

    V = c2w_b.shape[0]

    # 用"原始点云"初始化高斯：直接把各视角的点云合并（不做跨视角平均）
    Vv, H, W, _ = pts_b.shape
    mu = pts_b.reshape(-1, 3)  # [V*H*W,3]
    valid = torch.isfinite(mu).all(dim=-1)
    mu = mu[valid]
    if mu.numel() == 0:
        raise RuntimeError("No valid points to render.")
    sel = subsample_points(mu, args.max_gs)
    mu = mu[sel]
    M = mu.shape[0]

    # 颜色与尺度（简单：均匀灰色；尺度根据近邻并映射到像素半径）
    color = torch.ones(M, 3, device=device) * 0.5
    scale_world = compute_knn_scale(mu, anchors=min(10000, M), alpha=0.5)
    ref_v = 0
    scale_pix = adjust_scale_to_pixels(mu, scale_world, c2w_b[ref_v], K_b[ref_v], r_target=1.0, r_min=0.5, r_max=3.0)
    scale_pix = (scale_pix * float(args.scale_mult)).clamp_min(1e-4)

    # 初始化简单高斯模型
    model_config = Struct(
        init_opacity=0.8,
        init_scaling=0.008,
        num_basis_in=1,
        num_basis_blend=1,
        use_weight_proj=True,
        use_blend=False
    )
    gs_model = SimpleGaussianModel(model_config)
    gs_model.initialize_from_points(mu, colors=color, scales=scale_pix)

    bg_color = torch.ones(3, device=device)  # 每视角都用白背景

    ensure_dir(args.out_dir)

    # 加载 GT 图像
    print(f"\nLoading GT images...")
    gt_images = load_gt_images(batch, data_cfg, H_t, W_t, V, H, W, device)

    # ============ Step 1: First render (per-view) ============
    print(f"\n[Step 1] First render (per-view)...")
    imgs_1 = []
    for vi in range(V):
        # 从 c2w 得到 w2c = [R|t]
        c2w = c2w_b[vi].detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].astype(np.float32)
        t = w2c[:3, 3].astype(np.float32)

        K_np = K_b[vi].detach().cpu().numpy().astype(np.float32)

        cam = IntrinsicsCamera(
            K=K_np,
            R=R,
            T=t,
            width=int(W_t),
            height=int(H_t),
            znear=float(args.near),
            zfar=float(args.far),
        )

        # 获取对应视角的 target_image
        target_image_vi = None
        if gt_images is not None:
            target_image_vi = gt_images[vi:vi+1]  # [1,3,H,W]

        # 渲染单视角（SimpleGaussianModel 输出的属性是单 batch）
        gs_attrs = gs_model.get_gaussian_attributes()
        res_v = render_gs(
            camera=cam,
            bg_color=bg_color,
            gs=gs_attrs,
            target_image=target_image_vi,
            sh_degree=0,
            scaling_modifier=1.0,
        )
        img_v = res_v["color"]  # [3,H,W]
        imgs_1.append(img_v)
        print(f"  View {vi}: rendered (target_image={'provided' if target_image_vi is not None else 'None'})")

    out_1 = torch.stack(imgs_1, dim=0)  # [V,3,H,W]
    save_image(out_1, os.path.join(args.out_dir, f"simple_gs_render1.png"))
    print(f"  Saved: {os.path.join(args.out_dir, 'simple_gs_render1.png')}")

    # ============ Step 2: fast_forward ============
    print(f"\n[Step 2] Calling fast_forward...")
    # 从每个视角的渲染结果中提取 est_color 和 est_weight
    est_color_list = []
    est_weight_list = []
    
    for vi in range(V):
        c2w = c2w_b[vi].detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].astype(np.float32)
        t = w2c[:3, 3].astype(np.float32)
        K_np = K_b[vi].detach().cpu().numpy().astype(np.float32)

        cam = IntrinsicsCamera(
            K=K_np,
            R=R,
            T=t,
            width=int(W_t),
            height=int(H_t),
            znear=float(args.near),
            zfar=float(args.far),
        )

        # 获取对应视角的 target_image
        target_image_vi = None
        if gt_images is not None:
            target_image_vi = gt_images[vi:vi+1]  # [1,3,H,W]

        gs_attrs = gs_model.get_gaussian_attributes()
        res_v = render_gs(
            camera=cam,
            bg_color=bg_color,
            gs=gs_attrs,
            target_image=target_image_vi,
            sh_degree=0,
            scaling_modifier=1.0,
        )
        
        # 提取 est_color 和 est_weight（如果存在）
        if "est_color" in res_v:
            est_color_list.append(res_v["est_color"])
        if "est_weight" in res_v:
            est_weight_list.append(res_v["est_weight"])

    # 如果有 est_color 和 est_weight，进行 fast_forward
    if est_color_list and est_weight_list:
        est_color = torch.stack(est_color_list, dim=0)  # [V, N, 3] or similar
        est_weight = torch.stack(est_weight_list, dim=0)  # [V, N] or similar
        
        # 对所有视角的结果进行平均或聚合
        est_color_avg = est_color.mean(dim=0)  # [N, 3]
        est_weight_avg = est_weight.mean(dim=0)  # [N]
        
        gs_model.fast_forward(est_color_avg.unsqueeze(0), est_weight_avg.unsqueeze(0))
        print(f"  fast_forward completed")
    else:
        print(f"  Warning: est_color or est_weight not found in render_gs output, skipping fast_forward")

    # ============ Step 3: Second render (per-view) ============
    print(f"\n[Step 3] Second render after fast_forward (per-view)...")
    imgs_2 = []
    for vi in range(V):
        c2w = c2w_b[vi].detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].astype(np.float32)
        t = w2c[:3, 3].astype(np.float32)

        K_np = K_b[vi].detach().cpu().numpy().astype(np.float32)

        cam = IntrinsicsCamera(
            K=K_np,
            R=R,
            T=t,
            width=int(W_t),
            height=int(H_t),
            znear=float(args.near),
            zfar=float(args.far),
        )

        # 获取对应视角的 target_image
        target_image_vi = None
        if gt_images is not None:
            target_image_vi = gt_images[vi:vi+1]  # [1,3,H,W]

        # 渲染单视角（SimpleGaussianModel 输出的属性是单 batch）
        gs_attrs = gs_model.get_gaussian_attributes()
        res_v = render_gs(
            camera=cam,
            bg_color=bg_color,
            gs=gs_attrs,
            target_image=target_image_vi,
            sh_degree=0,
            scaling_modifier=1.0,
        )
        img_v = res_v["color"]  # [3,H,W]
        imgs_2.append(img_v)
        print(f"  View {vi}: rendered (target_image={'provided' if target_image_vi is not None else 'None'})")

    out_2 = torch.stack(imgs_2, dim=0)  # [V,3,H,W]
    save_image(out_2, os.path.join(args.out_dir, f"simple_gs_render2.png"))
    print(f"  Saved: {os.path.join(args.out_dir, 'simple_gs_render2.png')}")

    # ============ Step 4: Compare renders ============
    print(f"\n[Step 4] Comparing renders...")
    diff = (out_1 - out_2).abs().mean().item()
    print(f"  Mean absolute difference between render1 and render2: {diff:.6f}")
    
    for v in range(V):
        diff_v = (out_1[v] - out_2[v]).abs().mean().item()
        print(f"  View {v}: diff={diff_v:.6f}")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
