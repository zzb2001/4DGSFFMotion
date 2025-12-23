"""
Inference script for Trellis Minimal 4DGS
SLat Flow (image-conditioned) -> SLat Gaussian Decoder -> Sim(3) Canonicalizer -> gsplat renderer
- 使用多视图 feat_2d 作为条件，无需显式 canonical 构建
- 推理端完全前馈，按帧生成高斯并渲染/保存
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import re
from tqdm import tqdm
import json
import yaml
import numpy as np
import time
from PIL import Image
from torchvision.utils import save_image
try:
    from plyfile import PlyData, PlyElement
except Exception:  # pragma: no cover
    PlyData = None
    PlyElement = None
from safetensors.torch import load_file
from typing import Optional, Tuple, Dict
import math
import torch.nn.functional as F
from FF4DGSMotion.data.dataset import VoxelFF4DGSDataset
from FF4DGSMotion.models.FF4DGSMotion import Trellis4DGS4DCanonical
from FF4DGSMotion.camera.camera import IntrinsicsCamera
from FF4DGSMotion.diff_renderer.gaussian import render_gs, GaussianAttributes
from FF4DGSMotion.models._utils import rgb2sh0, matrix_to_quaternion


def _gaussian(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device, dtype=dtype)
    return gauss/gauss.sum()

def _create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype):
    _1D_window = _gaussian(window_size, 1.5, device, dtype).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, window: torch.Tensor = None, size_average: bool = True):
    # img: [N,C,H,W] in [0,1]
    device = img1.device
    dtype = img1.dtype
    channel = img1.size(1)
    if window is None:
        window = _create_window(window_size, channel, device, dtype)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-8)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1,2,3])

def ssim_ms_torch(img1: torch.Tensor, img2: torch.Tensor, scales: int = 3, window_size: int = 11):
    # Multi-scale SSIM via average pooling pyramid
    assert scales >= 1
    ssim_vals = []
    x, y = img1, img2
    for s in range(scales):
        ssim_vals.append(ssim_torch(x, y, window_size=window_size))
        if s < scales - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
    return torch.stack(ssim_vals).mean()

def crop_border_tensor(img: torch.Tensor, b: int) -> torch.Tensor:
    if b <= 0:
        return img
    return img[..., b:-b, b:-b] if img.shape[-1] > 2*b and img.shape[-2] > 2*b else img


def project_points_torch(Xw: torch.Tensor, c2w: torch.Tensor, K: torch.Tensor):
    # Xw: [M,3], c2w: [4,4] (c2w), K: [3,3]
    device = Xw.device
    dtype = Xw.dtype
    ones = torch.ones(Xw.shape[0], 1, device=device, dtype=dtype)
    Xw_h = torch.cat([Xw, ones], dim=1)  # [M,4]
    w2c = torch.inverse(c2w)
    Xc = (w2c @ Xw_h.t()).t()[:, :3]
    z = Xc[:, 2]
    uvw = (K @ Xc.t()).t()
    u = uvw[:, 0] / uvw[:, 2].clamp(min=1e-6)
    v = uvw[:, 1] / uvw[:, 2].clamp(min=1e-6)
    return u, v, z


def gaussian_color_init(mu: torch.Tensor,
                        scale: torch.Tensor,
                        images_tv: torch.Tensor,   # [V,H,W,3] in [0,1]
                        c2w_tv: torch.Tensor,      # [V,4,4]
                        K_tv: torch.Tensor,        # [V,3,3]
                        win: int = 5,
                        sigma_scale: float = 1.5) -> torch.Tensor:
    """
    基于多视角图像用局部高斯核做加权平均，近似 c_init = sum_ij w_ij I_ij / sum_ij w_ij。
    权重用像素邻域的各向同性高斯核，sigma 取自高斯 scale 的均值（像素域近似）。
    返回 [M,3]，范围裁剪到 [0,1]。
    """
    device = mu.device
    dtype = mu.dtype
    M = mu.shape[0]
    if M == 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)
    V, H, W, _ = images_tv.shape
    colors_acc = torch.zeros(M, 3, device=device, dtype=dtype)
    weights_vis = torch.zeros(M, 1, device=device, dtype=dtype)

    half = max(1, win // 2)
    du = torch.arange(-half, half + 1, device=device)
    dv = torch.arange(-half, half + 1, device=device)
    grid_u = du.view(1, -1).expand(1, du.numel())       # [1, K]
    grid_v = dv.view(-1, 1).expand(dv.numel(), 1).t()   # [1, K] after t()
    Kwin = du.numel()
    kernel_du = grid_u.reshape(1, Kwin)
    kernel_dv = grid_v.reshape(1, Kwin)
    sigma = scale.mean(dim=1) * float(sigma_scale)   # [M]
    sigma = sigma.clamp(min=1e-3)

    for v in range(V):
        u_pix, v_pix, z = project_points_torch(mu, c2w_tv[v], K_tv[v])
        valid = (z > 1e-4)
        if valid.sum() == 0:
            continue
        ui = u_pix.round().long().clamp(0, W - 1)
        vi = v_pix.round().long().clamp(0, H - 1)
        # neighbor coordinates per point
        uu = (ui.view(-1, 1) + kernel_du).clamp(0, W - 1)
        vv = (vi.view(-1, 1) + kernel_dv).clamp(0, H - 1)
        # sample pixels
        img = images_tv[v].to(device=device, dtype=dtype)  # [H,W,3]
        pix = img[vv, uu]  # [M,K,3] advanced indexing
        # weights: exp(-0.5 * (du^2+dv^2)/sigma^2)
        r2 = (kernel_du**2 + kernel_dv**2)  # [1,K]
        w = torch.exp(-0.5 * (r2 / (sigma.view(-1, 1)**2 + 1e-8)))  # [M,K]
        w = w * valid.view(-1, 1).to(dtype)  # mask by visibility
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        c_view = (w.unsqueeze(-1) * pix).sum(dim=1) / w_sum  # [M,3]
        colors_acc += c_view
        weights_vis += (w_sum > 0).to(dtype)

    colors_init = torch.where(weights_vis > 0, colors_acc / weights_vis.clamp(min=1.0), colors_acc*0)
    return colors_init.clamp(0.0, 1.0)



def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def custom_collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def prepare_batch(
    batch: dict,
    device: torch.device,
    debug: bool = False,
    ex4dgs_dir: Optional[str] = None,
    image_size: Optional[Tuple[int, int]] = None,
    config: Optional[dict] = None,
    mode: str = 'inference',
) -> dict:
    """
    Minimal batch preparation for baseline:
    Returns keys: feat_2d [T,V,H',W',C] (from dataset or zeros), conf [T,V,H,W,1],
                  camera_poses [T,V,4,4], camera_intrinsics [T,V,3,3], time_ids [T],
                  gt_images [T,V,H,W,3] (optional)
    """
    # Optional fields
    points = batch.get('points', None)
    if points is not None:
        points = points.to(device, non_blocking=True)  # [B,V,H,W,3]
    conf = batch['conf'].to(device, non_blocking=True)      # [B,V,H,W,1] or [B,V,H,W]
    time_indices = batch['time_idx'].to(device, non_blocking=True)  # [B]
    camera_poses = batch['camera_poses'].to(device, non_blocking=True)  # [B,V,4,4]
    camera_intrinsics_batch = batch.get('camera_intrinsics', None)
    feat_2d_batch = batch.get('feat_2d', None)
    if feat_2d_batch is not None:
        feat_2d_batch = feat_2d_batch.to(device, non_blocking=True)  # [B,V,H',W',C]

    if conf.dim() == 4:
        conf = conf.unsqueeze(-1)

    # Shapes (prefer conf if points is None)
    if points is not None:
        B, V, H, W = points.shape[0], points.shape[1], points.shape[2], points.shape[3]
    else:
        B, V, H, W = conf.shape[0], conf.shape[1], conf.shape[2], conf.shape[3]

    # T frames to use
    if config is not None:
        T = config.get('inference', {}).get('num_frames', 6) if mode == 'inference' else config.get('training', {}).get('num_frames', 2)
    else:
        T = B
    if B > T:
        if points is not None:
            points = points[:T]
        conf = conf[:T]
        time_indices = time_indices[:T]
        camera_poses = camera_poses[:T]
    else:
        T = B

    # Normalize camera_poses to [T, V, 4, 4] (c2w)
    if camera_poses.dim() == 3 and camera_poses.shape[-2:] == (4, 4):  # [V,4,4]
        camera_poses = camera_poses.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
    elif camera_poses.dim() == 2 and camera_poses.shape == (4, 4):  # [4,4]
        camera_poses = camera_poses.view(1, 1, 4, 4).expand(T, V, 4, 4).contiguous()

    # Build/normalize intrinsics to [T, V, 3, 3]
    def build_default_K(T, V, H, W, device, dtype):
        focal_length = max(H, W) * 1.2
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        K = torch.tensor([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], device=device, dtype=dtype)
        return K.view(1, 1, 3, 3).expand(T, V, 3, 3).contiguous()

    def to_K_3x3(batch_K, T, V, device, dtype):
        x = batch_K.to(device=device, dtype=dtype)
        if x.dim() >= 2 and x.shape[-2:] == (3, 3):
            if x.dim() == 4:  # [T?,V?,3,3]
                if x.shape[0] < T:
                    x = x.expand(T, -1, -1, -1)
                if x.shape[1] < V:
                    x = x.expand(x.shape[0], V, -1, -1)
                return x[:T, :V]
            elif x.dim() == 3:  # [V,3,3]
                x = x.unsqueeze(0).expand(T, -1, -1, -1)
                if x.shape[1] < V:
                    x = x.expand(T, V, 3, 3)
                return x[:, :V]
            elif x.dim() == 2:  # [3,3]
                return x.view(1, 1, 3, 3).expand(T, V, 3, 3).contiguous()
        # packed [*,*,4] -> fx,fy,cx,cy
        if x.shape[-1] == 4:
            fx, fy, cx, cy = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            K = torch.zeros(*x.shape[:-1], 3, 3, device=device, dtype=dtype)
            K[..., 0, 0] = fx
            K[..., 1, 1] = fy
            K[..., 0, 2] = cx
            K[..., 1, 2] = cy
            K[..., 2, 2] = 1.0
            if K.dim() == 5:  # [T,V,3,3]
                if K.shape[0] < T:
                    K = K.expand(T, -1, -1, -1, -1)
                if K.shape[1] < V:
                    K = K.expand(K.shape[0], V, -1, -1, -1)
                return K[:T, :V]
            elif K.dim() == 4:  # [V,3,3] or [T,3,3]
                if K.shape[0] == V:  # [V,3,3]
                    K = K.unsqueeze(0).expand(T, -1, -1, -1)
                    return K[:, :V]
                else:  # [T,3,3]
                    K = K.unsqueeze(1).expand(-1, V, -1, -1)
                    return K[:T, :V]
            elif K.dim() == 3:  # [3,3]
                return K.view(1, 1, 3, 3).expand(T, V, 3, 3).contiguous()
        # fallback default
        return build_default_K(T, V, H, W, device, dtype)

    if camera_intrinsics_batch is not None:
        if isinstance(camera_intrinsics_batch, torch.Tensor):
            camera_intrinsics = to_K_3x3(camera_intrinsics_batch[:T], T, V, device, camera_poses.dtype)
        else:
            camera_intrinsics = torch.as_tensor(camera_intrinsics_batch, device=device, dtype=camera_poses.dtype)
            camera_intrinsics = to_K_3x3(camera_intrinsics, T, V, device, camera_poses.dtype)
    else:
        # Rescale original intrinsics from config to target image_size
        data_cfg = config.get('data', {}) if config is not None else {}
        # target size
        if image_size is not None:
            H_t, W_t = image_size
        elif 'image_size' in data_cfg:
            H_t, W_t = tuple(data_cfg['image_size'])
        else:
            H_t, W_t = H, W
        K_cfg = data_cfg.get('K_orig', None)
        if K_cfg is not None:
            K_orig = torch.tensor(K_cfg, device=device, dtype=camera_poses.dtype)
            H0, W0 = data_cfg.get('orig_size', [H, W])
            sx = float(W_t) / float(W0)
            sy = float(H_t) / float(H0)
            S = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], device=device, dtype=camera_poses.dtype)
            K_new = (S @ K_orig)
        else:
            H0, W0 = data_cfg.get('orig_size', [H, W])
            f_orig = float(data_cfg.get('f_orig', max(H0, W0) * 1.2))
            cx0, cy0 = W0 / 2.0, H0 / 2.0
            sx = float(W_t) / float(W0)
            sy = float(H_t) / float(H0)
            fx = f_orig * sx
            fy = f_orig * sy
            cx_new = cx0 * sx
            cy_new = cy0 * sy
            K_new = torch.tensor([[fx, 0.0, cx_new], [0.0, fy, cy_new], [0.0, 0.0, 1.0]], device=device, dtype=camera_poses.dtype)
        camera_intrinsics = K_new.view(1,1,3,3).expand(T, V, 3, 3).contiguous()

    # Global points (average across views) — optional
    if points is not None:
        points_3d = points.reshape(T, V, H * W, 3).mean(dim=1)  # [T,N,3]
    else:
        points_3d = torch.zeros(T, 1, 3, device=device)

    # 2D features: prefer dataset-provided feat_2d; fallback to zeros
    if feat_2d_batch is not None:
        # Ensure T matches
        feat_2d = feat_2d_batch[:T]
    else:
        feat_2d_dim = 2048
        H_feat, W_feat = H // 14, W // 14
        feat_2d = torch.zeros(T, V, H_feat, W_feat, feat_2d_dim, device=device)

    time_ids = time_indices.long()

    # Load GT images from Ex4DGS (if provided)
    gt_images = None
    if ex4dgs_dir is not None and image_size is not None:
        gt_images_list = []
        for b in range(T):
            time_idx = time_indices[b].item()
            gt_imgs = load_gt_images_from_ex4dgs(time_idx, ex4dgs_dir, image_size)
            if gt_imgs is not None:
                gt_images_list.append(gt_imgs.to(device))  # [V,H,W,3]
            else:
                H_img, W_img = image_size
                gt_images_list.append(torch.zeros(V, H_img, W_img, 3, device=device))
        gt_images = torch.stack(gt_images_list, dim=0)  # [T,V,H,W,3]

    # Silhouette and visibility
    silhouette = batch.get('seganymo_mask', None)
    if silhouette is not None:
        silhouette = silhouette.to(device, non_blocking=True)[:T]  # [T,V,H,W]
    # Dynamic confidence / trajectories (SegAnyMo)
    dynamic_conf_tv = None
    dyn_conf_raw = batch.get('seganymo_dynamic_conf', None)
    if dyn_conf_raw is not None:
        dynamic_conf_tv = dyn_conf_raw.to(device, non_blocking=True).float()[:T]  # [T,V,H,W]

    dynamic_traj_tv = None
    dyn_traj_raw = batch.get('seganymo_dynamic_traj', None)
    if dyn_traj_raw is not None:
        dynamic_traj_tv = dyn_traj_raw.to(device, non_blocking=True).float()[:T]  # [T,V,2,H,W]

    # Optional sparse keypoints: [T,V,K,2] (or [T,V,K,>=2])
    keypoints_2d = None
    for key in ('keypoints_2d', 'kp_2d', 'seganymo_keypoints_2d', 'seganymo_kp_2d'):
        if key in batch and batch[key] is not None:
            keypoints_2d = batch[key]
            break
    if keypoints_2d is not None:
        if isinstance(keypoints_2d, torch.Tensor):
            keypoints_2d = keypoints_2d.to(device, non_blocking=True).float()
        else:
            keypoints_2d = torch.as_tensor(keypoints_2d, device=device).float()
        keypoints_2d = keypoints_2d[:T]
    # Compute conf_prob (probability) once
    conf_prob = torch.sigmoid(conf.squeeze(-1))  # [T,V,H,W]
    # visibility proxy
    seg_vis = batch.get('seganymo_visibility', None)
    if seg_vis is not None:
        seg_vis = seg_vis.to(device, non_blocking=True).float()[:T]  # [T,V,H,W]
        visibility = seg_vis.mean(dim=1)  # [T,H,W]
    else:
        visibility = conf_prob.mean(dim=1)  # [T,H,W]

    return {
        'points': points,  # keep per-view points for per-view chamfer
        'points_3d': points_3d,
        'feat_2d': feat_2d,
        'conf': conf,              # logits
        'conf_prob': conf_prob,    # probabilities
        'camera_poses': camera_poses,
        'camera_intrinsics': camera_intrinsics,
        'time_ids': time_ids,
        'gt_images': gt_images,
        'silhouette': silhouette,
        'visibility': visibility,
        'dynamic_conf_tv': dynamic_conf_tv,  # [T,V,H,W] per-view dyn prob (optional)
        'dynamic_traj_tv': dynamic_traj_tv,  # [T,V,2,H,W] per-view traj (optional)
        'keypoints_2d': keypoints_2d,  # [T,V,K,2] optional
    }


def _rotation_matrix_to_quaternion_numpy(R: np.ndarray) -> np.ndarray:
    original_shape = R.shape
    R_flat = R.reshape(-1, 3, 3)
    trace = np.trace(R_flat, axis1=1, axis2=2)
    quats = np.zeros((R_flat.shape[0], 4), dtype=R.dtype)
    mask1 = trace > 0
    if np.any(mask1):
        s = np.sqrt(trace[mask1] + 1.0) * 2
        quats[mask1, 0] = 0.25 * s
        quats[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s
        quats[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s
        quats[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if np.any(mask2):
        s = np.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
        quats[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s
        quats[mask2, 1] = 0.25 * s
        quats[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
        quats[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if np.any(mask3):
        s = np.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
        quats[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s
        quats[mask3, 1] = 0.25 * s
        quats[mask3, 2] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s
        quats[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if np.any(mask4):
        s = np.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
        quats[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s
        quats[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s
        quats[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s
        quats[mask4, 3] = 0.25 * s
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    quats = quats / norm
    if len(original_shape) == 3:
        return quats
    batch_dims = original_shape[:-2]
    return quats.reshape(*batch_dims, 4)


def save_gaussians_ply(gaussian_params: dict, output_path: str, sh_degree: int = 0):
    xyz = gaussian_params['mu']
    scale = gaussian_params['scale']
    color = gaussian_params['color']
    opacity = gaussian_params['opacity']
    num_gaussians = xyz.shape[0]
    rotation = np.zeros((num_gaussians, 4), dtype=np.float32)
    rotation[:, 0] = 1.0
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    if torch.is_tensor(scale):
        scale = scale.detach().cpu().numpy()
    if torch.is_tensor(opacity):
        opacity = opacity.detach().cpu().numpy()
    if torch.is_tensor(color):
        color = color.detach().cpu().numpy()
    if opacity.ndim > 1:
        opacity = opacity.squeeze()
    sh_coeffs = color.reshape(-1, 1, 3)
    M = xyz.shape[0]
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    elements = np.empty(M, dtype=np.dtype(dtype_list))
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = rotation[:, 1]
    elements['ny'] = rotation[:, 2]
    elements['nz'] = rotation[:, 3]
    elements['f_dc_0'] = sh_coeffs[:, 0, 0]
    elements['f_dc_1'] = sh_coeffs[:, 0, 1]
    elements['f_dc_2'] = sh_coeffs[:, 0, 2]
    elements['opacity'] = np.clip(opacity, 0.0, 1.0)
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    elements['rot_0'] = 1.0
    elements['rot_1'] = 0.0
    elements['rot_2'] = 0.0
    elements['rot_3'] = 0.0
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(output_path)
    print(f"  Saved Gaussian parameters to: {output_path} (M={M})")


def load_gt_images_from_ex4dgs(time_idx: int, ex4dgs_dir: str, image_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    camera_names = ['cam00', 'cam06', 'cam11', 'cam16']
    time_str = f"{time_idx:06d}"
    gt_images = []
    H, W = image_size
    for cam_name in camera_names:
        img_path = os.path.join(ex4dgs_dir, cam_name, f"{time_str}.png")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((W, H), Image.Resampling.LANCZOS)
                img_array = np.array(img).astype(np.float32) / 255.0
                gt_images.append(img_array)
            except Exception:
                return None
        else:
            return None
    if len(gt_images) != 4:
        return None
    return torch.from_numpy(np.stack(gt_images, axis=0))


def inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    config: dict,
    save_images: bool = True,
    save_gaussians: bool = True,
    ex4dgs_dir: Optional[str] = None,
    image_size: Optional[Tuple[int, int]] = None,
    use_fast_forward_init: bool = False,
    preset_aabb_from_points: bool = False,
) -> dict:
    model.eval()
    if save_images:
        images_dir = os.path.join(output_dir, 'rendered_images')
        os.makedirs(images_dir, exist_ok=True)
    if save_gaussians:
        gaussians_dir = os.path.join(output_dir, 'save_gaussians')
        os.makedirs(gaussians_dir, exist_ok=True)

    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            batch_data = prepare_batch(
                batch, device, debug=(batch_idx == 0),
                ex4dgs_dir=ex4dgs_dir, image_size=image_size,
                config=config, mode='inference'
            )
            points_3d = batch_data['points_3d']
            points = batch_data['points']
            feat_2d = batch_data['feat_2d']
            conf_seq = batch_data['conf']
            dyn_mask_tv = batch_data.get('dynamic_conf_tv', None)
            dyn_traj_tv = batch_data.get('dynamic_traj_tv', None)
            keypoints_2d = batch_data.get('keypoints_2d', None)
            camera_poses_seq = batch_data['camera_poses']
            camera_intrinsics_seq = batch_data['camera_intrinsics']
            time_ids = batch_data['time_ids']
            gt_images = batch_data.get('gt_images', None)

            # Optionally preset AABB tightly from points_3d (once before first forward)
            if preset_aabb_from_points and batch_idx == 0 and points_3d is not None and points_3d.numel() > 0:
                try:
                    if hasattr(model, 'reset_world_cache'):
                        model.reset_world_cache()
                    aabb = Trellis4DGS4DCanonical.estimate_points_aabb(points_3d)
                    model.set_world_aabb(aabb)
                except Exception as e:
                    print(f"[AABB preset] Failed to preset AABB from points: {e}")

            T = points_3d.shape[0]
            inference_start = time.time()
            H_t, W_t = config['data']['image_size']
            rendered_images_all = []
            rendered_images_static_all = []
            rendered_images_dynamic_all = []
            ## Debug dump: save model inputs for offline reproduction
            # os.makedirs('debug', exist_ok=True)
            # torch.save({
            #     'points_3d': points_3d,
            #     'feat_2d': feat_2d,
            #     'conf': conf_seq,
            #     'camera_poses': camera_poses_seq,
            #     'camera_intrinsics': camera_intrinsics_seq,
            #     'time_ids': time_ids,
            #     'build_canonical': False,
            # }, os.path.join('debug', f'infer_batch_{batch_idx:05d}.pth'))
            output = model(
                points_full=points,
                feat_2d=feat_2d,
                camera_poses=camera_poses_seq,
                camera_K=camera_intrinsics_seq,
                time_ids=time_ids,
                dyn_mask_2d=dyn_mask_tv,
                conf_2d=conf_seq,
            )

            mu_t, scale_t, color_t, alpha_t = output['mu_t'], output['scale_t'], output['color_t'], output['alpha_t']
            rot_t = output.get('rot_t', None)
            mu_t_s = output.get('mu_t_static', None)
            scale_t_s = output.get('scale_t_static', None)
            color_t_s = output.get('color_t_static', None)
            alpha_t_s = output.get('alpha_t_static', None)
            rot_t_s = output.get('rot_t_static', None)
            mu_t_d = output.get('mu_t_dynamic', None)
            scale_t_d = output.get('scale_t_dynamic', None)
            color_t_d = output.get('color_t_dynamic', None)
            alpha_t_d = output.get('alpha_t_dynamic', None)
            rot_t_d = output.get('rot_t_dynamic', None)
            has_split = (mu_t_s is not None) and (mu_t_d is not None)
            T, M = mu_t.shape[0], mu_t.shape[1]

            # 注意：scale 现在由网络直接输出并限幅，不再做像素半径校准（避免改变几何/尺度语义）

            render_start = time.time()
            for t in range(T):
                camera_poses_t = camera_poses_seq[t]            # [V,4,4] c2w
                camera_intrinsics_t = camera_intrinsics_seq[t]  # [V,3,3]

                # 预构建相机（每帧一次）
                cams = []
                for vi in range(camera_poses_t.shape[0]):
                    c2w = camera_poses_t[vi].detach().cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    R = w2c[:3, :3].astype(np.float32)
                    t_vec = w2c[:3, 3].astype(np.float32)
                    K_np = camera_intrinsics_t[vi].detach().cpu().numpy().astype(np.float32)
                    cams.append(IntrinsicsCamera(
                        K=K_np, R=R, T=t_vec,
                        width=int(W_t), height=int(H_t),
                        znear=0.01, zfar=100.0,
                    ))

                bg_color = torch.ones(3, device=device)
                scaling_modifier = float(config.get('render', {}).get('scaling_modifier', 1.0))

                def _render_set(mu_frame, scale_frame, rot_frame, color_frame, alpha_frame):
                    if mu_frame.numel() == 0:
                        Vn = camera_poses_t.shape[0]
                        return torch.zeros(Vn, H_t, W_t, 3, device=device, dtype=mu_frame.dtype)
                    # scale 由网络输出：仅做限幅，避免渲染数值不稳定
                    # 与训练保持一致：仅防止数值为 0，不再硬性截断上限
                    scale_frame = scale_frame.to(dtype=mu_frame.dtype).clamp(min=1e-6)
                                        # 训练阶段 alpha 已经是 sigmoid 输出，推理保持一致即可
                    opacity = alpha_frame if alpha_frame.dim() == 2 else alpha_frame.unsqueeze(-1)
                    # 若仍担心异常，可在调试中打开以下 clamp
                    # opacity = opacity.clamp(0.0, 1.0)
                    num_gs = mu_frame.shape[0]
                    if rot_frame is None:
                        rotation = torch.zeros(num_gs, 4, device=mu_frame.device, dtype=mu_frame.dtype)
                        rotation[:, 0] = 1.0
                    else:
                        rotation = matrix_to_quaternion(rot_frame).to(device=mu_frame.device, dtype=mu_frame.dtype)
                        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    sh = rgb2sh0(color_frame).unsqueeze(1)  # [M,1,3]
                    gs_attrs = GaussianAttributes(xyz=mu_frame, opacity=opacity, scaling=scale_frame, rotation=rotation, sh=sh)
                    imgs = []
                    for cam in cams:
                        res_v = render_gs(camera=cam, bg_color=bg_color, gs=gs_attrs, target_image=None, sh_degree=0, scaling_modifier=scaling_modifier)
                        imgs.append(res_v["color"])  # [3,H,W]
                    imgs_t_stacked = torch.stack(imgs, dim=0)  # [V,3,H,W]
                    return imgs_t_stacked.permute(0, 2, 3, 1).contiguous()  # [V,H,W,3]

                rot_frame = rot_t[t] if rot_t is not None else None
                rendered_images_all.append(_render_set(mu_t[t], scale_t[t], rot_frame, color_t[t], alpha_t[t]).unsqueeze(0))
                if has_split:
                    rot_s_frame = rot_t_s[t] if rot_t_s is not None else None
                    rot_d_frame = rot_t_d[t] if rot_t_d is not None else None
                    rendered_images_static_all.append(_render_set(mu_t_s[t], scale_t_s[t], rot_s_frame, color_t_s[t], alpha_t_s[t]).unsqueeze(0))
                    rendered_images_dynamic_all.append(_render_set(mu_t_d[t], scale_t_d[t], rot_d_frame, color_t_d[t], alpha_t_d[t]).unsqueeze(0))

            render_time = time.time() - render_start
            rendered_images = torch.cat(rendered_images_all, dim=0)  # [T,V,H,W,3]
            rendered_images_static = torch.cat(rendered_images_static_all, dim=0) if has_split else None
            rendered_images_dynamic = torch.cat(rendered_images_dynamic_all, dim=0) if has_split else None

            if save_images:
                T_render, N_views, H, W, C = rendered_images.shape
                B = T_render

                pred_images = rendered_images.permute(0, 1, 4, 2, 3).contiguous().clamp(0.0, 1.0)  # [B,V,3,H,W]
                if gt_images is not None:
                    gt_images_tensor = gt_images.permute(0, 1, 4, 2, 3).contiguous()  # [B,V,3,Hg,Wg]
                    Hg, Wg = gt_images_tensor.shape[-2:]
                    if (Hg != H) or (Wg != W):
                        gt_images_tensor = torch.nn.functional.interpolate(
                            gt_images_tensor.reshape(-1, 3, Hg, Wg),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False,
                        ).reshape(B, N_views, 3, H, W)
                    gt_images_tensor = gt_images_tensor.clamp(0.0, 1.0)
                else:
                    gt_images_tensor = torch.zeros(B, N_views, 3, H, W, device=device, dtype=pred_images.dtype)

                def ensure_chw(x: torch.Tensor) -> torch.Tensor:
                    if x.dim() != 3:
                        return x
                    if x.shape[0] == 3:
                        return x
                    if x.shape[1] == 3:
                        return x.permute(1, 0, 2)
                    if x.shape[2] == 3:
                        return x.permute(2, 0, 1)
                    return x

                def save_pred_gt_grid(pred_tvchw: torch.Tensor, suffix: str):
                    grid_rows = []
                    for view_idx in range(min(4, N_views)):
                        row = torch.cat([ensure_chw(pred_tvchw[b, view_idx]) for b in range(B)], dim=2)
                        grid_rows.append(row)
                    for view_idx in range(min(4, N_views)):
                        row = torch.cat([ensure_chw(gt_images_tensor[b, view_idx]) for b in range(B)], dim=2)
                        grid_rows.append(row)
                    max_w = max(row.shape[2] for row in grid_rows)
                    grid_rows = [torch.nn.functional.pad(row, (0, max_w - row.shape[2], 0, 0)) for row in grid_rows]
                    grid_img = torch.cat(grid_rows, dim=1)
                    name = f"batch_{batch_idx:05d}_{suffix}_pred_gt_grid.png"
                    save_image(grid_img, os.path.join(images_dir, name))

                save_pred_gt_grid(pred_images, 'all')
                if has_split and rendered_images_static is not None and rendered_images_dynamic is not None:
                    pred_s = rendered_images_static.permute(0, 1, 4, 2, 3).contiguous().clamp(0.0, 1.0)
                    pred_d = rendered_images_dynamic.permute(0, 1, 4, 2, 3).contiguous().clamp(0.0, 1.0)
                    save_pred_gt_grid(pred_s, 'static')
                    save_pred_gt_grid(pred_d, 'dynamic')

            # Metrics: MS-SSIM/L1 with border crop
            metrics_dir = os.path.join(output_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            metrics: Dict[str, float] = {}
            if gt_images is not None:
                border = int(config.get('loss_params', {}).get('ssim_border', 0))
                ws = int(config.get('loss_params', {}).get('ssim_window_size', 11))
                ms_scales = max(1, int(config.get('loss_params', {}).get('ssim_ms_scales', 3)))

                pred = rendered_images.permute(0,1,4,2,3).reshape(-1,3,H,W).clamp(0,1)
                gt   = gt_images.permute(0,1,4,2,3).reshape(-1,3,H,W).clamp(0,1)
                if border > 0:
                    pred = crop_border_tensor(pred, border)
                    gt   = crop_border_tensor(gt, border)
                # L1
                metrics['photo_l1'] = torch.abs(pred - gt).mean().item()
                # MS-SSIM
                metrics['ms_ssim'] = ssim_ms_torch(pred, gt, scales=ms_scales, window_size=ws).item()

            # save metrics json
            if metrics:
                with open(os.path.join(metrics_dir, f'batch_{batch_idx:05d}.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)

            if save_gaussians:
                for t in range(T):
                    time_idx = time_ids[t].item() if t < len(time_ids) else t
                    gaussian_dict = {
                        'mu': mu_t[t].cpu(),
                        'scale': scale_t[t].cpu(),
                        'color': color_t[t].cpu(),
                        'opacity': alpha_t[t].cpu().squeeze(-1),
                    }
                    ply_path = os.path.join(gaussians_dir, f'time_{time_idx:05d}_gaussians.ply')
                    save_gaussians_ply(gaussian_dict, ply_path, sh_degree=0)

            total_samples += T
            inference_time = time.time() - inference_start
            total_time += inference_time + render_time

    avg_time = total_time / total_samples if total_samples > 0 else 0.0
    return {'total_samples': total_samples, 'total_time': total_time, 'avg_time_per_sample': avg_time}


def _load_hf_backbone(model: Trellis4DGS4DCanonical, hf_dir: str, device: torch.device):
    try:
        pj = os.path.join(hf_dir, 'pipeline.json')
        with open(pj, 'r') as f:
            pipe = json.load(f)
        m = pipe['args']['models']
        def _ckpt_path(key):
            base = m[key]
            # prefer .safetensors
            p = os.path.join(hf_dir, base + '.safetensors') if not base.endswith('.safetensors') else os.path.join(hf_dir, base)
            if not os.path.exists(p):
                # maybe under ckpts subdir
                p2 = os.path.join(hf_dir, 'ckpts', os.path.basename(base) + '.safetensors')
                if os.path.exists(p2):
                    return p2
            # also try json sibling to infer folder
            if not os.path.exists(p):
                p3 = os.path.join(hf_dir, 'ckpts', os.path.basename(base))
                if os.path.isdir(p3):
                    # inside folder with .safetensors name
                    for fn in os.listdir(p3):
                        if fn.endswith('.safetensors'):
                            return os.path.join(p3, fn)
            return p
        flow_path = _ckpt_path('slat_flow_model')
        dec_path  = _ckpt_path('slat_decoder_gs')
        loaded = []
        def _safe_load(module: nn.Module, sd: dict, name: str):
            msd = module.state_dict()
            keep = {}
            drop = []
            for k, v in sd.items():
                if k in msd and msd[k].shape == v.shape:
                    keep[k] = v
                else:
                    drop.append(k)
            if keep:
                missing, unexpected = module.load_state_dict(keep, strict=False)
                print(f"[HF] {name}: loaded {len(keep)} tensors, skipped {len(drop)}, missing={len(missing)}, unexpected={len(unexpected)}")
                return True
            else:
                print(f"[HF] {name}: no matching tensors; skipped all {len(drop)} keys")
                return False
        if os.path.exists(flow_path):
            sd_flow = load_file(flow_path)
            flow_dtype = next(model.flow.parameters()).dtype
            flow_dev = next(model.flow.parameters()).device
            sd_flow = {k: v.to(device=flow_dev, dtype=flow_dtype) for k, v in sd_flow.items()}
            if _safe_load(model.flow, sd_flow, 'flow'):
                loaded.append('flow')
        else:
            print(f"[HF] Flow checkpoint not found at {flow_path}")
        if os.path.exists(dec_path):
            sd_dec = load_file(dec_path)
            dec_dtype = next(model.decoder.parameters()).dtype
            dec_dev = next(model.decoder.parameters()).device
            sd_dec = {k: v.to(device=dec_dev, dtype=dec_dtype) for k, v in sd_dec.items()}
            if _safe_load(model.decoder, sd_dec, 'decoder'):
                loaded.append('decoder')
        else:
            print(f"[HF] Decoder checkpoint not found at {dec_path}")
        return loaded
    except Exception as e:
        print(f"[HF] Failed to load HF backbone: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Inference Baseline 4DGS')
    parser.add_argument('--config', type=str, default='configs/anchorwarp_4dgs.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to training checkpoint (.pth). If omitted, use randomly initialized model')
    parser.add_argument('--hf_dir', type=str, default='weights/TRELLIS-image-large', help='Directory containing HuggingFace weights (ckpts)')
    parser.add_argument('--output_dir', type=str, default='results_train/AnchorWarp4DGS/inference', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (should equal T, overrides inference.num_frames)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers (overrides config)')
    parser.add_argument('--save_images', action='store_true', default=True, help='Save rendered images')
    parser.add_argument('--save_gaussians', action='store_false', default=True, help='Save Gaussian parameters')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val'], help='Which split to use for inference')
    parser.add_argument('--use_fast_forward_init', action='store_true', default=True, help='Use fast_forward for color self-correction during inference')
    parser.add_argument('--preset_aabb_from_points', action='store_true', default=False, help='Estimate tight AABB from points_3d once and preset to model before forward (disabled by default to match training)')
    parser.add_argument('--decoder_xyz_space', type=str, default=None, choices=['grid','world'], help='Override decoder xyz space mapping (legacy flag)')
    parser.add_argument('--decoder_xyz_mode', type=str, default=None, choices=['grid_abs','grid_offset','world_abs','world_offset'], help='Override decoder xyz mapping mode')
    parser.add_argument('--require_preset_aabb', action='store_true', default=False, help='Require preset AABB and error if missing')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    image_dir = config['data'].get('image_dir', None)
    if image_dir is None:
        pi3_results_dir = config['data']['pi3_results_dir']
        test_time_dir = os.path.join(pi3_results_dir, 'time_00')
        if os.path.exists(test_time_dir):
            image_files = [f for f in os.listdir(test_time_dir) if f.endswith(('.png', '.jpg')) and 'view' in f.lower()]
            if len(image_files) >= 4:
                image_dir = pi3_results_dir
                print(f"Found images in pi3_results_dir: {image_dir}")
        if image_dir is None:
            possible_dirs = ['assets/zzb/t', 'assets/zzb/beef', 'data/beef']
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    test_time_dir = os.path.join(dir_path, 'time_00')
                    if os.path.exists(test_time_dir):
                        image_files = [f for f in os.listdir(test_time_dir) if f.endswith(('.png', '.jpg')) and 'view' in f.lower()]
                        if len(image_files) >= 4:
                            image_dir = dir_path
                            print(f"Found images in: {image_dir}")
                            break

    dataset = VoxelFF4DGSDataset(
        pi3_results_dir=config['data']['pi3_results_dir'],
        seganymo_dir=config['data']['seganymo_dir'],
        reloc3r_dir=config['data']['reloc3r_dir'],
        image_size=tuple(config['data']['image_size']),
        image_dir=image_dir,
        preload_all=bool(config.get('data', {}).get('preload_all', True)),
        sample_interval=int(config.get('data', {}).get('sample_interval', 30)),
    )

    if args.split == 'all':
        inference_dataset = dataset
    else:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        inference_dataset = train_dataset if args.split == 'train' else val_dataset

    print(f"Inference dataset size: {len(inference_dataset)}")

    T_inference = config.get('inference', {}).get('num_frames', 10)
    batch_size = args.batch_size if args.batch_size is not None else T_inference
    print(f"  DataLoader batch_size: {batch_size} (T={T_inference})")

    dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )

    # Trellis canonical 4D model
    model_config = config.get('model', {})

    # Model: 参数全部来自 configs/ff4dgsmotion.yaml 的 model 字段，结构在 FF4DGSMotion/models/FF4DGSMotion.py 内部构建
    model = Trellis4DGS4DCanonical(cfg=model_config).to(device)
    print("Using Trellis4DGS4DCanonical (FF4DGSMotion.py Step1~Step7)")

    # Load checkpoint from --resume (training ckpt)
    ckpt_path = args.resume
    if ckpt_path:
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            # Support both training checkpoints and raw state_dict
            state = checkpoint.get('model_state_dict', None) if isinstance(checkpoint, dict) else None
            if state is None and isinstance(checkpoint, dict):
                # Some checkpoints use 'state_dict'
                state = checkpoint.get('state_dict', None)
            try:
                if state is not None:
                    model.load_state_dict(state)
                else:
                    model.load_state_dict(checkpoint)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        else:
            print(f"Checkpoint not found: {ckpt_path}. Using randomly initialized model.")
    else:
        print("No checkpoint provided; using randomly initialized model.")

    ex4dgs_dir = config.get('data', {}).get('ex4dgs_dir', 'assets/Ex4DGS')
    if not os.path.isabs(ex4dgs_dir):
        ex4dgs_dir = os.path.abspath(ex4dgs_dir)

    stats = inference(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
        config=config,
        save_images=args.save_images,
        save_gaussians=args.save_gaussians,
        ex4dgs_dir=ex4dgs_dir,
        image_size=tuple(config['data']['image_size']),
        use_fast_forward_init=args.use_fast_forward_init,
        preset_aabb_from_points=args.preset_aabb_from_points,
    )

    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}")
    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Average time per sample: {stats['avg_time_per_sample']:.4f}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
