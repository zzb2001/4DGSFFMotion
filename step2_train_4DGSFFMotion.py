"""
Training script for Baseline 4DGS (GaussianTransformerBaselineModel)
- Removes Canonical/Anchor/Motion layers and all dynamic mask logic
- Trains only with image supervision (L1/SSIM via PhotoConsistencyLoss) and optional temporal smoothness
"""
import os
from torchvision.utils import save_image
import time
import argparse
import random
from datetime import datetime
from typing import Optional, Tuple, Dict
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from safetensors.torch import load_file
import json
import os
from PIL import Image
# save_image already imported
import math
from torchvision.utils import save_image
from typing import List

from FF4DGSMotion.data.dataset import VoxelFF4DGSDataset
from FF4DGSMotion.models.FF4DGSMotion import Trellis4DGS4DCanonical

from FF4DGSMotion.losses.photo_loss import PhotoConsistencyLoss
# from fused_ssim import fused_ssim
import torch.nn.functional as F
try:
    from lpips import LPIPS  # perceptual loss
except ImportError:
    LPIPS = None
from FF4DGSMotion.camera.camera import IntrinsicsCamera
from FF4DGSMotion.diff_renderer.gaussian import render_gs, GaussianAttributes
from FF4DGSMotion.models._utils import matrix_to_quaternion, rgb2sh0
LPIPS_MODEL = None


def _save_grid_6x4(x: torch.Tensor, name: str):
    """Save 6x4 grid (6 cols × 4 rows) single-channel debug image.
    Args:
        x: Tensor [6,4,H,W] or [6,4,H,W,1]
        name: filename (without extension) under debug/images/
    """
    if x is None:
        return
    if x.dim() == 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.dim() != 4:
        print(f"[debug] skip {name}, unexpected shape {tuple(x.shape)}")
        return
    try:
        # [6,4,H,W] -> [4,6,H,W] -> [24,1,H,W]
        x_cpu = x.detach().cpu()
        x_cpu = x_cpu.permute(1, 0, 2, 3).contiguous()
        x_cpu = x_cpu.view(-1, 1, x_cpu.shape[-2], x_cpu.shape[-1])
        save_image(x_cpu, f"debug/images/{name}.png", nrow=6, normalize=True)
    except Exception as e:
        print(f"[debug] failed to save {name}: {e}")
# pretty print non-zero train metrics
def _fmt_metrics(title: str, m: dict, keys: list[str] | None = None, tol: float = 1e-8) -> str:
    """Pretty-print metrics.
    If *keys* is None, print every key in *m* (except 'loss'/'stage').
    Only values with absolute magnitude larger than *tol* are shown.
    Additionally, the current *stage* (if present in *m*) will be
    appended to *title* so the caller doesn’t have to manually
    include it.
    """
    stage = m.get('stage', None)
    if stage is not None:
        title = f"{title}(Stage {int(stage)})"
    parts = [f"{title}: {m.get('loss', 0.0):.4f}"]
    if keys is None:
        keys = [k for k in m.keys() if k not in ('loss', 'stage')]
    for k in keys:
        v = float(m.get(k, 0.0))
        if abs(v) > tol:
            pretty = k.replace('_', ' ').title()
            parts.append(f"{pretty}: {v:.4f}")
    return " | ".join(parts)
# ---------- pretty-print helpers ----------

def _fmt_nonzero(prefix: str, kv: dict[str, float], tol: float = 1e-8) -> str:
    parts = [prefix]
    for k, v in kv.items():
        if abs(float(v)) > tol:
            parts.append(f"{k}:{v:.4g}")
    return " | ".join(parts)


def _gaussian(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).to(device=device, dtype=dtype)
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


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_cuda_devices(train_cfg: dict) -> list[int]:
    """
    可以是:
      - "cpu"
      - "0" / "0,1,2"
      - [0,1,2]
    """
    val = (train_cfg or {}).get("cuda", None)
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "cpu", "none"):
            return []
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [int(p) for p in parts]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    return []


class DistributedWindowBatchSampler(torch.utils.data.Sampler):
    """把 TimeWindowSampler 的 windows 按 rank 分片（每个 batch 仍是一整个 window）。"""

    def __init__(self, windows: list[list[int]], rank: int, world_size: int, shuffle: bool = True, seed: int = 0):
        self.windows = list(windows)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self.windows)))
        if self.shuffle:
            g = random.Random(self.seed + self.epoch)
            g.shuffle(order)
        order = order[self.rank :: self.world_size]
        for i in order:
            yield self.windows[i]

    def __len__(self):
        n = len(self.windows)
        return (n + self.world_size - 1 - self.rank) // self.world_size


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

# ---------- motion_scale warm-up & dynamic loss weights ----------
def _update_motion_scale(model, gstep: int, config: dict):
    """Linearly warm-up model.motion_scale (now Parameter)."""
    warm_cfg = config.get('training', {}).get('motion_scale_warmup', {})
    warmup = int(warm_cfg.get('steps', 2000))
    start = float(warm_cfg.get('start', 0.1))  # 与模型初始保持一致
    end = float(warm_cfg.get('end', 1.0))
    alpha = min(1.0, gstep / max(1, warmup))
    cur = start + alpha * (end - start)
    tgt = model.module if hasattr(model, 'module') else model
    if hasattr(tgt, 'motion_scale') and torch.is_tensor(tgt.motion_scale):
        with torch.no_grad():
            tgt.motion_scale.data.fill_(float(cur))
def prepare_batch(
    batch: dict,
    device: torch.device,
    ex4dgs_dir: Optional[str],
    image_size: Tuple[int, int],
    config: dict,
    mode: str,
) -> Dict[str, torch.Tensor]:
    """
    Minimal batch preparation for baseline training/validation
    Returns keys: feat_2d [T,V,H',W',C] (from dataset or zeros), conf [T,V,H,W,1],
                  camera_poses [T,V,4,4], camera_intrinsics [T,V,3,3], time_ids [T],
                  gt_images [T,V,H,W,3] (optional)
    """
    # Optional fields
    points = batch.get('points', None)
    if points is not None:
        points = points.to(device, non_blocking=True)  # [B,V,H,W,3]
    conf = batch['conf'].to(device, non_blocking=True)      # [B,V,H,W,1] or [B,V,H,W]
    time_idx = batch['time_idx'].to(device, non_blocking=True)  # [B]
    camera_poses = batch['camera_poses'].to(device, non_blocking=True)  # [B,V,4,4]
    camera_intrinsics_batch = batch.get('camera_intrinsics', None)
    feat_2d_batch = batch.get('feat_2d', None)
    flow_fwd = batch.get('flow_fwd', None)
    if feat_2d_batch is not None:
        feat_2d_batch = feat_2d_batch.to(device, non_blocking=True)  # [B,V,H',W',C]
        # 2D 特征在 AMP 训练下用 fp16 存储即可（显存减半）；验证默认保持 fp32，避免 dtype/bias 不匹配
        feat_fp16 = bool(config.get("training", {}).get("feat_2d_fp16", True))
        if mode == "train" and feat_fp16:
            feat_2d_batch = feat_2d_batch.to(dtype=torch.float16)
    if flow_fwd is not None:
        flow_fwd = flow_fwd.to(device, non_blocking=True).to(dtype=torch.float32)

    if conf.dim() == 4:
        conf = conf.unsqueeze(-1)

    # Determine shapes robustly
    if points is not None:
        B, V, H, W = points.shape[0], points.shape[1], points.shape[2], points.shape[3]
    else:
        B, V, H, W = conf.shape[0], conf.shape[1], conf.shape[2], conf.shape[3]

    # T frames to use
    seq_len = config.get('training', {}).get('sequence_length', config.get('training', {}).get('num_frames', 2))
    T = seq_len
    if B > T:
        if points is not None:
            points = points[:T]
        conf = conf[:T]
        time_idx = time_idx[:T]
        camera_poses = camera_poses[:T]
        if flow_fwd is not None and flow_fwd.dim() >= 4:
            flow_fwd = flow_fwd[:T]
    else:
        T = B

    # Normalize camera_poses to [T, V, 4, 4] (dataset provides c2w)
    if camera_poses.dim() == 3 and camera_poses.shape[-2:] == (4, 4):  # [V,4,4]
        camera_poses = camera_poses.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
    elif camera_poses.dim() == 2 and camera_poses.shape == (4, 4):  # [4,4]
        camera_poses = camera_poses.view(1, 1, 4, 4).expand(T, V, 4, 4).contiguous()

    # Intrinsics helpers -> normalize to [T,V,3,3]
    def build_default_K(T, V, H, W, device, dtype):
        focal_length = max(H, W) * 1.2
        cx, cy = W / 2.0, H / 2.0
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

    # Build/normalize intrinsics
    if camera_intrinsics_batch is not None:
        if isinstance(camera_intrinsics_batch, torch.Tensor):
            camera_intrinsics = to_K_3x3(camera_intrinsics_batch[:T], T, V, device, camera_poses.dtype)
        else:
            camera_intrinsics = torch.as_tensor(camera_intrinsics_batch, device=device, dtype=camera_poses.dtype)
            camera_intrinsics = to_K_3x3(camera_intrinsics, T, V, device, camera_poses.dtype)
    else:
        camera_intrinsics = build_default_K(T, V, H, W, device, camera_poses.dtype)

    # If dataset didn't provide intrinsics, rescale original intrinsics from config to target image_size
    if camera_intrinsics_batch is None and config is not None:
        data_cfg = config.get('data', {})
        H_t, W_t = image_size
        K_cfg = data_cfg.get('K_orig', None)
        if K_cfg is not None:
            K_orig = torch.tensor(K_cfg, device=device, dtype=camera_poses.dtype)
            H0, W0 = data_cfg.get('orig_size', [H, W])
            sx = float(W_t) / float(W0)
            sy = float(H_t) / float(H0)
            S = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], device=device, dtype=camera_poses.dtype)
            K_new = S @ K_orig
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
        camera_intrinsics = K_new.unsqueeze(0).unsqueeze(0).repeat(T, V, 1, 1)

    # Global points (average across views)
    # points_3d（每帧点云）很占显存且多数训练阶段不需要；改为在需要的 loss 内部按需计算
    points_3d = None

    # 2D features: prefer dataset-provided feat_2d; fallback to zeros
    if feat_2d_batch is not None:
        feat_2d = feat_2d_batch[:T]
    else:
        feat_2d_dim = 2048
        H_feat, W_feat = H // 14, W // 14
        feat_2d = torch.zeros(T, V, H_feat, W_feat, feat_2d_dim, device=device)

    # Compute conf_prob once (probabilities from logits)
    conf_prob = torch.sigmoid(conf.squeeze(-1))  # [T,V,H,W]
    # Per-point visibility (for chamfer weighting): prefer seganymo_visibility; else from conf_prob
    visibility = None
    seg_vis = batch.get('seganymo_visibility', None)
    if seg_vis is not None:
        seg_vis = seg_vis.to(device, non_blocking=True).float()  # [B,V,H,W]
        seg_vis = seg_vis[:T]
        visibility = seg_vis.mean(dim=1)  # [T,H,W]
    else:
        visibility = conf_prob.mean(dim=1)  # [T,H,W]

    # Dynamic confidence (SegAnyMo): average across views -> [T,H,W]
    dynamic_conf = None
    dynamic_conf_tv = None
    dyn_conf_raw = batch.get('seganymo_dynamic_conf', None)
    if dyn_conf_raw is not None:
        dyn_conf_raw = dyn_conf_raw.to(device, non_blocking=True).float()[:T]  # [T,V,H,W]
        dynamic_conf_tv = dyn_conf_raw
        dynamic_conf = dyn_conf_raw.mean(dim=1)  # [T,H,W]

    # Dynamic trajectories (SegAnyMo): per-view -> [T,V,2,H,W]
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
            keypoints_2d = keypoints_2d.to(device=device, non_blocking=True).float()
        else:
            keypoints_2d = torch.as_tensor(keypoints_2d, device=device).float()
        keypoints_2d = keypoints_2d[:T]

    # GT images if available
    gt_images = None
    if ex4dgs_dir is not None and image_size is not None:
        gt_images_list = []
        for b in range(T):
            t_idx = time_idx[b].item()
            imgs = load_gt_images_from_ex4dgs(t_idx, ex4dgs_dir, image_size)
            if imgs is not None:
                gt_images_list.append(imgs.to(device))
            else:
                H_img, W_img = image_size
                gt_images_list.append(torch.zeros(V, H_img, W_img, 3, device=device))
        gt_images = torch.stack(gt_images_list, dim=0)

    # Silhouette masks from SegAnyMo
    silhouette = batch.get('seganymo_mask', None)
    if silhouette is not None:
        silhouette = silhouette.to(device, non_blocking=True)  # [B,V,H,W]
        silhouette = silhouette[:T]

    return {
        'points': points[:T] if points is not None else None,  # [T,V,H,W,3]
        'points_3d': points_3d,
        'feat_2d': feat_2d,
        'conf': conf,                  # logits
        'conf_prob': conf_prob,        # probabilities
        'camera_poses': camera_poses,
        'camera_intrinsics': camera_intrinsics,
        'time_ids': time_idx.long(),
        'gt_images': gt_images,
        'silhouette': silhouette,
        'visibility': visibility,      # [T,H,W] per-pixel visibility proxy
        'dynamic_conf': dynamic_conf,  # [T,H,W] avg across views if available
        'dynamic_conf_tv': dynamic_conf_tv,  # [T,V,H,W] per-view dyn prob (optional)
        'dynamic_traj_tv': dynamic_traj_tv,  # [T,V,2,H,W] per-view traj (optional)
        'keypoints_2d': keypoints_2d,  # [T,V,K,2] optional
        'flow_fwd': flow_fwd,          # [T,V,H,W,2] optional
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    photo_loss_fn: PhotoConsistencyLoss,
    lpips_fn: Optional[nn.Module],
    device: torch.device,
    config: dict,
    writer: Optional[SummaryWriter],
    epoch: int,
    ex4dgs_dir: Optional[str],
    output_dir: Optional[str],
    is_rank0: bool = True,
) -> dict:
    model.train()
    total_loss = 0.0
    sum_photo = 0.0
    sum_ssim = 0.0
    sum_psnr = 0.0
    sum_lpips = 0.0
    sum_retarget = 0.0

    sum_sil = 0.0
    sum_alpha = 0.0
    sum_chamfer = 0.0
    sum_motion = 0.0
    sum_vel_high_smooth = 0.0  # velocity smoothness for motion_block
    sum_temporal = 0.0
    sum_imp_budget = 0.0
    sum_imp_geom = 0.0
    sum_exist_budget = 0.0
    sum_flow_sparse = 0.0
    sum_flow_reproj = 0.0
    sum_mask_dyn = 0.0
    sum_mask_sta = 0.0
    sum_mask_cross = 0.0
    sum_res_mag = 0.0
    sum_res_smooth = 0.0
    sum_imp_entropy = 0.0
    sum_imp_repel = 0.0
    sum_kp2d = 0.0
    sum_eval = 0.0
    # stage indicator for pretty printing
    current_stage = None
    n_batches = 0

    last_rendered_images = None
    last_gaussian_params = None
    last_gt_images = None
    last_time_indices = None

    lambda_smooth = float(config.get('loss_weights', {}).get('dynamic_temporal', 0.0))
    # Retargeting loss (Forge4D-style)
    lw_cfg = config.get("loss_weights", {}) or {}
    base_retarget = float(lw_cfg.get("retarget", 0.5))
    base_retarget_ssim = float(lw_cfg.get("retarget_ssim", 0.5))
    base_retarget_lpips = float(lw_cfg.get("retarget_lpips", 0.5))
    base_flow_sparse = float(lw_cfg.get("flow_sparse", 0.0))
    base_flow_reproj = float(lw_cfg.get("flow_reproj", 0.0))
    ret_cfg = (config.get("loss_params", {}) or {}).get("retarget", {}) or {}
    ret_detach_ref = bool(ret_cfg.get("detach_ref", True))
    ret_view_sub = int(ret_cfg.get("view_subsample", 0))
    # stage gating for retargeting:
    # - default: only Stage 2 enables retargeting (motion-focused stage)
    # - set loss_params.retarget.enable_stage1=true to also enable in Stage 1
    enable_retarget_stage1 = bool(ret_cfg.get("enable_stage1", False))
    flow_sparse_cfg = (config.get("loss_params", {}) or {}).get("flow_sparse", {}) or {}
    flow_sparse_tau = float(flow_sparse_cfg.get("tau_assoc", 4.0))
    flow_sparse_max_mu = int(flow_sparse_cfg.get("max_mu_samples", 4096))
    flow_sparse_use_cyclic = bool(flow_sparse_cfg.get("use_cyclic_mask", False))
    flow_reproj_cfg = (config.get("loss_params", {}) or {}).get("flow_reproj", {}) or {}
    flow_reproj_max_mu = int(flow_reproj_cfg.get("max_mu_samples", 4096))
    flow_reproj_use_cyclic = bool(flow_reproj_cfg.get("use_cyclic_mask", False))

    # ----------------------------
    # Two-stage training schedule
    # ----------------------------
    stage_cfg = config.get("training", {}).get("stages", {}) or {}
    stage1_epochs = int(stage_cfg.get("stage1_epochs", 100))
    # stage 1: epoch < stage1_epochs
    # stage 2: epoch >= stage1_epochs
    if epoch < stage1_epochs:
        stage = 1
    else:
        stage = 2

    current_stage = stage
    freeze_canonical = True if stage == 1 else False
    # Hard stage gating (clear and explicit)  # Two-stage
    if stage == 1:
        # ========== Stage 1: 静态重建（主要监督阶段） ==========
        # 所有核心外观监督全部放在第一阶段
        w_photo = 1.0        # 主 L1
        w_ssim = 0.5         # 结构一致性
        w_lpips = 0.5       # 感知一致性，先设置为 0
        w_kp2d = 0.15
        # 使用致密光流计算的损失（仅在 batch 含 flow_fwd 时启用）
        w_flow_reproj = 0.1 if (flow_fwd is not None and flow_fwd.numel() > 0) else 0.0
        w_sil = 0.0 #前景覆盖概率损失，当前实现有问题
        #----------------------------------------------------------------------------
        w_flow_sparse = 0.0 #用 kp_{t-1} - kp_t 作为 GT flow，再约束 proj(mu_{t-1}) - proj(mu_t)，没必要
        # Retargeting: 它的最优解是：mu_t ≈ mu_{t-1}，如果你在 stage1/early stage2 给了较大 w_retarget，它会 系统性压制 dxyz_f 的幅值
        w_retarget = 0.0 
        w_retarget_ssim = 0
        w_retarget_lpips = 0
        w_alpha = 0 #鼓励大多数 Gaussian 的 alpha → 0，不可取
        w_motion = 0.0 #监督一阶运动和二阶运动几乎别动
        w_res_mag = 0.0
        w_res_smooth = 0.0
        w_chamfer = 0.0 #修改成前景高斯（注意当前是通过位移实现的）以及points_full的chamfer距离损失，实现这一块还有点问题
    else:  # Stage 2
        # ========== Stage 2: 强运动 + 精细化 ==========
        # 外观监督降权，聚焦运动/几何
        w_photo = 0.2
        w_ssim = 0.1
        w_lpips = 0.0
        w_flow_reproj = 0.5
        w_kp2d = 0.5
        #----------------------------------------------------------------------------
        # Retargeting: default on in Stage 2 (motion-focused stage)
        w_retarget = 0
        w_retarget_ssim = 0
        w_retarget_lpips = 0

        w_motion = 0.0
        w_sil = 0.0
        w_alpha = 0.0
        w_chamfer = 0.0

    # Effective retarget weights (must be decided before the streaming render loop)
    if stage == 1 and not enable_retarget_stage1:
        w_retarget = 0.0
    else:
        w_retarget = base_retarget
    w_retarget_ssim = base_retarget_ssim
    w_retarget_lpips = base_retarget_lpips
    # Effective sparse flow weight (default: only Stage 2)
    w_flow_sparse = 0.0 if stage == 1 else base_flow_sparse
    w_flow_reproj = 0.0 if stage == 1 else base_flow_reproj

    # Adjust optimizer LR for importance head at Stage 3 (0.1x)
    def _set_imp_lr(scale: float):
        try:
            for g in optimizer.param_groups:
                base_lr = g.get('base_lr', g['lr'])
                if g.get('name', '') == 'imp':
                    g['lr'] = float(base_lr) * float(scale)
                else:
                    g['lr'] = float(base_lr)
        except Exception:
            pass
    if stage == 3:
        _set_imp_lr(0.1)
    else:
        _set_imp_lr(1.0)

    # Freeze/unfreeze modules per stage
    def _set_requires_grad(module: nn.Module, enabled: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = enabled

    def project_world_to_pixel(mu_world: torch.Tensor, c2w: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mu_world: [M,3], c2w: [4,4], K: [3,3]
        if mu_world.numel() == 0:
            return torch.zeros(0, 2, device=mu_world.device, dtype=mu_world.dtype), torch.zeros(0, device=mu_world.device, dtype=mu_world.dtype)
        w2c = torch.inverse(c2w)
        xyz_h = torch.cat([mu_world, torch.ones(mu_world.shape[0], 1, device=mu_world.device, dtype=mu_world.dtype)], dim=1)  # [M,4]
        Xc = (w2c @ xyz_h.t()).t()[:, :3]
        z = Xc[:, 2].clamp(min=1e-6)
        uvw = (K @ Xc.t()).t()
        u = uvw[:, 0] / uvw[:, 2].clamp(min=1e-6)
        v = uvw[:, 1] / uvw[:, 2].clamp(min=1e-6)
        uv = torch.stack([u, v], dim=-1)
        return uv, z

    # Stage 1：训练 Gaussians & importance，冻结 motion / residual
    if stage == 1:
        _set_requires_grad(getattr(model, "dynamic_anchor_motion", None), False)
        _set_requires_grad(getattr(model, "residual_motion_head", None), False)
        _set_requires_grad(getattr(model, "motion_point_aggregator", None), False)  # 冻结 motion_block 相关模块
        _set_requires_grad(getattr(model, "motion_block", None), False)
        _set_requires_grad(getattr(model, "feat_reduce", None), True)
        _set_requires_grad(getattr(model, "dual_slot_prior", None), True)
        _set_requires_grad(getattr(model, "point_aggregator", None), True)
        _set_requires_grad(getattr(model, "dyn_pred_token", None), True)
        _set_requires_grad(getattr(model, "imp_head", None), True)
        _set_requires_grad(getattr(model, "imp_time_emb", None), True)
        _set_requires_grad(getattr(model, "imp_view_emb", None), True)
        _set_requires_grad(getattr(model, "gaussian_head", None), True)
    # Stage 2：全部解冻
    else:
        _set_requires_grad(getattr(model, "dynamic_anchor_motion", None), True)
        _set_requires_grad(getattr(model, "residual_motion_head", None), True)
        _set_requires_grad(getattr(model, "motion_point_aggregator", None), True)  # Stage 2 解冻 motion_block
        _set_requires_grad(getattr(model, "motion_block", None), True)
        _set_requires_grad(getattr(model, "feat_reduce", None), True)
        _set_requires_grad(getattr(model, "dual_slot_prior", None), True)
        _set_requires_grad(getattr(model, "point_aggregator", None), True)
        _set_requires_grad(getattr(model, "dyn_pred_token", None), True)
        _set_requires_grad(getattr(model, "imp_head", None), True)
        _set_requires_grad(getattr(model, "imp_time_emb", None), True)
        _set_requires_grad(getattr(model, "imp_view_emb", None), True)
        _set_requires_grad(getattr(model, "gaussian_head", None), True)


        
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        global_step = epoch * len(dataloader) + batch_idx
        # 更新可学习 motion_scale（线性 warm-up）
        _update_motion_scale(model, global_step, config)

        batch_data = prepare_batch(
        batch, device,
        ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
        image_size=tuple(config['data']['image_size']),
        config=config, mode='train')
        points = batch_data.get('points', None)

        # --- 检验函数 debug save of batch inputs (first batch only) ---
        # if epoch == 0 and batch_idx == 0:
        #     def _dbg_save(x: torch.Tensor | None, name: str):
        #         if x is None:
        #             return
        #         try:
        #             _save_grid_6x4(x[:6, :4], name)
        #         except Exception:
        #             pass
        #     _dbg_save(batch_data.get('conf', None), 'conf')
        #     _dbg_save(batch_data.get('conf_prob', None), 'conf_prob')
        #     _dbg_save(batch_data.get('silhouette', None), 'silhouette')
        #     _dbg_save(batch_data.get('dynamic_conf_tv', None), 'dynamic_conf_tv')
        points_3d = batch_data.get('points_3d', None)
        feat_2d = batch_data['feat_2d']
        conf_seq = batch_data['conf']
        dyn_mask_tv = batch_data.get('dynamic_conf_tv', None)
        dyn_traj_tv = batch_data.get('dynamic_traj_tv', None)
        keypoints_2d = batch_data.get('keypoints_2d', None) #[6, 4, 176, 2]
        flow_fwd = batch_data.get('flow_fwd', None)
        silhouette = batch_data.get('silhouette', None)
        camera_poses_seq = batch_data['camera_poses']
        camera_intrinsics_seq = batch_data['camera_intrinsics']
        time_ids = batch_data['time_ids']
        gt_images_raw = batch_data.get('gt_images', None)
        gt_images = gt_images_raw.permute(0, 1, 4, 2, 3).contiguous() if gt_images_raw is not None else None
        T = int(time_ids.shape[0])

        optimizer.zero_grad(set_to_none=True)
        # torch.cuda.amp.autocast 已弃用，改用 torch.amp.autocast('cuda', ...)
        use_amp_ctx = (torch.amp.autocast('cuda') if scaler.is_enabled() else nullcontext())
        with use_amp_ctx:
            dyn_mask_in = dyn_mask_tv if dyn_mask_tv is not None else silhouette
            output = model(
                points_full=points,    # [T,V,H,W,3] (preferred for slot_dual tokenization)
                # points_3d=points_3d,   # [T,N,3] (optional; AABB/debug)
                feat_2d=feat_2d,       # [T,V,H'=,W'=,C]
                conf_2d=conf_seq,
                camera_poses=camera_poses_seq,
                camera_K=camera_intrinsics_seq,
                time_ids=time_ids,
                dyn_mask_2d=dyn_mask_in,
                build_canonical=False,
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
        dxyz_t = output.get('dxyz_t', None)
        dxyz_t_res = output.get("dxyz_t_res", None)
        velocity_high = output.get("velocity_high", None)  # [T-1, M, 3] velocity from motion_block
        sim3_s, sim3_R, sim3_t = output.get('sim3_s', None), output.get('sim3_R', None), output.get('sim3_t', None)
        dtype = mu_t.dtype
        M = mu_t.shape[1]

        H_t, W_t = config['data']['image_size']
        
        # 提前计算 w_lpips 用于循环中的条件检查
        # 注意：Stage 1 会硬编码 w_lpips = 0.5，Stage 2 使用配置值（通常为 0.0）
        lw_raw = config.get('loss_weights', {})
        def _sched(name: str, base: float):
            stage_cfg = config.get('schedule', {})
            after = int(stage_cfg.get(f"{name}_after", 0))
            return 0.0 if epoch < after else base
        lw = {k: _sched(k, float(v)) for k,v in lw_raw.items()}
        w_lpips_check = 0.5 if stage == 1 else float(lw.get('lpips', 0.0))
        
        # 流式渲染累加 loss：不再堆叠保存 [T,V,3,H,W]（显著降低显存峰值）
        photo_loss_acc = torch.tensor(0.0, device=device)
        ssim_loss_acc = torch.tensor(0.0, device=device)
        psnr_acc = torch.tensor(0.0, device=device)
        lpips_loss_acc = torch.tensor(0.0, device=device)
        lpips_count = 0
        retarget_loss_acc = torch.tensor(0.0, device=device)
        retarget_count = 0
        n_render = 0

        # 颜色初始化：第 1 个 epoch（epoch==0）使用 target_image 估计每个 Gaussian 的 DC 颜色（参考 test_render.py）
        # 注意：这里“写进模型”的方式是把它作为 epoch0 的监督信号，让 gaussian_head 学到合适的初始颜色；
        #      之后颜色仍由网络输出并参与更新（不会在后续 epoch 被固定覆盖）。
        # 默认禁用颜色初始化以避免 CUDA 错误，如需启用请在配置中设置 color_init_first_epoch: true
        do_color_init = bool(config.get('training', {}).get('color_init_first_epoch', False)) and (epoch == 0)
        debug_color_init = bool(config.get('training', {}).get('debug_color_init', False))
        # 颜色初始化策略：不是监督，而是“初始化引导渲染”，并逐步退火到网络自身输出
        # blend(t) 从 color_init_blend_start 线性衰减到 color_init_blend_end（按 step 计算）
        blend_start = float(config.get('training', {}).get('color_init_blend_start', 1.0))
        blend_end = float(config.get('training', {}).get('color_init_blend_end', 0.0))
        blend_steps = int(config.get('training', {}).get('color_init_blend_steps', 2000))
        # 当前 epoch 的全局 step（仅用于 blend 调度，不影响 optimizer step）
        global_step = epoch * len(dataloader) + batch_idx

        @torch.no_grad()
        def estimate_init_sh0_from_targets(
            mu_frame,
            scale_frame,
            rot_frame,
            color_frame_pred,
            alpha_frame,
            camera_poses_t,
            camera_intrinsics_t,
            target_images_t,  # [V,3,H,W]
        ):
            if target_images_t is None or mu_frame.numel() == 0:
                return None, None
            
            # # 安全检查：如果 Gaussian 数量过多或参数异常，跳过颜色初始化
            # num_gs = int(mu_frame.shape[0])
            # if num_gs > 50000:  # 超过 5 万个 Gaussian 时跳过
            #     print(f"[color_init] Too many Gaussians ({num_gs}), skipping color init")
            #     return None, None
            
            # 检查是否有异常值
            if torch.isnan(mu_frame).any() or torch.isinf(mu_frame).any():
                print(f"[color_init] NaN/Inf detected in mu_frame, skipping color init")
                return None, None
            if torch.isnan(scale_frame).any() or torch.isinf(scale_frame).any():
                print(f"[color_init] NaN/Inf detected in scale_frame, skipping color init")
                return None, None

            bg_color = torch.ones(3, device=device)
            num_gs = int(mu_frame.shape[0])
            C0 = 0.28209479177387814
            # 在 fast_forward init pass 中，render_gs CUDA kernel 一旦报错通常会延迟到后续同步点才抛出。
            # 这里提供一个开关，强制每次 render_gs 后同步，便于定位且避免异步错误污染后续计算。
            color_init_sync_cuda = bool(config.get("training", {}).get("color_init_sync_cuda", True))
            # diff_gaussian_rasterization 在某些路径（尤其带 target_image）对 N 过大非常不稳定，可能触发 illegal memory access。
            # 这里对 init pass 做安全上限：只在一个子集上做 fast_forward，其余点保持网络预测颜色。
            max_init_gs = int(config.get("training", {}).get("color_init_max_gaussians", 5000))
            # 进一步降低风险：颜色初始化只用少量视角（test_gs_render.py 用 per-view 聚合，但训练阶段可用 1~2 个视角即可）
            max_init_views = int(config.get("training", {}).get("color_init_max_views", 1))

            # sanitize inputs to avoid rasterizer undefined behavior on NaN/Inf
            mu_frame = torch.nan_to_num(mu_frame, nan=0.0, posinf=0.0, neginf=0.0)
            alpha_frame = torch.nan_to_num(alpha_frame, nan=0.0, posinf=0.0, neginf=0.0)
            scale_frame = torch.nan_to_num(scale_frame, nan=0.0, posinf=0.0, neginf=0.0)
            color_frame_pred = torch.nan_to_num(color_frame_pred, nan=0.0, posinf=0.0, neginf=0.0)

            opacity = alpha_frame.squeeze(-1) if alpha_frame.dim() > 1 else alpha_frame
            opacity = opacity.to(dtype=mu_frame.dtype).clamp(0.0, 1.0).contiguous()
            # 仿照 test_gs_render.py：init pass 临时提高可见性（更大的 scale、更高的 alpha），只为拿到可靠的 est_color/est_weight
            init_scale_mode = str(config.get('training', {}).get('color_init_scale_mode', 'pixel')).lower()
            init_min_scale = float(config.get('training', {}).get('color_init_min_scale', 1e-3))
            init_max_scale = float(config.get('training', {}).get('color_init_max_scale', 0.05))
            init_min_alpha = float(config.get('training', {}).get('color_init_min_alpha', 0.8))

            scale_frame = scale_frame.to(dtype=mu_frame.dtype)
            scale_init = scale_frame.clamp(min=init_min_scale, max=init_max_scale)
            opacity_init = opacity.clamp(min=init_min_alpha)

            if rot_frame is None:
                rotation = torch.zeros(num_gs, 4, device=mu_frame.device, dtype=mu_frame.dtype)
                rotation[:, 0] = 1.0
            else:
                rotation = matrix_to_quaternion(rot_frame).to(device=mu_frame.device, dtype=mu_frame.dtype)
                rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rotation = torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

            # 用当前网络预测颜色作为初始 SH 输入，rasterizer 会返回 est_color / est_weight
            color_frame_pred = color_frame_pred.to(dtype=mu_frame.dtype).clamp(0.0, 1.0)
            sh0_pred_full = rgb2sh0(color_frame_pred).to(dtype=torch.float32).contiguous()  # [M,3] (DC)

            # Optional sub-sampling for init pass stability
            if num_gs > max_init_gs > 0:
                sub_idx = torch.randperm(num_gs, device=mu_frame.device)[:max_init_gs]
            else:
                sub_idx = None

            if sub_idx is not None:
                mu_use = mu_frame[sub_idx]
                opacity_use = opacity_init[sub_idx]
                scale_use = scale_init[sub_idx]
                rot_use = rotation[sub_idx]
                sh0_use = sh0_pred_full[sub_idx].contiguous()
            else:
                mu_use = mu_frame
                opacity_use = opacity_init
                scale_use = scale_init
                rot_use = rotation
                sh0_use = sh0_pred_full

            sh_in = sh0_use.to(device=mu_frame.device, dtype=mu_frame.dtype).unsqueeze(1)  # [N,1,3]

            # collect per-view estimates like test_gs_render.py, then do fast_forward on sh0_pred
            est_color_list = []
            est_weight_list = []

            for vi in range(min(int(camera_poses_t.shape[0]), max(1, max_init_views))):
                c2w = camera_poses_t[vi].detach().cpu().numpy()
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3].astype(np.float32)
                t_vec = w2c[:3, 3].astype(np.float32)
                K_np = camera_intrinsics_t[vi].detach().cpu().numpy().astype(np.float32)

                cam = IntrinsicsCamera(
                    K=K_np, R=R, T=t_vec,
                    width=int(W_t), height=int(H_t),
                    znear=0.01, zfar=100.0,
                )

                gs_attrs = GaussianAttributes(
                    xyz=mu_use,
                    opacity=opacity_use,
                    scaling=scale_use,
                    rotation=rot_use,
                    sh=sh_in,
                )

                try:
                    res_v = render_gs(
                        camera=cam,
                        bg_color=bg_color,
                        gs=gs_attrs,
                        # 注意：本项目的 render_gs 期望单张图像 [3,H,W]（非 batch），否则可能触发 CUDA kernel 非法访问
                        target_image=torch.nan_to_num(
                            target_images_t[vi].to(device=mu_frame.device, dtype=mu_frame.dtype),
                            nan=0.0, posinf=0.0, neginf=0.0,
                        ).clamp(0.0, 1.0).contiguous(),
                        sh_degree=0,
                        scaling_modifier=1.0,
                    )
                    if mu_frame.is_cuda and (color_init_sync_cuda or debug_color_init):
                        torch.cuda.synchronize(device=mu_frame.device)
                except RuntimeError as e:
                    if "CUDA" in str(e) or "illegal memory" in str(e):
                        # CUDA 错误：跳过颜色初始化，使用网络预测颜色
                        print(f"[color_init] CUDA error in render_gs (view {vi}), skipping color init: {e}")
                        return None, None
                    else:
                        raise
                est_c = res_v.get("est_color", None)
                est_w = res_v.get("est_weight", None)
                if est_c is None or est_w is None:
                    continue
                # est_color 可能是 [M,3] 或 [M,1,3]（SH0 / RGB），统一到 [M,3]
                if est_c.dim() == 3 and est_c.shape[1] == 1:
                    est_c = est_c[:, 0, :]
                elif est_c.dim() != 2 or est_c.shape[-1] != 3:
                    continue
                # est_weight 可能是 [M] 或 [M,1]
                if est_w.dim() == 1:
                    est_w = est_w.unsqueeze(-1)  # [M,1]

                est_color_list.append(est_c.to(dtype=torch.float32))
                est_weight_list.append(est_w.to(dtype=torch.float32))

            if len(est_color_list) == 0 or len(est_weight_list) == 0:
                return None, None

            est_color_avg = torch.stack(est_color_list, dim=0).mean(dim=0).contiguous()   # [M,3]
            est_weight_avg = torch.stack(est_weight_list, dim=0).mean(dim=0).contiguous() # [M,1]

            if float(est_weight_avg.mean().item()) < 1e-6:
                if debug_color_init and (epoch == 0 and batch_idx == 0):
                    print(f"[color_init] est_weight_avg.mean() too small: {float(est_weight_avg.mean().item()):.4g}")
                return None, None

            # ---- fast_forward (like SimpleGaussianModel.fast_forward) ----
            # Updates sh0_pred in-place using est_color/est_weight. This is not a supervision loss; it produces a better init color.
            try:
                from diff_gaussian_rasterization import fast_forward as cuda_fast_forward  # type: ignore
            except Exception:
                cuda_fast_forward = None

            if cuda_fast_forward is not None:
                thr = float(config.get("training", {}).get("color_init_ff_weight_threshold", 0.05))
                with torch.no_grad():
                    cuda_fast_forward(
                        thr,
                        est_color_avg.contiguous(),
                        est_weight_avg[:, 0].clamp_min(0).contiguous(),
                        torch.zeros(sh0_use.shape[0], device=sh0_use.device, dtype=torch.bool),
                        sh0_use,  # [N,3] in-place
                    )
            else:
                # Fallback: weighted average in SH0 space (less robust than cuda fast_forward)
                sh0_use = (est_color_avg / est_weight_avg.clamp_min(1e-8)).contiguous()

            # write back to full tensor if subsampled
            if sub_idx is not None:
                sh0_pred_full[sub_idx] = sh0_use
                est_w_full = torch.zeros(num_gs, 1, device=mu_frame.device, dtype=mu_frame.dtype)
                est_w_full[sub_idx] = est_weight_avg.to(device=mu_frame.device, dtype=mu_frame.dtype)
                return sh0_pred_full.to(device=mu_frame.device, dtype=mu_frame.dtype), est_w_full
            else:
                return sh0_use.to(device=mu_frame.device, dtype=mu_frame.dtype), est_weight_avg.to(device=mu_frame.device, dtype=mu_frame.dtype)

        def render_set(mu_frame, scale_frame, rot_frame, color_frame, alpha_frame, camera_poses_t, camera_intrinsics_t, target_images_t=None, view_ids: Optional[list[int]] = None):
            # 统一策略：模型 forward 用 AMP，但渲染端强制 fp32（避免 autocast 在 render 内反复 promote/cast 造成显存波动）
            # torch.cuda.amp.autocast 已弃用，改用 torch.amp.autocast('cuda', enabled=...)
            with torch.amp.autocast('cuda', enabled=False):
                bg_color = torch.ones(3, device=device, dtype=torch.float32)
                max_scale = float(config.get('model', {}).get('max_scale', 0.05))
                # 与 inference 对齐：scale 仅限幅，不做像素半径校准
                scale_frame = scale_frame.to(dtype=mu_frame.dtype).clamp(min=1e-6, max=max_scale)

                init_sh0 = None
                init_weight = None
                if do_color_init and (target_images_t is not None):
                    init_sh0, init_weight = estimate_init_sh0_from_targets(
                        mu_frame=mu_frame,
                        scale_frame=scale_frame,
                        rot_frame=rot_frame,
                        color_frame_pred=color_frame,
                        alpha_frame=alpha_frame,
                        camera_poses_t=camera_poses_t,
                        camera_intrinsics_t=camera_intrinsics_t,
                        target_images_t=target_images_t,
                    )

                # 渲染颜色采用“退火混合”：
                #   sh_render = (1-β)*sh_pred + β*sh_init
                # 这样 epoch0 初期颜色接近 fast_forward 结果，同时梯度仍回传到网络颜色分支。
                # ---- [ADD] sanitize inputs for rasterizer stability ----
                mu_frame = torch.nan_to_num(mu_frame, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
                scale_frame = torch.nan_to_num(scale_frame, nan=1e-3, posinf=1e-3, neginf=1e-3).contiguous()
                alpha_frame = torch.nan_to_num(alpha_frame, nan=0.0, posinf=1.0, neginf=0.0).contiguous()
                if rot_frame is not None:
                    rot_frame = torch.nan_to_num(rot_frame, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

                # 强烈建议：render 端统一用 float32（很多 CUDA rasterizer 对 fp16 不稳）
                mu_frame = mu_frame.float()
                scale_frame = scale_frame.float()
                alpha_frame = alpha_frame.float()

                safe_color = torch.nan_to_num(color_frame, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                safe_color = safe_color.to(device=mu_frame.device, dtype=torch.float32).contiguous()
                sh_pred = rgb2sh0(safe_color).to(dtype=torch.float32).contiguous()  # [M,3]

                if do_color_init and init_sh0 is not None and blend_steps > 0:
                    # 线性退火
                    t = float(min(max(global_step, 0), blend_steps)) / float(blend_steps)
                    beta = (1.0 - t) * blend_start + t * blend_end
                    beta = float(max(0.0, min(1.0, beta)))
                    sh_render = (1.0 - beta) * sh_pred + beta * init_sh0.detach()
                elif do_color_init and init_sh0 is not None and blend_steps <= 0:
                    sh_render = init_sh0.detach()
                else:
                    sh_render = sh_pred
                sh_for_render = sh_render.to(dtype=torch.float32).contiguous().unsqueeze(1)  # [M,1,3]
                # 注意：默认不要对 render_gs 使用 no_grad（否则 photo/ssim 等监督无法回传到高斯参数）
                # 如需紧急排查 OOM，可在 config.training.render_no_grad=true 临时启用（会关闭渲染梯度）。
                render_no_grad = bool(config.get("training", {}).get("render_no_grad", False))
                if view_ids is None:
                    view_ids = list(range(camera_poses_t.shape[0]))
                for vi in view_ids:
                    c2w = camera_poses_t[vi].detach().cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    R = w2c[:3, :3].astype(np.float32)
                    t_vec = w2c[:3, 3].astype(np.float32)
                    K_np = camera_intrinsics_t[vi].detach().cpu().numpy().astype(np.float32)
                    cam = IntrinsicsCamera(
                        K=K_np, R=R, T=t_vec,
                        width=int(W_t), height=int(H_t),
                        znear=0.01, zfar=100.0,
                    )
                    opacity = alpha_frame.squeeze(-1) if alpha_frame.dim() > 1 else alpha_frame
                    opacity = opacity.to(dtype=mu_frame.dtype).clamp(0.0, 1.0)
                    num_gs = mu_frame.shape[0]
                    if rot_frame is None:
                        rotation = torch.zeros(num_gs, 4, device=mu_frame.device, dtype=mu_frame.dtype)
                        rotation[:, 0] = 1.0
                    else:
                        rotation = matrix_to_quaternion(rot_frame).to(device=mu_frame.device, dtype=mu_frame.dtype)
                        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    gs_attrs = GaussianAttributes(
                        xyz=mu_frame,
                        opacity=opacity,
                        scaling=scale_frame,
                        rotation=rotation,
                        sh=sh_for_render,
                    )
                    with (torch.no_grad() if render_no_grad else nullcontext()):
                        res_v = render_gs(camera=cam, bg_color=bg_color, gs=gs_attrs, target_image=None, sh_degree=0, scaling_modifier=1.0)
                    color = res_v["color"]
                    alpha = res_v.get("alpha", None)
                    del res_v
                    yield vi, color, alpha  # color:[3,H,W], alpha:[1,H,W] or [H,W]
                return

        # 流式渲染 + 直接累加 photo/ssim（不保存全量 rendered_images）
        # 注意：masked dyn/static/cross 仍保持关闭（stage2_weights 默认 0）；如需启用需单独做流式实现。
        # 可视化：保存一个小子集（用于 epoch_images），不是用于指标统计
        viz_cfg = config.get("training", {}).get("viz", {}) or {}
        viz_every = int(viz_cfg.get("every", 10))
        viz_only_one = bool(viz_cfg.get("only_one", True))
        do_viz = (batch_idx == 0) and (viz_every > 0) and ((epoch % viz_every) == 0)
        viz_T = int(viz_cfg.get("times", 4))
        viz_V = int(viz_cfg.get("views", 4))
        if not do_viz:
            viz_T, viz_V = 0, 0
        elif viz_only_one:
            viz_T, viz_V = 1, 1
        viz_pred = [[None for _ in range(viz_V)] for _ in range(viz_T)]
        viz_gt = [[None for _ in range(viz_V)] for _ in range(viz_T)]

        # ---- 新增：保存完整 T*V 视角的全部预测 / GT ----
        n_views_total = camera_poses_seq.shape[1]
        full_pred = [[None for _ in range(n_views_total)] for _ in range(T)]
        full_gt = [[None for _ in range(n_views_total)] for _ in range(T)] if gt_images is not None else None

        last_rendered_images = None
        # 如果需要计算 silhouette loss，需要保存渲染出的 alpha （一次性，float32，设备同预测）
        # Silhouette alpha collection (Stage1 requires it)
        need_alpha = (stage == 1)  # Stage1 uses silhouette loss
        full_alpha = [[None for _ in range(n_views_total)] for _ in range(T)] if need_alpha else None
        for t in range(T):
            camera_poses_t = camera_poses_seq[t]
            camera_intrinsics_t = camera_intrinsics_seq[t]
            rot_frame = rot_t[t] if rot_t is not None else None
            tgt_tv = gt_images[t] if (gt_images is not None and do_color_init) else None
            # view subsample for retargeting (to control compute/memory)
            V = int(camera_poses_t.shape[0])
            if ret_view_sub > 0 and V > ret_view_sub:
                view_ids_retarget = torch.randperm(V, device=device)[:ret_view_sub].tolist()
            else:
                view_ids_retarget = list(range(V))
            for vi, pred_chw, alpha_hw in render_set(
                mu_t[t], scale_t[t], rot_frame, color_t[t], alpha_t[t],
                camera_poses_t, camera_intrinsics_t,
                target_images_t=tgt_tv,
            ):
                # 收集 alpha 用于 silhouette loss（保持在 GPU 上以便反传）
                if need_alpha and (alpha_hw is not None):
                    a = alpha_hw
                    if a.dim() == 3 and a.shape[0] == 1:
                        a = a.squeeze(0)
                    full_alpha[t][vi] = a.float()

                if gt_images is not None:
                    gt_chw = gt_images[t, vi].to(device=device, dtype=pred_chw.dtype)
                    # 保存完整 T*V（放 CPU，避免显存爆）
                    full_pred[t][vi] = pred_chw.detach().cpu()
                    if full_gt is not None:
                        full_gt[t][vi] = gt_chw.detach().cpu()
                    photo_loss_acc = photo_loss_acc + torch.nn.functional.l1_loss(pred_chw, gt_chw, reduction='mean')
                    if float(config.get('loss_weights', {}).get('ssim', 0.0)) > 0:
                        ssim_loss_acc = ssim_loss_acc + (1.0 - ssim_torch(
                            pred_chw.unsqueeze(0).clamp(0, 1),
                            gt_chw.unsqueeze(0).clamp(0, 1),
                            window_size=11,
                        ))
                    # LPIPS 计算（内存优化版）
                    if (
                        lpips_fn is not None
                        and w_lpips_check > 0
                        and t == 0
                        and vi == 0
                    ):
                        pred_nchw = pred_chw.unsqueeze(0) * 2 - 1  # [1,3,H,W] 归一化到 [-1,1]
                        gt_nchw = gt_chw.unsqueeze(0) * 2 - 1      # [1,3,H,W] 归一化到 [-1,1]
                        with torch.no_grad():  # 关键：LPIPS不需要梯度
                            try:
                                lp = lpips_fn(pred_nchw, gt_nchw).mean()
                                lpips_loss_acc = lpips_loss_acc + lp
                                lpips_count += 1
                            except torch.cuda.OutOfMemoryError:
                                print("OOM in LPIPS, skipping for this item.")
                    mse = torch.mean((pred_chw - gt_chw) ** 2).clamp_min(1e-8)
                    psnr_acc = psnr_acc + (10.0 * torch.log10(1.0 / mse))
                    if t < viz_T and vi < viz_V:
                        viz_pred[t][vi] = pred_chw.detach().cpu()
                        viz_gt[t][vi] = gt_chw.detach().cpu()
                n_render += 1
                del pred_chw

        # Retargeting loss (GT-based): render frame-t Gaussians under (t-1) cameras and match GT images at (t-1).
        # This provides a real supervisory signal for motion, instead of render-vs-render self-consistency.
        if w_retarget > 0 and t >= 1 and len(view_ids_retarget) > 0 and gt_images is not None:
            # Use cameras from frame (t-1)
            camera_poses_ref = camera_poses_seq[t - 1]
            camera_intrinsics_ref = camera_intrinsics_seq[t - 1]

            # "Warp" strategy:
            # Option A (recommended first): no explicit warp; use mu_t[t] directly.
            # This forces frame-t geometry/motion to explain frame-(t-1) images under (t-1) cameras.
            mu_warp = mu_t[t]
            scale_warp = scale_t[t]
            color_warp = color_t[t]
            alpha_warp = alpha_t[t]
            rot_warp = rot_frame  # keep consistent with your current rot choice

            for vi in view_ids_retarget:
                # GT image at t-1, view vi
                # gt_images 已经是 [T,V,3,H,W] 格式，所以 gt_images[t-1, vi] 是 [3,H,W]
                gt_ref_chw = gt_images[t - 1, vi].to(device=device, dtype=mu_warp.dtype)  # [3,H,W]

                warp_it = render_set(
                    mu_warp, scale_warp, rot_warp, color_warp, alpha_warp,
                    camera_poses_ref, camera_intrinsics_ref,
                    target_images_t=None,
                    view_ids=[vi],
                )
                try:
                    _, warp_chw, _ = next(warp_it)  # [3,H,W] - render_set returns (vi, color, alpha)
                except StopIteration:
                    continue

                # L2 / MSE
                l2 = torch.mean((warp_chw - gt_ref_chw) ** 2)
                ret = l2

                # SSIM loss
                if w_retarget_ssim > 0:
                    ret = ret + w_retarget_ssim * (1.0 - ssim_torch(
                        warp_chw.unsqueeze(0).clamp(0, 1),
                        gt_ref_chw.unsqueeze(0).clamp(0, 1),
                        window_size=11,
                    ))

                # LPIPS: DO NOT use no_grad if you want it to affect training.
                # Freeze LPIPS net params elsewhere (lpips_fn.eval(); requires_grad_(False)),
                # but allow gradients to flow to warp_chw.
                if lpips_fn is not None and w_retarget_lpips > 0:
                    warp_nchw = warp_chw.unsqueeze(0) * 2 - 1
                    gt_nchw = gt_ref_chw.unsqueeze(0) * 2 - 1
                    ret = ret + w_retarget_lpips * lpips_fn(warp_nchw, gt_nchw).mean()

                retarget_loss_acc = retarget_loss_acc + ret
                retarget_count += 1
                del warp_chw, gt_ref_chw


        # ------ 保存完整 T*V 视角的所有预测 / GT ------
        if full_pred[0][0] is not None:
            pred_stack = []
            gt_stack = [] if full_gt is not None else None
            for tt in range(T):
                row_p = []
                row_g = [] if gt_stack is not None else None
                for vv in range(n_views_total):
                    p = full_pred[tt][vv]
                    if p is None:
                        p = torch.zeros(3, H_t, W_t)
                    row_p.append(p)
                    if gt_stack is not None:
                        g = full_gt[tt][vv]
                        if g is None:
                            g = torch.zeros(3, H_t, W_t)
                        row_g.append(g)
                pred_stack.append(torch.stack(row_p, dim=0))
                if gt_stack is not None:
                    gt_stack.append(torch.stack(row_g, dim=0))
            last_rendered_images = torch.stack(pred_stack, dim=0)
            if gt_stack is not None:
                last_gt_images = torch.stack(gt_stack, dim=0)
        else:
            # 将可视化子集整理为 [T,V,3,H,W]  (保持早期逻辑不变)
            if gt_images is not None and viz_T > 0 and viz_V > 0 and viz_pred[0][0] is not None:
                pred_stack = []
                gt_stack = []
                for tt in range(viz_T):
                    row_p = []
                    row_g = []
                    for vv in range(viz_V):
                        p = viz_pred[tt][vv]
                        g = viz_gt[tt][vv]
                        if p is None:
                            p = torch.zeros(3, H_t, W_t)
                        if g is None:
                            g = torch.zeros(3, H_t, W_t)
                        row_p.append(p)
                        row_g.append(g)
                    pred_stack.append(torch.stack(row_p, dim=0))
                    gt_stack.append(torch.stack(row_g, dim=0))
                last_rendered_images = torch.stack(pred_stack, dim=0)
                last_gt_images = torch.stack(gt_stack, dim=0)


        # Loss weights
        lw_raw = config.get('loss_weights', {})
        # schedule weights by epoch
        def _sched(name: str, base: float):
            stage_cfg = config.get('schedule', {})
            after = int(stage_cfg.get(f"{name}_after", 0))
            return 0.0 if epoch < after else base
        lw = {k: _sched(k, float(v)) for k,v in lw_raw.items()}
        w_photo = float(lw.get('photo_l1', 1.0))
        w_ssim = float(lw.get('ssim', 0.0))
        w_lpips = float(lw.get('lpips', 0.0))
        w_sil = float(lw.get('silhouette', 0.0))
        w_alpha = float(lw.get('alpha_l1', 0.0))
        w_chamfer = float(lw.get('chamfer', 0.0))
        w_motion = float(lw.get('motion_reg', 0.0))
        w_vel_high_smooth = float(lw.get('velocity_high_smooth', 0.0))  # velocity smoothness for motion_block
        w_kp2d = 0.0  # default, will override per stage
        w_temp_vel = float(lw.get('temporal_vel', 0.0))
        w_temp_acc = float(lw.get('temporal_acc', 0.0))
        w_cycle = float(lw.get('cycle_consistency', 0.0))

        w_dyn = float(lw.get('mask_dyn', 0.0))
        w_sta = float(lw.get('mask_sta', 0.0))
        w_cross = float(lw.get('mask_cross', 0.0))
        # Stage C: differentiable importance selector regularization (默认 0，不影响现有训练)
        w_imp_budget = float(lw.get("imp_budget", 0.0))
        w_imp_entropy = float(lw.get("imp_entropy", 0.0))
        w_imp_repel = float(lw.get("imp_repel", 0.0))




        if gt_images is not None and n_render > 0:
            photo_loss = photo_loss_acc / float(n_render)
        else:
            photo_loss = torch.tensor(0.0, device=device)

        # Mask-decomposed losses：为避免 OOM，流式版本需另行实现；当前保持关闭（stage2_weights 默认为 0）
        mask_dyn_loss = torch.tensor(0.0, device=device)
        mask_sta_loss = torch.tensor(0.0, device=device)
        mask_cross_loss = torch.tensor(0.0, device=device)

        if gt_images is not None and w_ssim > 0 and n_render > 0:
            ssim_loss = ssim_loss_acc / float(n_render)
        else:
            ssim_loss = torch.tensor(0.0, device=device)

        # LPIPS loss
        if lpips_fn is not None and gt_images is not None and lpips_count > 0:
            lpips_loss = lpips_loss_acc / float(lpips_count)
        else:
            lpips_loss = torch.tensor(0.0, device=device)

        # Retargeting loss (Forge4D-style)
        if retarget_count > 0:
            retarget_loss = retarget_loss_acc / float(retarget_count)
        else:
            retarget_loss = torch.tensor(0.0, device=device)

        psnr_val = (psnr_acc / float(n_render)).detach() if (gt_images is not None and n_render > 0) else torch.tensor(0.0, device=device)

        # Silhouette Loss (streaming, memory-safe)
        silhouette = batch_data.get('silhouette', None)
        sil_loss = torch.zeros((), device=device, dtype=torch.float32)

        if silhouette is not None and w_sil > 0:
            # full_alpha: list[T][V] each is alpha [H,W] or [H,W,1] (float)
            # If alpha was not collected, we cannot supervise silhouette.
            if full_alpha is None:
                # No rendered alpha available -> skip supervision
                full_alpha = [[None for _ in range(n_views_total)] for _ in range(T)]

            sil = silhouette

            # --- normalize sil shape ---
            # allow [T,V,H,W] or [T,V,H,W,1]
            while sil.dim() > 4 and sil.shape[-1] == 1:
                sil = sil.squeeze(-1)
            if sil.dim() != 4:
                raise ValueError(
                    f"silhouette must be [T,V,H,W] or [T,V,H,W,1], got {tuple(sil.shape)}"
                )
            sil = sil.float().clamp(0.0, 1.0)

            # --- optional morphology config (applied per-frame after resize to alpha grid) ---
            morph_cfg = config.get('loss_params', {}).get('silhouette_morph', {})
            op = str(morph_cfg.get('op', 'none')).lower()
            k = int(morph_cfg.get('k', 3))
            if k % 2 == 0:
                k += 1

            sil_loss_acc = torch.zeros((), device=device, dtype=torch.float32)
            sil_count = 0

            # Stream over (t, v): compute loss using per-view alpha without stacking [T,V,H,W]
            for tt in range(T):
                for vv in range(n_views_total):
                    alpha_tv = full_alpha[tt][vv]
                    if alpha_tv is None:
                        # Do NOT treat missing alpha as zeros; just skip
                        continue

                    # alpha_tv should be [H,W] or [H,W,1]
                    ra = alpha_tv
                    if ra.dim() == 3 and ra.shape[-1] == 1:
                        ra = ra.squeeze(-1)
                    if ra.dim() != 2:
                        raise ValueError(
                            f"rendered alpha at (t={tt}, v={vv}) must be [H,W] or [H,W,1], got {tuple(alpha_tv.shape)}"
                        )
                    ra = ra.float().clamp(0.0, 1.0)
                    Hr, Wr = ra.shape[-2], ra.shape[-1]

                    # take GT silhouette for (t,v): [H,W]
                    sil_tv = sil[tt, vv]  # [Hs,Ws]
                    if sil_tv.dim() != 2:
                        raise ValueError(
                            f"silhouette slice at (t={tt}, v={vv}) must be [H,W], got {tuple(sil_tv.shape)}"
                        )

                    # --- align resolution BEFORE morphology (resize GT to rendered alpha grid) ---
                    if sil_tv.shape[-2:] != (Hr, Wr):
                        sil_b = sil_tv[None, None, ...]  # [1,1,Hs,Ws]
                        sil_b = F.interpolate(sil_b, size=(Hr, Wr), mode='nearest')
                        sil_tv = sil_b[0, 0]

                    sil_tv = sil_tv.float().clamp(0.0, 1.0)

                    # --- optional morphology on aligned grid (applied to GT only) ---
                    if op in ['dilate', 'erode'] and k >= 1:
                        sil_b = sil_tv[None, None, ...]  # [1,1,H,W]
                        pad = k // 2
                        if op == 'dilate':
                            sil_b = F.max_pool2d(sil_b, kernel_size=k, stride=1, padding=pad)
                        else:
                            sil_b = 1.0 - F.max_pool2d(1.0 - sil_b, kernel_size=k, stride=1, padding=pad)
                        sil_tv = sil_b[0, 0].clamp(0.0, 1.0)

                    # --- compute loss in float32 for stability ---
                    sil_loss_acc = sil_loss_acc + (ra - sil_tv).abs().mean()
                    sil_count += 1

            if sil_count > 0:
                sil_loss = sil_loss_acc / float(sil_count)
            else:
                # no valid alpha was collected => no supervision
                sil_loss = torch.zeros((), device=device, dtype=torch.float32)



        # Alpha sparsity (per-Gaussian)
        alpha_sparse = torch.abs(alpha_t).mean() if w_alpha > 0 else torch.tensor(0.0, device=device)

        # Motion regularization using SegAnyMo dynamic confidence (lower conf -> stronger penalty)
        motion_reg = torch.tensor(0.0, device=device)
        if w_motion > 0 and dxyz_t is not None and dxyz_t.shape[0] >= 3:
            # velocity & acceleration smoothness (L1)
            vel = (dxyz_t[1:] - dxyz_t[:-1]).abs().mean(dim=2)   # [T-1,M]
            acc = (dxyz_t[2:] - 2*dxyz_t[1:-1] + dxyz_t[:-2]).abs().mean(dim=2)  # [T-2,M]
            vel_loss = vel.mean()
            acc_loss = acc.mean()
            motion_reg = 0.5 * vel_loss + 0.5 * acc_loss    #一阶二阶平滑

        # Velocity smoothness loss for motion_block (L_vel_smooth = |v_t - v_{t-1}|.mean())
        # 只作用在 motion_block 分支的 velocity，避免高频 jitter
        vel_high_smooth = torch.tensor(0.0, device=device)
        if w_vel_high_smooth > 0 and velocity_high is not None and velocity_high.shape[0] >= 2:
            # velocity_high: [T-1, M, 3]
            # 计算相邻时刻 velocity 的差异: |v_t - v_{t-1}|
            vel_diff = torch.abs(velocity_high[1:] - velocity_high[:-1])  # [T-2, M, 3]
            vel_high_smooth = vel_diff.mean()  # L1 smoothness

        # # Stage 3: residual motion regularization (stronger than anchor motion)
        # w_res_mag = float(stage_cfg.get("stage3_residual_mag", 0.0)) if stage == 3 else 0.0
        # w_res_smooth = float(stage_cfg.get("stage3_residual_smooth", 0.0)) if stage == 3 else 0.0
        # res_mag = torch.tensor(0.0, device=device)
        # res_smooth = torch.tensor(0.0, device=device)
        # if stage == 3 and dxyz_t_res is not None:
        #     # magnitude
        #     if w_res_mag > 0:
        #         res_mag = dxyz_t_res.pow(2).sum(dim=-1).mean()
        #     # first-order temporal smoothness on residual
        #     if w_res_smooth > 0 and dxyz_t_res.shape[0] >= 2:
        #         res_smooth = torch.abs(dxyz_t_res[1:] - dxyz_t_res[:-1]).mean()



        # Simple/Weighted Chamfer (per frame, using points_3d transformed by Sim3)
        # --------------------------------------------------------
        # Displacement Chamfer Loss (Stage2 / Stage3)
        # Chamfer( mu_t - mu_0 , X_t - X_0 )
        # --------------------------------------------------------

        chamfer_loss = torch.tensor(0.0, device=device)

        # ========= 配置 =========
        dyn_thresh = config.get('loss_params', {}).get('chamfer', {}).get('dyn_thresh', 0.5)
        min_dyn_pts = config.get('loss_params', {}).get('chamfer', {}).get('min_dyn_pts', 128)
        K_ds = config.get('loss_params', {}).get('chamfer', {}).get('max_pairs', 2048)
        use_no_grad = config.get('loss_params', {}).get('chamfer', {}).get('no_grad', True)

        # stage gating：是否启用 chamfer（由外部 w_chamfer 控制）
        if w_chamfer > 0 and points is not None and points.numel() > 0:

            mu_0 = mu_t[0].detach()        # [M,3]
            pts_0 = points[0].reshape(points.shape[1], -1, 3).mean(dim=0).detach()  # [N,3]

            ctx = torch.no_grad() if use_no_grad else nullcontext()
            with ctx:
                loss_acc = torch.tensor(0.0, device=device)
                valid_t = 0

                # ===== 高斯侧：动态 mask（只算一次）=====
                dyn_mask_mu = torch.ones(mu_0.shape[0], device=device, dtype=torch.bool)

                mu_0_dyn = mu_0[dyn_mask_mu]
                if mu_0_dyn.shape[0] < min_dyn_pts:
                    # 动态点太少，直接跳过 Chamfer
                    chamfer_loss = torch.tensor(0.0, device=device)
                else:
                    for t in range(1, T):
                        mu = mu_t[t]              # [M,3]
                        pts = points[t].reshape(points.shape[1], -1, 3).mean(dim=0)  # [N,3]

                        if mu.numel() == 0 or pts.numel() == 0:
                            continue

                        # ===== displacement =====
                        d_mu_all  = mu  - mu_0          # [M,3]
                        d_pts_all = pts - pts_0         # [N,3]

                        # ===== 仅取动态高斯 =====
                        d_mu = d_mu_all[dyn_mask_mu]    # [Md,3]

                        # ===== points 侧：用位移幅度筛动态前景 =====
                        pts_disp_norm = d_pts_all.norm(dim=1)
                        dyn_mask_pts = pts_disp_norm > pts_disp_norm.mean()  # 自适应阈值
                        d_pts = d_pts_all[dyn_mask_pts]  # [Nd,3]

                        if d_mu.shape[0] < min_dyn_pts or d_pts.shape[0] < min_dyn_pts:
                            continue

                        # ===== subsample（防 OOM）=====
                        m_idx = torch.randperm(d_mu.shape[0], device=device)[:min(d_mu.shape[0], K_ds)]
                        n_idx = torch.randperm(d_pts.shape[0], device=device)[:min(d_pts.shape[0], K_ds)]

                        d_mu_s  = d_mu[m_idx]
                        d_pts_s = d_pts[n_idx]

                        # ===== displacement Chamfer =====
                        dists = torch.cdist(d_mu_s, d_pts_s, p=2)
                        chamfer_t = (
                            dists.min(dim=1).values.mean() +
                            dists.min(dim=0).values.mean()
                        )

                        loss_acc = loss_acc + chamfer_t
                        valid_t += 1

                if valid_t > 0:
                    chamfer_loss = loss_acc / valid_t



        # ------------- Keypoint 2D reprojection loss -------------
        kp2d_loss = torch.tensor(0.0, device=device)
        if w_kp2d > 0 and keypoints_2d is not None and keypoints_2d.numel() > 0:
            # keypoints_2d: [T,V,K,2] in pixel (x,y)
            K_mat = camera_intrinsics_seq  # [T,V,3,3]
            C2W = camera_poses_seq        # [T,V,4,4]
            T_kp, V_kp, Kk = keypoints_2d.shape[0], keypoints_2d.shape[1], keypoints_2d.shape[2]
            # soft association temperature
            tau = 4.0
            knn = int(config.get('loss_params', {}).get('kp2d', {}).get('knn', 64))
            knn_chunk = int(config.get('loss_params', {}).get('kp2d', {}).get('knn_chunk', 4096))

            def _softmin_pred_uv_topk(
                kp_xy: torch.Tensor,  # [K,2]
                proj_xy: torch.Tensor,  # [Mv,2]
                tau_val: float,
                knn_val: int,
                chunk: int,
            ) -> torch.Tensor:
                # 避免一次性构造 [K,Mv] 的 cdist（显存杀手）；改为分块扫描 + 维护 TopK。
                K_local = int(kp_xy.shape[0])
                M_local = int(proj_xy.shape[0])
                if M_local == 0:
                    return torch.zeros(K_local, 2, device=kp_xy.device, dtype=kp_xy.dtype)
                knn_eff = min(int(knn_val), M_local)

                best_d2 = torch.full((K_local, knn_eff), float("inf"), device=kp_xy.device, dtype=kp_xy.dtype)
                best_xy = torch.zeros((K_local, knn_eff, 2), device=kp_xy.device, dtype=kp_xy.dtype)

                for start in range(0, M_local, max(1, int(chunk))):
                    end = min(M_local, start + max(1, int(chunk)))
                    proj_c = proj_xy[start:end]  # [C,2]
                    C_local = int(proj_c.shape[0])
                    if C_local == 0:
                        continue
                    d2_sq = (kp_xy[:, None, :] - proj_c[None, :, :]).pow(2).sum(dim=-1)  # [K,C]

                    # 当前 chunk 的 topk（按距离）
                    k_chunk = min(knn_eff, C_local)
                    d2_sq_k, idx_k = torch.topk(d2_sq, k=k_chunk, dim=1, largest=False)
                    proj_k = proj_c[idx_k]  # [K,knn,2]

                    # 与历史 topk 合并再取 topk
                    cand_d2 = torch.cat([best_d2, d2_sq_k], dim=1)  # [K,knn + k_chunk]
                    cand_xy = torch.cat([best_xy, proj_k], dim=1)   # [K,knn + k_chunk,2]
                    new_d2, new_idx = torch.topk(cand_d2, k=knn_eff, dim=1, largest=False)
                    best_xy = cand_xy.gather(1, new_idx.unsqueeze(-1).expand(-1, -1, 2))
                    best_d2 = new_d2

                d = best_d2.clamp_min(1e-12).sqrt()  # 与原 torch.cdist(p=2) 对齐
                w = torch.softmax(-d / float(tau_val), dim=1)  # [K,knn]
                return (w.unsqueeze(-1) * best_xy).sum(dim=1)  # [K,2]

            kp_err_acc = 0.0
            cnt = 0
            for t in range(min(T, T_kp)):
                mu = mu_t[t]  # [M,3]
                if mu.numel() == 0:
                    continue
                M_t = mu.shape[0]
                for v in range(V_kp):
                    kp_pix = keypoints_2d[t, v]  # [K,2]
                    Kk_t = kp_pix.shape[0]
                    if Kk_t == 0:
                        continue
                    # 投影 mu -> [M,2]
                    c2w = C2W[t, v]  # [4,4]
                    w2c = torch.inverse(c2w)
                    xyz_h = torch.cat([mu, torch.ones(M_t,1, device=mu.device,dtype=mu.dtype)], dim=1)  # [M,4]
                    Xc = (w2c @ xyz_h.t()).t()[:, :3]          # [M,3]
                    z = Xc[:, 2].clamp(min=1e-4)               # 深度
                    uvw = (K_mat[t, v] @ Xc.t()).t()           # [M,3]
                    u = uvw[:, 0] / uvw[:, 2]
                    vv = uvw[:, 1] / uvw[:, 2]
                    proj = torch.stack([u, vv], dim=-1)        # [M,2]
                    # 有效可见 mask (z>0 且在图内)
                    H_img, W_img = H_t, W_t
                    vis_mask = (z > 1e-4) & (proj[:,0]>=0) & (proj[:,0]<W_img) & (proj[:,1]>=0) & (proj[:,1]<H_img)
                    if vis_mask.sum() == 0:
                        continue
                    proj_vis = proj[vis_mask]                # [Mv,2]
                    # softmin → TopK 近邻（显存从 O(K*Mv) 降到 O(K*knn)）
                    kp_xy = kp_pix.to(device=proj_vis.device, dtype=proj_vis.dtype)
                    pred_uv = _softmin_pred_uv_topk(kp_xy, proj_vis, tau, knn, knn_chunk)  # [K,2]
                    err = (pred_uv - kp_pix.to(dtype=pred_uv.dtype,device=pred_uv.device)).abs().mean()
                    kp_err_acc = kp_err_acc + err
                    cnt += 1
            if cnt > 0:
                kp2d_loss = kp_err_acc / float(cnt)

        # # ------------- Sparse flow alignment (Forge4D-style, 2A) -------------
        # flow_sparse_loss = torch.tensor(0.0, device=device)
        # if w_flow_sparse > 0 and keypoints_2d is not None and keypoints_2d.numel() > 0 and T > 1:
        #     if flow_sparse_use_cyclic and batch_idx == 0 and epoch == 0:
        #         print("[flow_sparse] use_cyclic_mask is set but dense cyclic masking is not implemented for sparse flow; ignoring.")
        #     H_img, W_img = H_t, W_t
        #     loss_acc = 0.0
        #     cnt_flow = 0
        #     for t in range(1, T):
        #         M_t = min(mu_t[t].shape[0], mu_t[t - 1].shape[0])
        #         if M_t == 0:
        #             continue
        #         if flow_sparse_max_mu > 0 and M_t > flow_sparse_max_mu:
        #             idx = torch.randperm(M_t, device=device)[:flow_sparse_max_mu]
        #         else:
        #             idx = torch.arange(M_t, device=device)
        #         mu_t_s = mu_t[t][idx]
        #         mu_tm1_s = mu_t[t - 1][idx]
        #         for v in range(keypoints_2d.shape[1]):
        #             kp_t = keypoints_2d[t, v]  # [K,2]
        #             kp_tm1 = keypoints_2d[t - 1, v]
        #             if kp_t.numel() == 0:
        #                 continue
        #             uv_t, z_t = project_world_to_pixel(mu_t_s, camera_poses_seq[t, v], camera_intrinsics_seq[t, v])
        #             uv_tm1, _ = project_world_to_pixel(mu_tm1_s, camera_poses_seq[t - 1, v], camera_intrinsics_seq[t - 1, v])
        #             # visibility mask (in-front + inside image)
        #             vis = (z_t > 1e-6) & (uv_t[:, 0] >= 0) & (uv_t[:, 0] < W_img) & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < H_img)
        #             if vis.sum() < 4:
        #                 continue
        #             uv_t = uv_t[vis]
        #             uv_tm1 = uv_tm1[vis]
        #             d2 = (kp_t[:, None, :] - uv_t[None, :, :]).pow(2).sum(dim=-1)
        #             w = torch.softmax(-d2 / max(1e-6, flow_sparse_tau ** 2), dim=1)
        #             pred_uv_t = w @ uv_t
        #             pred_uv_tm1 = w @ uv_tm1
        #             flow_pred = pred_uv_tm1 - pred_uv_t
        #             flow_gt = kp_tm1 - kp_t
        #             loss_acc = loss_acc + (flow_pred - flow_gt).pow(2).sum(dim=-1).mean()
        #             cnt_flow += 1
        #     if cnt_flow > 0:
        #         flow_sparse_loss = torch.tensor(loss_acc / float(cnt_flow), device=device)

        # ------------- Dense flow reprojection (Forge4D-style, 2B) -------------
        flow_reproj_loss = torch.tensor(0.0, device=device)
        if w_flow_reproj > 0 and isinstance(flow_fwd, torch.Tensor) and flow_fwd.numel() > 0 and T > 1:
            if flow_reproj_use_cyclic and batch_idx == 0 and epoch == 0:
                print("[flow_reproj] use_cyclic_mask is set but cyclic masking is not implemented; ignoring.")
            H_img, W_img = H_t, W_t
            loss_acc = 0.0
            cnt_flow = 0
            for t in range(T - 1):
                if t >= flow_fwd.shape[0]:
                    break
                M_t = min(mu_t[t].shape[0], mu_t[t + 1].shape[0])
                if M_t == 0:
                    continue
                if flow_reproj_max_mu > 0 and M_t > flow_reproj_max_mu:
                    idx = torch.randperm(M_t, device=device)[:flow_reproj_max_mu]
                else:
                    idx = torch.arange(M_t, device=device)
                mu_t_s = mu_t[t][idx]
                mu_tp1_s = mu_t[t + 1][idx]
                for v in range(flow_fwd.shape[1]):
                    uv_t, z_t = project_world_to_pixel(mu_t_s, camera_poses_seq[t, v], camera_intrinsics_seq[t, v])
                    uv_tp1, _ = project_world_to_pixel(mu_tp1_s, camera_poses_seq[t + 1, v], camera_intrinsics_seq[t + 1, v])
                    vis = (z_t > 1e-6) & (uv_t[:, 0] >= 0) & (uv_t[:, 0] < W_img) & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < H_img)
                    if vis.sum() < 4:
                        continue
                    uv_t = uv_t[vis]
                    uv_tp1 = uv_tp1[vis]
                    flow_pred = uv_tp1 - uv_t
                    # sample GT flow at uv_t
                    flow_map = flow_fwd[t, v].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
                    x = uv_t[:, 0]
                    y = uv_t[:, 1]
                    x_norm = (x / max(1.0, W_img - 1)) * 2.0 - 1.0
                    y_norm = (y / max(1.0, H_img - 1)) * 2.0 - 1.0
                    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
                    flow_gt = F.grid_sample(flow_map, grid, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).t()
                    loss_acc = loss_acc + (flow_pred - flow_gt).pow(2).sum(dim=-1).mean()
                    cnt_flow += 1
            if cnt_flow > 0:
                # loss_acc 已经是 Tensor，避免 torch.tensor(tensor) 触发 copy-construct warning
                flow_reproj_loss = (loss_acc / float(cnt_flow))


        # # ----- Stage C regularization (budget/entropy + repulsion) -----
        # # Initialize entropy placeholders for clarity even if selector disabled
        # imp_entropy = torch.tensor(0.0, device=device)
        # imp_entropy_reg = torch.tensor(0.0, device=device)
        # imp_budget_loss = torch.tensor(0.0, device=device)

        # imp_logits = output.get("imp_logits", None)
        # if imp_logits is not None and (w_imp_budget > 0.0 or w_imp_entropy > 0.0):
        #     # OOM 防御：对超大 imp_logits 做随机采样估计（保留梯度，但显著降低峰值显存）
        #     sel_cfg = config.get("loss_params", {}).get("selector", {}) or {}
        #     sample_n = int(sel_cfg.get("sample", 262144))  # 0 表示不采样（全量，风险较高）

        #     flat_logits = imp_logits.reshape(-1)
        #     n_all = int(flat_logits.numel())
        #     if sample_n > 0 and n_all > sample_n:
        #         idx = torch.randint(0, n_all, (sample_n,), device=flat_logits.device)
        #         logits_use = flat_logits[idx]
        #     else:
        #         logits_use = flat_logits

        #     imp_probs_raw = torch.sigmoid(logits_use)
        #     eps_p = 1e-6  # avoid exact 0/1 which lead to log(0)=nan
        #     imp_probs = imp_probs_raw.clamp(min=eps_p, max=1.0 - eps_p)
        #     imp_probs = torch.nan_to_num(imp_probs, nan=0.5, posinf=1.0 - eps_p, neginf=eps_p)

        #     # 让 probs 的均值接近预算比例，避免 logits 全饱和（硬选 k 固定，主要是 stabilizer）
        #     target = float(mu_t.shape[1]) / float(max(1, n_all))
        #     imp_budget_loss = (imp_probs.mean() - target) ** 2

        #     # 最大化熵：loss = -H(p)
        #     ent = -(imp_probs * torch.log(imp_probs) + (1.0 - imp_probs) * torch.log(1.0 - imp_probs))
        #     imp_entropy = torch.nan_to_num(ent, nan=0.0).mean()  # ensure finite
        #     imp_entropy_reg = -imp_entropy

        # xyz_f0 = output.get("xyz_f0", None)
        # if xyz_f0 is not None and w_imp_repel > 0.0:
        #     rp_cfg = config.get("loss_params", {}).get("imp_repel", {}) or {}
        #     rp_num = int(rp_cfg.get("num_points", 4096))
        #     rp_knn = int(rp_cfg.get("knn", 8))
        #     rp_sigma = float(rp_cfg.get("sigma", 0.05))
        #     rp_num = int(min(max(0, rp_num), int(xyz_f0.shape[0])))
        #     rp_knn = int(min(max(1, rp_knn), max(1, rp_num - 1)))
        #     if rp_num >= 2 and rp_sigma > 0:
        #         idx = torch.randperm(int(xyz_f0.shape[0]), device=xyz_f0.device)[:rp_num]
        #         x = xyz_f0[idx].to(dtype=torch.float32)
        #         d2 = torch.cdist(x, x, p=2.0).pow(2)
        #         d2 = d2 + torch.eye(rp_num, device=d2.device, dtype=d2.dtype) * 1e6
        #         knn_d2 = torch.topk(d2, k=rp_knn, dim=-1, largest=False).values
        #         sigma2 = float(max(1e-8, rp_sigma * rp_sigma))
        #         imp_repel_loss = torch.exp(-knn_d2 / sigma2).mean().to(device=device, dtype=mu_t.dtype)

        # ===== quick diagnostics (only first batch per epoch) =====
        if batch_idx == 0:
            print(_fmt_nonzero(f'[Diag Stage{stage}]', {
                # Stage1 active weights
                'photo': w_photo, 'ssim': w_ssim, 'lpips': w_lpips, 'kp2d': w_kp2d, 'flow_dp': w_flow_reproj,
                'sil': w_sil,
                # keep for later (commented in tensorboard section)
                # 'alpha': w_alpha, 'motion': w_motion, 'vel': w_temp_vel, 'acc': w_temp_acc,
                # 'chamf': w_chamfer, 'resMag': w_res_mag, 'resSm': w_res_smooth,
            }))
        # ===== compose final loss with explicit term variables =====
        term_photo = w_photo * photo_loss
        term_ssim  = w_ssim * ssim_loss
        term_lp = w_lpips * lpips_loss
        term_retarget = w_retarget * retarget_loss
        # term_flow_sparse = w_flow_sparse * flow_sparse_loss
        term_flow_reproj = w_flow_reproj * flow_reproj_loss
        term_alpha = w_alpha * alpha_sparse
        term_sil   = w_sil * sil_loss
        term_cham  = w_chamfer * chamfer_loss
        term_motion = w_motion * motion_reg
        term_vel_high_smooth = w_vel_high_smooth * vel_high_smooth  # velocity smoothness for motion_block
        # term_imp_budget = w_imp_budget * imp_budget_loss
        # term_imp_entropy = w_imp_entropy * imp_entropy_reg
        term_kp2d = w_kp2d * kp2d_loss
        # term_temp = lambda_smooth * temporal_smooth
        loss = (
            term_photo + term_ssim + term_lp + term_retarget + term_flow_reproj + term_sil + term_alpha +
            term_cham + term_motion + term_vel_high_smooth + term_kp2d +
            w_dyn * mask_dyn_loss + w_sta * mask_sta_loss + w_cross * mask_cross_loss
            # term_imp_budget + term_imp_entropy  + term_flow_sparse + term_temp +
        )

        # 累积 eval_loss（photo+ssim+chamfer，用于日志对齐）
        eval_loss_batch = (w_photo * photo_loss + w_ssim * ssim_loss + w_chamfer * chamfer_loss)
        sum_eval += float(eval_loss_batch.item())

        if batch_idx == 0 and epoch % 1 == 0:
            print(_fmt_nonzero(f'[TermContrib Stage{stage}]', {
                        'photo': term_photo, 'ssim': term_ssim, 'lpips': term_lp, 'retarget': term_retarget,  'flow_dp': term_flow_reproj, 'sil': term_sil,
                        'kp2d': term_kp2d, 'motion': term_motion, 'velHighSm': term_vel_high_smooth, 'chamfer': term_cham,}))
                        #'flow_sp': term_flow_sparse,  'vel': term_temp_vel, 'acc': term_temp_acc,'resMag': term_res_mag, 'resSm': term_res_smooth}
        #【TODO】将rendered_images 和 gt_images [4, 4, 3, 378, 504]绘制在一张大图上，大图两行每行 16 张图
        # rendered_images_concat = torch.cat([rendered_images, gt_images], dim=0).view(32, 3, 378, 504)
        # save_image(rendered_images_concat, 'debug/images/rendered_gt_images.png')

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        sum_photo += float(photo_loss.item())
        sum_ssim += float(ssim_loss.item())
        sum_lpips += float(lpips_loss.item())
        sum_retarget += float(retarget_loss.item())
        # sum_flow_sparse += float(flow_sparse_loss.item())
        sum_flow_reproj += float(flow_reproj_loss.item())
        sum_psnr += float(psnr_val.item())
        sum_sil += float(sil_loss.item())
        sum_alpha += float(alpha_sparse.item())
        sum_chamfer += float(chamfer_loss.item())
        sum_motion += float(motion_reg.item())
        sum_vel_high_smooth += float(vel_high_smooth.item())
        # sum_temporal += float(temporal_smooth.item())
        # sum_imp_budget += float(imp_budget_loss.item())
        # optional losses
        sum_mask_dyn += float(mask_dyn_loss.item())
        sum_mask_sta += float(mask_sta_loss.item())
        sum_mask_cross += float(mask_cross_loss.item())
            # sum_res_mag += float(res_mag.item())
            # sum_res_smooth += float(res_smooth.item())
            # sum_imp_entropy += float(imp_entropy.item() if 'imp_entropy' in locals() else 0.0)
            # sum_imp_repel += float(imp_repel_loss.item() if 'imp_repel_loss' in locals() else 0.0)
        sum_kp2d += float(kp2d_loss.item())

        n_batches += 1

        # Save last batch artifacts (after streaming, rendered_images 不再存在)
        # last_rendered_images 已在流式渲染中保存为 t==0,v==0 的一张图（若有 GT）
        last_gaussian_params = {
            'mu': mu_t.detach().cpu(),
            'scale': scale_t.detach().cpu(),
            'color': color_t.detach().cpu(),
            'opacity': alpha_t.detach().cpu(),
        }
        last_gt_images = gt_images.detach().cpu() if gt_images is not None else None
        last_time_indices = time_ids.detach().cpu()

        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            denom = max(1, n_batches)
            writer.add_scalar('Loss/Total', total_loss / denom, global_step)
            writer.add_scalar('Loss/Photo', sum_photo / denom, global_step)
            writer.add_scalar('Loss/SSIM', sum_ssim / denom, global_step)
            writer.add_scalar('Loss/LPIPS', sum_lpips / denom, global_step)
            # ==== Only log active Stage-1 losses ====
            writer.add_scalar('Loss/FlowDense', sum_flow_reproj / denom, global_step)
            writer.add_scalar('Loss/Silhouette', sum_sil / denom, global_step)
            writer.add_scalar('Loss/KP2D', sum_kp2d / denom, global_step)
            writer.add_scalar('Metric/PSNR', sum_psnr / denom, global_step)
            # writer.add_scalar('Loss/Retarget', sum_retarget / denom, global_step)  # inactive in Stage-1
            # writer.add_scalar('Loss/FlowSparse', sum_flow_sparse / denom, global_step)  # unused
            # writer.add_scalar('Loss/AlphaSparse', sum_alpha / denom, global_step)
            # writer.add_scalar('Loss/Chamfer', sum_chamfer / denom, global_step)
            # writer.add_scalar('Loss/MotionReg', sum_motion / denom, global_step)
            writer.add_scalar('Loss/VelHighSmooth', sum_vel_high_smooth / denom, global_step)
            # writer.add_scalar('Loss/TemporalSmooth', sum_temporal / denom, global_step)
            # writer.add_scalar('Loss/ImpBudget', sum_imp_budget / denom, global_step)

            # ==== Log loss weights for transparency ====
            writer.add_scalar('Weight/Photo', w_photo, global_step)
            writer.add_scalar('Weight/SSIM', w_ssim, global_step)
            writer.add_scalar('Weight/LPIPS', w_lpips, global_step)
            writer.add_scalar('Weight/FlowDense', w_flow_reproj, global_step)
            writer.add_scalar('Weight/Silhouette', w_sil, global_step)
            writer.add_scalar('Weight/KP2D', w_kp2d, global_step)
            writer.add_scalar('Weight/VelHighSmooth', w_vel_high_smooth, global_step)
            # Retargeting debug: verify motion is being trained
            try:
                if mu_t is not None and T > 1:
                    writer.add_scalar('Diag/MuDelta_t1', (mu_t[1] - mu_t[0]).abs().mean().item(), global_step)
                if dxyz_t is not None:
                    writer.add_scalar('Diag/DxyzMean', dxyz_t.abs().mean().item(), global_step)
            except Exception:
                pass

            if do_color_init:
                # 记录当前退火系数（越大越依赖 init_sh0）
                if blend_steps > 0:
                    t = float(min(max(epoch * len(dataloader) + batch_idx, 0), blend_steps)) / float(blend_steps)
                    beta = (1.0 - t) * blend_start + t * blend_end
                    beta = float(max(0.0, min(1.0, beta)))
                else:
                    beta = float(blend_start)
                writer.add_scalar('Train/ColorInitBlend', beta, epoch * len(dataloader) + batch_idx)

        # free
        del mu_t, scale_t, color_t, alpha_t, output
        if gt_images is not None:
            del gt_images
        # 不要在训练 loop 中调用 empty_cache（会破坏 allocator cache，导致碎片化/更早 OOM）

    # Save grid image / PLYs for last batch (rank0 only)
    if output_dir is not None and is_rank0:
        images_dir = os.path.join(output_dir, 'epoch_images')
        gaussians_dir = os.path.join(output_dir, 'epoch_gaussians')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(gaussians_dir, exist_ok=True)

        def ensure_tv_chw(img: torch.Tensor) -> torch.Tensor:
            if img is None:
                return None
            # 兼容流式渲染：last_rendered_images 可能仅保存了一张 [C,H,W]
            if img.dim() == 3 and img.shape[0] == 3:
                return img.unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
            if img.dim() == 4 and img.shape[-1] == 3:
                return img.unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()  # [1,V,3,H,W]
            if img.dim() != 5:
                raise ValueError("Expected image tensor with 5 dims [T,V,C,H,W] or [T,V,H,W,C].")
            if img.shape[2] == 3:
                return img
            if img.shape[-1] == 3:
                return img.permute(0, 1, 4, 2, 3).contiguous()
            raise ValueError("Image tensor does not have a channel dimension of size 3.")

        pred_images = ensure_tv_chw(last_rendered_images).contiguous().clamp(0.0, 1.0)
        T, N_views, _, H, W = pred_images.shape

        if last_gt_images is not None:
            gt_images_tensor = ensure_tv_chw(last_gt_images)
            Hg, Wg = gt_images_tensor.shape[-2:]
            if (Hg != H) or (Wg != W):
                gt_images_tensor = torch.nn.functional.interpolate(
                    gt_images_tensor.reshape(-1, 3, Hg, Wg), size=(H, W), mode='bilinear', align_corners=False
                ).reshape(T, N_views, 3, H, W)
            gt_images_tensor = gt_images_tensor.clamp(0.0, 1.0)
        else:
            gt_images_tensor = torch.zeros(T, N_views, 3, H, W, device=pred_images.device, dtype=pred_images.dtype)

        views_to_show = min(4, N_views)
        times_to_show = min(4, T)
        grid_rows = []
        for v in range(views_to_show):
            gt_row = torch.cat([gt_images_tensor[t, v] for t in range(times_to_show)], dim=2)
            pred_row = torch.cat([pred_images[t, v] for t in range(times_to_show)], dim=2)
            grid_rows.append(gt_row)
            grid_rows.append(pred_row)

        grid_img = torch.cat(grid_rows, dim=1) if grid_rows else torch.empty(0, device=pred_images.device)
        img_path = os.path.join(images_dir, f'epoch_{epoch:05d}_pred_gt_grid.png')
        save_image(grid_img, img_path)
        if writer is not None:
            writer.add_image('Epoch/GridImage', grid_img, epoch)

        # Save PLYs
        def save_gaussians_ply(gaussian_params: dict, output_path: str, sh_degree: int = 0):
            xyz = gaussian_params['mu']
            scale = gaussian_params['scale']
            color = gaussian_params['color']
            opacity = gaussian_params['opacity']
            if torch.is_tensor(xyz): xyz = xyz.numpy()
            if torch.is_tensor(scale): scale = scale.numpy()
            if torch.is_tensor(color): color = color.numpy()
            if torch.is_tensor(opacity): opacity = opacity.numpy()
            if opacity.ndim > 1: opacity = opacity.squeeze()
            M = xyz.shape[0]
            rotation = np.zeros((M, 4), dtype=np.float32); rotation[:, 0] = 1.0
            dtype_list = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ]
            elements = np.empty(M, dtype=np.dtype(dtype_list))
            elements['x'] = xyz[:, 0]; elements['y'] = xyz[:, 1]; elements['z'] = xyz[:, 2]
            elements['nx'] = rotation[:, 1]; elements['ny'] = rotation[:, 2]; elements['nz'] = rotation[:, 3]
            sh = color.reshape(-1, 1, 3)
            elements['f_dc_0'] = sh[:, 0, 0]; elements['f_dc_1'] = sh[:, 0, 1]; elements['f_dc_2'] = sh[:, 0, 2]
            elements['opacity'] = np.clip(opacity, 0.0, 1.0)
            elements['scale_0'] = scale[:, 0]; elements['scale_1'] = scale[:, 1]; elements['scale_2'] = scale[:, 2]
            elements['rot_0'] = 1.0; elements['rot_1'] = 0.0; elements['rot_2'] = 0.0; elements['rot_3'] = 0.0
            PlyData([PlyElement.describe(elements, 'vertex')]).write(output_path)
        if last_gaussian_params is not None and last_time_indices is not None:
            B_ply = last_time_indices.shape[0]
            for b in range(B_ply):
                time_idx = last_time_indices[b].item()
                gaussian_dict = {
                    k: (v[b] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                    for k, v in last_gaussian_params.items()
                }
                ply_path = os.path.join(gaussians_dir, f'epoch_{epoch:05d}_time_{time_idx:05d}_gaussians.ply')
                save_gaussians_ply(gaussian_dict, ply_path, sh_degree=0)

    denom = max(1, n_batches)
    metrics: dict[str, float | int] = {
        'loss_total': total_loss / denom,
        'loss_eval': sum_eval / denom,
        'photo_loss': sum_photo / denom,
        'ssim_loss': sum_ssim / denom,
        'lpips_loss': sum_lpips / denom,  # stage1 active
        # 'retarget_loss': sum_retarget / denom,
        # 'flow_sparse_loss': sum_flow_sparse / denom,
        'flow_dense_loss': sum_flow_reproj / denom,
        'silhouette_loss': sum_sil / denom,
        'retarget_loss': sum_retarget / denom,
        'flow_sparse_loss': sum_flow_sparse / denom,
        'flow_reproj_loss': sum_flow_reproj / denom,
        'psnr': sum_psnr / denom,
        'silhouette_loss': sum_sil / denom,
        'alpha_sparsity': sum_alpha / denom,
        'chamfer_loss': sum_chamfer / denom,
        'motion_reg': sum_motion / denom,
        'temporal_smooth': sum_temporal / denom,
        'imp_budget_loss': sum_imp_budget / denom,
        'mask_dyn_loss': sum_mask_dyn / denom,
        'mask_sta_loss': sum_mask_sta / denom,
        'mask_cross_loss': sum_mask_cross / denom,
        'residual_mag': sum_res_mag / denom,
        'residual_smooth': sum_res_smooth / denom,
        'imp_entropy': sum_imp_entropy / denom,
        'imp_repel_loss': sum_imp_repel / denom,
        'kp2d_loss': sum_kp2d / denom,
        'stage': current_stage,
    }
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    photo_loss_fn: PhotoConsistencyLoss,
    lpips_fn: Optional[nn.Module],
    device: torch.device,
    config: dict,
    writer: Optional[SummaryWriter],
    epoch: int,
    ex4dgs_dir: Optional[str],
) -> dict:
    model.eval()
    total_loss = 0.0
    sum_photo = 0.0
    sum_ssim = 0.0
    sum_lpips = 0.0
    sum_sil = 0.0
    sum_alpha = 0.0
    sum_chamfer = 0.0
    sum_motion = 0.0
    sum_vel_high_smooth = 0.0  # velocity smoothness for motion_block
    sum_temporal = 0.0
    sum_psnr = 0.0  # PSNR for validation
    n_batches = 0
    lambda_smooth = float(config.get('loss_weights', {}).get('dynamic_temporal', 0.0))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            batch_data = prepare_batch(
                batch, device,
                ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
                image_size=tuple(config['data']['image_size']),
                config=config, mode='val'
            )
            points = batch_data.get('points', None)
            points_3d = batch_data.get('points_3d', None)
            feat_2d = batch_data['feat_2d']
            conf_seq = batch_data['conf']
            dyn_mask_tv = batch_data.get('dynamic_conf_tv', None)
            dyn_traj_tv = batch_data.get('dynamic_traj_tv', None)
            keypoints_2d = batch_data.get('keypoints_2d', None)
            flow_fwd = batch_data.get('flow_fwd', None)
            camera_poses_seq = batch_data['camera_poses']
            camera_intrinsics_seq = batch_data['camera_intrinsics']
            time_ids = batch_data['time_ids']
            gt_images = batch_data.get('gt_images', None)
            if gt_images is not None:
                gt_images = gt_images.permute(0, 1, 4, 2, 3).contiguous()  # [T,V,3,H,W]
            silhouette = batch_data.get('silhouette', None)
            T = int(time_ids.shape[0])
            H_t, W_t = config['data']['image_size']
            # 提前计算 w_lpips_check 用于循环中的条件检查（验证时使用配置值）
            lw_raw = config.get('loss_weights', {})
            w_lpips_check_val = float(lw_raw.get('lpips', 0.0))
            # 每个 batch 的 loss 累加器
            photo_loss_acc = torch.tensor(0.0, device=device)
            ssim_loss_acc = torch.tensor(0.0, device=device)
            lpips_loss_acc = torch.tensor(0.0, device=device)
            psnr_acc = torch.tensor(0.0, device=device)  # PSNR accumulator for validation
            lpips_count = 0
            n_render = 0

            dyn_mask_in = dyn_mask_tv if dyn_mask_tv is not None else silhouette
            # 与训练保持一致：验证阶段也应遵循 stage 的 freeze_canonical 设定，
            # 否则 train/val 前向不一致，会导致 photo/ssim 指标差异异常。
            output = model(
                points_full=points,    # [T,V,H,W,3] (preferred for slot_dual tokenization)
                # points_3d=points_3d,   # [T,N,3] (optional; AABB/debug)
                feat_2d=feat_2d,       # [T,V,H'=,W'=,C]
                conf_2d=conf_seq,
                camera_poses=camera_poses_seq,
                camera_K=camera_intrinsics_seq,
                time_ids=time_ids,
                dyn_mask_2d=dyn_mask_in,
                build_canonical=False,
            )
            mu_t, scale_t, color_t, alpha_t = output['mu_t'], output['scale_t'], output['color_t'], output['alpha_t']
            rot_t = output.get('rot_t', None)
            dxyz_t = output.get('dxyz_t', None)
            velocity_high = output.get("velocity_high", None)  # [T-1, M, 3] velocity from motion_block
            sim3_s, sim3_R, sim3_t = output.get('sim3_s', None), output.get('sim3_R', None), output.get('sim3_t', None)

            for t in range(T):
                mu_frame = mu_t[t]
                scale_frame = scale_t[t]
                color_frame = color_t[t]
                alpha_frame = alpha_t[t]
                rot_frame = rot_t[t] if rot_t is not None else None

                camera_poses_t = camera_poses_seq[t]          # [V,4,4]
                camera_intrinsics_t = camera_intrinsics_seq[t]
                
                # 逐视角渲染：使用 IntrinsicsCamera + render_gs（参照 test_render.py）
                bg_color = torch.ones(3, device=device)  # 白背景
                max_scale = float(config.get('model', {}).get('max_scale', 0.05))
                # 与 inference 对齐：scale 仅限幅，不做像素半径校准
                scale_frame = scale_frame.to(dtype=mu_frame.dtype).clamp(min=1e-6, max=max_scale)
                
                for vi in range(camera_poses_t.shape[0]):
                    # 从 c2w 得到 w2c = [R|t]
                    c2w = camera_poses_t[vi].detach().cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    R = w2c[:3, :3].astype(np.float32)
                    t_vec = w2c[:3, 3].astype(np.float32)
                    
                    K_np = camera_intrinsics_t[vi].detach().cpu().numpy().astype(np.float32)
                    
                    cam = IntrinsicsCamera(
                        K=K_np,
                        R=R,
                        T=t_vec,
                        width=int(W_t),
                        height=int(H_t),
                        znear=0.01,
                        zfar=100.0,
                    )
                    
                    # 构建高斯属性对象
                    # 处理不透明度维度
                    max_scale = float(config.get('model', {}).get('max_scale', 0.05))
                    opacity = alpha_frame.squeeze(-1) if alpha_frame.dim() > 1 else alpha_frame
                    opacity = opacity.to(dtype=mu_frame.dtype).clamp(0.0, 1.0)
                    
                    # 旋转：优先使用模型输出的 rot_t（矩阵），渲染端需要四元数 (WXYZ)
                    num_gs = mu_frame.shape[0]
                    if rot_frame is None:
                        rotation = torch.zeros(num_gs, 4, device=mu_frame.device, dtype=mu_frame.dtype)
                        rotation[:, 0] = 1.0  # 单位四元数 [1, 0, 0, 0]
                    else:
                        rotation = matrix_to_quaternion(rot_frame).to(device=mu_frame.device, dtype=mu_frame.dtype)
                        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    
                    # 创建球谐系数（仅 DC 分量）
                    sh = rgb2sh0(color_frame).unsqueeze(1)  # [M, 3] -> [M, 1, 3]
                    
                    gs_attrs = GaussianAttributes(
                        xyz=mu_frame,
                        opacity=opacity,
                        scaling=scale_frame,
                        rotation=rotation,
                        sh=sh,
                    )
                    
                    # 渲染单视角
                    res_v = render_gs(
                        camera=cam,
                        bg_color=bg_color,
                        gs=gs_attrs,
                        target_image=None,
                        sh_degree=0,
                        scaling_modifier=1.0,
                    )
                    pred_chw = res_v["color"]  # [3,H,W]
                    del res_v
                    if gt_images is not None:
                        gt_chw = gt_images[t, vi].to(device=device, dtype=pred_chw.dtype)
                        photo_loss_acc = photo_loss_acc + F.l1_loss(pred_chw.clamp(0, 1), gt_chw.clamp(0, 1), reduction='mean')
                        if float(config.get('loss_weights', {}).get('ssim', 0.0)) > 0:
                            ssim_loss_acc = ssim_loss_acc + (1.0 - ssim_torch(
                                pred_chw.unsqueeze(0).clamp(0, 1),
                                gt_chw.unsqueeze(0).clamp(0, 1),
                                window_size=11,
                        ))
                        # PSNR 计算（验证时统计）
                        mse = torch.mean((pred_chw.clamp(0, 1) - gt_chw.clamp(0, 1)) ** 2).clamp_min(1e-8)
                        psnr_acc = psnr_acc + (10.0 * torch.log10(1.0 / mse))
                        # LPIPS 计算（验证时仅统计）
                        if (
                            lpips_fn is not None
                            and w_lpips_check_val > 0
                            and t == 0
                            and vi == 0
                        ):
                            pred_nchw = pred_chw.unsqueeze(0) * 2 - 1  # [1,3,H,W] 归一化到 [-1,1]
                            gt_nchw = gt_chw.unsqueeze(0) * 2 - 1      # [1,3,H,W] 归一化到 [-1,1]
                            lpips_loss_acc = lpips_loss_acc + lpips_fn(pred_nchw, gt_nchw).mean()
                            lpips_count += 1
                    n_render += 1
                    del pred_chw

            # Loss weights
            lw = config.get('loss_weights', {})
            w_photo = float(lw.get('photo_l1', 1.0))
            w_ssim = float(lw.get('ssim', 0.0))
            w_sil = float(lw.get('silhouette', 0.0))
            w_alpha = float(lw.get('alpha_l1', 0.0))
            w_chamfer = float(lw.get('chamfer', 0.0))
            w_motion = float(lw.get('motion_reg', 0.0))
            w_vel_high_smooth = float(lw.get('velocity_high_smooth', 0.0))  # velocity smoothness for motion_block

            if gt_images is not None and w_photo > 0 and n_render > 0:
                photo_loss = photo_loss_acc / float(n_render)
            else:
                photo_loss = torch.tensor(0.0, device=device)

            if gt_images is not None and w_ssim > 0 and n_render > 0:
                ssim_loss = ssim_loss_acc / float(n_render)
            else:
                ssim_loss = torch.tensor(0.0, device=device)

            # LPIPS loss (validation)
            if lpips_fn is not None and gt_images is not None and lpips_count > 0:
                lpips_loss = lpips_loss_acc / float(lpips_count)
            else:
                lpips_loss = torch.tensor(0.0, device=device)

            # Silhouette (disabled - no rendered_alpha available)
            sil_loss = torch.tensor(0.0, device=device)

            # Alpha sparsity
            alpha_sparse = torch.abs(alpha_t).mean() if w_alpha > 0 else torch.tensor(0.0, device=device)

            # Motion regularization (与训练流程保持一致：velocity & acceleration smoothness)
            motion_reg = torch.tensor(0.0, device=device)
            if w_motion > 0 and dxyz_t is not None and dxyz_t.shape[0] >= 3:
                # velocity & acceleration smoothness (L1) - 与训练流程保持一致
                vel = (dxyz_t[1:] - dxyz_t[:-1]).abs().mean(dim=2)   # [T-1,M]
                acc = (dxyz_t[2:] - 2*dxyz_t[1:-1] + dxyz_t[:-2]).abs().mean(dim=2)  # [T-2,M]
                vel_loss = vel.mean()
                acc_loss = acc.mean()
                motion_reg = 0.5 * vel_loss + 0.5 * acc_loss    #一阶二阶平滑

            # Velocity smoothness loss for motion_block (L_vel_smooth = |v_t - v_{t-1}|.mean())
            # 只作用在 motion_block 分支的 velocity，避免高频 jitter
            vel_high_smooth = torch.tensor(0.0, device=device)
            if w_vel_high_smooth > 0 and velocity_high is not None and velocity_high.shape[0] >= 2:
                # velocity_high: [T-1, M, 3]
                # 计算相邻时刻 velocity 的差异: |v_t - v_{t-1}|
                vel_diff = torch.abs(velocity_high[1:] - velocity_high[:-1])  # [T-2, M, 3]
                vel_high_smooth = vel_diff.mean()  # L1 smoothness

            # Chamfer
            chamfer_loss = torch.tensor(0.0, device=device)
            if w_chamfer > 0 and sim3_s is not None and points is not None and points.numel() > 0:
                max_m = 4096
                for t in range(T):
                    mu = mu_t[t]
                    pts = points[t].reshape(points.shape[1], -1, 3).mean(dim=0)  # [N,3]
                    if pts.numel() == 0:
                        continue
                    s_t = sim3_s[t]
                    R_t = sim3_R[t]
                    t_t = sim3_t[t]
                    pts_can = s_t * (pts @ R_t.T) + t_t
                    m_idx = torch.randperm(mu.shape[0], device=device)[:min(mu.shape[0], max_m)]
                    n_idx = torch.randperm(pts_can.shape[0], device=device)[:min(pts_can.shape[0], max_m)]
                    mu_s = mu[m_idx]
                    pts_s = pts_can[n_idx]
                    dists = torch.cdist(mu_s, pts_s, p=2)
                    chamfer_t = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
                    chamfer_loss = chamfer_loss + chamfer_t
                chamfer_loss = chamfer_loss / max(T,1)

            temporal_smooth = torch.tensor(0.0, device=device)
            if lambda_smooth > 0 and T >= 3:
                x_prev = mu_t[:-2]
                x_curr = mu_t[1:-1]
                x_next = mu_t[2:]
                temporal_smooth = ((x_next - 2 * x_curr + x_prev) ** 2).mean()

            loss = (
                w_photo * photo_loss +
                w_ssim * ssim_loss +
                w_chamfer * chamfer_loss +
                w_motion * motion_reg +
                w_vel_high_smooth * vel_high_smooth
            )

            total_loss += loss.item()
            sum_photo += float(photo_loss.item())
            sum_ssim += float(ssim_loss.item())
            sum_lpips += float(lpips_loss.item())
            sum_sil += float(sil_loss.item())
            sum_alpha += float(alpha_sparse.item())
            sum_chamfer += float(chamfer_loss.item())
            sum_motion += float(motion_reg.item())
            sum_vel_high_smooth += float(vel_high_smooth.item())
            sum_temporal += float(temporal_smooth.item())
            # 计算并累积 PSNR
            psnr_val = (psnr_acc / float(n_render)).detach() if (gt_images is not None and n_render > 0) else torch.tensor(0.0, device=device)
            sum_psnr += float(psnr_val.item())
            n_batches += 1

            del mu_t, scale_t, color_t, alpha_t
            if gt_images is not None:
                del gt_images

    denom = max(1, n_batches)
    return {
        'loss': total_loss / denom,  # 修改为 'loss' 以便 _fmt_metrics 正确显示
        'loss_eval': total_loss / denom,
        'photo_loss': sum_photo / denom,
        'ssim_loss': sum_ssim / denom,
        'lpips_loss': sum_lpips / denom,
        'silhouette_loss': sum_sil / denom,
        'alpha_sparsity': sum_alpha / denom,
        'chamfer_loss': sum_chamfer / denom,
        'motion_reg': sum_motion / denom,
        'velocity_high_smooth': sum_vel_high_smooth / denom,
        'temporal_smooth': sum_temporal / denom,
        'psnr': sum_psnr / denom,  # PSNR for validation
    }


def _load_hf_backbone(model: Trellis4DGS4DCanonical, hf_dir: str) -> list:
    try:
        pj = os.path.join(hf_dir, 'pipeline.json')
        with open(pj, 'r') as f:
            pipe = json.load(f)
        m = pipe['args']['models']
        def _ckpt_path(key):
            base = m[key]
            p = os.path.join(hf_dir, base + '.safetensors') if not base.endswith('.safetensors') else os.path.join(hf_dir, base)
            if not os.path.exists(p):
                p2 = os.path.join(hf_dir, 'ckpts', os.path.basename(base) + '.safetensors')
                if os.path.exists(p2):
                    return p2
            if not os.path.exists(p):
                p3 = os.path.join(hf_dir, 'ckpts', os.path.basename(base))
                if os.path.isdir(p3):
                    for fn in os.listdir(p3):
                        if fn.endswith('.safetensors'):
                            return os.path.join(p3, fn)
            return p
        flow_path = _ckpt_path('slat_flow_model')
        dec_path  = _ckpt_path('slat_decoder_gs')
        loaded = []
        if os.path.exists(flow_path):
            sd_flow = load_file(flow_path)
            flow_dtype = next(model.flow.parameters()).dtype
            flow_dev = next(model.flow.parameters()).device
            sd_flow = {k: v.to(device=flow_dev, dtype=flow_dtype) for k, v in sd_flow.items()}
            # safe partial load: only keys with matching shapes
            msd = model.flow.state_dict()
            keep = {}
            drop = []
            for k, v in sd_flow.items():
                if k in msd and msd[k].shape == v.shape:
                    keep[k] = v
                else:
                    drop.append(k)
            if keep:
                missing, unexpected = model.flow.load_state_dict(keep, strict=False)
                print(f"[HF] flow: loaded {len(keep)} tensors, skipped {len(drop)}, missing={len(missing)}, unexpected={len(unexpected)}")
                loaded.append('flow')
            else:
                print(f"[HF] flow: no matching tensors; skipped all {len(drop)} keys")
        else:
            print(f"[HF] Flow checkpoint not found at {flow_path}")
        if os.path.exists(dec_path):
            sd_dec = load_file(dec_path)
            # safe partial load for decoder
            dec_dtype = next(model.decoder.parameters()).dtype
            dec_dev = next(model.decoder.parameters()).device
            sd_dec = {k: v.to(device=dec_dev, dtype=dec_dtype) for k, v in sd_dec.items()}
            msd_dec = model.decoder.state_dict()
            keep_dec = {}
            drop_dec = []
            for k, v in sd_dec.items():
                if k in msd_dec and msd_dec[k].shape == v.shape:
                    keep_dec[k] = v
                else:
                    drop_dec.append(k)
            if keep_dec:
                missing, unexpected = model.decoder.load_state_dict(keep_dec, strict=False)
                print(f"[HF] decoder: loaded {len(keep_dec)} tensors, skipped {len(drop_dec)}, missing={len(missing)}, unexpected={len(unexpected)}")
                loaded.append('decoder')
            else:
                print(f"[HF] decoder: no matching tensors; skipped all {len(drop_dec)} keys")
        else:
            print(f"[HF] Decoder checkpoint not found at {dec_path}")
        return loaded
    except Exception as e:
        print(f"[HF] Failed to load HF backbone: {e}")
        return []

def _ddp_setup(rank: int, world_size: int, port: int):
    if world_size <= 1:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(int(port)))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _train_worker(local_rank: int, world_size: int, config: dict, args, output_dir: str, visible_devices: list[int]):
    # map local_rank -> real cuda device id
    use_cuda = torch.cuda.is_available() and len(visible_devices) > 0
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    rank = int(local_rank)
    port = int(config.get("training", {}).get("ddp_port", 29500))
    _ddp_setup(rank=rank, world_size=world_size, port=port)
    is_rank0 = (rank == 0)
    
    # Initialize LPIPS model once (rank0 only)
    lpips_fn = None
    if LPIPS is not None and is_rank0:
        try:
            lpips_fn = LPIPS(net='vgg').to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            print("[LPIPS] Initialized VGG LPIPS model (rank0)")
        except Exception as e:
            lpips_fn = None
            print(f"[LPIPS] Not available ({e}); LPIPS loss disabled")

    # TensorBoard (rank0 only)
    writer = None
    if is_rank0:
        log_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    # Dataset and loaders
    sequence_length = config.get('training', {}).get('sequence_length', config.get('training', {}).get('num_frames', 2))
    window_size = sequence_length
    data_cfg = config.get("data", {})
    dataset = VoxelFF4DGSDataset(
        pi3_results_dir=data_cfg['pi3_results_dir'],
        seganymo_dir=data_cfg['seganymo_dir'],
        reloc3r_dir=data_cfg['reloc3r_dir'],
        image_size=tuple(data_cfg['image_size']),
        use_sim3_normalize=data_cfg.get('use_sim3_normalize', False),
        window_size=window_size,
        preload_all=bool(data_cfg.get("preload_all", True)),
        sample_interval=int(data_cfg.get("sample_interval", 30)),
        image_dir=data_cfg.get("image_dir", None),
        flow_fwd_path=data_cfg.get("flow_fwd_path", None),
    )

    # Deterministic, non-overlapping split by contiguous indices to avoid any
    # possible window overlap between train/val (especially when windows use
    # consecutive time steps). We take the first 90% samples for training and
    # the remaining 10% for validation.
    _total = len(dataset)
    train_size = int(0.9 * _total)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, _total))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # TimeWindowSampler keeps time steps consecutive
    from FF4DGSMotion.data.dataset import TimeWindowSampler, WindowBatchSampler
    window_sampler = TimeWindowSampler(train_dataset, window_size=window_size, shuffle=False)
    windows = list(getattr(window_sampler, "windows", list(window_sampler)))

    if world_size > 1:
        batch_sampler = DistributedWindowBatchSampler(
            windows=windows,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=int(config.get("training", {}).get("seed", 0)),
        )
    else:
        batch_sampler = WindowBatchSampler(window_sampler, shuffle=True)

    # mp ctx
    mp_ctx = "spawn" if (torch.cuda.is_available() and int(config['training']['num_workers']) > 0) else None
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=int(config['training']['num_workers']),
        multiprocessing_context=mp_ctx,
        pin_memory=torch.cuda.is_available(),
    )

    # Validation loader (rank0 only to save compute)
    val_loader = None
    if is_rank0:
        from FF4DGSMotion.data.dataset import TimeWindowSampler, WindowBatchSampler
        # Ensure we have at least one validation window even when val set shorter than training window_size
        val_window_size = min(window_size, len(val_dataset)) if len(val_dataset) > 0 else 0
        if val_window_size == 0:
            val_loader = None
        else:
            val_window_sampler = TimeWindowSampler(val_dataset, window_size=val_window_size, shuffle=False)
            # If dataset too sparse (e.g., large sample_interval), sampler may yield 0 windows.
            has_windows = len(getattr(val_window_sampler, "windows", list(val_window_sampler))) > 0
            if has_windows:
                val_batch_sampler = WindowBatchSampler(val_window_sampler, shuffle=False)
                val_loader = DataLoader(
                    val_dataset,
                    batch_sampler=val_batch_sampler,
                    num_workers=int(config['training']['num_workers']),
                    multiprocessing_context=mp_ctx,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                # Fallback: simple sequential loader with batch_size = 1
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=int(config['training']['num_workers']),
                    multiprocessing_context=mp_ctx,
                    pin_memory=torch.cuda.is_available(),
                )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=int(config['training']['num_workers']),
            multiprocessing_context=mp_ctx,
            pin_memory=torch.cuda.is_available(),
        )
    # Model
    mc = config.get('model', {})
    model = Trellis4DGS4DCanonical(cfg=mc).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    photo_loss_fn = PhotoConsistencyLoss(loss_type='l1')

    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    # Build optimizer with param groups so we can downscale imp_head lr at Stage 3
    imp_modules = [getattr(model, 'imp_head', None), getattr(model, 'imp_time_emb', None), getattr(model, 'imp_view_emb', None)]
    imp_params = []
    for m in imp_modules:
        if m is not None:
            imp_params.extend(list(p for p in m.parameters() if p.requires_grad))
    imp_param_ids = {id(p) for p in imp_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in imp_param_ids]
    param_groups = []
    if other_params:
        param_groups.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'base', 'base_lr': lr})
    if imp_params:
        param_groups.append({'params': imp_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'imp', 'base_lr': lr})
    optimizer = optim.Adam(param_groups)

    has_fp16_params = any(p.dtype == torch.float16 for p in model.parameters())
    use_amp = torch.cuda.is_available() and (not has_fp16_params)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    if args.resume and is_rank0:
        checkpoint = torch.load(args.resume, map_location=device,weights_only=False)
        sd = checkpoint.get('model_state_dict', checkpoint)
        target_model = model.module if world_size > 1 else model
        missing, unexpected = target_model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[Resume] model strict=False: missing={len(missing)} unexpected={len(unexpected)}")
        opt_sd = checkpoint.get('optimizer_state_dict', None)
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
            except ValueError as e:
                print(f"[Resume] Skip loading optimizer state due to param_groups mismatch: {e}")
        # 我们保存的 checkpoint['epoch'] 是 1-based（ep+1）；用户希望“从 Epoch K/.. 开始继续训练”
        # 因此将 start_epoch 设为 (K-1)（0-based），重新开始该 epoch 的训练更稳妥。
        ckpt_epoch = int(checkpoint.get('epoch', 0))
        start_epoch = max(0, ckpt_epoch-1)
        print(f"Resumed from checkpoint epoch={ckpt_epoch} -> start_epoch={start_epoch} (0-based)")
    if world_size > 1:
        dist.barrier()

    num_epochs = int(config['training']['num_epochs'])
    # 跟踪最佳指标：PSNR 最大，photo_loss 和 ssim_loss 最小
    best_psnr = float('-inf')  # PSNR 越大越好
    best_photo_loss = float('inf')  # photo_loss 越小越好
    best_ssim_loss = float('inf')  # ssim_loss 越小越好

    for ep in range(start_epoch, num_epochs):
        if world_size > 1 and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(ep)
        if is_rank0:
            print(f"\nEpoch {ep + 1}/{num_epochs} (world_size={world_size})")

        train_metrics = train_epoch(
            model.module if world_size > 1 else model,
            train_loader, optimizer, scaler,
            photo_loss_fn, lpips_fn, device, config, writer, ep,
            ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
            output_dir=output_dir,
            is_rank0=is_rank0,
        )

        # Only rank0 runs validation + saves checkpoints
        if is_rank0:
            train_keys = [
                'photo_loss','ssim_loss','lpips_loss','alpha_sparsity','chamfer_loss','motion_reg',
                'temporal_smooth','imp_budget_loss','imp_geom_loss','exist_budget_loss',
                'flow_reproj_loss','residual_mag','residual_smooth'
            ]
            # 打印训练指标（只显示总损失和主要子损失）
            train_display = {
                'loss': train_metrics['loss_total'],
                'stage': train_metrics.get('stage', 1),
                'photo_loss': train_metrics['photo_loss'],
                'ssim_loss': train_metrics['ssim_loss'],
                'lpips_loss': train_metrics['lpips_loss'],
                'retarget_loss': train_metrics.get('retarget_loss', 0.0),
                'flow_sparse_loss': train_metrics.get('flow_sparse_loss', 0.0),
                'flow_reproj_loss': train_metrics.get('flow_reproj_loss', 0.0),
                'alpha_sparsity': train_metrics['alpha_sparsity'],
                'kp2d_loss': train_metrics.get('kp2d_loss', 0.0),
                'motion_reg': train_metrics.get('motion_reg', 0.0),
                'psnr': train_metrics.get('psnr', 0.0),
            }
            print(_fmt_metrics("Train", train_display, ['photo_loss','ssim_loss','lpips_loss','retarget_loss','flow_sparse_loss','flow_reproj_loss','alpha_sparsity','kp2d_loss','motion_reg','psnr']))

            if val_loader is not None:
                val_metrics = validate(
                    model.module if world_size > 1 else model,
                    val_loader,
                    photo_loss_fn, lpips_fn, device, config, writer, ep,
                    ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
                )
                # 只打印验证集的主要指标（与训练格式一致）
                val_display_keys = ['photo_loss','ssim_loss','lpips_loss','alpha_sparsity']
                print(_fmt_metrics("Val", val_metrics, val_display_keys))
            else:
                val_metrics = {'loss': float('inf')}

            # Save checkpoint
            ckpt = {
                'epoch': ep + 1,
                'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            # 始终保存 latest.pth
            torch.save(ckpt, os.path.join(output_dir, 'latest.pth'))
            
            # 根据验证指标保存最佳权重（如果有验证数据）
            if val_loader is not None:
                val_psnr = val_metrics.get('psnr', float('-inf'))
                val_photo_loss = val_metrics.get('photo_loss', float('inf'))
                val_ssim_loss = val_metrics.get('ssim_loss', float('inf'))
                
                # 保存 PSNR 最大的权重
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(ckpt, os.path.join(output_dir, 'best_psnr.pth'))
                    print(f"[Checkpoint] Saved best_psnr.pth (PSNR: {val_psnr:.4f})")
                
                # 保存 photo_loss 最小的权重
                if val_photo_loss < best_photo_loss:
                    best_photo_loss = val_photo_loss
                    torch.save(ckpt, os.path.join(output_dir, 'best_photo_loss.pth'))
                    print(f"[Checkpoint] Saved best_photo_loss.pth (photo_loss: {val_photo_loss:.6f})")
                
                # 保存 ssim_loss 最小的权重
                if val_ssim_loss < best_ssim_loss:
                    best_ssim_loss = val_ssim_loss
                    torch.save(ckpt, os.path.join(output_dir, 'best_ssim_loss.pth'))
                    print(f"[Checkpoint] Saved best_ssim_loss.pth (ssim_loss: {val_ssim_loss:.6f})")

        if world_size > 1:
            dist.barrier()

    if writer is not None:
        writer.close()
    _ddp_cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train Baseline 4DGS')
    parser.add_argument('--hf_dir', type=str, default='weights/TRELLIS-image-large', help='Directory containing HuggingFace weights (ckpts)')
    parser.add_argument('--freeze_flow', action='store_true', help='Freeze flow module parameters')
    parser.add_argument('--freeze_decoder', action='store_true', help='Freeze decoder module parameters')
    parser.add_argument('--freeze_motion', action='store_true', help='Freeze motion head parameters (gz_proj + motion_head)')
    parser.add_argument('--config', type=str, default='configs/anchorwarp_4dgs.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (if not provided, will create timestamped subdirectory)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for validation (should equal T, overrides config)')
    # GPU control (CLI): prefer this over YAML; also works with CUDA_VISIBLE_DEVICES env.
    parser.add_argument('--cuda', type=str, default=None, help='GPU list like \"0\" or \"0,1\"; use \"cpu\" to force CPU. If omitted, uses CUDA_VISIBLE_DEVICES or default cuda:0.')
    parser.add_argument('--ddp_port', type=int, default=None, help='DDP master port override (default: config.training.ddp_port or 29500)')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    # Output dir
    base_dir = args.output_dir or 'results_train/AnchorWarp4DGS/train'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and copy config
    config = load_config(args.config)
    import shutil
    shutil.copy2(args.config, os.path.join(args.output_dir, 'config.yaml'))
    # Backup key source files for this run
    try:
        backup_dir = os.path.join(args.output_dir, 'code')
        os.makedirs(backup_dir, exist_ok=True)
        to_backup = [
            ('FF4DGSMotion/models/FF4DGSMotion.py', 'FF4DGSMotion.py'),
            ('step2_train_4DGSFFMotion.py', 'step2_train_4DGSFFMotion.py'),
            ('configs/ff4dgsmotion.yaml', 'ff4dgsmotion.yaml'),
        ]
        for src, name in to_backup:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(backup_dir, name))
    except Exception as e:
        print(f"[Backup] Skipped code backup due to error: {e}")
    # DDP device selection from CLI/env
    train_cfg = config.get("training", {})
    if args.ddp_port is not None:
        train_cfg["ddp_port"] = int(args.ddp_port)
        config["training"] = train_cfg

    if args.cuda is not None:
        cuda_arg = str(args.cuda).strip()
        if cuda_arg.lower() in ("cpu", "none", ""):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_arg

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_env is None or visible_env.strip() == "":
        cuda_devices = []
    else:
        cuda_devices = _parse_cuda_devices({"cuda": visible_env})
    if cuda_devices:
        print(f"[CUDA] visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    world_size = max(1, len(cuda_devices)) if (torch.cuda.is_available() and cuda_devices) else 1
    if world_size > 1:
        print(f"[DDP] Spawning {world_size} processes")
        mp.spawn(_train_worker, args=(world_size, config, args, args.output_dir, cuda_devices), nprocs=world_size, join=True)
        return

    # single process
    _train_worker(0, 1, config, args, args.output_dir, cuda_devices)
    return



if __name__ == '__main__':
    main()
