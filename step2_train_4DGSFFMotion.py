"""
Training script for Baseline 4DGS (GaussianTransformerBaselineModel)
- Removes Canonical/Anchor/Motion layers and all dynamic mask logic
- Trains only with image supervision (L1/SSIM via PhotoConsistencyLoss) and optional temporal smoothness
"""
import os
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
from torchvision.utils import save_image
import math
from typing import List

from FF4DGSMotion.data.dataset import VoxelFF4DGSDataset
from FF4DGSMotion.models.FF4DGSMotion import Trellis4DGS4DCanonical

from FF4DGSMotion.losses.photo_loss import PhotoConsistencyLoss
from fused_ssim import fused_ssim
import torch.nn.functional as F
from FF4DGSMotion.camera.camera import IntrinsicsCamera
from FF4DGSMotion.diff_renderer.gaussian import render_gs, GaussianAttributes
from FF4DGSMotion.models._utils import matrix_to_quaternion, rgb2sh0

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
    if feat_2d_batch is not None:
        feat_2d_batch = feat_2d_batch.to(device, non_blocking=True)  # [B,V,H',W',C]

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
    points_3d = points.reshape(T, V, H * W, 3).mean(dim=1) if points is not None else torch.zeros(T, 1, 3, device=device)  # [T,N,3]

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
            keypoints_2d = keypoints_2d.to(device, non_blocking=True).float()
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
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    photo_loss_fn: PhotoConsistencyLoss,
    device: torch.device,
    config: dict,
    writer: Optional[SummaryWriter],
    epoch: int,
    ex4dgs_dir: Optional[str],
    output_dir: Optional[str],
) -> dict:
    model.train()
    total_loss = 0.0
    sum_photo = 0.0
    sum_ssim = 0.0
    sum_sil = 0.0
    sum_alpha = 0.0
    sum_chamfer = 0.0
    sum_motion = 0.0
    sum_temporal = 0.0
    n_batches = 0

    last_rendered_images = None
    last_gaussian_params = None
    last_gt_images = None
    last_time_indices = None

    lambda_smooth = float(config.get('loss_weights', {}).get('dynamic_temporal', 0.0))

    # ----------------------------
    # Three-stage training schedule
    # ----------------------------
    stage_cfg = config.get("training", {}).get("stages", {}) or {}
    stage1_epochs = int(stage_cfg.get("stage1_epochs", 5))
    stage2_epochs = int(stage_cfg.get("stage2_epochs", 20))
    # stage 3: epoch >= stage1_epochs + stage2_epochs
    if epoch < stage1_epochs:
        stage = 1
    elif epoch < stage1_epochs + stage2_epochs:
        stage = 2
    else:
        stage = 3

    # Freeze/unfreeze modules per stage
    def _set_requires_grad(module: nn.Module, enabled: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = enabled

    # Stage 1: freeze motion heads
    if stage == 1:
        _set_requires_grad(getattr(model, "dynamic_anchor_motion", None), False)
        _set_requires_grad(getattr(model, "residual_motion_head", None), False)
    # Stage 2: unfreeze anchor motion, keep residual frozen
    elif stage == 2:
        _set_requires_grad(getattr(model, "dynamic_anchor_motion", None), True)
        _set_requires_grad(getattr(model, "residual_motion_head", None), False)
    # Stage 3: unfreeze both
    else:
        _set_requires_grad(getattr(model, "dynamic_anchor_motion", None), True)
        _set_requires_grad(getattr(model, "residual_motion_head", None), True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch_data = prepare_batch(
            batch, device,
            ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
            image_size=tuple(config['data']['image_size']),
            config=config, mode='train'
        )
        points = batch_data.get('points', None)
        points_3d = batch_data['points_3d']
        feat_2d = batch_data['feat_2d']
        conf_seq = batch_data['conf']
        dyn_mask_tv = batch_data.get('dynamic_conf_tv', None)
        dyn_traj_tv = batch_data.get('dynamic_traj_tv', None)
        keypoints_2d = batch_data.get('keypoints_2d', None)
        silhouette = batch_data.get('silhouette', None)
        camera_poses_seq = batch_data['camera_poses']
        camera_intrinsics_seq = batch_data['camera_intrinsics']
        time_ids = batch_data['time_ids']
        gt_images_raw = batch_data.get('gt_images', None)
        gt_images = gt_images_raw.permute(0, 1, 4, 2, 3).contiguous() if gt_images_raw is not None else None
        T = points_3d.shape[0]

        optimizer.zero_grad(set_to_none=True)
        use_amp_ctx = autocast() if scaler.is_enabled() else nullcontext()
        with use_amp_ctx:
            dyn_mask_in = dyn_mask_tv if dyn_mask_tv is not None else silhouette
            output = model(
                points_full=points,
                feat_2d=feat_2d,
                camera_poses=camera_poses_seq,
                camera_K=camera_intrinsics_seq,
                time_ids=time_ids,
                dyn_mask_2d=dyn_mask_in,
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
        dxyz_t = output.get('dxyz_t', None)
        dxyz_t_res = output.get("dxyz_t_res", None)
        sim3_s, sim3_R, sim3_t = output.get('sim3_s', None), output.get('sim3_R', None), output.get('sim3_t', None)
        dtype = mu_t.dtype
        M = mu_t.shape[1]

        H_t, W_t = config['data']['image_size']
        rendered_images_list = []
        rendered_images_static_list = []
        rendered_images_dynamic_list = []

        # 颜色初始化：第 1 个 epoch（epoch==0）使用 target_image 估计每个 Gaussian 的 DC 颜色（参考 test_render.py）
        # 注意：这里“写进模型”的方式是把它作为 epoch0 的监督信号，让 gaussian_head 学到合适的初始颜色；
        #      之后颜色仍由网络输出并参与更新（不会在后续 epoch 被固定覆盖）。
        do_color_init = bool(config.get('training', {}).get('color_init_first_epoch', True)) and (epoch == 0)
        debug_color_init = bool(config.get('training', {}).get('debug_color_init', False))
        # 颜色初始化策略：不是监督，而是“初始化引导渲染”，并逐步退火到网络自身输出
        # blend(t) 从 color_init_blend_start 线性衰减到 color_init_blend_end（按 step 计算）
        blend_start = float(config.get('training', {}).get('color_init_blend_start', 1.0))
        blend_end = float(config.get('training', {}).get('color_init_blend_end', 0.0))
        blend_steps = int(config.get('training', {}).get('color_init_blend_steps', 2000))
        # 当前 epoch 的全局 step（仅用于 blend 调度，不影响 optimizer step）
        global_step = epoch * len(dataloader) + batch_idx

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

            bg_color = torch.ones(3, device=device)
            num_gs = int(mu_frame.shape[0])
            C0 = 0.28209479177387814
            # 在 fast_forward init pass 中，render_gs CUDA kernel 一旦报错通常会延迟到后续同步点才抛出。
            # 这里提供一个开关，强制每次 render_gs 后同步，便于定位且避免异步错误污染后续计算。
            color_init_sync_cuda = bool(config.get("training", {}).get("color_init_sync_cuda", True))

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
            sh0_pred = rgb2sh0(color_frame_pred).to(dtype=torch.float32).contiguous()  # [M,3] (DC)
            sh_in = sh0_pred.to(dtype=mu_frame.dtype).unsqueeze(1)  # [M,1,3] for render_gs input

            # collect per-view estimates like test_gs_render.py, then do fast_forward on sh0_pred
            est_color_list = []
            est_weight_list = []

            for vi in range(camera_poses_t.shape[0]):
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
                    xyz=mu_frame,
                    opacity=opacity_init,
                    scaling=scale_init,
                    rotation=rotation,
                    sh=sh_in,
                )

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
                        torch.zeros(num_gs, device=sh0_pred.device, dtype=torch.bool),
                        sh0_pred,  # [M,3] in-place
                    )
            else:
                # Fallback: weighted average in SH0 space (less robust than cuda fast_forward)
                sh0_pred = (est_color_avg / est_weight_avg.clamp_min(1e-8)).contiguous()

            return sh0_pred.to(device=mu_frame.device, dtype=mu_frame.dtype), est_weight_avg.to(device=mu_frame.device, dtype=mu_frame.dtype)

        def render_set(mu_frame, scale_frame, rot_frame, color_frame, alpha_frame, camera_poses_t, camera_intrinsics_t, target_images_t=None):
            bg_color = torch.ones(3, device=device)
            imgs = []
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
            sh_pred = rgb2sh0(color_frame.to(dtype=mu_frame.dtype).clamp(0.0, 1.0))  # [M,3]
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
            sh_for_render = sh_render.unsqueeze(1)  # [M,1,3]
            for vi in range(camera_poses_t.shape[0]):
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
                res_v = render_gs(camera=cam, bg_color=bg_color, gs=gs_attrs, target_image=None, sh_degree=0, scaling_modifier=1.0)
                imgs.append(res_v["color"])  # [3,H,W]
            return torch.stack(imgs, dim=0), init_sh0, init_weight  # ([V,3,H,W], [M,3] or None, [M,1] or None)

        for t in range(T):
            camera_poses_t = camera_poses_seq[t]
            camera_intrinsics_t = camera_intrinsics_seq[t]
            rot_frame = rot_t[t] if rot_t is not None else None
            tgt_tv = gt_images[t] if (gt_images is not None and do_color_init) else None
            imgs_v, init_sh0, init_weight = render_set(
                mu_t[t], scale_t[t], rot_frame, color_t[t], alpha_t[t],
                camera_poses_t, camera_intrinsics_t,
                target_images_t=tgt_tv,
            )
            rendered_images_list.append(imgs_v.unsqueeze(0))
            if has_split:
                rot_s_frame = rot_t_s[t] if rot_t_s is not None else None
                rot_d_frame = rot_t_d[t] if rot_t_d is not None else None
                imgs_s, _, _ = render_set(mu_t_s[t], scale_t_s[t], rot_s_frame, color_t_s[t], alpha_t_s[t], camera_poses_t, camera_intrinsics_t, target_images_t=None)
                imgs_d, _, _ = render_set(mu_t_d[t], scale_t_d[t], rot_d_frame, color_t_d[t], alpha_t_d[t], camera_poses_t, camera_intrinsics_t, target_images_t=None)
                rendered_images_static_list.append(imgs_s.unsqueeze(0))
                rendered_images_dynamic_list.append(imgs_d.unsqueeze(0))

            # 不再做“颜色初始化监督 loss”

        rendered_images = torch.cat(rendered_images_list, dim=0)  # [T,V,3,H,W]
        rendered_images_static = torch.cat(rendered_images_static_list, dim=0) if has_split else None  # [T,V,3,H,W]
        rendered_images_dynamic = torch.cat(rendered_images_dynamic_list, dim=0) if has_split else None  # [T,V,3,H,W]
        # SimpleGaussianModel + render_gs_batch 暂时不返回 alpha/depth，后面 Silhouette / alpha 正则可先置 0
        rendered_alpha = None


        # Loss weights
        lw = config.get('loss_weights', {})
        w_photo = float(lw.get('photo_l1', 1.0))
        w_ssim = float(lw.get('ssim', 0.0))
        w_sil = float(lw.get('silhouette', 0.0))
        w_alpha = float(lw.get('alpha_l1', 0.0))
        w_chamfer = float(lw.get('chamfer', 0.0))
        w_motion = float(lw.get('motion_reg', 0.0))
        w_temp_vel = float(lw.get('temporal_vel', 0.0))
        w_temp_acc = float(lw.get('temporal_acc', 0.0))
        w_cycle = float(lw.get('cycle_consistency', 0.0))

        # Override weights per stage (per your design)
        stage_w = stage_cfg.get(f"stage{stage}_weights", {}) or {}
        # Stage 1: only photo/ssim + very small chamfer; disable dyn/motion/temporal
        if stage == 1:
            w_motion = 0.0
            lambda_smooth = 0.0
            w_dyn = 0.0
            w_sta = 0.0
            w_cross = 0.0
            # use provided stage override or clamp to very small
            w_chamfer = float(stage_w.get("chamfer", min(w_chamfer, 0.01)))
        # Stage 2: enable motion reg + temporal smooth + dyn/static losses
        elif stage == 2:
            w_motion = float(stage_w.get("motion_reg", w_motion))
            lambda_smooth = float(stage_w.get("temporal_smooth", lambda_smooth))
            w_dyn = float(stage_w.get("mask_dyn", w_dyn))
            w_sta = float(stage_w.get("mask_sta", w_sta))
            w_cross = float(stage_w.get("mask_cross", w_cross))
            w_chamfer = float(stage_w.get("chamfer", w_chamfer))
        # Stage 3: same as stage2 plus residual regs (handled below)
        else:
            w_motion = float(stage_w.get("motion_reg", w_motion))
            lambda_smooth = float(stage_w.get("temporal_smooth", lambda_smooth))
            w_dyn = float(stage_w.get("mask_dyn", w_dyn))
            w_sta = float(stage_w.get("mask_sta", w_sta))
            w_cross = float(stage_w.get("mask_cross", w_cross))
            w_chamfer = float(stage_w.get("chamfer", w_chamfer))

        # Photo Loss: [T,V,3,H,W]
        if gt_images is not None and w_photo > 0:
            pred_chw = rendered_images    # [T,V,3,H,W]
            tgt_chw  = gt_images          # [T,V,3,H,W]
            Tv, Th, Tw = pred_chw.shape[1], pred_chw.shape[3], pred_chw.shape[4]
            pred_bchw = pred_chw.reshape(-1, 3, Th, Tw)
            tgt_bchw  = tgt_chw.reshape(-1, 3, Th, Tw)
            photo_loss = torch.nn.functional.l1_loss(pred_bchw, tgt_bchw, reduction='mean')
        else:
            photo_loss = torch.tensor(0.0, device=device)

        # Mask-decomposed losses (Stage 4.1): requires static/dynamic rendering + dyn mask
        w_dyn = float(lw.get('mask_dyn', 0.0))
        w_sta = float(lw.get('mask_sta', 0.0))
        w_cross = float(lw.get('mask_cross', 0.0))
        mask_dyn_loss = torch.tensor(0.0, device=device)
        mask_sta_loss = torch.tensor(0.0, device=device)
        mask_cross_loss = torch.tensor(0.0, device=device)
        if gt_images is not None and has_split and (w_dyn > 0 or w_sta > 0 or w_cross > 0):
            mask = dyn_mask_tv if dyn_mask_tv is not None else silhouette
            if mask is not None:
                if mask.dim() == 5 and mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                mask = mask.to(device=device, dtype=rendered_images.dtype).clamp(0.0, 1.0)  # [T,V,H,W]
                # resize if needed
                if mask.shape[-2:] != rendered_images.shape[-2:]:
                    Tm, Vm, Hm, Wm = mask.shape
                    Hr, Wr = rendered_images.shape[-2:]
                    mask_b = mask.reshape(-1, 1, Hm, Wm)
                    mask_b = F.interpolate(mask_b, size=(Hr, Wr), mode='nearest')
                    mask = mask_b.reshape(Tm, Vm, Hr, Wr)
                m_dyn = mask.unsqueeze(2)  # [T,V,1,H,W]
                m_sta = (1.0 - mask).unsqueeze(2)
                err = torch.abs(rendered_images - gt_images)
                if w_dyn > 0:
                    mask_dyn_loss = (m_dyn * err).mean()
                if w_sta > 0:
                    mask_sta_loss = (m_sta * err).mean()
                if w_cross > 0 and rendered_images_static is not None and rendered_images_dynamic is not None:
                    mask_cross_loss = (m_dyn * torch.abs(rendered_images_static)).mean() + (m_sta * torch.abs(rendered_images_dynamic)).mean()

        # SSIM Loss (1 - SSIM) using fused_ssim on [B,3,H,W]
        if gt_images is not None and w_ssim > 0:
            # Both tensors are assumed [T,V,3,H,W]
            Tn, Vn, Cn, Hn, Wn = rendered_images.shape
            pred_bchw = rendered_images.reshape(-1, Cn, Hn, Wn).clamp(0,1)
            gt_bchw   = gt_images.reshape(-1, Cn, Hn, Wn).clamp(0,1)
            ssim_val = ssim_torch(pred_bchw, gt_bchw, window_size=11)
            ssim_loss = (1.0 - ssim_val.mean())
        else:
            ssim_loss = torch.tensor(0.0, device=device)

        # Silhouette Loss
        silhouette = None #batch_data.get('silhouette', None)
        if silhouette is not None and w_sil > 0:
            sil = silhouette.clamp(0, 1)

            # Prepare rendered alpha to be 4D [T,V,H,W]
            ra = rendered_alpha
            if ra.dim() == 5 and ra.shape[-1] == 1:
                ra = ra.squeeze(-1)

            # Step 1: Ensure sil is 4D [T,V,H,W] (remove trailing singleton dims if present)
            while sil.dim() > 4:
                sil = sil.squeeze(-1)

            # Step 2: Align spatial resolution to rendered alpha (BEFORE morphology)
            if sil.shape[-2:] != ra.shape[-2:]:
                T_, V_, Hs, Ws = sil.shape
                Hr, Wr = ra.shape[-2:]
                sil_b = sil.reshape(-1, 1, Hs, Ws).float()
                sil_b = F.interpolate(sil_b, size=(Hr, Wr), mode='nearest')
                sil = sil_b.reshape(T_, V_, Hr, Wr)

            # Step 3: Apply optional morphology (dilate/erode) on aligned resolution
            morph_cfg = config.get('loss_params', {}).get('silhouette_morph', {})
            op = str(morph_cfg.get('op', 'none')).lower()
            k = int(morph_cfg.get('k', 3))
            if k % 2 == 0:
                k = k + 1

            if op in ['dilate', 'erode'] and k >= 1:
                T_, V_, H_, W_ = sil.shape
                sil_b = sil.reshape(-1, 1, H_, W_)
                pad = k // 2
                if op == 'dilate':
                    sil_b = F.max_pool2d(sil_b, kernel_size=k, stride=1, padding=pad)
                else:  # erode
                    sil_b = 1.0 - F.max_pool2d(1.0 - sil_b, kernel_size=k, stride=1, padding=pad)
                sil = sil_b.reshape(T_, V_, H_, W_)

            # Step 4: Ensure dtype matches before computing loss
            sil = sil.to(dtype=ra.dtype, device=ra.device)

            # Final safety alignment: handle any leftover mismatch (transpose or resize)
            if sil.shape != ra.shape:
                # try swapped H/W
                if sil.shape[-2] == ra.shape[-1] and sil.shape[-1] == ra.shape[-2]:
                    sil = sil.permute(0,1,3,2)
                # resize if still mismatched
                if sil.shape[-2:] != ra.shape[-2:]:
                    T_, V_, Hs, Ws = sil.shape
                    Hr, Wr = ra.shape[-2:]
                    sil_b = sil.reshape(-1, 1, Hs, Ws).float()
                    sil_b = F.interpolate(sil_b, size=(Hr, Wr), mode='nearest')
                    sil = sil_b.reshape(T_, V_, Hr, Wr)

            # Step 5: Compute loss
            sil_loss = torch.abs(ra - sil).mean()
        else:
            sil_loss = torch.tensor(0.0, device=device)

        # Alpha sparsity (per-Gaussian)
        # if w_alpha > 0:
        #     alpha_sparse = torch.abs(alpha_t).mean()
        # else:
        alpha_sparse = torch.tensor(0.0, device=device)

        # Motion regularization using SegAnyMo dynamic confidence (lower conf -> stronger penalty)

        color_init_loss = torch.tensor(0.0, device=device)
        motion_reg = torch.tensor(0.0, device=device)
        if w_motion > 0 and dxyz_t is not None:
            dyn_conf = batch_data.get('dynamic_conf', None)  # [T,H,W]
            if dyn_conf is not None:
                w_t = 1.0 - dyn_conf.mean(dim=(1,2))  # [T]
                # per-frame mean L2 of displacements
                per_t = (dxyz_t.pow(2).sum(dim=2).mean(dim=1))  # [T]
                motion_reg = (w_t * per_t).mean()
            else:
                motion_reg = (dxyz_t.pow(2).sum(dim=2).mean())

        # Stage 3: residual motion regularization (stronger than anchor motion)
        w_res_mag = float(stage_cfg.get("stage3_residual_mag", 0.0)) if stage == 3 else 0.0
        w_res_smooth = float(stage_cfg.get("stage3_residual_smooth", 0.0)) if stage == 3 else 0.0
        res_mag = torch.tensor(0.0, device=device)
        res_smooth = torch.tensor(0.0, device=device)
        if stage == 3 and dxyz_t_res is not None:
            # magnitude
            if w_res_mag > 0:
                res_mag = dxyz_t_res.pow(2).sum(dim=-1).mean()
            # first-order temporal smoothness on residual
            if w_res_smooth > 0 and dxyz_t_res.shape[0] >= 2:
                res_smooth = torch.abs(dxyz_t_res[1:] - dxyz_t_res[:-1]).mean()

        # Simple/Weighted Chamfer (per frame, using points_3d transformed by Sim3)
        chamfer_loss = torch.tensor(0.0, device=device)
        if w_chamfer > 0 and sim3_s is not None and points_3d is not None and points_3d.numel() > 0:
            ch_cfg = config.get('loss_params', {}).get('chamfer', {})
            weighted = bool(ch_cfg.get('weighted', True))
            use_alpha = bool(ch_cfg.get('use_alpha', True))
            use_vis = bool(ch_cfg.get('use_visibility', True))
            max_m = int(ch_cfg.get('max_pairs', 4096))
            eps = 1e-8
            vis_map = batch_data.get('visibility', None)  # [T,H,W]
            for t in range(T):
                mu = mu_t[t]  # [M,3]
                pts = points_3d[t]  # [N,3] with N=H*W
                if pts.numel() == 0:
                    continue
                # Transform points by Sim3
                s_t = sim3_s[t]
                R_t = sim3_R[t]
                t_t = sim3_t[t]
                pts_can = s_t * (pts @ R_t.T) + t_t  # [N,3]
                # subsample
                m_idx = torch.randperm(mu.shape[0], device=device)[:min(mu.shape[0], max_m)]
                n_idx = torch.randperm(pts_can.shape[0], device=device)[:min(pts_can.shape[0], max_m)]
                mu_s = mu[m_idx]
                pts_s = pts_can[n_idx]
                dists = torch.cdist(mu_s, pts_s, p=2)
                # weights
                if weighted:
                    # mu weights from alpha
                    if use_alpha:
                        w_mu = alpha_t[t].squeeze(-1)[m_idx].clamp(min=0.0)
                    else:
                        w_mu = torch.ones_like(m_idx, dtype=dtype, device=device)
                    # pts weights from visibility
                    if use_vis and vis_map is not None:
                        H = vis_map.shape[1]
                        W = vis_map.shape[2]
                        w_pts_full = vis_map[t].reshape(-1)  # [N]
                        w_pts = w_pts_full[n_idx].clamp(min=0.0)
                    else:
                        w_pts = torch.ones_like(n_idx, dtype=dtype, device=device)
                    # mu->pts
                    min_mu = dists.min(dim=1).values
                    chamfer_mu = (w_mu * min_mu).sum() / (w_mu.sum() + eps)
                    # pts->mu
                    min_pts = dists.min(dim=0).values
                    chamfer_pts = (w_pts * min_pts).sum() / (w_pts.sum() + eps)
                    chamfer_t = chamfer_mu + chamfer_pts
                else:
                    chamfer_t = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
                chamfer_loss = chamfer_loss + chamfer_t
            chamfer_loss = chamfer_loss / max(T,1)

        temporal_smooth = torch.tensor(0.0, device=device)
        if lambda_smooth > 0 and T >= 3:
            # simple 2nd-order smoothness on positions
            x_prev = mu_t[:-2]
            x_curr = mu_t[1:-1]
            x_next = mu_t[2:]
            temporal_smooth = ((x_next - 2 * x_curr + x_prev) ** 2).mean()

        loss = (
            w_photo * photo_loss +
            w_ssim * ssim_loss +
            w_sil * sil_loss +
            w_alpha * alpha_sparse +
            w_chamfer * chamfer_loss +
            w_motion * motion_reg +
            lambda_smooth * temporal_smooth
            + w_dyn * mask_dyn_loss
            + w_sta * mask_sta_loss
            + w_cross * mask_cross_loss
            + w_res_mag * res_mag
            + w_res_smooth * res_smooth
        )
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
        sum_sil += float(sil_loss.item())
        sum_alpha += float(alpha_sparse.item())
        sum_chamfer += float(chamfer_loss.item())
        sum_motion += float(motion_reg.item())
        sum_temporal += float(temporal_smooth.item())
        n_batches += 1

        # Save last batch artifacts
        last_rendered_images = rendered_images.detach().cpu()
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
            writer.add_scalar('Loss/Silhouette', sum_sil / denom, global_step)
            writer.add_scalar('Loss/AlphaSparse', sum_alpha / denom, global_step)
            writer.add_scalar('Loss/Chamfer', sum_chamfer / denom, global_step)
            writer.add_scalar('Loss/MotionReg', sum_motion / denom, global_step)
            writer.add_scalar('Loss/TemporalSmooth', sum_temporal / denom, global_step)
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
        del rendered_images_list, rendered_images, mu_t, scale_t, color_t, alpha_t, output
        if gt_images is not None:
            del gt_images
        torch.cuda.empty_cache()

    # Save grid image and PLYs for last batch
    if output_dir is not None and last_rendered_images is not None:
        images_dir = os.path.join(output_dir, 'epoch_images')
        gaussians_dir = os.path.join(output_dir, 'epoch_gaussians')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(gaussians_dir, exist_ok=True)

        def ensure_tv_chw(img: torch.Tensor) -> torch.Tensor:
            if img is None:
                return None
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
    return {
        'loss': total_loss / denom,
        'photo_loss': sum_photo / denom,
        'ssim_loss': sum_ssim / denom,
        'silhouette_loss': sum_sil / denom,
        'alpha_sparsity': sum_alpha / denom,
        'chamfer_loss': sum_chamfer / denom,
        'motion_reg': sum_motion / denom,
        'temporal_smooth': sum_temporal / denom,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    photo_loss_fn: PhotoConsistencyLoss,
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
    sum_sil = 0.0
    sum_alpha = 0.0
    sum_chamfer = 0.0
    sum_motion = 0.0
    sum_temporal = 0.0
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
            points_3d = batch_data['points_3d']
            feat_2d = batch_data['feat_2d']
            conf_seq = batch_data['conf']
            dyn_mask_tv = batch_data.get('dynamic_conf_tv', None)
            dyn_traj_tv = batch_data.get('dynamic_traj_tv', None)
            keypoints_2d = batch_data.get('keypoints_2d', None)
            camera_poses_seq = batch_data['camera_poses']
            camera_intrinsics_seq = batch_data['camera_intrinsics']
            time_ids = batch_data['time_ids']
            gt_images = batch_data.get('gt_images', None)
            if gt_images is not None:
                gt_images = gt_images.permute(0, 1, 4, 2, 3).contiguous()  # [T,V,3,H,W]
            silhouette = batch_data.get('silhouette', None)
            T = points_3d.shape[0]
            H_t, W_t = config['data']['image_size']
            rendered_images_list = []

            dyn_mask_in = dyn_mask_tv if dyn_mask_tv is not None else silhouette
            output = model(
                points_full=points,
                feat_2d=feat_2d,
                camera_poses=camera_poses_seq,
                camera_K=camera_intrinsics_seq,
                time_ids=time_ids,
                dyn_mask_2d=dyn_mask_in,
                conf_2d=conf_seq,
            )
            mu_t, scale_t, color_t, alpha_t = output['mu_t'], output['scale_t'], output['color_t'], output['alpha_t']
            rot_t = output.get('rot_t', None)
            dxyz_t = output.get('dxyz_t', None)
            sim3_s, sim3_R, sim3_t = output.get('sim3_s', None), output.get('sim3_R', None), output.get('sim3_t', None)

            rendered_images_list = []
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
                imgs_t = []
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
                    img_v = res_v["color"]  # [3,H,W]
                    imgs_t.append(img_v)
                
                # 堆叠视角 [V,3,H,W] -> 转换为 [V,H,W,3]
                imgs_t_stacked = torch.stack(imgs_t, dim=0)  # [V,3,H,W]
                imgs_t_hwc = imgs_t_stacked.permute(0, 2, 3, 1).contiguous()  # [V,H,W,3]
                rendered_images_list.append(imgs_t_hwc.unsqueeze(0))  # [1,V,H,W,3]
            
            rendered_images = torch.cat(rendered_images_list, dim=0)  # [T,V,H,W,3]
            # Convert to [T,V,3,H,W] for loss calculation
            rendered_images = rendered_images.permute(0, 1, 4, 2, 3).contiguous()  # [T,V,3,H,W]

            # Loss weights
            lw = config.get('loss_weights', {})
            w_photo = float(lw.get('photo_l1', 1.0))
            w_ssim = float(lw.get('ssim', 0.0))
            w_sil = float(lw.get('silhouette', 0.0))
            w_alpha = float(lw.get('alpha_l1', 0.0))
            w_chamfer = float(lw.get('chamfer', 0.0))
            w_motion = float(lw.get('motion_reg', 0.0))

            # Photo Loss: directly use [T,V,3,H,W]
            if gt_images is not None and w_photo > 0:
                pred_chw = rendered_images    # [T,V,3,H,W]
                tgt_chw  = gt_images          # [T,V,3,H,W]
                Tn,Vn,Cn,Hn,Wn = pred_chw.shape
                pred_bchw = pred_chw.reshape(-1, Cn, Hn, Wn)
                tgt_bchw  = tgt_chw.reshape(-1, Cn, Hn, Wn)
                # resize GT to match pred if needed
                if tgt_bchw.shape[-2:] != pred_bchw.shape[-2:]:
                    tgt_bchw = F.interpolate(tgt_bchw, size=pred_bchw.shape[-2:], mode='bilinear', align_corners=False)
                photo_loss = F.l1_loss(pred_bchw.clamp(0,1), tgt_bchw.clamp(0,1), reduction='mean')
            else:
                photo_loss = torch.tensor(0.0, device=device)

            # SSIM with fused_ssim on [B,3,H,W]
            if gt_images is not None and w_ssim > 0:
                Tn,Vn,Cn,Hn,Wn = rendered_images.shape
                pred_bchw = rendered_images.reshape(-1, Cn, Hn, Wn).clamp(0,1)
                gt_bchw   = gt_images.reshape(-1, Cn, Hn, Wn).clamp(0,1)
                if gt_bchw.shape[-2:] != pred_bchw.shape[-2:]:
                    gt_bchw = F.interpolate(gt_bchw, size=pred_bchw.shape[-2:], mode='bilinear', align_corners=False)
                ssim_val = ssim_torch(pred_bchw, gt_bchw, window_size=11)
                ssim_loss = (1.0 - ssim_val.mean())
            else:
                ssim_loss = torch.tensor(0.0, device=device)

            # Silhouette (disabled - no rendered_alpha available)
            sil_loss = torch.tensor(0.0, device=device)

            # Alpha sparsity
            alpha_sparse = torch.abs(alpha_t).mean() if w_alpha > 0 else torch.tensor(0.0, device=device)

            # Motion regularization using SegAnyMo dynamic confidence (lower conf -> stronger penalty)
            motion_reg = torch.tensor(0.0, device=device)
            if w_motion > 0 and dxyz_t is not None:
                dyn_conf = batch_data.get('dynamic_conf', None)  # [T,H,W]
                if dyn_conf is not None:
                    w_t = 1.0 - dyn_conf.mean(dim=(1,2))  # [T]
                    # per-frame mean L2 of displacements
                    per_t = (dxyz_t.pow(2).sum(dim=2).mean(dim=1))  # [T]
                    motion_reg = (w_t * per_t).mean()
                else:
                    motion_reg = (dxyz_t.pow(2).sum(dim=2).mean())

            # Chamfer
            chamfer_loss = torch.tensor(0.0, device=device)
            if w_chamfer > 0 and sim3_s is not None and points_3d is not None and points_3d.numel() > 0:
                max_m = 4096
                for t in range(T):
                    mu = mu_t[t]
                    pts = points_3d[t]
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
                w_sil * sil_loss +
                w_alpha * alpha_sparse +
                w_chamfer * chamfer_loss +
                w_motion * motion_reg +
                lambda_smooth * temporal_smooth
            )

            total_loss += loss.item()
            sum_photo += float(photo_loss.item())
            sum_ssim += float(ssim_loss.item())
            sum_sil += float(sil_loss.item())
            sum_alpha += float(alpha_sparse.item())
            sum_chamfer += float(chamfer_loss.item())
            sum_motion += float(motion_reg.item())
            sum_temporal += float(temporal_smooth.item())
            n_batches += 1

            del rendered_images_list, rendered_images, mu_t, scale_t, color_t, alpha_t
            if gt_images is not None:
                del gt_images

    denom = max(1, n_batches)
    return {
        'loss': total_loss / denom,
        'photo_loss': sum_photo / denom,
        'ssim_loss': sum_ssim / denom,
        'silhouette_loss': sum_sil / denom,
        'alpha_sparsity': sum_alpha / denom,
        'chamfer_loss': sum_chamfer / denom,
        'motion_reg': sum_motion / denom,
        'temporal_smooth': sum_temporal / denom,
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
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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
        T_training = sequence_length
        val_batch_size = args.batch_size if args.batch_size is not None else T_training
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    has_fp16_params = any(p.dtype == torch.float16 for p in model.parameters())
    use_amp = torch.cuda.is_available() and (not has_fp16_params)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    if args.resume and is_rank0:
        checkpoint = torch.load(args.resume, map_location=device)
        sd = checkpoint.get('model_state_dict', checkpoint)
        if world_size > 1:
            model.module.load_state_dict(sd)
        else:
            model.load_state_dict(sd)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    if world_size > 1:
        dist.barrier()

    num_epochs = int(config['training']['num_epochs'])
    best_val = float('inf')

    for ep in range(start_epoch, num_epochs):
        if world_size > 1 and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(ep)
        if is_rank0:
            print(f"\nEpoch {ep + 1}/{num_epochs} (world_size={world_size})")

        train_metrics = train_epoch(
            model.module if world_size > 1 else model,
            train_loader, optimizer, scaler,
            photo_loss_fn, device, config, writer, ep,
            ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
            output_dir=output_dir,
        )

        # Only rank0 runs validation + saves checkpoints
        if is_rank0:
            print(f"Train Loss: {train_metrics['loss']:.4f} | Photo: {train_metrics['photo_loss']:.4f} | SSIM: {train_metrics['ssim_loss']:.4f} | Alpha: {train_metrics['alpha_sparsity']:.4f} | Silhouette: {train_metrics['silhouette_loss']:.4f} | Chamfer: {train_metrics['chamfer_loss']:.4f} | MotionReg: {train_metrics['motion_reg']:.4f} | TemporalSmooth: {train_metrics['temporal_smooth']:.4f}")

            if val_loader is not None:
                val_metrics = validate(
                    model.module if world_size > 1 else model,
                    val_loader,
                    photo_loss_fn, device, config, writer, ep,
                    ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
                )
                print(f"Val   Loss: {val_metrics['loss']:.4f} | Photo: {val_metrics['photo_loss']:.4f} | SSIM: {val_metrics['ssim_loss']:.4f} | Alpha: {val_metrics['alpha_sparsity']:.4f} | Silhouette: {val_metrics['silhouette_loss']:.4f} | Chamfer: {val_metrics['chamfer_loss']:.4f} | MotionReg: {val_metrics['motion_reg']:.4f} | TemporalSmooth: {val_metrics['temporal_smooth']:.4f}")
            else:
                val_metrics = {'loss': float('inf')}

            # Save checkpoint
            ckpt = {
                'epoch': ep + 1,
                'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            torch.save(ckpt, os.path.join(output_dir, 'latest.pth'))
            if val_metrics.get('loss', float('inf')) < best_val:
                best_val = float(val_metrics['loss'])
                torch.save(ckpt, os.path.join(output_dir, 'best.pth'))

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

    # Dataset and loaders
    sequence_length = config.get('training', {}).get('sequence_length', config.get('training', {}).get('num_frames', 2))
    window_size = sequence_length
    dataset = VoxelFF4DGSDataset(
        pi3_results_dir=config['data']['pi3_results_dir'],
        seganymo_dir=config['data']['seganymo_dir'],
        reloc3r_dir=config['data']['reloc3r_dir'],
        image_size=tuple(config['data']['image_size']),
        use_sim3_normalize=config.get('data', {}).get('use_sim3_normalize', False),
        window_size=window_size,
        preload_all=bool(config.get('data', {}).get('preload_all', True)),
        sample_interval=int(config.get('data', {}).get('sample_interval', 30)),
        image_dir=config.get('data', {}).get('image_dir', None),
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # TimeWindowSampler keeps time steps consecutive
    from FF4DGSMotion.data.dataset import TimeWindowSampler, WindowBatchSampler
    window_sampler = TimeWindowSampler(train_dataset, window_size=window_size, shuffle=False)
    batch_sampler = WindowBatchSampler(window_sampler, shuffle=True)

    # 重要：CUDA 环境下用 DataLoader(num_workers>0, 默认 fork) 容易触发 “CUDA initialization error/worker aborted”
    # 这里在 CUDA 可用且 num_workers>0 时强制使用 spawn 上下文，避免 fork 继承 CUDA 上下文。
    mp_ctx = "spawn" if (torch.cuda.is_available() and int(config['training']['num_workers']) > 0) else None
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=int(config['training']['num_workers']),
        multiprocessing_context=mp_ctx,
        pin_memory=torch.cuda.is_available(),
    )

    # Validation loader
    sequence_length = config.get('training', {}).get('sequence_length', config.get('training', {}).get('num_frames', 2))
    T_training = sequence_length
    val_batch_size = args.batch_size if args.batch_size is not None else T_training
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=int(config['training']['num_workers']),
        multiprocessing_context=mp_ctx,
        pin_memory=torch.cuda.is_available(),
    )

    # Model: 参数全部来自 configs/ff4dgsmotion.yaml 的 model 字段，结构在 FF4DGSMotion/models/FF4DGSMotion.py 内部构建
    mc = config.get('model', {})
    model = Trellis4DGS4DCanonical(cfg=mc).to(device)

    # Apply fine-grained freezing per CLI flags
    def _set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag
    if args.freeze_flow:
        # coarse/anchor 分支
        if hasattr(model, "feat_reduce"):
            _set_requires_grad(model.feat_reduce, False)
        if hasattr(model, "dual_slot_prior"):
            _set_requires_grad(model.dual_slot_prior, False)
        if hasattr(model, "dyn_pred_token") and (model.dyn_pred_token is not None):
            _set_requires_grad(model.dyn_pred_token, False)
        print("[Freeze] feat_reduce / dual_slot_prior / dyn_pred_token frozen")
    if args.freeze_decoder:
        # 外观解码分支
        if hasattr(model, "point_aggregator"):
            _set_requires_grad(model.point_aggregator, False)
        if hasattr(model, "gaussian_head"):
            _set_requires_grad(model.gaussian_head, False)
        print("[Freeze] point_aggregator and gaussian_head frozen")
    if args.freeze_motion:
        # motion 分支
        if hasattr(model, "dynamic_anchor_motion"):
            _set_requires_grad(model.dynamic_anchor_motion, False)
        if hasattr(model, "residual_motion_head") and (model.residual_motion_head is not None):
            _set_requires_grad(model.residual_motion_head, False)
        print("[Freeze] dynamic_anchor_motion (+ residual_motion_head) frozen")

    # Loss function
    # PhotoConsistencyLoss only supports loss_type ('l1' or 'l2')
    photo_loss_fn = PhotoConsistencyLoss(loss_type='l1')

    # Optimizer
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    # Enable AMP only if model parameters are not in FP16
    has_fp16_params = any(p.dtype == torch.float16 for p in model.parameters())
    use_amp = torch.cuda.is_available() and (not has_fp16_params)
    if has_fp16_params:
        print('[AMP] Detected FP16 model parameters; disabling GradScaler and autocast to avoid unscale errors.')
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from epoch {start_epoch}")

    num_epochs = config['training']['num_epochs']
    best_val = float('inf')

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            photo_loss_fn, device, config, writer, epoch,
            ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
            output_dir=args.output_dir,
        )
        print(f"Train Loss: {train_metrics['loss']:.4f} | Photo: {train_metrics['photo_loss']:.4f} | SSIM: {train_metrics['ssim_loss']:.4f} | Alpha: {train_metrics['alpha_sparsity']:.4f} | Silhouette: {train_metrics['silhouette_loss']:.4f} | Chamfer: {train_metrics['chamfer_loss']:.4f} | MotionReg: {train_metrics['motion_reg']:.4f} | TemporalSmooth: {train_metrics['temporal_smooth']:.4f}")

        val_metrics = validate(
            model, val_loader,
            photo_loss_fn, device, config, writer, epoch,
            ex4dgs_dir=config.get('data', {}).get('ex4dgs_dir', None),
        )
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Photo: {val_metrics['photo_loss']:.4f} | SSIM: {val_metrics['ssim_loss']:.4f} | Alpha: {val_metrics['alpha_sparsity']:.4f} | Silhouette: {val_metrics['silhouette_loss']:.4f} | Chamfer: {val_metrics['chamfer_loss']:.4f} | MotionReg: {val_metrics['motion_reg']:.4f} | TemporalSmooth: {val_metrics['temporal_smooth']:.4f}")

        if writer is not None:
            writer.add_scalar('Epoch/Loss', train_metrics['loss'], epoch)
            writer.add_scalar('Epoch/PhotoLoss', train_metrics['photo_loss'], epoch)
            writer.add_scalar('Epoch/SSIMLoss', train_metrics['ssim_loss'], epoch)
            writer.add_scalar('Epoch/AlphaSparsity', train_metrics['alpha_sparsity'], epoch)
            writer.add_scalar('Epoch/SilhouetteLoss', train_metrics['silhouette_loss'], epoch)
            writer.add_scalar('Epoch/ChamferLoss', train_metrics['chamfer_loss'], epoch)
            writer.add_scalar('Epoch/MotionReg', train_metrics['motion_reg'], epoch)
            writer.add_scalar('Epoch/TemporalSmooth', train_metrics['temporal_smooth'], epoch)
            writer.add_scalar('Epoch/ValLoss', val_metrics['loss'], epoch)
            writer.add_scalar('Epoch/ValPhotoLoss', val_metrics['photo_loss'], epoch)
            writer.add_scalar('Epoch/ValSSIMLoss', val_metrics['ssim_loss'], epoch)
            writer.add_scalar('Epoch/ValAlphaSparsity', val_metrics['alpha_sparsity'], epoch)
            writer.add_scalar('Epoch/ValSilhouetteLoss', val_metrics['silhouette_loss'], epoch)
            writer.add_scalar('Epoch/ValChamferLoss', val_metrics['chamfer_loss'], epoch)
            writer.add_scalar('Epoch/ValMotionReg', val_metrics['motion_reg'], epoch)
            writer.add_scalar('Epoch/ValTemporalSmooth', val_metrics['temporal_smooth'], epoch)

        # Save checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            print(f"Saved best model (val_loss: {best_val:.4f})")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
