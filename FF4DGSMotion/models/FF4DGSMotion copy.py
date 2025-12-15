"""
4DGS Canonical Model - 重构版本
移除所有 Trellis 依赖，使用以下三个核心模块：
1. Point Downsampling / Clustering (Voxel Grid + 可选 KMeans)
2. Multi-view Feature Aggregator (Transformer-based)
3. Gaussian Head (MLP)
4. TimeWarpMotionHead (保留)

特点：
- 无需 Global Sim(3)，直接在世界坐标系中定义 canonical 高斯
- 世界 AABB 和体素中心缓存以避免重复计算
- 支持稀疏体素选择和流 t-scalar 控制
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Union
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from plyfile import PlyData, PlyElement
except Exception:  # pragma: no cover
    PlyData = None
    PlyElement = None
import numpy as np


class SlotAttention(nn.Module):
    """
    Slot Attention (Locatello et al.) 的简化实现。

    输入：tokens [B,N,D]
    输出：
        slots [B,M,D]
        attn  [B,N,M]  (每个 token 对每个 slot 的 soft assignment)
    """
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.iters = int(iters)
        self.eps = float(eps)

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)
        self.project_q = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        nn.init.xavier_uniform_(self.project_k.weight)
        nn.init.xavier_uniform_(self.project_v.weight)
        nn.init.xavier_uniform_(self.project_q.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        token_weight: Optional[torch.Tensor] = None,
        slots_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() != 3:
            raise ValueError(f"SlotAttention expects inputs [B,N,D], got {inputs.shape}")
        B, N, D = inputs.shape
        if D != self.dim:
            raise ValueError(f"SlotAttention dim mismatch: inputs D={D}, expected {self.dim}")

        token_w = None
        if token_weight is not None:
            if token_weight.dim() == 2:
                token_w = token_weight
            elif token_weight.dim() == 3 and token_weight.shape[-1] == 1:
                token_w = token_weight.squeeze(-1)
            else:
                raise ValueError(f"token_weight must be [B,N] or [B,N,1], got {token_weight.shape}")
            if token_w.shape[0] != B or token_w.shape[1] != N:
                raise ValueError(f"token_weight shape mismatch: expected [B,N]=[{B},{N}], got {token_w.shape}")
            token_w = token_w.to(dtype=inputs.dtype, device=inputs.device).clamp(min=0.0)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B,N,D]
        v = self.project_v(inputs)  # [B,N,D]

        # learnable slots (Gaussian init per-batch) or externally provided init
        if slots_init is not None:
            if slots_init.dim() != 3:
                raise ValueError(f"slots_init must be [B,M,D], got {slots_init.shape}")
            if slots_init.shape[0] != B or slots_init.shape[1] != self.num_slots or slots_init.shape[2] != D:
                raise ValueError(
                    f"slots_init shape mismatch: expected [{B},{self.num_slots},{D}], got {tuple(slots_init.shape)}"
                )
            slots = slots_init.to(device=inputs.device, dtype=inputs.dtype)
        else:
            mu = self.slots_mu.expand(B, self.num_slots, -1)
            sigma = torch.exp(self.slots_logsigma).expand(B, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)

        scale = D ** -0.5
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)  # [B,M,D]

            # logits: [B,N,M]
            logits = torch.einsum('bnd,bmd->bnm', k, q) * scale
            attn = torch.softmax(logits, dim=-1)  # over slots
            if token_w is not None:
                attn = attn * token_w.unsqueeze(-1)
            attn = attn + self.eps

            # normalize over tokens for each slot: sum_n a_{n,m} = 1
            attn_norm = attn / attn.sum(dim=1, keepdim=True)

            # updates: [B,M,D]
            updates = torch.einsum('bnd,bnm->bmd', v, attn_norm)

            # slot update via GRU
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D),
            ).view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class SlotAttentionGaussianPrior(nn.Module):
    """
    使用 Slot Attention 做可微的 Point-to-Gaussian soft clustering prior。

    输入：
        token_feat: [B,N,C]   (可来自多视角/多帧的 2D 特征或其它点特征)
        token_xyz:  [B,N,3]   (token 对应的 3D 坐标)
    输出：
        mu: [B,M,3]           (每个 slot 的 soft cluster center)
        normal: [B,M,3]       (由 slot feature 预测)
        radius: [B,M,1]       (由 slot feature 预测)
        confidence: [B,M,1]   (soft assignment 质量/cluster mass)
        slot_feat: [B,M,D]    (canonical shape token，可直接送进 head)
    """
    def __init__(
        self,
        num_slots: int,
        token_dim: int,
        slot_dim: int = 256,
        iters: int = 3,
        mlp_hidden: int = 256,
        radius_min: float = 1e-4,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.token_dim = int(token_dim)
        self.slot_dim = int(slot_dim)
        self.radius_min = float(radius_min)

        self.token_proj = nn.Linear(token_dim, slot_dim)
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=iters,
            hidden_dim=mlp_hidden,
        )

        self.normal_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, 3),
        )
        self.radius_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, 1),
        )

        nn.init.xavier_uniform_(self.token_proj.weight)
        if self.token_proj.bias is not None:
            nn.init.zeros_(self.token_proj.bias)

    def forward(
        self,
        token_feat: torch.Tensor,
        token_xyz: torch.Tensor,
        token_weight: Optional[torch.Tensor] = None,
        slots_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if token_feat.dim() != 3 or token_xyz.dim() != 3:
            raise ValueError(f"SlotPrior expects token_feat/token_xyz [B,N,*], got {token_feat.shape}, {token_xyz.shape}")
        if token_feat.shape[:2] != token_xyz.shape[:2]:
            raise ValueError(f"SlotPrior B,N mismatch: {token_feat.shape} vs {token_xyz.shape}")
        if token_xyz.shape[-1] != 3:
            raise ValueError(f"token_xyz last dim must be 3, got {token_xyz.shape}")

        B, N, _ = token_feat.shape
        x = self.token_proj(token_feat)  # [B,N,D]
        slot_feat, attn = self.slot_attn(x, token_weight=token_weight, slots_init=slots_init)  # [B,M,D], [B,N,M]

        # attn_mass: [B,M]，每个 slot 分到的 token 权重占比
        if token_weight is None:
            denom = float(max(1, N))
            attn_mass = attn.sum(dim=1) / denom
        else:
            if token_weight.dim() == 3 and token_weight.shape[-1] == 1:
                tw = token_weight.squeeze(-1)
            elif token_weight.dim() == 2:
                tw = token_weight
            else:
                raise ValueError(f"token_weight must be [B,N] or [B,N,1], got {token_weight.shape}")
            denom = tw.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]
            attn_mass = attn.sum(dim=1) / denom  # [B,M]

        # 计算每个 slot 的 soft center μ：对 token_xyz 做加权平均
        weights = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)  # [B,N,M], sum_n=1
        mu = torch.einsum('bnm,bnd->bmd', weights, token_xyz)  # [B,M,3]

        normal = F.normalize(self.normal_head(slot_feat), dim=-1)  # [B,M,3]
        radius = F.softplus(self.radius_head(slot_feat)) + self.radius_min  # [B,M,1]
        confidence = attn_mass.unsqueeze(-1).clamp(min=0.0, max=1.0)  # [B,M,1]

        return {
            'mu': mu,
            'normal': normal,
            'radius': radius,
            'confidence': confidence,
            'slot_feat': slot_feat,
            'attn': attn,
        }


class DualSlotAttentionGaussianPrior(nn.Module):
    """
    两路 SlotAttention（方案 B1）：分别生成 static / dynamic anchors。

    通过 token_p_dyn（动态概率）对 token 做软门控：
        w_static = (1 - p_dyn) * p_vis
        w_dynamic = p_dyn * p_vis
    """
    def __init__(
        self,
        num_slots_static: int,
        num_slots_dynamic: int,
        token_dim: int,
        slot_dim: int = 256,
        iters: int = 3,
        mlp_hidden: int = 256,
        radius_min: float = 1e-4,
    ):
        super().__init__()
        self.static_prior = SlotAttentionGaussianPrior(
            num_slots=num_slots_static,
            token_dim=token_dim,
            slot_dim=slot_dim,
            iters=iters,
            mlp_hidden=mlp_hidden,
            radius_min=radius_min,
        )
        self.dynamic_prior = SlotAttentionGaussianPrior(
            num_slots=num_slots_dynamic,
            token_dim=token_dim,
            slot_dim=slot_dim,
            iters=iters,
            mlp_hidden=mlp_hidden,
            radius_min=radius_min,
        )

    @staticmethod
    def _as_bn1(x: torch.Tensor, name: str, B: int, N: int) -> torch.Tensor:
        if x is None:
            return None
        if x.dim() == 2:
            y = x.unsqueeze(-1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            y = x
        else:
            raise ValueError(f"{name} must be [B,N] or [B,N,1], got {x.shape}")
        if y.shape[0] != B or y.shape[1] != N:
            raise ValueError(f"{name} shape mismatch: expected [B,N,*]=[{B},{N},*], got {y.shape}")
        return y

    def forward(
        self,
        token_feat: torch.Tensor,
        token_xyz: torch.Tensor,
        token_p_dyn: torch.Tensor,
        token_p_vis: Optional[torch.Tensor] = None,
        slots_init_static: Optional[torch.Tensor] = None,
        slots_init_dynamic: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if token_feat.dim() != 3 or token_xyz.dim() != 3:
            raise ValueError(f"DualSlotPrior expects token_feat/token_xyz [B,N,*], got {token_feat.shape}, {token_xyz.shape}")
        if token_feat.shape[:2] != token_xyz.shape[:2]:
            raise ValueError(f"DualSlotPrior B,N mismatch: {token_feat.shape} vs {token_xyz.shape}")
        B, N = token_feat.shape[0], token_feat.shape[1]

        p_dyn = self._as_bn1(token_p_dyn, "token_p_dyn", B, N).clamp(0.0, 1.0)
        p_vis = self._as_bn1(token_p_vis, "token_p_vis", B, N)
        if p_vis is None:
            p_vis = torch.ones_like(p_dyn)
        else:
            p_vis = p_vis.clamp(0.0, 1.0)

        w_dyn = (p_dyn * p_vis).squeeze(-1)          # [B,N]
        w_sta = ((1.0 - p_dyn) * p_vis).squeeze(-1)  # [B,N]

        out_s = self.static_prior(token_feat, token_xyz, token_weight=w_sta, slots_init=slots_init_static)
        out_d = self.dynamic_prior(token_feat, token_xyz, token_weight=w_dyn, slots_init=slots_init_dynamic)

        return {
            'static': out_s,
            'dynamic': out_d,
        }


class SurfelExtractor(nn.Module):
    """
    SURFEL 提取器 - 使用局部 PCA 从点云中提取表面元素
    
    对每个点的 K-近邻邻域进行 PCA，得到：
    - μ_j: 局部中心（PCA 均值）
    - R_j: 局部坐标系（PCA 特征向量）
    - s_j: 局部半径（PCA 特征值的平方根）
    
    输入：points_3d [T,N,3] 或 [N,3]
    输出：
        - mu: [M,3] SURFEL 中心
        - normal: [M,3] 主法线方向（最小特征值对应的特征向量）
        - radius: [M,1] 局部半径
        - confidence: [M,1] 置信度（基于特征值分布）
    """
    def __init__(
        self,
        k_neighbors: int = 16,
        use_confidence_weighting: bool = True,
        confidence_threshold: float = 0.1,
    ):
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.use_confidence_weighting = bool(use_confidence_weighting)
        self.confidence_threshold = float(confidence_threshold)

    @staticmethod
    def _local_pca_fast(
        points: torch.Tensor,
        k: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        快速局部 PCA - 使用 KNN 而非全距离矩阵
        
        Args:
            points: [N,3]
            k: K-近邻数量
            
        Returns:
            centers: [N,3] 局部中心
            normals: [N,3] 主法线（最小特征值对应）
            radii: [N,1] 局部半径（最大特征值）
            eigenvalues: [N,3] 三个特征值（降序）
        """
        device = points.device
        dtype = points.dtype
        N = points.shape[0]

        # AMP 下 CUDA 的 eigh 不支持 fp16；这里禁用 autocast 并强制 fp32
        if points.is_cuda:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            autocast_ctx = None

        if autocast_ctx is None:
            points_f = points.to(dtype=torch.float32)
            dists = torch.cdist(points_f, points_f)  # [N,N]
            _, indices = torch.topk(dists, k=min(k, N), dim=1, largest=False)  # [N,k]
            neighbors = points_f[indices]  # [N,k,3]
            centers = neighbors.mean(dim=1)  # [N,3]
            neighbors_centered = neighbors - centers.unsqueeze(1)  # [N,k,3]
            cov = torch.bmm(neighbors_centered.transpose(1, 2), neighbors_centered) / max(1, k - 1)  # [N,3,3]
            cov = cov.to(dtype=torch.float32)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # [N,3], [N,3,3]
        else:
            with autocast_ctx:
                points_f = points.to(dtype=torch.float32)
                dists = torch.cdist(points_f, points_f)  # [N,N]
                _, indices = torch.topk(dists, k=min(k, N), dim=1, largest=False)  # [N,k]
                neighbors = points_f[indices]  # [N,k,3]
                centers = neighbors.mean(dim=1)  # [N,3]
                neighbors_centered = neighbors - centers.unsqueeze(1)  # [N,k,3]
                cov = torch.bmm(neighbors_centered.transpose(1, 2), neighbors_centered) / max(1, k - 1)  # [N,3,3]
                cov = cov.to(dtype=torch.float32)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # [N,3], [N,3,3]
        
        # 特征值升序排列，需要反转
        eigenvalues = eigenvalues.flip(dims=[1])  # [N,3] 降序
        eigenvectors = eigenvectors.flip(dims=[2])  # [N,3,3]
        
        # 最小特征值对应的特征向量（法线）
        normals = eigenvectors[:, :, -1]  # [N,3]
        
        # 局部半径（最大特征值的平方根）
        radii = torch.sqrt(eigenvalues[:, 0:1].clamp(min=1e-6))  # [N,1]
        
        return (
            centers.to(device=device, dtype=dtype),
            normals.to(device=device, dtype=dtype),
            radii.to(device=device, dtype=dtype),
            eigenvalues.to(device=device, dtype=dtype),
        )

    def forward(
        self,
        points_3d: torch.Tensor,
        fps_target: int = 20000,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points_3d: [T,N,3] 或 [N,3]
            fps_target: FPS 目标点数（避免 PCA 输入过大）
            
        Returns:
            dict with keys:
                - mu: [M,3] SURFEL 中心
                - normal: [M,3] 法线
                - radius: [M,1] 局部半径
                - confidence: [M,1] 置信度
        """
        if points_3d is None or points_3d.numel() == 0:
            device = points_3d.device if points_3d is not None else torch.device('cpu')
            return {
                'mu': torch.zeros(0, 3, device=device),
                'normal': torch.zeros(0, 3, device=device),
                'radius': torch.zeros(0, 1, device=device),
                'confidence': torch.zeros(0, 1, device=device),
            }
        
        device = points_3d.device
        dtype = points_3d.dtype
        
        # 时序汇合
        if points_3d.dim() == 3:
            points_all = points_3d.reshape(-1, 3)
        else:
            points_all = points_3d.view(-1, 3)
        
        # 过滤无效点
        valid_mask = torch.isfinite(points_all).all(dim=-1)
        points_all = points_all[valid_mask]
        
        if points_all.numel() == 0:
            return {
                'mu': torch.zeros(0, 3, device=device, dtype=dtype),
                'normal': torch.zeros(0, 3, device=device, dtype=dtype),
                'radius': torch.zeros(0, 1, device=device, dtype=dtype),
                'confidence': torch.zeros(0, 1, device=device, dtype=dtype),
            }
        
        # 【改进】Step 1: 随机子采样 - 避免 FPS 的 NxN cdist OOM
        # 从 200k 点中随机采样到 fps_target（20k），足以进行 PCA
        if points_all.shape[0] > fps_target:
            # 随机子采样，避免 cdist OOM
            rand_idx = torch.randperm(points_all.shape[0], device=points_all.device)[:fps_target]
            points_pca = points_all[rand_idx]
        else:
            points_pca = points_all
        
        # 【改进】Step 2: 快速 PCA（仅在 fps_target 个点上）
        centers, normals, radii, eigenvalues = self._local_pca_fast(
            points_pca,
            k=min(self.k_neighbors, points_pca.shape[0]),
        )
        
        # 计算置信度（基于特征值分布）
        if self.use_confidence_weighting:
            # 置信度 = 1 - (λ_min / λ_max)
            # λ_min 越小，表面越平坦，置信度越高
            lambda_min = eigenvalues[:, -1:].clamp(min=1e-6)
            lambda_max = eigenvalues[:, 0:1].clamp(min=1e-6)
            confidence = 1.0 - (lambda_min / lambda_max).clamp(0, 1)
        else:
            confidence = torch.ones_like(radii)
        
        return {
            'mu': centers.to(device=device, dtype=dtype),
            'normal': normals.to(device=device, dtype=dtype),
            'radius': radii.to(device=device, dtype=dtype),
            'confidence': confidence.to(device=device, dtype=dtype),
        }

    @staticmethod
    def _farthest_point_sampling(
        points: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        简单的最远点采样（FPS）
        
        Args:
            points: [N,3]
            num_samples: 目标采样数
            
        Returns:
            indices: [num_samples]
        """
        N = points.shape[0]
        device = points.device
        
        if num_samples >= N:
            return torch.arange(N, device=device)
        
        # 初始化：随机选择第一个点
        selected_indices = [torch.randint(0, N, (1,), device=device).item()]
        
        # 距离矩阵
        dists = torch.cdist(points, points)  # [N,N]
        
        # 迭代选择
        for _ in range(num_samples - 1):
            # 已选点的索引
            selected = torch.tensor(selected_indices, device=device)
            
            # 计算每个未选点到已选点的最小距离
            min_dists = dists[selected].min(dim=0)[0]  # [N]
            
            # 排除已选点
            min_dists[selected] = -1e9
            
            # 选择最远的点
            next_idx = min_dists.argmax().item()
            selected_indices.append(next_idx)
        
        return torch.tensor(selected_indices, device=device, dtype=torch.long)


class WeightedFPS(nn.Module):
    """
    Weighted Farthest Point Sampling (加权最远点采样)
    
    从 M 个点中选择 K 个点，使得：
    1. 点之间的距离尽可能远（FPS）
    2. 考虑置信度权重（高置信度的点优先被选中）
    
    输入：
        - points: [M,3] 点坐标
        - weights: [M,1] 置信度权重
        - num_samples: 目标采样数 K
        
    输出：
        - indices: [K] 选中的点的索引
        - selected_points: [K,3]
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
        points: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: [M,3]
            weights: [M,1] or [M]
            num_samples: K
            
        Returns:
            indices: [K]
            selected_points: [K,3]
        """
        M = points.shape[0]
        device = points.device
        dtype = points.dtype
        
        if num_samples >= M:
            return torch.arange(M, device=device), points
        
        # 归一化权重
        if weights.dim() == 2:
            weights = weights.squeeze(-1)
        weights = weights / (weights.sum() + 1e-8)
        
        # 初始化：根据权重选择第一个点
        first_idx = torch.multinomial(weights, 1).item()
        selected_indices = [first_idx]
        
        # 距离矩阵
        dists = torch.cdist(points, points)  # [M,M]
        
        # 迭代选择
        for _ in range(num_samples - 1):
            # 已选点的索引
            selected = torch.tensor(selected_indices, device=device)
            
            # 计算每个未选点到已选点的最小距离
            min_dists = dists[selected].min(dim=0)[0]  # [M]
            
            # 加入权重：distance_score = min_dist * weight
            # 选择 distance_score 最大的点
            score = min_dists * weights
            
            # 排除已选点
            score[selected] = -1e9
            
            # 选择得分最高的点
            next_idx = score.argmax().item()
            selected_indices.append(next_idx)
        
        indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
        selected_points = points[indices]
        
        return indices, selected_points


class PointDownsampler(nn.Module):
    """
    Point 下采样/聚类模块
    输入：points_3d [T,N,3]
    输出：mu [M,3] - canonical 高斯中心
    
    方法：
    1. 时序汇合：concat 所有时间帧的点
    2. Voxel 下采样：固定 voxel size 做体素划分，每个 voxel 取均值
    3. (可选) KMeans refine：对 voxel centers 做小规模 KMeans
    
    细粒度控制：
    - voxel_size: 体素大小（场景尺度相关）
    - use_kmeans_refine: 是否进行 KMeans 精化
    - kmeans_iterations: KMeans 迭代次数
    - adaptive_voxel: 是否根据 AABB 自适应调整 voxel_size
    - target_num_gaussians: 目标高斯数量（用于反推 voxel_size）
    """
    def __init__(
        self,
        voxel_size: float = 0.02,
        use_kmeans_refine: bool = False,
        kmeans_iterations: int = 10,
        adaptive_voxel: bool = True,
        target_num_gaussians: Optional[int] = None,
    ):
        super().__init__()
        self.voxel_size = float(voxel_size)
        self.use_kmeans_refine = bool(use_kmeans_refine)
        self.kmeans_iterations = int(kmeans_iterations)
        self.adaptive_voxel = bool(adaptive_voxel)
        self.target_num_gaussians = target_num_gaussians

    @staticmethod
    def voxel_downsample(
        points: torch.Tensor,
        voxel_size: float,
    ) -> torch.Tensor:
        """
        Voxel 下采样：将点云按 voxel_size 划分，每个 voxel 取所有点的均值
        
        Args:
            points: [N,3] 点云
            voxel_size: 体素大小
            
        Returns:
            voxel_centers: [M,3] 体素中心
        """
        if points.numel() == 0:
            return points
        
        device = points.device
        dtype = points.dtype
        
        # 计算体素索引
        voxel_indices = torch.floor(points / voxel_size).long()
        
        # 创建唯一的体素 ID
        # 使用哈希方法避免内存溢出
        min_idx = voxel_indices.min(dim=0)[0]
        max_idx = voxel_indices.max(dim=0)[0]
        
        # 重新映射到 [0, max_range)
        voxel_indices_shifted = voxel_indices - min_idx
        
        # 计算线性索引（假设范围不超过 int32）
        range_x = max_idx[0] - min_idx[0] + 1
        range_y = max_idx[1] - min_idx[1] + 1
        
        linear_idx = (
            voxel_indices_shifted[:, 0] * (range_y * (max_idx[2] - min_idx[2] + 1)) +
            voxel_indices_shifted[:, 1] * (max_idx[2] - min_idx[2] + 1) +
            voxel_indices_shifted[:, 2]
        )
        
        # 按体素分组求均值
        unique_idx, inverse_indices = torch.unique(linear_idx, return_inverse=True)
        
        voxel_centers_list = []
        for i in range(len(unique_idx)):
            mask = inverse_indices == i
            voxel_centers_list.append(points[mask].mean(dim=0))
        
        voxel_centers = torch.stack(voxel_centers_list, dim=0)
        return voxel_centers

    @staticmethod
    def kmeans_refine(
        points: torch.Tensor,
        num_clusters: int,
        iterations: int = 10,
    ) -> torch.Tensor:
        """
        KMeans 聚类：对点进行 KMeans 聚类，返回聚类中心
        
        Args:
            points: [N,3]
            num_clusters: 目标聚类数
            iterations: KMeans 迭代次数
            
        Returns:
            centers: [num_clusters,3]
        """
        if points.shape[0] <= num_clusters:
            return points
        
        device = points.device
        dtype = points.dtype
        N = points.shape[0]
        
        # 随机初始化聚类中心
        indices = torch.randperm(N, device=device)[:num_clusters]
        centers = points[indices].clone()
        
        for _ in range(iterations):
            # 计算每个点到各聚类中心的距离
            distances = torch.cdist(points, centers)  # [N, num_clusters]
            
            # 分配点到最近的聚类
            assignments = distances.argmin(dim=1)
            
            # 更新聚类中心
            new_centers = []
            for k in range(num_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centers.append(points[mask].mean(dim=0))
                else:
                    new_centers.append(centers[k])
            centers = torch.stack(new_centers, dim=0)
        
        return centers

    def forward(
        self,
        points_3d: Optional[torch.Tensor],
        world_aabb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points_3d: [T,N,3] 或 [N,3]
            world_aabb: [2,3] 可选的世界 AABB（用于确定 voxel_size）
            
        Returns:
            mu: [M,3] canonical 高斯中心
        """
        if points_3d is None or points_3d.numel() == 0:
            return torch.zeros(0, 3, device=points_3d.device if points_3d is not None else torch.device('cpu'))
        
        device = points_3d.device
        dtype = points_3d.dtype
        
        # 时序汇合
        if points_3d.dim() == 3:
            points_all = points_3d.reshape(-1, 3)  # [T*N, 3]
        else:
            points_all = points_3d.view(-1, 3)
        
        # 过滤无效点
        valid_mask = torch.isfinite(points_all).all(dim=-1)
        points_all = points_all[valid_mask]
        
        if points_all.numel() == 0:
            return torch.zeros(0, 3, device=device, dtype=dtype)
        
        # 【改进】自适应 voxel_size 计算
        voxel_size = self.voxel_size
        if self.adaptive_voxel and world_aabb is not None:
            # 根据 AABB 自动调整 voxel_size
            # extent 是场景的最大尺度（单位：米）
            extent = (world_aabb[1] - world_aabb[0]).max().item()
            
            # 【修复】避免 extent 过小导致 voxel_size 过小
            # 策略：根据 target_num_gaussians 或保持最小粒度
            if self.target_num_gaussians is not None:
                # 根据目标高斯数量反推 voxel_size
                # 假设体素数量 ≈ (extent / voxel_size)^3
                # 则 voxel_size ≈ extent / (target_num_gaussians^(1/3))
                target_voxel_size = extent / max(2.0, (self.target_num_gaussians ** (1.0/3.0)))
                voxel_size = target_voxel_size
            else:
                # 保守策略：voxel_size 不小于 extent / 200
                # 这样即使 extent=1，voxel_size 也不会小于 0.005
                voxel_size = max(self.voxel_size, extent / 200.0)
        
        mu = self.voxel_downsample(points_all, voxel_size)
        
        # (可选) KMeans refine
        if self.use_kmeans_refine and mu.shape[0] > 1:
            # 【改进】更智能的 target_num 选择
            if self.target_num_gaussians is not None:
                target_num = self.target_num_gaussians
            else:
                # 默认：保留 50% 的体素
                target_num = max(1, mu.shape[0] // 2)
            
            if target_num < mu.shape[0]:
                mu = self.kmeans_refine(mu, target_num, self.kmeans_iterations)
        
        return mu.to(device=device, dtype=dtype)


class PerGaussianAggregator(nn.Module):
    """
    Multi-view Feature Aggregator (优化版本)
    对每个 μ_j，从多视角 feat_2d 中聚合一个局部外观+几何特征 g_j
    
    优化策略：
    1. 视角筛选：过滤不可见视角（z <= 0）
    2. 按 viewing angle 与 depth 排序，只取 top-K（默认 4）
    3. 降维：d_model=256（而非 512）
    4. 降层数：num_layers=1（而非 2）
    
    步骤：
    1. 投影 μ_j 到所有 (t,v) 视角
    2. 可见性过滤
    3. 按视角质量排序，取 top-K
    4. Bilinear sample 特征
    5. 加入时间/视角位置编码
    6. 通过 Transformer 聚合
    """
    def __init__(
        self,
        feat_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        hidden_dim: int = 256,
        time_emb_dim: int = 32,
        view_emb_dim: int = 32,
        max_views: int = 32,
        topk_views: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_emb_dim = time_emb_dim
        self.view_emb_dim = view_emb_dim
        self.topk_views = topk_views
        
        # 位置编码
        self.view_emb = nn.Embedding(max_views, view_emb_dim)
        
        # 特征投影（降维：C + Dt + Dv + 3 + 1 → hidden_dim）
        feat_input_dim = feat_dim + time_emb_dim + view_emb_dim + 3 + 1  # +3 for normal, +1 for radius
        self.feat_proj = nn.Linear(feat_input_dim, hidden_dim)
        
        # Transformer 层（降层数：1 层而非 2）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 压缩位置/几何编码，避免过大拼接维度
        pos_in_dim = time_emb_dim + view_emb_dim + 3 + 1  # Dt + Dv + normal(3) + radius(1)
        self.pos_proj = nn.Linear(pos_in_dim, 32)

        # 重新定义输入投影：仅 feat_dim + 32 的小维度
        feat_input_dim = feat_dim + 32
        self.feat_proj = nn.Linear(feat_input_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

    @staticmethod
    def _posenc_t(t_scalar: torch.Tensor, dim: int) -> torch.Tensor:
        """
        时间位置编码
        Args:
            t_scalar: [T] in [0,1]
            dim: 编码维度
        Returns:
            emb: [T, dim]
        """
        device = t_scalar.device
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, 8, steps=half, device=device))
        phases = t_scalar.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    @staticmethod
    def _project_points(
        Xw: torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        投影世界坐标到像素坐标
        Args:
            Xw: [M,3] 世界坐标
            c2w: [4,4] camera-to-world
            K: [3,3] 内参
        Returns:
            u, v, z: [M] 像素坐标和深度
        """
        Xw32 = Xw.to(torch.float32)
        c2w32 = c2w.to(torch.float32)
        K32 = K.to(torch.float32)
        
        w2c32 = torch.inverse(c2w32)
        M = Xw32.shape[0]
        Xw_h = torch.cat([Xw32, torch.ones(M, 1, device=Xw32.device, dtype=torch.float32)], dim=1)
        Xc = (w2c32 @ Xw_h.t()).t()[:, :3]
        
        z = Xc[:, 2]
        uvw = (K32 @ Xc.t()).t()
        u = uvw[:, 0] / (uvw[:, 2].clamp(min=1e-6))
        v = uvw[:, 1] / (uvw[:, 2].clamp(min=1e-6))
        
        return u.to(Xw.dtype), v.to(Xw.dtype), z.to(Xw.dtype)

    @staticmethod
    def _bilinear_sample(
        feat: torch.Tensor,
        u_pix: torch.Tensor,
        v_pix: torch.Tensor,
        H_img: float,
        W_img: float,
    ) -> torch.Tensor:
        """
        Bilinear sample from feature map
        Args:
            feat: [H',W',C]
            u_pix, v_pix: [M]
            H_img, W_img: 原始图像大小
        Returns:
            sampled: [M,C]
        """
        Hp, Wp, C = feat.shape
        dtype = feat.dtype
        device = feat.device
        
        # 缩放到特征图大小
        u_feat = u_pix.to(dtype) * (Wp / W_img)
        v_feat = v_pix.to(dtype) * (Hp / H_img)
        
        # 归一化到 [-1,1]
        x = 2.0 * (u_feat / max(1, Wp - 1)) - 1.0
        y = 2.0 * (v_feat / max(1, Hp - 1)) - 1.0
        
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2).to(dtype=dtype)
        feat_chw = feat.permute(2, 0, 1).unsqueeze(0)  # [1,C,H',W']
        
        sampled = F.grid_sample(feat_chw, grid, mode='bilinear', align_corners=True)
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [M,C]
        
        return sampled

    def forward(
        self,
        mu: torch.Tensor,                    # [M,3] 高斯中心
        feat_2d: torch.Tensor,               # [T,V,H',W',C]
        camera_poses: torch.Tensor,          # [T,V,4,4]
        camera_intrinsics: torch.Tensor,     # [T,V,3,3]
        time_ids: Optional[torch.Tensor] = None,
        surfel_normal: Optional[torch.Tensor] = None,  # [M,3] SURFEL 法线
        surfel_radius: Optional[torch.Tensor] = None,  # [M,1] SURFEL 半径
    ) -> torch.Tensor:
        """
        Args:
            mu: [M,3]
            feat_2d: [T,V,H',W',C]
            camera_poses: [T,V,4,4]
            camera_intrinsics: [T,V,3,3]
            time_ids: [T]
            surfel_normal: [M,3] 可选的 SURFEL 法线
            surfel_radius: [M,1] 可选的 SURFEL 半径
            
        Returns:
            g: [M,C] canonical feature
        """
        device = feat_2d.device
        dtype = feat_2d.dtype
        T, V, Hp, Wp, C = feat_2d.shape
        M = mu.shape[0]
        
        if M == 0:
            return torch.zeros(0, self.feat_dim, device=device, dtype=dtype)
        
        # 计算原始图像大小（从内参推断）
        cx = camera_intrinsics[0, 0, 0, 2]
        cy = camera_intrinsics[0, 0, 1, 2]
        W_img = 2.0 * cx
        H_img = 2.0 * cy
        
        # 时间编码
        if time_ids is None:
            time_ids = torch.arange(T, device=device)
        t_min = time_ids.min()
        t_max = torch.clamp(time_ids.max(), min=t_min + 1)
        t_norm = (time_ids.float() - t_min.float()) / (t_max.float() - t_min.float())
        t_emb_all = self._posenc_t(t_norm, self.time_emb_dim)  # [T, Dt]
        
        # 【优化】预计算所有视角的投影和可见性
        # 用于后续的视角筛选和排序
        view_scores = []  # [T, V, M] 视角质量分数
        
        for t in range(T):
            for v in range(V):
                c2w = camera_poses[t, v]
                K = camera_intrinsics[t, v]
                
                # 投影
                u, vv, z = self._project_points(mu, c2w, K)
                
                # 可见性：z > 0 且在图像范围内
                in_image = (u >= 0) & (u < W_img) & (vv >= 0) & (vv < H_img)
                visible = (z > 1e-4) & in_image
                
                # 视角质量分数：基于深度和法线方向
                if surfel_normal is not None:
                    # 计算相机到点的方向
                    cam_pos = c2w[:3, 3]
                    direction = mu - cam_pos.unsqueeze(0)  # [M, 3]
                    direction = F.normalize(direction, dim=-1)
                    
                    # 法线与视角方向的夹角（越小越好）
                    normal_w = F.normalize(surfel_normal, dim=-1)
                    view_angle = (direction * normal_w).sum(dim=-1)  # [M]
                    view_angle = torch.clamp(view_angle, 0, 1)  # 只取正向
                else:
                    view_angle = torch.ones(M, device=device, dtype=dtype)
                
                # 深度权重（近的点权重高）
                depth_weight = 1.0 / (z.clamp(min=0.1) + 1e-6)
                
                # 综合分数
                score = view_angle * depth_weight
                score = score * visible.float()  # 不可见的视角分数为 0
                
                view_scores.append(score)
        
        view_scores = torch.stack(view_scores, dim=0)  # [T*V, M]
        
        # 【优化】对每个高斯，选择 top-K 视角
        # 重新组织为 [M, T*V] 便于 topk
        view_scores_t = view_scores.t()  # [M, T*V]
        
        # 获取 top-K 视角索引
        topk_num = min(self.topk_views, T * V)
        topk_scores, topk_indices = torch.topk(view_scores_t, k=topk_num, dim=1)  # [M, K]
        
        # 仅为 Top-K 视角构建特征序列，显著降低序列长度
        K_sel = topk_num
        hidden = self.feat_proj.out_features
        features_stacked = torch.zeros(M, K_sel, hidden, device=device, dtype=dtype)
        
        # 遍历所有 (t,v) 对，并填充对应的 (m,k) 位置
        tv_idx = 0
        for t in range(T):
            for v in range(V):
                c2w = camera_poses[t, v]
                K = camera_intrinsics[t, v]
                
                # 找出哪些 (m,k) 选择了当前 tv
                match = (topk_indices == tv_idx).nonzero(as_tuple=False)  # [Q, 2] (m_idx, k_idx)
                if match.numel() == 0:
                    tv_idx += 1
                    continue
                m_idx = match[:, 0]
                k_idx = match[:, 1]
                
                # 投影（仅对被选中的 M_idx 进行采样）
                u, vv, z = self._project_points(mu[m_idx], c2w, K)  # [Q]
                
                # Sample 特征
                feat_tv = feat_2d[t, v]
                sampled_feat = self._bilinear_sample(feat_tv, u, vv, H_img=float(H_img), W_img=float(W_img))  # [Q, C]
                
                # 可见性掩码
                valid_mask = (z > 1e-4).float().unsqueeze(-1)
                sampled_feat = sampled_feat * valid_mask
                
                # SURFEL 几何信息（仅用于位置编码压缩）
                if surfel_normal is not None and surfel_radius is not None:
                    c2w32 = c2w.to(torch.float32)
                    w2c32 = torch.inverse(c2w32)
                    R_w2c = w2c32[:3, :3]
                    normal_cam = (R_w2c @ surfel_normal[m_idx].t()).t()  # [Q,3]
                    normal_cam = F.normalize(normal_cam, dim=-1)
                    radius_feat = surfel_radius[m_idx].to(dtype)  # [Q,1]
                else:
                    Q = m_idx.shape[0]
                    normal_cam = torch.zeros(Q, 3, device=device, dtype=dtype)
                    radius_feat = torch.zeros(Q, 1, device=device, dtype=dtype)
                
                # 时间/视角编码
                t_emb = t_emb_all[t].expand(m_idx.shape[0], -1)
                v_safe = min(v, self.view_emb.num_embeddings - 1)
                v_emb = self.view_emb(torch.tensor(v_safe, device=device)).expand(m_idx.shape[0], -1)
                
                # 压缩到 32 维
                pos_vec = torch.cat([normal_cam.to(dtype), radius_feat, t_emb, v_emb], dim=-1)  # [Q, 3+1+Dt+Dv]
                pos32 = self.pos_proj(pos_vec)  # [Q, 32]
                
                # 拼接并投影
                feat_with_pos = torch.cat([sampled_feat, pos32], dim=-1)  # [Q, C+32]
                feat_hidden = self.feat_proj(feat_with_pos)  # [Q, hidden]
                
                # 写入 (m_idx, k_idx)
                features_stacked[m_idx, k_idx, :] = feat_hidden
                
                tv_idx += 1
        
        # Transformer 聚合（序列长度 = K_sel）
        features_agg = self.transformer(features_stacked)  # [M, K, hidden]
        
        # 仅针对 Top-K 的权重进行加权池化
        sel_scores = torch.gather(view_scores_t, 1, topk_indices)  # [M, K]
        weights = sel_scores.unsqueeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        g = (features_agg * weights).sum(dim=1)  # [M, hidden]
        
        # 输出投影
        g = self.out_proj(g)  # [M, feat_dim]
        
        return g


class GaussianHead(nn.Module):
    """
    Gaussian Head - MLP 预测高斯参数（SURFEL 版本）
    
    输入：g_j [M,C] canonical feature
    输出：
        - color c_j [M,3]
        - opacity o_j [M,1]
        - (可选) scale delta Δs_j [M,3]
        - (可选) rotation delta ΔR_j [M,6]
    
    注意：R_j 和 s_j 直接来自 SURFEL，不再由 Head 预测
    """
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        color_dim: int = 3,
        use_scale_refine: bool = False,
        use_rot_refine: bool = False,
        opacity_init_bias: float = -2.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.color_dim = color_dim
        self.use_scale_refine = bool(use_scale_refine)
        self.use_rot_refine = bool(use_rot_refine)
        self.opacity_init_bias = float(opacity_init_bias)
        
        # MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # 必需输出头
        self.fc_opac = nn.Linear(hidden_dim, 1)     # opacity
        self.fc_color = nn.Linear(hidden_dim, color_dim)  # RGB
        
        # (可选) 微调头
        if use_scale_refine:
            self.fc_scale_delta = nn.Linear(hidden_dim, 3)  # log-scale delta
        
        if use_rot_refine:
            self.fc_rot_delta = nn.Linear(hidden_dim, 6)  # 6D rotation delta
        
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # 让初始 opacity 较小：sigmoid(-2)≈0.12，避免 early training 全遮挡
        if hasattr(self, "fc_opac") and isinstance(self.fc_opac, nn.Linear) and self.fc_opac.bias is not None:
            nn.init.constant_(self.fc_opac.bias, self.opacity_init_bias)

    @staticmethod
    def _rot_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        将 6D 旋转表示转换为 3x3 旋转矩阵
        使用 Gram-Schmidt 正交化
        
        Args:
            rot_6d: [M,6]
        Returns:
            rot_mat: [M,3,3]
        """
        batch_size = rot_6d.shape[0]
        
        # 前两列
        col1 = rot_6d[:, :3]
        col2 = rot_6d[:, 3:6]
        
        # 正交化
        col1 = F.normalize(col1, dim=-1)
        col2 = col2 - (col1 * col2).sum(dim=-1, keepdim=True) * col1
        col2 = F.normalize(col2, dim=-1)
        
        # 第三列（叉积）
        col3 = torch.cross(col1, col2, dim=-1)
        
        # 拼接
        rot_mat = torch.stack([col1, col2, col3], dim=-1)  # [M,3,3]
        
        return rot_mat

    def forward(
        self,
        g: torch.Tensor,  # [M,C]
        surfel_scale: Optional[torch.Tensor] = None,  # [M,3] SURFEL 尺度
        surfel_rot: Optional[torch.Tensor] = None,    # [M,3,3] SURFEL 旋转
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: [M,C] canonical feature
            surfel_scale: [M,3] 可选的 SURFEL 初始尺度
            surfel_rot: [M,3,3] 可选的 SURFEL 初始旋转
            
        Returns:
            dict with keys:
                - color: [M,3]
                - opacity: [M,1]
                - scale: [M,3] (SURFEL scale 或微调后)
                - rot: [M,3,3] (SURFEL rot 或微调后)
                - (可选) scale_delta: [M,3]
                - (可选) rot_delta: [M,6]
        """
        h = self.mlp(g)  # [M, hidden_dim]
        
        # 必需输出
        opac_logit = self.fc_opac(h)  # [M,1]
        color_logit = self.fc_color(h)  # [M,color_dim]
        
        opacity = torch.sigmoid(opac_logit)  # [M,1]
        color = torch.sigmoid(color_logit)  # [M,color_dim]
        
        result = {
            'color': color,
            'opacity': opacity,
        }
        
        # 处理尺度和旋转
        if surfel_scale is not None:
            scale = surfel_scale.clone()
        else:
            # 如果没有 SURFEL，使用默认值
            scale = torch.ones(h.shape[0], 3, device=h.device, dtype=h.dtype)
        
        if surfel_rot is not None:
            rot = surfel_rot.clone()
        else:
            # 如果没有 SURFEL，使用单位矩阵
            M = h.shape[0]
            rot = torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0).expand(M, -1, -1)
        
        # (可选) 微调尺度
        if self.use_scale_refine:
            scale_delta_logit = self.fc_scale_delta(h)  # [M,3]
            scale_delta = torch.tanh(scale_delta_logit) * 0.2  # 限制在 [-0.2, 0.2]
            scale = scale * torch.exp(scale_delta.clamp(min=-5, max=5))
            result['scale_delta'] = scale_delta
        
        # (可选) 微调旋转
        if self.use_rot_refine:
            rot_delta_6d = self.fc_rot_delta(h)  # [M,6]
            rot_delta_mat = self._rot_6d_to_matrix(rot_delta_6d * 0.1)  # 缩放以限制幅度
            rot = torch.bmm(rot, rot_delta_mat)  # [M,3,3]
            result['rot_delta'] = rot_delta_6d
        
        result['scale'] = scale
        result['rot'] = rot
        
        return result


class EGNNLayer(nn.Module):
    """最小 EGNN layer（E(n)-equivariant），用于 anchor 的 SE(3) 先验 refine。"""
    def __init__(self, feat_dim: int, hidden_dim: int = 128, coord_scale: float = 0.1):
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,N,3], h: [B,N,F]
        B, N, _ = x.shape
        Fdim = h.shape[-1]

        xi = x.unsqueeze(2)  # [B,N,1,3]
        xj = x.unsqueeze(1)  # [B,1,N,3]
        dx = xi - xj         # [B,N,N,3]
        dist2 = (dx ** 2).sum(dim=-1, keepdim=True)  # [B,N,N,1]

        hi = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B,N,N,F]
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B,N,N,F]
        e_in = torch.cat([hi, hj, dist2], dim=-1)  # [B,N,N,2F+1]
        m_ij = self.edge_mlp(e_in)                 # [B,N,N,H]

        # coord update
        c_ij = self.coord_mlp(m_ij)                # [B,N,N,1]
        dx_update = (dx * c_ij).sum(dim=2)         # [B,N,3]
        x = x + self.coord_scale * dx_update

        # node update
        m_i = m_ij.sum(dim=2)                      # [B,N,H]
        h_in = torch.cat([h, m_i], dim=-1)         # [B,N,F+H]
        dh = self.node_mlp(h_in)                   # [B,N,F]
        h = self.norm(h + dh)

        return x, h


class EGNNRefiner(nn.Module):
    """堆叠多层 EGNN refine anchors。"""
    def __init__(self, feat_dim: int, num_layers: int = 2, hidden_dim: int = 128, coord_scale: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(feat_dim=feat_dim, hidden_dim=hidden_dim, coord_scale=coord_scale)
            for _ in range(int(num_layers))
        ])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, h = layer(x, h)
        return x, h


class AnchorChildDecoder(nn.Module):
    """
    Anchor → Child 展开（两路可用不同超参）：
    - 输入 anchor 的 μ / normal / radius / feat
    - 输出 K 个 child 的 μ0 / scale0 / feat（后续送入 GaussianHead / MotionHead）
    """
    def __init__(
        self,
        in_dim: int,
        num_children: int,
        hidden_dim: int = 256,
        mode: str = "static",  # ["static","dynamic"]
        z_compress: float = 0.25,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_children = int(num_children)
        self.mode = str(mode).lower()
        if self.mode not in ("static", "dynamic"):
            raise ValueError(f"mode must be one of ['static','dynamic'], got {mode}")
        self.z_compress = float(z_compress)

        self.child_embed = nn.Parameter(torch.randn(1, 1, self.num_children, in_dim) * 0.02)
        # 让 child 初始就“贴表面”：用一个固定模板（局部切平面圆盘/椭圆盘），MLP 只学习残差
        self.register_buffer("child_template", self._make_child_template(self.num_children, mode=self.mode).view(1, 1, self.num_children, 3))
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_offset = nn.Linear(hidden_dim, 3)  # local offset (unit cube)
        self.to_scale_delta = nn.Linear(hidden_dim, 3)
        self.to_feat_delta = nn.Linear(hidden_dim, in_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # residual init：offset/scale/feat 的最后线性层置零，初始输出≈模板（更稳定、更快收敛）
        nn.init.zeros_(self.to_offset.weight)
        nn.init.zeros_(self.to_offset.bias)
        nn.init.zeros_(self.to_scale_delta.weight)
        nn.init.zeros_(self.to_scale_delta.bias)
        nn.init.zeros_(self.to_feat_delta.weight)
        nn.init.zeros_(self.to_feat_delta.bias)

    @staticmethod
    def _make_child_template(num_children: int, mode: str = "static") -> torch.Tensor:
        """
        返回 [K,3] 的局部模板点（范围约 [-1,1]）。
        - static：切平面圆盘（z≈0）
        - dynamic：轻微厚度的椭圆盘（z≈0.2*sin）
        """
        K = int(num_children)
        if K <= 0:
            return torch.zeros(0, 3)
        if K == 1:
            return torch.zeros(1, 3)

        mode = str(mode).lower()
        # 使用“向日葵/黄金角”在圆盘上均匀铺点
        idx = torch.arange(K, dtype=torch.float32)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        r = torch.sqrt((idx + 0.5) / float(K)).clamp(0.0, 1.0)
        theta = idx * float(golden_angle)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        if mode == "dynamic":
            z = 0.2 * torch.sin(theta)
        else:
            z = torch.zeros_like(x)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _build_rotation_from_normal(normal: torch.Tensor) -> torch.Tensor:
        # normal: [...,3] -> rot: [...,3,3] with z-axis = normal
        if normal.shape[-1] != 3:
            raise ValueError(f"normal must have last dim = 3, got {tuple(normal.shape)}")
        device, dtype = normal.device, normal.dtype

        n = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp(min=1e-6)

        # Choose a reference axis that is not (nearly) collinear with n to avoid numerical issues.
        mask = (torch.abs(n[..., 0]) < 0.9)  # [...]
        a1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        a2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        a = torch.where(mask.unsqueeze(-1), a1, a2)  # [...,3]

        # Gram-Schmidt: project a onto the plane orthogonal to n.
        t = a - (a * n).sum(dim=-1, keepdim=True) * n
        t = t / torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=1e-6)
        b = torch.cross(n, t, dim=-1)
        rot = torch.stack([t, b, n], dim=-1)  # [...,3,3]
        return rot

    def forward(
        self,
        anchor_mu: torch.Tensor,       # [B,M,3]
        anchor_feat: torch.Tensor,     # [B,M,D]
        anchor_normal: torch.Tensor,   # [B,M,3]
        anchor_radius: torch.Tensor,   # [B,M,1]
    ) -> Dict[str, torch.Tensor]:
        B, M, _ = anchor_mu.shape
        K = self.num_children

        base = anchor_feat.unsqueeze(2) + self.child_embed  # [B,M,K,D]
        h = self.mlp(self.norm(base))  # [B,M,K,H]

        offset_res = torch.tanh(self.to_offset(h))  # [B,M,K,3] in [-1,1]
        offset_local = self.child_template.to(dtype=h.dtype, device=h.device) + 0.2 * offset_res
        if self.mode == "static":
            offset_local[..., 2] = offset_local[..., 2] * self.z_compress

        r = anchor_radius.unsqueeze(2)  # [B,M,1,1]
        offset_local = offset_local * r  # scale by radius

        rot = self._build_rotation_from_normal(anchor_normal)  # [B,M,3,3]
        offset_world = torch.einsum('bmij,bmkj->bmki', rot, offset_local)  # [B,M,K,3]
        child_mu = anchor_mu.unsqueeze(2) + offset_world  # [B,M,K,3]

        # Base scale: static 更扁，dynamic 更饱满
        r1 = anchor_radius.unsqueeze(2).expand(-1, -1, K, -1)  # [B,M,K,1]
        if self.mode == "static":
            base_scale = torch.cat([r1, r1, r1 * self.z_compress], dim=-1)  # [B,M,K,3]
        else:
            base_scale = r1.expand(-1, -1, -1, 3)
        scale_delta = torch.tanh(self.to_scale_delta(h)) * 0.2
        child_scale = base_scale * torch.exp(scale_delta.clamp(min=-5, max=5))

        feat_delta = self.to_feat_delta(h)
        child_feat = anchor_feat.unsqueeze(2) + feat_delta  # [B,M,K,D]

        child_normal = anchor_normal.unsqueeze(2).expand(-1, -1, K, -1)  # [B,M,K,3]
        child_rot = rot.unsqueeze(2).expand(-1, -1, K, -1, -1)  # [B,M,K,3,3]

        # Flatten children
        child_mu = child_mu.reshape(B, M * K, 3)
        child_scale = child_scale.reshape(B, M * K, 3)
        child_feat = child_feat.reshape(B, M * K, self.in_dim)
        child_normal = child_normal.reshape(B, M * K, 3)
        child_rot = child_rot.reshape(B, M * K, 3, 3)

        return {
            'mu': child_mu,
            'scale': child_scale,
            'feat': child_feat,
            'normal': child_normal,
            'rot': child_rot,
        }


class TimeWarpMotionHead(nn.Module):
    """
    Per-Gaussian time-warp: 为每个高斯生成时间动态
    输入：z_g [M,motion_dim]，时间 ID
    输出：per-frame 的 xyz, scale, color, opacity 偏移
    """
    def __init__(
        self,
        in_dim: int,
        motion_dim: int = 128,
        time_emb_dim: int = 32,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, motion_dim), nn.SiLU(),
            nn.Linear(motion_dim, motion_dim), nn.SiLU(),
        )
        self.z_proj = nn.Linear(in_dim, motion_dim)
        self.cond_dim = int(cond_dim)
        self.cond_proj = nn.Linear(cond_dim, motion_dim) if self.cond_dim > 0 else None
        self.out_mlp = nn.Sequential(
            nn.Linear(motion_dim, motion_dim), nn.SiLU(),
            nn.Linear(motion_dim, 3 + 3 + 3 + 1),  # dx, dlog_s, dc, dσ
        )
        # 【稳定性】将 MotionHead 最后一层初始化为零，使初始输出为 0
        if isinstance(self.out_mlp[-1], nn.Linear):
            nn.init.zeros_(self.out_mlp[-1].weight)
            if self.out_mlp[-1].bias is not None:
                nn.init.zeros_(self.out_mlp[-1].bias)
        self.time_emb_dim = time_emb_dim

    @staticmethod
    def _posenc_t(t_scalar: torch.Tensor, dim: int) -> torch.Tensor:
        """时间位置编码"""
        device = t_scalar.device
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, 8, steps=half, device=device))
        phases = t_scalar.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        z_g: torch.Tensor,
        T: int,
        t_ids: torch.Tensor,
        xyz: torch.Tensor,
        scale: torch.Tensor,
        color: torch.Tensor,
        alpha: torch.Tensor,
        disable_color_delta: bool = True,
        motion_cond: Optional[torch.Tensor] = None,  # [T,cond_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_g: [M, in_dim]
            T: 时间帧数
            t_ids: [T]
            xyz: [M,3]
            scale: [M,3]
            color: [M,3]
            alpha: [M,1]
            disable_color_delta: 是否禁用颜色偏移（默认 True，禁用）
            
        Returns:
            xyz_t, scale_t, color_t, alpha_t, dxyz_t: 各为 [T,M,...]
            
        说明：
            - 颜色来自 Stage1 canonical，应保持固定
            - motion 仅作用于 xyz 和 scale，不应频繁更新 color
            - 未来可扩展为 SE(3) motion basis
        """
        M = z_g.shape[0]
        
        # 时间编码
        t_min = t_ids.min()
        t_max = torch.clamp(t_ids.max(), min=t_min + 1)
        t_norm = (t_ids.float() - t_min.float()) / (t_max.float() - t_min.float())
        t_emb = self._posenc_t(t_norm, self.time_emb_dim)  # [T, Dt]
        
        # 投影
        gate = self.time_mlp(t_emb)  # [T, motion_dim]
        if motion_cond is not None:
            if self.cond_proj is None:
                raise ValueError("motion_cond is provided but cond_dim=0 for TimeWarpMotionHead")
            if motion_cond.dim() != 2 or motion_cond.shape[0] != T or motion_cond.shape[1] != self.cond_dim:
                raise ValueError(f"motion_cond must be [T,{self.cond_dim}], got {motion_cond.shape}")
            gate = gate + self.cond_proj(motion_cond.to(device=gate.device, dtype=gate.dtype))
        z_m = self.z_proj(z_g)  # [M, motion_dim]
        
        # 广播和融合
        z_tm = gate.unsqueeze(1) * z_m.unsqueeze(0)  # [T, M, motion_dim]
        out = self.out_mlp(z_tm.view(T * M, -1))  # [T*M, 10]
        out = out.view(T, M, -1)
        
        dxyz = out[..., 0:3]
        dlog_s = out[..., 3:6]
        dc = out[..., 6:9]
        dσ = out[..., 9:10]
        
        # 应用偏移
        xyz_t = xyz.unsqueeze(0) + dxyz  # [T,M,3]
        scale_t = scale.unsqueeze(0) * torch.exp(dlog_s.clamp(min=-5, max=5))  # [T,M,3]
        
        # 【改进】禁用颜色变化（canonical 颜色应固定）
        color_t = color.unsqueeze(0).expand(T, -1, -1)  # [T,M,3]
        
        alpha_t = (alpha.unsqueeze(0) + dσ).clamp(0.0, 1.0)  # [T,M,1]
        
        return xyz_t, scale_t, color_t, alpha_t, dxyz


class AnchorDeltaHead(nn.Module):
    """
    Anchor-level motion head：只预测动态 anchor 的平移 Δxyz(t)。

    - 低频建模：T×Md 的小张量
    - 用于把运动分发到细节点云
    """
    def __init__(
        self,
        in_dim: int,
        motion_dim: int = 128,
        time_emb_dim: int = 32,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.time_emb_dim = int(time_emb_dim)
        self.motion_dim = int(motion_dim)
        self.cond_dim = int(cond_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, motion_dim), nn.SiLU(),
            nn.Linear(motion_dim, motion_dim), nn.SiLU(),
        )
        self.z_proj = nn.Linear(in_dim, motion_dim)
        self.cond_proj = nn.Linear(cond_dim, motion_dim) if self.cond_dim > 0 else None

        self.out_mlp = nn.Sequential(
            nn.Linear(motion_dim, motion_dim), nn.SiLU(),
            nn.Linear(motion_dim, 3),  # dx
        )
        if isinstance(self.out_mlp[-1], nn.Linear):
            nn.init.zeros_(self.out_mlp[-1].weight)
            if self.out_mlp[-1].bias is not None:
                nn.init.zeros_(self.out_mlp[-1].bias)

    @staticmethod
    def _posenc_t(t_scalar: torch.Tensor, dim: int) -> torch.Tensor:
        device = t_scalar.device
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, 8, steps=half, device=device))
        phases = t_scalar.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        z: torch.Tensor,                 # [M,in_dim]
        t_ids: torch.Tensor,             # [T]
        motion_cond: Optional[torch.Tensor] = None,  # [T,cond_dim]
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be [M,D], got {tuple(z.shape)}")
        if t_ids.dim() != 1:
            raise ValueError(f"t_ids must be [T], got {tuple(t_ids.shape)}")

        M = z.shape[0]
        T = int(t_ids.shape[0])

        t_min = t_ids.min()
        t_max = torch.clamp(t_ids.max(), min=t_min + 1)
        t_norm = (t_ids.float() - t_min.float()) / (t_max.float() - t_min.float())
        t_emb = self._posenc_t(t_norm, self.time_emb_dim)  # [T,Dt]

        gate = self.time_mlp(t_emb)  # [T,motion_dim]
        if motion_cond is not None:
            if self.cond_proj is None:
                raise ValueError("motion_cond is provided but cond_dim=0 for AnchorDeltaHead")
            if motion_cond.dim() != 2 or motion_cond.shape[0] != T or motion_cond.shape[1] != self.cond_dim:
                raise ValueError(f"motion_cond must be [T,{self.cond_dim}], got {tuple(motion_cond.shape)}")
            gate = gate + self.cond_proj(motion_cond.to(device=gate.device, dtype=gate.dtype))

        z_m = self.z_proj(z)  # [M,motion_dim]
        z_tm = gate.unsqueeze(1) * z_m.unsqueeze(0)  # [T,M,motion_dim]
        dxyz = self.out_mlp(z_tm.view(T * M, -1)).view(T, M, 3)
        return dxyz


class LowRankMotionBasisHead(nn.Module):
    """
    Learned 4D Basis Prior（低秩时空先验，完全可微）

    形式：
        Δμ_{j,t} = Σ_k w_{j,k} * B_k(t)

    - B_k(t): 由时间 embedding 生成的连续 basis（对所有高斯共享）
    - w_{j,k}: 由 canonical feature 预测的每个高斯的权重
    """
    def __init__(
        self,
        in_dim: int,
        num_bases: int = 8,
        basis_dim: int = 128,
        time_emb_dim: int = 32,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.num_bases = int(num_bases)
        self.time_emb_dim = int(time_emb_dim)
        self.cond_dim = int(cond_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, basis_dim),
            nn.SiLU(),
            nn.Linear(basis_dim, basis_dim),
            nn.SiLU(),
        )
        self.cond_proj = nn.Linear(cond_dim, basis_dim) if self.cond_dim > 0 else None
        self.basis_out = nn.Linear(basis_dim, num_bases * (3 + 3 + 1))  # dx, dlog_s, dσ
        self.weight_out = nn.Linear(in_dim, num_bases)

        # 初始化为 0，确保初始 motion 输出为 0（训练更稳定）
        nn.init.zeros_(self.basis_out.weight)
        if self.basis_out.bias is not None:
            nn.init.zeros_(self.basis_out.bias)

    @staticmethod
    def _posenc_t(t_scalar: torch.Tensor, dim: int) -> torch.Tensor:
        device = t_scalar.device
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, 8, steps=half, device=device))
        phases = t_scalar.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        z_g: torch.Tensor,
        T: int,
        t_ids: torch.Tensor,
        xyz: torch.Tensor,
        scale: torch.Tensor,
        color: torch.Tensor,
        alpha: torch.Tensor,
        disable_color_delta: bool = True,
        motion_cond: Optional[torch.Tensor] = None,  # [T,cond_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        M = z_g.shape[0]

        t_min = t_ids.min()
        t_max = torch.clamp(t_ids.max(), min=t_min + 1)
        t_norm = (t_ids.float() - t_min.float()) / (t_max.float() - t_min.float())
        t_emb = self._posenc_t(t_norm, self.time_emb_dim)  # [T, Dt]

        h_t = self.time_mlp(t_emb)  # [T, basis_dim]
        if motion_cond is not None:
            if self.cond_proj is None:
                raise ValueError("motion_cond is provided but cond_dim=0 for LowRankMotionBasisHead")
            if motion_cond.dim() != 2 or motion_cond.shape[0] != T or motion_cond.shape[1] != self.cond_dim:
                raise ValueError(f"motion_cond must be [T,{self.cond_dim}], got {motion_cond.shape}")
            h_t = h_t + self.cond_proj(motion_cond.to(device=h_t.device, dtype=h_t.dtype))
        basis = self.basis_out(h_t).view(T, self.num_bases, 3 + 3 + 1)  # [T,K,7]

        w = torch.softmax(self.weight_out(z_g), dim=-1)  # [M,K]

        dx_basis = basis[..., 0:3]      # [T,K,3]
        dlog_s_basis = basis[..., 3:6]  # [T,K,3]
        dσ_basis = basis[..., 6:7]      # [T,K,1]

        dxyz = torch.einsum('mk,tkd->tmd', w, dx_basis)        # [T,M,3]
        dlog_s = torch.einsum('mk,tkd->tmd', w, dlog_s_basis)  # [T,M,3]
        dσ = torch.einsum('mk,tkd->tmd', w, dσ_basis)          # [T,M,1]

        xyz_t = xyz.unsqueeze(0) + dxyz
        scale_t = scale.unsqueeze(0) * torch.exp(dlog_s.clamp(min=-5, max=5))
        color_t = color.unsqueeze(0).expand(T, -1, -1)  # canonical 颜色默认固定
        alpha_t = (alpha.unsqueeze(0) + dσ).clamp(0.0, 1.0)

        if not disable_color_delta:
            color_t = color.unsqueeze(0).expand(T, -1, -1)

        return xyz_t, scale_t, color_t, alpha_t, dxyz


class Trellis4DGS4DCanonical(nn.Module):
    """
    4D Gaussian Splatting in world-canonical coordinates (SURFEL 版本)
    
    核心流程：
    1. SURFEL 提取: points_3d -> μ_j, R_j, s_j, confidence
    2. Weighted FPS: 全局选点 (30k → 5k)
    3. Feature Aggregation: feat_2d + μ_j + R_j + s_j -> g_j
    4. Gaussian Head: g_j -> {c_j, o_j, (Δs_j, ΔR_j)}
    5. Motion Head: z_g + time -> per-frame 动态
    """
    def __init__(
        self,
        # SURFEL Extractor
        surfel_k_neighbors: int = 16,
        use_surfel_confidence: bool = True,
        
        # Weighted FPS
        target_num_gaussians: int = 20000,

        # Canonical Prior（方案2：Slot Attention 可微聚类）
        canonical_prior: str = "slot_dual",  # ["surfel", "slot", "slot_dual"]
        slot_iters: int = 3,
        slot_dim: int = 256,
        slot_mlp_hidden: int = 256,

        # Anchor–Child（动静态解耦）
        num_anchors_static: int = 4096,
        num_anchors_dynamic: int = 4096,
        children_per_anchor_static: int = 32,
        children_per_anchor_dynamic: int = 16,
        use_anchor_refiner: bool = False,
        anchor_refiner_layers: int = 2,
        anchor_refiner_hidden: int = 128,
        motion_cond_dim: int = 0,
        
        # Feature Aggregator
        feat_agg_dim: int = 256,
        feat_agg_layers: int = 1,
        feat_agg_heads: int = 4,
        time_emb_dim: int = 32,
        view_emb_dim: int = 32,
        top_k_views: int = 4,
        
        # 2D feature channels (input C)
        feat2d_in_dim: int = 2048,
        
        # Gaussian Head
        gaussian_head_hidden: int = 256,
        use_scale_refine: bool = False,
        use_rot_refine: bool = False,
        
        # Motion Head
        motion_dim: int = 128,
        use_low_rank_motion_basis: bool = False,
        motion_num_bases: int = 8,
        
        # World space config
        aabb_margin: float = 0.05,

        # Stage A/B/C：coarse anchors + fine points
        coarse_stride: int = 8,
        fine_num_points: int = 30000,
        fine_sample_mode: str = "topk_conf",  # ["topk_conf","random"]
        assign_topk: int = 8,
        assign_sigma: float = 0.05,

        # Token subsample（极端分辨率下可启用；0=禁用）
        max_token_points: int = 0,

        # Points-based init（加速收敛）：只用于初始化，不影响端到端梯度
        use_points_init: bool = False,
        points_init_max_tokens: int = 20000,
        points_init_knn_k: int = 32,
        points_init_override_geom: bool = True,
        points_init_radius_min: float = 1e-4,
        points_init_radius_max: float = 0.2,
    ):
        super().__init__()
        
        # 核心模块
        self.surfel_extractor = SurfelExtractor(
            k_neighbors=surfel_k_neighbors,
            use_confidence_weighting=use_surfel_confidence,
        )
        
        self.weighted_fps = WeightedFPS()

        self.canonical_prior = str(canonical_prior).lower()
        if self.canonical_prior not in ("surfel", "slot", "slot_dual"):
            raise ValueError(f"canonical_prior must be one of ['surfel','slot','slot_dual'], got {canonical_prior}")

        self.slot_dim = int(slot_dim)
        self.feat_agg_dim = int(feat_agg_dim)
        self.slot_to_feat = nn.Identity() if self.slot_dim == self.feat_agg_dim else nn.Linear(self.slot_dim, self.feat_agg_dim)

        if self.canonical_prior in ("slot", "slot_dual"):
            # token 特征以 Pi3 的 2D backbone feature 为主；xyz 只用于聚类中心/几何约束
            # Stage 1.1: 点级动静分解（soft）
            # p_dyn = f_dyn(token_xyz, token_feat, temporal statistics)
            self.dyn_pred = nn.Sequential(
                nn.LayerNorm(feat_agg_dim + 1),
                nn.Linear(feat_agg_dim + 1, feat_agg_dim),
                nn.GELU(),
                nn.Linear(feat_agg_dim, 1),
            )
            # 轻量几何 bias（small/low-dim/low-weight）：避免 slot 完全不看几何
            self.geo_bias_dim = 16
            self.geo_bias_scale = nn.Parameter(torch.tensor(0.05))  # 初值小：feat 主导
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(3, self.geo_bias_dim),
                nn.GELU(),
                nn.Linear(self.geo_bias_dim, self.geo_bias_dim),
                nn.GELU(),
            )
            self.geo_bias_to_feat = nn.Linear(self.geo_bias_dim, feat_agg_dim)
            nn.init.zeros_(self.geo_bias_to_feat.weight)
            if self.geo_bias_to_feat.bias is not None:
                nn.init.zeros_(self.geo_bias_to_feat.bias)
            if self.canonical_prior == "slot":
                self.slot_prior = SlotAttentionGaussianPrior(
                    num_slots=target_num_gaussians,
                    token_dim=feat_agg_dim,
                    slot_dim=slot_dim,
                    iters=slot_iters,
                    mlp_hidden=slot_mlp_hidden,
                )
            else:
                self.dual_slot_prior = DualSlotAttentionGaussianPrior(
                    num_slots_static=num_anchors_static,
                    num_slots_dynamic=num_anchors_dynamic,
                    token_dim=feat_agg_dim,
                    slot_dim=slot_dim,
                    iters=slot_iters,
                    mlp_hidden=slot_mlp_hidden,
                )
                self.static_child_decoder = AnchorChildDecoder(
                    in_dim=feat_agg_dim,
                    num_children=children_per_anchor_static,
                    hidden_dim=feat_agg_dim,
                    mode="static",
                    z_compress=0.25,
                )
                self.dynamic_child_decoder = AnchorChildDecoder(
                    in_dim=feat_agg_dim,
                    num_children=children_per_anchor_dynamic,
                    hidden_dim=feat_agg_dim,
                    mode="dynamic",
                    z_compress=1.0,
                )

                self.use_anchor_refiner = bool(use_anchor_refiner)
                if self.use_anchor_refiner:
                    raise ValueError("按当前方案不允许使用 EGNN（use_anchor_refiner 必须为 False）")
                self.anchor_refiner_static = None
                self.anchor_refiner_dynamic = None

                # 统一的几何头：从 refined anchor_feat 预测 normal / radius
                self.anchor_normal_head = nn.Sequential(
                    nn.LayerNorm(feat_agg_dim),
                    nn.Linear(feat_agg_dim, feat_agg_dim),
                    nn.GELU(),
                    nn.Linear(feat_agg_dim, 3),
                )
                self.anchor_radius_head = nn.Sequential(
                    nn.LayerNorm(feat_agg_dim),
                    nn.Linear(feat_agg_dim, feat_agg_dim),
                    nn.GELU(),
                    nn.Linear(feat_agg_dim, 1),
                )

                self.motion_cond_dim = int(motion_cond_dim)
                self.motion_cond_mlp = None
                if self.motion_cond_dim > 0:
                    # 默认从 (mean, std) of 2D traj（4维）映射到 cond_dim
                    self.motion_cond_mlp = nn.Sequential(
                        nn.Linear(4, self.motion_cond_dim),
                        nn.SiLU(),
                        nn.Linear(self.motion_cond_dim, self.motion_cond_dim),
                    )
                    # 关键点条件：从 (mean,std) of 3D keypoints（6维）映射到 cond_dim
                    self.keypoint_cond_mlp = nn.Sequential(
                        nn.Linear(6, self.motion_cond_dim),
                        nn.SiLU(),
                        nn.Linear(self.motion_cond_dim, self.motion_cond_dim),
                    )
        else:
            self.motion_cond_dim = int(motion_cond_dim)
            self.motion_cond_mlp = None
            self.keypoint_cond_mlp = None
        
        # 2D特征通道降维：C_in -> feat_agg_dim（例如 2048 -> 256）
        self.feat_reduce = nn.Conv2d(feat2d_in_dim, feat_agg_dim, kernel_size=1)
        # coarse slot token 的可训练投影（Pi3-native：feat 主导）
        self.slot_feat_proj = nn.Sequential(
            nn.LayerNorm(feat_agg_dim),
            nn.Linear(feat_agg_dim, feat_agg_dim),
        )
        nn.init.zeros_(self.slot_feat_proj[-1].bias)

        # 特征聚合器（降低 hidden_dim，并限制 top-K 视角）
        self.feature_aggregator = PerGaussianAggregator(
            feat_dim=feat_agg_dim,
            num_layers=1,  # 单层 Transformer
            num_heads=feat_agg_heads,
            hidden_dim=256,  # 降低 hidden_dim，减少激活内存
            time_emb_dim=time_emb_dim,
            view_emb_dim=view_emb_dim,
            topk_views=top_k_views,
        )
        
        self.gaussian_head = GaussianHead(
            in_dim=feat_agg_dim,
            hidden_dim=gaussian_head_hidden,
            color_dim=3,
            use_scale_refine=use_scale_refine,
            use_rot_refine=use_rot_refine,
        )
        
        self.use_low_rank_motion_basis = bool(use_low_rank_motion_basis)
        if self.use_low_rank_motion_basis:
            self.motion_head = LowRankMotionBasisHead(
                in_dim=motion_dim,
                num_bases=motion_num_bases,
                basis_dim=motion_dim,
                time_emb_dim=time_emb_dim,
                cond_dim=self.motion_cond_dim,
            )
        else:
            self.motion_head = TimeWarpMotionHead(
                in_dim=motion_dim,
                motion_dim=motion_dim,
                time_emb_dim=time_emb_dim,
                cond_dim=self.motion_cond_dim,
            )
        
        # 特征投影（用于 motion head）
        self.feat_to_motion = nn.Linear(feat_agg_dim, motion_dim)
        # anchor motion（只输出 Δxyz_d(t)）
        self.anchor_motion_head = AnchorDeltaHead(
            in_dim=motion_dim,
            motion_dim=motion_dim,
            time_emb_dim=time_emb_dim,
            cond_dim=0,
        )
        
        # 缓存（每个 scene 需要独立的缓存）
        self._world_cache: Dict[str, Optional[torch.Tensor]] = {
            'prepared': False,  # 标记是否已准备好 canonical
            'aabb': None,
            'surfel_mu': None,
            'surfel_normal': None,
            'surfel_radius': None,
            'surfel_confidence': None,
            'selected_indices': None,
        }
        
        self.aabb_margin = float(aabb_margin)
        self.target_num_gaussians = int(target_num_gaussians)
        self.max_token_points = int(max_token_points)
        self.coarse_stride = int(coarse_stride)
        self.fine_num_points = int(fine_num_points)
        self.fine_sample_mode = str(fine_sample_mode)
        self.assign_topk = int(assign_topk)
        self.assign_sigma = float(assign_sigma)

        # points init
        self.use_points_init = bool(use_points_init)
        if self.use_points_init:
            raise ValueError("按当前方案不允许使用 no_grad points_init（use_points_init 必须为 False）")
        self.points_init_max_tokens = int(points_init_max_tokens)
        self.points_init_knn_k = int(points_init_knn_k)
        self.points_init_override_geom = bool(points_init_override_geom)
        self.points_init_radius_min = float(points_init_radius_min)
        self.points_init_radius_max = float(points_init_radius_max)
        self._points_init_done = False

    def reset_cache(self):
        """
        重置缓存（多场景训练时必须调用）
        
        ⚠️ 重要说明：
        在训练多个场景时，每个新场景加载前必须显式调用此方法，
        否则会复用上一个场景的 canonical 数据，导致完全错误的结果。
        
        使用示例：
        ```python
        model = Trellis4DGS4DCanonical(...)
        
        # 场景 1
        model.reset_cache()  # ✅ 必须调用
        out1 = model(points_3d=pts1, feat_2d=feat1, ...)
        
        # 场景 2
        model.reset_cache()  # ✅ 必须调用
        out2 = model(points_3d=pts2, feat_2d=feat2, ...)
        ```
        
        缓存内容：
        - prepared: bool，标记 canonical 是否已准备
        - aabb: [2,3] 世界 AABB
        - surfel_mu: [M,3] SURFEL 中心
        - surfel_normal: [M,3] SURFEL 法线
        - surfel_radius: [M,1] SURFEL 半径
        - surfel_confidence: [M,1] SURFEL 置信度
        - selected_indices: [K] 加权 FPS 选中的索引
        """
        self._world_cache = {
            'prepared': False,
            'aabb': None,
            'surfel_mu': None,
            'surfel_normal': None,
            'surfel_radius': None,
            'surfel_confidence': None,
            'selected_indices': None,
        }
        self._points_init_done = False

    @staticmethod
    def _weighted_fps_centers(xyz: torch.Tensor, weight: torch.Tensor, num_centers: int) -> torch.Tensor:
        """
        xyz: [N,3], weight: [N] (>=0)
        返回 centers: [M,3]，用于初始化（no-grad 友好）
        """
        if xyz.dim() != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
        if weight.dim() != 1 or weight.shape[0] != xyz.shape[0]:
            raise ValueError(f"weight must be [N], got {tuple(weight.shape)}")
        N = xyz.shape[0]
        M = int(min(max(0, num_centers), N))
        if M == 0:
            return xyz.new_zeros(0, 3)

        w = weight.clamp(min=0.0)
        if float(w.sum()) <= 1e-8:
            w = torch.ones_like(w)

        # first: max weight
        first = torch.argmax(w)
        selected = [first]
        min_dist2 = torch.full((N,), float("inf"), device=xyz.device, dtype=xyz.dtype)
        for _ in range(1, M):
            last = selected[-1]
            d2 = ((xyz - xyz[last].unsqueeze(0)) ** 2).sum(dim=-1)
            min_dist2 = torch.minimum(min_dist2, d2)
            score = min_dist2 * (w + 1e-6)
            nxt = torch.argmax(score)
            selected.append(nxt)
        idx = torch.stack(selected, dim=0)
        return xyz[idx]

    @staticmethod
    def _subsample_by_weight(xyz: torch.Tensor, feat: torch.Tensor, weight: torch.Tensor, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按权重做无梯度子采样，返回 xyz/feat/weight 的子集。"""
        N = xyz.shape[0]
        K = int(min(max_tokens, N))
        if K <= 0 or N == 0:
            return xyz[:0], feat[:0], weight[:0]
        if N <= K:
            return xyz, feat, weight
        w = weight.clamp(min=0.0)
        if float(w.sum()) <= 1e-8:
            idx = torch.randperm(N, device=xyz.device)[:K]
        else:
            prob = (w + 1e-8) / (w.sum() + 1e-8)
            idx = torch.multinomial(prob, K, replacement=False)
        return xyz[idx], feat[idx], weight[idx]

    def _estimate_normal_radius_knn(self, centers: torch.Tensor, xyz: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        centers: [M,3], xyz: [N,3]
        返回：
          normal [M,3]：PCA 最小特征值对应向量
          radius [M,1]：邻域 RMS 距离（clamp）
        """
        orig_dtype = centers.dtype
        orig_device = centers.device
        M = centers.shape[0]
        if M == 0:
            return centers.new_zeros(0, 3), centers.new_zeros(0, 1)
        N = xyz.shape[0]
        kk = int(min(max(1, k), max(1, N)))

        # AMP 下即使输入是 fp32，一些算子也可能被 autocast 到 fp16，导致 eigh 报错；
        # 这里显式关闭 autocast，并强制使用 fp32 做 PCA 统计。
        if centers.is_cuda:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            autocast_ctx = torch.autocast(device_type=orig_device.type, enabled=False) if hasattr(torch, "autocast") else None

        if autocast_ctx is None:
            centers_f = centers.to(dtype=torch.float32)
            xyz_f = xyz.to(dtype=torch.float32)
            dist2 = torch.cdist(centers_f, xyz_f, p=2.0).pow(2)
            nn_d2, nn_idx = torch.topk(dist2, k=kk, dim=-1, largest=False)
            nn = xyz_f[nn_idx]
            mu = nn.mean(dim=1, keepdim=True)
            X = nn - mu
            cov = torch.matmul(X.transpose(1, 2), X) / float(max(1, kk))
            cov = cov.to(dtype=torch.float32)
            _, evecs = torch.linalg.eigh(cov)
            normal = evecs[..., 0]
            normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp(min=1e-6)
            radius = torch.sqrt(nn_d2.mean(dim=-1, keepdim=True).clamp(min=1e-10))
            radius = radius.clamp(min=self.points_init_radius_min, max=self.points_init_radius_max)
        else:
            with autocast_ctx:
                centers_f = centers.to(dtype=torch.float32)
                xyz_f = xyz.to(dtype=torch.float32)
                dist2 = torch.cdist(centers_f, xyz_f, p=2.0).pow(2)
                nn_d2, nn_idx = torch.topk(dist2, k=kk, dim=-1, largest=False)
                nn = xyz_f[nn_idx]
                mu = nn.mean(dim=1, keepdim=True)
                X = nn - mu
                cov = torch.matmul(X.transpose(1, 2), X) / float(max(1, kk))
                cov = cov.to(dtype=torch.float32)
                _, evecs = torch.linalg.eigh(cov)  # ascending (fp32)
                normal = evecs[..., 0]
                normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp(min=1e-6)
                radius = torch.sqrt(nn_d2.mean(dim=-1, keepdim=True).clamp(min=1e-10))
                radius = radius.clamp(min=self.points_init_radius_min, max=self.points_init_radius_max)

        return normal.to(device=orig_device, dtype=orig_dtype), radius.to(device=orig_device, dtype=orig_dtype)

    def reset_world_cache(self):
        """【废弃】使用 reset_cache() 代替"""
        self.reset_cache()

    def prepare_canonical(
        self,
        points_3d: torch.Tensor,
        use_temporal_aware: bool = True,
        token_feat: Optional[torch.Tensor] = None,
        token_xyz: Optional[torch.Tensor] = None,
    ):
        """
        前置准备 canonical 高斯（必须在 forward 前调用）
        
        【改进版】时间感知的动态采样
        
        流程：
        1. 分帧采样：每帧独立采样 k 个点，保留时间结构
        2. 去重合并：识别空间接近的点，合并为单一 SURFEL
        3. 时间置信度：计算点在时间上的稳定性
        4. SurfelExtractor：在去重点上做 PCA
        5. Weighted FPS：根据几何+时间置信度选点
        
        优点：
        - ✅ 区分静态点和动态点
        - ✅ 去重后点数更少，计算更快
        - ✅ 时间置信度更准确
        - ✅ 几何覆盖率更高
        
        Args:
            points_3d: [T,N,3] 点云
            use_temporal_aware: 是否使用时间感知采样（推荐 True）
            token_feat: [B,N,C] 可选的 token 特征（用于 slot prior）
            token_xyz: [B,N,3] 可选的 token 3D 坐标（用于 slot prior）
        """
        # 检查是否已准备
        if self._world_cache.get('prepared', False):
            return
        
        device = points_3d.device
        dtype = points_3d.dtype
        
        # Step 0: 估计 world AABB
        if self._world_cache['aabb'] is None:
            aabb = self.estimate_points_aabb(points_3d, margin=self.aabb_margin)
            self._world_cache['aabb'] = aabb

        # ======== Slot Attention Prior（方案2：可微 soft clustering） ========
        if self.canonical_prior == "slot" and token_xyz is not None:
            if token_xyz.dim() != 3 or token_xyz.shape[-1] != 3:
                raise ValueError(f"token_xyz must be [B,N,3], got {token_xyz.shape}")
            if token_feat is not None and (token_feat.dim() != 3 or token_feat.shape[:2] != token_xyz.shape[:2]):
                raise ValueError(f"token_feat must be [B,N,C] and match token_xyz, got {token_feat.shape} vs {token_xyz.shape}")
            # Pi3-native：slot attention 主要看 feat；xyz 只用于 μ 的落点
            if token_feat is None:
                if not hasattr(self, "geo_bias_mlp") or self.geo_bias_mlp is None:
                    raise ValueError("token_feat is None but geo_bias_mlp is not available to build feature tokens")
                token_feat_eff = self.geo_bias_to_feat(self.geo_bias_mlp(token_xyz.to(dtype=points_3d.dtype)))  # [B,N,C]
            else:
                token_feat_eff = token_feat
                if hasattr(self, "geo_bias_mlp") and self.geo_bias_mlp is not None:
                    geo_small = self.geo_bias_mlp(token_xyz.to(dtype=token_feat_eff.dtype))
                    geo_bias = self.geo_bias_to_feat(geo_small)
                    s = torch.clamp(self.geo_bias_scale, 0.0, 1.0)
                    token_feat_eff = token_feat_eff + s * geo_bias

            prior = self.slot_prior(token_feat_eff, token_xyz)
            mu = prior['mu'].squeeze(0)
            normal = prior['normal'].squeeze(0)
            radius = prior['radius'].squeeze(0)
            confidence = prior['confidence'].squeeze(0)

            self._world_cache.update({
                'surfel_mu': mu.to(device=device, dtype=dtype),
                'surfel_normal': normal.to(device=device, dtype=dtype),
                'surfel_radius': radius.to(device=device, dtype=dtype),
                'surfel_confidence': confidence.to(device=device, dtype=dtype),
                'selected_indices': None,
                'prepared': True,
            })
            return
        
        if use_temporal_aware and points_3d.dim() == 3:
            # ========== 时间感知采样 ==========
            T, N, _ = points_3d.shape
            k_per_frame = 2000  # 每帧采样 2k
            
            # Step 1: 分帧采样
            points_sampled_list = []
            frame_indices = []
            
            for t in range(T):
                pts_t = points_3d[t]  # [N, 3]
                valid_mask = torch.isfinite(pts_t).all(dim=-1)
                pts_valid = pts_t[valid_mask]
                
                if pts_valid.shape[0] > k_per_frame:
                    # 该帧点数过多，采样
                    idx = torch.randperm(pts_valid.shape[0], device=device)[:k_per_frame]
                    pts_sampled = pts_valid[idx]
                else:
                    # 该帧点数较少，全部保留
                    pts_sampled = pts_valid
                
                points_sampled_list.append(pts_sampled)
                frame_indices.append(torch.full((pts_sampled.shape[0],), t, dtype=torch.long, device=device))
            
            # 拼接所有帧的采样点
            points_all = torch.cat(points_sampled_list, dim=0)  # [T*k_per_frame, 3]
            frame_ids = torch.cat(frame_indices, dim=0)  # [T*k_per_frame]
            
            # Step 2: 去重合并（voxel grid）
            voxel_size = 0.01  # 1cm
            voxel_indices = torch.floor(points_all / voxel_size).long()
            
            # 创建唯一的 voxel ID
            unique_voxels, inverse_indices = torch.unique(
                voxel_indices, dim=0, return_inverse=True
            )
            
            # 每个 voxel 中的点进行平均和时间统计
            points_merged = []
            time_stability = []  # 该 voxel 出现的帧数 / 总帧数
            
            for i in range(len(unique_voxels)):
                mask = inverse_indices == i
                pts_in_voxel = points_all[mask]
                frames_in_voxel = frame_ids[mask]
                
                # 空间平均
                pt_merged = pts_in_voxel.mean(dim=0)
                points_merged.append(pt_merged)
                
                # 时间稳定性：出现的不同帧数 / 总帧数
                num_frames = len(torch.unique(frames_in_voxel))
                stability = num_frames / T
                time_stability.append(stability)
            
            points_merged = torch.stack(points_merged, dim=0)  # [M, 3]
            time_stability = torch.tensor(time_stability, device=device, dtype=dtype)  # [M]
            
        else:
            # ========== 原始采样（兼容非 3D 输入） ==========
            # 时序汇合
            if points_3d.dim() == 3:
                points_all = points_3d.reshape(-1, 3)
            else:
                points_all = points_3d.view(-1, 3)
            
            # 过滤无效点
            valid_mask = torch.isfinite(points_all).all(dim=-1)
            points_all = points_all[valid_mask]
            
            if points_all.shape[0] > 50000:
                # 随机子采样到 20k，避免 FPS 的 NxN cdist OOM
                rand_idx = torch.randperm(points_all.shape[0], device=device)[:50000]
                points_merged = points_all[rand_idx]
            else:
                points_merged = points_all
            
            time_stability = torch.ones(points_merged.shape[0], device=device, dtype=dtype)
        
        # Step 3: SurfelExtractor（在去重点上做 PCA）
        # 重要：SurfelExtractor 内部会做 fps_target 子采样（默认 20k），
        # 为避免 time_stability 与 surfel_confidence 维度不一致，这里先做同步子采样。
        surfel_fps_target = 20000
        if points_merged.shape[0] > surfel_fps_target:
            rand_idx = torch.randperm(points_merged.shape[0], device=device)[:surfel_fps_target]
            points_merged = points_merged[rand_idx]
            time_stability = time_stability[rand_idx]

        surfel_data = self.surfel_extractor(points_merged, fps_target=points_merged.shape[0])
        surfel_mu = surfel_data['mu']  # [N_surfel, 3]
        surfel_normal = surfel_data['normal']  # [N_surfel, 3]
        surfel_radius = surfel_data['radius']  # [N_surfel, 1]
        surfel_confidence = surfel_data['confidence']  # [N_surfel, 1]
        
        # Step 4: 融合时间和几何置信度
        # 综合置信度 = 几何置信度 × 时间稳定性
        # 这样既考虑了表面平坦度，也考虑了时间一致性
        combined_confidence = (surfel_confidence.squeeze(-1) * time_stability).unsqueeze(-1)  # [N_surfel, 1]
        
        N_surfel = surfel_mu.shape[0]
        
        # Step 5: Weighted FPS → 5k
        target_k = min(self.target_num_gaussians, N_surfel)
        selected_indices, mu = self.weighted_fps.forward(
            surfel_mu,
            combined_confidence,
            target_k,
        )
        
        # 对应的法线、半径、置信度
        surfel_normal = surfel_normal[selected_indices]
        surfel_radius = surfel_radius[selected_indices]
        surfel_confidence = combined_confidence[selected_indices]
        
        # 缓存结果
        self._world_cache.update({
            'surfel_mu': mu,
            'surfel_normal': surfel_normal,
            'surfel_radius': surfel_radius,
            'surfel_confidence': surfel_confidence,
            'selected_indices': selected_indices,
            'prepared': True,
        })

    def set_world_aabb(self, aabb: torch.Tensor):
        """设置世界 AABB"""
        if not torch.is_tensor(aabb):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        self._world_cache['aabb'] = aabb.to(device=dev, dtype=dt)

    @staticmethod
    def estimate_points_aabb(points_3d: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
        """从点云估计 AABB"""
        if points_3d is None or points_3d.numel() == 0:
            raise ValueError("points_3d is empty")
        
        if points_3d.dim() == 3:
            pts = points_3d.reshape(-1, 3)
        else:
            pts = points_3d.view(-1, 3)
        
        device, dtype = pts.device, pts.dtype
        mask = torch.isfinite(pts).all(dim=-1)
        pts = pts[mask]
        
        if pts.numel() == 0:
            raise ValueError("points_3d has no finite entries")
        
        minb = torch.quantile(pts, 0.01, dim=0)
        maxb = torch.quantile(pts, 0.99, dim=0)
        extent = (maxb - minb).clamp(min=1e-6)
        center = (minb + maxb) * 0.5
        minb = center - extent * (0.5 + margin)
        maxb = center + extent * (0.5 + margin)
        
        return torch.stack([minb, maxb], dim=0).to(device=device, dtype=dtype)

    @staticmethod
    def _soft_assign_topk(
        x: torch.Tensor,           # [M,3]
        anchors: torch.Tensor,     # [A,3]
        topk: int = 8,
        sigma: float = 0.05,
        chunk: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对每个点 x 找 top-k anchors 并做 RBF-softmax assignment（可微 w.r.t x/anchors）。

        Returns:
          idx: [M,K] long
          w:   [M,K] float
          d2:  [M,K] float (squared dist)
        """
        if x.dim() != 2 or x.shape[-1] != 3:
            raise ValueError(f"x must be [M,3], got {tuple(x.shape)}")
        if anchors.dim() != 2 or anchors.shape[-1] != 3:
            raise ValueError(f"anchors must be [A,3], got {tuple(anchors.shape)}")

        M = int(x.shape[0])
        A = int(anchors.shape[0])
        if A == 0:
            dev, dt = x.device, x.dtype
            return (
                torch.zeros(M, 0, device=dev, dtype=torch.long),
                torch.zeros(M, 0, device=dev, dtype=dt),
                torch.zeros(M, 0, device=dev, dtype=dt),
            )
        K = int(min(max(1, topk), A))
        if M == 0:
            dev, dt = x.device, x.dtype
            return (
                torch.zeros(0, K, device=dev, dtype=torch.long),
                torch.zeros(0, K, device=dev, dtype=dt),
                torch.zeros(0, K, device=dev, dtype=dt),
            )

        idx_out = torch.empty(M, K, device=x.device, dtype=torch.long)
        w_out = torch.empty(M, K, device=x.device, dtype=x.dtype)
        d2_out = torch.empty(M, K, device=x.device, dtype=x.dtype)

        sigma2 = float(max(1e-8, sigma * sigma))
        anchors32 = anchors.to(dtype=torch.float32)
        for i0 in range(0, M, int(max(1, chunk))):
            x_chunk = x[i0:i0 + chunk]  # [m,3]
            d2 = torch.cdist(x_chunk.to(dtype=torch.float32), anchors32, p=2.0).pow(2)  # [m,A] fp32
            d2k, idx = torch.topk(d2, k=K, dim=-1, largest=False)
            logits = -d2k / (2.0 * sigma2)
            w = torch.softmax(logits, dim=-1)
            m = int(x_chunk.shape[0])
            idx_out[i0:i0 + m] = idx
            w_out[i0:i0 + m] = w.to(dtype=x.dtype)
            d2_out[i0:i0 + m] = d2k.to(dtype=x.dtype)

        return idx_out, w_out, d2_out

    def forward(
        self,
        points_full: Optional[torch.Tensor] = None,        # [T,V,H,W,3]
        points_3d: Optional[torch.Tensor] = None,          # [T,N,3]
        feat_2d: Optional[torch.Tensor] = None,            # [T,V,H',W',C]
        camera_poses: Optional[torch.Tensor] = None,       # [T,V,4,4]
        camera_K: Optional[torch.Tensor] = None,           # [T,V,3,3] (alias)
        camera_intrinsics: Optional[torch.Tensor] = None,  # [T,V,3,3]
        time_ids: Optional[torch.Tensor] = None,           # [T]
        dyn_mask_2d: Optional[torch.Tensor] = None,        # [T,V,H,W] or [T,V,H,W,1]
        conf_2d: Optional[torch.Tensor] = None,            # [T,V,H,W] or [T,V,H,W,1]
        motion_cond: Optional[torch.Tensor] = None,        # [T,Cm]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward（推荐版本，按 Stage0~Stage6）：
        
        Args:
            points_full: [T,V,H,W,3]  (高质量稠密多视角 3D 点图)
            points_3d: [T,N,3]   (可选/兼容；surfel 分支或 AABB 辅助)
            feat_2d: [T,V,H',W',C]
            camera_poses: [T,V,4,4]
            camera_K/camera_intrinsics: [T,V,3,3]
            time_ids: [T]
            dyn_mask_2d: [T,V,H,W] or [T,V,H,W,1] (optional)
            conf_2d: [T,V,H,W] or [T,V,H,W,1] (optional)
            motion_cond: [T,Cm] (optional)
            
        Returns:
            dict with keys:
                - mu_t: [T,M,3] per-frame 高斯中心
                - scale_t: [T,M,3] per-frame 尺度
                - rot_t: [T,M,3,3] per-frame 旋转矩阵
                - color_t: [T,M,3] per-frame 颜色
                - alpha_t: [T,M,1] per-frame 不透明度
                - dxyz_t: [T,M,3] 动态偏移
                - world_aabb: [2,3]
                - surfel_mu: [M,3] SURFEL 中心
                - surfel_normal: [M,3] SURFEL 法线
                - surfel_radius: [M,1] SURFEL 半径
        """
        assert feat_2d is not None, "feat_2d is required"
        assert camera_poses is not None, "camera_poses is required"
        if camera_K is None:
            camera_K = kwargs.get("camera_K", None)
        if camera_K is not None:
            camera_intrinsics = camera_K
        assert camera_intrinsics is not None, "camera_K/camera_intrinsics is required"
        assert time_ids is not None, "time_ids is required"
        
        device = feat_2d.device
        dtype = feat_2d.dtype
        T = feat_2d.shape[0]

        # 兼容旧调用：points/points_full/conf/dyn_mask 作为 kwargs 传入
        if points_full is None:
            points_full = kwargs.get("points_full", None)
        points_map = points_full if points_full is not None else kwargs.get("points", None)  # [T,V,H,W,3]
        conf = conf_2d if conf_2d is not None else (kwargs.get("conf_2d", None) or kwargs.get("conf", None))
        dyn_mask = dyn_mask_2d if dyn_mask_2d is not None else (kwargs.get("dyn_mask_2d", None) or kwargs.get("dyn_mask", None))
        dyn_traj = kwargs.get("dyn_traj", None)
        keypoints_2d = kwargs.get("keypoints_2d", None)
        build_canonical = bool(kwargs.get("build_canonical", False))
        # 兼容：有些旧代码把 [T,N,3] 误传到 points 参数里
        if points_map is not None and points_map.dim() == 3 and points_3d is None:
            points_3d = points_map
            points_map = None
        if points_map is not None and (points_map.dim() != 5 or points_map.shape[-1] != 3):
            raise ValueError(f"points must be [T,V,H,W,3], got {tuple(points_map.shape)}")

        # Step 1: 2D 特征降维（slot prior / aggregator 复用）
        # feat_2d: [T,V,H',W',C_in] -> [T,V,H',W',C_out]
        T_, V_, Hp_, W_, Cin_ = feat_2d.shape
        feat_nchw = feat_2d.permute(0, 1, 4, 2, 3).contiguous().view(T_ * V_, Cin_, Hp_, W_)
        feat_red_nchw = self.feat_reduce(feat_nchw)  # [T*V, C_out, H', W']
        C_out = feat_red_nchw.shape[1]
        feat_red = feat_red_nchw.view(T_, V_, C_out, Hp_, W_).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,H',W',C_out]

        # ======== Stage 0: points -> points_ds（对齐到 H',W'，可微） ========
        points_ds = None  # [T,V,H',W',3]
        H_img = None
        W_img = None
        if self.canonical_prior in ("slot", "slot_dual"):
            if points_map is not None:
                if points_map.shape[0] != T_ or points_map.shape[1] != V_:
                    raise ValueError(f"points must be [T,V,H,W,3]=[{T_},{V_},H,W,3], got {tuple(points_map.shape)}")
                points_map = points_map.to(device=device, dtype=dtype)
                H_img, W_img = int(points_map.shape[2]), int(points_map.shape[3])
                pts_tvchw = points_map.permute(0, 1, 4, 2, 3).contiguous().view(T_ * V_, 3, H_img, W_img)
                pts_ds = F.interpolate(pts_tvchw, size=(Hp_, W_), mode="area")  # [T*V,3,H',W']
                points_ds = pts_ds.view(T_, V_, 3, Hp_, W_).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,H',W',3]
                points_ds = torch.nan_to_num(points_ds, nan=0.0, posinf=0.0, neginf=0.0)
            elif points_3d is not None:
                if points_3d.dim() != 3 or points_3d.shape[-1] != 3:
                    raise ValueError(f"points_3d must be [T,N,3], got {tuple(points_3d.shape)}")
                if points_3d.shape[0] != T_:
                    raise ValueError(f"T mismatch: points_3d T={points_3d.shape[0]} vs feat_2d T={T_}")

                cx0 = float(camera_intrinsics[0, 0, 0, 2].detach().item())
                cy0 = float(camera_intrinsics[0, 0, 1, 2].detach().item())
                W_img = int(round(2.0 * cx0))
                H_img = int(round(2.0 * cy0))
                if W_img <= 0 or H_img <= 0:
                    raise ValueError(f"Invalid inferred H/W from camera_intrinsics: H={H_img}, W={W_img}")
                N_pix = int(H_img * W_img)
                if points_3d.shape[1] == N_pix:
                    pts_t1hw3 = points_3d.to(device=device, dtype=dtype).view(T_, 1, H_img, W_img, 3)
                    pts_t1hw3 = torch.nan_to_num(pts_t1hw3, nan=0.0, posinf=0.0, neginf=0.0)
                    pts_bchw = pts_t1hw3.permute(0, 1, 4, 2, 3).contiguous().view(T_, 3, H_img, W_img)  # [T,3,H,W]
                    pts_ds = F.interpolate(pts_bchw, size=(Hp_, W_), mode="area")  # [T,3,H',W']
                    points_ds_1 = pts_ds.view(T_, 1, 3, Hp_, W_).permute(0, 1, 3, 4, 2).contiguous()  # [T,1,H',W',3]
                    points_ds = points_ds_1.repeat(1, V_, 1, 1, 1)  # [T,V,H',W',3]

        # ======== Branch 1: SURFEL（旧版，含缓存） ========
        if self.canonical_prior == "surfel":
            points_3d_eff = points_3d
            if points_3d_eff is None:
                if points_map is None:
                    raise ValueError("canonical_prior='surfel' 需要 points_3d，或提供 points[T,V,H,W,3] 以便展开为点云")
                if points_map.shape[0] != T:
                    raise ValueError(f"T mismatch between feat_2d (T={T}) and points (T={points_map.shape[0]})")
                points_3d_eff = points_map.reshape(T, -1, 3)
            self.prepare_canonical(points_3d_eff)

            world_aabb = self._world_cache['aabb']
            mu = self._world_cache['surfel_mu']              # [M,3]
            surfel_normal = self._world_cache['surfel_normal']  # [M,3]
            surfel_radius = self._world_cache['surfel_radius']  # [M,1]

            M = mu.shape[0]
            if M == 0:
                return {
                    'mu_t': torch.zeros(T, 0, 3, device=device, dtype=dtype),
                    'scale_t': torch.zeros(T, 0, 3, device=device, dtype=dtype),
                    'rot_t': torch.zeros(T, 0, 3, 3, device=device, dtype=dtype),
                    'color_t': torch.zeros(T, 0, 3, device=device, dtype=dtype),
                    'alpha_t': torch.zeros(T, 0, 1, device=device, dtype=dtype),
                    'dxyz_t': torch.zeros(T, 0, 3, device=device, dtype=dtype),
                    'world_aabb': world_aabb,
                    'surfel_mu': mu,
                    'surfel_normal': surfel_normal,
                    'surfel_radius': surfel_radius,
                }

            g = self.feature_aggregator(
                mu, feat_red, camera_poses, camera_intrinsics, time_ids,
                surfel_normal=surfel_normal,
                surfel_radius=surfel_radius,
            )

            surfel_rot = self._build_rotation_from_normal(surfel_normal)  # [M,3,3]
            gaussian_params = self.gaussian_head(
                g,
                surfel_scale=surfel_radius.expand(-1, 3),
                surfel_rot=surfel_rot,
            )

            scale = gaussian_params['scale']
            opacity = gaussian_params['opacity']
            color = gaussian_params['color']

            z_g = self.feat_to_motion(g)
            xyz_t, scale_t, color_t, alpha_t, dxyz_t = self.motion_head(
                z_g, T=T, t_ids=time_ids,
                xyz=mu, scale=scale, color=color, alpha=opacity,
                disable_color_delta=True,
            )

            return {
                'mu_t': xyz_t,
                'scale_t': scale_t,
                'rot_t': surfel_rot.unsqueeze(0).expand(T, -1, -1, -1),
                'color_t': color_t,
                'alpha_t': alpha_t,
                'dxyz_t': dxyz_t,
                'world_aabb': world_aabb,
                'surfel_mu': mu,
                'surfel_normal': surfel_normal,
                'surfel_radius': surfel_radius,
            }

        # ======== Slot tokens (shared) ========
        def _build_slot_tokens_from_points() -> Tuple[torch.Tensor, torch.Tensor, int, int]:
            if points_ds is None or H_img is None or W_img is None:
                raise ValueError("slot-based canonical_prior 需要 points 或可 reshape 的 points_3d，用于 Stage0 生成 points_ds[T,V,H',W',3]")

            token_xyz_ = points_ds.view(1, -1, 3)            # [1,N,3]
            token_feat_ = feat_red.view(1, -1, C_out)          # [1,N,C]

            # Pi3-native：slot 注意力空间以 feat 为主；xyz 仅用于 μ 的加权落点
            if hasattr(self, "geo_bias_mlp") and self.geo_bias_mlp is not None:
                geo_small = self.geo_bias_mlp(token_xyz_)               # [1,N,16]
                geo_bias = self.geo_bias_to_feat(geo_small)             # [1,N,C]
                s = torch.clamp(self.geo_bias_scale, 0.0, 1.0)
                token_feat_ = token_feat_ + s * geo_bias

            return token_xyz_, token_feat_, H_img, W_img

        # ======== Branch 2: Slot（单路，可微 clustering） ========
        if self.canonical_prior == "slot":
            token_xyz, token_feat_eff, _, _ = _build_slot_tokens_from_points()
            prior = self.slot_prior(token_feat_eff, token_xyz)

            mu = prior['mu'].squeeze(0)              # [M,3]
            surfel_normal = prior['normal'].squeeze(0)  # [M,3]
            surfel_radius = prior['radius'].squeeze(0)  # [M,1]
            g = self.slot_to_feat(prior['slot_feat'].squeeze(0))  # [M,C]

            surfel_rot = self._build_rotation_from_normal(surfel_normal)
            gaussian_params = self.gaussian_head(
                g,
                surfel_scale=surfel_radius.expand(-1, 3),
                surfel_rot=surfel_rot,
            )
            scale = gaussian_params['scale']
            opacity = gaussian_params['opacity']
            color = gaussian_params['color']

            z_g = self.feat_to_motion(g)
            xyz_t, scale_t, color_t, alpha_t, dxyz_t = self.motion_head(
                z_g, T=T, t_ids=time_ids,
                xyz=mu, scale=scale, color=color, alpha=opacity,
                disable_color_delta=True,
                motion_cond=None,
            )

            world_aabb = self._world_cache.get('aabb', None)
            if world_aabb is None:
                try:
                    if points_3d is not None:
                        world_aabb = self.estimate_points_aabb(points_3d, margin=self.aabb_margin)
                    else:
                        # 用下采样后的 token_xyz 估计，避免直接展开稠密 points
                        world_aabb = self.estimate_points_aabb(token_xyz.squeeze(0), margin=self.aabb_margin)
                except Exception:
                    world_aabb = torch.zeros(2, 3, device=device, dtype=dtype)

            return {
                'mu_t': xyz_t,
                'scale_t': scale_t,
                'rot_t': surfel_rot.unsqueeze(0).expand(T, -1, -1, -1),
                'color_t': color_t,
                'alpha_t': alpha_t,
                'dxyz_t': dxyz_t,
                'world_aabb': world_aabb,
                'surfel_mu': mu,
                'surfel_normal': surfel_normal,
                'surfel_radius': surfel_radius,
            }

        # ======== Branch 3: Slot-Dual（coarse anchors + fine points） ========
        if self.canonical_prior == "slot_dual":
            if points_map is None:
                raise ValueError("canonical_prior='slot_dual' 需要 points_full/points[T,V,H,W,3]（用于 coarse anchors + fine points）")
            if points_ds is None:
                raise ValueError("slot_dual: points_ds 未构造成功，请检查 points_full 与 feat_2d 的 T/V 对齐")

            # ==========================
            # Stage A：coarse anchors（结构/运动骨架）
            # ==========================
            # 在 feat_2d 的 (H',W') 上进一步降采样为 (Hc,Wc)，用于 slot（降低算力、更稳）
            stride = max(1, int(self.coarse_stride))
            Hc = int(max(1, (Hp_ + stride - 1) // stride))
            Wc = int(max(1, (W_ + stride - 1) // stride))

            pts_tvchw = points_ds.permute(0, 1, 4, 2, 3).contiguous().view(T_ * V_, 3, Hp_, W_)
            pts_c = F.interpolate(pts_tvchw, size=(Hc, Wc), mode="area")  # [T*V,3,Hc,Wc]
            points_coarse = pts_c.view(T_, V_, 3, Hc, Wc).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,Hc,Wc,3]
            points_coarse = torch.nan_to_num(points_coarse, nan=0.0, posinf=0.0, neginf=0.0)
            xyz_c = points_coarse.view(1, -1, 3)  # [1,Nc,3]

            feat_tvchw = feat_red.permute(0, 1, 4, 2, 3).contiguous().view(T_ * V_, C_out, Hp_, W_)
            feat_c = F.interpolate(feat_tvchw, size=(Hc, Wc), mode="area")
            feat_coarse = feat_c.view(T_, V_, C_out, Hc, Wc).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,Hc,Wc,C]
            token_feat = feat_coarse.view(1, -1, C_out)  # [1,Nc,C]
            token_feat = self.slot_feat_proj(token_feat)  # [1,Nc,C]

            # ---- p_vis / p_dyn（可微、可监督）----
            def _as_tvhw(x: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
                if x is None:
                    return None
                x = x.to(device=device, dtype=dtype)
                if x.dim() == 5 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                if x.dim() == 3:
                    x = x.unsqueeze(1).expand(T_, V_, -1, -1).contiguous()
                if x.dim() != 4:
                    raise ValueError(f"{name} must be [T,V,H,W] or [T,V,H,W,1], got {tuple(x.shape)}")
                if x.shape[0] != T_ or x.shape[1] != V_:
                    raise ValueError(f"{name} T/V mismatch: expected [{T_},{V_},H,W], got {tuple(x.shape)}")
                return x

            token_p_vis = None
            conf_eff = _as_tvhw(conf, "conf_2d/conf")  # [T,V,H,W]
            if conf_eff is not None:
                conf_x = conf_eff
                conf_prob = torch.sigmoid(conf_x) if (conf_x.min() < 0.0 or conf_x.max() > 1.0) else conf_x.clamp(0.0, 1.0)
                conf_ds = F.interpolate(
                    conf_prob.reshape(T_ * V_, 1, conf_prob.shape[-2], conf_prob.shape[-1]),
                    size=(Hc, Wc),
                    mode="area",
                )
                token_p_vis = conf_ds.view(1, -1, 1).clamp(0.0, 1.0)

            token_p_dyn_sup = None
            dyn_eff = _as_tvhw(dyn_mask, "dyn_mask_2d/dyn_mask")  # [T,V,H,W]
            if dyn_eff is not None:
                dyn_x = dyn_eff
                dyn_prob = torch.sigmoid(dyn_x) if (dyn_x.min() < 0.0 or dyn_x.max() > 1.0) else dyn_x.clamp(0.0, 1.0)
                dyn_ds = F.interpolate(
                    dyn_prob.reshape(T_ * V_, 1, dyn_prob.shape[-2], dyn_prob.shape[-1]),
                    size=(Hc, Wc),
                    mode="area",
                )
                token_p_dyn_sup = dyn_ds.view(1, -1, 1).clamp(0.0, 1.0)

            # temporal stat：从 coarse points 的时间方差得到 1 维标量
            var_tv = points_coarse.var(dim=0, unbiased=False)  # [V,Hc,Wc,3]
            motion_stat = var_tv.mean(dim=-1, keepdim=True).unsqueeze(0).expand(T_, -1, -1, -1, -1).contiguous()
            motion_stat = motion_stat.view(1, -1, 1)  # [1,Nc,1]

            if self.max_token_points > 0 and token_feat.shape[1] > self.max_token_points:
                idx_tok = torch.randperm(token_feat.shape[1], device=device)[: self.max_token_points]
                token_feat = token_feat[:, idx_tok]
                xyz_c = xyz_c[:, idx_tok]
                motion_stat = motion_stat[:, idx_tok]
                if token_p_vis is not None:
                    token_p_vis = token_p_vis[:, idx_tok]
                if token_p_dyn_sup is not None:
                    token_p_dyn_sup = token_p_dyn_sup[:, idx_tok]

            if token_p_dyn_sup is not None:
                token_p_dyn = token_p_dyn_sup
            else:
                dyn_in = torch.cat([token_feat, motion_stat], dim=-1).squeeze(0)  # [Nc,C+1]
                token_p_dyn = torch.sigmoid(self.dyn_pred(dyn_in)).view(1, -1, 1)

            if token_p_vis is None:
                token_p_vis = torch.ones_like(token_p_dyn)

            # ---- Stage A4: dual slot → anchors（只输出 anchors，不决定最终 M_full）----
            dual = self.dual_slot_prior(
                token_feat=token_feat,
                token_xyz=xyz_c,
                token_p_dyn=token_p_dyn,
                token_p_vis=token_p_vis,
                slots_init_static=None,
                slots_init_dynamic=None,
            )
            s = dual["static"]
            d = dual["dynamic"]

            anchors_mu_s0 = torch.nan_to_num(s["mu"].squeeze(0), nan=0.0, posinf=0.0, neginf=0.0)  # [Ms,3]
            anchors_mu_d0 = torch.nan_to_num(d["mu"].squeeze(0), nan=0.0, posinf=0.0, neginf=0.0)  # [Md,3]
            anchors_n_s0 = F.normalize(s["normal"].squeeze(0), dim=-1)  # [Ms,3]
            anchors_r_s0 = torch.nan_to_num(s["radius"].squeeze(0), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)  # [Ms,1]
            anchors_n_d0 = F.normalize(d["normal"].squeeze(0), dim=-1)  # [Md,3]
            anchors_r_d0 = torch.nan_to_num(d["radius"].squeeze(0), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)  # [Md,1]
            anchors_feat_s = self.slot_to_feat(s["slot_feat"]).squeeze(0)  # [Ms,C]
            anchors_feat_d = self.slot_to_feat(d["slot_feat"]).squeeze(0)  # [Md,C]
            Ms = int(anchors_mu_s0.shape[0])
            Md = int(anchors_mu_d0.shape[0])

            # ==========================
            # Stage B：anchor motion（低频，仅作用于 dynamic anchors）
            # ==========================
            if Md > 0:
                z_d = self.feat_to_motion(anchors_feat_d)  # [Md,motion_dim]
                dxyz_anchor_d = self.anchor_motion_head(z_d, t_ids=time_ids, motion_cond=None)  # [T,Md,3]
                anchors_mu_d_t = anchors_mu_d0.unsqueeze(0) + dxyz_anchor_d  # [T,Md,3]
            else:
                dxyz_anchor_d = torch.zeros(T_, 0, 3, device=device, dtype=dtype)
                anchors_mu_d_t = torch.zeros(T_, 0, 3, device=device, dtype=dtype)
            anchors_mu_s_t = anchors_mu_s0.unsqueeze(0).expand(T_, -1, -1)  # [T,Ms,3]

            # ==========================
            # Stage C：fine points（最终高斯点数 M_full 来自 points_full 的可控子采样）
            # ==========================
            points_t0 = points_map[0].to(device=device, dtype=dtype)  # [V,H,W,3]
            V_img, H_img_full, W_img_full = int(points_t0.shape[0]), int(points_t0.shape[1]), int(points_t0.shape[2])
            flat_xyz0 = torch.nan_to_num(points_t0.view(-1, 3), nan=0.0, posinf=0.0, neginf=0.0)  # [V*H*W,3]
            N_total = int(flat_xyz0.shape[0])
            M_full = int(min(max(0, self.fine_num_points), N_total))

            if M_full == 0:
                world_aabb = self._world_cache.get("aabb", None)
                if world_aabb is None:
                    world_aabb = torch.zeros(2, 3, device=device, dtype=dtype)
                    return {
                        "mu_t": torch.zeros(T_, 0, 3, device=device, dtype=dtype),
                        "scale_t": torch.zeros(T_, 0, 3, device=device, dtype=dtype),
                        "rot_t": torch.zeros(T_, 0, 3, 3, device=device, dtype=dtype),
                        "color_t": torch.zeros(T_, 0, 3, device=device, dtype=dtype),
                        "alpha_t": torch.zeros(T_, 0, 1, device=device, dtype=dtype),
                    "anchors_mu_static": anchors_mu_s0,
                    "anchors_mu_dynamic": anchors_mu_d0,
                    "anchors_mu_static_t": anchors_mu_s_t,
                    "anchors_mu_dynamic_t": anchors_mu_d_t,
                    "dxyz_anchor_dynamic": dxyz_anchor_d,
                        "assign_idx_dynamic": torch.zeros(0, 0, device=device, dtype=torch.long),
                        "assign_w_dynamic": torch.zeros(0, 0, device=device, dtype=dtype),
                        "dxyz_t": torch.zeros(T_, 0, 3, device=device, dtype=dtype),
                        "dxyz_t_dynamic": torch.zeros(T_, 0, 3, device=device, dtype=dtype),
                        "world_aabb": world_aabb,
                    }

            conf_flat0 = None
            if conf is not None:
                conf0 = conf.to(device=device, dtype=dtype)
                if conf0.dim() == 5 and conf0.shape[-1] == 1:
                    conf0 = conf0.squeeze(-1)
                if conf0.dim() == 4:
                    conf0 = conf0[0]  # [V,H,W]
                elif conf0.dim() == 3:
                    conf0 = conf0[0].unsqueeze(0).expand(V_img, -1, -1).contiguous()
                else:
                    conf0 = None
                if conf0 is not None:
                    conf_prob0 = torch.sigmoid(conf0) if (conf0.min() < 0.0 or conf0.max() > 1.0) else conf0.clamp(0.0, 1.0)
                    conf_flat0 = conf_prob0.reshape(-1)

            if self.fine_sample_mode == "topk_conf" and conf_flat0 is not None:
                _, fine_idx = torch.topk(conf_flat0, k=M_full, largest=True)
            else:
                if conf_flat0 is not None:
                    p = (conf_flat0 + 1e-8)
                    p = p / p.sum().clamp(min=1e-8)
                    fine_idx = torch.multinomial(p, num_samples=M_full, replacement=False)
                else:
                    fine_idx = torch.randperm(N_total, device=device)[:M_full]

            xyz_f0 = flat_xyz0[fine_idx]  # [M,3]

            # 解析 (v,y,x) 以便从 dyn_mask/conf/coarse 上取值
            HW = int(H_img_full * W_img_full)
            v_idx = (fine_idx // HW).clamp(0, V_img - 1)
            rem = fine_idx - v_idx * HW
            y_idx = (rem // W_img_full).clamp(0, H_img_full - 1)
            x_idx = (rem - y_idx * W_img_full).clamp(0, W_img_full - 1)

            # p_dyn_f：优先 dyn_mask 监督，否则用 coarse token 的 p_dyn 近似
            if dyn_mask is not None:
                dyn0 = dyn_mask.to(device=device, dtype=dtype)
                if dyn0.dim() == 5 and dyn0.shape[-1] == 1:
                    dyn0 = dyn0.squeeze(-1)
                if dyn0.dim() == 4:
                    dyn0 = dyn0[0]  # [V,H,W]
                elif dyn0.dim() == 3:
                    dyn0 = dyn0[0].unsqueeze(0).expand(V_img, -1, -1).contiguous()
                else:
                    dyn0 = None
                if dyn0 is None:
                    p_dyn_f = torch.ones(M_full, 1, device=device, dtype=dtype)
                else:
                    dyn_flat0 = dyn0.reshape(-1)
                    dyn_prob0 = torch.sigmoid(dyn_flat0) if (dyn_flat0.min() < 0.0 or dyn_flat0.max() > 1.0) else dyn_flat0.clamp(0.0, 1.0)
                    p_dyn_f = dyn_prob0[fine_idx].view(M_full, 1)
            else:
                p_dyn_c_map = token_p_dyn.view(T_, V_, Hc, Wc, 1)[0]  # [V,Hc,Wc,1]
                yc = ((y_idx.to(torch.long) * Hc) // max(1, H_img_full)).clamp(0, Hc - 1)
                xc = ((x_idx.to(torch.long) * Wc) // max(1, W_img_full)).clamp(0, Wc - 1)
                p_dyn_f = p_dyn_c_map[v_idx, yc, xc, :].contiguous().view(M_full, 1).clamp(0.0, 1.0)

            # ---- Stage C2/C3：soft assignment + motion 分发（只用 dynamic anchors）----
            if Md > 0:
                idx_d, w_d, _ = self._soft_assign_topk(
                    xyz_f0,
                    anchors_mu_d0,
                    topk=max(1, self.assign_topk),
                    sigma=max(1e-6, self.assign_sigma),
                    chunk=4096,
                )  # [M,K]
                dxyz_sel = dxyz_anchor_d[:, idx_d]  # [T,M,K,3]
                dxyz_f = (w_d.unsqueeze(0).unsqueeze(-1) * dxyz_sel).sum(dim=2)  # [T,M,3]
            else:
                idx_d = torch.zeros(M_full, 0, device=device, dtype=torch.long)
                w_d = torch.zeros(M_full, 0, device=device, dtype=dtype)
                dxyz_f = torch.zeros(T_, M_full, 3, device=device, dtype=dtype)

            dxyz_f = dxyz_f * p_dyn_f.view(1, M_full, 1)
            mu_t = xyz_f0.unsqueeze(0) + dxyz_f  # [T,M,3]

            # ---- 用 all anchors 给 fine points 推断 normal/radius（用于 aggregator 的几何编码）----
            anchors_mu_all = torch.cat([anchors_mu_s0, anchors_mu_d0], dim=0) if (Ms + Md) > 0 else torch.zeros(0, 3, device=device, dtype=dtype)
            anchors_n_all = torch.cat([anchors_n_s0, anchors_n_d0], dim=0) if (Ms + Md) > 0 else torch.zeros(0, 3, device=device, dtype=dtype)
            anchors_r_all = torch.cat([anchors_r_s0, anchors_r_d0], dim=0) if (Ms + Md) > 0 else torch.zeros(0, 1, device=device, dtype=dtype)

            if anchors_mu_all.numel() > 0:
                idx_a, w_a, _ = self._soft_assign_topk(
                    xyz_f0,
                    anchors_mu_all,
                    topk=max(1, self.assign_topk),
                    sigma=max(1e-6, self.assign_sigma),
                    chunk=4096,
                )
                n_sel = anchors_n_all[idx_a]  # [M,K,3]
                r_sel = anchors_r_all[idx_a]  # [M,K,1]
                n_f = (w_a.unsqueeze(-1) * n_sel).sum(dim=1)
                n_f = F.normalize(n_f, dim=-1)
                r_f = (w_a.unsqueeze(-1) * r_sel).sum(dim=1).clamp_min(1e-6)
            else:
                idx_a = torch.zeros(M_full, 0, device=device, dtype=torch.long)
                w_a = torch.zeros(M_full, 0, device=device, dtype=dtype)
                n_f = torch.zeros(M_full, 3, device=device, dtype=dtype)
                n_f[:, 2] = 1.0
                r_f = torch.full((M_full, 1), 1e-2, device=device, dtype=dtype)

            # ==========================
            # Stage D：细节点高斯属性（feat-driven，几何不变）
            # ==========================
            t_ref = 0
            g_f = self.feature_aggregator(
                mu=xyz_f0,
                feat_2d=feat_red[t_ref:t_ref + 1],
                camera_poses=camera_poses[t_ref:t_ref + 1],
                camera_intrinsics=camera_intrinsics[t_ref:t_ref + 1],
                time_ids=time_ids[t_ref:t_ref + 1],
                surfel_normal=n_f,
                surfel_radius=r_f,
            )  # [M,C]

            rot_init = self._build_rotation_from_normal(n_f)  # [M,3,3]
            scale_init = r_f.expand(-1, 3)
            gaussian_params = self.gaussian_head(g_f, surfel_scale=scale_init, surfel_rot=rot_init)
            scale_0 = gaussian_params["scale"]
            rot_0 = gaussian_params["rot"]
            color_0 = gaussian_params["color"]
            alpha_0 = gaussian_params["opacity"]

            scale_t = scale_0.unsqueeze(0).expand(T_, -1, -1)
            rot_t = rot_0.unsqueeze(0).expand(T_, -1, -1, -1)
            color_t = color_0.unsqueeze(0).expand(T_, -1, -1)
            alpha_t = alpha_0.unsqueeze(0).expand(T_, -1, -1)

            world_aabb = self._world_cache.get("aabb", None)
            if world_aabb is None:
                try:
                    world_aabb = self.estimate_points_aabb(xyz_f0, margin=self.aabb_margin)
                except Exception:
                    world_aabb = torch.zeros(2, 3, device=device, dtype=dtype)

                return {
                    "mu_t": mu_t,
                    "scale_t": scale_t,
                    "rot_t": rot_t,
                    "color_t": color_t,
                    "alpha_t": alpha_t,
                # coarse anchors（骨架）
                "anchors_mu_static": anchors_mu_s0,
                "anchors_mu_dynamic": anchors_mu_d0,
                "anchors_mu_static_t": anchors_mu_s_t,
                "anchors_mu_dynamic_t": anchors_mu_d_t,
                "anchors_feat_static": anchors_feat_s,
                "anchors_feat_dynamic": anchors_feat_d,
                "dxyz_anchor_dynamic": dxyz_anchor_d,
                # assignment / motion 分发（细节点）
                "assign_idx_dynamic": idx_d,
                "assign_w_dynamic": w_d,
                "assign_idx_all": idx_a,
                    "assign_w_all": w_a,
                    "p_dyn_f": p_dyn_f,
                    "dxyz_t": dxyz_f,
                    "dxyz_t_dynamic": dxyz_f,
                    # canonical surfel-like（细节点）
                    "surfel_mu": xyz_f0,
                    "surfel_normal": n_f,
                "surfel_radius": r_f,
                "world_aabb": world_aabb,
                # coarse token debug
                "token_p_dyn_coarse": token_p_dyn,
                "token_p_vis_coarse": token_p_vis,
            }

        raise ValueError(f"Unsupported canonical_prior: {self.canonical_prior}")

    @staticmethod
    def _build_rotation_from_normal(normal: torch.Tensor) -> torch.Tensor:
        """
        从法线向量构造旋转矩阵（标准 Gram-Schmidt 版本）
        法线作为 Z 轴，自动构造 X、Y 轴
        
        使用标准 Gram-Schmidt 正交化，避免数值不稳定：
        1. 选择一个任意向量 a
        2. 如果 a 与 n 接近平行，切换到另一个向量
        3. Gram-Schmidt: t = a - (a·n)n，然后归一化
        4. b = n × t（叉积得到第三个正交向量）
        
        Args:
            normal: [M,3] 单位法线向量
            
        Returns:
            rot: [M,3,3] 旋转矩阵 [x_axis, y_axis, z_axis]
        """
        M = normal.shape[0]
        device = normal.device
        dtype = normal.dtype
        
        # 归一化法线（确保是单位向量）
        n = F.normalize(normal, dim=-1)  # [M,3]
        
        # 选择参考向量 a
        # 策略：如果 |n[0]| < 0.9，使用 (1,0,0)；否则使用 (0,1,0)
        a = torch.zeros(M, 3, device=device, dtype=dtype)
        mask = (torch.abs(n[:, 0]) < 0.9)
        a[mask, 0] = 1.0
        a[~mask, 1] = 1.0
        
        # Gram-Schmidt 正交化：t = a - (a·n)n
        dot_an = (a * n).sum(dim=-1, keepdim=True)  # [M,1]
        t = a - dot_an * n  # [M,3]
        
        # 归一化 t（添加小 epsilon 避免除以零）
        t_norm = torch.norm(t, dim=-1, keepdim=True).clamp(min=1e-6)
        t = t / t_norm  # [M,3]
        
        # 计算第三个正交向量：b = n × t
        b = torch.cross(n, t, dim=-1)  # [M,3]
        
        # 拼接成旋转矩阵：[t, b, n]
        # 其中 t 是 X 轴，b 是 Y 轴，n 是 Z 轴
        rot = torch.stack([t, b, n], dim=-1)  # [M,3,3]
        
        return rot


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import os
    import glob
    
    config = load_config('configs/ff4dgsmotion.yaml')
    model_config = config.get('model', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Trellis4DGS4DCanonical(
        # SURFEL 参数
        surfel_k_neighbors=int(model_config.get('surfel_k_neighbors', 16)),
        use_surfel_confidence=bool(model_config.get('use_surfel_confidence', True)),
        target_num_gaussians=int(model_config.get('target_num_gaussians', 5000)),
        
        # Feature Aggregator 参数
        feat_agg_dim=int(model_config.get('feat_agg_dim', 256)),
        feat_agg_layers=int(model_config.get('feat_agg_layers', 2)),
        feat_agg_heads=int(model_config.get('feat_agg_heads', 4)),
        time_emb_dim=int(model_config.get('time_emb_dim', 32)),
        view_emb_dim=int(model_config.get('view_emb_dim', 32)),
        
        # Gaussian Head 参数
        gaussian_head_hidden=int(model_config.get('gaussian_head_hidden', 256)),
        use_scale_refine=bool(model_config.get('use_scale_refine', False)),
        use_rot_refine=bool(model_config.get('use_rot_refine', False)),
        
        # Motion Head 参数
        motion_dim=int(model_config.get('motion_dim', 128)),
        
        # World space 参数
        aabb_margin=float(model_config.get('aabb_margin', 0.05)),
    ).to(device)
    
    # Load latest debug dump from step2_inference_AnchorWarp4DGS.py
    dbg_dir = '/home/star/zzb/TRELLIS/debug'
    pattern = os.path.join(dbg_dir, 'infer_batch_*.pth')
    files = glob.glob(pattern)

    latest = max(files, key=os.path.getmtime)
    print(f"[debug-load] loading {latest}")
    pack = torch.load(latest, map_location=device)

    # Expect keys: points (preferred) or points_3d, plus core camera/feature fields.
    for k in ['feat_2d', 'camera_poses', 'camera_intrinsics', 'time_ids']:
        if k not in pack:
            raise KeyError(f"Missing key '{k}' in {latest}")

    points = pack.get('points', None)
    points_3d = pack.get('points_3d', None)
    if points is not None:
        points = points.to(device)
    if points_3d is not None:
        points_3d = points_3d.to(device)
    if points is None and points_3d is None:
        raise KeyError(f"Missing 'points' (dense map) and 'points_3d' in {latest}")

    feat_2d = pack['feat_2d'].to(device)
    conf = pack.get('conf', None)
    if conf is not None:
        conf = conf.to(device)
    camera_poses = pack['camera_poses'].to(device)
    camera_intrinsics = pack['camera_intrinsics'].to(device)
    time_ids = pack['time_ids'].to(device)
    build_canonical = bool(pack.get('build_canonical', False))

    with torch.no_grad():
        out = model(
            points_full=points,    # [T,V,H,W,3] (preferred for slot_dual tokenization)
            # points_3d=points_3d,   # [T,N,3] (optional; AABB/debug)
            feat_2d=feat_2d,       # [T,V,H'=,W'=,C]
            conf_2d=conf,
            camera_poses=camera_poses,
            camera_K=camera_intrinsics,
            time_ids=time_ids,
            build_canonical=build_canonical,
        )
    print("[debug-load] forward finished. Outputs:", [k for k in out.keys()])
