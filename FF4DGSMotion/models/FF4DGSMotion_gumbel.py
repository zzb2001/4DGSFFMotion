from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_tvhw(x: Optional[torch.Tensor], T: int, V: int, name: str) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.dim() == 3:
        x = x.unsqueeze(1).expand(T, V, -1, -1).contiguous()
    if x.dim() != 4:
        raise ValueError(f"{name} must be [T,V,H,W] or [T,V,H,W,1], got {tuple(x.shape)}")
    if x.shape[0] != T or x.shape[1] != V:
        raise ValueError(f"{name} T/V mismatch: expected [{T},{V},H,W], got {tuple(x.shape)}")
    return x


def _soft_assign_topk(
    x: torch.Tensor,        # [M,3]
    anchors: torch.Tensor,  # [A,3]
    topk: int,
    sigma: float,
    chunk: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2 or x.shape[-1] != 3:
        raise ValueError(f"x must be [M,3], got {tuple(x.shape)}")
    if anchors.dim() != 2 or anchors.shape[-1] != 3:
        raise ValueError(f"anchors must be [A,3], got {tuple(anchors.shape)}")

    M = int(x.shape[0])
    A = int(anchors.shape[0])
    if A == 0:
        return (
            torch.zeros(M, 0, device=x.device, dtype=torch.long),
            torch.zeros(M, 0, device=x.device, dtype=x.dtype),
        )
    K = int(min(max(1, topk), A))
    if M == 0:
        return (
            torch.zeros(0, K, device=x.device, dtype=torch.long),
            torch.zeros(0, K, device=x.device, dtype=x.dtype),
        )

    idx_out = torch.empty(M, K, device=x.device, dtype=torch.long)
    w_out = torch.empty(M, K, device=x.device, dtype=x.dtype)
    sigma2 = float(max(1e-8, sigma * sigma))
    anchors32 = anchors.to(dtype=torch.float32)

    for i0 in range(0, M, int(max(1, chunk))):
        x_chunk = x[i0:i0 + chunk]
        d2 = torch.cdist(x_chunk.to(dtype=torch.float32), anchors32, p=2.0).pow(2)
        d2k, idx = torch.topk(d2, k=K, dim=-1, largest=False)
        w = torch.softmax(-d2k / (2.0 * sigma2), dim=-1)
        m = int(x_chunk.shape[0])
        idx_out[i0:i0 + m] = idx
        w_out[i0:i0 + m] = w.to(dtype=x.dtype)

    return idx_out, w_out


def _thin_by_voxel_keep_max(
    xyz: torch.Tensor,      # [N,3]
    score: torch.Tensor,    # [N]
    voxel_size: float,
    eps_tie: float = 1e-6,
) -> torch.Tensor:
    """
    Density-aware thinning without reconstruction:
    - compute voxel index = floor(xyz / voxel_size)
    - hash voxel index to 1D
    - keep the point with max(score) per voxel (tie-broken by tiny noise)

    returns:
        keep_idx: [M] indices into xyz (M <= N)
    """
    if xyz.numel() == 0:
        return torch.zeros(0, device=xyz.device, dtype=torch.long)
    if xyz.dim() != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
    if score.dim() != 1 or score.shape[0] != xyz.shape[0]:
        raise ValueError(f"score must be [N], got {tuple(score.shape)} vs N={xyz.shape[0]}")

    N = int(xyz.shape[0])
    device = xyz.device

    vs = float(max(1e-9, voxel_size))
    # anchor to min to avoid huge indices
    xyz_min = xyz.amin(dim=0, keepdim=True)
    vidx = torch.floor((xyz - xyz_min) / vs).to(torch.int64)  # [N,3]

    # simple 3D hash to 1D
    hx = vidx[:, 0] * 73856093
    hy = vidx[:, 1] * 19349663
    hz = vidx[:, 2] * 83492791
    h = (hx ^ hy ^ hz)  # [N]

    uniq_h, inv = torch.unique(h, return_inverse=True)
    num_vox = int(uniq_h.shape[0])

    # tie-break with tiny noise
    score2 = score + eps_tie * torch.rand_like(score)

    max_per_vox = torch.full((num_vox,), -1e30, device=device, dtype=score2.dtype)
    max_per_vox.scatter_reduce_(0, inv, score2, reduce="amax", include_self=True)

    keep_mask = score2 >= (max_per_vox[inv] - 1e-12)
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1)

    if keep_idx.numel() > num_vox:
        h_keep = h[keep_idx]
        order = torch.argsort(h_keep)
        keep_idx = keep_idx[order]
        h_keep = h_keep[order]
        new_vox = torch.ones_like(h_keep, dtype=torch.bool)
        new_vox[1:] = h_keep[1:] != h_keep[:-1]
        keep_idx = keep_idx[new_vox]

    return keep_idx


def _sample_gumbel(shape, device, dtype, eps=1e-8):
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u.clamp(min=eps, max=1.0 - eps)))


def _gumbel_topk_straight_through_sparse(logits: torch.Tensor, k: int, tau: float):
    """
    Sparse ST Gumbel-TopK:
    - forward: hard top-k indices
    - backward: gradients only through selected logits

    Returns:
        idx: [k] indices of selected items (hard)
        y_st: [k] straight-through values corresponding to y[idx]
    """
    N = logits.shape[0]
    k = int(min(max(1, k), int(N)))
    device, dtype = logits.device, logits.dtype

    # Gumbel noise
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)))
    y = (logits + g) / float(max(1e-6, tau))

    # hard selection
    idx = torch.topk(y, k=k, largest=True).indices  # [k]

    # straight-through trick (sparse)
    y_hard = y[idx]
    y_soft = y[idx]
    y_st = y_hard.detach() + (y_soft - y_soft.detach())
    return idx, y_st


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8, hidden_dim: int = 256):
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

    def forward(
        self,
        inputs: torch.Tensor,                    # [B,N,D]
        token_weight: Optional[torch.Tensor] = None,  # [B,N] or [B,N,1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() != 3:
            raise ValueError(f"inputs must be [B,N,D], got {tuple(inputs.shape)}")
        B, N, D = inputs.shape
        if D != self.dim:
            raise ValueError(f"inputs D={D} but slot dim={self.dim}")

        tw = None
        if token_weight is not None:
            if token_weight.dim() == 3 and token_weight.shape[-1] == 1:
                tw = token_weight.squeeze(-1)
            elif token_weight.dim() == 2:
                tw = token_weight
            else:
                raise ValueError(f"token_weight must be [B,N] or [B,N,1], got {tuple(token_weight.shape)}")
            tw = tw.to(device=inputs.device, dtype=inputs.dtype).clamp(min=0.0)

        x = self.norm_inputs(inputs)
        k = self.project_k(x)
        v = self.project_v(x)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = torch.exp(self.slots_logsigma).expand(B, self.num_slots, -1).clamp(min=1e-6)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            q = self.project_q(slots_n)  # [B,M,D]

            attn_logits = torch.einsum("bnd,bmd->bnm", k, q) * (D ** -0.5)  # [B,N,M]
            attn = torch.softmax(attn_logits, dim=-1)  # [B,N,M]

            if tw is not None:
                attn = attn * tw.unsqueeze(-1)

            attn = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
            updates = torch.einsum("bnm,bnd->bmd", attn, v)

            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D),
            ).view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class SlotAttentionGaussianPrior(nn.Module):
    def __init__(self, dim: int, num_slots: int, iters: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.token_proj = nn.Linear(self.dim, self.dim)
        self.slot_attn = SlotAttention(num_slots=num_slots, dim=self.dim, iters=iters, hidden_dim=hidden_dim)

    def forward(
        self,
        token_feat: torch.Tensor,   # [B,N,C]
        token_xyz: torch.Tensor,    # [B,N,3]
        token_weight: Optional[torch.Tensor] = None,  # [B,N] or [B,N,1]
    ) -> Dict[str, torch.Tensor]:
        if token_feat.dim() != 3 or token_xyz.dim() != 3:
            raise ValueError(f"token_feat/token_xyz must be [B,N,*], got {tuple(token_feat.shape)}, {tuple(token_xyz.shape)}")
        if token_xyz.shape[-1] != 3:
            raise ValueError(f"token_xyz last dim must be 3, got {tuple(token_xyz.shape)}")
        if token_feat.shape[:2] != token_xyz.shape[:2]:
            raise ValueError(f"B,N mismatch: {tuple(token_feat.shape)} vs {tuple(token_xyz.shape)}")

        x = self.token_proj(token_feat)
        slots, attn = self.slot_attn(x, token_weight=token_weight)  # [B,M,C], [B,N,M]
        w = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        mu = torch.einsum("bnm,bnd->bmd", w, token_xyz)  # [B,M,3]
        return {"mu": mu, "slot_feat": slots, "attn": attn}


class DualSlotAttentionGaussianPrior(nn.Module):
    def __init__(
        self,
        dim: int,
        num_static_slots: int,
        num_dynamic_slots: int,
        iters: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hd = int(hidden_dim) if hidden_dim is not None else int(dim)
        self.static_prior = SlotAttentionGaussianPrior(dim=dim, num_slots=num_static_slots, iters=iters, hidden_dim=hd)
        self.dynamic_prior = SlotAttentionGaussianPrior(dim=dim, num_slots=num_dynamic_slots, iters=iters, hidden_dim=hd)

    def forward(
        self,
        token_feat: torch.Tensor,         # [B,N,C]
        token_xyz: torch.Tensor,          # [B,N,3]
        token_p_dyn: torch.Tensor,        # [B,N,1] or [B,N]
        token_p_vis: Optional[torch.Tensor] = None,  # [B,N,1] or [B,N]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if token_feat.shape[:2] != token_xyz.shape[:2]:
            raise ValueError("token_feat/token_xyz B,N mismatch")
        B, N = token_feat.shape[0], token_feat.shape[1]

        def _bn1(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.dim() == 2:
                y = x.unsqueeze(-1)
            elif x.dim() == 3 and x.shape[-1] == 1:
                y = x
            else:
                raise ValueError(f"{name} must be [B,N] or [B,N,1], got {tuple(x.shape)}")
            if y.shape[0] != B or y.shape[1] != N:
                raise ValueError(f"{name} shape mismatch: expected [{B},{N},*], got {tuple(y.shape)}")
            return y

        p_dyn = _bn1(token_p_dyn, "token_p_dyn").clamp(0.0, 1.0)
        if token_p_vis is None:
            p_vis = torch.ones_like(p_dyn)
        else:
            p_vis = _bn1(token_p_vis, "token_p_vis").clamp(0.0, 1.0)

        w_dyn = (p_dyn * p_vis).squeeze(-1)           # [B,N]
        w_sta = ((1.0 - p_dyn) * p_vis).squeeze(-1)   # [B,N]

        out_s = self.static_prior(token_feat, token_xyz, token_weight=w_sta)
        out_d = self.dynamic_prior(token_feat, token_xyz, token_weight=w_dyn)
        return {"static": out_s, "dynamic": out_d}


class AnchorDeltaHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = 3,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.time_emb_dim = int(time_emb_dim)

        self.z_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
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

    def forward(self, z: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be [M,D], got {tuple(z.shape)}")
        if t_ids.dim() != 1:
            raise ValueError(f"t_ids must be [T], got {tuple(t_ids.shape)}")

        M = int(z.shape[0])
        T = int(t_ids.shape[0])
        t_min = t_ids.min()
        t_max = torch.clamp(t_ids.max(), min=t_min + 1)
        t_norm = (t_ids.float() - t_min.float()) / (t_max.float() - t_min.float())
        t_emb = self._posenc_t(t_norm, self.time_emb_dim)  # [T,Dt]

        gate = self.time_mlp(t_emb)  # [T,H]
        z_h = self.z_proj(z)         # [M,H]
        fused = gate.unsqueeze(1) * z_h.unsqueeze(0)  # [T,M,H]
        out = self.out_mlp(fused.reshape(T * M, -1)).view(T, M, self.out_dim)
        return out


class MultiViewPointAggregator(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_layers: int,
        num_heads: int,
        time_emb_dim: int,
        view_emb_dim: int,
        topk_views: int = 4,
        hidden_dim: int = 256,
        max_views: int = 64,
        chunk_points: int = 20000,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.view_emb_dim = int(view_emb_dim)
        self.topk_views = int(topk_views)
        self.chunk_points = int(chunk_points)

        self.view_emb = nn.Embedding(int(max_views), self.view_emb_dim)
        self.pos_proj = nn.Linear(self.time_emb_dim + self.view_emb_dim, 32)
        self.feat_proj = nn.Linear(self.feat_dim + 32, int(hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.out_proj = nn.Linear(int(hidden_dim), self.feat_dim)

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

    @staticmethod
    def _project_points(Xw: torch.Tensor, c2w: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Xw32 = Xw.to(torch.float32)
        c2w32 = c2w.to(torch.float32)
        K32 = K.to(torch.float32)
        w2c32 = torch.inverse(c2w32)
        M = Xw32.shape[0]
        Xw_h = torch.cat([Xw32, torch.ones(M, 1, device=Xw32.device, dtype=torch.float32)], dim=1)
        Xc = (w2c32 @ Xw_h.t()).t()[:, :3]
        z = Xc[:, 2]
        uvw = (K32 @ Xc.t()).t()
        u = uvw[:, 0] / uvw[:, 2].clamp(min=1e-6)
        v = uvw[:, 1] / uvw[:, 2].clamp(min=1e-6)
        return u.to(Xw.dtype), v.to(Xw.dtype), z.to(Xw.dtype)

    @staticmethod
    def _bilinear_sample(feat: torch.Tensor, u_pix: torch.Tensor, v_pix: torch.Tensor, H_img: float, W_img: float) -> torch.Tensor:
        Hp, Wp, C = feat.shape
        u_feat = u_pix * (Wp / max(1.0, W_img))
        v_feat = v_pix * (Hp / max(1.0, H_img))
        x = 2.0 * (u_feat / max(1.0, Wp - 1)) - 1.0
        y = 2.0 * (v_feat / max(1.0, Hp - 1)) - 1.0
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2).to(dtype=feat.dtype)
        feat_chw = feat.permute(2, 0, 1).unsqueeze(0)  # [1,C,Hp,Wp]
        sampled = F.grid_sample(feat_chw, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [M,C]

    def forward(
        self,
        xyz: torch.Tensor,                 # [M,3]
        feat_2d: torch.Tensor,             # [T,V,H',W',C]
        camera_poses: torch.Tensor,        # [T,V,4,4]
        camera_intrinsics: torch.Tensor,   # [T,V,3,3]
        time_ids: torch.Tensor,            # [T]
    ) -> torch.Tensor:
        # 注意：当 M 很大（例如 100k）时，直接把 batch=M 喂给 Transformer/MHA
        # 在某些 CUDA 内核实现（尤其是 SDPA/FlashAttention 路径）会触发 invalid configuration。
        # 这里按点分块，保证每块的 batch 不会过大。
        device = feat_2d.device
        dtype = feat_2d.dtype
        T, V, Hp, Wp, C = feat_2d.shape
        M = int(xyz.shape[0])
        if M == 0:
            return torch.zeros(0, self.feat_dim, device=device, dtype=dtype)

        if self.chunk_points > 0 and M > self.chunk_points:
            outs = []
            for i0 in range(0, M, self.chunk_points):
                outs.append(
                    self.forward(
                        xyz[i0:i0 + self.chunk_points],
                        feat_2d,
                        camera_poses,
                        camera_intrinsics,
                        time_ids,
                    )
                )
            return torch.cat(outs, dim=0)

        cx = camera_intrinsics[0, 0, 0, 2]
        cy = camera_intrinsics[0, 0, 1, 2]
        W_img = 2.0 * cx
        H_img = 2.0 * cy

        t_min = time_ids.min()
        t_max = torch.clamp(time_ids.max(), min=t_min + 1)
        t_norm = (time_ids.float() - t_min.float()) / (t_max.float() - t_min.float())
        t_emb_all = self._posenc_t(t_norm, self.time_emb_dim)  # [T,Dt]

        # view score: prefer visible and closer
        view_scores = []
        for t in range(T):
            for v in range(V):
                u, vv, z = self._project_points(xyz, camera_poses[t, v], camera_intrinsics[t, v])
                in_img = (u >= 0) & (u < W_img) & (vv >= 0) & (vv < H_img)
                visible = (z > 1e-4) & in_img
                score = (1.0 / (z.clamp(min=0.1) + 1e-6)) * visible.to(dtype)
                view_scores.append(score)
        view_scores = torch.stack(view_scores, dim=0)  # [T*V, M]
        scores_m = view_scores.t()  # [M, T*V]

        Ksel = int(min(max(1, self.topk_views), T * V))
        _, topk_idx = torch.topk(scores_m, k=Ksel, dim=1, largest=True)  # [M,K]

        hidden = int(self.feat_proj.out_features)
        seq = torch.zeros(M, Ksel, hidden, device=device, dtype=dtype)

        tv = 0
        for t in range(T):
            for v in range(V):
                match = (topk_idx == tv).nonzero(as_tuple=False)
                if match.numel() == 0:
                    tv += 1
                    continue
                m_idx = match[:, 0]
                k_idx = match[:, 1]
                u, vv, z = self._project_points(xyz[m_idx], camera_poses[t, v], camera_intrinsics[t, v])
                sampled = self._bilinear_sample(feat_2d[t, v], u, vv, float(H_img), float(W_img))  # [Q,C]
                sampled = sampled * (z > 1e-4).to(dtype).unsqueeze(-1)

                v_safe = min(v, self.view_emb.num_embeddings - 1)
                v_emb = self.view_emb(torch.tensor(v_safe, device=device)).expand(m_idx.shape[0], -1)
                t_emb = t_emb_all[t].expand(m_idx.shape[0], -1)
                pos32 = self.pos_proj(torch.cat([t_emb, v_emb], dim=-1))

                h = self.feat_proj(torch.cat([sampled, pos32], dim=-1))
                seq[m_idx, k_idx, :] = h
                tv += 1

        seq = self.transformer(seq)  # [M,K,H]
        g = seq.mean(dim=1)
        return self.out_proj(g)


class GaussianHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, use_scale_refine: bool = False, use_rot_refine: bool = False, opacity_init_bias: float = -2.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_scale_refine = bool(use_scale_refine)
        self.use_rot_refine = bool(use_rot_refine)

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.fc_opac = nn.Linear(self.hidden_dim, 1)
        self.fc_color = nn.Linear(self.hidden_dim, 3)
        self.fc_scale = nn.Linear(self.hidden_dim, 3)
        self.fc_rot6 = nn.Linear(self.hidden_dim, 6)

        nn.init.constant_(self.fc_opac.bias, float(opacity_init_bias))

    @staticmethod
    def _rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
        a1 = x[..., 0:3]
        a2 = x[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    def forward(self, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.mlp(g)
        opacity = torch.sigmoid(self.fc_opac(h))
        color = torch.sigmoid(self.fc_color(h))
        scale = F.softplus(self.fc_scale(h)) + 1e-4
        rot = self._rot6d_to_matrix(self.fc_rot6(h))
        return {"opacity": opacity, "color": color, "scale": scale, "rot": rot}


@dataclass(frozen=True)
class ModelCfg:
    in_feat_dim: int = 768
    feat_agg_dim: int = 256
    feat_agg_layers: int = 1
    feat_agg_heads: int = 4
    time_emb_dim: int = 32
    view_emb_dim: int = 32
    gaussian_head_hidden: int = 256
    use_scale_refine: bool = False
    use_rot_refine: bool = False
    motion_dim: int = 128

    num_anchors_static: int = 4096
    num_anchors_dynamic: int = 4096
    slot_iters: int = 3

    coarse_stride: int = 8
    fine_num_points: int = 100000
    fine_sample_mode: str = "topk_conf"  # ["topk_conf","random"]
    assign_topk: int = 8
    assign_sigma: float = 0.05
    # ===== Differentiable importance selection (Stage C) =====
    imp_candidate_points: int = 200000   # 候选集大小（从 TVHW 中抽取多少候选点再做可微选择）
    imp_gumbel_tau: float = 1.0          # Gumbel-Softmax 温度（训练可退火到 0.2）
    imp_mlp_hidden: int = 256            # importance head hidden
    imp_use_xyz: bool = True             # 是否把 xyz 输入 importance head
    imp_use_conf: bool = True            # 是否把 conf 输入 importance head
    imp_use_dyn: bool = True             # 是否把 dyn 输入 importance head（若有）
    imp_mix_uniform: float = 0.2         # proposal 分布：uniform 权重
    imp_mix_conf: float = 0.6            # proposal 分布：conf 权重
    imp_mix_dyn: float = 0.2             # proposal 分布：dyn 权重（若有）
    imp_tau_min: float = 0.2             # 可选：训练后期最小 tau
    # Proposal / thinning (two-stage proposal)
    imp_pre_sample_factor: float = 3.0    # N_pre = factor * Ncand
    imp_thin_keep_factor: float = 2.0     # N_thin target = keep_factor * Ncand
    imp_voxel_ratio: float = 1.0          # voxel_size multiplier
    imp_min_voxel_size: float = 1e-4      # minimum voxel size to avoid zero
    # Safety cap for slot-attention tokens (coarse branch). Prevents OOM in [B,N,M] attention.
    max_coarse_tokens: int = 8000

    use_dyn_pred_token: bool = True
    use_residual_motion: bool = False

    top_k_views: int = 4
    point_agg_chunk: int = 20000
    max_scale: float = 0.05
    min_scale: float = 1e-4

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelCfg":
        kwargs = dict(d or {})
        if "cond_channels" in kwargs and "in_feat_dim" not in kwargs:
            kwargs["in_feat_dim"] = int(kwargs["cond_channels"])
        if "feat2d_in_dim" in kwargs and "in_feat_dim" not in kwargs:
            kwargs["in_feat_dim"] = int(kwargs["feat2d_in_dim"])
        return ModelCfg(**{k: v for k, v in kwargs.items() if k in ModelCfg.__annotations__})


class Trellis4DGS4DCanonical(nn.Module):
    """
    统一入口（供 step2_train / step2_inference 调用）：
    - 结构（模块）全部定义在本文件
    - 超参全部从 cfg（通常来自 configs/ff4dgsmotion.yaml 的 model 字段）读取
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = ModelCfg.from_dict(cfg or {})
        self._world_aabb: Optional[torch.Tensor] = None

        # Step 1: feat_reduce（2D → token 特征）
        self.feat_reduce = nn.Sequential(
            nn.Conv2d(int(self.cfg.in_feat_dim), int(self.cfg.feat_agg_dim), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.cfg.feat_agg_dim), int(self.cfg.feat_agg_dim), 1),
        )

        # Step 2: Dual Slot Prior（canonical anchors）
        self.dual_slot_prior = DualSlotAttentionGaussianPrior(
            dim=int(self.cfg.feat_agg_dim),
            num_static_slots=int(self.cfg.num_anchors_static),
            num_dynamic_slots=int(self.cfg.num_anchors_dynamic),
            iters=int(self.cfg.slot_iters),
            hidden_dim=int(self.cfg.feat_agg_dim),
        )

        # Step 3: Dynamic Anchor Motion Head
        self.dynamic_anchor_motion = AnchorDeltaHead(
            in_dim=int(self.cfg.feat_agg_dim),
            hidden_dim=int(self.cfg.motion_dim),
            out_dim=3,
            time_emb_dim=int(self.cfg.time_emb_dim),
        )

        # Step 4: Multi-view Point Aggregator（fine 点）
        self.point_aggregator = MultiViewPointAggregator(
            feat_dim=int(self.cfg.feat_agg_dim),
            num_layers=int(self.cfg.feat_agg_layers),
            num_heads=int(self.cfg.feat_agg_heads),
            time_emb_dim=int(self.cfg.time_emb_dim),
            view_emb_dim=int(self.cfg.view_emb_dim),
            topk_views=int(self.cfg.top_k_views),
            hidden_dim=int(self.cfg.feat_agg_dim),
            chunk_points=int(getattr(self.cfg, "point_agg_chunk", 20000)),
        )

        # Step 5: Gaussian Head（属性解码）
        self.gaussian_head = GaussianHead(
            in_dim=int(self.cfg.feat_agg_dim),
            hidden_dim=int(self.cfg.gaussian_head_hidden),
            use_scale_refine=bool(self.cfg.use_scale_refine),
            use_rot_refine=bool(self.cfg.use_rot_refine),
        )

        # Step 6（可选）：dyn_pred_token（学习动静）
        if bool(self.cfg.use_dyn_pred_token):
            self.dyn_pred_token = nn.Sequential(
                nn.Linear(int(self.cfg.feat_agg_dim), int(self.cfg.feat_agg_dim) // 2),
                nn.ReLU(inplace=True),
                nn.Linear(int(self.cfg.feat_agg_dim) // 2, 1),
            )
        else:
            self.dyn_pred_token = None

        # Step 7（可选）：residual motion（细节点残差）
        if bool(self.cfg.use_residual_motion):
            self.residual_motion_head = AnchorDeltaHead(
                in_dim=int(self.cfg.feat_agg_dim),
                hidden_dim=max(1, int(self.cfg.motion_dim) // 2),
                out_dim=3,
                time_emb_dim=int(self.cfg.time_emb_dim),
            )
        else:
            self.residual_motion_head = None

        # ===== Stage C: differentiable importance field (learnable point selector) =====
        imp_in = int(self.cfg.feat_agg_dim)
        if bool(self.cfg.imp_use_xyz):
            imp_in += 3
        if bool(self.cfg.imp_use_conf):
            imp_in += 1
        if bool(self.cfg.imp_use_dyn):
            imp_in += 1

        # time/view embedding（轻量）
        self.imp_time_emb = nn.Embedding(512, 16)
        self.imp_view_emb = nn.Embedding(64, 16)

        self.imp_head = nn.Sequential(
            nn.Linear(imp_in + 32, int(self.cfg.imp_mlp_hidden)),
            nn.SiLU(),
            nn.Linear(int(self.cfg.imp_mlp_hidden), int(self.cfg.imp_mlp_hidden)),
            nn.SiLU(),
            nn.Linear(int(self.cfg.imp_mlp_hidden), 1),
        )

    def reset_world_cache(self):
        self._world_aabb = None

    def set_world_aabb(self, aabb: torch.Tensor):
        if not torch.is_tensor(aabb):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        self._world_aabb = aabb.to(device=dev, dtype=dt)

    @staticmethod
    def estimate_points_aabb(points_3d: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
        if points_3d is None or points_3d.numel() == 0:
            raise ValueError("points_3d is empty")
        if points_3d.dim() == 3:
            pts = points_3d.reshape(-1, 3)
        else:
            pts = points_3d.view(-1, 3)
        mask = torch.isfinite(pts).all(dim=-1)
        pts = pts[mask]
        if pts.numel() == 0:
            raise ValueError("points_3d has no finite entries")
        device, dtype = pts.device, pts.dtype
        minb = torch.quantile(pts, 0.01, dim=0)
        maxb = torch.quantile(pts, 0.99, dim=0)
        extent = (maxb - minb).clamp(min=1e-6)
        center = (minb + maxb) * 0.5
        minb = center - extent * (0.5 + float(margin))
        maxb = center + extent * (0.5 + float(margin))
        return torch.stack([minb, maxb], dim=0).to(device=device, dtype=dtype)

    def forward(
        self,
        points_full: Optional[torch.Tensor] = None,   # [T,V,H,W,3]
        feat_2d: Optional[torch.Tensor] = None,       # [T,V,H',W',Cin]
        camera_poses: Optional[torch.Tensor] = None,  # [T,V,4,4] (c2w)
        camera_K: Optional[torch.Tensor] = None,      # [T,V,3,3] (alias)
        camera_intrinsics: Optional[torch.Tensor] = None,  # [T,V,3,3]
        time_ids: Optional[torch.Tensor] = None,      # [T]
        dyn_mask_2d: Optional[torch.Tensor] = None,   # [T,V,H,W] or [T,V,H,W,1]
        conf_2d: Optional[torch.Tensor] = None,       # [T,V,H,W] or [T,V,H,W,1]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if points_full is None:
            points_full = kwargs.get("points_full", None)
        if points_full is None:
            points_full = kwargs.get("points", None)
        if points_full is None:
            raise ValueError("points_full is required (expected [T,V,H,W,3])")
        if feat_2d is None:
            feat_2d = kwargs.get("feat_2d", None)
        if feat_2d is None:
            raise ValueError("feat_2d is required (expected [T,V,H',W',Cin])")
        if camera_poses is None:
            camera_poses = kwargs.get("camera_poses", None)
        if camera_poses is None:
            raise ValueError("camera_poses is required (expected [T,V,4,4])")
        if camera_intrinsics is None:
            camera_intrinsics = camera_K
        if camera_intrinsics is None:
            camera_intrinsics = kwargs.get("camera_intrinsics", None) or kwargs.get("camera_K", None)
        if camera_intrinsics is None:
            raise ValueError("camera_intrinsics/camera_K is required (expected [T,V,3,3])")
        if time_ids is None:
            time_ids = kwargs.get("time_ids", None)
        if time_ids is None:
            raise ValueError("time_ids is required (expected [T])")

        device = feat_2d.device
        dtype = feat_2d.dtype
        T, V, H, W, _ = points_full.shape
        _, _, Hp, Wp, Cin = feat_2d.shape

        # Step 1: feat_reduce（2D → token 特征）
        feat_nchw = feat_2d.permute(0, 1, 4, 2, 3).contiguous().view(T * V, Cin, Hp, Wp)
        feat_red = self.feat_reduce(feat_nchw)  # [T*V,C,H',W']
        C = int(feat_red.shape[1])
        feat_red = feat_red.view(T, V, C, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,H',W',C]

        # Stage A1: points_coarse（只为 anchors/motion，降低算力）
        stride = max(1, int(self.cfg.coarse_stride))
        Hc = int(max(1, (H + stride - 1) // stride))
        Wc = int(max(1, (W + stride - 1) // stride))

        pts_tvchw = points_full.permute(0, 1, 4, 2, 3).contiguous().view(T * V, 3, H, W)
        pts_c = F.interpolate(pts_tvchw, size=(Hc, Wc), mode="area")  # [T*V,3,Hc,Wc]
        points_coarse = pts_c.view(T, V, 3, Hc, Wc).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,Hc,Wc,3]
        points_coarse = torch.nan_to_num(points_coarse, nan=0.0, posinf=0.0, neginf=0.0)

        feat_tvchw = feat_red.permute(0, 1, 4, 2, 3).contiguous().view(T * V, C, Hp, Wp)
        feat_c = F.interpolate(feat_tvchw, size=(Hc, Wc), mode="bilinear", align_corners=False)
        feat_coarse = feat_c.view(T, V, C, Hc, Wc).permute(0, 1, 3, 4, 2).contiguous()  # [T,V,Hc,Wc,C]

        token_xyz = points_coarse.reshape(1, -1, 3)  # [1,Nc,3]
        token_feat = feat_coarse.reshape(1, -1, C)   # [1,Nc,C]
        Nc = int(token_xyz.shape[1])

        # Stage A3: p_vis / p_dyn
        token_p_vis = torch.ones((1, Nc, 1), device=device, dtype=dtype)
        conf_eff = _as_tvhw(conf_2d, T, V, "conf_2d")
        if conf_eff is not None:
            conf_prob = torch.sigmoid(conf_eff) if (conf_eff.min() < 0.0 or conf_eff.max() > 1.0) else conf_eff.clamp(0.0, 1.0)
            conf_ds = F.interpolate(conf_prob.reshape(T * V, 1, H, W), size=(Hc, Wc), mode="area")
            token_p_vis = conf_ds.view(1, -1, 1).clamp(0.0, 1.0)

        dyn_eff = _as_tvhw(dyn_mask_2d, T, V, "dyn_mask_2d")
        if dyn_eff is not None:
            dyn_prob = torch.sigmoid(dyn_eff) if (dyn_eff.min() < 0.0 or dyn_eff.max() > 1.0) else dyn_eff.clamp(0.0, 1.0)
            dyn_ds = F.interpolate(dyn_prob.reshape(T * V, 1, H, W), size=(Hc, Wc), mode="area")
            token_p_dyn = dyn_ds.view(1, -1, 1).clamp(0.0, 1.0)
        elif self.dyn_pred_token is not None:
            token_p_dyn = torch.sigmoid(self.dyn_pred_token(token_feat)).view(1, Nc, 1)
        else:
            token_p_dyn = torch.full((1, Nc, 1), 0.5, device=device, dtype=dtype)

        # --- OOM guard: subsample coarse tokens before slot attention ---
        max_tokens = int(getattr(self.cfg, "max_coarse_tokens", 0))
        if max_tokens > 0 and Nc > max_tokens:
            # Prefer sampling visible tokens; fall back to uniform.
            p = token_p_vis.view(-1)
            if not torch.isfinite(p).all() or float(p.sum().item()) <= 0:
                p = torch.ones_like(p)
            p = (p + 1e-8) / (p.sum().clamp_min(1e-8))
            idx = torch.multinomial(p, num_samples=max_tokens, replacement=False)
            token_xyz = token_xyz[:, idx]
            token_feat = token_feat[:, idx]
            token_p_vis = token_p_vis[:, idx]
            token_p_dyn = token_p_dyn[:, idx]
            Nc = int(max_tokens)

        # Step 2: Dual Slot Prior（canonical anchors）
        anchors = self.dual_slot_prior(
            token_feat=token_feat,
            token_xyz=token_xyz,
            token_p_dyn=token_p_dyn,
            token_p_vis=token_p_vis,
        )
        mu_s0 = anchors["static"]["mu"].squeeze(0)       # [Ms,3]
        mu_d0 = anchors["dynamic"]["mu"].squeeze(0)      # [Md,3]
        feat_d = anchors["dynamic"]["slot_feat"].squeeze(0)  # [Md,C]
        Ms = int(mu_s0.shape[0])
        Md = int(mu_d0.shape[0])

        # Step 3: Dynamic Anchor Motion Head（Δxyz_d(t)）
        if Md > 0:
            dxyz_anchor_d = self.dynamic_anchor_motion(feat_d, time_ids)  # [T,Md,3]
        else:
            dxyz_anchor_d = torch.zeros(T, 0, 3, device=device, dtype=dtype)

        # ============================================================
        # Stage C (NEW): Differentiable importance field over points_full[T,V,H,W,3]
        #   - build candidate set from all TVHW points (proposal)
        #   - learn importance logits per candidate (differentiable)
        #   - ST-Gumbel TopK selects M_full points (hard forward, soft gradients)
        # ============================================================

        # ----- flatten TVHW for candidates -----
        # points_full: [T,V,H,W,3]
        flat_xyz_all = torch.nan_to_num(points_full.reshape(-1, 3), nan=0.0, posinf=0.0, neginf=0.0)
        N_total = int(flat_xyz_all.shape[0])

        M_full = int(min(max(0, int(self.cfg.fine_num_points)), N_total))
        if M_full == 0:
            return {
                "mu_t": torch.zeros(T, 0, 3, device=device, dtype=dtype),
                "scale_t": torch.zeros(T, 0, 3, device=device, dtype=dtype),
                "rot_t": torch.zeros(T, 0, 3, 3, device=device, dtype=dtype),
                "color_t": torch.zeros(T, 0, 3, device=device, dtype=dtype),
                "alpha_t": torch.zeros(T, 0, 1, device=device, dtype=dtype),
                "dxyz_t": torch.zeros(T, 0, 3, device=device, dtype=dtype),
                "anchors_mu_static": mu_s0,
                "anchors_mu_dynamic": mu_d0,
                "dxyz_anchor_dynamic": dxyz_anchor_d,
                "assign_idx_dynamic": torch.zeros(0, 0, device=device, dtype=torch.long),
                "assign_w_dynamic": torch.zeros(0, 0, device=device, dtype=dtype),
            }

        # ----- proposal distribution (non-topk): mixture of uniform + conf + dyn -----
        # conf_eff/dyn_eff are [T,V,H,W] or None
        conf_flat = None
        if conf_eff is not None:
            conf_flat = conf_eff.to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0)

        dyn_flat = None
        if dyn_eff is not None:
            dyn_flat = dyn_eff.to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0)

        w_uni = float(getattr(self.cfg, "imp_mix_uniform", 0.2))
        w_conf = float(getattr(self.cfg, "imp_mix_conf", 0.6))
        w_dyn = float(getattr(self.cfg, "imp_mix_dyn", 0.2))

        p = torch.ones(N_total, device=device, dtype=dtype) * max(0.0, w_uni)
        if conf_flat is not None:
            p = p + max(0.0, w_conf) * (conf_flat + 1e-6)
        if dyn_flat is not None:
            p = p + max(0.0, w_dyn) * (dyn_flat + 1e-6)

        # avoid degenerate
        p = (p + 1e-8)
        p = p / p.sum().clamp(min=1e-8)

        # -------------------------
        # Two-stage proposal:
        #   1) pre-sample from p (bigger pool)
        #   2) density-aware thinning (keep max-score per voxel)
        #   3) multinomial again to final Ncand
        # -------------------------
        Ncand = int(min(max(1024, int(getattr(self.cfg, "imp_candidate_points", 200000))), N_total))

        pre_factor = float(getattr(self.cfg, "imp_pre_sample_factor", 3.0))
        keep_factor = float(getattr(self.cfg, "imp_thin_keep_factor", 2.0))

        N_pre = int(min(N_total, max(Ncand, int(pre_factor * Ncand))))
        # Pre-sample (non-diff proposal, fine)
        pre_idx = torch.multinomial(p, num_samples=N_pre, replacement=False)  # [N_pre]

        xyz_pre = flat_xyz_all[pre_idx]              # [N_pre,3]
        p_pre = p[pre_idx].contiguous()              # [N_pre]

        # Choose voxel_size adaptively from pre-sample bounding box
        # heuristic: voxel_size ~ scene_diag / sqrt(Ncand) (scaled)
        xyz_min = xyz_pre.amin(dim=0)
        xyz_max = xyz_pre.amax(dim=0)
        scene_diag = torch.norm((xyz_max - xyz_min).clamp(min=1e-9)).item()
        voxel_ratio = float(getattr(self.cfg, "imp_voxel_ratio", 1.0))
        min_vs = float(getattr(self.cfg, "imp_min_voxel_size", 1e-4))
        voxel_size = max(min_vs, voxel_ratio * (scene_diag / (max(1.0, float(Ncand)) ** 0.5)))

        # Thinning: keep best per voxel according to p_pre (you can use conf/dyn mixed score too)
        keep_in_pre = _thin_by_voxel_keep_max(xyz_pre, p_pre, voxel_size=voxel_size)  # indices into pre pool

        # Optionally cap thinning pool size for speed (if too many voxels)
        N_thin_target = int(min(int(keep_factor * Ncand), int(keep_in_pre.numel())))
        if keep_in_pre.numel() > N_thin_target:
            # If thinning produced too many points, downsample them proportional to p_pre
            p_keep = p_pre[keep_in_pre]
            p_keep = (p_keep + 1e-8) / p_keep.sum().clamp(min=1e-8)
            sub = torch.multinomial(p_keep, num_samples=N_thin_target, replacement=False)
            keep_in_pre = keep_in_pre[sub]

        thin_idx = pre_idx[keep_in_pre]       # indices into full set
        p_thin = p[thin_idx].contiguous()
        p_thin = (p_thin + 1e-8) / p_thin.sum().clamp(min=1e-8)

        # Final candidates from thinned pool
        if thin_idx.numel() <= Ncand:
            cand_idx = thin_idx
        else:
            sel = torch.multinomial(p_thin, num_samples=Ncand, replacement=False)
            cand_idx = thin_idx[sel]

        xyz_cand = flat_xyz_all[cand_idx]  # [Ncand,3]

        # ----- gather candidate feat from feat_red (needs mapping flat idx -> t,v,y,x, then to H',W') -----
        # feat_red: [T,V,Hp,Wp,C], points_full: [T,V,H,W,3]
        # decode flat index in TVHW
        HW = int(H * W)
        VHW = int(V * HW)
        t_idx = (cand_idx // VHW).clamp(0, T - 1)
        rem0 = cand_idx - t_idx * VHW
        v_idx = (rem0 // HW).clamp(0, V - 1)
        rem1 = rem0 - v_idx * HW
        y_idx = (rem1 // W).clamp(0, H - 1)
        x_idx = (rem1 - y_idx * W).clamp(0, W - 1)

        # map (y,x) in image to (yp,xp) in feat map (Hp,Wp)
        yp = ((y_idx.to(torch.long) * Hp) // max(1, H)).clamp(0, Hp - 1)
        xp = ((x_idx.to(torch.long) * Wp) // max(1, W)).clamp(0, Wp - 1)

        feat_cand = feat_red[t_idx, v_idx, yp, xp, :]  # [Ncand,C]

        # ----- build importance head inputs -----
        # time/view embedding (32 dims)
        time_ids_long = time_ids.to(torch.long)
        t_emb = self.imp_time_emb(time_ids_long[t_idx].clamp(min=0, max=self.imp_time_emb.num_embeddings - 1))
        v_emb = self.imp_view_emb(v_idx.clamp(min=0, max=self.imp_view_emb.num_embeddings - 1))
        pos_emb = torch.cat([t_emb, v_emb], dim=-1)  # [Ncand,32]

        inputs = [feat_cand, pos_emb]
        if bool(getattr(self.cfg, "imp_use_xyz", True)):
            inputs.insert(1, xyz_cand)  # after feat
        if bool(getattr(self.cfg, "imp_use_conf", True)):
            if conf_flat is not None:
                inputs.append(conf_flat[cand_idx].view(-1, 1))
            else:
                inputs.append(torch.zeros(Ncand, 1, device=device, dtype=dtype))
        if bool(getattr(self.cfg, "imp_use_dyn", True)):
            if dyn_flat is not None:
                inputs.append(dyn_flat[cand_idx].view(-1, 1))
            else:
                inputs.append(torch.zeros(Ncand, 1, device=device, dtype=dtype))

        imp_in = torch.cat(inputs, dim=-1)  # [Ncand, D]
        imp_logits = self.imp_head(imp_in).squeeze(-1)  # [Ncand]

        # ----- differentiable top-k selection (ST) -----
        tau = float(kwargs.get("imp_gumbel_tau", getattr(self.cfg, "imp_gumbel_tau", 1.0)))
        tau_min = float(getattr(self.cfg, "imp_tau_min", 0.2))
        tau = max(tau_min, tau)  # you can anneal in training by editing cfg.imp_gumbel_tau each epoch

        if self.training:
            idx_hard, y_st = _gumbel_topk_straight_through_sparse(imp_logits, k=M_full, tau=tau)
        else:
            N = int(imp_logits.shape[0])
            k = int(min(max(1, M_full), N))
            y = imp_logits / float(max(1e-6, tau))
            idx_hard = torch.topk(y, k=k, largest=True).indices
            y_st = y[idx_hard]
        # forward: hard index selection
        xyz_f0 = xyz_cand[idx_hard]  # [M,3]

        # ----- p_dyn_f: also selected sparsely -----
        if dyn_flat is not None:
            dyn_cand = dyn_flat[cand_idx].view(-1, 1)  # [Ncand,1]
            p_dyn_f = dyn_cand[idx_hard].clamp(0.0, 1.0)  # [M,1]
        else:
            p_dyn_f = torch.full((M_full, 1), 0.5, device=device, dtype=dtype)

        # ============================================================
        # Stage C2/C3: soft assignment → distribute anchor motion to selected fine points
        # ============================================================
        if Md > 0:
            idx_d, w_d = _soft_assign_topk(
                xyz_f0,
                mu_d0,
                topk=int(self.cfg.assign_topk),
                sigma=float(self.cfg.assign_sigma),
                chunk=4096,
            )  # [M,K]
            dxyz_sel = dxyz_anchor_d[:, idx_d]  # [T,M,K,3]
            dxyz_f = (w_d.unsqueeze(0).unsqueeze(-1) * dxyz_sel).sum(dim=2)  # [T,M,3]
        else:
            idx_d = torch.zeros(M_full, 0, device=device, dtype=torch.long)
            w_d = torch.zeros(M_full, 0, device=device, dtype=dtype)
            dxyz_f = torch.zeros(T, M_full, 3, device=device, dtype=dtype)

        # gate motion by dynamic probability
        dxyz_f = dxyz_f * p_dyn_f.view(1, M_full, 1)

        # Step 7: residual motion（可选）
        if self.residual_motion_head is not None:
            z_f = self.point_aggregator(
                xyz_f0,
                feat_red[0:1],             # use t=0 features for residual input; ok
                camera_poses[0:1],
                camera_intrinsics[0:1],
                time_ids[0:1],
            )  # [M,C]
            dxyz_res = self.residual_motion_head(z_f, time_ids)  # [T,M,3]
            dxyz_f = dxyz_f + p_dyn_f.view(1, M_full, 1) * dxyz_res
        else:
            dxyz_res = None

        mu_t = xyz_f0.unsqueeze(0) + dxyz_f  # [T,M,3]


        # Step 4/5: 聚合外观特征 → 解码 Gaussian 属性（不改几何）
        g = self.point_aggregator(xyz_f0, feat_red, camera_poses, camera_intrinsics, time_ids)  # [M,C]
        gauss = self.gaussian_head(g)
        # scale 有界化：避免直接 clamp(max) 导致触顶后梯度为 0（scale_t“学不动”）
        max_scale = float(getattr(self.cfg, "max_scale", 0.05))
        min_scale = float(getattr(self.cfg, "min_scale", 1e-4))
        # gaussian_head 里 scale 用 softplus 输出，可能远大于 max_scale；用 sigmoid 压到 (0,1) 再映射到 [min,max]
        s01 = torch.sigmoid(gauss["scale"])
        gauss["scale"] = (min_scale + (max_scale - min_scale) * s01).clamp(min=min_scale, max=max_scale)

        color_t = gauss["color"].unsqueeze(0).expand(T, -1, -1)
        alpha_t = gauss["opacity"].unsqueeze(0).expand(T, -1, -1)
        scale_t = gauss["scale"].unsqueeze(0).expand(T, -1, -1)
        rot_t = gauss["rot"].unsqueeze(0).expand(T, -1, -1, -1)

        return {
            "mu_t": mu_t,
            "scale_t": scale_t,
            "rot_t": rot_t,
            "color_t": color_t,
            "alpha_t": alpha_t,
            "dxyz_t": dxyz_f,
            "dxyz_t_res": dxyz_res,
            "anchors_mu_static": mu_s0,
            "anchors_mu_dynamic": mu_d0,
            "dxyz_anchor_dynamic": dxyz_anchor_d,
            "assign_idx_dynamic": idx_d,
            "assign_w_dynamic": w_d,
            "p_dyn_f": p_dyn_f,
            # optional debug/regularization hooks
            "xyz_f0": xyz_f0,
            "imp_logits": imp_logits,
        }


if __name__ == "__main__":
    # 用 YAML 配置做一次最小 smoke-test（不依赖 diff_gaussian_rasterization）
    cfg = load_config("configs/ff4dgsmotion.yaml").get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Trellis4DGS4DCanonical(cfg=cfg).to(device)

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
