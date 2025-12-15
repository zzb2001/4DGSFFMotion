"""Voxelization utilities: Morton encoding, voxel indexing"""
import torch
import numpy as np
from typing import Tuple, Optional


def morton_encode(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Morton (Z-order) encoding"""
    morton = torch.zeros_like(x, dtype=torch.int64)
    x = x.clamp(0, (1 << 21) - 1)
    y = y.clamp(0, (1 << 21) - 1)
    z = z.clamp(0, (1 << 21) - 1)
    for i in range(21):
        morton |= (x & (1 << i)).long() << (3 * i)
        morton |= (y & (1 << i)).long() << (3 * i + 1)
        morton |= (z & (1 << i)).long() << (3 * i + 2)
    return morton


def points_to_voxel_indices(points: torch.Tensor, voxel_size: float, 
                            bounds: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert 3D points to voxel indices"""
    min_bound, max_bound = bounds
    normalized = (points - min_bound) / (max_bound - min_bound)
    voxel_coords = (normalized / voxel_size).floor().long()
    max_voxel = int((max_bound - min_bound) / voxel_size)
    voxel_coords = voxel_coords.clamp(0, max_voxel - 1)
    return voxel_coords[..., 0], voxel_coords[..., 1], voxel_coords[..., 2]


def voxel_indices_to_centers(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor,
                             voxel_size: float, bounds: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Convert voxel indices to voxel center coordinates"""
    min_bound, max_bound = bounds
    x = (ix.float() + 0.5) * voxel_size + min_bound
    y = (iy.float() + 0.5) * voxel_size + min_bound
    z = (iz.float() + 0.5) * voxel_size + min_bound
    return torch.stack([x, y, z], dim=-1)


def scatter_add_weighted(values: torch.Tensor, indices: torch.Tensor,
                        weights: Optional[torch.Tensor] = None,
                        dim_size: Optional[int] = None) -> torch.Tensor:
    """Weighted scatter_add"""
    if dim_size is None:
        if indices.numel() > 0:
            dim_size = int(indices.max().item()) + 1
        else:
            dim_size = 0
    if dim_size == 0:
        return torch.zeros(0, values.shape[1], device=values.device, dtype=values.dtype)
    if weights is not None:
        weighted_values = values * weights.unsqueeze(-1)
    else:
        weighted_values = values
    aggregated = torch.zeros(dim_size, values.shape[1], device=values.device, dtype=values.dtype)
    aggregated.index_add_(0, indices, weighted_values)
    if weights is None:
        counts = torch.bincount(indices, minlength=dim_size).float().unsqueeze(-1).clamp(min=1)
        aggregated = aggregated / counts
    return aggregated


def compute_voxel_resolution(num_points: int, target_num_voxels: int = 120000) -> int:
    """Compute appropriate voxel resolution"""
    resolution = int(np.cbrt(target_num_voxels))
    if resolution <= 128:
        return 128
    elif resolution <= 192:
        return 192
    elif resolution <= 256:
        return 256
    else:
        return 256




