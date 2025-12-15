"""Sinkhorn optimal transport for slot matching"""
import torch


def sinkhorn_knopp(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, 
                   num_iter: int = 100, lambda_reg: float = 1.0) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for optimal transport"""
    u = torch.ones_like(a) / a.shape[0]
    v = torch.ones_like(b) / b.shape[0]
    K = torch.exp(-C / lambda_reg)
    for _ in range(num_iter):
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P




