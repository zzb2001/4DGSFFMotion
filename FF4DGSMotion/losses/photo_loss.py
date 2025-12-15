"""Photo-consistency loss: relative view synthesis"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PhotoConsistencyLoss(nn.Module):
    """Photo-consistency loss between views"""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        rendered_images: torch.Tensor,  # [B, N, H, W, 3]
        target_images: torch.Tensor,  # [B, N, H, W, 3]
        masks: Optional[torch.Tensor] = None,  # [B, N, H, W]
    ) -> torch.Tensor:
        """
        Compute photo-consistency loss
        
        Args:
            rendered_images: Rendered images from different views
            target_images: Target images
            masks: Optional masks for valid regions
            
        Returns:
            loss: Scalar loss
        """
        if masks is not None:
            # Apply masks
            rendered_masked = rendered_images * masks.unsqueeze(-1)
            target_masked = target_images * masks.unsqueeze(-1)
        else:
            rendered_masked = rendered_images
            target_masked = target_images
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(rendered_masked, target_masked, reduction='mean')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(rendered_masked, target_masked, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class RelativeViewSynthesisLoss(nn.Module):
    """Relative view synthesis loss (warping between views)"""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        rendered_view_i: torch.Tensor,  # [B, H, W, 3]
        rendered_view_j: torch.Tensor,  # [B, H, W, 3]
        warped_view_i_to_j: torch.Tensor,  # [B, H, W, 3] (warped from i to j)
        masks: Optional[torch.Tensor] = None,  # [B, H, W]
    ) -> torch.Tensor:
        """
        Compute relative view synthesis loss
        
        Args:
            rendered_view_i: Rendered image from view i
            rendered_view_j: Rendered image from view j
            warped_view_i_to_j: View i warped to view j's viewpoint
            masks: Optional masks
            
        Returns:
            loss: Scalar loss
        """
        if masks is not None:
            rendered_j_masked = rendered_view_j * masks.unsqueeze(-1)
            warped_masked = warped_view_i_to_j * masks.unsqueeze(-1)
        else:
            rendered_j_masked = rendered_view_j
            warped_masked = warped_view_i_to_j
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(warped_masked, rendered_j_masked, reduction='mean')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(warped_masked, rendered_j_masked, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss

