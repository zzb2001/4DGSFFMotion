"""Gaussian renderer using gsplat"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import math
import os
from torchvision.utils import save_image


try:
    from gsplat import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False
    print("Warning: gsplat not available, using dummy renderer")


class GaussianRenderer(nn.Module):
    """Gaussian splatting renderer using gsplat"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (378, 574),
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 1e-10,
        far_plane: Optional[float] = None,
        radius_clip: float = 0.1,
        rasterize_mode: str = 'classic',
        sh_degree: int = 0,  # Spherical harmonics degree (0 for RGB only)
        # Debug options
        debug_save_points: bool = False,
        debug_dir: str = "debug/plys",
        debug_save_images: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.radius_clip = radius_clip
        self.rasterize_mode = rasterize_mode
        self.sh_degree = sh_degree
        self.debug_save_points = bool(debug_save_points)
        self.debug_dir = str(debug_dir)
        self.debug_save_images = bool(debug_save_images)
    
    @staticmethod
    def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrices to quaternions (w, x, y, z).
        
        Args:
            R: Rotation matrices [..., 3, 3]
            
        Returns:
            quats: Quaternions [..., 4] in (w, x, y, z) format
        """
        # Ensure R is at least 2D
        original_shape = R.shape
        if R.dim() < 2:
            R = R.unsqueeze(0)
        
        # Flatten batch dimensions
        batch_dims = R.shape[:-2]
        R_flat = R.view(-1, 3, 3)  # [N, 3, 3]
        
        # Trace of rotation matrix
        trace = R_flat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [N]
        
        # Compute quaternion components based on trace
        quats = torch.zeros(R_flat.shape[0], 4, device=R.device, dtype=R.dtype)  # [N, 4]
        
        # Case 1: trace > 0
        mask1 = trace > 0
        if mask1.any():
            s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
            quats[mask1, 0] = 0.25 * s  # w
            quats[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # x
            quats[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # y
            quats[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # z
        
        # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
        mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
        if mask2.any():
            s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2  # s = 4 * qx
            quats[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s  # w
            quats[mask2, 1] = 0.25 * s  # x
            quats[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s  # y
            quats[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s  # z
        
        # Case 3: R[1,1] > R[2,2]
        mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
        if mask3.any():
            s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2  # s = 4 * qy
            quats[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s  # w
            quats[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s  # x
            quats[mask3, 2] = 0.25 * s  # y
            quats[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s  # z
        
        # Case 4: otherwise
        mask4 = (~mask1) & (~mask2) & (~mask3)
        if mask4.any():
            s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2  # s = 4 * qz
            quats[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s  # w
            quats[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s  # x
            quats[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s  # y
            quats[mask4, 3] = 0.25 * s  # z
        
        # Normalize quaternions
        quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
        
        # Reshape to original batch dimensions
        quats = quats.view(*batch_dims, 4)  # [..., 4]
        
        return quats
    
    @property
    def has_gsplat(self):
        return HAS_GSPLAT
    
    def forward(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera_poses: torch.Tensor,  # [B, N, 4, 4] c2w
        intrinsics: Optional[torch.Tensor] = None,  # [B, N, 3, 3]
    ) -> torch.Tensor:
        """
        Render Gaussians
        
        Args:
            gaussian_params: dict with keys:
                - mu: [B, num_gaussians, 3] positions (world coordinates)
                - scale: [B, num_gaussians, 3] scales
                - color: [B, num_gaussians, 3] colors (RGB, range [0, 1])
                - opacity: [B, num_gaussians, 1] opacities
                - rotation: Optional [B, num_gaussians, 4] quaternions (w, x, y, z)
            camera_poses: Camera-to-world poses [B, N, 4, 4]
            intrinsics: Optional camera intrinsics [B, N, 3, 3]
            
        Returns:
            rendered_images: [B, N, H, W, 3] RGB images
        """
        if not self.has_gsplat:
            # Dummy renderer (returns zeros)
            B, N = camera_poses.shape[:2]
            H, W = self.image_size
            return torch.zeros(B, N, H, W, 3, device=camera_poses.device)
        
        B, N = camera_poses.shape[:2]
        H, W = self.image_size
        device = camera_poses.device
        
        # Extract parameters
        mu = gaussian_params['mu']  # [B, num_gaussians, 3]
        scale = gaussian_params['scale']  # [B, num_gaussians, 3]
        color = gaussian_params['color']  # [B, num_gaussians, 3]
        opacity = gaussian_params['opacity'].squeeze(-1)  # [B, num_gaussians]
        
        # Rotation: if not provided, use identity quaternions
        if 'rotation' in gaussian_params:
            rotation = gaussian_params['rotation']  # Could be [B, num_gaussians, 4] or [B, num_gaussians, 3, 3]
            
            # Check if rotation is matrices (3x3) or quaternions (4)
            if rotation.dim() == 4 and rotation.shape[-1] == 3 and rotation.shape[-2] == 3:
                # Rotation matrices [B, num_gaussians, 3, 3] -> convert to quaternions [B, num_gaussians, 4]
                rotation = self._rotation_matrix_to_quaternion(rotation)
            elif rotation.dim() == 3 and rotation.shape[-1] == 4:
                # Already quaternions [B, num_gaussians, 4]
                pass
            else:
                # Unexpected shape, use identity quaternions
                num_gaussians = mu.shape[1]
                rotation = torch.zeros(B, num_gaussians, 4, device=device)
                rotation[:, :, 0] = 1.0  # w component
        else:
            # Default: identity quaternion (w=1, x=0, y=0, z=0)
            num_gaussians = mu.shape[1]
            rotation = torch.zeros(B, num_gaussians, 4, device=device)
            rotation[:, :, 0] = 1.0  # w component
        
        # Default intrinsics if not provided
        if intrinsics is None:
            # Approximate focal length (assuming FOV ~60 degrees)
            focal = H * 0.5 / math.tan(math.radians(30))
            K = torch.tensor([
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1],
            ], device=device, dtype=torch.float32).unsqueeze(0).expand(B, N, -1, -1)
        else:
            K = intrinsics
        
        rendered_images = []
        
        for b in range(B):
            batch_images = []
            for view_idx in range(N):
                # Get parameters for this batch and view
                mu_b = mu[b]  # [num_gaussians, 3]
                scale_b = scale[b]  # [num_gaussians, 3]
                rotation_b = rotation[b]  # [num_gaussians, 4]
                opacity_b = opacity[b]  # [num_gaussians]
                color_b = color[b]  # [num_gaussians, 3]
                
                # Get camera pose
                c2w = camera_poses[b, view_idx]  # [4, 4]
                w2c = torch.inverse(c2w)  # [4, 4] world-to-camera
                
                # Get intrinsics
                K_b = K[b, view_idx]  # [3, 3]
                
                # Check if intrinsics are normalized (fx < 1)
                K_denorm = K_b.clone()
                if K_b[0, 0] < 1.0 or K_b[1, 1] < 1.0:
                    # Denormalize
                    K_denorm[0, 0] = K_b[0, 0] * W  # fx
                    K_denorm[1, 1] = K_b[1, 1] * H  # fy
                    K_denorm[0, 2] = K_b[0, 2] * W  # cx
                    K_denorm[1, 2] = K_b[1, 2] * H  # cy
                
                # Prepare SH coefficients
                # For sh_degree=0, we only need SH0 (constant term)
                # SH0 coefficient is just the RGB color
                if self.sh_degree == 0:
                    # Simple RGB: use color directly as SH0
                    # gsplat expects [N, 1, 3] for sh_degree=0
                    feature = color_b.unsqueeze(1)  # [num_gaussians, 1, 3]
                else:
                    # For higher SH degrees, we need to convert color to SH coefficients
                    # Simplified: use color as SH0, zeros for higher orders
                    num_sh_coeffs = (self.sh_degree + 1) ** 2
                    feature = torch.zeros(mu_b.shape[0], num_sh_coeffs, 3, device=device)
                    feature[:, 0, :] = color_b  # SH0 = color
                
                # Ensure background color is on correct device
                background_color_device = self.background_color.to(device)
                
                # Render using gsplat.rasterization
                rendering, alpha, _ = rasterization(
                    means=mu_b.float(),  # [num_gaussians, 3] world coordinates
                    quats=rotation_b.float(),  # [num_gaussians, 4] quaternions (w, x, y, z)
                    scales=scale_b.float(),  # [num_gaussians, 3]
                    opacities=opacity_b.float(),  # [num_gaussians]
                    colors=feature.float(),  # [num_gaussians, num_sh_coeffs, 3]
                    viewmats=w2c.unsqueeze(0).float(),  # [1, 4, 4] world-to-camera
                    Ks=K_denorm.unsqueeze(0).float(),  # [1, 3, 3] intrinsics
                    width=W,
                    height=H,
                    sh_degree=self.sh_degree,
                    render_mode="RGB",  # RGB only (no depth)
                    packed=False,
                    near_plane=self.near_plane,
                    backgrounds=background_color_device.unsqueeze(0),  # [1, 3]
                    radius_clip=self.radius_clip,
                    rasterize_mode=self.rasterize_mode,
                )  # rendering: [1, H, W, 3], alpha: [1, H, W]
                
                # Extract RGB image
                rendering_img = rendering.squeeze(0)  # [H, W, 3]
                rendering_img = rendering_img.clamp(0.0, 1.0)
                
                batch_images.append(rendering_img)
            
            rendered_images.append(torch.stack(batch_images, dim=0))  # [N, H, W, 3]
        
        return torch.stack(rendered_images, dim=0)  # [B, N, H, W, 3]
    
    def render_with_depth(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera_poses: torch.Tensor,  # [B, N, 4, 4] c2w
        intrinsics: Optional[torch.Tensor] = None,  # [B, N, 3, 3]
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians with depth
        
        Returns:
            dict with keys:
            - images: [B, N, H, W, 3] RGB images
            - depth: [B, N, H, W] depth maps
            - alpha: [B, N, H, W] alpha maps
        """
        if not self.has_gsplat:
            B, N = camera_poses.shape[:2]
            H, W = self.image_size
            device = camera_poses.device
            return {
                'images': torch.zeros(B, N, H, W, 3, device=device),
                'depth': torch.zeros(B, N, H, W, device=device),
                'alpha': torch.zeros(B, N, H, W, device=device),
            }
        
        B, N = camera_poses.shape[:2]
        H, W = self.image_size
        device = camera_poses.device
        
        # Extract parameters
        mu = gaussian_params['mu']
        scale = gaussian_params['scale']
        color = gaussian_params['color']
        opacity = gaussian_params['opacity'].squeeze(-1)
        
        if 'rotation' in gaussian_params:
            rotation = gaussian_params['rotation']
            # Accept either rotation matrices [B, M, 3, 3] or quaternions [B, M, 4]
            if rotation.dim() == 4 and rotation.shape[-2:] == (3, 3):
                rotation = self._rotation_matrix_to_quaternion(rotation)
            elif rotation.dim() == 3 and rotation.shape[-1] == 4:
                pass  # already quaternions
            else:
                # Fallback to identity quaternions if unexpected shape
                num_gaussians = mu.shape[1]
                rotation = torch.zeros(B, num_gaussians, 4, device=device, dtype=mu.dtype)
                rotation[:, :, 0] = 1.0
        else:
            num_gaussians = mu.shape[1]
            rotation = torch.zeros(B, num_gaussians, 4, device=device, dtype=mu.dtype)
            rotation[:, :, 0] = 1.0
        
        if intrinsics is None:
            focal = H * 0.5 / math.tan(math.radians(30))
            K = torch.tensor([
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1],
            ], device=device, dtype=torch.float32).unsqueeze(0).expand(B, N, -1, -1)
        else:
            K = intrinsics
        
        rendered_images = []
        rendered_depths = []
        rendered_alphas = []

        for b in range(B):
            mu_b = mu[b].contiguous()              # [M,3]
            scale_b = scale[b].contiguous()        # [M,3]
            rotation_b = rotation[b].contiguous()  # [M,4]
            opacity_b = opacity[b].contiguous()    # [M]
            color_b = color[b].contiguous()        # [M,3]

            # Build SH features once
            if self.sh_degree == 0:
                feature = color_b.unsqueeze(1).contiguous()  # [M,1,3]
            else:
                num_sh_coeffs = (self.sh_degree + 1) ** 2
                feature = torch.zeros(mu_b.shape[0], num_sh_coeffs, 3, device=device)
                feature[:, 0, :] = color_b

            # Precompute all view matrices and intrinsics for this batch
            w2c_all = torch.linalg.inv(camera_poses[b]).contiguous()  # [N,4,4]
            K_all = K[b].clone().contiguous()                         # [N,3,3]
            # Denormalize intrinsics if needed
            mask_norm = (K_all[:, 0, 0] < 1.0) | (K_all[:, 1, 1] < 1.0)
            if mask_norm.any():
                K_all[mask_norm, 0, 0] = K_all[mask_norm, 0, 0] * W
                K_all[mask_norm, 1, 1] = K_all[mask_norm, 1, 1] * H
                K_all[mask_norm, 0, 2] = K_all[mask_norm, 0, 2] * W
                K_all[mask_norm, 1, 2] = K_all[mask_norm, 1, 2] * H

            bg = self.background_color.to(device).expand(N, -1).contiguous()  # [N,3]

            try:
                with torch.inference_mode():
                    rendering, alpha, _ = rasterization(
                        means=mu_b,
                        quats=rotation_b,
                        scales=scale_b,
                        opacities=opacity_b,           # use provided opacities
                        colors=feature,
                        viewmats=w2c_all,              # [N,4,4]
                        Ks=K_all,                      # [N,3,3]
                        width=W,
                        height=H,
                        sh_degree=self.sh_degree,
                        render_mode="RGB+D",
                        packed=False,
                        near_plane=self.near_plane,
                        backgrounds=bg,                # [N,3]
                        radius_clip=self.radius_clip,
                        rasterize_mode=self.rasterize_mode,
                    )  # rendering: [N,H,W,4], alpha: [N,H,W]
                # Split RGB and depth
                rgb, depth = torch.split(rendering, [3, 1], dim=-1)  # [N,H,W,3], [N,H,W,1]
                rgb = rgb.clamp(0.0, 1.0)
                depth = depth.squeeze(-1)

                # Debug save per view if enabled
                if self.debug_save_images:
                    img_dir = os.path.join(self.debug_dir, 'images')
                    os.makedirs(img_dir, exist_ok=True)
                    for v in range(N):
                        out_path = os.path.join(img_dir, f"rgb_depth_{b}_{v}.png")
                        rgb_chw = rgb[v].permute(2, 0, 1).contiguous()   # [3,H,W]
                        d = depth[v]
                        d_ = d.clone()
                        d_[~torch.isfinite(d_)] = 0
                        if torch.isfinite(d_).any():
                            d_min = torch.quantile(d_.flatten(), 0.01)
                            d_max = torch.quantile(d_.flatten(), 0.99)
                            denom = (d_max - d_min).clamp(min=1e-6)
                            d_norm = ((d_ - d_min) / denom).clamp(0.0, 1.0)
                        else:
                            d_norm = torch.zeros_like(d_)
                        d_rgb = d_norm.unsqueeze(0).repeat(3, 1, 1)
                        vis = torch.cat([rgb_chw, d_rgb], dim=2).clamp(0.0, 1.0)
                        save_image(vis.detach().cpu(), out_path)

                rendered_images.append(rgb)     # [N,H,W,3]
                rendered_depths.append(depth)   # [N,H,W]
                rendered_alphas.append(alpha)   # [N,H,W]
            except Exception as e:
                print(f"Rendering error (batched): {e}")
                # Fallback zeros for this batch
                rendered_images.append(torch.zeros(N, H, W, 3, device=device))
                rendered_depths.append(torch.zeros(N, H, W, device=device))
                rendered_alphas.append(torch.zeros(N, H, W, device=device))

        return {
            'images': torch.stack(rendered_images, dim=0),  # [B, N, H, W, 3]
            'depth': torch.stack(rendered_depths, dim=0),    # [B, N, H, W]
            'alpha': torch.stack(rendered_alphas, dim=0),    # [B, N, H, W]
        }
