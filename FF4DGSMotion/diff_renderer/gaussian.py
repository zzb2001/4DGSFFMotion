import torch
import math
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        _RasterizeGaussians,
    )  # type: ignore
except Exception:  # pragma: no cover
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _RasterizeGaussians = None
from camera import Camera
from dataclasses import dataclass
from contextlib import nullcontext


@dataclass
class GaussianAttributes:
    xyz: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    sh: torch.Tensor


def render_gs(
    camera: Camera,
    bg_color: torch.Tensor,
    gs: GaussianAttributes,
    target_image: torch.Tensor = None,
    sh_degree: int = 0,
    scaling_modifier: float = 1.0
) -> dict[str, torch.Tensor]:
    if GaussianRasterizationSettings is None or GaussianRasterizer is None:
        raise ImportError("diff_gaussian_rasterization is required for render_gs()")
    # diff_gaussian_rasterization CUDA kernel expects fp32 inputs; AMP/fp16 will crash.
    # Cast all inputs to float32 (keeps gradients), and disable autocast for the rasterizer call.
    xyz = gs.xyz.float().contiguous()
    opacity = gs.opacity.float()
    # The rasterizer backward expects gradients shaped like opacities input.
    # Use [N,1] (canonical 3DGS convention) to avoid grad shape mismatch.
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(-1)  # [N,1]
    elif opacity.dim() == 2 and opacity.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"gs.opacity must be [N] or [N,1], got {tuple(opacity.shape)}")
    opacity = opacity.contiguous()
    scaling = gs.scaling.float().contiguous()
    rotation = gs.rotation.float().contiguous()
    sh = gs.sh.float().contiguous()
    bg_color = bg_color.float()
    if target_image is not None:
        target_image = target_image.to(device=xyz.device, dtype=torch.float32, non_blocking=True)
    # torch.cuda.amp.autocast 已弃用，改用 torch.amp.autocast('cuda', enabled=...)
    autocast_ctx = torch.amp.autocast('cuda', enabled=False) if xyz.is_cuda else nullcontext()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    if screenspace_points.requires_grad: # requires_grad == False when inference
        screenspace_points.retain_grad()

    # Set up rasterization configuration
    tanfovx = math.tan(camera.fov_x * 0.5)
    tanfovy = math.tan(camera.fov_y * 0.5)

    viewmatrix = camera.get_w2v
    projmatrix = camera.get_full_proj
    with autocast_ctx:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix.transpose(0, 1).contiguous().to(device='cuda', non_blocking=True).float(),
            projmatrix=projmatrix.transpose(0, 1).contiguous().to(device='cuda', non_blocking=True).float(),
            sh_degree=sh_degree,
            campos=camera.get_pos.contiguous().to(device='cuda', non_blocking=True).float(),
            prefiltered=False,
            debug=False
        )

        # Prefer calling the underlying autograd.Function directly to avoid potential
        # shape squeezing in wrapper code that can cause invalid opacities gradients.
        if _RasterizeGaussians is not None:
            colors_precomp = torch.empty(0, device=xyz.device, dtype=torch.float32)
            cov3D_precomp = torch.empty(0, device=xyz.device, dtype=torch.float32)
            target_image_t = target_image if target_image is not None else torch.empty(0, device=xyz.device, dtype=torch.float32)
            color, alpha, est_color, est_weight, radii = _RasterizeGaussians.apply(
                xyz,
                screenspace_points,
                sh,
                colors_precomp,
                opacity,         # [N,1] to match backward grad shape
                scaling,
                rotation,
                cov3D_precomp,
                target_image_t,
                raster_settings,
            )
        else:
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            # Rasterize visible Gaussians to image.
            color, alpha, est_color, est_weight, radii = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=sh,
                colors_precomp=None,
                opacities=opacity,
                scales=scaling,
                rotations=rotation,
                cov3D_precomp=None,
                target_image=target_image
            )
    return {"color": color, "alpha": alpha, "est_color": est_color, "est_weight": est_weight, "radii": radii}


def render_gs_batch( # legacy
    camera: Camera,
    bg_color: torch.Tensor,
    gs: GaussianAttributes,
    target_image: torch.Tensor = None,
    sh_degree: int = 0,
    scaling_modifier: float = 1.0
) -> dict[str, torch.Tensor]:
    if GaussianRasterizationSettings is None or GaussianRasterizer is None:
        raise ImportError("diff_gaussian_rasterization is required for render_gs_batch()")
    # diff_gaussian_rasterization CUDA kernel expects fp32 inputs; AMP/fp16 will crash.
    xyz = gs.xyz.float().contiguous()
    opacity = gs.opacity.float()
    # Use [B,N,1] for batch (canonical 3DGS convention) to avoid grad shape mismatch.
    if opacity.dim() == 2:
        opacity = opacity.unsqueeze(-1)  # [B,N,1]
    elif opacity.dim() == 3 and opacity.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"gs.opacity must be [B,N] or [B,N,1], got {tuple(opacity.shape)}")
    opacity = opacity.contiguous()
    scaling = gs.scaling.float().contiguous()
    rotation = gs.rotation.float().contiguous()
    sh = gs.sh.float().contiguous()
    bg_color = bg_color.float()
    if target_image is not None:
        target_image = target_image.to(device=xyz.device, dtype=torch.float32, non_blocking=True)
    autocast_ctx = torch.cuda.amp.autocast(enabled=False) if xyz.is_cuda else nullcontext()

    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    if screenspace_points.requires_grad: # requires_grad == False when inference
        screenspace_points.retain_grad()

    tanfovx = math.tan(camera.fov_x * 0.5)
    tanfovy = math.tan(camera.fov_y * 0.5)

    viewmatrix = camera.get_w2v
    projmatrix = camera.get_full_proj
    bg_color = bg_color.reshape(-1, 3)
    if bg_color.shape[0] == 1:
        bg_color = bg_color.repeat(xyz.shape[0], 1)

    bs = xyz.shape[0]
    color_list = [] 
    alpha_list = [] 
    est_color_list = [] 
    est_weight_list = [] 
    radii_list = []
    for i in range(bs):
        with autocast_ctx:
            raster_settings = GaussianRasterizationSettings(
                image_height=int(camera.height),
                image_width=int(camera.width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color[i],
                scale_modifier=scaling_modifier,
                viewmatrix=viewmatrix.transpose(0, 1).contiguous().cuda().float(),
                projmatrix=projmatrix.transpose(0, 1).contiguous().cuda().float(),
                sh_degree=sh_degree,
                campos=camera.get_pos.contiguous().cuda().float(),
                prefiltered=False,
                debug=False
            )

            if _RasterizeGaussians is not None:
                ti = target_image[i] if target_image is not None else None
                colors_precomp = torch.empty(0, device=xyz.device, dtype=torch.float32)
                cov3D_precomp = torch.empty(0, device=xyz.device, dtype=torch.float32)
                target_image_t = ti if ti is not None else torch.empty(0, device=xyz.device, dtype=torch.float32)
                color, alpha, est_color, est_weight, radii = _RasterizeGaussians.apply(
                    xyz[i],
                    screenspace_points[i],
                    sh[i],
                    colors_precomp,
                    opacity[i],       # [N,1]
                    scaling[i],
                    rotation[i],
                    cov3D_precomp,
                    target_image_t,
                    raster_settings,
                )
            else:
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                color, alpha, est_color, est_weight, radii = rasterizer(
                    means3D=xyz[i],
                    means2D=screenspace_points[i],
                    shs=sh[i],
                    colors_precomp=None,
                    opacities=opacity[i],
                    scales=scaling[i],
                    rotations=rotation[i],
                    cov3D_precomp=None,
                    target_image=target_image[i] if target_image is not None else None
                )
        color_list.append(color)
        alpha_list.append(alpha)
        est_color_list.append(est_color)
        est_weight_list.append(est_weight)
        radii_list.append(radii)

    return {
        "color": torch.stack(color_list), 
        "alpha": torch.stack(alpha_list), 
        "est_color": torch.stack(est_color_list), 
        "est_weight": torch.stack(est_weight_list), 
        "radii": torch.stack(radii_list)
    }
