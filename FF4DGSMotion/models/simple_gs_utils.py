# model/AnchorWarp4DGS/models/simple_gs_utils.py
import math
import torch
from typing import Tuple, Dict, Optional
from torchvision.utils import save_image
from FF4DGSMotion.models.simple_gaussian import SimpleGaussianModel
from FF4DGSMotion.models.mv_reconstruction import render_gs_batch
from _utils import Struct


# 控制由 Trellis 传入的 scale_t 映射到 SimpleGaussianModel 时的全局缩放因子：
#   - 值 < 1.0  会整体缩小高斯（更“细小”）
#   - 值 > 1.0  会整体放大高斯（更“膨胀”）
# 注意：这个因子直接作用在 scale_t → scale_pix 上，比改 model_config.init_scaling 更直观，
# 因为当前路径里我们总是显式传入 scales，init_scaling 只在 scales=None 时才起效。
INIT_SCALE_MULT: float = 0.75


def build_opengl_proj_from_K(
    K: torch.Tensor, H: int, W: int,
    near: float = 0.01, far: float = 100.0
) -> torch.Tensor:
    """
    从 3x3 内参 K 构造 OpenGL 风格 4x4 投影矩阵（和 test_gs_render.py 同号约定），
    保证图像不倒、不旋转 180°。
    K: [3,3]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    proj = K.new_zeros(4, 4)

    # 注意这里的正负号和 test_gs_render.py 一致
    proj[0, 0] = 2.0 * fx / W
    proj[1, 1] = -2.0 * fy / H
    proj[0, 2] = 1.0 - 2.0 * cx / W
    proj[1, 2] = 2.0 * cy / H - 1.0
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def compute_fov_from_K(K: torch.Tensor, H: int, W: int) -> Tuple[float, float]:
    """
    从相机内参估算水平/垂直 FOV（弧度）。
    """
    fx, fy = K[0, 0].item(), K[1, 1].item()
    fov_x = 2.0 * math.atan(W / (2.0 * fx))
    fov_y = 2.0 * math.atan(H / (2.0 * fy))
    return fov_x, fov_y


def build_gs_model_from_trellis_output(
    mu_t: torch.Tensor,      # [M,3]
    scale_t: torch.Tensor,   # [M,3] 或 [M]（Trellis 输出）
    color_t: torch.Tensor,   # [M,3]  (Trellis 输出颜色，可以先当作初始颜色)
    alpha_t: torch.Tensor,   # [M,1]  (Trellis 输出不透明度，用于初始化)
) -> Tuple[SimpleGaussianModel, torch.Tensor]:
    """
    把 Trellis 的一帧高斯参数封装成 SimpleGaussianModel。
    """
    M = mu_t.shape[0]
    device = mu_t.device

    # scale: 简化成一维像素尺度输入给 SimpleGaussianModel（保持和 test_gs_render.py 一致）
    if scale_t.ndim == 2 and scale_t.shape[1] == 3:
        scale_pix = scale_t.norm(dim=-1) / math.sqrt(3.0)   # [M]
    else:
        scale_pix = scale_t.view(-1)                        # [M]

    model_config = Struct(
        init_opacity=0.8,
        # 仅在 initialize_from_points(...) 未显式提供 scales 时生效；
        # 这里 Trellis 始终传入 scale_pix_device，因此 init_scaling 只作为兜底默认值。
        # 如需全局调节当前初始化大小，请优先修改上面的 INIT_SCALE_MULT。
        init_scaling=0.005,
        num_basis_in=1,
        num_basis_blend=1,
        use_weight_proj=True,   # 和 test_gs_render.py 保持一致
        use_blend=False
    )

    gs_model = SimpleGaussianModel(model_config)
    # 确保输入张量在正确的设备上
    mu_t_device = mu_t.to(device=device, dtype=torch.float32)
    color_t_device = color_t.to(device=device, dtype=torch.float32)

    # 统一控制 Trellis 输出的尺度在 SimpleGaussianModel 中的初始大小
    # （当前默认 0.25 倍，如果觉得仍然太大/太小，可直接修改 INIT_SCALE_MULT）
    scale_pix_device = (scale_pix * INIT_SCALE_MULT).to(device=device, dtype=torch.float32)
    gs_model.initialize_from_points(mu_t_device, colors=color_t_device, scales=scale_pix_device)

    return gs_model, scale_pix_device


def render_one_frame_simple_gs(
    mu_t: torch.Tensor,          # [M,3]
    scale_t: torch.Tensor,       # [M,3] or [M]
    color_t: torch.Tensor,       # [M,3]
    alpha_t: torch.Tensor,       # [M,1]
    camera_poses_t: torch.Tensor,# [V,4,4] c2w
    camera_intrinsics_t: torch.Tensor,  # [V,3,3]
    H: int,
    W: int,
    gt_images_t: Optional[torch.Tensor] = None,  # [V,H,W,3] 或 None
    do_fast_forward: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    单帧渲染：构造 SimpleGaussianModel + render_gs_batch。
    如果 do_fast_forward=True 且提供 gt_images_t，则：
      - 先渲染一遍得到 est_color / est_weight
      - 调 fast_forward 更新颜色
      - 再渲染一次作为最终输出
    返回:
      {
        "color": [V,H,W,3],  # 最终渲染结果
        "est_color": [V,H,W,3] or None,
        "est_weight": [V,H,W] or None,
    }
    """
    device = mu_t.device
    dtype = mu_t.dtype
    V = camera_poses_t.shape[0]

    # 1) 构造 SimpleGaussianModel
    # 验证输入张量的有效性
    if not torch.isfinite(mu_t).all():
        raise ValueError("mu_t contains NaN/Inf")
    if not torch.isfinite(scale_t).all():
        raise ValueError("scale_t contains NaN/Inf")
    if not torch.isfinite(color_t).all():
        raise ValueError("color_t contains NaN/Inf")
    if not torch.isfinite(alpha_t).all():
        raise ValueError("alpha_t contains NaN/Inf")
    
    # 同步 CUDA，确保之前的操作完成
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    
    gs_model, scale_pix = build_gs_model_from_trellis_output(mu_t, scale_t, color_t, alpha_t)
    
    # 再次同步，确保模型初始化完成
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)

    # 2) 准备相机参数（确保 dtype/device 一致且张量连续）
    # 先同步 CUDA，确保之前的操作完成
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    
    camera_poses_t = camera_poses_t.to(device=device, dtype=dtype).contiguous()
    camera_intrinsics_t = camera_intrinsics_t.to(device=device, dtype=dtype).contiguous()
    
    # 再次同步，确保转换完成
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    
    if not torch.isfinite(camera_poses_t).all() or not torch.isfinite(camera_intrinsics_t).all():
        raise ValueError("camera_poses_t or camera_intrinsics_t contains NaN/Inf")

    cam_view_mats = torch.inverse(camera_poses_t)  # [V,4,4] w2c
    cam_proj_mats = torch.empty((V, 4, 4), device=device, dtype=dtype)
    cam_positions = torch.empty((V, 3), device=device, dtype=dtype)

    for v in range(V):
        c2w = camera_poses_t[v]
        cam_positions[v] = c2w[:3, 3]
        K = camera_intrinsics_t[v]
        cam_proj_mats[v] = build_opengl_proj_from_K(K, H, W)

    # FOV
    K_ref = camera_intrinsics_t[0]
    fov_x, fov_y = compute_fov_from_K(K_ref, H, W)

    bg_color = torch.ones(V, 3, device=device) * 0.5

    # 3) 第一次渲染（拿 est_color / est_weight 做 fast_forward 初始化）
    # 验证 gs_model 参数的有效性
    try:
        gs_attrs = gs_model.get_batch_gaussian_attributes(V)
        # 验证 gs_attrs 中的张量是否有效
        for attr_name in ['xyz', 'opacity', 'scaling', 'rotation', 'features_dc']:
            if hasattr(gs_attrs, attr_name):
                attr_val = getattr(gs_attrs, attr_name)
                if not torch.isfinite(attr_val).all():
                    raise ValueError(f"gs_attrs.{attr_name} contains NaN/Inf")
    except Exception as e:
        raise RuntimeError(f"Failed to get gaussian attributes: {e}") from e
    
    target_image_chw = None
    if gt_images_t is not None:
        # gt: [V,H,W,3] -> [V,3,H,W]
        target_image_chw = gt_images_t.permute(0, 3, 1, 2).contiguous()
        # 验证 target_image 有效性
        if not torch.isfinite(gt_images_t).all():
            print(f"[WARNING] gt_images_t contains NaN/Inf, setting to None")
            gt_images_t = None

    # 同步 CUDA 操作，确保之前的错误能被及时捕获
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    
    # 验证所有输入张量的有效性
    try:
        if not torch.isfinite(cam_view_mats).all():
            raise ValueError("cam_view_mats contains NaN/Inf")
        if not torch.isfinite(cam_proj_mats).all():
            raise ValueError("cam_proj_mats contains NaN/Inf")
        if not torch.isfinite(cam_positions).all():
            raise ValueError("cam_positions contains NaN/Inf")
        if not torch.isfinite(bg_color).all():
            raise ValueError("bg_color contains NaN/Inf")
    except Exception as e:
        raise RuntimeError(f"Input validation failed before render_gs_batch: {e}") from e
    
    try:
        result_1 = render_gs_batch(
            fov_x, fov_y,
            W, H,
            cam_view_mats, cam_proj_mats, cam_positions,
            bg_color, gs_attrs,
            # render_gs_batch 期望输入为 [V,H,W,3]（HWC），内部会自行转 CHW
            target_image=gt_images_t if gt_images_t is not None else None,
        )
        # 同步 CUDA，确保渲染完成
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)
    except RuntimeError as e:
        if "CUDA" in str(e) or "illegal memory access" in str(e).lower():
            raise RuntimeError(f"CUDA error in render_gs_batch. Input shapes: mu_t={mu_t.shape}, V={V}, H={H}, W={W}. "
                             f"Check if gs_attrs are valid and all tensors are on the correct device.") from e
        raise
    est_color_1 = result_1["est_color"]   # shape may be [V,N,3] or [V,H,W,3]
    est_weight_1 = result_1["est_weight"] # shape may be [V,N] or [V,H,W]
    image_1 = result_1["color"]           # [V,H,W,3]

    # Prepare inputs for fast_forward if shapes are per-gaussian
    can_fast_forward = False
    est_color_ff = None
    est_weight_ff = None
    # Case A: [V,N,3] and [V,N]
    if est_color_1.dim() == 3 and est_weight_1.dim() == 2 and est_color_1.shape[0] == est_weight_1.shape[0] and est_color_1.shape[2] == 3:
        can_fast_forward = True
        est_color_ff = est_color_1                    # [V,N,3]
        est_weight_ff = est_weight_1                  # [V,N]
    # Case B: already reduced [N,3] and [N]
    elif est_color_1.dim() == 2 and est_weight_1.dim() == 1 and est_color_1.shape[1] == 3:
        can_fast_forward = True
        est_color_ff = est_color_1.unsqueeze(0)       # treat as [1,N,3]
        est_weight_ff = est_weight_1.unsqueeze(0)     # treat as [1,N]
    # Case C: per-pixel maps -> cannot feed kernel safely
    else:
        can_fast_forward = False

    if do_fast_forward and gt_images_t is not None and can_fast_forward:
        # 4) fast_forward 更新 Gaussian 颜色（内部做 no_grad + 形状/设备检查）
        gs_model.fast_forward(est_color_ff, est_weight_ff)
        # 5) 第二次渲染作为最终输出
        gs_attrs_2 = gs_model.get_batch_gaussian_attributes(V)
        result_2 = render_gs_batch(
            fov_x, fov_y,
            W, H,
            cam_view_mats, cam_proj_mats, cam_positions,
            bg_color, gs_attrs_2,
            # keep HWC for target_image, renderer will handle CHW internally
            target_image=gt_images_t,
        )
        image_final = result_2["color"]
        est_color = est_color_1
        est_weight = est_weight_1
    else:
        # 不做 fast_forward，就直接把第一次结果当最终结果
        image_final = image_1
        est_color = est_color_1
        est_weight = est_weight_1

    out = {
        "color": image_final,       # [V,H,W,3]
        "est_color": est_color,     # may be per-pixel or per-gaussian
        "est_weight": est_weight,   # may be per-pixel or per-gaussian
        "gs_model": gs_model,
    }
    if can_fast_forward:
        out["est_color_gs"] = est_color_ff  # [V,N,3] or [1,N,3]
        out["est_weight_gs"] = est_weight_ff  # [V,N] or [1,N]
    return out
