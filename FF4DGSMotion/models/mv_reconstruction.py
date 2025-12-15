import torch
import numpy as np
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from nvdiffrast import torch as dr
# from fused_ssim import fused_ssim   #实现使用了11x11大小可变的卷积核。其权重已硬编码​​，这也是其速度更快的另一个原因。此实现目前仅支持二维图像，但通道数和批量大小可变
import lpips
import math

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_renderer import GaussianAttributes
from _utils import Struct, l1_loss, ssim, get_expon_lr_func, create_window, _ssim
from torchvision.utils import save_image, make_grid
from plyfile import PlyData, PlyElement
import os
# from .binding import BindingModel


def compute_face_normal(verts: torch.Tensor, faces: torch.Tensor):
    faces = faces.to(torch.int64)
    i0 = faces[..., 0]
    i1 = faces[..., 1]
    i2 = faces[..., 2]

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=-1)
    return face_normals


def vis_shading_mesh(
    glctx: dr.RasterizeGLContext,
    width: int, height: int,
    cam_proj_mats: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    verts_pad = torch.nn.functional.pad(vertices, [0, 1], value=1.0)
    verts_clip = torch.matmul(verts_pad, cam_proj_mats.transpose(-1, -2))

    rast_out, _ = dr.rasterize(glctx, verts_clip, faces.to(torch.int32), resolution=(height, width))

    face_id = rast_out[..., 3:] # face_id == 0 when no triangle
    mask = face_id > 0
    face_id = torch.clamp_min(face_id - 1, 0)

    face_normals = compute_face_normal(vertices, faces) # [B, F, 3]
    normal_map = torch.gather(
        input=face_normals[:, None, None, :, :].expand(-1, height, width, -1, -1), # [B, H, W, F, 3],
        dim=-2,
        index=face_id[:, :, :, :, None].expand(-1, -1, -1, -1, 3).long() # [B, H, W, 1, 3]
    ).squeeze(-2) # [B, H, W, 3]

    shading = (normal_map * 0.5 + 0.5) * mask
    return shading, mask


def render_gs_batch(
    fov_x: float, fov_y: float,
    width: int, height: int,
    cam_view_mats: torch.Tensor,
    cam_proj_mats: torch.Tensor,
    cam_positions: torch.Tensor,
    bg_colors: torch.Tensor,
    gs: GaussianAttributes,
    target_image: torch.Tensor = None,
    sh_degree: int = 0,
    scaling_modifier: float = 1.0,
    debug_save_points: bool = False
) -> dict[str, torch.Tensor]:

    screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
    if screenspace_points.requires_grad: # requires_grad == False when inference
        screenspace_points.retain_grad()

    tanfovx = math.tan(fov_x * 0.5)
    tanfovy = math.tan(fov_y * 0.5)

    batch_size = gs.xyz.shape[0]

    cam_view_mats = cam_view_mats.reshape(batch_size, 4, 4).transpose(1, 2)
    cam_proj_mats = cam_proj_mats.reshape(batch_size, 4, 4).transpose(1, 2)
    cam_positions = cam_positions.reshape(batch_size, 3)
    bg_colors = bg_colors.reshape(batch_size, 3)

    bs = gs.xyz.shape[0]
    color_list = [] 
    alpha_list = [] 
    est_color_list = [] 
    est_weight_list = [] 
    radii_list = []
    for i in range(bs):
        cam_view_mat = cam_view_mats[i].contiguous()
        cam_proj_mat = cam_proj_mats[i].contiguous()
        cam_position = cam_positions[i].contiguous()
        bg_color = bg_colors[i].contiguous()

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=cam_view_mat,
            projmatrix=cam_proj_mat,
            sh_degree=sh_degree,
            campos=cam_position,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        if debug_save_points:
            # 将当前视角的高斯中心点保存为 PLY
            pts = gs.xyz[i].detach().cpu().numpy()  # [N,3]
            if pts.ndim == 2 and pts.shape[1] >= 3:
                vertex = np.empty(pts.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4')])
                vertex['x'] = pts[:,0]
                vertex['y'] = pts[:,1]
                vertex['z'] = pts[:,2]
                ply = PlyData([PlyElement.describe(vertex, 'vertex')])
                out_path = os.path.join('debug/plys', f'points_cam_after_{i:02d}.ply')
                ply.write(out_path)

            
        color, alpha, est_color, est_weight, radii = rasterizer(
            means3D=gs.xyz[i],
            means2D=screenspace_points[i],
            shs=gs.sh[i],
            colors_precomp=None,
            opacities=gs.opacity[i],
            scales=gs.scaling[i],
            rotations=gs.rotation[i],
            cov3D_precomp=None,
            target_image=target_image[i] if target_image is not None else None  #target_image[i] [3, 378, 504]
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



    def __init__(self,
        fov_x: float, fov_y: float,
        width: int, height: int,

        gaussian_model: BindingModel,
        batch_size: int,
        recon_config: Struct,
        tb_writer: SummaryWriter = None
    ):
        self.batch_size = batch_size
        self.recon_config = recon_config
        self.tb_writer = tb_writer
        self.gaussian_model = gaussian_model
        self.bg_color = torch.tensor(self.recon_config.bg_color, dtype=torch.float32, device='cuda')

        self.fov_x, self.fov_y = fov_x, fov_y
        self.width, self.height = width, height

        gs_params, bs_params, adapter_params = gaussian_model.training_params(self.recon_config)
        params = gs_params + bs_params + adapter_params
        self.optimizer = Adam(params=params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.recon_config.position_lr * self.recon_config.scene_extent,
            lr_final=self.recon_config.position_lr_final * self.recon_config.scene_extent,
            lr_delay_mult=self.recon_config.position_lr_delay_mult,
            max_steps=self.recon_config.position_lr_max_steps
        )

        self.iteration = 0
        self.perceptual_model = None
        self.global_lr_scale = 1.0


    def perceptual_loss(self, image: torch.Tensor, gt_image: torch.Tensor):
        if self.iteration > 20000: # hard coded
            if self.perceptual_model is None: # load lpips model
                self.perceptual_model = lpips.LPIPS(net='vgg').cuda()
            image = image * 2.0 - 1.0
            gt_image = gt_image * 2.0 - 1.0
            return self.perceptual_model(image, gt_image).mean()
        else:
            return 0.0
    

    def ssim_loss(self, image: torch.Tensor, gt_image: torch.Tensor):
        return 1.0 - fused_ssim(image, gt_image)
    

    def update_learning_rate(self):
        xyz_lr = self.xyz_scheduler_args(self.iteration) * self.recon_config.scene_extent
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz": 
                param_group['lr'] = xyz_lr * self.global_lr_scale
            elif param_group["name"] == "opacity": 
                param_group['lr'] = self.recon_config.opacity_lr * self.global_lr_scale
            elif param_group["name"] == "scaling": 
                param_group['lr'] = self.recon_config.scaling_lr * self.global_lr_scale
            elif param_group["name"] == "rotation": 
                param_group['lr'] = self.recon_config.rotation_lr * self.global_lr_scale
            elif param_group["name"] == "f_dc": 
                param_group['lr'] = self.recon_config.feature_lr * self.global_lr_scale
            elif param_group["name"] == "xyz_b": 
                param_group['lr'] = xyz_lr * self.recon_config.position_b_lr_scale * self.global_lr_scale
            elif param_group["name"] == "rotation_b": 
                param_group['lr'] = self.recon_config.rotaion_b_lr_scale * self.recon_config.rotation_lr * self.global_lr_scale
            elif param_group["name"] == "f_dc_b": 
                param_group['lr'] = self.recon_config.feature_b_lr_scale * self.recon_config.feature_lr * self.global_lr_scale
            elif param_group["name"] == "weight_module": 
                param_group['lr'] = self.recon_config.weight_module_lr * self.global_lr_scale

    def step(self, 
        gt_image: torch.Tensor, 
        template_mesh: torch.Tensor,
        blend_weight: torch.Tensor,
        cam_view_mats: torch.Tensor,
        cam_proj_mats: torch.Tensor,
        cam_positions: torch.Tensor
    ):
        batch_size = gt_image.shape[0]
        self.update_learning_rate()
        gt_rgb, gt_mask = gt_image[:, :3], gt_image[:, 3:4]
        
        # background processing
        if self.recon_config.random_bg_color: bg_color = torch.rand([batch_size, 3, 1, 1], dtype=torch.float32, device='cuda')
        else: bg_color = self.bg_color.reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1)
        gt_rgb = gt_rgb * gt_mask + bg_color * (1.0 - gt_mask)

        # blend & bind
        blend_weight = None if self.iteration < self.recon_config.blend_start_iter else blend_weight
        gaussian = self.gaussian_model.gaussian_deform_batch(template_mesh, blend_weight)
        self.optimizer.zero_grad(set_to_none = True)

        # batch render
        result_pkg = render_gs_batch(
            self.fov_x, self.fov_y,
            self.width, self.height,
            cam_view_mats, cam_proj_mats, cam_positions,
            bg_color, gaussian, gt_image
        )
        image, alpha, est_color, est_weight = result_pkg["color"], result_pkg["alpha"], result_pkg["est_color"], result_pkg["est_weight"]

        # loss
        l1_loss_val = l1_loss(image, gt_rgb) if self.recon_config.lambda_l1 > 0.0 else 0.0
        ssim_loss_val = self.ssim_loss(image, gt_rgb) if self.recon_config.lambda_ssim > 0.0 else 0.0
        lpips_loss_val = self.perceptual_loss(image, gt_rgb) if self.recon_config.lambda_lpips > 0.0 else 0.0
        alpha_loss_val = l1_loss(alpha, gt_mask) if self.recon_config.lambda_alpha > 0.0 else 0.0
        sparsity_loss_val = self.gaussian_model.sparsity_loss(blend_weight) if self.recon_config.lambda_sparsity > 0.0 else 0.0
        orth_loss_val = self.gaussian_model.orth_loss() if self.recon_config.lambda_orth > 0.0 and blend_weight is not None else 0.0
        total_loss = self.recon_config.lambda_l1 * l1_loss_val + \
            self.recon_config.lambda_ssim * ssim_loss_val + \
            self.recon_config.lambda_lpips * lpips_loss_val + \
            self.recon_config.lambda_alpha * alpha_loss_val + \
            self.recon_config.lambda_sparsity * sparsity_loss_val +\
            self.recon_config.lambda_orth * orth_loss_val
        
        # optimize
        total_loss.backward()
        self.optimizer.step()

        # fast forward
        if self.recon_config.use_fast_forward:
            self.gaussian_model.fast_forward(est_color, est_weight)

        # log
        with torch.no_grad():
            if self.tb_writer is not None:
                if self.recon_config.lambda_l1 > 0.0:
                    self.tb_writer.add_scalar('train_loss/l1_loss', l1_loss_val.item(), self.iteration)
                if self.recon_config.lambda_ssim > 0.0:
                    self.tb_writer.add_scalar('train_loss/ssim_loss', ssim_loss_val.item(), self.iteration)
                if self.recon_config.lambda_lpips > 0.0:
                    self.tb_writer.add_scalar('train_loss/lpips_loss', lpips_loss_val.item(), self.iteration)
                if self.recon_config.lambda_alpha > 0.0:
                    self.tb_writer.add_scalar('train_loss/alpha_loss', alpha_loss_val.item(), self.iteration)
                if self.recon_config.lambda_alpha > 0.0:
                    self.tb_writer.add_scalar('train_loss/orth_loss', orth_loss_val.item(), self.iteration)
                self.tb_writer.add_scalar('train_loss/total_loss', total_loss.item(), self.iteration)
            
            if self.iteration % 3000 == 0:
                self.tb_writer.add_image('pred', image, self.iteration, dataformats='NCHW')
                self.tb_writer.add_image('gt', gt_rgb, self.iteration, dataformats='NCHW')

                shading, mask = vis_shading_mesh(self.gaussian_model.glctx, self.width, self.height, cam_proj_mats, template_mesh, self.gaussian_model.template_faces)
                shading, mask = shading.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)
                mesh_images = (gt_rgb * 0.3 + shading * 0.7) * mask + gt_rgb * (~mask)
                self.tb_writer.add_image('mesh', mesh_images, self.iteration, dataformats='NCHW')

        self.iteration += batch_size