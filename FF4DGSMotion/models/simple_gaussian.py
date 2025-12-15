# ============================================
# simple_gaussian.py  ——  PATCHED VERSION
# ============================================
import torch
from torch.nn import Parameter
import sys
sys.path.append('FF4DGSMotion')
sys.path.append('FF4DGSMotion/models')
from diff_renderer import GaussianAttributes
from _utils import Struct, rgb2sh0
from diff_gaussian_rasterization import fast_forward as cuda_fast_forward
from .gaussian import GaussianModel


class SimpleGaussianModel(GaussianModel):

    def __init__(self, model_config: Struct):
        super().__init__(model_config)

        # [PATCH] 永久记录哪些 Gaussians 在首次 FF 后已经被初始化颜色
        self.ff_color_fixed = False

    # ----------------------------------------------------
    # initialize_from_points —— 初始化模型
    # ----------------------------------------------------
    def initialize_from_points(self, points, colors=None, scales=None):
        num_gaussian = points.shape[0]
        dev = points.device

        # xyz
        xyz = points.clone().detach()

        # opacity
        opacity = self.inv_opactity_act(
            torch.full([num_gaussian, 1], self.model_config.init_opacity,
                       dtype=torch.float32, device=dev)
        )

        # scaling
        if scales is None:
            scaling = self.inv_scaling_act(
                torch.full([num_gaussian, 3], self.model_config.init_scaling,
                           dtype=torch.float32, device=dev)
            )
        else:
            if scales.ndim == 1:
                scales = scales.unsqueeze(-1).expand(-1, 3)
            scaling = self.inv_scaling_act(scales)

        # rotation
        rotation = torch.zeros([num_gaussian, 4], dtype=torch.float32, device=dev)
        rotation[:, 0] = 1.0

        # feature color
        if colors is None:
            colors = torch.rand([num_gaussian, 3], device=dev)
        feature = rgb2sh0(colors).unsqueeze(1)

        # bases
        nb = self.model_config.num_basis_in
        xyz_b = torch.zeros([nb, num_gaussian, 3], dtype=torch.float32, device=dev)
        feature_b = torch.zeros([nb, num_gaussian, 1, 3], dtype=torch.float32, device=dev)
        rotation_b = torch.zeros([nb, num_gaussian, 4], dtype=torch.float32, device=dev)

        # register parameters
        self._xyz = Parameter(xyz)
        self._opacity = Parameter(opacity)
        self._scaling = Parameter(scaling)
        self._rotation = Parameter(rotation)
        self._feature_dc = Parameter(feature)

        self._xyz_b = Parameter(xyz_b)
        self._feature_b = Parameter(feature_b)
        self._rotation_b = Parameter(rotation_b)

    # ----------------------------------------------------
    # fast_forward —— 只在第一次调用时写入颜色
    # ----------------------------------------------------
    @torch.no_grad()
    def fast_forward(self, est_color, est_weight):

        # [PATCH] 如果颜色已经初始化过，后续不再更新
        if self.ff_color_fixed:
            return

        if est_color.ndim == 3:
            est_color = est_color.sum(dim=0)
        if est_weight.ndim == 2:
            est_weight = est_weight.sum(dim=0)

        N = self._feature_dc.shape[0]
        assert est_color.shape == (N, 3)
        assert est_weight.shape[0] == N

        feature_dc = self._feature_dc.reshape(-1, 3)

        cuda_fast_forward(
            0.05,       # weight threshold
            est_color.contiguous(),
            est_weight.clamp_min(0).contiguous(),
            torch.zeros(N, device=feature_dc.device, dtype=torch.bool),
            feature_dc,
        )

        # [PATCH] 写一次以后标记永久不再更新
        self.ff_color_fixed = True

    # ----------------------------------------------------
    # attributes
    # ----------------------------------------------------
    def get_gaussian_attributes(self, *args, **kwargs):
        return GaussianAttributes(
            self._xyz,
            self.opacity_act(self._opacity),
            self.scaling_act(self._scaling),
            self.rotation_act(self._rotation),
            self._feature_dc,
        )

    def get_batch_gaussian_attributes(self, B, *args, **kwargs):
        return GaussianAttributes(
            self._xyz.unsqueeze(0).expand(B, -1, -1),
            self.opacity_act(self._opacity).unsqueeze(0).expand(B, -1, -1),
            self.scaling_act(self._scaling).unsqueeze(0).expand(B, -1, -1),
            self.rotation_act(self._rotation).unsqueeze(0).expand(B, -1, -1),
            self._feature_dc.unsqueeze(0).expand(B, -1, -1, -1),
        )
