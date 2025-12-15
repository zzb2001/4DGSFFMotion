from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
from plyfile import PlyData, PlyElement
import sys
sys.path.append('model/AnchorWarp4DGS')
sys.path.append('model/AnchorWarp4DGS/models')
from diff_renderer import GaussianAttributes
from _utils import Struct, inverse_sigmoid


class GaussianModel:
    """
    Simplified Gaussian Model for direct point cloud to Gaussian conversion.
    No blending, no basis decomposition - direct parameters only.
    """
    
    def __init__(self, model_config: Struct):
        self.model_config = model_config
        self._xyz = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._feature_dc = torch.empty(0)

        self.opacity_act = torch.sigmoid
        self.inv_opactity_act = inverse_sigmoid
        self.scaling_act = torch.exp
        self.inv_scaling_act = torch.log
        self.rotation_act = lambda x: torch.nn.functional.normalize(x, dim=-1)
    
    def initialize_from_points(self, points: torch.Tensor, colors: Optional[torch.Tensor] = None,
                              scales: Optional[torch.Tensor] = None, opacities: Optional[torch.Tensor] = None):
        """
        Initialize Gaussians directly from point cloud.
        
        Args:
            points: [N, 3] point cloud positions
            colors: [N, 3] optional point colors (default: random)
            scales: [N] or [N, 3] optional point scales (default: from config)
            opacities: [N, 1] optional point opacities (default: from config)
        """
        num_gaussian = points.shape[0]
        device = points.device
        
        # Initialize positions
        xyz = points.clone().detach()
        
        # Initialize opacity
        if opacities is None:
            opacity = self.inv_opactity_act(
                torch.full([num_gaussian, 1], self.model_config.init_opacity, 
                          dtype=torch.float32, device=device)
            )
        else:
            opacity = self.inv_opactity_act(opacities.clamp_min(1e-6))
        
        # Initialize scaling
        if scales is None:
            scaling = self.inv_scaling_act(
                torch.full([num_gaussian, 3], self.model_config.init_scaling, 
                          dtype=torch.float32, device=device)
            )
        else:
            if scales.ndim == 1:
                scales = scales.unsqueeze(-1).expand(-1, 3)
            scaling = self.inv_scaling_act(scales.clamp_min(1e-6))
        
        # Initialize rotation (identity quaternion)
        rotation = torch.zeros([num_gaussian, 4], dtype=torch.float32, device=device)
        rotation[:, 0] = 1.0
        
        # Initialize color/feature
        if colors is None:
            colors = torch.rand([num_gaussian, 3], dtype=torch.float32, device=device)
        
        # Convert RGB to SH coefficients (DC component only)
        from utils import rgb2sh0
        feature = rgb2sh0(colors).unsqueeze(1)  # [N, 1, 3]
        
        # Set as parameters
        self._xyz = Parameter(xyz.requires_grad_(True))
        self._opacity = Parameter(opacity.requires_grad_(True))
        self._scaling = Parameter(scaling.requires_grad_(True))
        self._rotation = Parameter(rotation.requires_grad_(True))
        self._feature_dc = Parameter(feature.requires_grad_(True))
    
    def get_attributes(self, blend_weight: Optional[torch.Tensor] = None) -> GaussianAttributes:
        """
        Get current Gaussian attributes.
        
        Args:
            blend_weight: ignored (for compatibility)
            
        Returns:
            GaussianAttributes with current Gaussian parameters
        """
        return GaussianAttributes(
            self._xyz,
            self.opacity_act(self._opacity),
            self.scaling_act(self._scaling),
            self.rotation_act(self._rotation),
            self._feature_dc
        )
    
    def get_batch_attributes(self, batch_size: int, blend_weight: Optional[torch.Tensor] = None) -> GaussianAttributes:
        """
        Get Gaussian attributes expanded to batch size.
        
        Args:
            batch_size: batch size to expand to
            blend_weight: ignored (for compatibility)
            
        Returns:
            GaussianAttributes with batch-expanded parameters
        """
        return GaussianAttributes(
            self._xyz.unsqueeze(0).expand(batch_size, -1, -1),
            self.opacity_act(self._opacity).unsqueeze(0).expand(batch_size, -1, -1),
            self.scaling_act(self._scaling).unsqueeze(0).expand(batch_size, -1, -1),
            self.rotation_act(self._rotation).unsqueeze(0).expand(batch_size, -1, -1),
            self._feature_dc.unsqueeze(0).expand(batch_size, -1, -1, -1)
        )
    
    # Legacy methods for compatibility (no-op)
    def get_attributes_torch(self, blend_weight: Optional[torch.Tensor] = None):
        """Legacy method - same as get_attributes"""
        return self.get_attributes(blend_weight)
    
    def get_batch_attributes_torch(self, batch_size: int, blend_weight: Optional[torch.Tensor] = None):
        """Legacy method - same as get_batch_attributes"""
        return self.get_batch_attributes(batch_size, blend_weight)
    
    def sparsity_loss(self, blend_weight: torch.Tensor = None):
        """No sparsity loss without blending"""
        return 0.0

    def training_params(self, args):
        """
        Get training parameters.
        
        Returns:
            Tuple of (gs_params, bs_params, adapter_params)
            - gs_params: Gaussian parameters
            - bs_params: empty (no basis)
            - adapter_params: empty (no adapter)
        """
        gs_params = [
            {'params': [self._xyz], 'lr': args.position_lr * args.scene_extent, "name": "xyz"},
            {'params': [self._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            {'params': [self._feature_dc], 'lr': args.feature_lr, "name": "f_dc"}
        ]
        bs_params = []  # No basis parameters
        adapter_params = []  # No adapter parameters
        return gs_params, bs_params, adapter_params

    @torch.no_grad()
    def save_ply(self, path: str):
        """Save Gaussians to PLY file"""
        xyz = self._xyz.cpu().numpy()
        num_gs = xyz.shape[0]
        normal = np.zeros_like(xyz)
        opacity = self._opacity.cpu().numpy()
        scaling = self._scaling.cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        f_dc = self._feature_dc.cpu().transpose(1, 2).flatten(start_dim=1).numpy()
        f_rest = np.zeros([num_gs, 45], dtype=xyz.dtype)

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'opacity']
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        
        dtype_full = [(attribute, 'f4') for attribute in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, opacity, scaling, rotation, f_dc, f_rest), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @torch.no_grad()
    def load_ply(self, path: str, train: bool = False):
        """Load Gaussians from PLY file"""
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"], dtype=np.float32),
            np.asarray(plydata.elements[0]["y"], dtype=np.float32),
            np.asarray(plydata.elements[0]["z"], dtype=np.float32)
        ), axis=1)
        num_gaussian = xyz.shape[0]

        opacity = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[..., np.newaxis]
        assert opacity.shape[0] == num_gaussian

        scaling_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scaling_names = sorted(scaling_names, key=lambda x: int(x.split('_')[-1]))
        scaling = np.zeros((num_gaussian, len(scaling_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scaling_names):
            scaling[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)
        assert scaling.shape[0] == num_gaussian

        rotation = np.stack((
            np.asarray(plydata.elements[0]["rot_0"], dtype=np.float32),
            np.asarray(plydata.elements[0]["rot_1"], dtype=np.float32),
            np.asarray(plydata.elements[0]["rot_2"], dtype=np.float32),
            np.asarray(plydata.elements[0]["rot_3"], dtype=np.float32)
        ), axis=1)
        assert rotation.shape[0] == num_gaussian

        feature_dc = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"], dtype=np.float32),
            np.asarray(plydata.elements[0]["f_dc_1"], dtype=np.float32),
            np.asarray(plydata.elements[0]["f_dc_2"], dtype=np.float32)
        ), axis=1).reshape([num_gaussian, 3, 1])
        assert feature_dc.shape[0] == num_gaussian

        self._xyz = torch.from_numpy(xyz).cuda()
        self._opacity = torch.from_numpy(opacity).cuda()
        self._scaling = torch.from_numpy(scaling).cuda()
        self._rotation = torch.from_numpy(rotation).cuda()
        self._feature_dc = torch.from_numpy(feature_dc).transpose(1, 2).contiguous().cuda()

        if train:
            self._xyz = Parameter(self._xyz.requires_grad_(True))
            self._opacity = Parameter(self._opacity.requires_grad_(True))
            self._scaling = Parameter(self._scaling.requires_grad_(True))
            self._rotation = Parameter(self._rotation.requires_grad_(True))
            self._feature_dc = Parameter(self._feature_dc.requires_grad_(True))

    @torch.no_grad()
    def capture(self):
        """Capture current state"""
        return (
            self._xyz,
            self._opacity,
            self._scaling,
            self._rotation,
            self._feature_dc
        )
    
    @torch.no_grad()
    def restore(self, params):
        """Restore from captured state"""
        self._xyz.copy_(params[0])
        self._opacity.copy_(params[1])
        self._scaling.copy_(params[2])
        self._rotation.copy_(params[3])
        self._feature_dc.copy_(params[4])

    @torch.no_grad()
    def restore_from_optimizer(self, optimizer: torch.optim.Optimizer):
        """Restore from optimizer state"""
        for group in optimizer.param_groups:
            if group['name'] == 'xyz':
                self._xyz = group['params'][0]
            elif group['name'] == 'opacity':
                self._opacity = group['params'][0]
            elif group['name'] == 'scaling':
                self._scaling = group['params'][0]
            elif group['name'] == 'rotation':
                self._rotation = group['params'][0]
            elif group['name'] == 'f_dc':
                self._feature_dc = group['params'][0]
