"""
Geometric Embedding Module (GEM) for self-supervised monocular depth.
Encodes physical geometric constraints into feature-level representations:
  1. Camera ray direction embedding (perspective geometry)
  2. Ground plane depth map (absolute scale anchor)
  3. Learnable scale recovery head

All three components have configurable on/off for ablation studies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeometricEmbedding(nn.Module):
    """Build geometric augmentation channels from camera parameters.

    Input:  B batch size, H height, W width
            K:   camera intrinsics [B, 4, 4]
            h_c: camera height above ground (scalar or [B])
            theta: camera pitch angle in radians (scalar or [B])

    Output: [B, C_geom, H, W] where C_geom depends on enabled components.
    """

    def __init__(self, enable_ray=True, enable_ground=True, ray_channels='all'):
        super().__init__()
        self.enable_ray = enable_ray
        self.enable_ground = enable_ground
        self.ray_channels = ray_channels  # 'all', 'rx_only', 'ry_only', 'rz_only', 'rx_rz'
        # Precompute 1/H and 1/W for coordinate scaling
        self._cache = {}  # (H,W) → grid

    def _ray_channel_count(self):
        if not self.enable_ray: return 0
        if self.ray_channels == 'all': return 3
        if self.ray_channels in ('rx_only', 'ry_only', 'rz_only'): return 1
        if self.ray_channels == 'rx_rz': return 2
        return 3

    def _get_grid(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._cache:
            v = torch.arange(0, H, device=device, dtype=torch.float32)  # [0, H-1]
            u = torch.arange(0, W, device=device, dtype=torch.float32)  # [0, W-1]
            uu, vv = torch.meshgrid(u, v, indexing='xy')
            self._cache[key] = (uu, vv)
        return self._cache[key]

    def build_ray_directions(self, K, H, W):
        """Compute normalized camera ray direction for each pixel.

        r_uv = normalize(K^{-1} · [u, v, 1]^T)
        Returns [B, 3, H, W], each channel is (rx, ry, rz) at scale H×W.
        """
        uu, vv = self._get_grid(H, W, K.device)
        B = K.shape[0]
        # Pixels → normalized coordinates
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=0)  # [3, H, W]
        uv1 = uv1.unsqueeze(0).expand(B, -1, -1, -1)           # [B, 3, H, W]
        uv1_flat = uv1.reshape(B, 3, -1)                        # [B, 3, H*W]

        # K^{-1} (batch matmul)
        inv_K = torch.inverse(K[:, :3, :3])                     # [B, 3, 3]
        rays_flat = torch.bmm(inv_K, uv1_flat)                  # [B, 3, H*W]

        # Normalize (safety: clamp min norm)
        norm = torch.norm(rays_flat, dim=1, keepdim=True).clamp(min=1e-8)
        rays_flat = rays_flat / norm

        rays = rays_flat.reshape(B, 3, H, W)
        return rays

    def build_ground_depth(self, h_c, theta, K, H, W):
        """Compute ground-plane depth hypothesis for every pixel.

        Under flat ground assumption (y=0 in camera frame):
            Z_ground(v) = h_c / tan(theta + atan((v - c_y) / f_y))

        h_c:  scalar or [B, 1] — camera height above ground (meters)
        theta: scalar or [B, 1] — camera pitch angle (radians, positive = looking down)
        K:      [B, 4, 4]
        Returns [B, 1, H, W]
        """
        B = K.shape[0]
        device = K.device

        # Handle scalar inputs
        if not isinstance(h_c, torch.Tensor):
            h_c = torch.full((B, 1), h_c, device=device, dtype=torch.float32)
        elif h_c.dim() == 0:
            h_c = h_c.view(1, 1).expand(B, 1)
        elif h_c.dim() == 1:
            h_c = h_c.view(B, 1)
        if not isinstance(theta, torch.Tensor):
            theta = torch.full((B, 1), theta, device=device, dtype=torch.float32)
        elif theta.dim() == 0:
            theta = theta.view(1, 1).expand(B, 1)
        elif theta.dim() == 1:
            theta = theta.view(B, 1)

        # Camera intrinsics
        c_y = K[:, 1, 2]  # principal point y (in pixel space)
        f_y = K[:, 1, 1]  # focal length y

        # Pixel v coordinates (top→bottom)
        _, vv = self._get_grid(H, W, device)
        vv = vv.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

        # Angle from optical axis to each row
        v_angle_offset = torch.atan((vv - c_y.view(B, 1, 1)) / f_y.view(B, 1, 1))
        total_angle = theta.view(B, 1, 1) + v_angle_offset

        # Depth = h_c / tan(total_angle)
        tan_val = torch.tan(total_angle).clamp(min=1e-6)
        depth_ground = (h_c.view(B, 1, 1) / tan_val).unsqueeze(1)  # [B, 1, H, W]

        # Clip to reasonable range
        depth_ground = depth_ground.clamp(0.1, 200.0)

        # Log-scale to compress dynamic range
        depth_ground = torch.log1p(depth_ground) / 5.0  # normalize

        return depth_ground

    def forward(self, K, h_c=None, theta=None, H=192, W=640):
        """Build geometric feature channels.

        Args:
            K: camera intrinsics [B, 4, 4]
            h_c: camera height (optional, for ground plane)
            theta: camera pitch (optional, for ground plane)
            H, W: output resolution

        Returns:
            Tensor [B, C_out, H, W] — geometric features to concat with RGB.
            C_out = 3 (ray) + 1 (ground) depending on enable flags.
            All channels normalized to approximately the same range as ground plane.
        """
        channels = []
        if self.enable_ray:
            rays = self.build_ray_directions(K, H, W)
            # Normalize ray directions to match ground plane range [~0.4, ~1.1]
            rays = (rays + 1.0) * (1.1 - 0.4) / 2.0 + 0.4  # → [0.4, 1.1]
            # Select specific channels to reduce redundancy
            if self.ray_channels == 'rx_only':
                rays = rays[:, 0:1]  # only r_x (horizontal, unique vs ground)
            elif self.ray_channels == 'ry_only':
                rays = rays[:, 1:2]  # only r_y (vertical, redundant with ground)
            elif self.ray_channels == 'rz_only':
                rays = rays[:, 2:3]  # only r_z (depth direction)
            elif self.ray_channels == 'rx_rz':
                rays = rays[:, [0, 2]]  # r_x + r_z
            # 'all': keep all 3
            channels.append(rays)
        if self.enable_ground and h_c is not None and theta is not None:
            gd = self.build_ground_depth(h_c, theta, K, H, W)
            channels.append(gd)

        if not channels:
            # Return empty tensor (shouldn't happen; defensive)
            return torch.zeros(K.shape[0], 0, H, W, device=K.device)

        return torch.cat(channels, dim=1)


class ScaleRecoveryHead(nn.Module):
    """Learned affine scale recovery from bottleneck features.

    Input:  bottleneck feature [B, C, H/32, W/32]
    Output: scale factor s [B, 1]   →  depth = s × raw_depth

    Architecture: GlobalAvgPool → MLP(512→128→1) → sigmoid → scale
    Neutral initialization: sigmoid output ≈ (1.0 - min_scale)/(max_scale - min_scale)
    so that initial scale ≈ 1.0 (identity).
    """

    def __init__(self, in_channels=512, hidden=128):
        super().__init__()
        self.min_scale = 0.1
        self.max_scale = 50.0
        # Target neutral value in sigmoid space
        neutral_scale = 1.0
        neutral_sigmoid = (neutral_scale - self.min_scale) / (self.max_scale - self.min_scale)
        # Inverse sigmoid to get the bias for the last linear layer
        # logit = ln(x/(1-x))
        import math
        neutral_logit = math.log(neutral_sigmoid / (1 - neutral_sigmoid + 1e-12) + 1e-12)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Initialize last linear layer to produce neutral_logit regardless of input
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, neutral_logit)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bottleneck_feat):
        """bottleneck_feat: [B, C, H', W']"""
        x = self.mlp(bottleneck_feat)
        s = self.sigmoid(x)  # [B, 1]，值在 [0, 1]
        scale = self.min_scale + s * (self.max_scale - self.min_scale)
        return scale


def get_kitti_params(batch_size=8, device='cuda'):
    """Return standard KITTI camera intrinsics and ground parameters.

    KITTI stereo: f_x ≈ 0.58*W, f_y ≈ 1.92*H, c_x ≈ 0.5*W, c_y ≈ 0.5*H
    Camera height ≈ 1.65m, pitch ≈ -0.03 rad (slightly looking down).
    """
    W, H = 640, 192
    K_np = np.array([
        [0.58 * W, 0, 0.5 * W, 0],
        [0, 1.92 * H, 0.5 * H, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    K = torch.from_numpy(K_np).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    h_c = 1.65   # meters (KITTI camera height)
    theta = -0.03  # radians (slight downward pitch)

    return K, h_c, theta
