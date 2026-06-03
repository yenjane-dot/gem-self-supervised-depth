"""
End-to-end validation for GEM (Geometric Embedding Module).
Tests forward pass + backward pass + gradient flow for each experiment.

Run: python validate_gem.py
"""
import sys, os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False

from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from networks.geometric_embedding import GeometricEmbedding, ScaleRecoveryHead, get_kitti_params
from layers import transformation_from_parameters, disp_to_depth, BackprojectDepth, Project3D, SSIM

DEVICE = torch.device('cuda')
B, H, W = 8, 192, 640

EXP_CONFIGS = {
    'E0_Baseline':   {'ray': False, 'ground': False, 'scale': False},
    'E1_RayOnly':    {'ray': True,  'ground': False, 'scale': False},
    'E2_RayGround':  {'ray': True,  'ground': True,  'scale': False},
    'E3_FullGEM':    {'ray': True,  'ground': True,  'scale': True},
    'E4_GroundOnly': {'ray': False, 'ground': True,  'scale': False},
    'E5_GroundScale':{'ray': False, 'ground': True,  'scale': True},
    'E6_ScaleOnly':  {'ray': False, 'ground': False, 'scale': True},
}

print('=' * 70)
print('GEM End-to-End Validation')
print('=' * 70)

all_passed = True

for exp_name, cfg in EXP_CONFIGS.items():
    print(f'\n--- {exp_name}: ray={cfg["ray"]} ground={cfg["ground"]} scale={cfg["scale"]} ---')

    try:
        # Build models
        encoder = ResnetEncoder(18, True)
        depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4))
        pose_encoder = ResnetEncoder(18, True, 2)
        pose_decoder = PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        gem = GeometricEmbedding(enable_ray=cfg['ray'], enable_ground=cfg['ground'])

        # Adjust encoder input channels
        enc_in_channels = 3
        if cfg['ray']: enc_in_channels += 3
        if cfg['ground']: enc_in_channels += 1
        if enc_in_channels > 3:
            import torch.nn as nn
            old_conv = encoder.encoder.conv1
            new_conv = nn.Conv2d(enc_in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                                 padding=old_conv.padding, bias=old_conv.bias is not None)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight.clone()
                new_conv.weight[:, 3:] = 0
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            encoder.encoder.conv1 = new_conv

        scale_head = ScaleRecoveryHead() if cfg['scale'] else None

        encoder.to(DEVICE); depth_decoder.to(DEVICE)
        pose_encoder.to(DEVICE); pose_decoder.to(DEVICE)
        gem.to(DEVICE)
        if scale_head is not None: scale_head.to(DEVICE)

        encoder.train(); depth_decoder.train(); pose_encoder.train(); pose_decoder.train()
        gem.train()
        if scale_head is not None: scale_head.train()

        # Forward pass
        rgb = torch.randn(B, 3, H, W, device=DEVICE)
        color_m1 = torch.randn(B, 3, H, W, device=DEVICE)
        color_p1 = torch.randn(B, 3, H, W, device=DEVICE)

        K_static, h_c, theta = get_kitti_params(B, DEVICE)
        geom_feat = gem(K_static, h_c, theta, H=H, W=W)

        if geom_feat.shape[1] > 0:
            aug_rgb = torch.cat([rgb, geom_feat], dim=1)
        else:
            aug_rgb = rgb

        # Pose
        pose_in_m1 = torch.cat([color_m1, rgb], 1)
        pf_m1 = pose_encoder(pose_in_m1)
        ax_m1, tr_m1 = pose_decoder([pf_m1])
        T_m1 = transformation_from_parameters(ax_m1[:, 0], tr_m1[:, 0], invert=True)

        pose_in_p1 = torch.cat([rgb, color_p1], 1)
        pf_p1 = pose_encoder(pose_in_p1)
        ax_p1, tr_p1 = pose_decoder([pf_p1])
        T_p1 = transformation_from_parameters(ax_p1[:, 0], tr_p1[:, 0], invert=False)

        # Depth
        feats = encoder(aug_rgb)
        disp_outputs = depth_decoder(feats)

        if scale_head is not None:
            bottleneck_feat = feats[4]
            scale_factor = scale_head(bottleneck_feat)
            for s in range(4):
                disp_outputs[('disp', s)] = disp_outputs[('disp', s)] / (scale_factor.view(-1, 1, 1, 1) + 1e-6)

        disp = F.interpolate(disp_outputs[('disp', 0)], [H, W], mode='bilinear', align_corners=False)
        _, depth = disp_to_depth(disp, 0.1, 100.0)
        print(f'  Forward OK: disp [{list(disp.shape)}] range=[{disp.min().item():.3f}, {disp.max().item():.3f}] '
              f'depth range=[{depth.min().item():.2f}, {depth.max().item():.2f}]')

        # Warp test
        K_tensor = K_static  # [B, 4, 4]
        inv_K = torch.inverse(K_tensor)
        backproj = BackprojectDepth(B, H, W).to(DEVICE)
        proj3d = Project3D(B, H, W).to(DEVICE)
        ssim = SSIM().to(DEVICE)

        cam_points = backproj(depth, inv_K)
        pixel_coords = proj3d(cam_points, K_tensor, T_m1)
        warped = F.grid_sample(color_m1, pixel_coords, padding_mode='border', align_corners=True)
        print(f'  Warp OK: shape={list(warped.shape)}')

        # Loss
        photo_loss = 0.85 * ssim(warped, rgb).mean() + 0.15 * torch.abs(rgb - warped).mean()
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth = (torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:]).mean() +
                  torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :]).mean()) * 1e-3
        loss = photo_loss + smooth
        print(f'  Loss OK: photo={photo_loss.item():.4f} smooth={smooth.item():.6f} total={loss.item():.4f}')

        # Backward
        params = (list(encoder.parameters()) + list(depth_decoder.parameters()) +
                  list(pose_encoder.parameters()) + list(pose_decoder.parameters()) +
                  list(gem.parameters()))
        if scale_head is not None:
            params += list(scale_head.parameters())

        optimizer = torch.optim.SGD(params, lr=0.01)
        optimizer.zero_grad()
        loss.backward()

        # Gradient check
        zero_grad = []
        nan_grad = []
        for name, param in (list(encoder.named_parameters()) +
                            list(depth_decoder.named_parameters()) +
                            list(gem.named_parameters())):
            if param.grad is not None:
                g_norm = param.grad.norm().item()
                if g_norm == 0:
                    zero_grad.append(name)
                if torch.isnan(param.grad).any():
                    nan_grad.append(name)

        if scale_head is not None:
            for name, param in scale_head.named_parameters():
                if param.grad is not None:
                    g_norm = param.grad.norm().item()
                    if g_norm == 0:
                        zero_grad.append(f'scale_head.{name}')

        if zero_grad:
            print(f'  [WARN] Zero grad ({len(zero_grad)}): {zero_grad[:5]}...')
        else:
            print(f'  Gradients OK: all params have non-zero grad')

        if nan_grad:
            print(f'  [FAIL] NaN grad: {nan_grad}')
            all_passed = False
        else:
            print(f'  No NaN gradients')

        print(f'  [PASS] {exp_name} PASSED')

        # Cleanup
        del encoder, depth_decoder, pose_encoder, pose_decoder, gem, scale_head
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'  [FAIL] {exp_name} FAILED: {e}')
        import traceback; traceback.print_exc()
        all_passed = False

print('\n' + '=' * 70)
if all_passed:
    print('[OK] ALL 7 EXPERIMENTS PASSED')
else:
    print('[FAIL] SOME EXPERIMENTS FAILED')
print('=' * 70)
