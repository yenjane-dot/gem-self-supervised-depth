"""
GEM (Geometric Embedding Module) — unified training script.
Supports 7 ablation experiments via command-line flags.

Usage:
  Baseline:                python train_gem.py --exp e0
  Ray Only:                python train_gem.py --exp e1
  Ray+Ground:              python train_gem.py --exp e2
  Full GEM:                python train_gem.py --exp e3
  Ground Only:             python train_gem.py --exp e4
  Ground+Scale:            python train_gem.py --exp e5
  Scale Only:              python train_gem.py --exp e6
  RayOnly Norm:            python train_gem.py --exp e7
  RayGround Norm:          python train_gem.py --exp e8
  Ground + r_x Only:       python train_gem.py --exp e9
"""
import sys, os, time, argparse, gc, numpy as np
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = torch.device('cuda')

from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from networks.geometric_embedding import GeometricEmbedding, ScaleRecoveryHead, get_kitti_params
from layers import transformation_from_parameters, disp_to_depth, BackprojectDepth, Project3D, SSIM
from torch.utils.data import DataLoader
from datasets.kitti_custom_depth import KITTICustomDepthDataset
import torch.optim as optim

# ── Parse experiment ──
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='e0', choices=['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9'])
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--resume', type=str, default='')
args = parser.parse_args(sys.argv[1:])  # only pars our own args

EXP_CONFIG = {
    'e0': {'ray': False, 'ground': False, 'scale': False, 'name': 'Baseline', 'ray_ch': 'all'},
    'e1': {'ray': True,  'ground': False, 'scale': False, 'name': 'Ray_Only', 'ray_ch': 'all'},
    'e2': {'ray': True,  'ground': True,  'scale': False, 'name': 'Ray+Ground', 'ray_ch': 'all'},
    'e3': {'ray': True,  'ground': True,  'scale': True,  'name': 'Full_GEM', 'ray_ch': 'all'},
    'e4': {'ray': False, 'ground': True,  'scale': False, 'name': 'Ground_Only', 'ray_ch': 'all'},
    'e5': {'ray': False, 'ground': True,  'scale': True,  'name': 'Ground+Scale', 'ray_ch': 'all'},
    'e6': {'ray': False, 'ground': False, 'scale': True,  'name': 'Scale_Only', 'ray_ch': 'all'},
    'e7': {'ray': True,  'ground': False, 'scale': False, 'name': 'RayOnly_Norm', 'ray_ch': 'all'},
    'e8': {'ray': True,  'ground': True,  'scale': False, 'name': 'RayGround_Norm', 'ray_ch': 'all'},
    'e9': {'ray': True,  'ground': True,  'scale': False, 'name': 'Ground_rxOnly', 'ray_ch': 'rx_only'},
}
cfg = EXP_CONFIG[args.exp]

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

OUT_DIR = f'outputs/gem/{args.exp}_{cfg["name"].lower()}'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'=== GEM Experiment: {args.exp} ({cfg["name"]}) ===')
print(f'Ray={cfg["ray"]} Ground={cfg["ground"]} Scale={cfg["scale"]}')
print(f'Epochs: {args.epochs}, Output: {OUT_DIR}')

# ── Options (manual — bypass MonodepthOptions to avoid argparse conflicts) ──
class SimpleOpt: pass
opt = SimpleOpt()
opt.data_path = os.environ.get('KITTI_DATA_PATH', './kitti')
opt.height = 192; opt.width = 640
opt.num_scales = 4
opt.batch_size = 8
opt.min_depth = 0.1; opt.max_depth = 100.0
opt.disparity_smoothness = 1e-3
opt.num_epochs = args.epochs
opt.learning_rate = 1e-4
opt.scheduler_step_size = 15
opt.pose_model_input = 'pairs'
opt.frame_ids = [0, -1, 1]
opt.scales = [0, 1, 2, 3]

# ── Models ──
start_epoch = 0
encoder = ResnetEncoder(18, True).to(DEVICE)
depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4)).to(DEVICE)
pose_encoder = ResnetEncoder(18, True, 2).to(DEVICE)
pose_decoder = PoseDecoder(pose_encoder.num_ch_enc, 1, 2).to(DEVICE)

# GEM components
gem = GeometricEmbedding(enable_ray=cfg['ray'], enable_ground=cfg['ground'],
                         ray_channels=cfg.get('ray_ch', 'all')).to(DEVICE)
scale_head = ScaleRecoveryHead().to(DEVICE) if cfg['scale'] else None

# Determine encoder input channels
enc_in_channels = 3  # RGB
if cfg['ray']: enc_in_channels += gem._ray_channel_count()
if cfg['ground']: enc_in_channels += 1
if enc_in_channels > 3:
    # Replace encoder's first conv to accept extra channels
    old_conv = encoder.encoder.conv1
    new_conv = nn.Conv2d(enc_in_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                         padding=old_conv.padding, bias=old_conv.bias is not None).to(DEVICE)
    # Copy RGB weights from pretrained; extra channels → zero init
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight.clone()
        new_conv.weight[:, 3:] = 0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    encoder.encoder.conv1 = new_conv

# Handle resume
if args.resume:
    ck = torch.load(args.resume, map_location='cpu', weights_only=False)
    encoder.load_state_dict(ck['encoder'])
    depth_decoder.load_state_dict(ck['depth'])
    pose_encoder.load_state_dict(ck.get('pose_enc', ck.get('pose_encoder')))
    pose_decoder.load_state_dict(ck.get('pose_dec', ck.get('pose_decoder')))
    if 'gem' in ck: gem.load_state_dict(ck['gem'])
    if 'scale_head' in ck and scale_head is not None: scale_head.load_state_dict(ck['scale_head'])
    start_epoch = ck.get('epoch', -1) + 1
    print(f'Resumed from {args.resume}, start_epoch={start_epoch}')

params = (list(encoder.parameters()) + list(depth_decoder.parameters()) +
          list(pose_encoder.parameters()) + list(pose_decoder.parameters()) +
          list(gem.parameters()))
if scale_head is not None:
    params += list(scale_head.parameters())
print(f'Params: {sum(p.numel() for p in params)/1e6:.2f}M')

optimizer = optim.Adam(params, lr=opt.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_step_size, gamma=0.1)
for _ in range(start_epoch): scheduler.step()

# ── Dataset ──
with open('splits/custom_30drivers/train_files.txt') as f:
    train_lines = [l.strip() for l in f if l.strip()]

def filter_boundary(lines):
    max_frames = {}
    for l in lines:
        parts = l.strip().split()
        if len(parts) < 3: continue
        fi = int(parts[1]); side = parts[2] if len(parts) > 2 else 'l'
        key = (parts[0], side); max_frames[key] = max(max_frames.get(key, -1), fi)
    res = []
    for l in lines:
        parts = l.strip().split()
        if len(parts) < 3: continue
        fi = int(parts[1]); side = parts[2] if len(parts) > 2 else 'l'
        mx = max_frames.get((parts[0], side), fi)
        if fi >= 1 and fi + 1 <= mx and fi - 1 >= 0 and side in ['l','r','2','3']:
            res.append(l.strip())
    return res

train_filenames = filter_boundary(train_lines)
print(f'Train samples: {len(train_filenames)}')

ds = KITTICustomDepthDataset(
    opt.data_path, train_filenames, opt.height, opt.width,
    frame_idxs=[0, -1, 1], num_scales=opt.num_scales,
    is_train=True, img_ext='.png'
)
dl = DataLoader(ds, opt.batch_size, True, num_workers=0, pin_memory=True, drop_last=True)

# ── Loss helpers ──
backproj = BackprojectDepth(opt.batch_size, opt.height, opt.width).to(DEVICE)
proj3d = Project3D(opt.batch_size, opt.height, opt.width).to(DEVICE)
ssim_loss = SSIM()

def smooth_loss(d, c):
    gdx = torch.abs(d[:,:,:,:-1] - d[:,:,:,1:])
    gdy = torch.abs(d[:,:,:-1,:] - d[:,:,1:,:])
    gcx = torch.abs(c[:,:,:,:-1] - c[:,:,:,1:]).mean(1, keepdim=True)
    gcy = torch.abs(c[:,:,:-1,:] - c[:,:,1:,:]).mean(1, keepdim=True)
    return (gdx * torch.exp(-gcx)).mean() + (gdy * torch.exp(-gcy)).mean()

# ── Precompute camera params ──
K_static, h_c_static, theta_static = get_kitti_params(opt.batch_size, DEVICE)

# ── Training ──
print(f'\nTraining {args.epochs} epochs, {len(dl)} batches/epoch...')
total_t0 = time.time()

for epoch in range(start_epoch, args.epochs):
    encoder.train(); depth_decoder.train(); pose_encoder.train(); pose_decoder.train()
    gem.train()
    if scale_head is not None: scale_head.train()

    ep_t0 = time.time(); ep_loss = 0.0

    for batch_idx, inputs in enumerate(dl):
        B_act = inputs['color', 0, 0].shape[0]
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(DEVICE)

        # ── Geometric embedding ──
        rgb = inputs['color_aug', 0, 0]
        geom_feat = gem(K_static[:B_act], h_c_static, theta_static,
                        H=opt.height, W=opt.width)
        if geom_feat.shape[1] > 0:
            aug_rgb = torch.cat([rgb, geom_feat], dim=1)
        else:
            aug_rgb = rgb

        # ── Pose ──
        outputs = {}
        for fi in [-1, 1]:
            pin = torch.cat([inputs['color_aug', fi, 0], rgb], 1) if fi < 0 else torch.cat([rgb, inputs['color_aug', fi, 0]], 1)
            pf = pose_encoder(pin)
            ax, tr = pose_decoder([pf])
            outputs[('cam_T_cam', 0, fi)] = transformation_from_parameters(ax[:, 0], tr[:, 0], invert=(fi < 0))

        # ── Depth ──
        feats = encoder(aug_rgb)
        disp_outputs = depth_decoder(feats)

        # Apply scale head if enabled
        if scale_head is not None:
            bottleneck_feat = feats[4]  # lowest resolution
            scale_factor = scale_head(bottleneck_feat)  # [B, 1]
            # Scale the output disparity (higher scale → larger depth → smaller disparity)
            # Actually: depth = s * d_raw, so disp = 1/(s * d_raw) = disp_raw / s
            for scale in range(opt.num_scales):
                disp_outputs[('disp', scale)] = disp_outputs[('disp', scale)] / (scale_factor.view(-1, 1, 1, 1) + 1e-6)
        outputs.update(disp_outputs)

        # ── Warping ──
        for scale in range(opt.num_scales):
            disp = outputs[('disp', scale)]
            d_up = F.interpolate(disp, [opt.height, opt.width], mode='bilinear', align_corners=False)
            _, depth = disp_to_depth(d_up, opt.min_depth, opt.max_depth)
            outputs[('depth', 0, scale)] = depth

            for fi in [-1, 1]:
                T = outputs[('cam_T_cam', 0, fi)]
                pc = proj3d(backproj(depth, inputs[('inv_K', 0)]), inputs[('K', 0)], T)
                outputs[('sample', fi, scale)] = pc
                outputs[('color', fi, scale)] = F.grid_sample(inputs['color', fi, 0], pc, padding_mode='border', align_corners=True)

        # ── Loss ──
        total = torch.tensor(0.0, device=DEVICE)
        target = inputs['color', 0, 0]
        for scale in range(opt.num_scales):
            loss = torch.tensor(0.0, device=DEVICE)
            reproj, ident = [], []

            for fi in [-1, 1]:
                pred = outputs[('color', fi, scale)]
                diff = torch.abs(target - pred)
                reproj.append(0.85 * ssim_loss(pred, target).mean(1, True) + 0.15 * diff.mean(1, True))
            reproj = torch.cat(reproj, 1)

            for fi in [-1, 1]:
                p = inputs['color', fi, 0]
                d = torch.abs(target - p)
                ident.append(0.85 * ssim_loss(p, target).mean(1, True) + 0.15 * d.mean(1, True))
            ident_cat = torch.cat(ident, 1)
            ident_cat = ident_cat + torch.randn(ident_cat.shape, device=DEVICE) * 1e-5

            combined = torch.cat([ident_cat, reproj], 1)
            to_opt, _ = torch.min(combined, dim=1)
            loss += to_opt.mean()

            disp_sc = outputs[('disp', scale)]
            mean_disp = disp_sc.mean(2, True).mean(3, True)
            norm_disp = disp_sc / (mean_disp + 1e-7)
            color_f = F.interpolate(target, size=disp_sc.shape[-2:], mode='bilinear', align_corners=False)
            loss += opt.disparity_smoothness * smooth_loss(norm_disp, color_f) / (2 ** scale)
            total += loss

        losses = {'loss': total / opt.num_scales}
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        ep_loss += losses['loss'].item()

        if batch_idx % 500 == 0:
            dt = time.time() - ep_t0
            print(f'  [{batch_idx}] loss={losses["loss"].item():.4f} t={dt:.0f}s', flush=True)

    scheduler.step()
    ep_dt = time.time() - ep_t0
    ep_avg = ep_loss / len(dl)
    print(f'[{args.exp} Epoch {epoch+1}/{args.epochs}] avg_loss={ep_avg:.4f} time={ep_dt:.0f}s lr={optimizer.param_groups[0]["lr"]:.2e}')

    # Save checkpoint
    ckpt_dict = {
        'encoder': encoder.state_dict(), 'depth': depth_decoder.state_dict(),
        'pose_enc': pose_encoder.state_dict(), 'pose_dec': pose_decoder.state_dict(),
        'gem': gem.state_dict(), 'epoch': epoch, 'exp': args.exp, 'seed': SEED
    }
    if scale_head is not None:
        ckpt_dict['scale_head'] = scale_head.state_dict()
    ckpt_path = os.path.join(OUT_DIR, f'epoch{epoch}.pth')
    torch.save(ckpt_dict, ckpt_path)

    # Also save latest for eval
    torch.save(ckpt_dict, os.path.join(OUT_DIR, 'model.pth'))

total_dt = time.time() - total_t0
print(f'\n[{args.exp}] Training done in {total_dt/60:.1f}min')
print(f'Model saved to {OUT_DIR}/model.pth')
