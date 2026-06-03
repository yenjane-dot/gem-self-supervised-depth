"""
Ground Only — 20 epoch training.
Ground plane depth map as extra input channel.
Ray embedding and scale head completely removed.
Saves every 2 epochs for eval.
"""
import sys, os, time, argparse, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))
sys.stdout.reconfigure(line_buffering=True)

import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = torch.device('cuda')
torch.backends.cudnn.benchmark = True

from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from networks.geometric_embedding import GeometricEmbedding, get_kitti_params
from layers import transformation_from_parameters, disp_to_depth, BackprojectDepth, Project3D, SSIM
from torch.utils.data import DataLoader
from datasets.kitti_custom_depth import KITTICustomDepthDataset
import torch.optim as optim

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

OUT_DIR = 'outputs/gem/e4_ground_only_20ep'
os.makedirs(OUT_DIR, exist_ok=True)

# Redirect stdout to log file for orphaned execution
log_file = open(os.path.join(OUT_DIR, 'train_restart.log'), 'a')
sys.stdout = log_file
sys.stderr = log_file

print(f'Seed={SEED}, Output={OUT_DIR}', flush=True)

# ── Options ──
class O: pass
opt = O()
opt.data_path = os.environ.get('KITTI_DATA_PATH', './kitti'); opt.height = 192; opt.width = 640
opt.num_scales = 4; opt.batch_size = 8
opt.min_depth = 0.1; opt.max_depth = 100.0
opt.disparity_smoothness = 1e-3
opt.num_epochs = 20
opt.learning_rate = 1e-4; opt.scheduler_step_size = 15

# ── Models ──
encoder = ResnetEncoder(18, True).to(DEVICE)
depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4)).to(DEVICE)
pose_encoder = ResnetEncoder(18, True, 2).to(DEVICE)
pose_decoder = PoseDecoder(pose_encoder.num_ch_enc, 1, 2).to(DEVICE)
gem = GeometricEmbedding(enable_ray=False, enable_ground=True).to(DEVICE)

# 4-channel input (3 RGB + 1 ground)
old_conv = encoder.encoder.conv1
new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                     stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None).to(DEVICE)
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight.clone()
    new_conv.weight[:, 3:] = 0
    if old_conv.bias is not None: new_conv.bias.copy_(old_conv.bias)
encoder.encoder.conv1 = new_conv

start_epoch = 0
resume_path = os.path.join(OUT_DIR, 'epoch_latest.pth')
if os.path.exists(resume_path):
    ck = torch.load(resume_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(ck['encoder'])
    depth_decoder.load_state_dict(ck['depth'])
    pose_encoder.load_state_dict(ck['pose_enc'])
    pose_decoder.load_state_dict(ck['pose_dec'])
    gem.load_state_dict(ck['gem'])
    start_epoch = ck['epoch'] + 1
    print(f'Resumed from epoch {start_epoch}', flush=True)

params = (list(encoder.parameters()) + list(depth_decoder.parameters()) +
          list(pose_encoder.parameters()) + list(pose_decoder.parameters()) +
          list(gem.parameters()))
print(f'Params: {sum(p.numel() for p in params)/1e6:.2f}M', flush=True)

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
        max_frames[(parts[0], side)] = max(max_frames.get((parts[0], side), -1), fi)
    res = []
    for l in lines:
        parts = l.strip().split()
        if len(parts) < 3: continue
        fi = int(parts[1]); side = parts[2] if len(parts) > 2 else 'l'
        mx = max_frames.get((parts[0], side), fi)
        if fi >= 1 and fi + 1 <= mx and fi - 1 >= 0 and side in ['l','r','2','3']:
            res.append(l.strip())
    return res

train_files = filter_boundary(train_lines)
print(f'Train: {len(train_files)} samples', flush=True)

ds = KITTICustomDepthDataset(opt.data_path, train_files, opt.height, opt.width,
                              frame_idxs=[0, -1, 1], num_scales=opt.num_scales,
                              is_train=True, img_ext='.png')
dl = DataLoader(ds, opt.batch_size, True, num_workers=0, pin_memory=True, drop_last=True)
print(f'Batches: {len(dl)}', flush=True)

backproj = BackprojectDepth(opt.batch_size, opt.height, opt.width).to(DEVICE)
proj3d = Project3D(opt.batch_size, opt.height, opt.width).to(DEVICE)
ssim = SSIM().to(DEVICE)
K_static, h_c, theta = get_kitti_params(opt.batch_size, DEVICE)

def smooth_loss(d, c):
    gdx = torch.abs(d[:,:,:,:-1] - d[:,:,:,1:])
    gdy = torch.abs(d[:,:,:-1,:] - d[:,:,1:,:])
    gcx = torch.abs(c[:,:,:,:-1] - c[:,:,:,1:]).mean(1, keepdim=True)
    gcy = torch.abs(c[:,:,:-1,:] - c[:,:,1:,:]).mean(1, keepdim=True)
    return (gdx * torch.exp(-gcx)).mean() + (gdy * torch.exp(-gcy)).mean()

print(f'\nTraining 20 epochs...', flush=True)
total_t0 = time.time()

for epoch in range(start_epoch, opt.num_epochs):
    encoder.train(); depth_decoder.train(); pose_encoder.train(); pose_decoder.train()
    gem.train()
    ep_t0 = time.time(); ep_loss = 0.0

    for bi, inputs in enumerate(dl):
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(DEVICE)

        rgb = inputs['color_aug', 0, 0]
        with torch.no_grad():
            geo = gem(K_static, h_c, theta, H=opt.height, W=opt.width)
        aug = torch.cat([rgb, geo], dim=1)

        # Pose
        outputs = {}
        for fi in [-1, 1]:
            pin = torch.cat([inputs['color_aug', fi, 0], rgb], 1) if fi < 0 else torch.cat([rgb, inputs['color_aug', fi, 0]], 1)
            pf = pose_encoder(pin); ax, tr = pose_decoder([pf])
            outputs[('cam_T_cam', 0, fi)] = transformation_from_parameters(ax[:, 0], tr[:, 0], invert=(fi < 0))

        # Depth
        feats = encoder(aug)
        outputs.update(depth_decoder(feats))

        # Warp
        for sc in range(opt.num_scales):
            disp = outputs[('disp', sc)]
            d_up = F.interpolate(disp, [opt.height, opt.width], mode='bilinear', align_corners=False)
            _, depth = disp_to_depth(d_up, opt.min_depth, opt.max_depth)
            outputs[('depth', 0, sc)] = depth
            for fi in [-1, 1]:
                T = outputs[('cam_T_cam', 0, fi)]
                pc = proj3d(backproj(depth, inputs[('inv_K', 0)]), inputs[('K', 0)], T)
                outputs[('color', fi, sc)] = F.grid_sample(inputs['color', fi, 0], pc, padding_mode='border', align_corners=True)

        # Loss
        total = torch.tensor(0.0, device=DEVICE)
        tgt = inputs['color', 0, 0]
        for sc in range(opt.num_scales):
            loss = torch.tensor(0.0, device=DEVICE)
            reproj, ident = [], []
            for fi in [-1, 1]:
                p = outputs[('color', fi, sc)]
                diff = torch.abs(tgt - p)
                reproj.append(0.85 * ssim(p, tgt).mean(1, True) + 0.15 * diff.mean(1, True))
            reproj = torch.cat(reproj, 1)
            for fi in [-1, 1]:
                p = inputs['color', fi, 0]
                d = torch.abs(tgt - p)
                ident.append(0.85 * ssim(p, tgt).mean(1, True) + 0.15 * d.mean(1, True))
            ic = torch.cat(ident, 1)
            ic = ic + torch.randn(ic.shape, device=DEVICE) * 1e-5
            combined = torch.cat([ic, reproj], 1)
            to_opt, _ = torch.min(combined, dim=1)
            loss += to_opt.mean()
            ds = outputs[('disp', sc)]
            md = ds.mean(2, True).mean(3, True)
            nd = ds / (md + 1e-7)
            cf = F.interpolate(tgt, size=ds.shape[-2:], mode='bilinear', align_corners=False)
            loss += opt.disparity_smoothness * smooth_loss(nd, cf) / (2 ** sc)
            total += loss

        loss_val = total / opt.num_scales
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        ep_loss += loss_val.item()

        if bi % 500 == 0:
            dt = time.time() - ep_t0
            print(f'  [{bi}] loss={loss_val.item():.4f} t={dt:.0f}s', flush=True)

    scheduler.step()
    ep_dt = time.time() - ep_t0
    ep_avg = ep_loss / len(dl)
    print(f'[E{epoch+1:02d}/{opt.num_epochs}] avg_loss={ep_avg:.4f} time={ep_dt/60:.0f}m lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)

    ckpt = dict(encoder=encoder.state_dict(), depth=depth_decoder.state_dict(),
                pose_enc=pose_encoder.state_dict(), pose_dec=pose_decoder.state_dict(),
                gem=gem.state_dict(), epoch=epoch, seed=SEED)
    torch.save(ckpt, os.path.join(OUT_DIR, f'epoch_{epoch:02d}.pth'))
    torch.save(ckpt, os.path.join(OUT_DIR, 'epoch_latest.pth'))

total_dt = time.time() - total_t0
print(f'\nDone! Total: {total_dt/3600:.1f}h', flush=True)
