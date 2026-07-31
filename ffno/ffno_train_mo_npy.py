#!/usr/bin/env python3
# ffno_multi_obstacle_incompressible_multiframe_mask.py
#
# Train FFNO on multi-obstacle flow (velocity_x, velocity_y, pressure) with static obstacle mask,
# conditioning on a history window (multi-frame).
#
# Expected shapes (time-first):
#   velocity_x: (N, T_total, H, W)
#   velocity_y: (N, T_total, H, W)
#   pressure:   (N, T_total, H, W)
#   mask:       (N, H, W)   boolean-ish
#
# Conditioning:
#   At each step, input features per pixel = concat([vx,vy,p] over T_in frames, mask) -> (B,H,W,3*T_in+1)
#   Output per step = (vx,vy,p) at next frame -> (B,H,W,3)
#
# By default this keeps your original "masked-to-fluid loss" behavior (like your vorticity script),
# but you can turn it OFF (paper-style) via --no_mask_loss.

import os
import shutil
import time
import gc
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import wandb

from utilities import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="FFNO Multi-Obstacle (vx,vy,p multiframe + mask conditioning)")
parser.add_argument("--save", type=str, default="output_isotime", help="Directory to save model/logs.")
parser.add_argument("--width", type=int, default=64, help="Model width.")
parser.add_argument("--modes", type=int, default=16, help="Fourier modes.")
parser.add_argument("--n_layers", type=int, default=24, help="Number of FFNO layers.")
parser.add_argument("--budget", type=float, default=100, help="Time budget in seconds.")
parser.add_argument("--ndata", type=int, default=1000, help="Training data size.")
parser.add_argument("--data_path", type=str, default="../generated_large_g256_fp32/", help="Path to .npy data.")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
parser.add_argument("--inference_batch_size", type=int, default=100, help="Batch size for validation and inference.")
parser.add_argument("--reduced_resolution", type=int, default=1, help="Downsample factor for H/W.")
parser.add_argument("--noise_std", type=float, default=0.01, help="Input Gaussian noise std.")

# multiframe + horizon
parser.add_argument("--T_in", type=int, default=5, help="History window length (frames).")
parser.add_argument("--T_total_used", type=int, default=21, help="Total frames used from each traj.")
# (rollout steps) T = T_total_used - T_in

# mask semantics and loss options
parser.add_argument("--mask_is_obstacle", action="store_true",
                    help="If set, mask==1 means obstacle (default assumption). If not set, mask==1 means fluid.")
parser.add_argument("--zero_pred_in_obstacle", action="store_true",
                    help="If set, force prediction to 0 inside obstacles (often stabilizes rollout).")
parser.add_argument("--no_mask_loss", action="store_true",
                    help="If set, DO NOT mask the loss to fluid (paper-style: train on all pixels).")

# normalization (defaults from your MultiObstacleIncompressible constants)
parser.add_argument("--mean_vx", type=float, default=0.9619638919830322)
parser.add_argument("--mean_vy", type=float, default=-7.730342986178584e-06)
parser.add_argument("--mean_p",  type=float, default=-0.39420366287231445)
parser.add_argument("--std_vx",  type=float, default=0.6491641402244568)
parser.add_argument("--std_vy",  type=float, default=0.5268091559410095)
parser.add_argument("--std_p",   type=float, default=1.060120940208435)
parser.add_argument("--single_frame_initialization", action="store_true", default=False,
                    help="Initialize the history with repeated copies of its first frame.")

args = parser.parse_args()

# -------------------------
# FFNO modules (your original, with small change: out_dim param)
# -------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
        else:
            self.forecast_ff = None

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x: [B, H, W, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape

        # Y
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')

        # X
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]
        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')
        return x


class FNOFactorized2DBlock(nn.Module):
    def __init__(self, modes, width, input_dim, out_dim,
                 dropout=0.0, in_dropout=0.0,
                 n_layers=4, share_weight: bool = False,
                 share_fork=False, factor=2,
                 ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, use_fork=False, mode='full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork

        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = FeedForward(
                    width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = FeedForward(
                width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(
                in_dim=width,
                out_dim=width,
                n_modes=modes,
                forecast_ff=self.forecast_ff,
                backcast_ff=self.backcast_ff,
                fourier_weight=self.fourier_weight,
                factor=factor,
                ff_weight_norm=ff_weight_norm,
                n_ff_layers=n_ff_layers,
                layer_norm=layer_norm,
                use_fork=use_fork,
                dropout=dropout,
                mode=mode))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, out_dim, wnorm=ff_weight_norm))

    def forward(self, x, **kwargs):
        # x: (B,H,W,input_dim)
        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        forecast_list = []
        for i in range(self.n_layers):
            b, f = self.spectral_layers[i](x)

            if self.use_fork:
                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b

        if not self.use_fork:
            forecast = self.out(b)

        return {'forecast': forecast, 'forecast_list': forecast_list}


# -------------------------
# Training config (your style)
# -------------------------
ntrain = int(args.ndata)
ntest = 100
noise_std = float(args.noise_std)

modes = int(args.modes)
width = int(args.width)
n_layers = int(args.n_layers)

time_budget = float(args.budget)
per_step_time = 0.41
per_data_generation_time = 180
per_sample_downsample_time = 0

sub = int(args.reduced_resolution)

downsample_time = per_sample_downsample_time * ntrain
generation_time = per_data_generation_time * ntrain
train_budget = time_budget - generation_time - downsample_time
if train_budget <= 0:
    print("no train budget")
    raise SystemExit(0)

print(f"train budget: {train_budget:.2f}")

batch_size = int(args.batch_size)
inference_batch_size = int(args.inference_batch_size)
learning_rate = 1e-3

scheduler_step = max(1, int((train_budget // per_step_time) * 0.1))
scheduler_gamma = 0.5
print(learning_rate, scheduler_step, scheduler_gamma)

# multiframe horizon
T_in = int(args.T_in)
T_total_used = int(args.T_total_used)
assert T_total_used > T_in, "Need T_total_used > T_in"
T = T_total_used - T_in  # rollout steps per traj chunk
T_rollout = T + (T_in - 1 if args.single_frame_initialization else 0)

# normalization tensors
MEAN = torch.tensor([args.mean_vx, args.mean_vy, args.mean_p], dtype=torch.float32)  # (3,)
STD  = torch.tensor([args.std_vx,  args.std_vy,  args.std_p ], dtype=torch.float32)  # (3,)

# -------------------------
# Mask helpers + loss
# -------------------------
def to_fluid_mask(mask):
    # mask: (B,H,W,1) in {0,1}
    if args.mask_is_obstacle:
        return 1.0 - mask  # 1=fluid
    else:
        return mask         # already 1=fluid

def l2_loss(out, y):
    # out,y: (B,H,W,C)
    diff = out - y
    B = diff.shape[0]
    return torch.norm(diff.reshape(B, -1), p=2, dim=1).mean()

def masked_l2_loss(out, y, fluid):
    # out,y: (B,H,W,C), fluid: (B,H,W,1)
    # compute RMSE over fluid pixels, averaged over channels
    diff = (out - y) * fluid  # broadcast over C
    denom = fluid.sum().clamp_min(1.0) * out.shape[-1]
    mse = (diff * diff).sum() / denom
    return torch.sqrt(mse + 1e-12)

def traj_l2(preds, targets, fluid=None):
    # preds/targets: (B,H,W,T,C)
    diff = preds - targets
    if fluid is not None:
        diff = diff * fluid.unsqueeze(-2)  # (B,H,W,1)->(B,H,W,1,1) over T,C
    B = diff.shape[0]
    return torch.norm(diff.reshape(B, -1), p=2, dim=1)

def traj_rel_l2(preds, targets, fluid=None, eps=1e-12):
    diff = preds - targets
    if fluid is not None:
        f = fluid.unsqueeze(-2)
        diff = diff * f
        targets = targets * f
    B = diff.shape[0]
    num = torch.norm(diff.reshape(B, -1), p=2, dim=1)
    den = torch.norm(targets.reshape(B, -1), p=2, dim=1).clamp_min(eps)
    return num / den


# -------------------------
# Data loader
# -------------------------
def load_data(file_index, n):
    """
    Loads:
      data_velocity_x_{file_index}.npy  (N,T,H,W)
      data_velocity_y_{file_index}.npy  (N,T,H,W)
      data_pressure_{file_index}.npy    (N,T,H,W)
      data_mask_{file_index}.npy        (N,H,W)
    and returns a loader over:
      x:    (B,H',W',T_in,3)
      y:    (B,H',W',T,3)
      mask: (B,H',W',1)
    """
    if file_index == "val":
        vx = np.load(os.path.join(args.data_path, f"data_velocity_x_val.npy"), mmap_mode="r")
        vy = np.load(os.path.join(args.data_path, f"data_velocity_y_val.npy"), mmap_mode="r")
        pp = np.load(os.path.join(args.data_path, f"data_pressure_val.npy"),   mmap_mode="r")
        mm = np.load(os.path.join(args.data_path, f"data_mask_val.npy"),       mmap_mode="r")
    else:
        vx = np.load(os.path.join(args.data_path, f"data_velocity_x_{file_index}.npy"), mmap_mode="r")
        vy = np.load(os.path.join(args.data_path, f"data_velocity_y_{file_index}.npy"), mmap_mode="r")
        pp = np.load(os.path.join(args.data_path, f"data_pressure_{file_index}.npy"),   mmap_mode="r")
        mm = np.load(os.path.join(args.data_path, f"data_mask_{file_index}.npy"),       mmap_mode="r")

    vx = torch.from_numpy(vx).float()  # (N,T,H,W)
    vy = torch.from_numpy(vy).float()
    pp = torch.from_numpy(pp).float()
    mask = torch.from_numpy(mm).float()  # (N,H,W)

    n = min(int(n), vx.shape[0])

    # slice time and downsample space
    vx_x = vx[:n, :T_in, ::sub, ::sub]                 # (n,T_in,H',W')
    vy_x = vy[:n, :T_in, ::sub, ::sub]
    pp_x = pp[:n, :T_in, ::sub, ::sub]

    vx_y = vx[:n, T_in:T_in + T, ::sub, ::sub]         # (n,T,H',W')
    vy_y = vy[:n, T_in:T_in + T, ::sub, ::sub]
    pp_y = pp[:n, T_in:T_in + T, ::sub, ::sub]

    mask = mask[:n, ::sub, ::sub]                      # (n,H',W')

    # stack channels: (n,T_in,H',W',3) and (n,T,H',W',3)
    x_t = torch.stack([vx_x, vy_x, pp_x], dim=-1)       # (n,T_in,H',W',3)
    y_t = torch.stack([vx_y, vy_y, pp_y], dim=-1)       # (n,T,H',W',3)

    # normalize physical channels only
    # broadcast MEAN/STD: (3,) -> (1,1,1,1,3)
    x_t = (x_t - MEAN.view(1, 1, 1, 1, 3)) / STD.view(1, 1, 1, 1, 3)
    y_t = (y_t - MEAN.view(1, 1, 1, 1, 3)) / STD.view(1, 1, 1, 1, 3)

    # permute to time-last for easier windowing: (n,H',W',T_in,3), (n,H',W',T,3)
    x = x_t.permute(0, 2, 3, 1, 4).contiguous()
    y = y_t.permute(0, 2, 3, 1, 4).contiguous()

    # mask static feature: (n,H',W',1)
    mask = mask.unsqueeze(-1).contiguous()

    if file_index == "val":
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y, mask),
            batch_size=inference_batch_size,
            shuffle=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y, mask),
            batch_size=batch_size,
            shuffle=True
        )
    return loader


train_loader = load_data(0, min(2200, ntrain))
print("load train data finished")
val_loader = load_data("val", ntest)
print("load val data finished")

output_path = f"ffno_multiobs_incomp_B{int(time_budget)}_n{ntrain}_w{width}_m{modes}_batch{batch_size}_Tin{T_in}_T{T_total_used}"
local_ckpt = os.path.join(args.save, "model", f"{output_path}.pt")

# -------------------------
# wandb
# -------------------------
wandb.init(
    project="ffno_multi_obstacle_velocity_pressure",
    group=f"B{int(time_budget)}",
    name=f"incomp_n{ntrain}_B{int(time_budget)}_Tin{T_in}_T{T_total_used}",
)

# -------------------------
# model/optim
# input_dim = 3*T_in + 1 (mask)
# out_dim   = 3 (vx,vy,p)
# -------------------------
in_dim = 3 * T_in + 1
out_dim = 3

model = FNOFactorized2DBlock(
    modes=modes,
    width=width,
    input_dim=in_dim,
    out_dim=out_dim,
    dropout=0.0,
    in_dropout=0.0,
    n_layers=n_layers,
    share_weight=False,
    share_fork=False,
    factor=2,
    ff_weight_norm=False,
    n_ff_layers=2,
    gain=1,
    layer_norm=False,
    use_fork=False,
    mode='full'
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

if use_checkpoint:
    ckpt = torch.load(ckpt_pth, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state", None) is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])

# -------------------------
# Warm-up (one step)
# -------------------------
model.train()
ep = 0
cur_total_time = 0.0
if use_checkpoint:
    ep = int(ckpt["epoch"])
    cur_total_time = float(ckpt["train_time"])

for xx, yy, mask in train_loader:
    xx = xx.to(device)      # (B,H,W,T_in,3)
    yy = yy.to(device)      # (B,H,W,T,3)
    xx, yy = apply_single_frame_initialization(
        xx, yy, args.single_frame_initialization
    )
    mask = mask.to(device)  # (B,H,W,1)
    fluid = to_fluid_mask(mask)

    for t in range(T_rollout):
        optimizer.zero_grad(set_to_none=True)

        if t == 0:
            win = xx[..., :T_in, :]  # (B,H,W,T_in,3)
        elif t < T_in:
            win = torch.cat([xx[..., -T_in + t:, :], yy[..., :t, :]], dim=-2)
        else:
            win = yy[..., t - T_in:t, :]

        # flatten time*channels: (B,H,W,T_in,3) -> (B,H,W,3*T_in)
        inp = win.reshape(win.shape[0], win.shape[1], win.shape[2], 3 * T_in)
        inp = torch.cat([inp, mask], dim=-1)  # (B,H,W,3*T_in+1)

        if noise_std > 0:
            inp = inp + noise_std * torch.randn_like(inp)

        y = yy[..., t, :]  # (B,H,W,3)

        out = model(inp)['forecast']  # (B,H,W,3)

        if args.zero_pred_in_obstacle:
            out = out * fluid  # broadcast over 3

        loss = l2_loss(out, y) if args.no_mask_loss else masked_l2_loss(out, y, fluid)
        loss.backward()
        optimizer.step()
        break
    break

print("warm up finished")
gc.collect()
torch.cuda.empty_cache()

print(f"train budget: {train_budget:.2f}")

# -------------------------
# Train (time-budgeted)
# -------------------------
while cur_total_time <= train_budget:
    print(f"Epoch {ep+1}, time used: {cur_total_time:.2f}s")
    ep += 1
    trained_data = 0
    train_loss_sum = 0.0

    model.train()

    for xx, yy, mask in train_loader:
        xx = xx.to(device, non_blocking=True)
        yy = yy.to(device, non_blocking=True)
        xx, yy = apply_single_frame_initialization(
            xx, yy, args.single_frame_initialization
        )
        mask = mask.to(device, non_blocking=True)
        fluid = to_fluid_mask(mask)

        B = xx.shape[0]
        trained_data += B

        for t in range(T_rollout):
            optimizer.zero_grad(set_to_none=True)

            if t == 0:
                win = xx[..., :T_in, :]
            elif t < T_in:
                win = torch.cat([xx[..., -T_in + t:, :], yy[..., :t, :]], dim=-2)
            else:
                win = yy[..., t - T_in:t, :]

            inp = win.reshape(B, win.shape[1], win.shape[2], 3 * T_in)  # (B,H,W,3*T_in)
            inp = torch.cat([inp, mask], dim=-1)                        # (B,H,W,3*T_in+1)
            if noise_std > 0:
                inp = inp + noise_std * torch.randn_like(inp)

            y = yy[..., t, :]                                           # (B,H,W,3)

            t_step_s = time.time()
            out = model(inp)['forecast']                                # (B,H,W,3)
            if args.zero_pred_in_obstacle:
                out = out * fluid

            loss = l2_loss(out, y) if args.no_mask_loss else masked_l2_loss(out, y, fluid)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_time = time.time() - t_step_s

            # print(f"  step time: {step_time:.4f}s, loss: {loss.item():.6f}")

            cur_total_time += step_time
            train_loss_sum += float(loss.item())

            if cur_total_time > train_budget:
                break

        if cur_total_time > train_budget or trained_data >= ntrain:
            break

    if (ep % 10 == 0) or (cur_total_time > train_budget):
        denom = max(1, trained_data) * T_rollout
        wandb.log({
            "epoch": ep,
            "step": scheduler.last_epoch,
            "train_time": cur_total_time,
            "lr": optimizer.param_groups[0]['lr'],
            "loss": train_loss_sum / denom,
        })

    if ep % 100 == 0:
        ckpt_out = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "epoch": ep,
            "train_time": cur_total_time,
        }
        torch.save(ckpt_out, local_ckpt)

# Save final model
final_model_path = os.path.join(args.save, "model", output_path)
torch.save(model, final_model_path)
shutil.copy(final_model_path, os.path.join(home_path, args.save, "model"))

# -------------------------
# Eval (autoregressive rollout)
# -------------------------
print("Start evaluation...")

model.eval()

val_step_loss_sum = 0.0
worst_loss = -math.inf
worst_idx = -1
global_base = 0
dnrmse_sum = 0.0
dnrmse_count = 0

with torch.no_grad():
    for xx, yy, mask in val_loader:
        xx, yy = apply_single_frame_initialization(
            xx, yy, args.single_frame_initialization
        )
        B = xx.shape[0]

        # keep window on CPU like your original eval
        window = xx[..., :T_in, :]     # (B,H,W,T_in,3) on CPU
        mask_cpu = mask               # (B,H,W,1) on CPU

        preds_list = []
        for t in range(T_rollout):
            win_d = window.contiguous().to(device, non_blocking=True)        # (B,H,W,T_in,3)
            mask_d = mask_cpu.contiguous().to(device, non_blocking=True)     # (B,H,W,1)
            fluid_d = to_fluid_mask(mask_d)                                  # (B,H,W,1)

            inp = win_d.reshape(B, win_d.shape[1], win_d.shape[2], 3 * T_in) # (B,H,W,3*T_in)
            inp = torch.cat([inp, mask_d], dim=-1)                           # (B,H,W,3*T_in+1)

            out = model(inp)['forecast']                                     # (B,H,W,3)
            if args.zero_pred_in_obstacle:
                out = out * fluid_d

            y = yy[..., t, :].contiguous().to(device, non_blocking=True)     # (B,H,W,3)
            step_loss = l2_loss(out, y) if args.no_mask_loss else masked_l2_loss(out, y, fluid_d)
            val_step_loss_sum += float(step_loss.item())

            out_cpu = out.detach().cpu().unsqueeze(-2)                       # (B,H,W,1,3)
            preds_list.append(out_cpu)

            # shift window: drop oldest frame, append new pred as last frame
            # window is (B,H,W,T_in,3), out_cpu is (B,H,W,1,3)
            window = torch.cat([window[..., 1:, :], out_cpu], dim=-2)

            del win_d, out, y, mask_d, fluid_d, inp

        preds = torch.cat(preds_list, dim=-2)                                # (B,H,W,T,3)

        fluid_cpu = to_fluid_mask(mask_cpu) if not args.no_mask_loss else None  # (B,H,W,1) or None
        per_sample = traj_l2(preds, yy, fluid=fluid_cpu) / T_rollout
        # per_sample = traj_rel_l2(preds, yy, fluid=fluid_cpu)
        dnrmse_batch = dimensionwise_nrmse(preds, yy)
        dnrmse_sum += float(dnrmse_batch.sum().item())
        dnrmse_count += int(dnrmse_batch.shape[0])

        batch_worst_loss, batch_worst_j = torch.max(per_sample, dim=0)
        batch_worst_loss = float(batch_worst_loss.item())
        batch_worst_j = int(batch_worst_j.item())

        if batch_worst_loss > worst_loss:
            worst_loss = batch_worst_loss
            worst_idx = global_base + batch_worst_j

        global_base += B

val_step_loss = val_step_loss_sum / (ntest * T_rollout)
test_dnrmse = dnrmse_sum / max(1, dnrmse_count)

wandb.log({
    "worst_l2_loss": worst_loss,
    "worst_traj_idx": worst_idx,
    "test_step_loss": val_step_loss,
    "test/dnrmse": test_dnrmse,
})

print(f"val_step_loss: {val_step_loss}")
print(f"test dNRMSE: {test_dnrmse}")
print(f"[TEST] worst trajectory idx = {worst_idx}, worst L2 loss = {worst_loss}")
