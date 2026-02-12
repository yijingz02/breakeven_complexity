import argparse
import os
import shutil
import time
import gc
import math

import yaml
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from utilities import *

# Unified FFNO trainer that uses a YAML config to specify dataset and training parameters.
# Usage: python ffno_train.py --config ffno/configs/gs.yaml

parser = argparse.ArgumentParser(description="Unified FFNO Trainer (config-driven)")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
parser.add_argument("--override", nargs="*", help="Optional key=value overrides for config top-level keys.")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f) or {}

# apply simple overrides (key=value) to top-level keys in cfg
if args.override:
    for item in args.override:
        if "=" in item:
            k, v = item.split("=", 1)
            # try to parse numbers and booleans
            if v.lower() in ("true", "false"):
                val = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        val = float(v)
                        # if integer-like
                        if val.is_integer():
                            val = int(val)
                    else:
                        val = int(v)
                except Exception:
                    val = v
            cfg[k] = val

# helper to read keys with defaults
def _cfg_get(k, default=None):
    return cfg.get(k, default)

# ---------- model definitions (copied & adapted) ----------
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
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape
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
    def __init__(self, modes, width, input_dim=12, dropout=0.0, in_dropout=0.0,
                 n_layers=4, share_weight: bool = False,
                 share_fork=False, factor=2,
                 ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, use_fork=False, mode='full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
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
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
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
            WNLinear(128, 2, wnorm=ff_weight_norm))

    def forward(self, x, **kwargs):
        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        forecast_list = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, f = layer(x)
            if self.use_fork:
                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)
            x = x + b
        if not self.use_fork:
            forecast = self.out(b)
        return {'forecast': forecast, 'forecast_list': forecast_list}

# ---------- end model defs ----------

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read config values with defaults
ntrain = int(_cfg_get('ndata', 1000))
ntest = int(_cfg_get('ntest', 100))
noise_std = float(_cfg_get('noise_std', 0.01))

modes = int(_cfg_get('modes', 16))
width = int(_cfg_get('width', 64))
n_layers = int(_cfg_get('n_layers', 16))

time_budget = float(_cfg_get('budget', 100.0))
per_step_time = float(_cfg_get('per_step_time', 0.17))
per_data_generation_time = float(_cfg_get('per_data_generation_time', 0.47509707514047617))
per_sample_downsample_time = float(_cfg_get('per_sample_downsample_time', 0))

sub = int(_cfg_get('reduced_resolution', 1))
reduced_resolution_t = int(_cfg_get('reduced_resolution_t', 1))

S = int(_cfg_get('grid', 64))
T_in = int(_cfg_get('T_in', 10))
T_total = int(_cfg_get('T_total', 200))
T = T_total - T_in
batch_size = int(_cfg_get('batch_size', 100))
learning_rate = float(_cfg_get('learning_rate', 0.001))

# compute budget
downsample_time = per_sample_downsample_time * ntrain
generation_time = per_data_generation_time * ntrain
train_budget = time_budget - generation_time - downsample_time
if train_budget <= 0:
    raise SystemExit("no train budget")

scheduler_step = int((train_budget // per_step_time) * 0.1)
if scheduler_step < 1:
    scheduler_step = 1
scheduler_gamma = float(_cfg_get('scheduler_gamma', 0.5))

print(f"train budget: {train_budget:.2f}")
print(learning_rate, scheduler_step, scheduler_gamma)

# data loader helper (expects .npy files named data_{idx}_u.npy/data_{idx}_v.npy or dataset-specific keys)
def load_data_np(data_path, file_index, n, sub, S, T_in, T):
    # generic loader for u/v style data
    u_p = os.path.join(data_path, f"data_{file_index}_u.npy")
    v_p = os.path.join(data_path, f"data_{file_index}_v.npy")
    if os.path.exists(u_p) and os.path.exists(v_p):
        u = np.load(u_p, mmap_mode='r+')
        v = np.load(v_p, mmap_mode='r+')
        fu = torch.from_numpy(u).float()
        fv = torch.from_numpy(v).float()
        train_x_u = fu[:n, ::sub, ::sub, :T_in]
        train_x_v = fv[:n, ::sub, ::sub, :T_in]
        train_x = torch.stack([train_x_u, train_x_v], dim=-1)
        train_y_u = fu[:n, ::sub, ::sub, T_in:T+T_in]
        train_y_v = fv[:n, ::sub, ::sub, T_in:T+T_in]
        train_y = torch.stack([train_y_u, train_y_v], dim=-1)
        train_x = train_x.reshape(n, S//sub, S//sub, T_in, 2)
        train_y = train_y.reshape(n, S//sub, S//sub, T, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        return loader
    else:
        raise FileNotFoundError(f"Expected data at {u_p} and {v_p}")

# prepare output dirs
save_dir = _cfg_get('save', 'output_isotime')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'log'), exist_ok=True)
home_path = _cfg_get('home_path', '.')
os.makedirs(os.path.join(home_path, save_dir), exist_ok=True)

# build model
model = FNOFactorized2DBlock(
    modes=modes,
    width=width,
    input_dim=2*T_in,
    dropout=float(_cfg_get('dropout', 0.0)),
    in_dropout=float(_cfg_get('in_dropout', 0.0)),
    n_layers=n_layers,
    share_weight=_cfg_get('share_weight', False),
    share_fork=_cfg_get('share_fork', False),
    factor=_cfg_get('factor', 2),
    ff_weight_norm=_cfg_get('ff_weight_norm', False),
    n_ff_layers=_cfg_get('n_ff_layers', 2),
    gain=_cfg_get('gain', 1),
    layer_norm=_cfg_get('layer_norm', False),
    use_fork=_cfg_get('use_fork', False),
    mode=_cfg_get('mode', 'full')
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
loss_fn = LpLoss()

# simple data file handling: load first chunk and a val chunk
data_path = _cfg_get('data_path', '.')
train_loader = load_data_np(data_path, 0, min(2000, ntrain), sub, S, T_in, T)
val_loader = load_data_np(data_path, 'val', 100, sub, S, T_in, T)

# Training loop (kept concise; follows original style)
model.train()
cur_total_time = 0.0
ep = 0
best_full_loss = float('inf')

# warmup
for xx, yy in train_loader:
    for t in range(0, T):
        optimizer.zero_grad(set_to_none=True)
        if t == 0:
            inp = xx[..., :T_in, :]
        elif t < T_in:
            inp = torch.cat([xx[..., -T_in + t:, :], yy[..., :t, :]], dim=-2)
        else:
            inp = yy[..., t - T_in:t, :]
        inp = inp.reshape(batch_size, S, S, T_in * 2)
        if noise_std > 0:
            inp = inp + noise_std * torch.randn_like(inp)
        y = yy[..., t, :]
        inp = inp.to(device)
        y = y.to(device)
        out = model(inp)['forecast']
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        inp = inp.to('cpu')
        y = y.to('cpu')
        break
    break

print('warm up finished')

# main loop with budget
while cur_total_time < train_budget:
    ep += 1
    trained_data = 0
    train_step_loss = 0
    model.train()

    # use several files if present (0..9 like original)
    for file_index in range(10):
        if trained_data >= ntrain or cur_total_time >= train_budget:
            break
        try:
            train_loader = load_data_np(data_path, file_index, min(2000, ntrain - trained_data), sub, S, T_in, T)
        except FileNotFoundError:
            break

        for xx, yy in train_loader:
            trained_data += batch_size
            for t in range(0, T):
                if t == 0:
                    inp = xx[..., :T_in, :]
                elif t < T_in:
                    tmp_x = xx[..., -T_in + t:, :].to(device)
                    tmp_y = yy[..., :t, :].to(device)
                    inp = torch.cat([tmp_x, tmp_y], dim=-2)
                    tmp_x = tmp_x.to('cpu')
                    tmp_y = tmp_y.to('cpu')
                else:
                    inp = yy[..., t - T_in:t, :]
                inp = inp.reshape(batch_size, S, S, T_in * 2)
                if noise_std > 0:
                    inp = inp + noise_std * torch.randn_like(inp)
                y = yy[..., t, :]
                inp = inp.to(device)
                y = y.to(device)
                t_step_s = time.time()
                out = model(inp)['forecast']
                loss = loss_fn(out, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                t_step_e = time.time()
                step_time = t_step_e - t_step_s
                cur_total_time += step_time
                train_step_loss += float(loss.item())
                inp = inp.to('cpu')
                y = y.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
                if cur_total_time >= train_budget:
                    break
            if cur_total_time >= train_budget or trained_data >= ntrain:
                break

    print(f"Epoch {ep} time {cur_total_time:.2f} loss {(train_step_loss / max(1, trained_data) / max(1, T)):.6f}")

    if ep % 100 == 0 or cur_total_time >= train_budget:
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": ep,
            "scheduler_state": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_time": cur_total_time,
        }
        torch.save(ckpt, os.path.join(save_dir, 'model', 'ffno_checkpoint'))
        shutil.copy(os.path.join(save_dir, 'model', 'ffno_checkpoint'), os.path.join(home_path, save_dir, 'model'))

# final save
torch.save(model, os.path.join(save_dir, 'model', 'ffno_model.pt'))
shutil.copy(os.path.join(save_dir, 'model', 'ffno_model.pt'), os.path.join(home_path, save_dir, 'model'))

print('Training finished')
