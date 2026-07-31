
import torch
import torch.nn as nn
from einops import rearrange

from utilities import *
import argparse

import operator
from functools import reduce
from functools import partial
import time

import scipy.io
from torch.profiler import profile, ProfilerActivity, record_function, schedule

import gc
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description="KS FFNO Trainer")
parser.add_argument(
    "--save",
    type=str,
    default="output_isotime",
    help="File name to save the trained model."
)
parser.add_argument(
    "--width",
    type=int,
    # default=8,
    default=64,
    help="Model width."
)
parser.add_argument(
    "--modes",
    type=int,
    # default=8,
    default=16,
    help="Model modes."
)
parser.add_argument(
    "--n_layers",
    type=int,
    # default=8,
    default=24,
    help="Model modes."
)
parser.add_argument(
    "--budget",
    type=float,
    default=100,
    help="Time budget in seconds."
)
parser.add_argument(
    "--ndata",
    type=int,
    default=1000,
    help="Training data size."
)
parser.add_argument(
    "--data_path",
    type=str,
    default="../generated_large_g256_fp32/",
    help="Path to data."
)
parser.add_argument(
    "--grid",
    type=int,
    default=64,
    help="Grid size."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="Batch size for training."
)
parser.add_argument(
    "--inference_batch_size",
    type=int,
    default=100,
    help="Batch size for validation and inference."
)
parser.add_argument(
    "--reduced_resolution",
    type=int,
    default=1,
    help="Downsample factor for grid size."
)
parser.add_argument(
    "--reduced_resolution_t",
    type=int,
    default=1,
    help="Downsample factor for time step (for evaluation)."
)
parser.add_argument("--use_autoregressive", action="store_true", default=False, help="Use autoregressive rollout during training.")
parser.add_argument("--single_frame_initialization", action="store_true", default=False,
                    help="Initialize the history with repeated copies of its first frame.")
args = parser.parse_args()

import os
import shutil
if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists(args.save + "/model/"):
    os.makedirs(args.save + "/model")

if not os.path.exists(args.save + "/log/"):
    os.makedirs(args.save + "/log/")

home_path = "."
if not os.path.exists(home_path + args.save):
    os.makedirs(home_path + args.save)

if not os.path.exists(home_path + args.save + "/model/"):
    os.makedirs(home_path + args.save + "/model")

if not os.path.exists(home_path + args.save + "/log/"):
    os.makedirs(home_path + args.save + "/log/")

################################################################
# Model
################################################################
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
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
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

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

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
            WNLinear(128, 1, wnorm=ff_weight_norm))

    def forward(self, x, **kwargs):
        # x.shape == [n_batches, *dim_sizes, input_size]
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

        return {
            'forecast': forecast,
            'forecast_list': forecast_list,
        }


################################################################
# configs
################################################################

ntrain = args.ndata
ntest = 100
noise_std = 0.01

modes = args.modes
width = args.width
n_layers = args.n_layers

time_budget = args.budget
per_step_time = 0.41
per_data_generation_time = 0.06673403584957123
per_sample_downsample_time = 0

sub = args.reduced_resolution
reduced_resolution_t = args.reduced_resolution_t

downsample_time = per_sample_downsample_time * ntrain
generation_time = per_data_generation_time * ntrain
train_budget = time_budget - generation_time - downsample_time

if train_budget <= 0:
    print("no train budget")
    exit()

print(f"train budget: {train_budget}")

batch_size = args.batch_size
inference_batch_size = args.inference_batch_size
learning_rate = 0.001

scheduler_step = int((train_budget // per_step_time) * 0.1)
scheduler_gamma = 0.5

print(learning_rate, scheduler_step, scheduler_gamma)

runtime = np.zeros(2, )
t1 = time.time()

S = args.grid
T_in = 5
T = 50 - T_in
T_rollout = T + (T_in - 1 if args.single_frame_initialization else 0)
T_eval = 999
step = 1

################################################################
# load data
################################################################

def load_data(file_index, n):
    u = np.load(args.data_path + f"data_{file_index}_u.npy", mmap_mode="r+")
    fields_u = torch.from_numpy(u).float()

    train_x = fields_u[:n,::sub,::sub,:T_in]
    train_y = fields_u[:n,::sub,::sub,T_in:T+T_in]
    train_x = train_x.reshape(n, S//sub, S//sub, T_in, 1)
    train_y = train_y.reshape(n, S//sub, S//sub, T, 1)

    print("data size:", train_x.shape)
    print("data device", train_x.device)

    if file_index == "val":
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=inference_batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    return train_loader

t1 = time.time()

train_loader = load_data(0, min(20000, ntrain))
cur_data_file_loaded = 0
print("load train data finished")

val_loader = load_data("val", 100)
print("load val data finished")

t2 = time.time()

print('preprocessing finished, time used:', t2-t1)

output_path = f"ffno_KS_NvsB_budget{int(time_budget)}_n{ntrain}_w{width}_m{modes}_batch{batch_size}"

################################################################
# training and evaluation
################################################################
import os

use_checkpoint = False
ckpt_pth = None

print("<- Train:", output_path, " ->")

if os.path.isfile(home_path + args.save + "/model/" + output_path):
    print("Trained.")
    exit()

if os.path.isfile(args.save + "/model/" + output_path + "_checkpoint"):
    print("Checkpoint found at machine. Uses checkpoint.")
    use_checkpoint = True
    ckpt_pth = args.save + "/model/" + output_path + "_checkpoint"
elif os.path.isfile(home_path + args.save + "/model/" + output_path + "_checkpoint"):
    print("Checkpoint found at home dir. Uses checkpoint.")
    use_checkpoint = True
    ckpt_pth = home_path + args.save + "/model/" + output_path + "_checkpoint"
else:
    print("No checkpoint found. New training.")

wandb.init(
    project=f"ffno_ks_2D",
    group=f"B{int(time_budget)}",
    name=f"n{ntrain}_B{int(time_budget)}"
)

model = FNOFactorized2DBlock(
    modes=modes,
    width=width,
    input_dim=T_in,
    # out_dim=1,
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
    use_fork=False,          # classic: use backcast path → final head
    mode='full'              # full fourier transform mode
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
loss_fn   = LpLoss()

if use_checkpoint:
    ckpt = torch.load(ckpt_pth)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])

loss_fn = LpLoss(size_average=False)

model.train()

# warm up
ep = 0
cur_total_time = 0

best_full_loss = float('inf')
best_step_loss = float('inf')

if use_checkpoint:
    ep = ckpt["epoch"]
    cur_total_time = ckpt["train_time"]

# warm-up
for xx, yy in train_loader:
    xx = xx.to(device)
    yy = yy.to(device)
    xx, yy = apply_single_frame_initialization(
        xx, yy, args.single_frame_initialization
    )

    B = xx.shape[0]

    for t in range(T_rollout):
        optimizer.zero_grad(set_to_none=True)

        if t == 0:
            win = xx[..., :T_in, :]
        elif t < T_in:
            win = torch.cat(
                [xx[..., -T_in + t:, :],  
                 yy[..., :t, :]],       
                dim=-2       
            )
        else:
            win = yy[..., t - T_in:t, :]

        inp = win.squeeze(-1)

        if noise_std > 0:
            inp = inp + noise_std * torch.randn_like(inp)

        y = yy[..., t, :]

        out = model(inp)['forecast']
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        break

    break

print("warm up finished")

gc.collect()
torch.cuda.empty_cache()

while cur_total_time <= train_budget:
    # print("ep:", ep)
    ep += 1

    trained_data = 0
    train_step_loss = 0

    model.train()

    for i in range(1):
        for xx, yy in train_loader:
            trained_data += batch_size

            xx = xx.to(device)
            yy = yy.to(device)
            xx, yy = apply_single_frame_initialization(
                xx, yy, args.single_frame_initialization
            )

            # Initialize autoregressive state if needed
            if args.use_autoregressive:
                ar_state = xx.clone()  # Start with initial condition

            B = xx.shape[0]

            for t in range(T_rollout):
                optimizer.zero_grad(set_to_none=True)

                if not args.use_autoregressive:
                    # Teacher forcing: use ground truth
                    if t == 0:
                        win = xx[..., :T_in, :]
                    elif t < T_in:
                        win = torch.cat(
                            [xx[..., -T_in + t:, :],  
                            yy[..., :t, :]],       
                            dim=-2       
                        )
                    else:
                        win = yy[..., t - T_in:t, :]
                else:
                    # Autoregressive: use model predictions
                    if t == 0:
                        win = xx[..., :T_in, :]  # Use initial condition
                    else:
                        # Use last T_in timesteps from autoregressive state
                        win = ar_state[..., -T_in:, :]

                inp = win.squeeze(-1)

                if noise_std > 0:
                    inp = inp + noise_std * torch.randn_like(inp)

                y = yy[..., t, :]

                t_step_s = time.time()

                out = model(inp)['forecast']
                loss = loss_fn(out, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update autoregressive state with prediction
                if args.use_autoregressive:
                    ar_state = torch.cat([ar_state[..., 1:, :], out.unsqueeze(-1)], dim=-2)

                t_step_e = time.time()
                step_time = t_step_e - t_step_s
                cur_total_time += step_time
                train_step_loss += float(loss.item())

            if cur_total_time > train_budget:
                break

            if trained_data >= ntrain:
                break

        if cur_total_time > train_budget:
            break

        if trained_data >= ntrain:
            break

    if ep % 10 == 0 or cur_total_time > train_budget:
        wandb.log({
            "epoch": ep,
            "step": scheduler.last_epoch,
            "train_time": cur_total_time,
            "lr": optimizer.param_groups[0]['lr'],
            "loss": train_step_loss / trained_data / T_rollout,
        })

    if ep % 100 == 0:
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": ep,        
            "scheduler_state": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_time": cur_total_time,
        }
        torch.save(ckpt, args.save + "/model/" + output_path + "_checkpoint")
        shutil.copy(args.save + "/model/" + output_path + "_checkpoint", home_path + args.save + "/model/")

torch.save(model, args.save + "/model/" + output_path)
shutil.copy(args.save + "/model/" + output_path, home_path + args.save + "/model/")

import math
def traj_l2(preds, targets):
    """
    preds, targets: (B, S, S, T, 1)  or (B, S, S, T) 都行
    return: per-sample L2 over (S,S,T,channels) => shape (B,)
    """
    if preds.dim() == 4:   # (B,S,S,T)
        preds = preds.unsqueeze(-1)
    if targets.dim() == 4:
        targets = targets.unsqueeze(-1)

    diff = preds - targets                       # (B,S,S,T,1)
    B = diff.shape[0]
    diff_flat = diff.reshape(B, -1)
    return torch.norm(diff_flat, p=2, dim=1)     # (B,)


model.eval()

val_step_loss = 0
full_loss_sum = 0
dnrmse_sum = 0.0
dnrmse_count = 0
worst_loss = -math.inf
worst_idx  = -1

global_base = 0

with torch.no_grad():
    for xx, yy in val_loader:
        xx, yy = apply_single_frame_initialization(
            xx, yy, args.single_frame_initialization
        )
        B = xx.shape[0]

        window = xx[..., :T_in, :]     # CPU: (B,S,S,T_in,1)

        preds_list = []
        for t in range(T_rollout):
            inp = window.squeeze(-1).contiguous().to(device, non_blocking=True) 
            out = model(inp)['forecast']                  

            y   = yy[..., t, :].contiguous().to(device, non_blocking=True)
            val_step_loss += loss_fn(out, y).item()

            preds_list.append(out.detach().cpu().unsqueeze(-2))

            out_cpu = out.detach().cpu().unsqueeze(-2)                           # (B,S,S,1,1)
            window = torch.cat([window[..., 1:, :], out_cpu], dim=-2)            # (B,S,S,T_in,1)

            del inp, out, y

        preds = torch.cat(preds_list, dim=-2)
        per_sample = traj_l2(preds, yy)
        dnrmse_batch = dimensionwise_nrmse(preds, yy)

        full_loss_sum += float(per_sample.sum().item())
        dnrmse_sum += float(dnrmse_batch.sum().item())
        dnrmse_count += int(dnrmse_batch.shape[0])

        batch_worst_loss, batch_worst_j = torch.max(per_sample, dim=0)
        batch_worst_loss = float(batch_worst_loss.item())
        batch_worst_j = int(batch_worst_j.item())

        if batch_worst_loss > worst_loss:
            worst_loss = batch_worst_loss
            worst_idx = global_base + batch_worst_j

        global_base += B

val_step_loss /= (ntest * T_rollout)
test_full_loss = full_loss_sum / max(1, ntest)
test_dnrmse = dnrmse_sum / max(1, dnrmse_count)

wandb.log({
    "worst_l2_loss": worst_loss,
    "worst_traj_idx": worst_idx,
    "test_step_loss": val_step_loss,
    "test_full_loss": test_full_loss,
    "test/dnrmse": test_dnrmse,
})

print(f"val_step_loss: {val_step_loss}")
print(f"test_full_loss: {test_full_loss}")
print(f"test dNRMSE: {test_dnrmse}")
print(f"[TEST] worst trajectory idx = {worst_idx}, worst L2 loss = {worst_loss}")
