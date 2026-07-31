import torch
import torch.nn as nn
from einops import rearrange

from utilities import *
import argparse

import operator
from functools import reduce
from functools import partial
import time
import math
import scipy.io
import gc
import wandb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description="3D kuramoto sivashinsky simulation")
parser.add_argument("--save", type=str, default="output_isotime", help="File name to save the trained model.")
parser.add_argument("--width", type=int, default=64, help="Model width.")
parser.add_argument("--modes", type=int, default=16, help="Model modes.")
parser.add_argument("--n_layers", type=int, default=24, help="Model layers.")
parser.add_argument("--budget", type=float, default=100, help="Time budget in seconds.")
parser.add_argument("--ndata", type=int, default=1000, help="Training data size.")
parser.add_argument("--data_path", type=str, default="../generated_large_g256_fp32/", help="Path to data.")
parser.add_argument("--grid", type=int, default=64, help="Grid size (assumed isotropic X, Y, Z).")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training. Keep VERY low for 3D.")
parser.add_argument("--inference_batch_size", type=int, default=2, help="Batch size for validation and inference.")
parser.add_argument("--reduced_resolution", type=int, default=1, help="Downsample factor for grid size.")
parser.add_argument("--reduced_resolution_t", type=int, default=1, help="Downsample factor for time step.")
parser.add_argument("--project_name", type=str, default="ffno_ks_3d", help="Weights & Biases project name.")
parser.add_argument("--use_autoregressive", action="store_true", default=False, help="Use autoregressive rollout during training.")
parser.add_argument("--single_frame_initialization", action="store_true", default=False,
                    help="Initialize the history with repeated copies of its first frame.")
args = parser.parse_args()

################################################################
# Model (3D FNO)
################################################################
class SpectralConv3d(nn.Module):
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
            # Initialize 3 separate weights for X, Y, and Z axes
            for _ in range(3):
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
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b x y z i -> b i x y z')
        B, I, X, Y, Z = x.shape

        # # # Dimension Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, I, X, Y, Z // 2 + 1)
        if self.mode == 'full':
            out_ft[:, :, :, :, :self.n_modes] = torch.einsum(
                "bixyz,ioz->boxyz",
                x_ftz[:, :, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :, :self.n_modes] = x_ftz[:, :, :, :, :self.n_modes]
        xz = torch.fft.irfft(out_ft, n=Z, dim=-1, norm='ortho')

        # # # Dimension Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, X, Y // 2 + 1, Z)
        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes, :] = torch.einsum(
                "bixyz,ioy->boxyz",
                x_fty[:, :, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes, :] = x_fty[:, :, :, :self.n_modes, :]
        xy = torch.fft.irfft(out_ft, n=Y, dim=-2, norm='ortho')

        # # # Dimension X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, X // 2 + 1, Y, Z)
        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :, :] = torch.einsum(
                "bixyz,iox->boxyz",
                x_ftx[:, :, :self.n_modes, :, :],
                torch.view_as_complex(self.fourier_weight[2]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :, :] = x_ftx[:, :, :self.n_modes, :, :]
        xx = torch.fft.irfft(out_ft, n=X, dim=-3, norm='ortho')

        # # # Combine Dimensions # # #
        x_combined = xx + xy + xz
        x_combined = rearrange(x_combined, 'b i x y z -> b x y z i')
        return x_combined


class FNOFactorized3DBlock(nn.Module):
    def __init__(self, modes, width, input_dim=5, dropout=0.0, in_dropout=0.0,
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
            for _ in range(3):
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv3d(in_dim=width,
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
per_data_generation_time = 2.711006149959564
per_sample_downsample_time = 0

sub = args.reduced_resolution
reduced_resolution_t = args.reduced_resolution_t

batch_size = args.batch_size
inference_batch_size = args.inference_batch_size
learning_rate = 0.001
scheduler_gamma = 0.5

S = args.grid
T_in = 5
T = 50 - T_in
T_rollout = T + (T_in - 1 if args.single_frame_initialization else 0)

# Base budget calculation
downsample_time = per_sample_downsample_time * ntrain
generation_time = per_data_generation_time * ntrain
train_budget = time_budget - generation_time - downsample_time

if train_budget <= 0:
    print("no train budget")
    exit()

print(f"Base train budget remaining: {train_budget:.2f}s")

################################################################
# load data
################################################################
def load_data(file_index, n):
    u = np.load(args.data_path + f"data_{file_index}_u.npy", mmap_mode="r+")
    fields_u = torch.from_numpy(u).float()

    # 3D Slicing -> shape: (N, X, Y, Z, Time)
    train_x = fields_u[:n, ::sub, ::sub, ::sub, :T_in]
    train_y = fields_u[:n, ::sub, ::sub, ::sub, T_in:T+T_in]
    
    # Leave T_in channels as the final dimension
    train_x = train_x.reshape(n, S//sub, S//sub, S//sub, T_in)
    train_y = train_y.reshape(n, S//sub, S//sub, S//sub, T)

    print(f"File {file_index} data size:", train_x.shape)

    if file_index == "val":
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=inference_batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    return train_loader

t1 = time.time()
train_loader = load_data(0, min(100, ntrain))
print("load train data finished")

val_loader = load_data("val", 100)
print("load val data finished")
print('preprocessing finished, time used:', time.time()-t1)

output_path = f"ffno_KS_3D_budget{int(time_budget)}_n{ntrain}_w{width}_m{modes}_batch{batch_size}"

################################################################
# Checkpointing and Model Init
################################################################
wandb.init(project=args.project_name, group=f"B{int(time_budget)}", name=f"n{ntrain}_B{int(time_budget)}")

model = FNOFactorized3DBlock(
    modes=modes,
    width=width,
    input_dim=T_in,
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
loss_fn = LpLoss(size_average=False)
scaler = torch.cuda.amp.GradScaler() # AMP setup for memory saving

ep = 0
cur_total_time = 0

################################################################
# Dynamic Warmup & Scheduler configuration (Option A: Skip on Resume)
################################################################
model.train()

if not use_checkpoint:
    print("Starting 3D warmup and step time profiling...")
    warmup_steps = 10  # Reduced to 10 for 3D to save initialization time
    warmup_time = 0.0
    steps_completed = 0

    fake_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        xx, yy = apply_single_frame_initialization(
            xx, yy, args.single_frame_initialization, time_dim=-1
        )

        for t in range(T_rollout):
            if steps_completed >= warmup_steps: break
            
            # 3D Sliding Window
            if t == 0:
                inp = xx 
            elif t < T_in:
                inp = torch.cat([xx[..., t:], yy[..., :t]], dim=-1)
            else:
                inp = yy[..., t - T_in:t]

            y = yy[..., t:t+1]

            if noise_std > 0:
                inp = inp + noise_std * torch.randn_like(inp)

            torch.cuda.synchronize()
            t_start = time.time()

            optimizer.zero_grad(set_to_none=True)
            
            out = model(inp)['forecast']
            loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            fake_scheduler.step()

            torch.cuda.synchronize()
            t_end = time.time()

            warmup_time += (t_end - t_start)
            steps_completed += 1

        if steps_completed >= warmup_steps: break

    per_step_time = warmup_time / max(1, warmup_steps)
    print(f"Calculated per_step_time: {per_step_time:.5f}s over {warmup_steps} steps")

    scheduler_step = int((train_budget / per_step_time) * 0.1)
    if scheduler_step == 0: scheduler_step = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    cur_total_time += warmup_time

else:
    print("Checkpoint found! Resuming standard run...")
    ckpt = torch.load(ckpt_pth)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    scheduler.load_state_dict(ckpt["scheduler_state"])
    
    # Attempt to load scaler state if it was saved in the previous run
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    torch.set_rng_state(ckpt["torch_rng_state"])
    ep = ckpt["epoch"]
    cur_total_time = ckpt["train_time"]

gc.collect()
torch.cuda.empty_cache()

################################################################
# Training Loop
################################################################
cur_data_file_loaded = 0
num_files = 50

while cur_total_time <= train_budget:
    ep += 1
    trained_data = 0
    train_step_loss = 0
    model.train()

    for i in range(num_files):
        if cur_data_file_loaded != i and trained_data < ntrain:
            train_loader = load_data(i, min(100, ntrain - trained_data))
            cur_data_file_loaded = i

        for xx, yy in train_loader:
            trained_data += batch_size
            xx = xx.to(device)
            yy = yy.to(device)
            xx, yy = apply_single_frame_initialization(
                xx, yy, args.single_frame_initialization, time_dim=-1
            )

            # Initialize autoregressive state if needed
            if args.use_autoregressive:
                ar_state = xx.clone()  # Start with initial condition

            for t in range(T_rollout):
                if not args.use_autoregressive:
                    # Teacher forcing: use ground truth
                    if t == 0:
                        inp = xx 
                    elif t < T_in:
                        inp = torch.cat([xx[..., t:], yy[..., :t]], dim=-1)
                    else:
                        inp = yy[..., t - T_in:t]
                else:
                    # Autoregressive: use model predictions
                    if t == 0:
                        inp = xx  # Use initial condition
                    else:
                        # Use last T_in timesteps from autoregressive state
                        inp = ar_state[..., -T_in:]

                y = yy[..., t:t+1]

                if noise_std > 0:
                    inp = inp + noise_std * torch.randn_like(inp)

                t_step_s = time.time()

                optimizer.zero_grad(set_to_none=True)
                
                out = model(inp)['forecast']
                loss = loss_fn(out, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()

                # Update autoregressive state with prediction
                if args.use_autoregressive:
                    ar_state = torch.cat([ar_state[..., 1:], out], dim=-1)

                t_step_e = time.time()
                step_time = t_step_e - t_step_s
                cur_total_time += step_time
                train_step_loss += float(loss.item())

            if cur_total_time > train_budget or trained_data >= ntrain:
                break

        if cur_total_time > train_budget or trained_data >= ntrain:
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
            "scaler_state": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_time": cur_total_time,
        }
        torch.save(ckpt, args.save + "/model/" + output_path + "_checkpoint")
        
torch.save(model, args.save + "/model/" + output_path)

################################################################
# Evaluation
################################################################
def traj_l2(preds, targets):
    """
    preds, targets: (B, X, Y, Z, T) 
    """
    diff = preds - targets                       
    B = diff.shape[0]
    diff_flat = diff.reshape(B, -1)
    return torch.norm(diff_flat, p=2, dim=1)     

model.eval()
val_step_loss = 0
worst_loss = -math.inf
worst_idx  = -1
global_base = 0
full_loss_sum = 0
n_eval = 0
dnrmse_sum = 0.0

total_inference_time = 0.0 

with torch.no_grad():
    for xx, yy in val_loader:
        xx, yy = apply_single_frame_initialization(
            xx, yy, args.single_frame_initialization, time_dim=-1
        )
        B = xx.shape[0]
        window = xx # (B, X, Y, Z, T_in)

        preds_list = []
        for t in range(T_rollout):
            inp = window.contiguous().to(device, non_blocking=True) 
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_inf_start = time.time()
            
            out = model(inp)['forecast']                  
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_inf_end = time.time()
            total_inference_time += (t_inf_end - t_inf_start)

            y = yy[..., t:t+1].contiguous().to(device, non_blocking=True)
            val_step_loss += loss_fn(out, y).item()

            preds_list.append(out.detach().cpu())
            out_cpu = out.detach().cpu()
            
            window = torch.cat([window[..., 1:], out_cpu], dim=-1) 

            del inp, out, y

        preds = torch.cat(preds_list, dim=-1)
        per_sample = traj_l2(preds, yy)
        dnrmse_batch = dimensionwise_nrmse(preds, yy, channel_dim=None)

        full_loss_sum += float(loss_fn(preds, yy).item())
        dnrmse_sum += float(dnrmse_batch.sum().item())
        n_eval += B

        batch_worst_loss, batch_worst_j = torch.max(per_sample, dim=0)
        batch_worst_loss = float(batch_worst_loss.item())
        batch_worst_j = int(batch_worst_j.item())

        if batch_worst_loss > worst_loss:
            worst_loss = batch_worst_loss
            worst_idx = global_base + batch_worst_j

        global_base += B

val_step_loss /= (max(1, n_eval) * T_rollout)
test_full_loss = full_loss_sum / max(1, n_eval)
test_dnrmse = dnrmse_sum / max(1, n_eval)

avg_rollout_time_per_traj = total_inference_time / max(1, ntest)

wandb.log({
    "worst_l2_loss": worst_loss,
    "worst_traj_idx": worst_idx,
    "test_step_loss": val_step_loss,
    "test_full_loss": test_full_loss,
    "test/dnrmse": test_dnrmse,
    "total_inference_time": total_inference_time,
    "avg_rollout_time_per_traj": avg_rollout_time_per_traj
})

print(f"val_step_loss: {val_step_loss}")
print(f"test_full_loss: {test_full_loss}")
print(f"test dNRMSE: {test_dnrmse}")
print(f"[TEST] worst trajectory idx = {worst_idx}, worst L2 loss = {worst_loss}")
print(f"[TEST] Total Rollout Time: {total_inference_time:.4f}s")
print(f"[TEST] Avg Rollout Time per Trajectory: {avg_rollout_time_per_traj:.4f}s")
