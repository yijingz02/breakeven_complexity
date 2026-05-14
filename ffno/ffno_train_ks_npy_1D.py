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
from torch.profiler import profile, ProfilerActivity, record_function, schedule

import gc
import wandb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description="1D kuramoto sivashinsky simulation")
parser.add_argument("--save", type=str, default="output_isotime", help="File name to save the trained model.")
parser.add_argument("--width", type=int, default=64, help="Model width.")
parser.add_argument("--modes", type=int, default=16, help="Model modes.")
parser.add_argument("--n_layers", type=int, default=24, help="Model layers.")
parser.add_argument("--budget", type=float, default=100, help="Time budget in seconds.")
parser.add_argument("--ndata", type=int, default=1000, help="Training data size.")
parser.add_argument("--data_path", type=str, default="../generated_large_g256_fp32/", help="Path to data.")
parser.add_argument("--grid", type=int, default=64, help="Grid size.")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
parser.add_argument("--reduced_resolution", type=int, default=1, help="Downsample factor for grid size.")
parser.add_argument("--reduced_resolution_t", type=int, default=1, help="Downsample factor for time step (for evaluation).")
parser.add_argument("--use_autoregressive", action="store_true", default=False, help="Use autoregressive rollout during training.")
args = parser.parse_args()

################################################################
# Model (1D FNO)
################################################################
class SpectralConv1d(nn.Module):
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
        # x.shape == [batch_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m i -> b i m')
        # x.shape == [batch_size, in_dim, grid_size]

        B, I, M = x.shape

        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1]

        out_ft = x_ft.new_zeros(B, I, M // 2 + 1)
        
        if self.mode == 'full':
            out_ft[:, :, :self.n_modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes] = x_ft[:, :, :self.n_modes]

        x = torch.fft.irfft(out_ft, n=M, dim=-1, norm='ortho')
        
        x = rearrange(x, 'b i m -> b m i')
        # x.shape == [batch_size, grid_size, out_dim]
        return x

class FNOFactorized1DBlock(nn.Module):
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
            weight = torch.FloatTensor(width, width, modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param, gain=gain)
            self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv1d(in_dim=width,
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
        # x.shape == [n_batches, grid_size, input_size]
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
per_data_generation_time = 0.0003920316696166992
per_sample_downsample_time = 0

sub = args.reduced_resolution
reduced_resolution_t = args.reduced_resolution_t

batch_size = args.batch_size
learning_rate = 0.001
scheduler_gamma = 0.5

S = args.grid
T_in = 5
T = 50 - T_in

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

    # 1D Slicing -> shape: (N, S, T)
    train_x = fields_u[:n, ::sub, :T_in]
    train_y = fields_u[:n, ::sub, T_in:T+T_in]
    
    # Leave T_in channels as the final dimension
    train_x = train_x.reshape(n, S//sub, T_in)
    train_y = train_y.reshape(n, S//sub, T)

    print(f"File {file_index} data size:", train_x.shape)

    if file_index == "val":
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=10, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    return train_loader

t1 = time.time()
train_loader = load_data(0, min(20000, ntrain))
print("load train data finished")

val_loader = load_data("val", 100)
print("load val data finished")
print('preprocessing finished, time used:', time.time()-t1)

output_path = f"ffno_KS_NvsB_budget{int(time_budget)}_n{ntrain}_w{width}_m{modes}_batch{batch_size}"

################################################################
# Checkpointing and Model Init
################################################################
wandb.init(project=f"ffno_ks_1D", group=f"B{int(time_budget)}", name=f"n{ntrain}_B{int(time_budget)}")

model = FNOFactorized1DBlock(
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

ep = 0
cur_total_time = 0

################################################################
# Dynamic Warmup & Scheduler configuration (Option A: Skip on Resume)
################################################################
model.train()

if not use_checkpoint:
    print("Starting warmup and step time profiling...")
    warmup_steps = 20
    warmup_time = 0.0
    steps_completed = 0

    # Fake scheduler to simulate the exact computational overhead
    fake_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(T):
            if steps_completed >= warmup_steps: break
            
            # 1D Sliding Window formulation
            if t == 0:
                inp = xx 
            elif t < T_in:
                inp = torch.cat([xx[..., t:], yy[..., :t]], dim=-1)
            else:
                inp = yy[..., t - T_in:t]

            # Preserve the channel dimension for loss calculation
            y = yy[..., t:t+1]

            if noise_std > 0:
                inp = inp + noise_std * torch.randn_like(inp)

            torch.cuda.synchronize()
            t_start = time.time()

            out = model(inp)['forecast']
            loss = loss_fn(out, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Step the fake scheduler to include its execution time
            fake_scheduler.step()

            torch.cuda.synchronize()
            t_end = time.time()

            warmup_time += (t_end - t_start)
            steps_completed += 1

        if steps_completed >= warmup_steps: break

    per_step_time = warmup_time / max(1, warmup_steps)
    print(f"Calculated per_step_time: {per_step_time:.5f}s over {warmup_steps} steps")

    # Set up the real scheduler based on the empirical calculation
    scheduler_step = int((train_budget / per_step_time) * 0.1)
    if scheduler_step == 0: scheduler_step = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    cur_total_time += warmup_time

else:
    print("Checkpoint found! Resuming standard run (skipping warmup profiling)...")
    ckpt = torch.load(ckpt_pth)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    
    # Initialize a placeholder scheduler so we can overwrite it with the true saved state
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    scheduler.load_state_dict(ckpt["scheduler_state"])
    
    torch.set_rng_state(ckpt["torch_rng_state"])
    ep = ckpt["epoch"]
    cur_total_time = ckpt["train_time"]

gc.collect()
torch.cuda.empty_cache()

################################################################
# Training Loop
################################################################
while cur_total_time <= train_budget:
    ep += 1
    trained_data = 0
    train_step_loss = 0
    model.train()

    for xx, yy in train_loader:
        trained_data += batch_size
        xx = xx.to(device)
        yy = yy.to(device)

        # Initialize autoregressive state if needed
        if args.use_autoregressive:
            ar_state = xx.clone()  # Start with initial condition

        for t in range(T):
            # 1D Sliding Window formulation
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

            out = model(inp)['forecast']
            loss = loss_fn(out, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
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

    if ep % 10 == 0 or cur_total_time > train_budget:
        wandb.log({
            "epoch": ep,
            "step": scheduler.last_epoch,
            "train_time": cur_total_time,
            "lr": optimizer.param_groups[0]['lr'],
            "loss": train_step_loss / trained_data / T,
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

torch.save(model, args.save + "/model/" + output_path)

################################################################
# Evaluation
################################################################
def traj_l2(preds, targets):
    """
    preds, targets: (B, S, T) 
    return: per-sample L2 over (S,T) => shape (B,)
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

# Track purely the forward pass time
total_inference_time = 0.0 

with torch.no_grad():
    for xx, yy in val_loader:
        B = xx.shape[0]
        window = xx # (B, S, T_in)

        preds_list = []
        for t in range(T):
            inp = window.contiguous().to(device, non_blocking=True) 
            
            # --- Start Inference Timer ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inf_start = time.time()
            
            out = model(inp)['forecast']                  
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inf_end = time.time()
            total_inference_time += (t_inf_end - t_inf_start)
            # --- End Inference Timer ---

            # Loss computation (excluded from timer)
            y = yy[..., t:t+1].contiguous().to(device, non_blocking=True)
            val_step_loss += loss_fn(out, y).item()

            preds_list.append(out.detach().cpu())
            out_cpu = out.detach().cpu()
            
            window = torch.cat([window[..., 1:], out_cpu], dim=-1) 

            del inp, out, y

        preds = torch.cat(preds_list, dim=-1)
        per_sample = traj_l2(preds, yy)

        full_loss_sum += float(loss_fn(preds, yy).item())
        n_eval += B

        batch_worst_loss, batch_worst_j = torch.max(per_sample, dim=0)
        batch_worst_loss = float(batch_worst_loss.item())
        batch_worst_j = int(batch_worst_j.item())

        if batch_worst_loss > worst_loss:
            worst_loss = batch_worst_loss
            worst_idx = global_base + batch_worst_j

        global_base += B

val_step_loss /= (max(1, n_eval) * T)
test_full_loss = full_loss_sum / max(1, n_eval)

# Calculate average rollout time per trajectory
avg_rollout_time_per_traj = total_inference_time / max(1, ntest)

wandb.log({
    "worst_l2_loss": worst_loss,
    "worst_traj_idx": worst_idx,
    "test_step_loss": val_step_loss,
    "test_full_loss": test_full_loss,
    "total_inference_time": total_inference_time,
    "avg_rollout_time_per_traj": avg_rollout_time_per_traj
})

print(f"val_step_loss: {val_step_loss}")
print(f"test_full_loss: {test_full_loss}")
print(f"[TEST] worst trajectory idx = {worst_idx}, worst L2 loss = {worst_loss}")
print(f"[TEST] Total Rollout Time: {total_inference_time:.4f}s")
print(f"[TEST] Avg Rollout Time per Trajectory: {avg_rollout_time_per_traj:.4f}s")