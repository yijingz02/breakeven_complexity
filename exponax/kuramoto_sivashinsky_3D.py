import jax
import exponax as ex
import jax.numpy as jnp
from jax import config, lax
import jax.tree_util as jtu
from typing import Callable, Union, Any

config.update("jax_enable_x64", False)

import numpy as np
import torch
import torch.nn.functional as F
import gc
import time
import argparse

# ==========================================
# 1. SETUP & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="3D KS: Low-Res Simulation Benchmark")
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--ndata", type=int, default=5000)
parser.add_argument("--nfile", type=int, default=50)
parser.add_argument("--grid", type=int, default=64, help="Ground Truth Grid")
parser.add_argument("--simulation_time", type=int, default=10)
parser.add_argument("--target_save", type=int, default=50)
parser.add_argument("--dt", type=float, default=0.001) 
parser.add_argument("--L", type=float, default=50.0)
parser.add_argument("--batch_size", type=int, default=1) # Smaller batches for 3D stability
parser.add_argument("--val_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fold_in_process", action="store_true")
args = parser.parse_args()

PyTree = Any

# ==========================================
# 2. CORE UTILITIES
# ==========================================

def get_stepper(grid_size):
    return ex.stepper.KuramotoSivashinsky(
        num_spatial_dims=3,
        domain_extent=args.L,
        num_points=grid_size,
        dt=args.dt,
    )

def rollout_save_step(stepper_fn, n, save_step=1):
    step = jax.jit(lambda u: stepper_fn(u), donate_argnums=(0,))
    def rollout_stepper_fn(u0: PyTree):
        def advance_k(u):
            return lax.fori_loop(0, save_step, lambda _, u_: step(u_), u)
        def outer(u, _):
            u = advance_k(u)
            return u, u
        num = n // save_step
        uT, hist = lax.scan(outer, u0, xs=None, length=num)
        return hist
    return rollout_stepper_fn

def sample_u(key, grid_size):
    return ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=3
    )(num_points=grid_size, key=key)

def generate_batch(key, stepper, batch_size, steps, save_step, grid_size):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    u_0 = jax.vmap(lambda k: sample_u(k, grid_size))(keys)
    
    batched_rollout = jax.vmap(
        rollout_save_step(stepper, steps, save_step=save_step),
        in_axes=0,
    )
    trajectory = batched_rollout(u_0)
    # 3D trajectory shape returned from here is [N, T, 1, X, Y, Z] -> slice to [N, T, X, Y, Z]
    return key, trajectory[:, :, 0, :, :, :], u_0

# ==========================================
# 3. 3D SPECTRAL OPERATORS
# ==========================================

def spectral_downsample_3d(u, S_new):
    """Downsample [N, T, S, S, S] to [N, T, S_new, S_new, S_new]"""
    u = torch.from_numpy(np.array(u))
    N, T, S = u.shape[0], u.shape[1], u.shape[2]
    dims = (2, 3, 4)
    
    u_fft = torch.fft.fftshift(torch.fft.fftn(u, dim=dims), dim=dims)
    center, half = S // 2, S_new // 2
    start, end = center - half, center + half
    
    u_fft_crop = u_fft[:, :, start:end, start:end, start:end] * ((S_new / S)**3)
    u_coarse = torch.fft.ifftn(torch.fft.ifftshift(u_fft_crop, dim=dims), dim=dims).real
    return u_coarse.numpy()

def spectral_upsample_3d(u, S_orig):
    """Upsample [N, T, S_low, S_low, S_low] back to S_orig"""
    u = torch.from_numpy(np.array(u))
    N, T, S_low = u.shape[0], u.shape[1], u.shape[2]
    dims = (2, 3, 4)
    
    u_fft = torch.fft.fftshift(torch.fft.fftn(u, dim=dims), dim=dims)
    pad = (S_orig - S_low) // 2
    # Padding in 3D: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    u_fft_pad = F.pad(u_fft, (pad, pad, pad, pad, pad, pad)) * ((S_orig / S_low)**3)
    u_fine = torch.fft.ifftn(torch.fft.ifftshift(u_fft_pad, dim=dims), dim=dims).real
    return u_fine.numpy()

def calc_nrmse(true, pred):
    true_f = true.reshape(true.shape[0], true.shape[1], -1)
    pred_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
    diff_norm = np.linalg.norm(true_f - pred_f, axis=-1)
    true_norm = np.linalg.norm(true_f, axis=-1)
    
    # Calculate per-sample error
    error = diff_norm / (true_norm + 1e-8)
    
    # Safely ignore any nan trajectories
    return np.nanmean(error)

# ==========================================
# 4. DATA GENERATION (GROUND TRUTH)
# ==========================================
ks_stepper = get_stepper(args.grid)
steps = int(args.simulation_time / args.dt)
save_step = max(1, int(steps / args.target_save))

key = jax.random.PRNGKey(args.seed)
if args.fold_in_process:
    key = jax.random.fold_in(key, jax.process_index())

total_time = 0
total_batch = 0

for i in range(args.nfile):
    data_cnt = 0
    u_list = []
    file_key = jax.random.fold_in(key, i)
    per_file = args.ndata // args.nfile

    while data_cnt < per_file:
        ts = time.time()
        file_key, u_trj, _ = generate_batch(file_key, ks_stepper, args.batch_size, steps, save_step, args.grid)
        te = time.time()
        
        u_list.append(np.array(u_trj, dtype=np.float32))
        data_cnt += args.batch_size
        total_time += (te - ts)
        total_batch += 1
        print(f"File {i} | Step Time: {te-ts:.2f}s")
        jax.clear_caches(); gc.collect()

    u_final = np.concatenate(u_list, axis=0)
    u_final = np.transpose(u_final, (0, 2, 3, 4, 1)) # [N, X, Y, Z, T]
    np.save(f"{args.save_path}/data_{i}_u.npy", u_final)
    del u_list, u_final; gc.collect()

print("\nTraining data generation complete.\n")

print(f"Average per batch time: {total_time / total_batch}")
print(f"Average per data time:  {total_time / args.ndata}")

# ==========================================
# 5. BENCHMARKING LOW-RES 3D SIMS
# ==========================================
print("\n" + "="*50 + "\nRUNNING 3D LOW-RES SIMULATION BENCHMARK\n" + "="*50)
# Note: Keep resolutions even and compatible with grid size
target_resolutions = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
bench_stats = {res: {"time": 0.0, "error": 0.0} for res in target_resolutions}

val_key = jax.random.fold_in(key, 10_000_000)
data_cnt = 0

# Added list to accumulate validation data
val_u_list = []

while data_cnt < args.val_size:
    val_key, u_gt, u_0_gt = generate_batch(val_key, ks_stepper, args.batch_size, steps, save_step, args.grid)
    u_gt_np = np.array(u_gt)
    
    # Append ground truth data to validation list
    val_u_list.append(u_gt_np.astype(np.float32))

    for res in target_resolutions:
        # 1. Downsample Init
        u_0_low = spectral_downsample_3d(u_0_gt, res)
        u_0_low_jax = jnp.array(u_0_low)

        # 2. Sim
        low_stepper = get_stepper(res)
        rollout_fn = jax.vmap(rollout_save_step(low_stepper, steps, save_step=save_step))
        
        ts = time.time()
        u_low_sim = rollout_fn(u_0_low_jax)
        jax.block_until_ready(u_low_sim)
        te = time.time()
        
        # 3. Stats
        u_low_np = np.array(u_low_sim[:, :, 0, :, :, :])
        u_recon = spectral_upsample_3d(u_low_np, args.grid)
        
        bench_stats[res]["time"] += (te - ts)
        bench_stats[res]["error"] += calc_nrmse(u_gt_np, u_recon)
        jax.clear_caches(); gc.collect()

    data_cnt += args.batch_size
    print(f"Benchmark Progress: {data_cnt}/{args.val_size}")

# Added section to format and save the validation data
print("\nSaving validation ground truth data...")
val_u_final = np.concatenate(val_u_list, axis=0)
val_u_final = np.transpose(val_u_final, (0, 2, 3, 4, 1)) # Transpose to [N, X, Y, Z, T]
np.save(f"{args.save_path}/data_val_u.npy", val_u_final)
del val_u_list, val_u_final; gc.collect()

# ==========================================
# 6. RESULTS LOG
# ==========================================
print("\n--- FINAL BENCHMARK RESULTS (PER TRAJECTORY) ---")
print(f"{'Resolution':<12} | {'Time/Data (s)':<18} | {'Avg nRMSE':<18}")
print("-" * 55)

total_samples = args.val_size 
num_batches = args.val_size / args.batch_size

for res in target_resolutions:
    avg_t_per_data = bench_stats[res]["time"] / total_samples
    avg_e_total = bench_stats[res]["error"] / num_batches
    
    print(f"{res:<12d} | {avg_t_per_data:<18.6f} | {avg_e_total:<18.6f}")

print(f"\nBenchmark completed with {total_samples} trajectories.")