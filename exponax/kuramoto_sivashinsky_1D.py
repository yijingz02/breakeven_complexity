import jax
import exponax as ex
import jax.numpy as jnp
from jax import config, lax
import jax.tree_util as jtu
from typing import Callable, Union, Any

config.update("jax_enable_x64", False)

import numpy as np
import gc
import time
import argparse
import math

# ==========================================
# 1. SETUP & ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="1D KS: High-Res Save & Low-Res Benchmark")
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--ndata", type=int, default=20000)
parser.add_argument("--nfile", type=int, default=1)
parser.add_argument("--grid", type=int, default=64, help="Ground Truth Grid Size")
parser.add_argument("--simulation_time", type=float, default=10.0)
parser.add_argument("--target_save", type=int, default=50)
parser.add_argument("--L", type=float, default=50.0)
parser.add_argument("--batch_size_gen", type=int, default=1000, help="Batch size for data generation")
parser.add_argument("--batch_size_bench", type=int, default=1000, help="Batch size for benchmark to saturate GPU")
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fold_in_process", action="store_true")
args = parser.parse_args()

PyTree = Any

# ==========================================
# 2. CORE UTILITIES
# ==========================================

def get_stepper(grid_size, dt):
    return ex.stepper.KuramotoSivashinsky(
        num_spatial_dims=1,
        domain_extent=args.L,
        num_points=grid_size,
        dt=dt,
    )

def rollout_save_step(stepper_fn, n, save_step=1):
    step = jax.jit(lambda u: stepper_fn(u), donate_argnums=(0,))
    def rollout_stepper_fn(u0):
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
        num_spatial_dims=1
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
    return key, trajectory[:, :, 0, :], u_0

# ==========================================
# 3. SPECTRAL OPERATORS & METRICS
# ==========================================

def spectral_resample(u, S_new):
    """
    Unified real FFT resampling function replacing the complex torch FFTs.
    Handles both downsampling and upsampling flawlessly for real signals.
    """
    S = u.shape[-1]
    if S_new == S:
        return u.copy()
    u_rfft = np.fft.rfft(u, axis=-1)
    if S_new < S:
        # Downsample: truncate high frequencies
        u_rfft_new = u_rfft[..., :S_new // 2 + 1].copy()
    else:
        # Upsample: pad with zeros for high frequencies
        pad = [(0, 0)] * (u.ndim - 1) + [(0, S_new // 2 + 1 - u_rfft.shape[-1])]
        u_rfft_new = np.pad(u_rfft, pad, mode='constant')
    u_rfft_new *= S_new / S
    return np.fft.irfft(u_rfft_new, n=S_new, axis=-1)

def calc_nrmse(true, pred):
    """
    Updated to compute global spatiotemporal variance across the trajectory
    instead of per-timestep spatial norms, smoothing out chaotic fluctuations.
    """
    mse = np.mean((pred - true) ** 2, axis=(1, 2))
    var = np.var(true, axis=(1, 2))
    return np.mean(np.sqrt(mse / (var + 1e-12)))

# ==========================================
# 4. EXECUTION: TRAINING DATA (FULL RESOLUTION)
# ==========================================
# Calculate evaluation interval based on total simulation time and targets
dt_save = args.simulation_time / args.target_save

gt_target_dt = 0.002
gt_steps_per_save = max(1, int(dt_save / gt_target_dt))
gt_actual_dt = dt_save / gt_steps_per_save
gt_total_steps = gt_steps_per_save * args.target_save

high_res_stepper = get_stepper(args.grid, gt_actual_dt)

key = jax.random.PRNGKey(args.seed)
if args.fold_in_process:
    key = jax.random.fold_in(key, jax.process_index())

total_time = 0.0
total_batch = 0

print(f"Generating training data at native resolution: {args.grid} with dt={gt_actual_dt:.5f}")
for i in range(args.nfile):
    u_list = []
    data_cnt = 0
    file_key = jax.random.fold_in(key, i)
    per_file = args.ndata // args.nfile

    while data_cnt < per_file:
        ts = time.time()
        file_key, u_trj, _ = generate_batch(
            file_key, high_res_stepper, args.batch_size_gen, 
            gt_total_steps, gt_steps_per_save, args.grid
        )
        te = time.time()

        total_time += (te - ts)
        total_batch += 1

        print(f"File {i} | Batch {data_cnt}/{per_file} | Time: {te-ts:.2f}s")
        u_list.append(np.transpose(np.array(u_trj), (0, 2, 1)).astype(np.float32))
        data_cnt += args.batch_size_gen

    np.save(f"{args.save_path}/data_{i}_u.npy", np.concatenate(u_list, axis=0))
    del u_list; gc.collect()

print("\nTraining data generation complete.\n")

print(f"Average per batch time: {total_time / total_batch}")
print(f"Average per data time:  {total_time / args.ndata}")

# ==========================================
# 5. EXECUTION: BENCHMARKING LOW-RES SIMS
# ==========================================
print("\n" + "="*50 + "\nRUNNING LOW-RES SIMULATION BENCHMARK\n" + "="*50)

target_resolutions = list(range(8, args.grid + 1, 2))
bench_stats = {res: {"time": 0.0, "error": 0.0, "dt": 0.0} for res in target_resolutions}

val_key = jax.random.fold_in(key, 10_000_000)
data_cnt = 0

# List to store the ground truth validation data
val_u_list = []

while data_cnt < args.val_size:
    # A. Ground Truth at full res
    val_key, u_gt, u_0_gt = generate_batch(
        val_key, high_res_stepper, args.batch_size_bench, 
        gt_total_steps, gt_steps_per_save, args.grid
    )
    u_gt_np = np.array(u_gt)
    
    # Save ground truth to the validation list
    val_u_list.append(np.transpose(u_gt_np, (0, 2, 1)).astype(np.float32))

    for res in target_resolutions:
        # B. Downsample Init Condition
        u_0_low = jnp.array(spectral_resample(u_0_gt, res))

        # Max dt_low capped at 0.2 to prevent numerical instability
        dt_low = 0.2
        fraction = (args.grid - res) / (args.grid - 8) if res != args.grid else 0.0
        target_dt = gt_target_dt + fraction * (dt_low - gt_target_dt)

        # Force dt to perfectly align with save steps
        steps_per_save = max(1, int(dt_save / target_dt))
        actual_dt = dt_save / steps_per_save
        total_steps = steps_per_save * args.target_save
        
        bench_stats[res]["dt"] = actual_dt

        # C. Run Sim
        low_stepper = get_stepper(res, actual_dt)
        rollout_fn = jax.vmap(rollout_save_step(low_stepper, total_steps, save_step=steps_per_save))
        
        ts = time.time()
        u_low_sim = rollout_fn(u_0_low)
        jax.block_until_ready(u_low_sim)
        te = time.time()
        
        # D. Record Stats
        u_low_np = np.array(u_low_sim[:, :, 0, :])
        u_recon = spectral_resample(u_low_np, args.grid)
        
        bench_stats[res]["time"] += (te - ts)
        bench_stats[res]["error"] += calc_nrmse(u_gt_np, u_recon)

    data_cnt += args.batch_size_bench
    print(f"Benchmark Progress: {data_cnt}/{args.val_size}")

# Save the accumulated validation data to disk
print("\nSaving validation ground truth data...")
np.save(f"{args.save_path}/data_val_u.npy", np.concatenate(val_u_list, axis=0))
del val_u_list; gc.collect()

# ==========================================
# 6. FINAL RESULTS
# ==========================================
print("\n--- FINAL BENCHMARK RESULTS (PER TRAJECTORY) ---")
print(f"{'Resolution':<12} | {'Actual dt':<12} | {'Time/Data (s)':<18} | {'Avg nRMSE':<18}")
print("-" * 65)

total_samples = args.val_size 
num_batches = args.val_size / args.batch_size_bench

for res in target_resolutions:
    avg_t_per_data = bench_stats[res]["time"] / total_samples
    avg_e_total = bench_stats[res]["error"] / num_batches
    actual_dt_used = bench_stats[res]["dt"]
    
    print(f"{res:<12d} | {actual_dt_used:<12.5f} | {avg_t_per_data:<18.6f} | {avg_e_total:<18.6f}")

print(f"\nBenchmark completed with {total_samples} trajectories.")