import jax
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", False)

from jax import lax
from typing import Callable, Union
import jax.tree_util as jtu
from typing import Any
PyTree = Any

import numpy as np
import torch

import gc
import time

import argparse
parser = argparse.ArgumentParser(description="2D Gray-scott Reaction Diffusion")
parser.add_argument(
    "--save_path",
    type=str,
    default="./",
    help="Path and name to save the data."
)
parser.add_argument(
    "--ndata",
    type=int,
    default=20000,
    help="Number of traj to generate."
)
parser.add_argument(
    "--nfile",
    type=int,
    default=1,
    help="Number of files to store data."
)
parser.add_argument(
    "--grid",
    type=int,
    default=256,
    help="Grid size."
)
parser.add_argument(
    "--simulation_time",
    type=int,
    default=10,
    help="Total physical simulation time (in s)."
)
parser.add_argument(
    "--target_save",
    type=int,
    default=50,
    help="Target saving frame."
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.1,
    help="dt value."
) 
parser.add_argument(
    "--L",
    type=float,
    default=50,
    help="Simulation spatial size."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="batch size."
)
parser.add_argument(
    "--val_size",
    type=int,
    default=1000,
    help="batch size."
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="PRNG seed for init sampling."
)
parser.add_argument(
    "--fold_in_process",
    action="store_true",
    help="If set, fold in jax.process_index() so multi-process runs don't duplicate inits.",
)
args = parser.parse_args()

def rollout_save_step(
    stepper_fn: Union[Callable[[PyTree], PyTree], Callable[[PyTree, PyTree], PyTree]],
    n: int,
    *,
    save_step: int = 1,
    include_init: bool = False,
):
    step = jax.jit(lambda u: stepper_fn(u), donate_argnums=(0,))

    def rollout_stepper_fn(u0: PyTree):
        def advance_k(u):
            def body(_, u_): return step(u_)
            return lax.fori_loop(0, save_step, body, u)

        def outer(u, _):
            u = advance_k(u)
            return u, u

        num = n // save_step
        uT, hist = lax.scan(outer, u0, xs=None, length=num)

        if include_init:
            init = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), u0)
            hist = jtu.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), init, hist)

        leftover = n - num * save_step
        uT = lax.fori_loop(float(0), float(leftover), lambda _, u: step(u), uT)

        return hist

    return rollout_stepper_fn

def sample_u(key):
    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=2
    )(num_points=args.grid, key=key)
    return u_0

def generate_batch(key, gen_stepper, batch_size, steps, save_step):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)

    u_0 = jax.vmap(sample_u)(keys)
    u_0_cpy = u_0.copy()

    batched_rollout = jax.vmap(
        rollout_save_step(gen_stepper, steps, save_step=save_step, include_init=False),
        in_axes=0,
    )
    trajectory = batched_rollout(u_0)
    u_trj = trajectory[:, :, 0, :, :]  # keep only channel 0

    return key, u_trj, u_0_cpy

def spectral_downsample(u, S_new):
    """
    Downsample via spectral interpolation.
    u: [N, T, S, S]
    S_new: new grid size
    to_field: whether turn to field
    returns u_coarse: [N, T, S_new, S_new], either tensor or field
    """

    u = np.asarray(u)
    u = torch.from_numpy(u)

    assert u.ndim == 4

    N, T, S, _ = u.shape
    dims_fft = (2, 3)

    u_fft = torch.fft.fft2(u, dim=dims_fft)
    u_fft = torch.fft.fftshift(u_fft, dim=dims_fft)

    center   = S // 2
    half_new = S_new // 2

    u_fft_crop = u_fft[:, :, (center - half_new):(center + half_new), (center - half_new):(center + half_new)]

    scale = (S_new / S) ** 2
    u_fft_crop = u_fft_crop * scale

    u_fft_crop = torch.fft.ifftshift(u_fft_crop, dim=dims_fft)
    u_coarse = torch.fft.ifft2(u_fft_crop, dim=dims_fft).real

    return u_coarse

ks_stepper = ex.stepper.KuramotoSivashinsky(
    num_spatial_dims=2,
    domain_extent=args.L,
    num_points=args.grid,
    dt=args.dt,
)

per_file = args.ndata // args.nfile
steps = int(args.simulation_time / args.dt)
save_step = max(1, int(steps / args.target_save))

key = jax.random.PRNGKey(args.seed)
if args.fold_in_process:
    key = jax.random.fold_in(key, jax.process_index())

total_time = 0
total_batch = 0

# generate train data
for i in range(args.nfile):
    data_cnt = 0
    u = []

    file_key = jax.random.fold_in(key, i)

    while data_cnt < per_file:
        ts = time.time()
        file_key, u_trj, _u0 = generate_batch(file_key, ks_stepper, args.batch_size, steps, save_step)
        u_trj = spectral_downsample(u_trj, 64)
        te = time.time()

        print(f"step time: {te - ts}")

        u_trj = np.array(u_trj, dtype=np.float32)
        u.append(u_trj)
        data_cnt += args.batch_size

        jax.clear_caches()
        gc.collect()

        total_time += (te - ts)
        total_batch += 1

    u = np.concatenate(u, axis=0)
    u = np.transpose(u, (0, 2, 3, 1))
    out_u = f"{args.save_path}/data_{i}_u.npy"

    print(f"Saving {out_u} shape={u.shape}\n")

    np.save(out_u, u)

# generate val file
data_cnt = 0
u = []
u0 = []

val_key = jax.random.fold_in(key, 10_000_000)

while data_cnt < args.val_size:
    val_key, u_trj, u_0 = generate_batch(val_key, ks_stepper, args.batch_size, steps, save_step)
    u_trj = spectral_downsample(u_trj, 64)
    u_0   = spectral_downsample(u_0, 64)

    u_trj = np.array(u_trj, dtype=np.float32)
    u_0   = np.array(u_0, dtype=np.float32)

    u.append(u_trj)
    u0.append(u_0)

    data_cnt += args.batch_size

    jax.clear_caches()
    gc.collect()

u = np.concatenate(u, axis=0)
u = np.transpose(u, (0, 2, 3, 1))
out_u = f"{args.save_path}/data_val_u.npy"
print(f"Saving {out_u} shape={u.shape}\n")
np.save(out_u, u)

u0 = np.concatenate(u0, axis=0)
u0 = np.transpose(u0, (0, 2, 3, 1))
out_u0 = f"{args.save_path}/data_val_u0.npy"
print(f"Saving {out_u0} shape={u0.shape}\n")
np.save(out_u0, u0)

print(f"\nAverage per batch time: {total_time / total_batch}")
print(f"Average per data time:  {total_time / total_batch / args.batch_size}")