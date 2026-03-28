import jax
import exponax as ex
import jax.numpy as jnp
from jax import config, lax
config.update("jax_enable_x64", False)

import jax.tree_util as jtu
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="1D Kuramoto-Sivashinsky")
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--ndata", type=int, default=128)
parser.add_argument("--grid", type=int, default=64)
parser.add_argument("--simulation_time", type=int, default=10)
parser.add_argument("--target_save", type=int, default=50)
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--L", type=float, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_test", type=int, default=20)
parser.add_argument("--grid_low", type=int, default=8)
parser.add_argument("--grid_high_test", type=int, default=64)
parser.add_argument("--dt_low", type=float, default=0.002)
parser.add_argument("--dt_high_test", type=float, default=0.2)
args = parser.parse_args()

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print()


def rollout(stepper_fn, total_steps, save_step):
    step = jax.jit(lambda u: stepper_fn(u), donate_argnums=(0,))

    def fn(u0):
        def advance(u):
            return lax.fori_loop(0, save_step, lambda _, u_: step(u_), u)
        def scan_body(u, _):
            u = advance(u)
            return u, u
        n_saves = total_steps // save_step
        _, hist = lax.scan(scan_body, u0, xs=None, length=n_saves)
        return hist

    return fn


def sample_u(key, num_points):
    return ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=1,
    )(num_points=num_points, key=key)


def spectral_resample(u, S_new):
    S = u.shape[-1]
    if S_new == S:
        return u.copy()
    u_rfft = np.fft.rfft(u, axis=-1)
    if S_new < S:
        u_rfft_new = u_rfft[..., :S_new // 2 + 1].copy()
    else:
        pad = [(0, 0)] * (u.ndim - 1) + [(0, S_new // 2 + 1 - u_rfft.shape[-1])]
        u_rfft_new = np.pad(u_rfft, pad, mode='constant')
    u_rfft_new *= S_new / S
    return np.fft.irfft(u_rfft_new, n=S_new, axis=-1)


def run_ks(u0_np, grid, L, dt, save_times):
    stepper = ex.stepper.KuramotoSivashinsky(
        num_spatial_dims=1, domain_extent=L, num_points=grid, dt=dt,
    )
    step_fn = jax.jit(lambda u: stepper(u))
    u0 = jnp.array(u0_np)

    save_at_sorted = sorted(set(int(round(t / dt)) for t in save_times))
    prev = 0
    increments = []
    for s in save_at_sorted:
        increments.append(s - prev)
        prev = s
    increments_arr = jnp.array(increments)

    def single_rollout(u0_single):
        def body(u, inc):
            u = lax.fori_loop(0, inc, lambda _, u_: step_fn(u_), u)
            return u, u
        _, saves = lax.scan(body, u0_single, increments_arr)
        return saves

    traj = jax.vmap(single_rollout)(u0)
    return np.array(traj)


def nrmse(pred, true):
    mse = np.mean((pred - true) ** 2, axis=(1, 2))
    var = np.var(true, axis=(1, 2))
    return np.mean(np.sqrt(mse / (var + 1e-12)))


def make_pairs(n, grid_high, grid_low, dt_low, dt_high, save_interval):
    pairs = []
    for i in range(n):
        frac = i / (n - 1)
        grid = round(grid_high + frac * (grid_low - grid_high))
        dt_raw = dt_low + frac * (dt_high - dt_low)
        k = max(1, round(save_interval / dt_raw))
        dt = save_interval / k
        pairs.append((grid, dt))
    return pairs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

save_interval = args.simulation_time / args.target_save
save_times = [(i + 1) * save_interval for i in range(args.target_save)]

print(f"=== 1D KS Resolution Study ===")
print(f"L={args.L}, grid={args.grid}, dt={args.dt}, T={args.simulation_time}")
print(f"ndata={args.ndata}, save_interval={save_interval}")
print()

key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, args.ndata)
u0_high = np.array(jax.vmap(lambda k: sample_u(k, args.grid))(keys))
print(f"Generated {args.ndata} ICs at grid={args.grid}")

print(f"Running ground truth: grid={args.grid}, dt={args.dt} ...")
ts = time.time()
gt_traj = run_ks(u0_high, args.grid, args.L, args.dt, save_times)
te = time.time()
gt_strip = gt_traj[:, :, 0, :]
print(f"  done in {te-ts:.1f}s, shape={gt_traj.shape}")
print()

pairs = make_pairs(args.n_test, args.grid_high_test, args.grid_low,
                   args.dt_low, args.dt_high_test, save_interval)

print(f"{'grid':>6s} {'dt':>10s} {'steps':>8s} {'nRMSE':>12s}")
print("-" * 42)

for grid_t, dt_t in pairs:
    n_steps = int(round(args.simulation_time / dt_t))

    u0_t = spectral_resample(u0_high[:, 0, :], grid_t)[:, np.newaxis, :]

    try:
        traj_t = run_ks(u0_t, grid_t, args.L, dt_t, save_times)
    except Exception as e:
        print(f"{grid_t:6d} {dt_t:10.5f} {n_steps:8d} {'FAILED':>12s}  ({e})")
        jax.clear_caches()
        continue

    strip_t = traj_t[:, :, 0, :]
    if not np.all(np.isfinite(strip_t)):
        print(f"{grid_t:6d} {dt_t:10.5f} {n_steps:8d} {'NaN/Inf':>12s}")
        jax.clear_caches()
        continue

    strip_up = spectral_resample(strip_t, args.grid)
    nf = min(strip_up.shape[1], gt_strip.shape[1])
    err = nrmse(strip_up[:, :nf, :], gt_strip[:, :nf, :])
    print(f"{grid_t:6d} {dt_t:10.5f} {n_steps:8d} {err:12.6f}")

    del traj_t, u0_t, strip_t, strip_up
    jax.clear_caches()
