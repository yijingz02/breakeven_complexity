import os
import sys
from collections import defaultdict
import h5py
import numpy as np
from flowgen import array2gif, array2png


def nrmse(pred, ref):
    return np.sqrt(np.square(pred-ref).sum() / np.square(ref).sum())

def flipped(array, q):
    output = np.flip(array.copy(), axis=-1)
    if q in {'vorticity', 'velocity_y'}:
        output *= -1.0
    elif q == 'vxvyp':
        n = output.shape[-1]
        output[...,n:-n,:] *= -1.0
    return output

def minflip_nrmse(pred, ref, q):
    return min(nrmse(pred, flipped(ref, q)), nrmse(pred, ref))

def avgflip_nrmse(pred, ref, q):
    return nrmse(pred + flipped(pred, q), ref + flipped(ref, q))


basedir = sys.argv[1] if len(sys.argv) > 1 else 'timeavg-Re200'
baseres = sys.argv[2] if len(sys.argv) > 2 else '1.0'
Re = int(basedir.split('Re')[1])

quantities = ['vorticity', 'velocity_x', 'velocity_y', 'pressure']
vxvyp_quantities = ['velocity_x', 'velocity_y', 'pressure']
print_quantities = ['vorticity', 'vxvyp']   # velocity_x, velocity_y, pressure printing disabled
EXPECTED_T = 201
LAST_N = 180                                # time-averaged metrics use last 180 timesteps
INST_SLICE = slice(1, 101 - int(Re/2.5))      # instantaneous metrics use timesteps 1..20

# Discover seeds (top-level subdirs of timeavg/)
seed_dirs = sorted(
    d for d in os.listdir(basedir)
    if os.path.isdir(os.path.join(basedir, d))
)
print(f'Simulation directory: {basedir}')
print(f'Base resolution: {baseres}')
print(f'Found seeds: {seed_dirs}')

# For each seed, discover coarse_size subdirs with complete data
seed_coarse = {}  # seed -> sorted list of (coarse_size_float, dirname)
for seed in seed_dirs:
    seed_path = os.path.join(basedir, seed)
    entries = []
    for d in os.listdir(seed_path):
        h5path = os.path.join(seed_path, d, 'flow.h5')
        if not os.path.isfile(h5path):
            continue
        with h5py.File(h5path, 'r') as f:
            nt = f[quantities[0]].shape[0]
            if nt == EXPECTED_T:
                entries.append((float(d), d))
            else:
                print(f'Skipping seed {seed}/{d}: has {nt} timesteps')
    entries.sort()
    seed_coarse[seed] = entries

# Collect union of all coarse_sizes
all_coarse = sorted(set(cs for entries in seed_coarse.values() for cs, _ in entries))

# For each seed, load reference (coarse_size=1.0)
ref_by_seed = {}
for seed in seed_dirs:
    ref_path = os.path.join(basedir, seed, baseres, 'flow.h5')
    if not os.path.isfile(ref_path):
        print(f'WARNING: no reference ({baseres}) for seed {seed}, skipping seed')
        continue
    with h5py.File(ref_path, 'r') as f:
        mask = np.array(f['mask'])
        data = {q: (np.array(f[q]) * mask).transpose(0, 2, 1) for q in quantities}
    data['vxvyp'] = np.hstack([data[q] for q in vxvyp_quantities])
    ref_by_seed[seed] = data

all_coarse = [cs for cs in all_coarse if float(cs) >= float(baseres)]

active_seeds = [s for s in seed_dirs if s in ref_by_seed]
print(f'Active seeds (with reference): {active_seeds}\n')

# Table header
tavg_cols = [f'tavg_{q}' for q in print_quantities]
inst_cols = [f'inst_{q}' for q in print_quantities]
all_cols = tavg_cols + inst_cols
header = f"{'coarse_size':>14s}"
for c in all_cols:
    header += f"  {c:>20s}"
header += f"  {'wallclock':>12s}  {'n_seeds':>7s}"
print(header)
print('-' * len(header))

for cs in all_coarse:
    # Accumulate per-seed results
    seed_tavg_nrmse = defaultdict(list)
    seed_inst_nrmse = defaultdict(list)
    wallclocks = []

    for seed in active_seeds:
        matches = [dn for cv, dn in seed_coarse.get(seed, []) if cv == cs]
        if not matches:
            continue
        dname = matches[0]
        h5path = os.path.join(basedir, seed, dname, 'flow.h5')
        ref = ref_by_seed[seed]

        with h5py.File(h5path, 'r') as f:
            mask = np.array(f['mask'])
            data = {q: (np.array(f[q]) * mask).transpose(0, 2, 1) for q in quantities}
            wallclock = float(np.array(f['wallclock']))
        data['vxvyp'] = np.hstack([data[q] for q in vxvyp_quantities])

        for q in print_quantities:
            seed_tavg_nrmse[q].append(avgflip_nrmse(data[q][-LAST_N:].mean(axis=0), ref[q][-LAST_N:].mean(axis=0), q))
            #seed_tavg_nrmse[q].append(nrmse(data[q][-LAST_N:].mean(axis=0), ref[q][-LAST_N:].mean(axis=0)))

        for q in print_quantities:
            seed_inst_nrmse[q].append(minflip_nrmse(data[q][INST_SLICE], ref[q][INST_SLICE], q))
            #seed_inst_nrmse[q].append(nrmse(data[q][INST_SLICE], ref[q][INST_SLICE]))

        wallclocks.append(wallclock)

    if not wallclocks:
        continue

    n_seeds = len(wallclocks)
    row = f"{cs:>14.4f}"
    for q in print_quantities:
        row += f"  {np.mean(seed_tavg_nrmse[q]):>20.6f}"
    for q in print_quantities:
        row += f"  {np.mean(seed_inst_nrmse[q]):>20.6f}"
    row += f"  {np.mean(wallclocks):>12.1f}  {n_seeds:>7d}"
    print(row)
