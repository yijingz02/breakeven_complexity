import h5py
import argparse
import os
import numpy as np

"""
python scOT/data_preprocessing.py \
    --path="data" \
    --file_prefix="train_" \
    --save_name="train_preprocessed_" \
    --file_count=2
"""

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Path to the data to be processed",
)
parser.add_argument(
    "--file_prefix",
    type=str,
    required=True,
    help="Prefix of file name.",
)
parser.add_argument(
    "--file_count",
    type=int,
    required=True,
    help="Number of files to be processed.",
)
parser.add_argument(
    "--save_name",
    type=str,
    required=True,
    help="Save name",
)
parser.add_argument(
    "--frames",
    type=int,
    default=200,
    help="Frames",
)
parser.add_argument(
    "--max_traj_per_file",
    type=int,
    default=2500,
    help="max traj per file",
)
parser.add_argument(
    "--N_val",
    type=int,
    default=100,
    help="N_val",
)

args = parser.parse_args()

frames = args.frames
max_traj_per_file = args.max_traj_per_file

for i in range(args.file_count):
    full_path = f"{args.path}/{args.file_prefix}{i}.mat"

    print(f"Processing file: {full_path}")

    with h5py.File(full_path, "r") as f:
        u_ds = f["u"][:frames, ...]
        v_ds = f["v"][:frames, ...]

        u = u_ds[..., : max_traj_per_file]
        v = v_ds[..., : max_traj_per_file]

        print(u.shape)
        print(v.shape)

        assert u.shape == v.shape

    u = np.transpose(u, (3, 1, 2, 0))
    v = np.transpose(v, (3, 1, 2, 0))

    out_u = f"{args.path}/{args.save_name}{i}_u.npy"
    out_v = f"{args.path}/{args.save_name}{i}_v.npy"

    print(f"Saving {out_u} shape={u.shape}")
    print(f"Saving {out_v} shape={v.shape}\n")

    np.save(out_u, u)
    np.save(out_v, v)

full_path = f"{args.path}/{args.file_prefix}val.mat"
print(f"Processing file: {full_path}")
with h5py.File(full_path, "r") as f:
    u_ds = f["u"][:frames, ...]
    v_ds = f["v"][:frames, ...]

    u = u_ds[..., : max_traj_per_file]
    v = v_ds[..., : max_traj_per_file]

    assert u_ds.shape == v_ds.shape

u = np.transpose(u, (3, 1, 2, 0))
v = np.transpose(v, (3, 1, 2, 0))

out_u = f"{args.path}/{args.save_name}val_u.npy"
out_v = f"{args.path}/{args.save_name}val_v.npy"

print(f"Saving {out_u} shape={u.shape}")
print(f"Saving {out_v} shape={v.shape}")

np.save(out_u, u)
np.save(out_v, v)

print("All files processed.")