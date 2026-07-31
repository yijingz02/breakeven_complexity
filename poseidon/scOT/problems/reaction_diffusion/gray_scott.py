import torch
import h5py
import torch
import scipy.io
import h5py
import numpy as np
import gc
import os
from scOT.problems.base import BaseTimeDataset
import time

class GrayScottMultiFile(BaseTimeDataset):
    def __init__(
        self,
        # file_prefix,
        # file_count,
        # max_traj_per_file=None,
        # num_trajectories=None,
        # resolution=64,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.file_count = kwargs.get("file_count")
        self.which = kwargs.get("which")
        self.data_path = kwargs.get("data_path")
        self.file_prefix = kwargs.get("file_prefix")
        self.file_indices = list(range(self.file_count))
        self.max_traj_per_file = kwargs.get("max_traj_per_file")
        self.num_trajectories = kwargs.get("num_trajectories")

        self.history = 10
        self.input_dim = 20 # u,v * 10 frames
        self.output_dim = 2

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        
        self.N_max = self.num_trajectories + self.N_val + self.N_test

        self.t_loading = 0

        self._current_file = None
        self._u = None
        self._v = None
        self._N_local = None

        self.res = 64
        self.resolution = 64

        self.label_description = "[u, v]"
        self.frames = 200
        self.T_len  = 10

        self.samples_per_traj = (
            self.frames - 1
            if self.single_frame_initialization
            else self.frames - self.T_len
        )

        self._traj_per_file = self.max_traj_per_file

        self._load_file(0)

        total_capacity = self.max_traj_per_file * len(self.file_indices)

        if self.num_trajectories is None:
            self._len = total_capacity
        else:
            self._len = min(self.num_trajectories, total_capacity)

        if self.max_traj_per_file is None:
            self.max_traj_per_file = self._N_local
        else:
            self.max_traj_per_file = min(self.max_traj_per_file, self._N_local)

        self._traj_per_file = self.max_traj_per_file
        self._len = self._traj_per_file * len(self.file_indices)

        self.pixel_mask = torch.tensor([False, False])
        
        self.constants = {
            "mean": torch.tensor([0.0, 0.0]),
            "std": torch.tensor([1.0, 1.0]),
            "time": 10.0,
        }

        self.post_init()

    def _filepath(self, file_idx):
        u_pth = f"{self.data_path}/{self.file_prefix}{self.file_indices[file_idx]}_u.npy"
        v_pth = f"{self.data_path}/{self.file_prefix}{self.file_indices[file_idx]}_v.npy"
        return u_pth, v_pth

    def _load_file(self, file_idx):
        if self.which == "val" or self.which == "test":
            self._u = None
            self._v = None
            gc.collect()

            u_pth = f"{self.data_path}/{self.file_prefix}val_u.npy"
            v_pth = f"{self.data_path}/{self.file_prefix}val_v.npy"
        
            u = np.load(u_pth, mmap_mode="r+")
            v = np.load(v_pth, mmap_mode="r+")

            self._u = torch.from_numpy(u).float()
            self._v = torch.from_numpy(v).float()
            self._N_local = self._u.shape[0]

            self._current_file = file_idx

        elif self.which == "train":
            if self._current_file == file_idx:
                return

            self._u = None
            self._v = None
            gc.collect()

            idx = self.file_indices[file_idx]
            u_pth, v_pth = self._filepath(idx)

            t_ss = time.time()

            u = np.load(u_pth, mmap_mode="r+")
            v = np.load(v_pth, mmap_mode="r+")

            # print(u.shape)

            self._u = torch.from_numpy(u).float()
            self._v = torch.from_numpy(v).float()
            self._N_local = self._u.shape[0]

            t_ee = time.time()

            t_l = t_ee - t_ss
            self.t_loading += t_l

            # print(f"Data loading time: {t_l}")
            # print(f"Accumulative time: {self.t_loading}")

            self._current_file = file_idx

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        traj_id = idx // self.samples_per_traj
        t1      = idx %  self.samples_per_traj
        t2      = t1 + (1 if self.single_frame_initialization else self.T_len)

        file_idx  = (traj_id % self.num_trajectories) // self.max_traj_per_file
        local_idx = (traj_id % self.num_trajectories) %  self.max_traj_per_file

        # print("idx:", idx)
        # print("local idx:", local_idx)
        # print("file idx", file_idx)
        # print(f"t1: {t1}")

        self._load_file(file_idx)

        traj_u = self._u[local_idx]
        traj_v = self._v[local_idx]

        if self.single_frame_initialization:
            indices = [max(0, t1 - self.history + 1 + i) for i in range(self.history)]
            u_in = traj_u[:, :, indices]
            v_in = traj_v[:, :, indices]
        else:
            u_in = traj_u[:, :, t1:t2]
            v_in = traj_v[:, :, t1:t2]
        u_out = traj_u[:, :, t2]
        v_out = traj_v[:, :, t2]

        inputs = torch.stack([u_in,  v_in], dim=0)
        labels = torch.stack([u_out, v_out], dim=0)

        inputs = (inputs - self.constants["mean"][:, None, None, None]) / self.constants["std"][:, None, None, None]
        labels = (labels - self.constants["mean"][:, None, None])       / self.constants["std"][:, None, None]

        if self.single_frame_initialization:
            inputs = inputs.permute(3, 0, 1, 2).reshape(
                self.input_dim, self.resolution, self.resolution
            )
        else:
            inputs = inputs.permute(0, 3, 1, 2).reshape(
                self.input_dim, self.resolution, self.resolution
            )
        time = 10.0

        # print("inputs shape:", inputs.shape)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }
