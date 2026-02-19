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

class NavierStokesMultiFile(BaseTimeDataset):
    def __init__(
        self,
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

        self.input_dim  = 1 # u * 1 frames
        self.output_dim = 1

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        
        self.N_max = self.num_trajectories + self.N_val + self.N_test

        self.t_loading = 0

        self._current_file = None
        self._u = None
        self._N_local = None

        self.res = 64
        self.resolution = 64

        self.label_description = "[u]"
        self.frames = 10
        self.T_len  = 1

        self.samples_per_traj = self.frames - self.T_len

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

        self.pixel_mask = torch.tensor([False])
        
        self.constants = {
            "mean": torch.tensor([-2.9261311139783255e-10]),
            "std": torch.tensor([0.26406050778680523]),
            "time": 1.0,
        }

        self.post_init()

    def _filepath(self, file_idx):
        u_pth = f"{self.data_path}/{self.file_prefix}{self.file_indices[file_idx]}_u.npy"
        return u_pth

    def _load_file(self, file_idx):
        if self.which == "val" or self.which == "test":
            self._u = None
            gc.collect()

            u_pth = f"{self.data_path}/{self.file_prefix}val_u.npy"
            
            u = np.load(u_pth, mmap_mode="r+")

            self._u = torch.from_numpy(u).float()
            self._N_local = self._u.shape[0]

            self._current_file = file_idx

        elif self.which == "train":
            if self._current_file == file_idx:
                return

            self._u = None
            gc.collect()

            idx = self.file_indices[file_idx]
            u_pth = self._filepath(idx)

            t_ss = time.time()

            u = np.load(u_pth, mmap_mode="r+")

            print(u.shape)

            self._u = torch.from_numpy(u).float()
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
        t2      = t1 + self.T_len

        file_idx  = (traj_id % self.num_trajectories) // self.max_traj_per_file
        local_idx = (traj_id % self.num_trajectories) %  self.max_traj_per_file

        self._load_file(file_idx)

        traj_u = self._u[local_idx]

        u_in  = traj_u[:, :, t1]
        u_out = traj_u[:, :, t2]
        
        inputs = u_in.unsqueeze(0)
        labels = u_out.unsqueeze(0) 

        inputs = (inputs - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]
        labels = (labels - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]

        time = 1.0

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class NavierStokesMultiFileMultiFrame(BaseTimeDataset):
    def __init__(
        self,
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

        self.input_dim  = 5 # u * 5 frames
        self.output_dim = 1

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        
        self.N_max = self.num_trajectories + self.N_val + self.N_test

        self.t_loading = 0

        self._current_file = None
        self._u = None
        self._N_local = None

        self.res = 64
        self.resolution = 64

        self.label_description = "[u]"
        self.frames = 10
        self.T_len  = 5

        self.samples_per_traj = self.frames - self.T_len

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

        self.pixel_mask = torch.tensor([False])
        
        self.constants = {
            "mean": torch.tensor([0.0]),
            "std": torch.tensor([1.0]),
            "time": 1.0,
        }

        self.post_init()

    def _filepath(self, file_idx):
        u_pth = f"{self.data_path}/{self.file_prefix}{self.file_indices[file_idx]}_u.npy"
        return u_pth

    def _load_file(self, file_idx):
        if self.which == "val" or self.which == "test":
            self._u = None
            gc.collect()

            u_pth = f"{self.data_path}/{self.file_prefix}val_u.npy"
            
            u = np.load(u_pth, mmap_mode="r+")

            self._u = torch.from_numpy(u).float()
            self._N_local = self._u.shape[0]

            self._current_file = file_idx

        elif self.which == "train":
            if self._current_file == file_idx:
                return

            self._u = None
            gc.collect()

            idx = self.file_indices[file_idx]
            u_pth = self._filepath(idx)

            t_ss = time.time()

            u = np.load(u_pth, mmap_mode="r+")

            print(u.shape)

            self._u = torch.from_numpy(u).float()
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
        t2      = t1 + self.T_len

        file_idx  = (traj_id % self.num_trajectories) // self.max_traj_per_file
        local_idx = (traj_id % self.num_trajectories) %  self.max_traj_per_file

        self._load_file(file_idx)

        traj_u = self._u[local_idx]

        u_in  = traj_u[:, :, t1:t2]
        u_out = traj_u[:, :, t2]
        
        inputs = u_in.unsqueeze(0)
        labels = u_out.unsqueeze(0) 

        inputs = (inputs - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]
        labels = (labels - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]

        inputs = inputs.permute(0, 3, 1, 2).reshape(self.input_dim, self.resolution, self.resolution)
        time = 1.0

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }
