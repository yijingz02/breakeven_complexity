import torch
import h5py

import scipy.io
import h5py
import numpy as np
import gc
import os
from scOT.problems.base import BaseTimeDataset
import time

from typing import Tuple


import gc
import time

import numpy as np
import torch

from scOT.problems.base import BaseTimeDataset


class MultiObstacleVorticity(BaseTimeDataset):
    """
    Vorticity dataset with "paper-style" mask handling:

      - mask is an INPUT CHANNEL (geometry encoding)
      - we DO NOT use mask to form pixel_mask / loss masking
      - we normalize only vorticity, not the mask channel

    Expected arrays:
      vorticity: (N,T,H,W)   (your current code uses traj_u[t])
      mask:     (N,H,W) boolean-ish

    Shapes:
      pixel_values: (1 + (1 if use_mask else 0), H, W)  -> [u, mask?]
      labels:       (1, H, W)                           -> [u_next]
      pixel_mask:   (1, H, W) all False
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_count = int(kwargs.get("file_count"))
        self.which = kwargs.get("which")  # "train" | "val" | "test"
        self.data_path = kwargs.get("data_path")
        self.file_prefix = kwargs.get("file_prefix", "")
        self.file_indices = list(range(self.file_count))

        self.max_traj_per_file = kwargs.get("max_traj_per_file")
        self.num_trajectories = kwargs.get("num_trajectories")

        self.use_mask = bool(kwargs.get("use_mask", True))
        self.mask_is_obstacle_true = bool(kwargs.get("mask_is_obstacle_true", True))

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        self.N_max = (self.num_trajectories or 0) + self.N_val + self.N_test

        self.t_loading = 0.0

        self._current_file = None
        self._u = None
        self._mask = None
        self._N_local = None

        self.resolution = int(kwargs.get("resolution", 64))

        self.label_description = "[u]"
        self.frames = int(kwargs.get("frames", 10))
        self.T_len = int(kwargs.get("T_len", 1))

        self.samples_per_traj = self.frames - self.T_len

        # channels: vorticity (+ optional mask)
        self.input_dim = 1 + (1 if self.use_mask else 0)
        self.output_dim = 1

        self.pixel_mask = torch.zeros((1, self.resolution, self.resolution), dtype=torch.bool)

        self.constants = {
            "mean": torch.tensor([4.5171726e-05], dtype=torch.float32),
            "std": torch.tensor([0.52408644], dtype=torch.float32),
            "time": 1.0,
        }

        # init sizing/length based on first load
        self._load_file(0)

        if self.max_traj_per_file is None:
            self.max_traj_per_file = int(self._N_local)
        else:
            self.max_traj_per_file = min(int(self.max_traj_per_file), int(self._N_local))

        self._traj_per_file = int(self.max_traj_per_file)

        # len = (#traj total across shards) * (#samples per traj)
        total_traj_capacity = self._traj_per_file * len(self.file_indices)
        if self.num_trajectories is None:
            self._num_traj_total = total_traj_capacity
        else:
            self._num_traj_total = min(int(self.num_trajectories), total_traj_capacity)

        self._len = self._num_traj_total * self.samples_per_traj

        self.post_init()

    def _filepath(self, file_idx: int):
        u_pth = f"{self.data_path}/{self.file_prefix}vorticity_{self.file_indices[file_idx]}.npy"
        mask_pth = f"{self.data_path}/{self.file_prefix}mask_{self.file_indices[file_idx]}.npy"
        return u_pth, mask_pth

    def _load_file(self, file_idx: int):
        if self.which in ("val", "test"):
            if self._current_file == 0 and self._u is not None:
                return

            self._u = None
            self._mask = None
            gc.collect()

            u_pth = f"{self.data_path}/{self.file_prefix}vorticity_val.npy"
            mask_pth = f"{self.data_path}/{self.file_prefix}mask_val.npy"

            u = np.load(u_pth, mmap_mode="r+")
            mask = np.load(mask_pth, mmap_mode="r+")

            self._u = torch.from_numpy(u).float()
            self._mask = mask if self.use_mask else None
            self._N_local = int(self._u.shape[0])
            self._current_file = 0
            return

        # train
        if self._current_file == file_idx:
            return

        self._u = None
        self._mask = None
        gc.collect()

        t_ss = time.time()
        u_pth, mask_pth = self._filepath(file_idx)

        u = np.load(u_pth, mmap_mode="r+")
        mask = np.load(mask_pth, mmap_mode="r+")

        self._u = torch.from_numpy(u).float()
        self._mask = mask if self.use_mask else None
        self._N_local = int(self._u.shape[0])
        self._current_file = file_idx

        self.t_loading += (time.time() - t_ss)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        traj_id = idx // self.samples_per_traj
        t1 = idx % self.samples_per_traj
        t2 = t1 + self.T_len

        # map traj_id to (file_idx, local_idx) within the *used* trajectory budget
        global_traj = int(traj_id % self._num_traj_total)
        file_idx = global_traj // self._traj_per_file
        local_idx = global_traj % self._traj_per_file

        self._load_file(int(file_idx))

        traj_u = self._u[int(local_idx)]   # (T,H,W)
        u_in = traj_u[int(t1)]             # (H,W)
        u_out = traj_u[int(t2)]            # (H,W)

        inputs = u_in.unsqueeze(0)         # (1,H,W)
        labels = u_out.unsqueeze(0)        # (1,H,W)

        # normalize vorticity only
        inputs = (inputs - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]
        labels = (labels - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]

        # mask as input channel (obstacle=1, fluid=0), NOT normalized
        if self.use_mask and (self._mask is not None):
            raw = np.asarray(self._mask[int(local_idx)]).astype(np.bool_, copy=False)  # (H,W)

            obst = raw if self.mask_is_obstacle_true else ~raw
            mask_chan = torch.from_numpy(obst.astype(np.float32, copy=False)).unsqueeze(0)  # (1,H,W)

            inputs = torch.cat([inputs, mask_chan], dim=0)  # (2,H,W)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": float(self.T_len) * float(self.constants["time"]),
            "pixel_mask": self.pixel_mask,  # (1,H,W) all False
        }


class MultiObstacleVorticityMultiFrame(BaseTimeDataset):
    """
    Multi-frame vorticity dataset with "paper-style" mask handling:

      - condition on `history` previous frames (default 5)
      - total frames = 21 by your spec
      - predict 1-step ahead from the last history frame (default T_len=1)
      - mask is an INPUT CHANNEL, not used for pixel/loss masking

    Shapes:
      pixel_values: (history + (1 if use_mask else 0), H, W) -> [u_t ... u_{t+history-1}, mask?]
      labels:       (1, H, W)                                -> [u_target]
      pixel_mask:   (1, H, W) all False
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_count = int(kwargs.get("file_count"))
        self.which = kwargs.get("which")  # "train" | "val" | "test"
        self.data_path = kwargs.get("data_path")
        self.file_prefix = kwargs.get("file_prefix", "")
        self.file_indices = list(range(self.file_count))

        self.max_traj_per_file = kwargs.get("max_traj_per_file")
        self.num_trajectories = kwargs.get("num_trajectories")

        self.use_mask = bool(kwargs.get("use_mask", True))
        self.mask_is_obstacle_true = bool(kwargs.get("mask_is_obstacle_true", True))

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        self.N_max = (self.num_trajectories or 0) + self.N_val + self.N_test

        self.t_loading = 0.0

        self._current_file = None
        self._u = None
        self._mask = None
        self._N_local = None

        self.resolution = int(kwargs.get("resolution", 64))

        self.label_description = "[u]"
        self.frames = int(kwargs.get("frames", 21))
        self.history = int(kwargs.get("history", 5))
        self.T_len = int(kwargs.get("T_len", 1))

        assert self.history >= 1
        assert self.frames >= self.history + self.T_len, (
            f"Need frames >= history + T_len. Got frames={self.frames}, history={self.history}, T_len={self.T_len}"
        )

        # t0 so input uses [t0..t0+history-1], target at t0+history-1+T_len
        self.samples_per_traj = self.frames - (self.history + self.T_len) + 1

        self.input_dim = self.history + (1 if self.use_mask else 0)
        self.output_dim = 1

        self.pixel_mask = torch.zeros((1, self.resolution, self.resolution), dtype=torch.bool)

        self.constants = {
            "mean": torch.tensor([4.5171726e-05], dtype=torch.float32),
            "std": torch.tensor([0.52408644], dtype=torch.float32),
            "time": 5.0,
        }

        # init sizing/length based on first load
        self._load_file(0)

        if self.max_traj_per_file is None:
            self.max_traj_per_file = int(self._N_local)
        else:
            self.max_traj_per_file = min(int(self.max_traj_per_file), int(self._N_local))

        self._traj_per_file = int(self.max_traj_per_file)

        total_traj_capacity = self._traj_per_file * len(self.file_indices)
        if self.num_trajectories is None:
            self._num_traj_total = total_traj_capacity
        else:
            self._num_traj_total = min(int(self.num_trajectories), total_traj_capacity)

        self._len = self._num_traj_total * self.samples_per_traj

        self.post_init()

    def _filepath(self, file_idx: int):
        u_pth = f"{self.data_path}/{self.file_prefix}vorticity_{self.file_indices[file_idx]}.npy"
        mask_pth = f"{self.data_path}/{self.file_prefix}mask_{self.file_indices[file_idx]}.npy"
        return u_pth, mask_pth

    def _load_file(self, file_idx: int):
        if self.which in ("val", "test"):
            if self._current_file == 0 and self._u is not None:
                return

            self._u = None
            self._mask = None
            gc.collect()

            u_pth = f"{self.data_path}/{self.file_prefix}vorticity_val.npy"
            mask_pth = f"{self.data_path}/{self.file_prefix}mask_val.npy"

            u = np.load(u_pth, mmap_mode="r+")
            mask = np.load(mask_pth, mmap_mode="r+")

            self._u = torch.from_numpy(u).float()
            self._mask = mask if self.use_mask else None
            self._N_local = int(self._u.shape[0])
            self._current_file = 0
            return

        # train
        if self._current_file == file_idx:
            return

        self._u = None
        self._mask = None
        gc.collect()

        t_ss = time.time()
        u_pth, mask_pth = self._filepath(file_idx)

        u = np.load(u_pth, mmap_mode="r+")
        mask = np.load(mask_pth, mmap_mode="r+")

        self._u = torch.from_numpy(u).float()
        self._mask = mask if self.use_mask else None
        self._N_local = int(self._u.shape[0])
        self._current_file = file_idx

        self.t_loading += (time.time() - t_ss)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        traj_id = idx // self.samples_per_traj
        t0 = idx % self.samples_per_traj

        global_traj = int(traj_id % self._num_traj_total)
        file_idx = global_traj // self._traj_per_file
        local_idx = global_traj % self._traj_per_file

        self._load_file(int(file_idx))

        traj_u = self._u[int(local_idx)]  # (T,H,W)

        # history window: (history,H,W)
        u_hist = traj_u[int(t0): int(t0 + self.history)]
        # target at last history + T_len
        t_target = int(t0 + self.history - 1 + self.T_len)
        u_tgt = traj_u[t_target]          # (H,W)

        inputs = u_hist                   # (history,H,W)
        labels = u_tgt.unsqueeze(0)       # (1,H,W)

        # normalize vorticity only
        inputs = (inputs - self.constants["mean"][0]) / self.constants["std"][0]
        labels = (labels - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]

        # mask as input channel (obstacle=1, fluid=0), NOT normalized
        if self.use_mask and (self._mask is not None):
            raw = np.asarray(self._mask[int(local_idx)]).astype(np.bool_, copy=False)  # (H,W)
            obst = raw if self.mask_is_obstacle_true else ~raw
            mask_chan = torch.from_numpy(obst.astype(np.float32, copy=False)).unsqueeze(0)  # (1,H,W)
            inputs = torch.cat([inputs, mask_chan], dim=0)  # (history+1,H,W)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": 5.0,
            "pixel_mask": self.pixel_mask,  # (1,H,W) all False
        }



class MultiObstacleIncompressible(BaseTimeDataset):
    """
    Loads multi-obstacle incompressible fields from .npy shards:
      {prefix}velocity_x_{fi}.npy
      {prefix}velocity_y_{fi}.npy
      {prefix}pressure_{fi}.npy
      {prefix}mask_{fi}.npy

    Each field array is either (N,T,H,W) or (N,H,W,T).
    Mask is (N,H,W) boolean by default.

    IMPORTANT (paper-style):
      - mask is treated as an INPUT CHANNEL (geometry encoding)
      - we DO NOT use mask to form pixel_mask / loss masking here
      - we normalize only (vx, vy, p), not the mask channel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.which         = kwargs["which"]          # "train" | "val" | "test"
        self.data_path     = kwargs["data_path"]
        self.file_prefix   = kwargs.get("file_prefix", "")
        self.file_count    = int(kwargs.get("file_count", 1))
        self.file_indices  = list(range(self.file_count))

        self.num_trajectories    = kwargs.get("num_trajectories", None)     # total desired (train)
        self.max_traj_per_file   = kwargs.get("max_traj_per_file", None)

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        self.N_max = self.num_trajectories + self.N_val + self.N_test

        self.resolution   = int(kwargs.get("resolution", 64))
        self.frames       = int(kwargs.get("frames", 10))
        self.T_len        = int(kwargs.get("T_len", 1))  # 1-step by default
        self.samples_per_traj = self.frames - self.T_len

        # mask semantics
        # if True: raw mask True = obstacle; if False: raw mask True = fluid
        self.mask_is_obstacle_true = bool(kwargs.get("mask_is_obstacle_true", True))
        self.use_mask = bool(kwargs.get("use_mask", True))

        # channels
        self.fields = ("velocity_x", "velocity_y", "pressure", "mask")
        self.label_description = "[vx, vy, p]"
        self.input_dim  = 4 if self.use_mask else 3
        self.output_dim = 3  # predict vx, vy, p

        # internal cache for current file
        self._current_file = None
        self._vx = None
        self._vy = None
        self._p  = None
        self._mask = None  # numpy memmap or ndarray
        self._layout = None

        # Build per-file trajectory counts (train only) so mapping is correct even if last file is short.
        self._build_index()

        # normalize only physical channels (vx, vy, p)
        self.constants = {
            "mean": torch.tensor([0.9619638919830322, -7.730342986178584e-06, -0.39420366287231445], dtype=torch.float32).view(3, 1, 1),
            "std":  torch.tensor([0.6491641402244568, 0.5268091559410095, 1.060120940208435], dtype=torch.float32).view(3, 1, 1),
            "time": 1.0,
        }

        self.pixel_mask = torch.tensor([False, False, False])

        self.post_init()

    # ---------- indexing / mapping ----------
    def _build_index(self):
        if self.which in ("val", "test"):
            vx = np.load(f"{self.data_path}/{self.file_prefix}{self.fields[0]}_val.npy", mmap_mode="r")
            N = int(vx.shape[0])
            self._num_traj_total = N if self.num_trajectories is None else min(int(self.num_trajectories), N)
            self._cum = np.array([self._num_traj_total], dtype=np.int64)
            self._file_sizes = [self._num_traj_total]
            self._len = self._num_traj_total * self.samples_per_traj
            return

        # train: compute per-file sizes
        sizes = []
        for fi in self.file_indices:
            vx = np.load(f"{self.data_path}/{self.file_prefix}{self.fields[0]}_{fi}.npy", mmap_mode="r")
            n = int(vx.shape[0])
            if self.max_traj_per_file is not None:
                n = min(n, int(self.max_traj_per_file))
            sizes.append(n)

        total_capacity = int(np.sum(sizes))
        self._num_traj_total = total_capacity if self.num_trajectories is None else min(int(self.num_trajectories), total_capacity)

        # trim sizes to match _num_traj_total
        trimmed = []
        remaining = self._num_traj_total
        for n in sizes:
            take = min(n, remaining)
            trimmed.append(take)
            remaining -= take
            if remaining <= 0:
                break

        self._file_sizes = trimmed
        self._cum = np.cumsum(self._file_sizes, dtype=np.int64)
        self._len = self._num_traj_total * self.samples_per_traj

    def __len__(self):
        return self._len

    def _traj_to_file_local(self, traj_id: int) -> Tuple[int, int]:
        file_idx = int(np.searchsorted(self._cum, traj_id, side="right"))
        prev = 0 if file_idx == 0 else int(self._cum[file_idx - 1])
        local_idx = int(traj_id - prev)
        return file_idx, local_idx

    # ---------- IO ----------
    def _infer_layout(self, arr: np.ndarray) -> str:
        # arr is 4D
        if arr.shape[2] == self.resolution and arr.shape[3] == self.resolution:
            return "NTHW"
        if arr.shape[1] == self.resolution and arr.shape[2] == self.resolution:
            return "NHWT"
        # fallback
        return "NTHW" if arr.shape[1] != arr.shape[2] else "NHWT"

    def _load_file(self, file_idx: int):
        if self.which in ("val", "test"):
            if self._current_file == 0 and self._vx is not None:
                return
            gc.collect()

            def load_field(name: str):
                return np.load(f"{self.data_path}/{self.file_prefix}{name}_val.npy", mmap_mode="r")

            vx = load_field("velocity_x")
            vy = load_field("velocity_y")
            p  = load_field("pressure")
            self._layout = self._infer_layout(vx)

            self._vx = torch.from_numpy(vx).float()
            self._vy = torch.from_numpy(vy).float()
            self._p  = torch.from_numpy(p).float()

            if self.use_mask:
                self._mask = np.load(f"{self.data_path}/{self.file_prefix}mask_val.npy", mmap_mode="r")
            else:
                self._mask = None

            self._current_file = 0
            return

        # train: load shard by file_idx into cache
        if self._current_file == file_idx:
            return

        gc.collect()
        fi = self.file_indices[file_idx]

        vx = np.load(f"{self.data_path}/{self.file_prefix}velocity_x_{fi}.npy", mmap_mode="r")
        vy = np.load(f"{self.data_path}/{self.file_prefix}velocity_y_{fi}.npy", mmap_mode="r")
        p  = np.load(f"{self.data_path}/{self.file_prefix}pressure_{fi}.npy",   mmap_mode="r")
        self._layout = self._infer_layout(vx)

        self._vx = torch.from_numpy(vx).float()
        self._vy = torch.from_numpy(vy).float()
        self._p  = torch.from_numpy(p).float()

        if self.use_mask:
            self._mask = np.load(f"{self.data_path}/{self.file_prefix}mask_{fi}.npy", mmap_mode="r")
        else:
            self._mask = None

        self._current_file = file_idx

    def _get_frame(self, tensor_4d: torch.Tensor, local_idx: int, t: int) -> torch.Tensor:
        # returns (H,W)
        if self._layout == "NTHW":
            return tensor_4d[local_idx, t]
        else:  # NHWT
            return tensor_4d[local_idx, :, :, t]

    def __getitem__(self, idx: int):
        traj_id = idx // self.samples_per_traj
        t1      = idx %  self.samples_per_traj
        t2      = t1 + self.T_len

        file_idx, local_idx = self._traj_to_file_local(int(traj_id))
        self._load_file(file_idx)

        # physical fields (H,W)
        vx_in = self._get_frame(self._vx, local_idx, t1)
        vy_in = self._get_frame(self._vy, local_idx, t1)
        p_in  = self._get_frame(self._p,  local_idx, t1)

        vx_out = self._get_frame(self._vx, local_idx, t2)
        vy_out = self._get_frame(self._vy, local_idx, t2)
        p_out  = self._get_frame(self._p,  local_idx, t2)

        x_phys = torch.stack([vx_in, vy_in, p_in], dim=0)      # (3,H,W)
        y_phys = torch.stack([vx_out, vy_out, p_out], dim=0)   # (3,H,W)

        x_phys = (x_phys - self.constants["mean"]) / self.constants["std"]
        y_phys = (y_phys - self.constants["mean"]) / self.constants["std"]

        if self.use_mask and (self._mask is not None):
            raw = np.asarray(self._mask[local_idx]).astype(np.bool_, copy=False)  # (H,W)

            obst = raw if self.mask_is_obstacle_true else ~raw
            mask_chan = torch.from_numpy(obst.astype(np.float32, copy=False)).unsqueeze(0)  # (1,H,W)

            inputs = torch.cat([x_phys, mask_chan], dim=0)  # (4,H,W): [vx, vy, p, mask]
        else:
            inputs = x_phys  # (3,H,W)

        labels = y_phys  # predict only (vx, vy, p)

        time = 1.0

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }



class MultiObstacleIncompressibleMultiframe(BaseTimeDataset):
    """
    Multi-obstacle incompressible flow dataset with *multi-frame conditioning*.

    Loads .npy shards:
      {prefix}velocity_x_{fi}.npy
      {prefix}velocity_y_{fi}.npy
      {prefix}pressure_{fi}.npy
      {prefix}mask_{fi}.npy

    Each field array is either (N,T,H,W) or (N,H,W,T).
    Mask is (N,H,W) boolean by default.

    Paper-style semantics:
      - mask is treated as an INPUT CHANNEL (geometry encoding)
      - we DO NOT use mask to form pixel_mask / loss masking here
      - we normalize only (vx, vy, p), not the mask channel

    Multiframe:
      - condition on `history` previous frames (default 5)
      - predict 1-step ahead (t -> t+T_len, default T_len=1)
      - total frames per trajectory is `frames` (you said 21)

    Shapes:
      inputs  : (3*history + (1 if use_mask else 0), H, W)
      labels  : (3, H, W)
      pixel_mask : (3, H, W) boolean (all False => no overwriting/masking)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.which         = kwargs["which"]          # "train" | "val" | "test"
        self.data_path     = kwargs["data_path"]
        self.file_prefix   = kwargs.get("file_prefix", "")
        self.file_count    = int(kwargs.get("file_count", 1))
        self.file_indices  = list(range(self.file_count))

        self.num_trajectories    = kwargs.get("num_trajectories", None)
        self.max_traj_per_file   = kwargs.get("max_traj_per_file", None)

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        self.resolution   = int(kwargs.get("resolution", 64))

        self.N_max = self.num_trajectories + self.N_val + self.N_test

        # total frames in each trajectory (you said 21)
        self.frames       = int(kwargs.get("frames", 21))

        # predict horizon (steps ahead); keep 1-step by default
        self.T_len        = int(kwargs.get("T_len", 1))

        # number of conditioning frames (you want 5)
        self.history      = int(kwargs.get("history", 5))
        assert self.history >= 1, "history must be >= 1"
        assert self.frames >= self.history + self.T_len, (
            f"Need frames >= history + T_len. Got frames={self.frames}, history={self.history}, T_len={self.T_len}"
        )

        # number of training samples per trajectory
        # choose t0 so that input uses [t0, ..., t0+history-1] and target is at t0+history-1+T_len
        self.samples_per_traj = self.frames - (self.history + self.T_len) + 1

        # mask semantics
        # if True: raw mask True = obstacle; if False: raw mask True = fluid
        self.mask_is_obstacle_true = bool(kwargs.get("mask_is_obstacle_true", True))
        self.use_mask = bool(kwargs.get("use_mask", True))

        # channels / metadata
        self.fields = ("velocity_x", "velocity_y", "pressure", "mask")
        self.label_description = "[vx, vy, p] @ t+T_len"

        self.input_dim  = 3 * self.history + (1 if self.use_mask else 0)
        self.output_dim = 3  # predict vx, vy, p

        # internal cache for current file
        self._current_file = None
        self._vx = None
        self._vy = None
        self._p  = None
        self._mask = None  # numpy memmap or ndarray
        self._layout = None

        # Build per-file trajectory counts (train only)
        self._build_index()

        # normalize only physical channels (vx, vy, p)
        self.constants = {
            "mean": torch.tensor(
                [0.9619638919830322, -7.730342986178584e-06, -0.39420366287231445],
                dtype=torch.float32
            ).view(3, 1, 1),
            "std": torch.tensor(
                [0.6491641402244568, 0.5268091559410095, 1.060120940208435],
                dtype=torch.float32
            ).view(3, 1, 1),
            "time": 1.0,
        }

        # IMPORTANT: pixel_mask must be (C,H,W) so collation yields (B,C,H,W)
        self.pixel_mask = torch.zeros((self.output_dim, self.resolution, self.resolution), dtype=torch.bool)

        self.post_init()

    # ---------- indexing / mapping ----------
    def _build_index(self):
        if self.which in ("val", "test"):
            vx = np.load(f"{self.data_path}/{self.file_prefix}{self.fields[0]}_val.npy", mmap_mode="r")
            N = int(vx.shape[0])
            self._num_traj_total = N if self.num_trajectories is None else min(int(self.num_trajectories), N)
            self._cum = np.array([self._num_traj_total], dtype=np.int64)
            self._file_sizes = [self._num_traj_total]
            self._len = self._num_traj_total * self.samples_per_traj
            return

        # train: compute per-file sizes
        sizes: List[int] = []
        for fi in self.file_indices:
            vx = np.load(f"{self.data_path}/{self.file_prefix}{self.fields[0]}_{fi}.npy", mmap_mode="r")
            n = int(vx.shape[0])
            if self.max_traj_per_file is not None:
                n = min(n, int(self.max_traj_per_file))
            sizes.append(n)

        total_capacity = int(np.sum(sizes))
        self._num_traj_total = total_capacity if self.num_trajectories is None else min(int(self.num_trajectories), total_capacity)

        # trim sizes to match _num_traj_total
        trimmed: List[int] = []
        remaining = self._num_traj_total
        for n in sizes:
            take = min(n, remaining)
            trimmed.append(take)
            remaining -= take
            if remaining <= 0:
                break

        self._file_sizes = trimmed
        self._cum = np.cumsum(self._file_sizes, dtype=np.int64)
        self._len = self._num_traj_total * self.samples_per_traj

    def __len__(self):
        return self._len

    def _traj_to_file_local(self, traj_id: int) -> Tuple[int, int]:
        file_idx = int(np.searchsorted(self._cum, traj_id, side="right"))
        prev = 0 if file_idx == 0 else int(self._cum[file_idx - 1])
        local_idx = int(traj_id - prev)
        return file_idx, local_idx

    # ---------- IO ----------
    def _infer_layout(self, arr: np.ndarray) -> str:
        # arr is 4D
        if arr.shape[2] == self.resolution and arr.shape[3] == self.resolution:
            return "NTHW"
        if arr.shape[1] == self.resolution and arr.shape[2] == self.resolution:
            return "NHWT"
        # fallback
        return "NTHW" if arr.shape[1] != arr.shape[2] else "NHWT"

    def _load_file(self, file_idx: int):
        if self.which in ("val", "test"):
            if self._current_file == 0 and self._vx is not None:
                return
            gc.collect()

            def load_field(name: str):
                return np.load(f"{self.data_path}/{self.file_prefix}{name}_val.npy", mmap_mode="r")

            vx = load_field("velocity_x")
            vy = load_field("velocity_y")
            p  = load_field("pressure")
            self._layout = self._infer_layout(vx)

            self._vx = torch.from_numpy(vx).float()
            self._vy = torch.from_numpy(vy).float()
            self._p  = torch.from_numpy(p).float()

            if self.use_mask:
                self._mask = np.load(f"{self.data_path}/{self.file_prefix}mask_val.npy", mmap_mode="r")
            else:
                self._mask = None

            self._current_file = 0
            return

        # train: load shard by file_idx into cache
        if self._current_file == file_idx:
            return

        gc.collect()
        fi = self.file_indices[file_idx]

        vx = np.load(f"{self.data_path}/{self.file_prefix}velocity_x_{fi}.npy", mmap_mode="r")
        vy = np.load(f"{self.data_path}/{self.file_prefix}velocity_y_{fi}.npy", mmap_mode="r")
        p  = np.load(f"{self.data_path}/{self.file_prefix}pressure_{fi}.npy",   mmap_mode="r")
        self._layout = self._infer_layout(vx)

        self._vx = torch.from_numpy(vx).float()
        self._vy = torch.from_numpy(vy).float()
        self._p  = torch.from_numpy(p).float()

        if self.use_mask:
            self._mask = np.load(f"{self.data_path}/{self.file_prefix}mask_{fi}.npy", mmap_mode="r")
        else:
            self._mask = None

        self._current_file = file_idx

    def _get_frame(self, tensor_4d: torch.Tensor, local_idx: int, t: int) -> torch.Tensor:
        # returns (H,W)
        if self._layout == "NTHW":
            return tensor_4d[local_idx, t]
        else:  # NHWT
            return tensor_4d[local_idx, :, :, t]

    def __getitem__(self, idx: int):
        traj_id = idx // self.samples_per_traj
        t0      = idx %  self.samples_per_traj  # start time of the history window

        file_idx, local_idx = self._traj_to_file_local(int(traj_id))
        self._load_file(file_idx)

        # Build history window: times [t0, ..., t0+history-1]
        x_list = []
        for k in range(self.history):
            t = t0 + k
            vx = self._get_frame(self._vx, local_idx, t)
            vy = self._get_frame(self._vy, local_idx, t)
            p  = self._get_frame(self._p,  local_idx, t)
            x_list.append(torch.stack([vx, vy, p], dim=0))  # (3,H,W)

        x_phys = torch.cat(x_list, dim=0)

        t_last   = t0 + self.history - 1
        t_target = t_last + self.T_len

        vx_out = self._get_frame(self._vx, local_idx, t_target)
        vy_out = self._get_frame(self._vy, local_idx, t_target)
        p_out  = self._get_frame(self._p,  local_idx, t_target)
        y_phys = torch.stack([vx_out, vy_out, p_out], dim=0)  # (3,H,W)

        mean_hist = self.constants["mean"].repeat(self.history, 1, 1)
        std_hist  = self.constants["std"].repeat(self.history, 1, 1)
        x_phys = (x_phys - mean_hist) / std_hist

        y_phys = (y_phys - self.constants["mean"]) / self.constants["std"]

        if self.use_mask and (self._mask is not None):
            raw = np.asarray(self._mask[local_idx]).astype(np.bool_, copy=False)  # (H,W)
            obst = raw if self.mask_is_obstacle_true else ~raw
            mask_chan = torch.from_numpy(obst.astype(np.float32, copy=False)).unsqueeze(0)  # (1,H,W)
            inputs = torch.cat([x_phys, mask_chan], dim=0)  # (3*history+1, H, W)
        else:
            inputs = x_phys  # (3*history, H, W)

        labels = y_phys  # (3,H,W)

        time = 5.0

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": self.pixel_mask,  # (3,H,W) all False
        }