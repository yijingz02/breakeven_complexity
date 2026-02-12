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

def estimate_vorticity_mean_std(
    data_path: str,
    file_prefix: str,
    file_indices,
    *,
    max_traj_per_file: int | None = None,
    max_trajs_total: int = 512,
    max_frames_per_traj: int | None = None,
    use_mask: bool = True,
    mask_is_fluid_true: bool = True,
    seed: int = 0,
):
    """
    Estimates mean/std of vorticity over TRAINING shards only, using random subsampling.
    Assumes each vorticity file stores one of:
      - (N, T, H, W)  (time-first)
      - (N, H, W, T)  (time-last)

    mask file assumed:
      - (N, H, W) boolean-ish
    If use_mask=True, we compute stats ONLY on fluid pixels.
    """
    rs = np.random.RandomState(seed)

    s1 = 0.0
    s2 = 0.0
    n  = 0

    file_indices = list(file_indices)
    if len(file_indices) == 0:
        raise ValueError("file_indices is empty")

    # distribute sample budget across files
    per_file = max(1, max_trajs_total // len(file_indices))

    for fi in file_indices:
        v_path = f"{data_path}/{file_prefix}vorticity_{fi}.npy"
        m_path = f"{data_path}/{file_prefix}mask_{fi}.npy"

        v = np.load(v_path, mmap_mode="r")   # memmap
        if use_mask:
            m = np.load(m_path, mmap_mode="r")
        else:
            m = None

        if v.ndim != 4:
            raise ValueError(f"{v_path} expected 4D, got {v.shape}")

        N = v.shape[0]
        if max_traj_per_file is not None:
            N = min(N, int(max_traj_per_file))

        take = min(per_file, N)
        if take <= 0:
            continue

        traj_idx = rs.choice(N, size=take, replace=False)

        # Determine layout
        # time-first: (N,T,H,W) -> v.shape[1] is T
        # time-last : (N,H,W,T) -> v.shape[-1] is T
        time_first = (v.shape[2] == v.shape[3]) and (v.shape[1] != v.shape[2])
        # This heuristic is not perfect; if you know your layout, replace this.
        # Better: assert based on expected H=W=64.
        if v.shape[2] == 64 and v.shape[3] == 64:
            # likely (N,T,64,64)
            layout = "NTHW"
        elif v.shape[1] == 64 and v.shape[2] == 64:
            # likely (N,64,64,T)
            layout = "NHWT"
        else:
            # fallback to guessing
            layout = "NTHW" if time_first else "NHWT"

        for j in traj_idx:
            if layout == "NTHW":
                traj = v[j]           # (T,H,W)
                T = traj.shape[0]
                if max_frames_per_traj is not None:
                    T = min(T, int(max_frames_per_traj))
                frame_idx = rs.choice(T, size=T if T <= 8 else 8, replace=False)
                frames = traj[frame_idx]  # (k,H,W)
            else:
                traj = v[j]           # (H,W,T)
                T = traj.shape[-1]
                if max_frames_per_traj is not None:
                    T = min(T, int(max_frames_per_traj))
                frame_idx = rs.choice(T, size=T if T <= 8 else 8, replace=False)
                frames = np.moveaxis(traj[..., frame_idx], -1, 0)  # -> (k,H,W)

            frames = frames.astype(np.float64, copy=False)

            if use_mask:
                mask = np.asarray(m[j]).astype(np.bool_, copy=False)  # (H,W)
                fluid = mask if mask_is_fluid_true else ~mask
                x = frames[:, fluid].reshape(-1)
            else:
                x = frames.reshape(-1)

            # accumulate
            s1 += x.sum()
            s2 += (x * x).sum()
            n  += x.size

    if n == 0:
        raise RuntimeError("No samples collected for mean/std (check masks / file paths).")

    mean = s1 / n
    var  = s2 / n - mean * mean
    std  = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std


class MultiObstacleVorticity(BaseTimeDataset):
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
        self._mask = None
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
        
        # mask_is_fluid_true = True
        # use_mask_for_stats = True
        # mean, std = estimate_vorticity_mean_std(
        #     data_path=self.data_path,
        #     file_prefix=self.file_prefix,
        #     file_indices=self.file_indices,
        #     max_traj_per_file=self.max_traj_per_file,
        #     max_trajs_total=self.num_trajectories,
        #     max_frames_per_traj=self.frames,
        #     use_mask=use_mask_for_stats,
        #     mask_is_fluid_true=mask_is_fluid_true,
        #     seed=0,
        # )

        # self.constants = {
        #     "mean": torch.tensor([mean], dtype=torch.float32),
        #     "std":  torch.tensor([std],  dtype=torch.float32),
        #     "time": 5.0,
        # }

        self._mask_is_fluid_true = mask_is_fluid_true

        self.constants = {
            "mean": torch.tensor([0.0]),
            "std": torch.tensor([1.0]),
            "time": 1.0,
        }

        self.post_init()

    def _filepath(self, file_idx):
        u_pth    = f"{self.data_path}/{self.file_prefix}vorticity_{self.file_indices[file_idx]}.npy"
        mask_pth = f"{self.data_path}/{self.file_prefix}mask_{self.file_indices[file_idx]}.npy"

        return u_pth, mask_pth

    def _load_file(self, file_idx):
        if self.which == "val" or self.which == "test":
            self._u    = None
            self._mask = None
            gc.collect()

            u_pth    = f"{self.data_path}/{self.file_prefix}vorticity_val.npy"
            mask_pth = f"{self.data_path}/{self.file_prefix}mask_val.npy"
            
            u    = np.load(u_pth, mmap_mode="r+")
            mask = np.load(mask_pth, mmap_mode="r+")

            self._u    = torch.from_numpy(u).float()
            self._mask = mask

            self._N_local = self._u.shape[0]

            self._current_file = file_idx

        elif self.which == "train":
            if self._current_file == file_idx:
                return

            self._u    = None
            self._mask = None

            gc.collect()

            idx = self.file_indices[file_idx]
            u_pth, mask_pth = self._filepath(idx)

            t_ss = time.time()

            u    = np.load(u_pth, mmap_mode="r+")
            mask = np.load(mask_pth, mmap_mode="r+")

            print(u.shape)

            self._u    = torch.from_numpy(u).float()
            self._mask = mask

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

        # u_in  = traj_u[:, :, t1]
        # u_out = traj_u[:, :, t2]
        u_in  = traj_u[t1]
        u_out = traj_u[t2]
        
        inputs = u_in.unsqueeze(0)
        labels = u_out.unsqueeze(0) 

        inputs = (inputs - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]
        labels = (labels - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]

        time = 1.0

        if self._mask is None:
            pixel_mask = torch.ones(inputs.shape[-2:], dtype=torch.bool)  # (H,W)
        else:
            raw = np.asarray(self._mask[local_idx]).astype(np.bool_, copy=False)
            pixel_mask = torch.from_numpy(~raw).unsqueeze(0)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": pixel_mask,
        }



class MultiObstacleVorticityMultiFrame(BaseTimeDataset):
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

        self.input_dim  = 10 # u * 5 frames
        self.output_dim = 1

        self.N_val  = kwargs.get("N_val")
        self.N_test = kwargs.get("N_val")
        
        self.N_max = self.num_trajectories + self.N_val + self.N_test

        self.t_loading = 0

        self._current_file = None
        self._u = None
        self._mask = None
        self._N_local = None

        self.res = 64
        self.resolution = 64

        self.label_description = "[u]"
        self.frames = 21
        self.T_len  = 10

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
        
        # mask_is_fluid_true = True
        # use_mask_for_stats = True
        # mean, std = estimate_vorticity_mean_std(
        #     data_path=self.data_path,
        #     file_prefix=self.file_prefix,
        #     file_indices=self.file_indices,
        #     max_traj_per_file=self.max_traj_per_file,
        #     max_trajs_total=self.num_trajectories,
        #     max_frames_per_traj=self.frames,
        #     use_mask=use_mask_for_stats,
        #     mask_is_fluid_true=mask_is_fluid_true,
        #     seed=0,
        # )

        # self.constants = {
        #     "mean": torch.tensor([mean], dtype=torch.float32),
        #     "std":  torch.tensor([std],  dtype=torch.float32),
        #     "time": 10.0,
        # }

        # self._mask_is_fluid_true = mask_is_fluid_true

        self.constants = {
            "mean": torch.tensor([0.0]),
            "std": torch.tensor([1.0]),
            "time": 10.0,
        }

        self.post_init()

    def _filepath(self, file_idx):
        u_pth    = f"{self.data_path}/{self.file_prefix}vorticity_{self.file_indices[file_idx]}.npy"
        mask_pth = f"{self.data_path}/{self.file_prefix}mask_{self.file_indices[file_idx]}.npy"

        return u_pth, mask_pth

    def _load_file(self, file_idx):
        if self.which == "val" or self.which == "test":
            self._u    = None
            self._mask = None
            gc.collect()

            u_pth    = f"{self.data_path}/{self.file_prefix}vorticity_val.npy"
            mask_pth = f"{self.data_path}/{self.file_prefix}mask_val.npy"
            
            u    = np.load(u_pth, mmap_mode="r+")
            mask = np.load(mask_pth, mmap_mode="r+")

            self._u    = torch.from_numpy(u).float()
            self._mask = mask

            self._N_local = self._u.shape[0]

            self._current_file = file_idx

        elif self.which == "train":
            if self._current_file == file_idx:
                return

            self._u    = None
            self._mask = None

            gc.collect()

            idx = self.file_indices[file_idx]
            u_pth, mask_pth = self._filepath(idx)

            t_ss = time.time()

            u    = np.load(u_pth, mmap_mode="r+")
            mask = np.load(mask_pth, mmap_mode="r+")

            print(u.shape)

            self._u    = torch.from_numpy(u).float()
            self._mask = mask

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

        u_in  = traj_u[t1:t2]
        u_out = traj_u[t2]
        
        inputs = u_in.unsqueeze(0)
        labels = u_out.unsqueeze(0) 

        inputs = (inputs - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]
        labels = (labels - self.constants["mean"][:, None, None]) / self.constants["std"][:, None, None]

        time = 10.0

        if self._mask is None:
            pixel_mask = torch.ones(inputs.shape[-2:], dtype=torch.bool)  # (H,W)
        else:
            raw = np.asarray(self._mask[local_idx]).astype(np.bool_, copy=False)
            pixel_mask = torch.from_numpy(~raw).unsqueeze(0)

        inputs = inputs.reshape(self.input_dim, self.resolution, self.resolution)
        
        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": pixel_mask,
        }
