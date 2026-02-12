from nsm.typing import *
from nsm.flow import Flow

import os
import glob
import logging
import numpy as np
import numpy.lib.format as npformat

# NOTE: jnp/jrand/utils/Grid/SEM/F are expected to come from nsm.typing import *
# If not, uncomment:
# import jax.numpy as jnp
# import jax.random as jrand
# import functools as F


def _npy_shape0(path: str) -> int:
  """
  Read .npy header to get shape[0] without keeping a memmap open.
  Avoids 'too many open files' during dataset index expansion.
  """
  with open(path, "rb") as f:
    version = npformat.read_magic(f)
    shape, fortran, dtype = npformat._read_array_header(f, version)  # header only
  return int(shape[0])


class NpyFlow(Flow):
  """
  Data-driven Gray-Scott flow backed by .npy shards.

  Each shard file: (N, T, H, W, C) where C=2 (u,v).
  """

  def __init__(
    self,
    base_dir: str,
    train_glob: str = "train/data_*.npy",
    val_glob: str   = "val/data_*.npy",
    test_glob: str  = "val/data_*.npy",
    t: float = 1.0,                  # "one step" time unit used in logging
    domain_size: float = 1.0,         # size L in each spatial dim
  ):
    self.base_dir = base_dir
    self.split_glob = {
      "train": train_glob,
      "val": val_glob,
      "test": test_glob,
    }

    self.t = float(t)
    self.ndim = 2
    self.L = float(domain_size)       # used to build Grid size vector
    self.odim = 2                     # u,v channels

    # Cache memmaps to avoid reopening the same shard for every sample.
    # This prevents OSError: [Errno 24] Too many open files.
    self._mmap_cache: dict[str, np.ndarray] = {}

  # ----------------------------------- FLOW ----------------------------------- #

  def solve(self, ic: Grid, ts: Maybe[Array] = None, aux_data: PyTree = None) -> Grid:
    """No numerical solver here (purely data-driven)."""
    raise NotImplementedError("GrayScottNpy is data-driven; no solver implemented.")

  def process(self, ic: Grid, out: DataType, aux_data: PyTree) -> Tuple[DataType, Maybe[Grid]]:
    """Align ic to out's grid (if needed). Return (processed_ic, solver_guess=None)."""
    if isinstance(out, Grid):
      return ic.interp(out, "pbc"), None
    if isinstance(out, SEM):
      return ic.interp(out, "pbc"), None
    return ic, None

  def project(self, u: DataType, u0: Maybe[DataType], out: Maybe[Grid]) -> DataType:
    """No correction of solver guess; prediction is directly used."""
    if isinstance(u, SEM) and out is not None:
      return u.eval(out.resolution)
    return u

  def metric(self, u: Grid, ut: Grid) -> PyTree:
    """Simple per-channel metrics like other flows."""
    u_  = jnp.moveaxis(u.value, -1, 0)   # (C,H,W)
    ut_ = jnp.moveaxis(ut.value, -1, 0)

    var = ["u", "v"] if u_.shape[0] == 2 else [f"c{i}" for i in range(u_.shape[0])]

    def named(metrics: List[Array]) -> Dict[str, Array]:
      return { var[i]: m for i, m in enumerate(metrics) }

    return {
      "corr": named([utils.corr(a, b) for a, b in zip(u_, ut_)]),
      "rel_err": named([utils.err(a, b) for a, b in zip(u_, ut_)]),
      "abs_err": named([utils.mae(a, b) for a, b in zip(u_, ut_)]),
    }

  def loss_data(self, u: Grid, ut: Grid) -> Array:
    """Relative error on full tensor."""
    return utils.err(u.value, ut.value)

  # ---------------------------------- DATASET --------------------------------- #

  @F.lru_cache
  def dataset_files(self, split: str) -> List[str]:
    pat = self.split_glob[split]
    shard_paths = sorted(glob.glob(os.path.join(self.base_dir, pat)))
    if len(shard_paths) == 0:
      raise FileNotFoundError(f"No shards for split={split}: {self.base_dir}/{pat}")

    items = []
    for sp in shard_paths:
      N = _npy_shape0(sp)  # header-only, avoids opening lots of memmaps
      items.extend([f"{sp}::{i}" for i in range(N)])

    logging.info(
      f"Found {len(shard_paths)} shard(s) for split={split}: {self.base_dir}/{pat} "
      f"-> expanded to {len(items)} trajectories"
    )
    return items

  def _load_shard(self, shard_path: str) -> np.ndarray:
    """
    Load shard as memmap and cache it to avoid too many open files.
    """
    arr = self._mmap_cache.get(shard_path)
    if arr is None:
      arr = np.load(shard_path, mmap_mode="r")  # (N,T,H,W,C)
      self._mmap_cache[shard_path] = arr
    return arr

  def dataset_transform(self, fname: str, window: Maybe[int], prng: Maybe[Array]) -> Flow.Data:
    shard_path, traj_str = fname.split("::")
    traj_id = int(traj_str)

    arr = self._load_shard(shard_path)     # (N,T,H,W,C)
    u = arr[traj_id]                       # (T,H,W,C)

    # optional windowing (must leave at least 2 frames)
    T = u.shape[0]
    if window is not None:
      if T < window + 1:
        raise ValueError(f"{shard_path} traj {traj_id}: T={T} too short for window={window}")
      if prng is None:
        prng = jrand.PRNGKey(0)
      s = int(jrand.randint(prng, (), 0, T - (window + 1) + 1))
      u = u[s:s + window + 1]              # (window+1,H,W,C)

    size = jnp.array([self.L, self.L], dtype=jnp.float32)

    data = []
    for n, (ic_np, ut_np) in enumerate(zip(u[:-1], u[1:])):
      ic = Grid(jnp.asarray(ic_np, jnp.float32), size)
      ut = Grid(jnp.asarray(ut_np, jnp.float32), size)
      data.append(Flow.Data_t(n * self.t, ic, ut))
    return data


class NpyFlowVorticity(Flow):
  """
  Data-driven flow backed by .npy shards, using ONLY one channel (default u).

  Each shard file: (N, T, H, W, C) where C=2 (u,v).
  Slices to (N, T, H, W, 1) by taking traj[..., channel_index:channel_index+1].
  """

  def __init__(
    self,
    base_dir: str,
    train_glob: str = "train/data_*.npy",
    val_glob: str   = "val/data_*.npy",
    test_glob: str  = "val/data_*.npy",
    t: float = 1.0,
    domain_size: float = 1.0,
    channel_index: int = 0,           # 0 -> u, 1 -> v
  ):
    self.base_dir = base_dir
    self.split_glob = {
      "train": train_glob,
      "val": val_glob,
      "test": test_glob,
    }

    self.t = float(t)
    self.ndim = 2
    self.L = float(domain_size)
    self.odim = 1
    self.channel_index = int(channel_index)

    # Cache memmaps to avoid too many open files.
    self._mmap_cache: dict[str, np.ndarray] = {}

  # ----------------------------------- FLOW ----------------------------------- #

  def solve(self, ic: Grid, ts: Maybe[Array] = None, aux_data: PyTree = None) -> Grid:
    raise NotImplementedError("Data-driven; no solver implemented.")

  def process(self, ic: Grid, out: DataType, aux_data: PyTree) -> Tuple[DataType, Maybe[Grid]]:
    if isinstance(out, Grid):
      return ic.interp(out, "pbc"), None
    if isinstance(out, SEM):
      return ic.interp(out, "pbc"), None
    return ic, None

  def project(self, u: DataType, u0: Maybe[DataType], out: Maybe[Grid]) -> DataType:
    if isinstance(u, SEM) and out is not None:
      return u.eval(out.resolution)
    return u

  def metric(self, u: Grid, ut: Grid) -> PyTree:
    u_  = jnp.moveaxis(u.value, -1, 0)   # (1,H,W)
    ut_ = jnp.moveaxis(ut.value, -1, 0)

    var = ["u"]

    def named(metrics: List[Array]) -> Dict[str, Array]:
      return { var[i]: m for i, m in enumerate(metrics) }

    return {
      "corr": named([utils.corr(a, b) for a, b in zip(u_, ut_)]),
      "rel_err": named([utils.err(a, b) for a, b in zip(u_, ut_)]),
      "abs_err": named([utils.mae(a, b) for a, b in zip(u_, ut_)]),
    }

  def loss_data(self, u: Grid, ut: Grid) -> Array:
    return utils.err(u.value, ut.value)

  # ---------------------------------- DATASET --------------------------------- #

  @F.lru_cache
  def dataset_files(self, split: str) -> List[str]:
    pat = self.split_glob[split]
    shard_paths = sorted(glob.glob(os.path.join(self.base_dir, pat)))
    if len(shard_paths) == 0:
      raise FileNotFoundError(f"No shards for split={split}: {self.base_dir}/{pat}")

    items = []
    for sp in shard_paths:
      N = _npy_shape0(sp)
      items.extend([f"{sp}::{i}" for i in range(N)])

    logging.info(
      f"Found {len(shard_paths)} shard(s) for split={split}: {self.base_dir}/{pat} "
      f"-> expanded to {len(items)} trajectories"
    )
    return items

  def _load_shard(self, shard_path: str) -> np.ndarray:
    arr = self._mmap_cache.get(shard_path)
    if arr is None:
      arr = np.load(shard_path, mmap_mode="r")  # (N,T,H,W,C)
      self._mmap_cache[shard_path] = arr
    return arr

  def dataset_transform(self, fname: str, window: Maybe[int], prng: Maybe[Array]) -> Flow.Data:
    shard_path, traj_str = fname.split("::")
    traj_id = int(traj_str)

    arr = self._load_shard(shard_path)          # (N,T,H,W,C)
    traj = arr[traj_id]                         # (T,H,W,C)

    # slice ONLY one channel, keep channel dim (T,H,W,1)
    c = self.channel_index
    traj = traj[..., c:c+1]

    # optional windowing
    T = traj.shape[0]
    if window is not None:
      if T < window + 1:
        raise ValueError(f"{shard_path} traj {traj_id}: T={T} too short for window={window}")
      if prng is None:
        prng = jrand.PRNGKey(0)
      s = int(jrand.randint(prng, (), 0, T - (window + 1) + 1))
      traj = traj[s:s + window + 1]             # (window+1,H,W,1)

    size = jnp.array([self.L, self.L], dtype=jnp.float32)

    data = []
    for n, (ic_np, ut_np) in enumerate(zip(traj[:-1], traj[1:])):
      ic = Grid(jnp.asarray(ic_np, jnp.float32), size)   # (H,W,1)
      ut = Grid(jnp.asarray(ut_np, jnp.float32), size)
      data.append(Flow.Data_t(n * self.t, ic, ut))

    return data