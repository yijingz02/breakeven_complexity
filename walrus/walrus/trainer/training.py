import copy
import gc
import logging
import os
import pickle
import time
from concurrent.futures import Future
from contextlib import nullcontext
from random import shuffle
from subprocess import CalledProcessError
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from the_well.benchmark.metrics import (
    VRMSE,
    make_video,
    validation_plots,
)
from the_well.benchmark.metrics.common import Metric
from the_well.data.datamodule import AbstractDataModule
from the_well.data.datasets import WellDataset
from the_well.data.utils import flatten_field_names
from torch.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from walrus.trainer.checkpoints import CheckPointLoader
from walrus.trainer.normalization_strat import (
    BaseRevNormalization,
    normalize_target,
)

logger = logging.getLogger(__name__)

def _check_finite(name: str, x, throw: bool = True):
        if x is None:
            return

        # Numpy arrays -> tensor (cpu)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Python number -> tensor
        if isinstance(x, (float, int, np.floating, np.integer)):
            x = torch.tensor(x)

        # If it's not a tensor now, just log what it is and return (or raise)
        if not torch.is_tensor(x):
            msg = f"[NONFINITE-CHECK-SKIP] {name}: expected Tensor, got {type(x)} value={repr(x)[:200]}"
            if throw:
                raise TypeError(msg)
            else:
                logger.warning(msg)
                return

        # Real check
        ok = torch.isfinite(x).all().item()
        if not ok:
            with torch.no_grad():
                bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:10]
                finite = x[torch.isfinite(x)]
                mn = finite.min().item() if finite.numel() else float("nan")
                mx = finite.max().item() if finite.numel() else float("nan")
                msg = (
                    f"[NONFINITE] {name} shape={tuple(x.shape)} dtype={x.dtype} "
                    f"device={x.device} min_finite={mn} max_finite={mx} "
                    f"nan={torch.isnan(x).sum().item()} "
                    f"posinf={torch.isposinf(x).sum().item()} "
                    f"neginf={torch.isneginf(x).sum().item()} "
                    f"bad_idx={bad.tolist()}"
                )
            if throw:
                raise RuntimeError(msg)
            else:
                logger.error(msg)

class RelativeL2(Metric):
    """
    Relative L2 error per (batch, time, channel):
        rel = ||pred-ref||_2 / (||ref||_2 + eps)
    where ||·||_2 is over spatial dims only (H,W[,D]).
    Returns tensor shaped (B, T, C) to match Trainer.split_up_losses().
    """
    def __init__(self, name: str = "RelL2"):
        super().__init__()
        self.name = name

    def __call__(self, y_pred, y_ref, metadata, eps: float = 1e-12):
        # y_pred/y_ref: (B, T, [H W D], C) where spatial dims are 2..(ndim-2)
        spatial_dims = tuple(range(2, y_ref.ndim - 1))
        diff2 = (y_pred - y_ref).float().pow(2).sum(dim=spatial_dims)   # (B,T,C)
        ref2  = (y_ref).float().pow(2).sum(dim=spatial_dims)           # (B,T,C)

        rel = torch.sqrt(diff2 + eps) / (torch.sqrt(ref2 + eps))       # (B,T,C)
        return {self.name: rel}


def dimensionwise_nrmse(y_pred, y_ref, eps=1e-12):
    """Return B,C relative L2 reduced over the complete rollout and spatial axes."""
    reduction_dims = tuple(range(1, y_ref.ndim - 1))
    numerator = (y_pred - y_ref).float().square().sum(dim=reduction_dims)
    denominator = y_ref.float().square().sum(dim=reduction_dims).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


def _canonical_multi_obstacle_field_name(name: str) -> str | None:
    normalized = str(name).lower().replace(" ", "_")
    aliases = {
        "vorticity": "vorticity",
        "omega": "vorticity",
        "velocity_x": "velocity_x",
        "velocity_0": "velocity_x",
        "velocity_x_0": "velocity_x",
        "vx": "velocity_x",
        "velx": "velocity_x",
        "velocity_y": "velocity_y",
        "velocity_1": "velocity_y",
        "velocity_y_0": "velocity_y",
        "vy": "velocity_y",
        "vely": "velocity_y",
        "pressure": "pressure",
        "p": "pressure",
    }
    return aliases.get(normalized)


def _time_avg_nrmse_per_sample(pred, ref, eps=1e-12):
    batch_size = pred.shape[0]
    numerator = (pred - ref).reshape(batch_size, -1).float().pow(2).sum(dim=1)
    denominator = ref.reshape(batch_size, -1).float().pow(2).sum(dim=1).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


def _flip_multi_obstacle_fields(fields, names):
    """Reflect B,...,C fields across the final spatial axis."""
    flipped = torch.flip(fields, dims=[-2])
    for channel_idx, name in enumerate(names):
        if name in {"vorticity", "velocity_x"}:
            flipped[..., channel_idx] *= -1.0
    return flipped


def multi_obstacle_time_avg_scores(
    predictions,
    references,
    field_names,
    *,
    last_n=160,
):
    """Return Poseidon-style per-trajectory flip/no-flip time-average NRMSE."""
    # Walrus validation tensors are B,T,[spatial...],C and are already in physical units.
    k = min(int(last_n), predictions.shape[1], references.shape[1])
    if k <= 0:
        return {}

    canonical_to_index = {}
    for channel_idx, field_name in enumerate(field_names):
        canonical_name = _canonical_multi_obstacle_field_name(field_name)
        if canonical_name is not None and canonical_name not in canonical_to_index:
            canonical_to_index[canonical_name] = channel_idx

    groups = {
        quantity: [quantity]
        for quantity in ["vorticity", "velocity_x", "velocity_y", "pressure"]
        if quantity in canonical_to_index
    }
    if {"velocity_x", "velocity_y"}.issubset(canonical_to_index):
        groups["velocity"] = ["velocity_x", "velocity_y"]
    if {"velocity_x", "velocity_y", "pressure"}.issubset(canonical_to_index):
        groups["vxvyp"] = ["velocity_x", "velocity_y", "pressure"]
    if canonical_to_index:
        groups["all_fields"] = [
            quantity
            for quantity in ["vorticity", "velocity_x", "velocity_y", "pressure"]
            if quantity in canonical_to_index
        ]

    pred_avg = predictions[:, -k:].float().mean(dim=1)
    ref_avg = references[:, -k:].float().mean(dim=1)
    scores = {}
    for group_name, names in groups.items():
        indices = [canonical_to_index[name] for name in names]
        pred_fields = pred_avg[..., indices]
        ref_fields = ref_avg[..., indices]
        scores[f"tavg_flip_{group_name}"] = _time_avg_nrmse_per_sample(
            pred_fields + _flip_multi_obstacle_fields(pred_fields, names),
            ref_fields + _flip_multi_obstacle_fields(ref_fields, names),
        )
        scores[f"tavg_noflip_{group_name}"] = _time_avg_nrmse_per_sample(
            pred_fields, ref_fields
        )
    return scores


def channelwise_physical_nrmse(
    predictions,
    references,
    field_names,
    *,
    fluid_mask=None,
    eps=1e-12,
):
    """Return B,C physical-unit rollout nRMSE for Walrus validation tensors."""
    pred = predictions.float().clone()
    ref = references.float().clone()
    if fluid_mask is None:
        fluid = torch.ones(
            pred.shape[0],
            *pred.shape[2:-1],
            device=pred.device,
            dtype=pred.dtype,
        )
    else:
        fluid = fluid_mask.to(device=pred.device, dtype=pred.dtype)
        fluid = (fluid > 0.5).to(pred.dtype)

    while fluid.ndim < pred.ndim - 1:
        fluid = fluid.unsqueeze(1)
    weight = fluid.unsqueeze(-1)
    spatial_dims = tuple(range(2, pred.ndim - 1))

    for channel_idx, field_name in enumerate(field_names):
        if _canonical_multi_obstacle_field_name(field_name) != "pressure":
            continue
        pressure_weight = weight[..., 0]
        pressure_count = pressure_weight.sum(
            dim=spatial_dims, keepdim=True
        ).clamp_min(1.0)
        pred[..., channel_idx] -= (
            (pred[..., channel_idx] * pressure_weight).sum(
                dim=spatial_dims, keepdim=True
            )
            / pressure_count
        )
        ref[..., channel_idx] -= (
            (ref[..., channel_idx] * pressure_weight).sum(
                dim=spatial_dims, keepdim=True
            )
            / pressure_count
        )

    reduction_dims = tuple(range(1, pred.ndim - 1))
    numerator = ((pred - ref).square() * weight).sum(dim=reduction_dims)
    denominator = (ref.square() * weight).sum(
        dim=reduction_dims
    ).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


def _breakflow_fluid_mask(batch, metadata, device):
    """Extract the raw BreakFlow mask, whose convention is 1=fluid."""
    constant_names = metadata.constant_field_names[0]
    if "mask" not in constant_names:
        return None
    mask_index = constant_names.index("mask")
    return batch["constant_fields"][..., mask_index].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


def _safe_gaussian_kde(x):
    """Return a callable kde(t) or None if scipy isn't available."""
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        return gaussian_kde(x)
    except Exception:
        return None

def save_rel_l2_distribution_plot(
    rel_l2_values: np.ndarray,
    out_dir: str,
    tag: str,
    bins: int = 60,
    log_x: bool = True,
):
    """
    Saves:
      - {out_dir}/{tag}_relL2_dist.npz  (raw + histogram + smooth curve)
      - {out_dir}/{tag}_relL2_dist.png
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    x = np.asarray(rel_l2_values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    x = x[x > 0]  # needed for log-x
    if x.size == 0:
        logger.warning("[REL_L2_DIST_PLOT] No positive finite RelL2 values to plot for %s", tag)
        return

    # Histogram binning
    if log_x:
        lo, hi = np.percentile(x, [0.5, 99.5])
        lo = max(lo, np.min(x[x > 0]))
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    else:
        lo, hi = np.percentile(x, [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)

    counts, edges = np.histogram(x, bins=edges)

    # Smooth overlay (KDE if available, else smoothed histogram in log space)
    kde_x = None
    kde_y = None
    kde = _safe_gaussian_kde(x)
    if kde is not None:
        kde_x = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), 400) if log_x else np.linspace(edges[0], edges[-1], 400)
        kde_y = kde(kde_x)
        # scale density to "count" scale for visual overlay
        bin_widths = np.diff(edges)
        typical_w = np.median(bin_widths)
        kde_y = kde_y * x.size * typical_w
    else:
        # fallback: smooth histogram counts (in log-x coordinate if requested)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kde_x = centers
        kde_y = counts.astype(float)
        # simple moving-average smoothing
        w = 7
        if kde_y.size >= w:
            kernel = np.ones(w, dtype=float) / w
            kde_y = np.convolve(kde_y, kernel, mode="same")

    # Save raw + plotting artifacts for exact reproduction
    npz_path = os.path.join(out_dir, f"{tag}_relL2_dist.npz")
    np.savez(
        npz_path,
        rel_l2=x,
        edges=edges,
        counts=counts,
        kde_x=kde_x,
        kde_y=kde_y,
        log_x=np.array([log_x], dtype=bool),
    )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(x, bins=edges, alpha=0.65)
    if kde_x is not None and kde_y is not None:
        ax.plot(kde_x, kde_y, linewidth=2.5)

    ax.set_title("Test loss distribution (hist + smooth overlay, log x)" if log_x else "Test loss distribution (hist + smooth overlay)")
    ax.set_xlabel("Per-trajectory relative L2 loss")
    ax.set_ylabel("Count")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)

    fig_path = os.path.join(out_dir, f"{tag}_relL2_dist.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    logger.info("[REL_L2_DIST_PLOT] Wrote %s and %s", npz_path, fig_path)
    return npz_path, fig_path

def log_all_time_metrics(
    time_logs: dict,
    output_dir: str,
    epoch_number: str = "0",
):
    """Plot loss over time for all time metrics.

    Args:
        time_logs: Dict of time metrics
        metadata: Metadata object associated with dset
        output_dir: Directory to save the plots
        epoch_number: Current epoch number
    """
    for dataset_name, logs in time_logs.items():
        os.makedirs(
            f"{output_dir}/{dataset_name}/rollout_losses/epoch_{epoch_number}",
            exist_ok=True,
        )
        print(
            "output_dir",
            f"{output_dir}/{dataset_name}/rollout_losses/epoch_{epoch_number}",
        )
        for k, v in logs.items():
            v = np.array(v)
            title = k.split("/")[-1]
            np.save(
                f"{output_dir}/{dataset_name}/rollout_losses/epoch_{epoch_number}/{title}.npy",
                v,
            )


def expand_mask_to_match(mask, target):
    """Expand mask of shape B H [W D] 1 to
    broadcast with tensor of given shape B T H [W D] C"""
    T = target.shape[1]
    C = target.shape[-1]
    expansion_tuple = (
        -1,
        T,
    )
    expansion_tuple = expansion_tuple + (-1,) * (len(target.shape) - 3) + (C,)
    mask = mask.unsqueeze(1).expand(*expansion_tuple)
    return mask


def get_grad_norm_local(model) -> torch.Tensor:
    """Computes grad norm for the specific device

    From https://github.com/pytorch/pytorch/issues/88621"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


def get_grad_norm_fsdp(
    model, rank, world_size, sharding_strategy=ShardingStrategy.FULL_SHARD
) -> torch.Tensor:
    """Combines grad norm for the specific device

    From https://github.com/pytorch/pytorch/issues/88621"""
    local_norm = get_grad_norm_local(model)
    op = torch.distributed.ReduceOp.SUM
    return_norm = local_norm.clone().detach().requires_grad_(False) ** 2
    dist.all_reduce(return_norm, op=op)
    if sharding_strategy == ShardingStrategy.NO_SHARD:
        return_norm = return_norm / world_size
    return return_norm**0.5


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5


class Trainer:
    grad_scaler: GradScaler

    def __init__(
        self,
        experiment_name: str,
        viz_folder: str,
        formatter: Callable,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        revin: BaseRevNormalization,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        prediction_type: str,
        max_epoch: int,
        val_frequency: int,
        rollout_val_frequency: int,
        max_rollout_steps: int,
        short_validation_length: int,
        checkpointer: CheckPointLoader,
        num_time_intervals: int,
        skip_checkpointing: bool = False,
        validation_suite: list[Metric] = [VRMSE()],
        validation_trajectory_metrics: Optional[list[Metric]] = None,
        batch_aggregation_fns: list[Callable] = [torch.mean],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        device_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
        sampling_rank_strategy: Literal["gpu", "node", "world"] = "node",
        reuse_batches: bool = False,
        distribution_type: str = "local",
        rank: int = 0,
        world_size: int = 1,
        enable_amp: bool = False,
        amp_type: str = "float16",  # bfloat not supported in FFT
        grad_acc_steps: int = 1,
        big_batch_multiplier: int = 1,
        big_batch_before: int = 0,
        clip_gradient: float = 0.0,
        loss_multiplier: float = 1.0,
        minimum_context: int = 1,
        validation_full_trajectory_ensemble_size: int = 1,
        validation_one_step_ensemble_size: int = 1,
        masked_loss_for_objects: bool = True,
        video_validation: bool = False,
        video_size_multiplier: float = 0.4,
        dump_prediction_to_disk: bool = False,
        num_detailed_logs: int = 3,
        image_validation: bool = False,
        skip_spectral_metrics: bool = False,
        gradient_log_level: int = 0,
        log_interval: int = 10,
        wandb_logging: bool = True,
        start_epoch: int = 1,
        lr_scheduler_per_step: bool = False,
        debug_mode: bool = False,
        start_val_loss: Optional[float] = None,
        epsilon: float = 1e-5,
        validation_epsilon: float = 1e-5,
        time_budget_s: Optional[float] = None,   # total budget in seconds (includes one-time data gen cost)
        gen_time_s: float = 0.0,                 # seconds per generated trajectory/sample (one-time)
        Ndata_generated: Optional[int] = None,   # fixed up-front #trajectories/samples generated
        save_path: Optional[str] = None,
        single_frame_initialization: bool = False,
    ):
        """
        Class in charge of the training loop. It performs train, validation and test.

        Parameters
        ----------
        experiment_name:
            The name of the training experiment to be run
        viz_folder:
            The folder where visualizations are saved
        formatter:
            Callable that initializes formatter object that maps between Well and model formats.
        model:
            PyTorch model used for training.
        datamodule:
            A datamodule that provides dataloaders for each split (train, valid, and test)
        optimizer:
            A Pytorch optimizer to perform the backprop (e.g. Adam)
        loss_fn:
            A loss function that evaluates the model predictions to be used for training
        prediction_type:
            The type of prediction to make. Options are "delta" or "full". "delta" predicts the change in the
            field from the previous timestep. "full" predicts the full field at the next timestep.
            This only affects training since validation losses are computed on full reconstructed fields
            either way.
        max_epoch:
            Number of epochs to train the model.
            One epoch correspond to a full loop over the datamodule's training dataloader which may be a subset
        val_frequency:
            The frequency in terms of number of epochs to perform one-step validation
        rollout_val_frequency:
            The frequency in terms of number of epochs to perform rollout validation
        max_rollout_steps:
            The maximum number of timesteps to rollout the model during long validation.
            Note: for historical reasons, redundant parameter is included in datamodule - both need to be set.
        num_time_intervals:
            The number of time intervals to bin the loss over for logging purposes.
        lr_scheduler:
            A Pytorch learning rate scheduler to update the learning rate during training
        device:
            A Pytorch device (e.g. "cuda" or "cpu")
        device_mesh:
            Device mesh used for distributed training
        reuse_batches:
            A boolean flag to reuse batches during training. If True, the same batch is used for
            two steps of training (separated by a new batch). If False, a new batch is used for each step.
            Mostly use if disk io is bottleneck.
        distribution_type:
            The type of distribution to use. Options are "local", "ddp", "fsdp", "hsdp"
        rank:
            The rank of the current GPU in the PyTorch world.
        world_size:
            The total number of GPUs in the PyTorch world
        enable_amp:
            A boolean flag to enable automatic mixed precision training
        amp_type:
            The type of automatic mixed precision to use. Options are "float16" or "bfloat16"
        grad_acc_steps:
            The number of gradient accumulation steps to perform between optimizer steps
        big_batch_multiplier:
            Use batches this many times larger via grad accumulation during the first big_batch_before epochs
            Was experimented with, but not actually used.
        big_batch_before:
            Use big batches for the first this many epochs
            Was experimented with, but not actually used.
        clip_gradient:
            The maximum gradient norm to clip to. If 0, no clipping is performed.
        loss_multiplier:
            A float to multiply the loss by before backpropagating. Useful for satisfying paranoia
            about underflow error. Generally won't matter when using Adam-family optimizers at FP32.
        minimum_context:
            The minimum number of timesteps needed to evaluate the loss for a given sample
        validation_full_trajectory_ensemble_size:
            The number of times to repeat the trajectory during validation. The final prediction is averaged over all sampled trajectories.
        validation_one_step_ensemble_size:
            The number of samples to draw per step. The final step is averaged over all sampled steps.
        video_validation:
            A boolean flag to enable saving rollouts to disk during validation
        dump_prediction_to_disk:
            A boolean flag to enable saving raw prediction and reference numpy arrays to disk during validation
        num_detailed_logs:
            The number of detailed logs (video, numpy dumps) to save during validation
        image_validation:
            A boolean flag to enable saving images to disk during validation
        skip_spectral_metrics:
            A boolean flag to skip spectral metrics during validation since they can be memory hungry.
        gradient_log_level:
            An integer representing the level of gradient logging. 0 is no logging, 1 full synced gradient only
        log_interval:
            An integer representing how often to log training information. This results in gpu-cpu sync.
        wandb_logging:
            A boolean flag to enable logging to Weights and Biases
        start_epoch:
            The epoch to start training from. Used for resuming training.
        lr_scheduler_per_step:
            A boolean flag to update the learning rate after each optimizer step instead of each epoch.
        debug_mode:
            A boolean flag to run a quick debug run. This limits all validation to small version.
        start_val_loss:
            The validation loss to start from. Used for resuming training.
        epsilon:
            A small float added to denominators in denominators during training for numerical stability. Reasonably varies between runs.
        validation_epsilon:
            A small float added to denominators in loss functions during validation for numerical stability. Should be kept consistent across runs.
        """
        self.experiment_name = experiment_name
        self.viz_folder = viz_folder
        self.wandb_logging = wandb_logging
        self.video_validation = video_validation
        self.image_validation = image_validation
        self.video_size_multiplier = video_size_multiplier
        self.dump_prediction_to_disk = dump_prediction_to_disk
        self.num_detailed_logs = num_detailed_logs
        self.gradient_log_level = gradient_log_level
        self.log_interval = log_interval
        self.device = device
        self.model = model
        self.single_frame_initialization = bool(single_frame_initialization)
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        self.skip_checkpointing = skip_checkpointing
        self.validation_suite = validation_suite
        self.validation_suite.append(self.loss_fn)
        self.validation_suite.append(RelativeL2())
        self.validation_trajectory_metrics = validation_trajectory_metrics
        self.batch_aggregation_fns = batch_aggregation_fns
        self.skip_spectral_metrics = skip_spectral_metrics
        # These starting parameters are just for resuming runs
        self.start_epoch = start_epoch
        self.start_val_loss = start_val_loss
        # Run logistics
        self.max_epoch = max_epoch
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps
        self.short_validation_length = short_validation_length
        self.num_time_intervals = num_time_intervals
        self.enable_amp = enable_amp
        self.grad_acc_steps = grad_acc_steps
        self.big_batch_multiplier = big_batch_multiplier
        self.big_batch_before = big_batch_before
        self.clip_gradient = clip_gradient
        self.loss_multiplier = loss_multiplier
        self.minimum_context = minimum_context
        self.validation_full_trajectory_ensemble_size = (
            validation_full_trajectory_ensemble_size
        )
        self.validation_one_step_ensemble_size = validation_one_step_ensemble_size
        self.lr_scheduler_per_step = lr_scheduler_per_step
        self.debug_mode = debug_mode
        self.masked_loss_for_objects = masked_loss_for_objects
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.checkpointer = checkpointer
        # If local or DDP, can use standard grad scaler
        if distribution_type.upper() in ["LOCAL", "DDP"]:
            self.grad_scaler = torch.GradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )
        # Otherwise need sharded version.
        else:
            self.grad_scaler = ShardedGradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )

        self.device_mesh = device_mesh
        self.is_distributed = device_mesh is not None
        self.distribution_type = distribution_type
        self.rank = rank
        self.world_size = world_size
        self.sampling_rank_strategy = sampling_rank_strategy
        self.reuse_batches = reuse_batches
        # Get derived rank info about which nodes must be synced from the device mesh
        if self.device_mesh is not None and "fsdp" in self.device_mesh.mesh_dim_names:
            self.sync_group = self.device_mesh.get_group(mesh_dim="fsdp")
            self.sync_group_size = self.sync_group.size()
        else:  # Local or DDP
            self.sync_group = None
            self.sync_group_size = 1
        # Local or DDP do not need all-gather to do forward pass
        self.sync_group_rank = self.rank // self.sync_group_size
        self.rank_in_sync_group = self.rank % self.sync_group_size
        self.num_sync_groups = self.world_size // self.sync_group_size
        if (
            self.sampling_rank_strategy == "gpu"
        ):  # This means sample different dataset per GPU
            self.sampling_rank = self.rank
        elif (
            self.sampling_rank_strategy == "node"
        ):  # This means sample different dataset per node
            self.sampling_rank = self.sync_group_rank
        else:  # This means sample single dataset per step
            self.sampling_rank = 0

        self.dset_metadata = self.datamodule.train_dataset.dset_to_metadata
        self.revin = revin(self.datamodule.train_dataset, self.device)
        self.model_epsilon = (
            epsilon  # Used for compatibility with configs from before split occurred
        )
        self.validation_epsilon = validation_epsilon

        self.formatter_dict = {}
        # Initial formatter for each dataset - right now these are all identical
        # but we might want to differentiate them in the future.
        for dset_name, metadata in self.dset_metadata.items():
            self.formatter_dict[metadata.dataset_name] = formatter()

        self.time_budget_s = None if time_budget_s is None else float(time_budget_s)
        self.gen_time_s = float(gen_time_s)
        self.save_path = save_path or os.path.join(
            os.environ.get("MODEL_STORAGE_ROOT", "checkpoints"), "walrus"
        )

        if Ndata_generated is not None:
            self.Ndata_generated = int(Ndata_generated)
        else:
            # Infer from metadata if possible (sum across training datasets)
            inferred = 0
            for _, meta in self.dset_metadata.items():
                n = getattr(meta, "n_trajectories", None)
                if n is None:
                    n = getattr(meta, "num_trajectories", None)
                if n is None:
                    raise AttributeError(
                        "Could not infer Ndata_generated from metadata "
                        "(expected meta.n_trajectories or meta.num_trajectories). "
                        "Pass Ndata_generated explicitly."
                    )
                inferred += int(n)
            self.Ndata_generated = inferred

        self.gen_cost_s = self.Ndata_generated * self.gen_time_s
        if self.time_budget_s is None:
            self.effective_train_budget_s = None
        else:
            self.effective_train_budget_s = float(self.time_budget_s - self.gen_cost_s)

        self.train_compute_time_s = 0.0  # accumulates fwd+bw+opt only (no dataloading)

        if self.rank == 0 and self.time_budget_s is not None:
            logger.info(
                "[TIME_BUDGET] time_budget_s=%.3f, Ndata_generated=%d, gen_time_s=%.6f => gen_cost_s=%.3f, "
                "effective_train_budget_s=%.3f",
                self.time_budget_s,
                self.Ndata_generated,
                self.gen_time_s,
                self.gen_cost_s,
                self.effective_train_budget_s,
            )

        self.do_infer_timing = False
        self.infer_total_time = 0
        self.infer_n_cnt = 0
        self.infer_cnt = 0

    @staticmethod
    def _single_frame_teacher_forcing_batch(batch):
        """Sample teacher-forced left-padded startup stages without growing B."""
        history = batch["input_fields"]
        targets = batch["output_fields"]
        history_length = history.shape[1]
        sequence = torch.cat([history, targets], dim=1)
        contexts = []
        stage_targets = []
        target_indices = torch.randint(
            1, sequence.shape[1], (history.shape[0],)
        )
        for sample_idx, target_idx_tensor in enumerate(target_indices):
            target_idx = int(target_idx_tensor.item())
            available = sequence[sample_idx:sample_idx + 1, :target_idx]
            first = sequence[sample_idx:sample_idx + 1, :1]
            if target_idx < history_length:
                padding = first.repeat(
                    1,
                    history_length - target_idx,
                    *([1] * (history.ndim - 2)),
                )
                context = torch.cat([padding, available], dim=1)
            else:
                context = available[:, -history_length:]
            contexts.append(context)
            stage_targets.append(
                sequence[sample_idx:sample_idx + 1, target_idx:target_idx + 1]
            )

        result = dict(batch)
        result["input_fields"] = torch.cat(contexts, dim=0)
        result["output_fields"] = torch.cat(stage_targets, dim=0)
        return result

    def save_model_if_necessary(
        self, epoch: int, validation_loss: float, last: bool = False
    ) -> Optional[Future]:
        """Save the model checkpoint.
        Force checkpointing if last.
        """
        checkpoint_future = self.checkpointer.save_if_necessary(
            self.model,
            self.optimizer,
            validation_loss,
            epoch,
            force=last,
            local=self.distribution_type.upper() in ["LOCAL", "DDP"],
        )
        return checkpoint_future

    def rollout_model(
        self,
        model,
        batch,
        formatter,
        train=True,
        fake_pass=False,
        full_rollout=False,
    ):
        """Rollout the model for as many steps as we have data for.
        predict_normalized: bool - If true, output normalized prediction. During one-step training,
            predict normalized values to reduce precision issues/extra FLOPs. During rollout,
            denormalize the output for loss calculation. If multiple steps used during training,
            throw error because not currently supported.
        """

        metadata = batch["metadata"]
        batch = {
            k: v.to(self.device, non_blocking=True)
            if k not in {"metadata", "boundary_conditions"}
            else v
            for k, v in batch.items()
        }
        single_frame_training = (
            train
            and self.single_frame_initialization
            and batch["input_fields"].shape[1] > 1
        )
        if single_frame_training:
            batch = self._single_frame_teacher_forcing_batch(batch)
        T_in = batch["input_fields"].shape[1]
        # Extract mask and move to device for loss eval
        if (
            self.masked_loss_for_objects
            and "mask" in batch["metadata"].constant_field_names[0]
        ):
            mask_index = batch["metadata"].constant_field_names[0].index("mask")
            mask = batch["constant_fields"][..., mask_index : mask_index + 1]
            mask = mask.to(self.device, dtype=torch.bool, non_blocking=True)
            if metadata.dataset_name.lower() == "breakflow":
                # PyFR/Analyze uses 1 for fluid and 0 for an obstacle, whereas
                # Walrus expects True to identify values that should be hidden.
                mask = ~mask
        else:
            mask = None

        if mask is not None:
            s = int(mask.sum().item())
            if s == 0:
                logger.warning(
                    "[NAN_DEBUG] object mask sum == 0 (dataset=%s). "
                    "Masked reductions like sum(masked)/mask.sum() can produce NaNs.",
                    metadata.dataset_name,
                )

        inputs, y_ref = formatter.process_input(
            batch,
            causal_in_time=model.causal_in_time and not single_frame_training,
            predict_delta=self.prediction_type == "delta",
            train=train,
        )
        if self.single_frame_initialization and not train and T_in > 1:
            y_ref = torch.cat([batch["input_fields"][:, 1:], y_ref], dim=1)
            batch["input_fields"] = batch["input_fields"][:, :1].repeat(
                1,
                T_in,
                *([1] * (batch["input_fields"].ndim - 2)),
            )

        # Inputs T B C H [W D], y_ref B T H [W D] C
        # If causal, during training don't include initial context in rollout length
        if full_rollout:
            max_rollout_steps = y_ref.shape[1]
        elif model.causal_in_time:
            max_rollout_steps = self.max_rollout_steps + (T_in - 1)
        else:
            max_rollout_steps = self.max_rollout_steps
        rollout_steps = min(
            y_ref.shape[1], max_rollout_steps
        )  # Number of timesteps in target
        train_rollout_limit = (
            1
            if single_frame_training
            else (T_in if (train and model.causal_in_time) else 1)
        )
        if rollout_steps > train_rollout_limit and train:
            raise ValueError("Multiple step prediction in train mode not yet supported")
        y_ref = y_ref[:, :rollout_steps]
        # Create a moving batch of one step at a time
        moving_batch = copy.deepcopy(batch)
        y_preds = []
        # Fake pass is just a convenience for distributed validation without communcation code
        if fake_pass:
            return y_ref, y_ref

        # Rollout the model - Causal in time gets more predictions from the first step
        for i in range(train_rollout_limit - 1, rollout_steps):
            # Don't fill causal_in_time here since that only affects y_ref
            inputs, _ = formatter.process_input(moving_batch)
            inputs = list(inputs)
            with torch.no_grad():
                normalization_stats = self.revin.compute_stats(
                    inputs[0], metadata, epsilon=self.model_epsilon
                )

            # try:
            #     # _check_finite("norm/sample_mean", normalization_stats.sample_mean)
            #     # _check_finite("norm/sample_std",  normalization_stats.sample_std)
            #     # _check_finite("norm/delta_mean",  normalization_stats.delta_mean)
            #     # _check_finite("norm/delta_std",   normalization_stats.delta_std)
            # except Exception as e:
            #     logger.error("[NAN_DEBUG] normalization_stats blew up (dataset=%s)", metadata.dataset_name)
            #     raise

            # if std is ~0, you will divide by ~0 -> Inf/NaN later
            # min_std = float(normalization_stats.sample_std.min().item())
            # min_dstd = float(normalization_stats.delta_std.min().item())
            # if min_std < 1e-12 or min_dstd < 1e-12:
            #     logger.warning(
            #         "[NAN_DEBUG] tiny std detected: min sample_std=%.3e, min delta_std=%.3e (dataset=%s)",
            #         min_std, min_dstd, metadata.dataset_name
            #     )

            # NOTE - Currently assuming only [0] (fields) needs normalization
            normalized_inputs = inputs[:]  # Map type bugs out
            normalized_inputs[0] = self.revin.normalize_stdmean(
                normalized_inputs[0], normalization_stats
            )

            # _check_finite("normalized_inputs[0]", normalized_inputs[0])
        
            if train:
                ensemble_size = 1
            elif self.do_infer_timing:
                ensemble_size = 1
            else:
                ensemble_size = self.validation_one_step_ensemble_size

            for jj in range(ensemble_size):
                t0 = time.time()
                y_pred_internal = model(
                    normalized_inputs[0],
                    normalized_inputs[1],
                    normalized_inputs[2].tolist(),
                    metadata=metadata,
                )
                t1 = time.time()
                if self.do_infer_timing:
                    self.infer_total_time += (t1 - t0)
                    # print(f"[INFER ONE STEP]: {t1 - t0}")

                if jj == 0:
                    y_pred = y_pred_internal.clone() / ensemble_size
                else:
                    y_pred = y_pred + y_pred_internal / ensemble_size

            # _check_finite("y_pred_internal (model output)", y_pred_internal)

            # During validation, don't maintain full inner predictions
            if single_frame_training:
                y_pred = y_pred[-1:]
            elif not train and model.causal_in_time:
                y_pred = y_pred[-1:]  # y_pred is T first, y_ref is not
            # Train used normalized values to avoid precision loss
            # Validation on the other hand, reconstructs predictions on original scale
            if train:
                pass  # Do nothing since we're computing loss on predicted value and normalizing "ref"
            elif self.prediction_type == "delta":
                # y_pred - (T_all or T=-1 depending on causal or not), B, C, H, [W, D]. Different from y_ref
                with torch.autocast(
                    self.device.type, enabled=False, dtype=self.amp_type
                ):
                    y_pred = inputs[0][
                        -y_pred.shape[0] :
                    ].float() + self.revin.denormalize_delta(
                        y_pred, normalization_stats
                    )  # Unnormalize delta and add to input
            elif self.prediction_type == "full":
                y_pred = self.revin.denormalize_stdmean(y_pred, normalization_stats)
            else:
                raise ValueError(
                    f"Invalid prediction type {self.prediction_type}. Valid types are delta/full"
                )
            y_pred = formatter.process_output(y_pred, metadata)[
                ..., : y_ref.shape[-1]
            ]  # Cut off constant channels

            # TODO - redo losses to accept losses since this will be more efficient there
            if mask is not None:
                mask_pred = expand_mask_to_match(mask, y_pred)
                y_pred.masked_fill_(mask_pred, 0)

            # NOTE (MM) - The person reading this might ask - why is the mask in-place above and a copy below?
            # The answer is that job failed using in_place below but not above. No idea why.
            if (
                batch["padded_field_mask"].shape[0] != y_pred.shape[-1]
            ):  # Quick hack for Neutron
                batch["padded_field_mask"] = batch["padded_field_mask"][
                    : y_pred.shape[-1]
                ]
            y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

            # If not last step, update moving batch for autoregressive prediction
            # TODO - for anyone updating this later, it's the primary reason why
            # multiple steps isn't currently supported since we want to recompute
            # normalization stats at each step, but also want to compute training loss
            # on normalized values
            if i != rollout_steps - 1:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
                )
            # For causal models, we get use full predictions for the first batch and
            # incremental predictions for subsequent batches - concat 1:T to y_ref for loss eval
            if model.causal_in_time and i == train_rollout_limit - 1:
                y_preds.append(y_pred)
            else:
                y_preds.append(y_pred[:, -1:])
        y_pred_out = torch.cat(y_preds, dim=1)
        # Post-processing y_ref depending on train - if train, normalize y_ref before loss calc
        # If not train, we already denormalized the prediction
        if train:
            mean = (
                normalization_stats.sample_mean
                if self.prediction_type == "full"
                else normalization_stats.delta_mean
            )
            std = (
                normalization_stats.sample_std
                if self.prediction_type == "full"
                else normalization_stats.delta_std
            )
            y_ref = normalize_target(y_ref, mean, std, formatter, metadata, self.device)

        # _check_finite("y_ref (post normalize_target)", y_ref)
        
        if mask is not None:
            mask_ref = expand_mask_to_match(mask, y_ref)
            y_ref.masked_fill_(mask_ref, 0)

        del moving_batch, batch, mask  # Free up batch memory when done
        return y_pred_out, y_ref

    def temporal_split_losses(
        self, loss_values, temporal_loss_intervals, loss_name, dset_name, fname="full"
    ):
        """Take time series of loss values in split them into aggregate metrics computed on
        different time intervals."""
        # loss_values is B, T here since we either selected from or averaged over fields
        new_losses = {}
        # Average over time interval
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_values.mean(dim=1)
        # Don't compute sublosses if we only have one interval
        if len(temporal_loss_intervals) <= 2:
            return new_losses
        # Break it down by time interval
        for k in range(len(temporal_loss_intervals) - 1):
            start_ind = temporal_loss_intervals[k]
            end_ind = temporal_loss_intervals[k + 1]
            time_str = f"{start_ind}:{end_ind}"
            loss_subset = loss_values[:, start_ind:end_ind].mean(
                1
            )  # Mean over time in interval
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={time_str}"] = loss_subset
        return new_losses

    def split_up_losses(
        self, loss_values, loss_name, dset_name, field_names: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Divide raw losses into per-field and mean values, as well as temporal splits.
        Returns new loss dict and time logs for plotting."""
        # loss_values is B, T, C here
        new_losses: dict[str, Any] = {}
        time_logs: dict[str, Any] = {}
        time_steps = loss_values.shape[1]  # we B, T, C
        num_time_intervals = min(time_steps, self.num_time_intervals)
        temporal_loss_intervals = np.linspace(0, np.log(time_steps), num_time_intervals)
        temporal_loss_intervals = [0] + [
            int(np.exp(x)) for x in temporal_loss_intervals
        ]
        # Split up losses by field
        for i, fname in enumerate(field_names):
            time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                :, :, i
            ].cpu()
            new_losses |= self.temporal_split_losses(
                loss_values[:, :, i],
                temporal_loss_intervals,
                loss_name,
                dset_name,
                fname,
            )
        # Compute average over all fields
        new_losses |= self.temporal_split_losses(
            loss_values.mean(2),
            temporal_loss_intervals,
            loss_name,
            dset_name,
            "full",
        )
        time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(2).cpu()
        # New losses - B, time_logs - B T
        return new_losses, time_logs

    @torch.no_grad()
    def _validation_rel_l2_only(self, dataloaders, valid_or_test="test", full=True, epoch=0):
        self.model.eval()
        metadatas = []

        # collect per-sample relL2 (already reduced over spatial dims by RelativeL2)
        rel_samples_local = []  # list of 1D torch tensors on CPU

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        for i, dataloader in enumerate(dataloaders):
            dataset = dataloader.dataset.sub_dsets[0]
            meta = dataset.metadata
            metadatas.append(meta)
            dset_name = meta.dataset_name

            rank_assignment = i % self.num_sync_groups
            if rank_assignment != self.sync_group_rank:
                continue

            denom = len(dataloader) if full else min(len(dataloader), self.short_validation_length)

            for j, batch in enumerate(dataloader):
                t0 = time.time()
                y_pred, y_ref = self.rollout_model(
                    self.model, batch, self.formatter_dict[dset_name], train=False
                )
                t1 = time.time()

                # respect padded_field_mask like before
                pfm = batch["padded_field_mask"]
                if pfm.shape[0] != y_pred.shape[-1]:
                    pfm = pfm[: y_pred.shape[-1]]
                y_pred = y_pred[..., pfm]
                y_ref  = y_ref[..., pfm]

                # RelativeL2 returns dict {"RelL2": (B,T,C)}
                rel = RelativeL2()(y_pred, y_ref, meta, eps=self.validation_epsilon)["RelL2"]  # (B,T,C)

                # reduce over time+channels -> per-sample (B,)
                rel_per_sample = rel.mean(dim=(1, 2)).detach().cpu()
                rel_samples_local.append(rel_per_sample)

                if (not full) and (j + 1) >= self.short_validation_length:
                    break

        if len(rel_samples_local) == 0:
            # nothing processed on this rank
            rel_all = torch.empty(0, dtype=torch.float32)
        else:
            rel_all = torch.cat(rel_samples_local, dim=0)  # CPU tensor (N,)

        # --- distributed gather: gather just this 1D tensor ---
        if self.is_distributed:
            # gather sizes
            local_n = torch.tensor([rel_all.numel()], dtype=torch.long)
            ns = [torch.zeros_like(local_n) for _ in range(self.world_size)]
            dist.all_gather(ns, local_n)
            ns = [int(x.item()) for x in ns]
            max_n = max(ns)

            # pad to max_n so all_gather works
            padded = torch.zeros(max_n, dtype=torch.float32)
            if rel_all.numel() > 0:
                padded[: rel_all.numel()] = rel_all

            gathered = [torch.zeros(max_n, dtype=torch.float32) for _ in range(self.world_size)]
            dist.all_gather(gathered, padded)

            rel_list = []
            for g, n in zip(gathered, ns):
                if n > 0:
                    rel_list.append(g[:n])
            rel_all = torch.cat(rel_list, dim=0) if len(rel_list) else torch.empty(0, dtype=torch.float32)

        # now rank0 has full distribution
        out = {}
        if self.rank == 0:
            rel_np = rel_all.numpy()
            mean_rel = float(rel_np.mean()) if rel_np.size else float("nan")

            # log scalar with the exact key you want
            # e.g. test_ns_exponax/full_RelL2_T=all_mean
            # if multiple datasets, you may want per-dset keys; here assume one dset
            dset = metadatas[0].dataset_name if len(metadatas) else "unknown"
            out[f"{valid_or_test}_{dset}/full_RelL2_T=all_mean"] = mean_rel
            out["epoch"] = epoch

            # plot + wandb
            out_dir = os.path.join(self.viz_folder, dset, "distributions")
            tag = f"{valid_or_test}_epoch{epoch}_rank{self.rank}"
            save_rel_l2_distribution_plot(rel_np, out_dir=out_dir, tag=tag, bins=60, log_x=True)

            fig_path = os.path.join(out_dir, f"{tag}_relL2_dist.png")
            npz_path = os.path.join(out_dir, f"{tag}_relL2_dist.npz")

            if self.wandb_logging:
                wandb.log({f"{dset}/{tag}/relL2_dist": wandb.Image(fig_path),
                        f"{valid_or_test}_{dset}/full_RelL2_T=all_mean": mean_rel,
                        "epoch": epoch})
                wandb.save(npz_path, base_path=self.viz_folder)

        # return signature-compatible (validation_loss, logs)
        # validation_loss can just be mean RelL2 (or NaN if not rank0)
        validation_loss = out.get(f"{valid_or_test}_{metadatas[0].dataset_name}/full_RelL2_T=all_mean", float("nan")) if self.rank == 0 else float("nan")
        return validation_loss, out

    # @torch.no_grad()
    # def validation_loop(
    #     self,
    #     dataloaders: list[WellDataset],
    #     valid_or_test: str = "valid",
    #     full=False,
    #     epoch: int = 0,
    #     only_rel_l2: bool = False,
    # ) -> tuple[float, dict[str, Any]]:
    #     """Run validation by looping over the dataloader.

    #     Validate the same dataset over FSDP groups since they're locked
    #     by syncs but distribute over replication (DDP) groups.

    #     WARNING: running DDP or HSDP with a number of datasets that either dont
    #     evenly divide the number of shard groups/GPUs or that have
    #     significantly varying sizes will probably result in a timeout at one of these barriers.
    #     """
    #     full = full if not self.debug_mode else False
    #     self.model.eval()

    #     if only_rel_l2:
    #         return self._validation_rel_l2_only(
    #             datalinfer_time_s_total += (t1 - t0)oaders=dataloaders,
    #             valid_or_test=valid_or_test,
    #             full=full,
    #             epoch=epoch,
    #         )

    #     validation_loss = 0.0
    #     metadatas = []
    #     # Timeouts get really annoying - barrier at start makes it slightly better
    #     if torch.distributed.is_initialized():
    #         torch.distributed.barrier()
    #     # Each dataset being validated gets separate loader
    #     rank_loss_dict = {}
    #     rank_time_logs = {}
    #     for i, dataloader in enumerate(dataloaders):
    #         # Grab metadata for the current dataset
    #         assert len(dataloader.dataset.sub_dsets) == 1, (
    #             "Only one dataset per validation dataloader"
    #         )
    #         dataset = dataloader.dataset.sub_dsets[
    #             0
    #         ]  # There is only one dset by design
    #         dset_time_logs = {}
    #         current_metadata = dataset.metadata
    #         metadatas.append(current_metadata)
    #         dset_name = current_metadata.dataset_name
    #         field_names = flatten_field_names(current_metadata, include_constants=False)
    #         # NOTE MM - We want each FSDP group to process a different dataset to accelerate
    #         # partial validation between epochs so we assign datasets to groups in round-robin fashion.
    #         # If we have more FSDP groups than datasets, some groups will do nothing which can lead to timeouts
    #         # during full validation.
    #         # Validation shared over FSDP group, then passed to local rank 0 for aggregation and logging.
    #         # For giant dataset or metric counts, this could get memory hungry on rank 0.
    #         rank_assignment = (
    #             i % self.num_sync_groups
    #         )  # Use same data across one sync group
    #         # Only print if we're doing something on this node
    #         if rank_assignment == self.sync_group_rank:
    #             logger.info(
    #                 f"Validating dataset {dataset.metadata.dataset_name} with full_trajectory_mode={dataset.full_trajectory_mode} on rank {self.rank}"
    #             )
    #         else:
    #             continue
    #         count = 0
    #         denom = (
    #             len(dataloader)
    #             if full
    #             else min(len(dataloader), self.short_validation_length)
    #         )
    #         with torch.autocast(
    #             device_type=self.device.type,
    #             enabled=self.enable_amp,
    #             dtype=self.amp_type,
    #         ):
    #             for j, batch in enumerate(dataloader):
    #                 # Validation datasets don't automatically add metadata
    #                 start_time = time.time()
    #                 # Rollout for length of target - fake pass if not evaluating on this node
    #                 # so that we get the right field names for reduction
    #                 for kk in range(self.validation_full_trajectory_ensemble_size):
    #                     t0 = time.time()
    #                     y_pred_internal, y_ref_internal = self.rollout_model(
    #                         self.model,
    #                         batch,
    #                         self.formatter_dict[dset_name],
    #                         train=False,
    #                         fake_pass=(
    #                             rank_assignment != self.sync_group_rank
    #                         ),  # Leftover from earlier design - should never be true due to continue above
    #                     )
    #                     t1 = time.time()

    #                     if kk == 0:
    #                         y_pred = (
    #                             y_pred_internal
    #                             / self.validation_full_trajectory_ensemble_size
    #                         )
    #                         y_ref = y_ref_internal  # This doesn't change
    #                     else:
    #                         y_pred += (
    #                             y_pred_internal
    #                             / self.validation_full_trajectory_ensemble_size
    #                         )
    #                 assert y_ref.shape == y_pred.shape, (
    #                     f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
    #                 )
    #                 # Go through losses
    #                 model_time = time.time() - start_time
    #                 if (
    #                     batch["padded_field_mask"].shape[0] != y_pred.shape[-1]
    #                 ):  # Quick hack for Neutron and the underscores in the dimension names...
    #                     batch["padded_field_mask"] = batch["padded_field_mask"][
    #                         : y_pred.shape[-1]
    #                     ]
    #                 # Don't evaluate loss of "padded" dimensions so we're not biased towards predicting 0
    #                 y_pred, y_ref = (
    #                     y_pred[..., batch["padded_field_mask"]],
    #                     y_ref[..., batch["padded_field_mask"]],
    #                 )

    #                 # Collecting names to make detailed output logs
    #                 used_field_names = [
    #                     f
    #                     for i, f in enumerate(field_names)
    #                     if batch["padded_field_mask"][i]
    #                 ]

    #                 # Iterate through all validation metrics and log them.
    #                 for loss_fn in self.validation_suite:
    #                     # Mean over batch and time per field
    #                     if (
    #                         self.skip_spectral_metrics
    #                         and "spectr" in loss_fn.__class__.__name__
    #                     ):
    #                         continue
    #                     # Loss fn expect B T [H W D] C where [H W D] are described by metadata
    #                     loss = loss_fn(
    #                         y_pred, y_ref, current_metadata, eps=self.validation_epsilon
    #                     )
    #                     # Some losses return multiple values for efficiency, so if not dict,
    #                     # wrap in dict here
    #                     if not isinstance(loss, dict):
    #                         loss = {loss_fn.__class__.__name__: loss}
    #                     # Split the losses and update the logging dictionary
    #                     for k, sub_loss in loss.items():
    #                         # Each loss is B, T, C
    #                         new_losses, new_time_logs = self.split_up_losses(
    #                             sub_loss, k, dset_name, used_field_names
    #                         )
    #                         # TODO Break spectral error into second category using new API
    #                         for loss_name, batch_losses in new_losses.items():
    #                             if loss_name in rank_loss_dict:
    #                                 rank_loss_dict[loss_name] = torch.cat(
    #                                     [rank_loss_dict[loss_name], batch_losses], dim=0
    #                                 )
    #                             else:
    #                                 rank_loss_dict[loss_name] = batch_losses
    #                             # Let's just store the VRMSE for printout since that's what we're actually looking at on aggregate.
    #                             if "full_VRMSE_T=all" in loss_name:
    #                                 vrmse = batch_losses.mean().item()
    #                         if dataset.full_trajectory_mode:
    #                             for loss_name, batch_time_logs in new_time_logs.items():
    #                                 if loss_name in dset_time_logs:
    #                                     # Time logs live on CPU for mem reasons
    #                                     dset_time_logs[loss_name] = torch.cat(
    #                                         [
    #                                             dset_time_logs[loss_name],
    #                                             batch_time_logs,
    #                                         ],
    #                                         dim=0,
    #                                     )
    #                                 else:
    #                                     dset_time_logs[loss_name] = batch_time_logs
    #                 # NOW do trajectory losses if we have them - can probably combine with above later
    #                 if dataset.full_trajectory_mode:
    #                     for traj_loss_fn in self.validation_trajectory_metrics or []:
    #                         traj_loss = traj_loss_fn(
    #                             y_pred, y_ref, current_metadata, batch["metadata"]
    #                         )
    #                         if not isinstance(traj_loss, dict):
    #                             traj_loss = {traj_loss_fn.__class__.__name__: traj_loss}
    #                         # Note - still use split up losses for per-field even if temporal not relevant
    #                         for k, sub_traj_loss in traj_loss.items():
    #                             new_traj_losses, _ = self.split_up_losses(
    #                                 sub_traj_loss, k, dset_name, used_field_names
    #                             )
    #                             for loss_name, batch_losses in new_traj_losses.items():
    #                                 if loss_name in rank_loss_dict:
    #                                     rank_loss_dict[loss_name] = torch.cat(
    #                                         [rank_loss_dict[loss_name], batch_losses],
    #                                         dim=0,
    #                                     )
    #                                 else:
    #                                     rank_loss_dict[loss_name] = batch_losses
    #                 total_time = time.time() - start_time
    #                 max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
    #                 # Only print out if local device actually doing something
    #                 if rank_assignment == self.sync_group_rank:
    #                     logger.info(
    #                         f"{valid_or_test}: {dset_name}, Batch {j + 1}/{denom}, Rank {self.rank:>3}: Field-time-averaged VRMSE {vrmse:7.4f}, mem {max_mem_GB:5.2f} GB, total_time {total_time:5.3f}s, model {model_time:5.4f}s"
    #                     )
    #                 if torch.cuda.is_available():
    #                     torch.cuda.reset_peak_memory_stats()
    #                 count += 1
    #                 # Do some detailed outputs on rank 0 if specified
    #                 if (
    #                     self.rank_in_sync_group == 0
    #                     and rank_assignment == self.sync_group_rank
    #                 ):
    #                     if dataset.full_trajectory_mode:
    #                         if self.video_validation and count < self.num_detailed_logs:
    #                             try:
    #                                 make_video(
    #                                     y_pred[0],  # First sample only in batch
    #                                     y_ref[0],  # First sample only in batch
    #                                     current_metadata,
    #                                     self.viz_folder,
    #                                     f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",  # For the file name
    #                                     field_name_overrides=used_field_names,  # Fields actually used
    #                                     size_multiplier=self.video_size_multiplier,  # Shrinking for bulk runs, but visuals tuned around 1
    #                                 )
    #                             except CalledProcessError as e:
    #                                 logger.warning(
    #                                     f"Error in making video due to FFMPEG: {e}. Skipping video."
    #                                 )
    #                         # Write out prediction and reference as npy for later analysis
    #                         if (
    #                             self.dump_prediction_to_disk
    #                             and count < self.num_detailed_logs
    #                         ):
    #                             dump_path = os.path.join(
    #                                 self.viz_folder, dset_name, "full_trajectory_dumps"
    #                             )
    #                             if not os.path.exists(dump_path):
    #                                 os.makedirs(dump_path, exist_ok=True)
    #                             np.save(
    #                                 os.path.join(
    #                                     dump_path,
    #                                     f"yref_{dset_name}_{valid_or_test}_epoch{epoch}_rank{self.rank}_{j}.npy",
    #                                 ),
    #                                 y_ref.cpu().numpy(),
    #                             )
    #                             np.save(
    #                                 os.path.join(
    #                                     dump_path,
    #                                     f"ypred_{dset_name}_{valid_or_test}_epoch{epoch}_rank{self.rank}_{j}.npy",
    #                                 ),
    #                                 y_pred.cpu().numpy(),
    #                             )
    #                             logger.info(f"Wrote out npy dumps to {dump_path}")
    #                 # For most per-"epoch" validations, we only do a configurably short subset
    #                 if not full and count >= self.short_validation_length:
    #                     break
    #             # Run some last outputs on rank 0 do get a general sense of what's going on
    #             if (
    #                 self.rank_in_sync_group == 0
    #                 and rank_assignment == self.sync_group_rank
    #             ):
    #                 if self.image_validation:
    #                     for plot_fn in validation_plots:
    #                         if (
    #                             self.skip_spectral_metrics
    #                             and "spectr" in plot_fn.__name__
    #                         ):
    #                             continue
    #                         plot_fn(
    #                             y_pred,
    #                             y_ref,
    #                             current_metadata,
    #                             self.viz_folder,  # Temporary until we port over the resume logic
    #                             f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",  # For the file name
    #                         )
    #                 if dataset.full_trajectory_mode:
    #                     if self.video_validation:
    #                         try:
    #                             make_video(
    #                                 y_pred[0],  # First sample only in batch
    #                                 y_ref[0],  # First sample only in batch
    #                                 current_metadata,
    #                                 self.viz_folder,
    #                                 f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",  # For the file name
    #                                 field_name_overrides=used_field_names,  # Fields actually used
    #                                 size_multiplier=self.video_size_multiplier,  # Shrinking for bulk runs, but visuals tuned around 1
    #                             )
    #                         except CalledProcessError as e:
    #                             logger.warning(
    #                                 f"Error in making video due to FFMPEG: {e}. Skipping video."
    #                             )
    #         if dataset.full_trajectory_mode:
    #             rank_time_logs[current_metadata.dataset_name] = dset_time_logs
        
    #     # If we're distributed, now send all per rank results to rank 0 for aggregation
    #     if self.is_distributed:
    #         # Wait for all ranks to finish
    #         logger.info(f"Rank {self.rank} waiting for barrier")
    #         dist.barrier()
    #         logger.info(f"Rank {self.rank} passed barrier")

    #         # If we're distributed, gather all batchwise losses onto rank 0 for summarization and logging
    #         object_loss_dict = {k: v.cpu() for k, v in rank_loss_dict.items()}
    #         loss_dicts = [None for _ in range(self.world_size)]
    #         time_logs_list = [None for _ in range(self.world_size)]
    #         dist.all_gather_object(loss_dicts, object_loss_dict)
    #         dist.all_gather_object(time_logs_list, rank_time_logs)
    #         loss_dict = {}
    #         # Iterate through loss dicts and recover all keys
    #         for ld in loss_dicts:
    #             for k, v in ld.items():
    #                 if k not in loss_dict:
    #                     loss_dict[k] = v
    #                 else:
    #                     loss_dict[k] = torch.cat([loss_dict[k], v], dim=0)
    #         time_logs = {}
    #         # Time logs have two levels {dataset: {lossname: B x T tensor}}
    #         for tl in time_logs_list:
    #             for dataset, dset_time_logs in tl.items():
    #                 if dataset not in time_logs:
    #                     time_logs[dataset] = {}
    #                 for k, v in dset_time_logs.items():
    #                     if k not in time_logs[dataset]:
    #                         time_logs[dataset][k] = v
    #                     else:
    #                         time_logs[dataset][k] = torch.cat(
    #                             [time_logs[dataset][k], v], dim=0
    #                         )
    #     else:  # If we're not distributed, just use the local rank dict
    #         loss_dict = rank_loss_dict
    #         time_logs = rank_time_logs

    #     # --- [REL_L2_DIST_PLOT] save per-sample RelL2 distribution for final eval on rank 0 ---
    #     if self.rank == 0 and valid_or_test in ("test", "rollout_test"):
    #         for meta in metadatas:
    #             dset = meta.dataset_name
    #             key = f"{dset}/full_RelL2_T=all"
    #             if key not in loss_dict:
    #                 logger.warning("[REL_L2_DIST_PLOT] Missing key %s in loss_dict.", key)
    #                 continue

    #             rel = loss_dict[key]
    #             rel_np = rel.detach().cpu().numpy() if isinstance(rel, torch.Tensor) else np.asarray(rel)

    #             out_dir = os.path.join(self.viz_folder, dset, "distributions")
    #             tag = f"{valid_or_test}_epoch{epoch}_rank{self.rank}"
    #             npz_path, fig_path = save_rel_l2_distribution_plot(
    #                 rel_l2_values=rel_np,
    #                 out_dir=out_dir,
    #                 tag=tag,
    #                 bins=60,
    #                 log_x=True,
    #             )

    #             # ---- W&B logging (image + file) ----
    #             if self.wandb_logging and npz_path is not None and fig_path is not None:
    #                 # log image
    #                 wandb.log(
    #                     {f"{dset}/{tag}/relL2_dist": wandb.Image(fig_path), "epoch": epoch},
    #                     commit=False,
    #                 )
    #                 # upload the npz as a file (simplest)
    #                 wandb.save(npz_path, base_path=self.viz_folder)


    #     # Single score validation loss is average of all losses on the training metric
    #     validation_loss = sum(
    #         [
    #             loss_dict[
    #                 f"{metadata.dataset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
    #             ]
    #             .mean()
    #             .item()  # Losses should all be B sized
    #             for metadata in metadatas
    #         ]
    #     ) / len(metadatas)
    #     aggregated_losses = {}
    #     aggregated_time_logs = {}
    #     for f in self.batch_aggregation_fns:
    #         for k, v in loss_dict.items():
    #             aggregated_losses[f"{valid_or_test}_{k}_{f.__name__}"] = f(
    #                 v
    #             )  # All losses by this point should be B sized
    #         # Two levels of time logs
    #         for dataset, dset_time_logs in time_logs.items():
    #             if dataset not in aggregated_time_logs:
    #                 aggregated_time_logs[dataset] = {}
    #             for k, v in dset_time_logs.items():
    #                 agged = f(v, dim=0)
    #                 if not isinstance(
    #                     agged, torch.Tensor
    #                 ):  # Some torch args return wrappers - hack for the common pattern
    #                     agged = (
    #                         agged.values
    #                     )  # For torch tensors wrapped in something else
    #                 aggregated_time_logs[dataset][
    #                     f"{valid_or_test}_{k}_{f.__name__}"
    #                 ] = agged
    #     # Simple function just saves all time logs in place
    #     if len(aggregated_time_logs) > 0 and self.rank == 0:
    #         log_all_time_metrics(
    #             aggregated_time_logs,
    #             self.viz_folder,
    #             f"{epoch}_rank{self.rank}_{valid_or_test}",  # For the file name
    #         )
    #         # Write the aggretated time log pickles for later analysis
    #         dump_path = os.path.join(
    #             self.viz_folder,
    #             "loss_dicts",
    #             f"{valid_or_test}_time_logs_epoch{epoch}_rank{self.rank}.pkl",
    #         )
    #         if not os.path.exists(dump_path):
    #             os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    #         with open(dump_path, "wb") as f:
    #             pickle.dump(aggregated_time_logs, f)
    #         logger.info(f"Wrote out time logs to {dump_path}")
    #     # Just write the pickles out.
    #     if len(aggregated_losses) > 0 and self.rank == 0:
    #         # Write the aggretated loss pickles for later analysis
    #         dump_path = os.path.join(
    #             self.viz_folder,
    #             "loss_dicts",
    #             f"{valid_or_test}_loss_dict_epoch{epoch}_rank{self.rank}.pkl",
    #         )
    #         if not os.path.exists(dump_path):
    #             os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    #         with open(dump_path, "wb") as f:
    #             pickle.dump(aggregated_losses, f)
    #         logger.info(f"Wrote out loss dict to {dump_path}")
    #     # Misc metrics
    #     aggregated_losses["param_norm"] = param_norm(self.model.parameters())
    #     return validation_loss, aggregated_losses

    @torch.no_grad()
    def validation_loop(
        self,
        dataloaders: list[WellDataset],
        valid_or_test: str = "valid",
        full: bool = False,
        epoch: int = 0,
        only_rel_l2: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        """Run validation by looping over the dataloader.

        Validate the same dataset over FSDP groups since they're locked
        by syncs but distribute over replication (DDP) groups.

        WARNING: running DDP or HSDP with a number of datasets that either dont
        evenly divide the number of shard groups/GPUs or that have
        significantly varying sizes will probably result in a timeout at one of these barriers.
        """
        full = full if not self.debug_mode else False
        self.model.eval()

        if only_rel_l2:
            return self._validation_rel_l2_only(
                dataloaders=dataloaders,
                valid_or_test=valid_or_test,
                full=full,
                epoch=epoch,
            )

        validation_loss = 0.0
        metadatas = []

        # Timeouts get really annoying - barrier at start makes it slightly better
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Each dataset being validated gets separate loader
        rank_loss_dict: dict[str, Any] = {}
        rank_time_logs: dict[str, Any] = {}

        for i, dataloader in enumerate(dataloaders):
            # Grab metadata for the current dataset
            assert len(dataloader.dataset.sub_dsets) == 1, (
                "Only one dataset per validation dataloader"
            )
            dataset = dataloader.dataset.sub_dsets[0]  # There is only one dset by design
            dset_time_logs: dict[str, Any] = {}

            current_metadata = dataset.metadata
            metadatas.append(current_metadata)

            dset_name = current_metadata.dataset_name
            field_names = flatten_field_names(current_metadata, include_constants=False)

            # NOTE MM - We want each FSDP group to process a different dataset to accelerate
            # partial validation between epochs so we assign datasets to groups in round-robin fashion.
            rank_assignment = i % self.num_sync_groups  # Use same data across one sync group

            # Only print if we're doing something on this node
            if rank_assignment == self.sync_group_rank:
                logger.info(
                    f"Validating dataset {dataset.metadata.dataset_name} "
                    f"with full_trajectory_mode={dataset.full_trajectory_mode} on rank {self.rank}"
                )
            else:
                continue

            count = 0
            denom = len(dataloader) if full else min(len(dataloader), self.short_validation_length)

            # ---- inference timing (per dataset) ----
            infer_time_s_total = 0.0   # total wall time spent in rollout_model (summed across batches)
            infer_samples_total = 0    # total #samples (sum of batch sizes)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.enable_amp,
                dtype=self.amp_type,
            ):
                for j, batch in enumerate(dataloader):
                    # Validation datasets don't automatically add metadata
                    start_time = time.time()

                    # Rollout for length of target
                    batch_B = None

                    for kk in range(self.validation_full_trajectory_ensemble_size):
                        t0 = time.time()
                        y_pred_internal, y_ref_internal = self.rollout_model(
                            self.model,
                            batch,
                            self.formatter_dict[dset_name],
                            train=False,
                            fake_pass=(rank_assignment != self.sync_group_rank),
                            full_rollout=(
                                dset_name.lower() == "breakflow"
                                and valid_or_test in ("test", "rollout_test")
                            ),
                        )
                        t1 = time.time()

                        infer_time_s_total += (t1 - t0)
                        if batch_B is None:
                            batch_B = int(y_ref_internal.shape[0])
                            infer_samples_total += batch_B

                        # logger.info(f"[INFER] step time: {(t1-t0) / batch_B}")
                        # logger.info(f"[INFER] step time: {(t1 - t0) / (batch_B * self.validation_full_trajectory_ensemble_size)}")

                        if kk == 0:
                            y_pred = y_pred_internal / self.validation_full_trajectory_ensemble_size
                            y_ref = y_ref_internal
                        else:
                            y_pred += y_pred_internal / self.validation_full_trajectory_ensemble_size

                    assert y_ref.shape == y_pred.shape, (
                        f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                    )

                    # model_time = end-to-end time for this batch (includes rollout + losses + overhead)
                    model_time = time.time() - start_time

                    if batch["padded_field_mask"].shape[0] != y_pred.shape[-1]:
                        # Quick hack for Neutron and underscores in the dimension names...
                        batch["padded_field_mask"] = batch["padded_field_mask"][: y_pred.shape[-1]]

                    # Don't evaluate loss of "padded" dimensions so we're not biased towards predicting 0
                    y_pred, y_ref = (
                        y_pred[..., batch["padded_field_mask"]],
                        y_ref[..., batch["padded_field_mask"]],
                    )

                    used_field_names = [
                        f for ii, f in enumerate(field_names) if batch["padded_field_mask"][ii]
                    ]

                    dnrmse_channel_scores = dimensionwise_nrmse(
                        y_pred, y_ref, eps=self.validation_epsilon
                    )
                    for channel_idx, field_name in enumerate(used_field_names):
                        loss_name = f"{dset_name}/dimensionwise_nrmse/{field_name}"
                        channel_scores = (
                            dnrmse_channel_scores[:, channel_idx].detach().cpu()
                        )
                        if loss_name in rank_loss_dict:
                            rank_loss_dict[loss_name] = torch.cat(
                                [rank_loss_dict[loss_name], channel_scores], dim=0
                            )
                        else:
                            rank_loss_dict[loss_name] = channel_scores
                    dnrmse_macro_name = f"{dset_name}/dnrmse"
                    dnrmse_macro_scores = (
                        dnrmse_channel_scores.mean(dim=1).detach().cpu()
                    )
                    if dnrmse_macro_name in rank_loss_dict:
                        rank_loss_dict[dnrmse_macro_name] = torch.cat(
                            [rank_loss_dict[dnrmse_macro_name], dnrmse_macro_scores],
                            dim=0,
                        )
                    else:
                        rank_loss_dict[dnrmse_macro_name] = dnrmse_macro_scores

                    if dset_name.lower() == "breakflow":
                        physical_batch_scores = channelwise_physical_nrmse(
                            y_pred,
                            y_ref,
                            used_field_names,
                            fluid_mask=_breakflow_fluid_mask(
                                batch,
                                current_metadata,
                                y_pred.device,
                            ),
                            eps=self.validation_epsilon,
                        )
                        for channel_idx, field_name in enumerate(used_field_names):
                            canonical_name = (
                                _canonical_multi_obstacle_field_name(field_name)
                                or str(field_name)
                            )
                            if canonical_name == "pressure":
                                canonical_name = "pressure_gauge_centered"
                            loss_name = (
                                f"{dset_name}/channelwise_physical_nrmse/"
                                f"{canonical_name}"
                            )
                            channel_scores = (
                                physical_batch_scores[:, channel_idx].detach().cpu()
                            )
                            if loss_name in rank_loss_dict:
                                rank_loss_dict[loss_name] = torch.cat(
                                    [rank_loss_dict[loss_name], channel_scores], dim=0
                                )
                            else:
                                rank_loss_dict[loss_name] = channel_scores

                        macro_loss_name = (
                            f"{dset_name}/channelwise_physical_nrmse_macro"
                        )
                        macro_scores = physical_batch_scores.mean(dim=1).detach().cpu()
                        if macro_loss_name in rank_loss_dict:
                            rank_loss_dict[macro_loss_name] = torch.cat(
                                [rank_loss_dict[macro_loss_name], macro_scores], dim=0
                            )
                        else:
                            rank_loss_dict[macro_loss_name] = macro_scores

                        time_avg_batch_scores = multi_obstacle_time_avg_scores(
                            y_pred,
                            y_ref,
                            used_field_names,
                            last_n=160,
                        )
                        for score_name, batch_scores in time_avg_batch_scores.items():
                            loss_name = f"{dset_name}/{score_name}_nrmse"
                            batch_scores = batch_scores.detach().cpu()
                            if loss_name in rank_loss_dict:
                                rank_loss_dict[loss_name] = torch.cat(
                                    [rank_loss_dict[loss_name], batch_scores], dim=0
                                )
                            else:
                                rank_loss_dict[loss_name] = batch_scores

                    # Iterate through all validation metrics and log them.
                    vrmse = float("nan")
                    for loss_fn in self.validation_suite:
                        if self.skip_spectral_metrics and "spectr" in loss_fn.__class__.__name__:
                            continue

                        loss = loss_fn(y_pred, y_ref, current_metadata, eps=self.validation_epsilon)

                        if not isinstance(loss, dict):
                            loss = {loss_fn.__class__.__name__: loss}

                        for k, sub_loss in loss.items():
                            new_losses, new_time_logs = self.split_up_losses(
                                sub_loss, k, dset_name, used_field_names
                            )

                            for loss_name, batch_losses in new_losses.items():
                                if loss_name in rank_loss_dict:
                                    rank_loss_dict[loss_name] = torch.cat(
                                        [rank_loss_dict[loss_name], batch_losses], dim=0
                                    )
                                else:
                                    rank_loss_dict[loss_name] = batch_losses

                                if "full_VRMSE_T=all" in loss_name:
                                    vrmse = batch_losses.mean().item()

                            if dataset.full_trajectory_mode:
                                for loss_name, batch_time_logs in new_time_logs.items():
                                    if loss_name in dset_time_logs:
                                        dset_time_logs[loss_name] = torch.cat(
                                            [dset_time_logs[loss_name], batch_time_logs],
                                            dim=0,
                                        )
                                    else:
                                        dset_time_logs[loss_name] = batch_time_logs

                    # Trajectory metrics (optional)
                    if dataset.full_trajectory_mode:
                        for traj_loss_fn in self.validation_trajectory_metrics or []:
                            traj_loss = traj_loss_fn(
                                y_pred, y_ref, current_metadata, batch["metadata"]
                            )
                            if not isinstance(traj_loss, dict):
                                traj_loss = {traj_loss_fn.__class__.__name__: traj_loss}

                            for k, sub_traj_loss in traj_loss.items():
                                new_traj_losses, _ = self.split_up_losses(
                                    sub_traj_loss, k, dset_name, used_field_names
                                )
                                for loss_name, batch_losses in new_traj_losses.items():
                                    if loss_name in rank_loss_dict:
                                        rank_loss_dict[loss_name] = torch.cat(
                                            [rank_loss_dict[loss_name], batch_losses], dim=0
                                        )
                                    else:
                                        rank_loss_dict[loss_name] = batch_losses

                    total_time = time.time() - start_time
                    max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3

                    if rank_assignment == self.sync_group_rank:
                        logger.info(
                            f"{valid_or_test}: {dset_name}, Batch {j + 1}/{denom}, Rank {self.rank:>3}: "
                            f"Field-time-averaged VRMSE {vrmse:7.4f}, mem {max_mem_GB:5.2f} GB, "
                            f"total_time {total_time:5.3f}s, model {model_time:5.4f}s"
                        )

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    count += 1

                    # Detailed outputs on rank 0 if specified
                    if self.rank_in_sync_group == 0 and rank_assignment == self.sync_group_rank:
                        if dataset.full_trajectory_mode:
                            if self.video_validation and count < self.num_detailed_logs:
                                try:
                                    make_video(
                                        y_pred[0],
                                        y_ref[0],
                                        current_metadata,
                                        self.viz_folder,
                                        f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",
                                        field_name_overrides=used_field_names,
                                        size_multiplier=self.video_size_multiplier,
                                    )
                                except CalledProcessError as e:
                                    logger.warning(
                                        f"Error in making video due to FFMPEG: {e}. Skipping video."
                                    )

                            if self.dump_prediction_to_disk and count < self.num_detailed_logs:
                                dump_path = os.path.join(
                                    self.viz_folder, dset_name, "full_trajectory_dumps"
                                )
                                os.makedirs(dump_path, exist_ok=True)
                                np.save(
                                    os.path.join(
                                        dump_path,
                                        f"yref_{dset_name}_{valid_or_test}_epoch{epoch}_rank{self.rank}_{j}.npy",
                                    ),
                                    y_ref.cpu().numpy(),
                                )
                                np.save(
                                    os.path.join(
                                        dump_path,
                                        f"ypred_{dset_name}_{valid_or_test}_epoch{epoch}_rank{self.rank}_{j}.npy",
                                    ),
                                    y_pred.cpu().numpy(),
                                )
                                logger.info(f"Wrote out npy dumps to {dump_path}")

                    if not full and count >= self.short_validation_length:
                        break

                # ---- print avg inference time per sample (once per dataset) ----
                if rank_assignment == self.sync_group_rank:
                    avg_s_per_sample = (
                        infer_time_s_total / float(infer_samples_total)
                        if infer_samples_total > 0
                        else float("nan")
                    )
                    logger.info(
                        f"{valid_or_test}: {dset_name}, Rank {self.rank:>3}: "
                        f"avg_infer_time {avg_s_per_sample:.6f} s/sample "
                        f"(total {infer_time_s_total:.3f}s over {infer_samples_total} samples)"
                    )
                    # optional: stash as a 1-element tensor so it can flow through aggregation
                    rank_loss_dict[f"{dset_name}/avg_infer_time_s_per_sample"] = torch.tensor(
                        [avg_s_per_sample], dtype=torch.float32
                    )

            # Run some last outputs on rank 0 to get a general sense of what's going on
            if self.rank_in_sync_group == 0 and rank_assignment == self.sync_group_rank:
                if self.image_validation:
                    for plot_fn in validation_plots:
                        if self.skip_spectral_metrics and "spectr" in plot_fn.__name__:
                            continue
                        plot_fn(
                            y_pred,
                            y_ref,
                            current_metadata,
                            self.viz_folder,
                            f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",
                        )
                if dataset.full_trajectory_mode and self.video_validation:
                    try:
                        make_video(
                            y_pred[0],
                            y_ref[0],
                            current_metadata,
                            self.viz_folder,
                            f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",
                            field_name_overrides=used_field_names,
                            size_multiplier=self.video_size_multiplier,
                        )
                    except CalledProcessError as e:
                        logger.warning(
                            f"Error in making video due to FFMPEG: {e}. Skipping video."
                        )

            if dataset.full_trajectory_mode:
                rank_time_logs[current_metadata.dataset_name] = dset_time_logs

        # If we're distributed, now send all per rank results to rank 0 for aggregation
        if self.is_distributed:
            logger.info(f"Rank {self.rank} waiting for barrier")
            dist.barrier()
            logger.info(f"Rank {self.rank} passed barrier")

            object_loss_dict = {k: v.cpu() for k, v in rank_loss_dict.items()}
            loss_dicts = [None for _ in range(self.world_size)]
            time_logs_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(loss_dicts, object_loss_dict)
            dist.all_gather_object(time_logs_list, rank_time_logs)

            loss_dict: dict[str, Any] = {}
            for ld in loss_dicts:
                for k, v in ld.items():
                    if k not in loss_dict:
                        loss_dict[k] = v
                    else:
                        loss_dict[k] = torch.cat([loss_dict[k], v], dim=0)

            time_logs: dict[str, Any] = {}
            for tl in time_logs_list:
                for dataset_name, dset_time_logs in tl.items():
                    if dataset_name not in time_logs:
                        time_logs[dataset_name] = {}
                    for k, v in dset_time_logs.items():
                        if k not in time_logs[dataset_name]:
                            time_logs[dataset_name][k] = v
                        else:
                            time_logs[dataset_name][k] = torch.cat(
                                [time_logs[dataset_name][k], v], dim=0
                            )
        else:
            loss_dict = rank_loss_dict
            time_logs = rank_time_logs

        # --- [REL_L2_DIST_PLOT] save per-sample RelL2 distribution for final eval on rank 0 ---
        if self.rank == 0 and valid_or_test in ("test", "rollout_test"):
            for meta in metadatas:
                dset = meta.dataset_name
                key = f"{dset}/full_RelL2_T=all"
                if key not in loss_dict:
                    logger.warning("[REL_L2_DIST_PLOT] Missing key %s in loss_dict.", key)
                    continue

                rel = loss_dict[key]
                rel_np = rel.detach().cpu().numpy() if isinstance(rel, torch.Tensor) else np.asarray(rel)

                out_dir = os.path.join(self.viz_folder, dset, "distributions")
                tag = f"{valid_or_test}_epoch{epoch}_rank{self.rank}"
                npz_path, fig_path = save_rel_l2_distribution_plot(
                    rel_l2_values=rel_np,
                    out_dir=out_dir,
                    tag=tag,
                    bins=60,
                    log_x=True,
                )

                if self.wandb_logging and npz_path is not None and fig_path is not None:
                    wandb.log(
                        {f"{dset}/{tag}/relL2_dist": wandb.Image(fig_path), "epoch": epoch},
                        commit=False,
                    )
                    wandb.save(npz_path, base_path=self.viz_folder)

            time_avg_wandb = {}
            for loss_name, values in loss_dict.items():
                marker = "/tavg_"
                if marker not in loss_name or not loss_name.endswith("_nrmse"):
                    continue
                _, score_name = loss_name.split(marker, 1)
                mode, quantity = score_name[:-len("_nrmse")].split("_", 1)
                mean_score = (
                    values.float().mean().item()
                    if isinstance(values, torch.Tensor)
                    else float(np.asarray(values).mean())
                )
                time_avg_wandb[f"test/{quantity}/tavg_{mode}_nrmse"] = mean_score

            if time_avg_wandb:
                logger.info(
                    "Multi-obstacle time-average statistics (last 160 rollout frames): %s",
                    time_avg_wandb,
                )
                if self.wandb_logging:
                    wandb.log(time_avg_wandb, commit=False)

            physical_nrmse_wandb = {}
            for loss_name, values in loss_dict.items():
                marker = "/channelwise_physical_nrmse/"
                if marker in loss_name:
                    channel_name = loss_name.split(marker, 1)[1]
                    physical_nrmse_wandb[
                        f"test/channelwise_physical_nrmse/{channel_name}"
                    ] = values.float().mean().item()
                elif loss_name.endswith("/channelwise_physical_nrmse_macro"):
                    macro_score = values.float().mean().item()
                    physical_nrmse_wandb[
                        "test/channelwise_physical_nrmse_macro"
                    ] = macro_score
                    physical_nrmse_wandb["test/nrmse"] = macro_score

            if physical_nrmse_wandb:
                logger.info(
                    "Channelwise physical-unit nRMSE: %s",
                    physical_nrmse_wandb,
                )
                if self.wandb_logging:
                    wandb.log(physical_nrmse_wandb, commit=False)

            dnrmse_wandb = {}
            for metadata in metadatas:
                dset_name = metadata.dataset_name
                key = f"{dset_name}/dnrmse"
                if key not in loss_dict:
                    continue
                mean_dnrmse = loss_dict[key].float().mean().item()
                dnrmse_wandb[f"test/{dset_name}/dnrmse"] = mean_dnrmse
            if len(dnrmse_wandb) == 1:
                dnrmse_wandb["test/dnrmse"] = next(iter(dnrmse_wandb.values()))
            if dnrmse_wandb:
                logger.info("Dimensionwise nRMSE: %s", dnrmse_wandb)
                if self.wandb_logging:
                    wandb.log(dnrmse_wandb, commit=False)

        # Single score validation loss is average of all losses on the training metric
        validation_loss = sum(
            [
                loss_dict[
                    f"{metadata.dataset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
                ]
                .mean()
                .item()
                for metadata in metadatas
            ]
        ) / len(metadatas)

        aggregated_losses: dict[str, Any] = {}
        aggregated_time_logs: dict[str, Any] = {}

        for f in self.batch_aggregation_fns:
            for k, v in loss_dict.items():
                # For scalar-ish one-element tensors (like avg_infer_time), mean is fine too.
                aggregated_losses[f"{valid_or_test}_{k}_{f.__name__}"] = f(v)

            for dataset_name, dset_time_logs in time_logs.items():
                if dataset_name not in aggregated_time_logs:
                    aggregated_time_logs[dataset_name] = {}
                for k, v in dset_time_logs.items():
                    agged = f(v, dim=0)
                    if not isinstance(agged, torch.Tensor):
                        agged = agged.values
                    aggregated_time_logs[dataset_name][f"{valid_or_test}_{k}_{f.__name__}"] = agged

        if len(aggregated_time_logs) > 0 and self.rank == 0:
            log_all_time_metrics(
                aggregated_time_logs,
                self.viz_folder,
                f"{epoch}_rank{self.rank}_{valid_or_test}",
            )
            dump_path = os.path.join(
                self.viz_folder,
                "loss_dicts",
                f"{valid_or_test}_time_logs_epoch{epoch}_rank{self.rank}.pkl",
            )
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "wb") as f:
                pickle.dump(aggregated_time_logs, f)
            logger.info(f"Wrote out time logs to {dump_path}")

        if len(aggregated_losses) > 0 and self.rank == 0:
            dump_path = os.path.join(
                self.viz_folder,
                "loss_dicts",
                f"{valid_or_test}_loss_dict_epoch{epoch}_rank{self.rank}.pkl",
            )
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "wb") as f:
                pickle.dump(aggregated_losses, f)
            logger.info(f"Wrote out loss dict to {dump_path}")

        aggregated_losses["param_norm"] = param_norm(self.model.parameters())
        return validation_loss, aggregated_losses
    
    
    def train_one_epoch(
        self, epoch: int, dataloader: DataLoader
    ) -> tuple[float, dict[str, Any]]:
        self.model.train()
        epoch_loss = 0.0
        avg_grad_norm = torch.tensor(0.0).to(self.device)
        last_grad_norm = torch.tensor(0.0).to(self.device)
        train_logs: dict[str, Any] = {}
        batch_start = time.time()
        interval_start = time.time()

        self.optimizer.zero_grad()
        overall_batch_queue = []
        current_batch_queue = []
        data_iter = iter(dataloader)
        i = 0

        # --- grad accumulation ---
        if epoch < self.big_batch_before:
            grad_acc_steps = self.grad_acc_steps * self.big_batch_multiplier
        else:
            grad_acc_steps = self.grad_acc_steps

        # --- LR logging ---
        lr_log_every = 50
        opt_step = 0

        def _get_lrs() -> list[float]:
            return [float(pg["lr"]) for pg in self.optimizer.param_groups]

        def _wandb_log(payload: dict[str, Any]):
            if self.wandb_logging and self.rank == 0:
                wandb.log(payload, commit=False)

        def _log_lr():
            lrs = [float(pg["lr"]) for pg in self.optimizer.param_groups]
            payload = {"epoch": int(epoch), "train/opt_step": int(opt_step)}

            if len(lrs) == 1:
                payload["train/lr"] = lrs[0]
            else:
                payload["train/lr"] = lrs[0]
                for gi, lr in enumerate(lrs):
                    payload[f"train/lr_group{gi}"] = lr

            if self.wandb_logging and self.rank == 0:
                wandb.log(payload, commit=False)

        if lr_log_every != 0:
            _log_lr()

        while i < len(dataloader):
            if self.reuse_batches and (
                len(overall_batch_queue) >= 2 * grad_acc_steps
                or len(current_batch_queue) > 0
            ):
                if len(current_batch_queue) == 0:
                    if grad_acc_steps > 1:
                        shuffle(overall_batch_queue)
                    current_batch_queue = overall_batch_queue[:grad_acc_steps]
                    overall_batch_queue = overall_batch_queue[grad_acc_steps:]
                batch = current_batch_queue.pop(0)
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                if self.reuse_batches:
                    overall_batch_queue.append(batch.copy())

            batch["padded_field_mask"] = batch["padded_field_mask"].to(
                self.device, non_blocking=True
            )

            update_grad = (i + 1) % grad_acc_steps == 0
            with (
                nullcontext()
                if (update_grad or self.distribution_type == "local")
                else self.model.no_sync()
            ):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.enable_amp,
                    dtype=self.amp_type,
                ):
                    data_time = time.time() - batch_start
                    current_metadata = batch["metadata"]
                    dset_name = current_metadata.dataset_name

                    ts = time.time()
                    y_pred, y_ref = self.rollout_model(
                        self.model, batch, self.formatter_dict[dset_name]
                    )
                    te = time.time()

                    B = int(y_ref.shape[0])

                    # logger.info(f"[INFER] step time: {(te-ts)/max(B,1)} s/sample")

                    if y_pred.shape[1] > self.minimum_context:
                        y_ref = y_ref[:, self.minimum_context :]
                        y_pred = y_pred[:, self.minimum_context :]

                    forward_time = time.time() - batch_start - data_time

                    assert y_ref.shape == y_pred.shape, (
                        f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                    )

                    loss = (
                        self.loss_multiplier
                        * self.loss_fn(
                            y_pred, y_ref, current_metadata, eps=self.model_epsilon
                        ).mean()
                        / grad_acc_steps
                    )

                    if not torch.isfinite(loss):
                        logger.error("[NAN_DEBUG] loss is non-finite on epoch=%d batch=%d dset=%s", epoch, i, dset_name)
                        _check_finite("y_pred (train)", y_pred, throw=False)
                        _check_finite("y_ref (train)", y_ref, throw=False)
                        for parameter_name, parameter in self.model.named_parameters():
                            if not torch.isfinite(parameter).all():
                                _check_finite(
                                    f"model parameter {parameter_name}",
                                    parameter,
                                    throw=False,
                                )
                                break
                        raise RuntimeError("Loss became NaN/Inf (see logs above)")

                    del y_pred, y_ref
                self.grad_scaler.scale(loss).backward()
                backward_time = time.time() - batch_start - forward_time - data_time

            if update_grad:
                if self.clip_gradient > 0 or self.gradient_log_level > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                if self.gradient_log_level == 1:
                    if hasattr(self.model, "sharding_strategy"):
                        last_grad_norm = get_grad_norm_fsdp(
                            self.model,
                            self.rank,
                            self.world_size,
                            self.model.sharding_strategy,
                        )
                    else:
                        last_grad_norm = get_grad_norm_local(self.model)
                        avg_grad_norm += last_grad_norm.detach() / (
                            len(dataloader) / grad_acc_steps
                        )
                if self.clip_gradient > 0:
                    if hasattr(self.model, "clip_grad_norm_"):
                        self.model.clip_grad_norm_(
                            self.clip_gradient,
                            norm_type=2.0,
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_gradient, norm_type=2.0
                        )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                if self.lr_scheduler_per_step and self.lr_scheduler:
                    self.lr_scheduler.step()

                opt_step += 1
                if lr_log_every > 0 and (opt_step == 1 or (opt_step % lr_log_every == 0)):
                    _log_lr()

            total_time = time.time() - batch_start
            optimizer_time = total_time - forward_time - backward_time - data_time

            # Syncing for all reduce anyway so may as well compute synchornous metrics
            epoch_loss += (grad_acc_steps * loss.detach()) / len(dataloader)

            ### [TIME_BUDGET] accumulate compute-only time and early-stop
            step_compute = float(forward_time + backward_time + optimizer_time)
            self.train_compute_time_s += step_compute

            if (
                self.effective_train_budget_s is not None
                and self.train_compute_time_s >= self.effective_train_budget_s
            ):
                logger.info(
                    "[TIME_BUDGET] Stop: train_compute_time_s=%.3f >= effective_train_budget_s=%.3f "
                    "(time_budget_s=%.3f, gen_cost_s=%.3f, Ndata_generated=%d, gen_time_s=%.6f). "
                    "Stopping at epoch %d, batch %d/%d.",
                    self.train_compute_time_s,
                    self.effective_train_budget_s,
                    self.time_budget_s,
                    self.gen_cost_s,
                    self.Ndata_generated,
                    self.gen_time_s,
                    epoch,
                    i + 1,
                    len(dataloader),
                )
                # Log budget info for wandb
                train_logs["train_compute_time_s"] = self.train_compute_time_s
                train_logs["effective_train_budget_s"] = self.effective_train_budget_s
                train_logs["gen_cost_s"] = self.gen_cost_s
                train_logs["Ndata_generated"] = self.Ndata_generated
                train_logs["gen_time_s"] = self.gen_time_s
                # break out of epoch loop
                i += 1
                break

            max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
            if i % self.log_interval == 0:
                timing = (time.time() - interval_start) / self.log_interval
                interval_start = time.time()
                logger.info(
                    f"Epoch {epoch:>4}, Batch {i + 1}/{len(dataloader)}, Rank {self.rank:>3}, SyncStep: {update_grad}:\n\t"
                    f"Data: {current_metadata.dataset_name:<32}, loss {(grad_acc_steps * loss.item()) ** 0.5:7.4f}, "
                    f"mem {max_mem_GB:5.2f} GB, total_time {timing:5.3f}s, data {data_time:5.4f}s, "
                    f"fwd {forward_time:5.3f}s, bw {backward_time:5.3f}s, opt {optimizer_time:5.3f}s, "
                    f"compute_total {self.train_compute_time_s:8.2f}s"
                )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            batch_start = time.time()

            train_logs["avg_data_loading_time"] = train_logs.get(
                "avg_data_loading_time", 0
            ) + data_time / len(dataloader)
            train_logs["avg_forward_time"] = train_logs.get(
                "avg_forward_time", 0
            ) + forward_time / len(dataloader)
            train_logs["avg_backward_time"] = train_logs.get(
                "avg_backward_time", 0
            ) + backward_time / len(dataloader)
            train_logs["avg_optimizer_time"] = train_logs.get(
                "avg_optimizer_time", 0
            ) + optimizer_time / len(dataloader)
            train_logs["avg_time_per_step"] = train_logs.get(
                "avg_time_per_step", 0
            ) + total_time / len(dataloader)
            train_logs["peak_memory"] = max(train_logs.get("peak_memory", 0), max_mem_GB)

            ### [TIME_BUDGET] also log compute-only time
            train_logs["train_compute_time_s"] = self.train_compute_time_s
            if self.effective_train_budget_s is not None:
                train_logs["effective_train_budget_s"] = self.effective_train_budget_s
                train_logs["gen_cost_s"] = self.gen_cost_s
                train_logs["Ndata_generated"] = self.Ndata_generated
                train_logs["gen_time_s"] = self.gen_time_s

            i += 1

        train_logs["train_loss"] = epoch_loss
        if self.gradient_log_level >= 1:
            train_logs["avg_grad_norm"] = avg_grad_norm.item()
        if self.lr_scheduler:
            if not self.lr_scheduler_per_step:
                self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs

    def validate_if_necessary(
        self,
        epoch: int,
        one_step_dataloaders: list[DataLoader],
        rollout_dataloaders: list[DataLoader],
        valid_or_test: Literal["valid", "test"] = "valid",
    ):
        """Check what type of validate/rollouts we need to do for a given epoch.

        Parameters
        ----------
        epoch: int
            The current epoch. Used for logging and saving checkpoints.
        one_step_dataloaders: list[DataLoader]
            List of dataloaders for one step validation
        rollout_dataloaders: list[DataLoader]
            List of dataloaders for rollout validation
        valid_or_test: str
            String to indicate if we are validating or testing. Options are "valid" or "test"
        """
        is_test = valid_or_test == "test"  # Check if test
        val_loss, rollout_val_loss = None, None
        # # First do one step checks = frequency, last epoch, or test. Only do full validation on last epoch or test
        # if epoch % self.val_frequency == 0 or epoch >= self.max_epoch or is_test:
        #     logger.info(
        #         f"Epoch {epoch}/{self.max_epoch}: starting {valid_or_test} validation"
        #     )
        #     val_loss, loss_dict = self.validation_loop(
        #         one_step_dataloaders,
        #         valid_or_test=valid_or_test,
        #         full=(epoch >= self.max_epoch or is_test),
        #         epoch=epoch,
        #     )
        #     logger.info(
        #         f"Epoch {epoch}/{self.max_epoch}: {valid_or_test} loss {val_loss}"
        #     )
        #     loss_dict |= {f"{valid_or_test}": val_loss, "epoch": epoch}

        #     if self.wandb_logging and self.rank == 0:
        #         wandb.log(loss_dict)

        # Rollout if frequency, last epoch, or if this is the test set
        if (
            epoch % self.rollout_val_frequency == 0
            or epoch >= self.max_epoch
            or is_test
        ):
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: starting rollout {valid_or_test} validation"
            )
            rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                rollout_dataloaders,
                valid_or_test=f"rollout_{valid_or_test}",
                full=True,
                # full=epoch >= self.max_epoch,
                epoch=epoch,
            )
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: rollout {valid_or_test} loss {rollout_val_loss}"
            )
            rollout_val_loss_dict |= {
                f"rollout_{valid_or_test}": rollout_val_loss,
                "epoch": epoch,
            }
            if self.wandb_logging and self.rank == 0:
                wandb.log(rollout_val_loss_dict)
        return val_loss, rollout_val_loss

    def _time_budget_exceeded(self) -> bool:
        if self.time_budget_s is None:
            return False

        # compute effective budget (your existing logic)
        gen_cost_s = 0.0
        if self.Ndata_generated is not None:
            gen_cost_s = float(self.Ndata_generated) * float(self.gen_time_s)

        effective_train_budget_s = float(self.time_budget_s) - gen_cost_s
        if effective_train_budget_s <= 0:
            effective_train_budget_s = 0.0

        return float(self.train_compute_time_s) >= effective_train_budget_s

    def _profile_optimizer_step_time(
        self,
        dataloader: DataLoader,
        warmup_optimizer_steps: int = 10,
        burn_in_optimizer_steps: int = 2,
    ) -> float:
        """
        Runs a short training-like loop to estimate seconds per *optimizer step*.
        Uses real forward/backward/step path, respects grad_acc_steps.
        Returns avg seconds / optimizer step (compute-only).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        data_iter = iter(dataloader)
        opt_steps_done = 0
        opt_step_times = []

        # match your epoch logic for grad_acc_steps
        grad_acc_steps = self.grad_acc_steps

        # microstep index
        i = 0
        while opt_steps_done < warmup_optimizer_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch["padded_field_mask"] = batch["padded_field_mask"].to(self.device, non_blocking=True)

            update_grad = ((i + 1) % grad_acc_steps == 0)

            # start timing ONLY for optimizer step window (across its microsteps)
            if update_grad:
                t0 = time.time()

            with (nullcontext() if (update_grad or self.distribution_type == "local") else self.model.no_sync()):
                with torch.autocast(device_type=self.device.type, enabled=self.enable_amp, dtype=self.amp_type):
                    current_metadata = batch["metadata"]
                    dset_name = current_metadata.dataset_name

                    y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter_dict[dset_name])
                    if y_pred.shape[1] > self.minimum_context:
                        y_ref = y_ref[:, self.minimum_context :]
                        y_pred = y_pred[:, self.minimum_context :]

                    loss = (
                        self.loss_multiplier
                        * self.loss_fn(y_pred, y_ref, current_metadata, eps=self.model_epsilon).mean()
                        / grad_acc_steps
                    )

            self.grad_scaler.scale(loss).backward()

            if update_grad:
                if self.clip_gradient > 0 or self.gradient_log_level > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                if self.clip_gradient > 0:
                    if hasattr(self.model, "clip_grad_norm_"):
                        self.model.clip_grad_norm_(self.clip_gradient, norm_type=2.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient, norm_type=2.0)

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                t1 = time.time()
                opt_steps_done += 1

                logger.info(f"[WARMUP] step time: {t1-t0}")

                # discard first few optimizer steps (compile/caches)
                if opt_steps_done > burn_in_optimizer_steps:
                    opt_step_times.append(t1 - t0)

            i += 1

        avg = float(np.mean(opt_step_times)) if len(opt_step_times) else float("nan")
        if self.rank == 0:
            logger.info("[WARMUP] avg seconds/optimizer_step = %.4f (from %d samples)", avg, len(opt_step_times))
        return avg

    def _reset_scheduler_from_budget(self, sec_per_opt_step: float):
        """
        Rebuild self.lr_scheduler so decay begins after ~10% of total optimizer steps,
        based on effective_train_budget_s and measured sec_per_opt_step.
        """
        if self.lr_scheduler is None:
            return
        if self.effective_train_budget_s is None:
            # no time budget => can't infer total steps; do nothing
            return
        if not np.isfinite(sec_per_opt_step) or sec_per_opt_step <= 0:
            return

        total_opt_steps = int(self.effective_train_budget_s / sec_per_opt_step)
        total_opt_steps = max(total_opt_steps, 1)

        warmup_steps = max(int(0.10 * total_opt_steps), 1)
        decay_steps = max(total_opt_steps - warmup_steps, 1)

        # Grab base LR(s) from optimizer
        base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # Example schedule: warm up from 0.1*lr -> 1.0*lr, then cosine to 0
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=decay_steps, eta_min=0.0
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        # IMPORTANT: scheduler steps must be per optimizer step for this to mean what you want
        self.lr_scheduler_per_step = True

        if self.rank == 0:
            logger.info(
                "[SCHED_RESET] total_opt_steps=%d warmup_steps=%d decay_steps=%d (decay starts at 10%%)",
                total_opt_steps, warmup_steps, decay_steps
            )
            logger.info("[SCHED_RESET] base_lrs=%s", str(base_lrs))

    @torch.no_grad()
    def profile_rollout_infer_time_on_train_batch(
        self,
        dataloader: DataLoader,
        num_batches: int = 10,
        burn_in: int = 2,
        train: bool = False,   # keep False to match validation-style rollout (denorm etc.)
    ) -> dict[str, float]:
        """
        Measures rollout_model() wall-time with the SAME batches as training loader.
        Times ONLY inference (rollout_model), no loss, no backward.

        Returns dict with avg seconds/sample and seconds/batch (local + global if distributed).
        """
        self.model.eval()
        self.do_infer_timing = True

        it = iter(dataloader)
        times = []
        n_samples = 0

        # optional: reduce noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # barrier so ranks start together
        if self.is_distributed:
            dist.barrier()

        for k in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)

            # NOTE: mirror your training path device moves
            batch["padded_field_mask"] = batch["padded_field_mask"].to(self.device, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Use the same formatter resolution as training
            dset_name = batch["metadata"].dataset_name

            t0 = 0
            t1 = 0
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.enable_amp,
                dtype=self.amp_type,
            ):
                t0 = time.time()
                _y_pred, _y_ref = self.rollout_model(
                    self.model,
                    batch,
                    self.formatter_dict[dset_name],
                    train=False,
                    fake_pass=False,
                )
                t1 = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            B = int(_y_ref.shape[0])
            n_samples += B

            print(f"[INFER TIME] per-sample={(t1 - t0) / max(B,1):.6f}s  batch_B={B}")

            # discard burn-in batches (compilation/caches)
            if k >= burn_in:
                times.append(t1 - t0)

        # local stats
        local_total_s = float(np.sum(times))
        local_batches = float(len(times))
        local_avg_s_per_batch = local_total_s / max(local_batches, 1.0)
        local_avg_s_per_sample = local_total_s / max(float(n_samples) * (local_batches / max(num_batches - burn_in, 1)), 1.0)

        out = {
            "local_avg_s_per_batch": local_avg_s_per_batch,
            "local_avg_s_per_sample": local_avg_s_per_sample,
        }

        # global aggregate (mean across ranks)
        if self.is_distributed:
            t = torch.tensor(
                [local_avg_s_per_batch, local_avg_s_per_sample],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= float(self.world_size)
            out["global_avg_s_per_batch"] = float(t[0].item())
            out["global_avg_s_per_sample"] = float(t[1].item())
        return out
    
    # @torch.no_grad()
    # def profile_rollout_infer_time_on_train_batch(
    #     self,
    #     dataloader: DataLoader,
    #     num_batches: int = 3,
    #     burn_in: int = 2,
    #     train: bool = False,   # keep False to match validation-style rollout (denorm etc.)
    # ) -> dict[str, float]:
    #     """
    #     Measures rollout_model() wall-time with the SAME batches as training loader.
    #     Times ONLY inference (rollout_model), no loss, no backward.

    #     Returns dict with avg seconds/sample and seconds/batch (local + global if distributed).
    #     """
    #     self.model.eval()
    #     self.do_infer_timing = True

    #     it = iter(dataloader)
    #     times = []
    #     n_samples = 0

    #     # optional: reduce noise
    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize()

    #     # barrier so ranks start together
    #     if self.is_distributed:
    #         dist.barrier()

    #     for k in range(num_batches):
    #         try:
    #             batch = next(it)
    #         except StopIteration:
    #             it = iter(dataloader)
    #             batch = next(it)

    #         batch["padded_field_mask"] = batch["padded_field_mask"].to(self.device, non_blocking=True)

    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()

    #         dset_name = batch["metadata"].dataset_name

    #         if k < burn_in:
    #             self.do_infer_timing = False

    #         with torch.autocast(
    #             device_type=self.device.type,
    #             enabled=self.enable_amp,
    #             dtype=self.amp_type,
    #         ):
    #             _y_pred, _y_ref = self.rollout_model(
    #                 self.model,
    #                 batch,
    #                 self.formatter_dict[dset_name],
    #                 train=False,
    #                 fake_pass=False,
    #             )

    #         if k < burn_in:
    #             self.do_infer_timing = True
    #             continue

    #         B = int(_y_ref.shape[0])
    #         n_samples += B
    #         self.infer_n_cnt += B
    #         self.infer_cnt += 1

    #         print(f"[INFER TIME]: total={self.infer_total_time}, sample={self.infer_n_cnt}")

    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()

    #     local_avg_s_per_batch  = self.infer_total_time / self.infer_cnt
    #     local_avg_s_per_sample = self.infer_total_time / self.infer_n_cnt

    #     out = {
    #         "local_avg_s_per_batch": local_avg_s_per_batch,
    #         "local_avg_s_per_sample": local_avg_s_per_sample,
    #     }

    #     self.do_infer_timing = False

    #     return out


    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        checkpoint_future = None
        val_loss = self.start_val_loss
        train_dataloader = self.datamodule.train_dataloader(self.sampling_rank)

        # infer_stats = self.profile_rollout_infer_time_on_train_batch(
        #     train_dataloader,
        #     num_batches=8,
        #     burn_in=2,
        #     train=False,
        # )
        # if self.rank == 0:
        #     logger.info(
        #         "[ROLLOUT_INFER][TRAIN_BS] global_avg_s_per_sample=%.6f, global_avg_s_per_batch=%.6f",
        #         infer_stats.get("global_avg_s_per_sample", infer_stats["local_avg_s_per_sample"]),
        #         infer_stats.get("global_avg_s_per_batch", infer_stats["local_avg_s_per_batch"]),
        #     )
        #     if self.wandb_logging:
        #         wandb.log({
        #             "rollout_infer/global_s_per_sample_train_bs": infer_stats.get("global_avg_s_per_sample", infer_stats["local_avg_s_per_sample"]),
        #             "rollout_infer/global_s_per_batch_train_bs": infer_stats.get("global_avg_s_per_batch", infer_stats["local_avg_s_per_batch"]),
        #         })

        # warm-up
        if self.rank == 0:
            logger.info("[WARMUP] starting scheduler warmup/profile...")

        sec_per_opt_step = self._profile_optimizer_step_time(
            train_dataloader,
            warmup_optimizer_steps=5,    # e.g. 5 optimizer steps
            burn_in_optimizer_steps=2,   # discard first 2
        )
        # all ranks should reset scheduler consistently
        if self.is_distributed:
            # broadcast sec_per_opt_step from rank0
            t = torch.tensor([sec_per_opt_step], device=self.device, dtype=torch.float64)
            dist.broadcast(t, src=0)
            sec_per_opt_step = float(t.item())

        self._reset_scheduler_from_budget(sec_per_opt_step)

        # profile rollout inference on training batches
        infer_stats = self.profile_rollout_infer_time_on_train_batch(
            train_dataloader,
            num_batches=8,
            burn_in=2,
            train=False,
        )
        if self.rank == 0:
            logger.info(
                "[ROLLOUT_INFER][TRAIN_BS] global_avg_s_per_sample=%.6f, global_avg_s_per_batch=%.6f",
                infer_stats.get("global_avg_s_per_sample", infer_stats["local_avg_s_per_sample"]),
                infer_stats.get("global_avg_s_per_batch", infer_stats["local_avg_s_per_batch"]),
            )
            if self.wandb_logging:
                wandb.log({
                    "rollout_infer/global_s_per_sample_train_bs": infer_stats.get("global_avg_s_per_sample", infer_stats["local_avg_s_per_sample"]),
                    "rollout_infer/global_s_per_batch_train_bs": infer_stats.get("global_avg_s_per_batch", infer_stats["local_avg_s_per_batch"]),
                })

        epoch = self.start_epoch
        for epoch in range(
            self.start_epoch, self.max_epoch + 1
        ):  # I like 1 indexing for epochs
            # NOTE - only update train sampler because we want to sample same valid data every time

            if self._time_budget_exceeded():
                logger.info(
                    f"[TIME_BUDGET] Stop BEFORE epoch {epoch}: "
                    f"train_compute_time_s={self.train_compute_time_s:.3f} exceeded budget."
                )
                break

            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            # Empty mem caches before train loop
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")
            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: training loss {train_loss:.4f}"
            )
            train_logs |= {"train": train_loss, "epoch": epoch}
            if self.wandb_logging and self.rank == 0:
                wandb.log(train_logs)
            # Empty mem caches before val
            torch.cuda.empty_cache()
            gc.collect()

            # # Recreate loader every time so we're using same data in val
            # val_dataloders = self.datamodule.val_dataloaders(
            #     replicas=self.sync_group_size,
            #     rank=self.rank_in_sync_group,
            #     full=(epoch >= self.max_epoch and not self.debug_mode),
            # )
            # rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders(
            #     replicas=self.sync_group_size,
            #     rank=self.rank_in_sync_group,
            #     full=(epoch >= self.max_epoch and not self.debug_mode),
            # )
            # maybe_val_loss, rollout_loss = self.validate_if_necessary(
            #     epoch, val_dataloders, rollout_val_dataloaders
            # )
            # val_loss = maybe_val_loss if maybe_val_loss is not None else val_loss
            
            if checkpoint_future is not None:
                logger.debug(
                    f"Wait for previous checkpointing {checkpoint_future} to complete."
                )
                checkpoint_future.result()  # Make sure previous checkpoint has finished before starting next.
            
            # Save "last" every epoch plus various intervals/best results
            if not self.skip_checkpointing:
                checkpoint_future = self.save_model_if_necessary(
                    epoch, val_loss, last=(epoch == self.max_epoch)
                )

        if self.rank == 0:
            budget_label = (
                "unlimited"
                if self.time_budget_s is None
                else str(int(self.time_budget_s))
            )
            out = os.path.join(
                self.save_path,
                f"B{budget_label}_N{int(self.Ndata_generated)}.pt",
            )
            os.makedirs(self.save_path, exist_ok=True)
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "train_compute_time": self.train_compute_time_s,
            }
            if self.lr_scheduler is not None:
                checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
            torch.save(
                checkpoint,
                out,
            )
            logger.info(f"[SAVE] Saved final checkpoint to {out}")
            if self.wandb_logging and wandb.run is not None:
                wandb.run.summary["checkpoint_path"] = out

        # Do test validation
        test_dataloaders = self.datamodule.test_dataloaders(
            self.sync_group_size, rank=self.rank_in_sync_group, full=not self.debug_mode
        )
        rollout_test_dataloaders = self.datamodule.rollout_test_dataloaders(
            replicas=self.sync_group_size,
            rank=self.rank_in_sync_group,
            full=not self.debug_mode,
        )
        self.validate_if_necessary(
            epoch, test_dataloaders, rollout_test_dataloaders, valid_or_test="test"
        )


    def validate(self):
        """Run validation and test. This is a stand alone path"""
        val_dataloders = self.datamodule.val_dataloaders(
            replicas=self.sync_group_size,
            rank=self.rank_in_sync_group,
            full=not self.debug_mode,
        )
        rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders(
            replicas=self.sync_group_size,
            rank=self.rank_in_sync_group,
            full=not self.debug_mode,
        )
        test_dataloaders = self.datamodule.test_dataloaders(
            replicas=self.sync_group_size,
            rank=self.rank_in_sync_group,
            full=not self.debug_mode,
        )
        rollout_test_dataloaders = self.datamodule.rollout_test_dataloaders(
            replicas=self.sync_group_size,
            rank=self.rank_in_sync_group,
            full=not self.debug_mode,
        )
        # Run validation and test
        self.validate_if_necessary(
            self.max_epoch + 1, val_dataloders, rollout_val_dataloaders
        )
        self.validate_if_necessary(
            self.max_epoch + 1,
            test_dataloaders,
            rollout_test_dataloaders,
            valid_or_test="test",
        )
