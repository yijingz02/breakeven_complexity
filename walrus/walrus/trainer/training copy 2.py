# --- PATCH: add compute-only time budget discounted by one-time data generation cost ---
# Changes are marked with "### [TIME_BUDGET]" comments.

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

# --- [REL_L2_DIST_PLOT] save raw per-sample RelL2 and plot histogram + smooth overlay (log-x) ---

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

def log_all_time_metrics(
    time_logs: dict,
    output_dir: str,
    epoch_number: str = "0",
):
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
    T = target.shape[1]
    C = target.shape[-1]
    expansion_tuple = (-1, T)
    expansion_tuple = expansion_tuple + (-1,) * (len(target.shape) - 3) + (C,)
    mask = mask.unsqueeze(1).expand(*expansion_tuple)
    return mask


def get_grad_norm_local(model) -> torch.Tensor:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


def get_grad_norm_fsdp(
    model, rank, world_size, sharding_strategy=ShardingStrategy.FULL_SHARD
) -> torch.Tensor:
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
        amp_type: str = "float16",
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
        ### [TIME_BUDGET] new knobs
        time_budget_s: Optional[float] = None,   # total budget in seconds (includes one-time data gen cost)
        gen_time_s: float = 0.0,                 # seconds per generated trajectory/sample (one-time)
        Ndata_generated: Optional[int] = None,   # fixed up-front #trajectories/samples generated
    ):
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
        self.start_epoch = start_epoch
        self.start_val_loss = start_val_loss
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

        if distribution_type.upper() in ["LOCAL", "DDP"]:
            self.grad_scaler = torch.GradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )
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

        if self.device_mesh is not None and "fsdp" in self.device_mesh.mesh_dim_names:
            self.sync_group = self.device_mesh.get_group(mesh_dim="fsdp")
            self.sync_group_size = self.sync_group.size()
        else:
            self.sync_group = None
            self.sync_group_size = 1

        self.sync_group_rank = self.rank // self.sync_group_size
        self.rank_in_sync_group = self.rank % self.sync_group_size
        self.num_sync_groups = self.world_size // self.sync_group_size

        if self.sampling_rank_strategy == "gpu":
            self.sampling_rank = self.rank
        elif self.sampling_rank_strategy == "node":
            self.sampling_rank = self.sync_group_rank
        else:
            self.sampling_rank = 0

        self.dset_metadata = self.datamodule.train_dataset.dset_to_metadata
        self.revin = revin(self.datamodule.train_dataset, self.device)
        self.model_epsilon = epsilon
        self.validation_epsilon = validation_epsilon

        self.formatter_dict = {}
        for dset_name, metadata in self.dset_metadata.items():
            self.formatter_dict[metadata.dataset_name] = formatter()

        ### [TIME_BUDGET] compute effective compute-only budget (discount one-time gen cost)
        self.time_budget_s = None if time_budget_s is None else float(time_budget_s)
        self.gen_time_s = float(gen_time_s)

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

    def save_model_if_necessary(
        self, epoch: int, validation_loss: float, last: bool = False
    ) -> Optional[Future]:
        checkpoint_future = self.checkpointer.save_if_necessary(
            self.model,
            self.optimizer,
            validation_loss,
            epoch,
            force=last,
            local=self.distribution_type.upper() in ["LOCAL", "DDP"],
        )
        return checkpoint_future

    # --- NOTE: rollout_model / validation_loop unchanged ---

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

        if epoch < self.big_batch_before:
            grad_acc_steps = self.grad_acc_steps * self.big_batch_multiplier
        else:
            grad_acc_steps = self.grad_acc_steps

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

                    y_pred, y_ref = self.rollout_model(
                        self.model, batch, self.formatter_dict[dset_name]
                    )
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

    # --- NOTE: validate_if_necessary unchanged ---

    def train(self):
        checkpoint_future = None
        val_loss = self.start_val_loss
        train_dataloader = self.datamodule.train_dataloader(self.sampling_rank)

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)

            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")
            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(f"Epoch {epoch}/{self.max_epoch}: training loss {train_loss:.4f}")
            train_logs |= {"train": train_loss, "epoch": epoch}
            if self.wandb_logging and self.rank == 0:
                wandb.log(train_logs)

            ### [TIME_BUDGET] if we hit budget, stop training entirely (no extra epochs)
            if (
                self.effective_train_budget_s is not None
                and self.train_compute_time_s >= self.effective_train_budget_s
            ):
                logger.info(
                    "[TIME_BUDGET] Training loop exiting early at epoch %d due to budget.",
                    epoch,
                )
                # Optionally save a final checkpoint as "last"
                if not self.skip_checkpointing:
                    if checkpoint_future is not None:
                        checkpoint_future.result()
                    checkpoint_future = self.save_model_if_necessary(
                        epoch, val_loss, last=True
                    )
                    if checkpoint_future is not None:
                        checkpoint_future.result()
                break

            torch.cuda.empty_cache()
            gc.collect()

            val_dataloders = self.datamodule.val_dataloaders(
                replicas=self.sync_group_size,
                rank=self.rank_in_sync_group,
                full=(epoch >= self.max_epoch and not self.debug_mode),
            )
            rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders(
                replicas=self.sync_group_size,
                rank=self.rank_in_sync_group,
                full=(epoch >= self.max_epoch and not self.debug_mode),
            )
            maybe_val_loss, rollout_loss = self.validate_if_necessary(
                epoch, val_dataloders, rollout_val_dataloaders
            )
            val_loss = maybe_val_loss if maybe_val_loss is not None else val_loss

            if checkpoint_future is not None:
                logger.debug(
                    f"Wait for previous checkpointing {checkpoint_future} to complete."
                )
                checkpoint_future.result()

            if not self.skip_checkpointing:
                checkpoint_future = self.save_model_if_necessary(
                    epoch, val_loss, last=(epoch == self.max_epoch)
                )

        # NOTE: you can choose whether to run test after early-stop.
        # Keeping original behavior here:
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
        self.validate_if_necessary(
            self.max_epoch + 1, val_dataloders, rollout_val_dataloaders
        )
        self.validate_if_necessary(
            self.max_epoch + 1,
            test_dataloaders,
            rollout_test_dataloaders,
            valid_or_test="test",
        )
