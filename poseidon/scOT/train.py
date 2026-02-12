"""
This script trains a scOT or pretrains Poseidon on a PDE dataset.
Can be also used for finetuning Poseidon.
Can be used in a single config or sweep setup.

Example usage:

accelerate launch scOT/train.py
    --config configs/gray_scott.yaml \
    --wandb_run_name test \
    --wandb_project_name test \
    --checkpoint_path check \
    --data_path data \
    --ndata=200 \
    --time_budget=1000
"""

import argparse
import torch
import wandb
import numpy as np
import random
import json
import psutil
import os
import gc
import time

import matplotlib.pyplot as plt
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
import matplotlib.pyplot as plt
import transformers
from accelerate.utils import broadcast_object_list
from scOT.trainer import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from scOT.model import ScOT, ScOTConfig
from mpl_toolkits.axes_grid1 import ImageGrid
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.utils import get_num_parameters, read_cli, get_num_parameters_no_embed, LpLoss
from scOT.metrics import relative_lp_error
from scOT.model_save import SaveEveryNEpochs

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}


def create_predictions_plot(predictions, labels, wandb_prefix):
    assert predictions.shape[0] >= 4

    indices = random.sample(range(predictions.shape[0]), 4)

    predictions = predictions[indices]
    labels = labels[indices]

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
    )

    vmax, vmin = max(predictions.max(), labels.max()), min(
        predictions.min(), labels.min()
    )

    for _i, ax in enumerate(grid):
        i = _i // 4
        j = _i % 4

        if i % 2 == 0:
            ax.imshow(
                predictions[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.imshow(
                labels[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
    plt.close()

def _smooth_1d(counts: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    """
    Gaussian smoothing in 'bin index' space (no scipy needed).
    sigma_bins ~ 1-3 usually looks good.
    """
    counts = counts.astype(np.float64)
    if sigma_bins <= 0:
        return counts

    radius = int(np.ceil(3 * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(counts, k, mode="same")

def plot_smoothed_hist_curve(
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    title: str,
    xlabel: str,
    log_x: bool = False,
    sigma_bins: float = 2.0,
):
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = _smooth_1d(counts, sigma_bins=sigma_bins)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(centers, smooth, linewidth=2.0)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count (smoothed)")
    ax.set_title(title)
    return fig

def per_traj_relative_l2(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12):
    """
    preds, labels: [N, ...]
    Returns:
    rel_l2: [N] tensor
    finite_mask: [N] bool tensor
    """
    assert preds.shape[0] == labels.shape[0]
    N = preds.shape[0]
    preds_f = preds.reshape(N, -1)
    labels_f = labels.reshape(N, -1)

    diff = preds_f - labels_f
    abs_l2 = torch.linalg.vector_norm(diff, ord=2, dim=1)
    denom = torch.linalg.vector_norm(labels_f, ord=2, dim=1).clamp_min(eps)
    rel_l2 = abs_l2 / denom

    finite_mask = torch.isfinite(rel_l2)
    return rel_l2, finite_mask

def plot_hist_with_smooth_overlay_counts(
    edges: np.ndarray,
    counts: np.ndarray,
    samples: np.ndarray,
    *,
    title: str,
    xlabel: str,
    log_x: bool = False,
    sigma_bins: float = 2.0,
):
    """
    One figure: histogram bars + smoothed-count line overlay (same y-axis = count).
    edges/counts should come from np.histogram(samples, ...)
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = _smooth_1d(counts, sigma_bins=sigma_bins)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # bars: use the SAME edges so the overlay matches exactly
    ax.hist(samples, bins=edges, alpha=0.8)

    # line: smoothed counts
    ax.plot(centers, smooth, linewidth=2.0)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    return fig

def save_loss_distribution(
    rel: np.ndarray,
    *,
    out_dir: str,
    prefix: str,
    bins: int = 50,
):
    """
    Save raw losses + reproducible histogram binning (linear and log bins).
    Returns the saved filepath.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rel = rel.astype(np.float64)

    # Linear histogram
    lin_counts, lin_edges = np.histogram(rel, bins=bins)

    fig_lin_smooth = plot_smoothed_hist_curve(
        lin_edges, lin_counts,
        title="Test loss distribution (smoothed, linear scale)",
        xlabel="Per-trajectory relative L2 loss",
        log_x=False,
        sigma_bins=2.0,   # tweak this
    )
    wandb.log({f"{prefix}/loss_dist_linear_smooth": wandb.Image(fig_lin_smooth)})
    plt.close(fig_lin_smooth)

    lin_counts, lin_edges = np.histogram(rel, bins=bins)
    fig_lin_overlay = plot_hist_with_smooth_overlay_counts(
        lin_edges, lin_counts, rel,
        title="Test loss distribution (hist + smooth overlay, linear)",
        xlabel="Per-trajectory relative L2 loss",
        log_x=False,
        sigma_bins=2.0,
    )
    wandb.log({f"{prefix}/loss_dist_linear_overlay": wandb.Image(fig_lin_overlay)})
    plt.close(fig_lin_overlay)

    # Log-x histogram (only for positive values)
    rel_pos = rel[rel > 0]
    if rel_pos.size > 0:
        lo, hi = rel_pos.min(), rel_pos.max()
        # avoid degenerate logspace
        if hi <= lo:
            log_edges = np.array([lo, lo * 1.000001], dtype=np.float64)
        else:
            log_edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        log_counts, _ = np.histogram(rel_pos, bins=log_edges)
        fig_log_smooth = plot_smoothed_hist_curve(
            log_edges, log_counts,
            title="Test loss distribution (smoothed, log scale)",
            xlabel="Per-trajectory relative L2 loss (log scale)",
            log_x=True,
            sigma_bins=2.0,  # tweak this
        )
        wandb.log({f"{prefix}/loss_dist_logx_smooth": wandb.Image(fig_log_smooth)})
        plt.close(fig_log_smooth)

        log_counts, _ = np.histogram(rel_pos, bins=log_edges)

        fig_log_overlay = plot_hist_with_smooth_overlay_counts(
            log_edges, log_counts, rel_pos,
            title="Test loss distribution (hist + smooth overlay, log x)",
            xlabel="Per-trajectory relative L2 loss",
            log_x=True,
            sigma_bins=2.0,
        )
        wandb.log({f"{prefix}/loss_dist_logx_overlay": wandb.Image(fig_log_overlay)})
        plt.close(fig_log_overlay)

    else:
        log_edges = np.array([], dtype=np.float64)
        log_counts = np.array([], dtype=np.int64)

    payload = dict(
        losses=rel,
        bins=bins,
        lin_edges=lin_edges,
        lin_counts=lin_counts,
        log_edges=log_edges,
        log_counts=log_counts,
    )

    fname = out_dir / f"{prefix}_loss_distribution_bins{bins}.npz"
    np.savez_compressed(fname, **payload)
    return str(fname)

def loss_distributions(
    rel_l2: torch.Tensor,
    finite_mask: torch.Tensor,
    *,
    wandb_prefix: str,
    out_dir: str,
    bins: int = 50,
    log_as_artifact: bool = True,
):
    """
    Creates two plots (linear x + log x) and saves reproducible data to npz.
    """
    rel = rel_l2[finite_mask].detach().float().cpu().numpy()
    if rel.size == 0:
        print(f"[WARN] {wandb_prefix}: no finite losses to plot.")
        return

    npz_path = save_loss_distribution(rel, out_dir=out_dir, prefix=wandb_prefix, bins=bins)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.hist(rel, bins=bins)
    ax1.set_xlabel("Per-trajectory relative L2 loss")
    ax1.set_ylabel("Count")
    ax1.set_title("Test loss distribution (linear scale)")
    wandb.log({
        f"{wandb_prefix}/loss_dist_linear": wandb.Image(fig1),
        f"{wandb_prefix}/loss_hist_linear": wandb.Histogram(rel),
    })
    plt.close(fig1)

    # ---- Plot 2: log-x (use only positive values)
    rel_pos = rel[rel > 0]
    if rel_pos.size > 0:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        lo, hi = rel_pos.min(), rel_pos.max()
        if hi > lo:
            edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
            ax2.hist(rel_pos, bins=edges)
        else:
            ax2.hist(rel_pos, bins=bins)
        ax2.set_xscale("log")
        ax2.set_xlabel("Per-trajectory relative L2 loss (log x)")
        ax2.set_ylabel("Count")
        ax2.set_title("Test loss distribution (log scale)")
        wandb.log({
            f"{wandb_prefix}/loss_dist_logx": wandb.Image(fig2),
            f"{wandb_prefix}/loss_hist_logx": wandb.Histogram(rel_pos),
        })
        plt.close(fig2)

    # Quantiles + record where data is saved
    wandb.log({
        f"{wandb_prefix}/loss_p50": float(np.percentile(rel, 50)),
        f"{wandb_prefix}/loss_p90": float(np.percentile(rel, 90)),
        f"{wandb_prefix}/loss_p95": float(np.percentile(rel, 95)),
        f"{wandb_prefix}/loss_p99": float(np.percentile(rel, 99)),
        f"{wandb_prefix}/loss_max": float(np.max(rel)),
        f"{wandb_prefix}/loss_npz_path": npz_path,
    })

    # Optional: log the npz as a W&B artifact so it's stored with the run
    if log_as_artifact:
        art = wandb.Artifact(name=f"{wandb_prefix}_loss_distribution", type="evaluation")
        art.add_file(npz_path)
        wandb.log_artifact(art)

def worst_traj_relative_l2(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12):
    """
    preds, labels: [N, ...]
    Per-trajectory relative L2: ||pred-label||_2 / (||label||_2 + eps)
    Returns: worst_idx, worst_rel_l2, worst_abs_l2_norm, mean_rel_l2
    """
    assert preds.shape[0] == labels.shape[0]
    N = preds.shape[0]

    preds_f = preds.reshape(N, -1)
    labels_f = labels.reshape(N, -1)

    diff = preds_f - labels_f
    abs_l2 = torch.linalg.vector_norm(diff, ord=2, dim=1)  # [N]
    denom = torch.linalg.vector_norm(labels_f, ord=2, dim=1).clamp_min(eps)
    rel_l2 = abs_l2 / denom  # [N]

    finite = torch.isfinite(rel_l2)
    if not torch.all(finite):
        rel_l2 = torch.where(
            finite, rel_l2, torch.tensor(float("-inf"), device=rel_l2.device)
        )

    worst_idx = torch.argmax(rel_l2).item()
    worst_rel = rel_l2[worst_idx].item()
    worst_abs = abs_l2[worst_idx].item()
    mean_rel = rel_l2[finite].mean().item() if torch.any(finite) else float("nan")
    return worst_idx, worst_rel, worst_abs, mean_rel

def compute_test_worst_and_log(
    *,
    predictions_obj,
    prefix: str,
    config: dict,
    RANK: int,
    ckpt_dir: str,
    do_plot: bool = True,
):
    """
    Given Trainer.predict(...) output, compute:
      - global relative L2 (LpLoss)
      - per-traj mean relative L2
      - worst traj relative L2 + index
    Log to wandb under `prefix/...` and optionally plot.
    """
    if not (RANK == 0 or RANK == -1):
        return

    preds_np  = predictions_obj.predictions
    labels_np = predictions_obj.label_ids

    if preds_np is None or labels_np is None:
        print(f"[WARN] {prefix}: predictions or labels missing; skipping worst-traj report.")
        return

    preds  = torch.from_numpy(preds_np).float()
    labels = torch.from_numpy(labels_np).float()

    rel_l2_vec, finite_mask = per_traj_relative_l2(preds, labels)
    loss_distributions(
        rel_l2_vec,
        finite_mask,
        wandb_prefix=prefix,
        out_dir=ckpt_dir,
        bins=50,
        log_as_artifact=True,
    )

    rel_full = rel_l2_vec.detach().float().cpu().numpy()
    rel_full[~finite_mask.detach().cpu().numpy()] = np.nan

    traj_id = np.arange(rel_full.shape[0], dtype=np.int32)

    scatter_neural_path = Path(ckpt_dir) / f"scatter_neural_{prefix}_evalS{config.get('eval_resolution', 64)}.npz"
    np.savez_compressed(
        scatter_neural_path,
        traj_id=traj_id,
        err_neural=rel_full.astype(np.float32),
        metric_name="rel_l2_per_traj",
        eval_S=np.int32(config.get("eval_resolution", 64)),
        worst_idx=np.int32(int(np.nanargmax(rel_full))),
        worst_val=np.float32(float(np.nanmax(rel_full))),
    )

    art = wandb.Artifact(name=f"scatter_neural_{prefix}", type="scatter_data")
    art.add_file(str(scatter_neural_path))
    wandb.log_artifact(art)

    # Global relative L2 (your existing convention)
    lp_loss = LpLoss(size_average=True)
    global_rel_l2 = lp_loss(preds, labels).item()

    # Per-trajectory mean + worst
    worst_idx, worst_rel_l2, worst_abs_l2, mean_rel_l2 = worst_traj_relative_l2(preds, labels)

    print(f"[{prefix.upper()}] preds shape: {tuple(preds.shape)}")
    print(
        f"[{prefix.upper()}] global_rel_l2={global_rel_l2:.6g} | "
        f"mean_per_traj_rel_l2={mean_rel_l2:.6g} | "
        f"worst_rel_l2={worst_rel_l2:.6g} (idx={worst_idx}) | "
        f"worst_abs_l2_norm={worst_abs_l2:.6g}"
    )

    metrics = {}
    for key, value in predictions_obj.metrics.items():
        metrics[f"{prefix}/" + key[1:]] = value

    metrics[f"{prefix}/relative_l2_loss"] = global_rel_l2
    metrics[f"{prefix}/mean_per_traj_relative_l2_loss"] = mean_rel_l2
    metrics[f"{prefix}/worst_relative_l2_loss"] = worst_rel_l2
    # metrics[f"{prefix}/worst_abs_l2_norm"] = worst_abs_l2
    metrics[f"{prefix}/worst_traj_idx"] = worst_idx

    wandb.log(metrics)

    if do_plot:
        tps = time.time()
        create_predictions_plot(preds_np, labels_np, wandb_prefix=prefix)
        tpe = time.time()
        print(f"[{prefix.upper()}] plot time: {tpe - tps:.3f}s")

def setup(params, model_map=True):
    config = None
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    CPU_CORES = len(psutil.Process().cpu_affinity())
    CPU_CORES = min(CPU_CORES, 16)
    print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
    if params.json_config:
        config = json.loads(params.config)
    else:
        config = params.config

    run_name = f"N{int(params.ndata)}_B{int(params.time_budget)}"
    if params.batch_size is not None:
        run_name += f"_batch{int(params.batch_size)}"

    if params.lr is not None:
        run_name += f"_lr{float(params.lr)}"

    group_name = f"B{int(params.time_budget)}"

    if RANK == 0 or RANK == -1:
        run = wandb.init(
            project=params.wandb_project_name, name=run_name, group=group_name, config=config
        )
        config = wandb.config
    else:

        def clean_yaml(config):
            d = {}
            for key, inner_dict in config.items():
                d[key] = inner_dict["value"]
            return d

        if not params.json_config:
            with open(params.config, "r") as s:
                config = yaml.safe_load(s)
            config = clean_yaml(config)
        run = None

    ckpt_dir = "./"
    if RANK == 0 or RANK == -1:
        if run.sweep_id is not None:
            ckpt_dir = (
                params.checkpoint_path
                + "/"
                + run.project
                + "/"
                # + run.sweep_id
                # + "/"
                + run.name
            )
        else:
            ckpt_dir = params.checkpoint_path + "/" + run.project + "/" + run.name
    if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ls = broadcast_object_list([ckpt_dir], from_process=0)
    ckpt_dir = ls[0]

    print(f'Model: Poseidon_{config["model_name"]}')

    if model_map and (
        type(config["model_name"]) == str and config["model_name"] in MODEL_MAP.keys()
    ):
        config = {**config, **MODEL_MAP[config["model_name"]]}
        if RANK == 0 or RANK == -1:
            wandb.config.update(MODEL_MAP[config["model_name"]], allow_val_change=True)

    if params.batch_size is not None:
        config["batch_size"] = int(params.batch_size)

    if params.lr is not None:
        config["lr"] = float(params.lr)

    return run, config, ckpt_dir, RANK, CPU_CORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scOT or pretrain Poseidon.")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Set this to a str pointing to a HF Hub model checkpoint or a directory with a scOT checkpoint if you want to finetune.",
    )
    parser.add_argument(
        "--replace_embedding_recovery",
        action="store_true",
        help="Set this if you have to replace the embeddings and recovery layers because you are not just using the density, velocity and pressure channels. Only relevant for finetuning.",
    )
    params = read_cli(parser).parse_args()
    run, config, ckpt_dir, RANK, CPU_CORES = setup(params)

    print("Configuration:", config)
    # print(f"lr: {params.lr}")

    try:
        ckpt = get_last_checkpoint(ckpt_dir)
        if ckpt is not None:
            params.resume_training = True
    except:
        pass

    def trained_model_exists(ckpt_dir):
        return os.path.isfile(os.path.join(ckpt_dir, "pytorch_model.bin"))

    if not params.do_inference and trained_model_exists(ckpt_dir) and not params.resume_training:
        if RANK == 0 or RANK == -1:
            print(f"[INFO] Trained model already exists at {ckpt_dir}. Skipping training.")
            if run is not None:
                wandb.log({"status": "skipped_existing_model"})
                wandb.finish()
        exit(0)

    if not params.resume_training:
        print("No checkpoint found, starting training from scratch.")
    else:
        print(f"Resuming training from checkpoint: {ckpt}")

    train_eval_set_kwargs = (
        {"just_velocities": True}
        if ("incompressible" in config["dataset"]) and params.just_velocities
        else {}
    )
    if params.move_data is not None:
        train_eval_set_kwargs["move_to_local_scratch"] = params.move_data
    if params.max_num_train_time_steps is not None:
        train_eval_set_kwargs["max_num_time_steps"] = params.max_num_train_time_steps
    if params.train_time_step_size is not None:
        train_eval_set_kwargs["time_step_size"] = params.train_time_step_size
    if params.train_small_time_transition:
        train_eval_set_kwargs["allowed_time_transitions"] = [1]
    
    time_budget = params.time_budget
    data_time = params.ndata * (config["per_data_generation_time"] + config["per_data_downsample_time"])
    train_time_budget = time_budget - data_time
    if train_time_budget <= 0:
        print("Training time budget less than or equal to 0 seconds, exiting.")
        exit(0)

    print("Total time budget (s):", time_budget)
    print("Data generation time (s):", data_time)
    print("Training time budget (s):", train_time_budget)

    eta_total_steps = (train_time_budget / config["eta_step_time"])
    warmup_steps = 0.1 * eta_total_steps

    train_dataset = get_dataset(
        dataset=config["dataset"],
        which="train",
        num_trajectories=params.ndata,
        max_traj_per_file=config["max_traj_per_file"],
        file_count=config["file_count"],
        file_prefix=config["file_prefix"],
        N_val=config["eval_ndata"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    # eval_dataset = get_dataset(
    #     dataset=config["dataset"],
    #     which="val",
    #     num_trajectories=config["eval_ndata"],
    #     max_traj_per_file=config["max_traj_per_file"],
    #     file_count=config["file_count"],
    #     file_prefix=config["file_prefix"],
    #     data_path=params.data_path,
    #     **train_eval_set_kwargs,
    # )

    config["effective_train_set_size"] = len(train_dataset)
    time_involved = isinstance(train_dataset, BaseTimeDataset) or (
        isinstance(train_dataset, torch.utils.data.ConcatDataset)
        and isinstance(train_dataset.datasets[0], BaseTimeDataset)
    )

    if not isinstance(train_dataset, torch.utils.data.ConcatDataset):
        resolution = train_dataset.resolution
        input_dim = train_dataset.input_dim
        output_dim = train_dataset.output_dim
        channel_slice_list = train_dataset.channel_slice_list
        printable_channel_description = train_dataset.printable_channel_description
    else:
        resolution = train_dataset.datasets[0].resolution
        input_dim = train_dataset.datasets[0].input_dim
        output_dim = train_dataset.datasets[0].output_dim
        channel_slice_list = train_dataset.datasets[0].channel_slice_list
        printable_channel_description = train_dataset.datasets[
            0
        ].printable_channel_description

    model_config = (
        ScOTConfig(
            image_size=resolution,
            patch_size=config["patch_size"],
            num_channels=input_dim,
            num_out_channels=output_dim,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            skip_connections=config["skip_connections"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=channel_slice_list,
            residual_model="convnext",
            use_conditioning=time_involved,
            learn_residual=False,
        )
        if params.finetune_from is None or params.replace_embedding_recovery
        else None
    )

    if params.finetune_from is not None:
        model = ScOT.from_pretrained(
            params.finetune_from, config=model_config, ignore_mismatched_sizes=True
        )
    else:
        model = ScOT(model_config)
    num_params = get_num_parameters(model)
    config["num_params"] = num_params
    num_params_no_embed = get_num_parameters_no_embed(model)
    config["num_params_wout_embed"] = num_params_no_embed
    if RANK == 0 or RANK == -1:
        print(f"Model size: {num_params}")
        print(f"Model size without embeddings: {num_params_no_embed}")

    def compute_metrics(eval_preds):
        channel_list = channel_slice_list

        def get_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_relative_l1_error": median_error,
                "mean_relative_l1_error": mean_error,
                "std_relative_l1_error": std_error,
                "min_relative_l1_error": min_error,
                "max_relative_l1_error": max_error,
            }

        error_statistics = [
            get_statistics(
                relative_lp_error(
                    eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                    p=1,
                    return_percent=True,
                )
            )
            for i in range(len(channel_list) - 1)
        ]

        if output_dim == 1:
            error_statistics = error_statistics[0]
            return error_statistics
        else:
            mean_over_means = np.mean(
                np.array(
                    [stats["mean_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            mean_over_medians = np.mean(
                np.array(
                    [stats["median_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            error_statistics_ = {
                "mean_relative_l1_error": mean_over_means,
                "mean_over_median_relative_l1_error": mean_over_medians,
            }
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[printable_channel_description[i] + "/" + key] = (
                        value
                    )
            return error_statistics_

    def get_trainer(model, config, params, compute_metrics, warmup_steps, train_time_budget, train_dataset):
        train_config = TrainingArguments(
            output_dir=ckpt_dir,
            overwrite_output_dir=True,
            # evaluation_strategy="epoch",
            evaluation_strategy="no",
            per_device_train_batch_size=min(100, config["batch_size"]),
            gradient_accumulation_steps=max(config["batch_size"] // 100, 1),
            per_device_eval_batch_size=config["batch_size"],
            eval_accumulation_steps=16,
            max_grad_norm=config["max_grad_norm"],
            num_train_epochs=config["num_epochs"],
            # max_steps=eta_total_steps,
            optim="adamw_torch",
            learning_rate=config["lr"],
            learning_rate_embedding_recovery=(
                None
                if (params.finetune_from is None or "lr_embedding_recovery" not in config)
                else config["lr_embedding_recovery"]
            ),
            learning_rate_time_embedding=(
                None
                if (params.finetune_from is None or "lr_time_embedding" not in config)
                else config["lr_time_embedding"]
            ),
            weight_decay=config["weight_decay"],
            adam_beta1=0.9,  # default
            adam_beta2=0.999,  # default
            adam_epsilon=1e-8,  # default
            lr_scheduler_type=config["lr_scheduler"],
            warmup_steps=warmup_steps,
            # warmup_ratio=config["warmup_ratio"],
            log_level="passive",
            logging_strategy="epoch",
            logging_steps=10,
            logging_nan_inf_filter=False,
            save_strategy="no",
            save_total_limit=1,
            seed=SEED,
            fp16=False,
            # dataloader_num_workers=CPU_CORES,
            dataloader_num_workers=0,
            # load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            # dataloader_pin_memory=True,
            dataloader_pin_memory=False,
            auto_find_batch_size=False,
            full_determinism=False,
            torch_compile=False,
            report_to="wandb",
            run_name=params.wandb_run_name,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=model,
            args=train_config,
            train_dataset=train_dataset,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            time_budget=train_time_budget,
            callbacks=[SaveEveryNEpochs(50)]
        )

        return trainer, train_config

    trainer, train_config = get_trainer(model, config, params, compute_metrics, warmup_steps, train_time_budget, train_dataset)

    eta_step_time = trainer.warmup(n_steps=10, discard_first=3)

    print(f"Warmed up.")
    print(f"Eta step time: {eta_step_time}")

    # get new trainer and with the new eta of step time
    gradient_accumulation_steps = max(config["batch_size"] // 100, 1)
    eta_step_time = eta_step_time * gradient_accumulation_steps

    eta_total_steps = (train_time_budget / eta_step_time)
    warmup_steps = 0.1 * eta_total_steps

    print(f"Total training steps estimated: {eta_total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    trainer, train_config = get_trainer(model, config, params, compute_metrics, warmup_steps, train_time_budget, train_dataset)

    if not params.do_inference:
        ts = time.time()
        try:
            if params.resume_training:
                trainer.train(resume_from_checkpoint=ckpt)
            else:
                trainer.train()
        except Exception as e:
            print(e)
            print("Training finished.")

        te = time.time()
        trainer.save_model(train_config.output_dir)

        if (RANK == 0 or RANK == -1) and params.push_to_hf_hub is not None:
            model.push_to_hub(params.push_to_hf_hub)
    
    else:
        print("Only inference: Skipping training.")

    del train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    do_test = (
        True
        if params.max_num_train_time_steps is None
        and params.train_time_step_size is None
        and not params.train_small_time_transition
        and not ".time" in config["dataset"]
        else False
    )
    if do_test:
        # quantization
        # model.quantize_linear(mode="8bit") 

        print("Testing...")
        test_set_kwargs = (
            {"just_velocities": True}
            if ("incompressible" in config["dataset"]) and params.just_velocities
            else {}
        )
        out_test_set_kwargs = (
            {"just_velocities": True}
            if ("incompressible" in config["dataset"]) and params.just_velocities
            else {}
        )
        if params.move_data is not None:
            test_set_kwargs["move_to_local_scratch"] = params.move_data
            out_test_set_kwargs["move_to_local_scratch"] = params.move_data
        if time_involved:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 14,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 20,
                "allowed_time_transitions": [1],
            }
        if "RayleighTaylor" in config["dataset"]:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 7,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 10,
                "allowed_time_transitions": [1],
            }

        test_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=config["eval_ndata"],
            max_traj_per_file=config["max_traj_per_file"],
            file_count=config["file_count"],
            file_prefix=config["file_prefix"],
            N_val=config["eval_ndata"],
            data_path=params.data_path,
            **test_set_kwargs,
        )
        try:
            out_dist_test_dataset = get_dataset(
                dataset=config["dataset"] + ".out",
                which="test",
                num_trajectories=config["num_trajectories"],
                data_path=params.data_path,
                **out_test_set_kwargs,
            )
        except:
            out_dist_test_dataset = None

        infer_ts = time.time()
        predictions = trainer.predict(test_dataset, metric_key_prefix="")
        infer_te = time.time()

        print(f'\nTotal inference time (with computing loss): {(infer_te - infer_ts) / config["eval_ndata"]}')

        preds = torch.from_numpy(predictions.predictions).float()
        labels = torch.from_numpy(predictions.label_ids).float()

        lp_loss = LpLoss(size_average=True)
        l2_rel = lp_loss(preds, labels).item()

        compute_test_worst_and_log(
            predictions_obj=predictions,
            prefix="test",
            config=config,
            RANK=RANK,
            ckpt_dir=train_config.output_dir,
            do_plot=True,
        )

        print("predictions shape:", preds.shape)

        if RANK == 0 or RANK == -1:
            metrics = {}
            for key, value in predictions.metrics.items():
                metrics["test/" + key[1:]] = value
            metrics["test/relative_l2_loss"] = l2_rel
            wandb.log(metrics)

            tps = time.time()
            create_predictions_plot(
                predictions.predictions,
                predictions.label_ids,
                wandb_prefix="test",
            )
            tpe = time.time()
            print("plot time:", tpe-tps)

        # evaluate on out-of-distribution test set
        if out_dist_test_dataset is not None:
            predictions = trainer.predict(out_dist_test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test_out_dist/" + key[1:]] = value
                metrics["test_out_dist/relative_l2_loss"] = l2_rel
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test_out_dist",
                )

        if time_involved and (test_set_kwargs["time_step_size"] // 2 > 0):
            trainer.set_ar_steps(test_set_kwargs["time_step_size"] // 2)
            predictions = trainer.predict(test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test/ar/" + key[1:]] = value
                metrics["test/ar/relative_l2_loss"] = l2_rel
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test/ar",
                )

            # evaluate on out-of-distribution test set
            if out_dist_test_dataset is not None:
                trainer.set_ar_steps(out_test_set_kwargs["time_step_size"] // 2)
                predictions = trainer.predict(
                    out_dist_test_dataset, metric_key_prefix=""
                )
                if RANK == 0 or RANK == -1:
                    metrics = {}
                    for key, value in predictions.metrics.items():
                        metrics["test_out_dist/ar/" + key[1:]] = value
                    metrics["test_out_dist/ar/relative_l2_loss"] = l2_rel
                    wandb.log(metrics)
                    create_predictions_plot(
                        predictions.predictions,
                        predictions.label_ids,
                        wandb_prefix="test_out_dist/ar",
                    )

    tee = time.time()

    # print("train time:", te-ts)
    print("test time:", tee-te)