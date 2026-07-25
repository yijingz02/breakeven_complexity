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
import base64
import io
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
from transformers.trainer_utils import get_last_checkpoint
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


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = np.square(ref).sum()
    if denom == 0:
        denom = 1e-10
    return float(np.sqrt(np.square(pred - ref).sum() / denom))


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


def _create_predictions_figure(predictions, labels, indices=None, value_range=None):
    if predictions.ndim == 5:
        predictions = predictions[:, 0]
    if labels.ndim == 5:
        labels = labels[:, 0]

    if indices is None:
        indices = list(range(min(4, predictions.shape[0])))
    else:
        indices = list(indices)[:4]

    predictions = predictions[indices]
    labels = labels[indices]

    fig = plt.figure(figsize=(7.0, 6.0))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1] * 2, 4), axes_pad=0.1
    )

    if value_range is None:
        vmax, vmin = max(predictions.max(), labels.max()), min(
            predictions.min(), labels.min()
        )
    else:
        vmin, vmax = value_range

    for _i, ax in enumerate(grid):
        i = _i // 4
        j = _i % 4

        if i < predictions.shape[1]:
            ax.imshow(
                predictions[j, i, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.imshow(
                labels[j, i - predictions.shape[1], :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(0.02, 0.75, "Prediction", rotation=90, va="center", ha="left", fontsize=10)
    fig.text(0.02, 0.25, "Ground Truth", rotation=90, va="center", ha="left", fontsize=10)

    return fig


def create_predictions_plot(predictions, labels, wandb_prefix):
    fig = _create_predictions_figure(predictions, labels)
    wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
    plt.close(fig)


def _figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=105)
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_step_slider_html(step_data_uris, step_indices, title: str) -> str:
    images_json = json.dumps(step_data_uris)
    steps_json = json.dumps([int(s) for s in step_indices])
    first_step = int(step_indices[0]) if len(step_indices) > 0 else 0
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; }}
        .wrap {{ max-width: 980px; margin: 0 auto; padding: 8px 12px; }}
        .title {{ font-size: 18px; margin-bottom: 8px; text-align: center; }}
        .meta {{ display: flex; justify-content: space-between; align-items: center; margin: 6px 0 10px 0; }}
        .step-label {{ font-size: 16px; }}
        img {{ width: min(880px, 100%); height: auto; display: block; margin: 0 auto; border: 0; }}
        input[type=range] {{ width: 100%; }}
    </style>
</head>
<body>
    <div class=\"wrap\">
        <div class=\"title\">{title}</div>
        <img id=\"rollout-image\" src=\"\" alt=\"rollout step\" />
        <div class=\"meta\">
            <div class=\"step-label\">Step <span id=\"step-value\">{first_step}</span></div>
        </div>
        <input id=\"step-slider\" type=\"range\" min=\"0\" max=\"{max(len(step_data_uris) - 1, 0)}\" value=\"0\" step=\"1\" />
    </div>
    <script>
        const images = {images_json};
        const steps = {steps_json};
        const slider = document.getElementById('step-slider');
        const img = document.getElementById('rollout-image');
        const stepValue = document.getElementById('step-value');

        function setStep(idx) {{
            const clamped = Math.max(0, Math.min(idx, images.length - 1));
            img.src = images[clamped];
            stepValue.textContent = steps[clamped];
            slider.value = clamped;
        }}

        slider.addEventListener('input', (event) => {{
            setStep(parseInt(event.target.value, 10));
        }});

        if (images.length > 0) {{
            setStep(0);
        }}
    </script>
</body>
</html>
"""

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


def _multi_obstacle_flip(field: np.ndarray, quantity: str) -> np.ndarray:
    flipped = np.flip(np.array(field, copy=True), axis=-1)
    if quantity == "velocity_x":
        flipped *= -1.0
    return flipped


def _denormalize_multi_obstacle_rollout(
    rollout: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    return rollout * std + mean


def _multi_obstacle_fluid_masks(dataset, num_trajectories: int) -> np.ndarray:
    """Return N,H,W masks using the dataset's configured mask convention."""
    raw = np.asarray(dataset._mask)
    raw = np.squeeze(raw)
    if raw.ndim == 2:
        raw = np.repeat(raw[None], num_trajectories, axis=0)
    if raw.ndim != 3 or raw.shape[0] < num_trajectories:
        raise ValueError(
            f"Expected at least {num_trajectories} masks shaped N,H,W, got {raw.shape}"
        )
    raw = raw[:num_trajectories] > 0.5
    if bool(getattr(dataset, "mask_is_obstacle_true", True)):
        return (~raw).astype(np.float64)
    return raw.astype(np.float64)


def _channelwise_physical_nrmse_multi_obstacle(
    predictions: np.ndarray,
    references: np.ndarray,
    fluid_masks: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return N,3 rollout nRMSE in physical units for vx, vy, and pressure."""
    pred = np.asarray(predictions[:, :, :3], dtype=np.float64).copy()
    ref = np.asarray(references[:, :, :3], dtype=np.float64).copy()
    weight = np.asarray(fluid_masks, dtype=np.float64)[:, None, None]

    pressure_weight = weight[:, :, 0]
    pressure_count = pressure_weight.sum(
        axis=(-2, -1), keepdims=True
    ).clip(min=1.0)
    pred[:, :, 2] -= (
        (pred[:, :, 2] * pressure_weight).sum(
            axis=(-2, -1), keepdims=True
        )
        / pressure_count
    )
    ref[:, :, 2] -= (
        (ref[:, :, 2] * pressure_weight).sum(
            axis=(-2, -1), keepdims=True
        )
        / pressure_count
    )

    numerator = (np.square(pred - ref) * weight).sum(axis=(1, 3, 4))
    denominator = (np.square(ref) * weight).sum(axis=(1, 3, 4))
    return np.sqrt(numerator / np.maximum(denominator, eps))


def _extract_multi_obstacle_reference_rollout(dataset, rollout_steps: int) -> np.ndarray:
    history = int(getattr(dataset, "history", 5))
    frame_axis = 1 if getattr(dataset, "_layout", "NTHW") == "NTHW" else 3
    total_frames = int(dataset._vx.shape[frame_axis])
    rollout_steps = min(int(rollout_steps), total_frames - history)
    if rollout_steps <= 0:
        raise ValueError(
            f"Cannot extract a rollout with history={history} from a trajectory with {total_frames} frames."
        )

    refs = []
    for traj_id in range(int(dataset._num_traj_total)):
        _, local_idx = dataset._traj_to_file_local(int(traj_id))
        traj = []
        for t in range(history, history + rollout_steps):
            vx = dataset._get_frame(dataset._vx, local_idx, t)
            vy = dataset._get_frame(dataset._vy, local_idx, t)
            p = dataset._get_frame(dataset._p, local_idx, t)
            traj.append(torch.stack([vx, vy, p], dim=0))
        refs.append(torch.stack(traj, dim=0))

    return torch.stack(refs, dim=0).cpu().numpy()


def _compute_multi_obstacle_time_avg_stats(
    predictions: np.ndarray,
    references: np.ndarray,
    *,
    last_n: int = 160,
) -> dict:
    quantities = ["velocity_x", "velocity_y", "pressure"]

    def _flip_stacked_fields(field_stack: np.ndarray, names) -> np.ndarray:
        out = np.array(field_stack, copy=True)
        for idx, name in enumerate(names):
            out[idx] = _multi_obstacle_flip(out[idx], name)
        return out

    last_n = min(int(last_n), predictions.shape[1], references.shape[1])
    if last_n <= 0:
        raise ValueError("last_n must be positive.")

    stats = {}
    for channel_idx, quantity in enumerate(quantities):
        flip_scores = []
        noflip_scores = []
        for traj_idx in range(predictions.shape[0]):
            pred_avg = predictions[traj_idx, -last_n:, channel_idx].mean(axis=0)
            ref_avg = references[traj_idx, -last_n:, channel_idx].mean(axis=0)

            flip_scores.append(
                nrmse(
                    pred_avg + _multi_obstacle_flip(pred_avg, quantity),
                    ref_avg + _multi_obstacle_flip(ref_avg, quantity),
                )
            )
            noflip_scores.append(nrmse(pred_avg, ref_avg))

        flip_scores = np.asarray(flip_scores, dtype=np.float64)
        noflip_scores = np.asarray(noflip_scores, dtype=np.float64)
        stats[f"tavg_flip_{quantity}"] = float(np.mean(flip_scores))
        stats[f"tavg_noflip_{quantity}"] = float(np.mean(noflip_scores))
        stats[f"worst_tavg_flip_{quantity}"] = float(np.max(flip_scores))
        stats[f"worst_tavg_noflip_{quantity}"] = float(np.max(noflip_scores))
        stats[f"worst_tavg_flip_{quantity}_traj_idx"] = int(np.argmax(flip_scores))
        stats[f"worst_tavg_noflip_{quantity}_traj_idx"] = int(np.argmax(noflip_scores))

    velocity_names = ["velocity_x", "velocity_y"]
    flip_scores_velocity = []
    noflip_scores_velocity = []
    flip_scores_all = []
    noflip_scores_all = []

    for traj_idx in range(predictions.shape[0]):
        pred_avg_all = predictions[traj_idx, -last_n:, :].mean(axis=0)
        ref_avg_all = references[traj_idx, -last_n:, :].mean(axis=0)

        pred_avg_velocity = pred_avg_all[:2]
        ref_avg_velocity = ref_avg_all[:2]

        flip_scores_velocity.append(
            nrmse(
                pred_avg_velocity + _flip_stacked_fields(pred_avg_velocity, velocity_names),
                ref_avg_velocity + _flip_stacked_fields(ref_avg_velocity, velocity_names),
            )
        )
        noflip_scores_velocity.append(nrmse(pred_avg_velocity, ref_avg_velocity))

        flip_scores_all.append(
            nrmse(
                pred_avg_all + _flip_stacked_fields(pred_avg_all, quantities),
                ref_avg_all + _flip_stacked_fields(ref_avg_all, quantities),
            )
        )
        noflip_scores_all.append(nrmse(pred_avg_all, ref_avg_all))

    flip_scores_velocity = np.asarray(flip_scores_velocity, dtype=np.float64)
    noflip_scores_velocity = np.asarray(noflip_scores_velocity, dtype=np.float64)
    flip_scores_all = np.asarray(flip_scores_all, dtype=np.float64)
    noflip_scores_all = np.asarray(noflip_scores_all, dtype=np.float64)

    stats["tavg_flip_velocity"] = float(np.mean(flip_scores_velocity))
    stats["tavg_noflip_velocity"] = float(np.mean(noflip_scores_velocity))
    stats["tavg_flip_all_fields"] = float(np.mean(flip_scores_all))
    stats["tavg_noflip_all_fields"] = float(np.mean(noflip_scores_all))
    stats["worst_tavg_flip_velocity"] = float(np.max(flip_scores_velocity))
    stats["worst_tavg_noflip_velocity"] = float(np.max(noflip_scores_velocity))
    stats["worst_tavg_flip_all_fields"] = float(np.max(flip_scores_all))
    stats["worst_tavg_noflip_all_fields"] = float(np.max(noflip_scores_all))
    stats["worst_tavg_flip_velocity_traj_idx"] = int(np.argmax(flip_scores_velocity))
    stats["worst_tavg_noflip_velocity_traj_idx"] = int(np.argmax(noflip_scores_velocity))
    stats["worst_tavg_flip_all_fields_traj_idx"] = int(np.argmax(flip_scores_all))
    stats["worst_tavg_noflip_all_fields_traj_idx"] = int(np.argmax(noflip_scores_all))

    return stats


def _compute_rollout_temporal_diagnostics(
    predictions: np.ndarray,
    references: np.ndarray,
) -> dict:
    steps = min(int(predictions.shape[1]), int(references.shape[1]))
    if steps <= 1:
        return {
            "pred_mean_step_delta": 0.0,
            "ref_mean_step_delta": 0.0,
            "pred_first_last_delta": 0.0,
            "ref_first_last_delta": 0.0,
            "pred_ref_step_delta_ratio": 0.0,
        }

    preds = predictions[:, :steps]
    refs = references[:, :steps]
    pred_step_delta = float(np.mean(np.abs(np.diff(preds, axis=1))))
    ref_step_delta = float(np.mean(np.abs(np.diff(refs, axis=1))))
    pred_first_last = float(np.mean(np.abs(preds[:, -1] - preds[:, 0])))
    ref_first_last = float(np.mean(np.abs(refs[:, -1] - refs[:, 0])))
    ratio = pred_step_delta / max(ref_step_delta, 1e-12)

    return {
        "pred_mean_step_delta": pred_step_delta,
        "ref_mean_step_delta": ref_step_delta,
        "pred_first_last_delta": pred_first_last,
        "ref_first_last_delta": ref_first_last,
        "pred_ref_step_delta_ratio": float(ratio),
    }


def _log_multi_obstacle_rollout_step_slider(
    predictions: np.ndarray,
    references: np.ndarray,
    *,
    wandb_prefix: str = "test",
    max_steps: int | None = None,
):
    if predictions.ndim != 5 or references.ndim != 5:
        raise ValueError(
            f"Expected rollout tensors with shape (N,T,C,H,W), got predictions {predictions.shape}, references {references.shape}"
        )

    total_steps = min(int(predictions.shape[1]), int(references.shape[1]))
    if total_steps <= 0:
        return

    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    sample_indices = list(range(min(4, predictions.shape[0])))
    panel_preds = predictions[sample_indices, :total_steps]
    panel_refs = references[sample_indices, :total_steps]
    panel_min = float(min(np.min(panel_preds), np.min(panel_refs)))
    panel_max = float(max(np.max(panel_preds), np.max(panel_refs)))

    step_data_uris = []
    step_indices = []
    for step_idx in range(total_steps):
        fig = _create_predictions_figure(
            predictions[:, step_idx],
            references[:, step_idx],
            indices=sample_indices,
            value_range=(panel_min, panel_max),
        )
        step_data_uris.append(_figure_to_data_uri(fig))
        step_indices.append(step_idx)

    html = _build_step_slider_html(
        step_data_uris,
        step_indices,
        title=f"{wandb_prefix}/rollout_predictions",
    )
    wandb.log({f"{wandb_prefix}/rollout_predictions": wandb.Html(html)})

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

    if "_infer" in params.checkpoint_path and params.do_inference:
        params.checkpoint_path = params.checkpoint_path.replace("_infer", "")

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

    if params.do_inference:
        ckpt_dir = ckpt_dir.replace("_infer", "")

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

    ckpt = None
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

    if params.do_inference and params.finetune_from is None:
        load_path = None
        if trained_model_exists(ckpt_dir):
            load_path = ckpt_dir
        elif ckpt is not None:
            load_path = ckpt

        print(f"ckpt_dir: {ckpt_dir}")
        print(f"load_path for inference: {load_path}")

        if load_path is None:
            raise FileNotFoundError(
                "Inference requested but no saved checkpoint was found. "
                "Please provide --finetune_from or ensure a checkpoint exists in the run directory."
            )

        if RANK == 0 or RANK == -1:
            print(f"Loading model weights for inference from: {load_path}")
        model = ScOT.from_pretrained(
            load_path,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
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

        # Handle both 4D (num_samples, channels, H, W) and 5D (num_samples, T, channels, H, W) predictions
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        if preds.ndim == 5 and labels.ndim == 5:
            # Match full rollout targets: flatten time into batch
            N, T, C, H, W = preds.shape
            preds = preds.reshape(N * T, C, H, W)
            labels = labels.reshape(N * T, C, H, W)
        elif preds.ndim == 5 and labels.ndim == 4:
            # Single-step targets: compare only first predicted step
            preds = preds[:, 0]
        elif preds.ndim == 4 and labels.ndim == 5 and labels.shape[1] == 1:
            labels = labels[:, 0]

        error_statistics = [
            get_statistics(
                relative_lp_error(
                    preds[:, channel_list[i] : channel_list[i + 1]],
                    labels[:, channel_list[i] : channel_list[i + 1]],
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

        if (
            params.do_inference
            or (
            "MultiObstacleIncompressibleMultiFrame" in config["dataset"]
            or "MultiObstacleIncompressibleMultiframe" in config["dataset"]
            )
        ):
            requested_total_frames = 200
            print(
                f"\nRunning multi-obstacle long-rollout evaluation for {requested_total_frames} total frames..."
            )
            rollout_history = 5
            long_rollout_set_kwargs = {
                "frames": rollout_history + 1,
                "history": rollout_history,
                "T_len": 1,
                "resolution": config["image_size"],
            }
            if params.move_data is not None:
                long_rollout_set_kwargs["move_to_local_scratch"] = params.move_data

            long_rollout_dataset = get_dataset(
                dataset=config["dataset"],
                which="test",
                num_trajectories=config["eval_ndata"],
                max_traj_per_file=config["max_traj_per_file"],
                file_count=config["file_count"],
                file_prefix=config["file_prefix"],
                N_val=config["eval_ndata"],
                data_path=params.data_path,
                **long_rollout_set_kwargs,
            )

            long_rollout_dataset._load_file(0)

            frame_axis = 1 if getattr(long_rollout_dataset, "_layout", "NTHW") == "NTHW" else 3
            available_rollout_steps = int(long_rollout_dataset._vx.shape[frame_axis]) - rollout_history
            long_rollout_steps = min(requested_total_frames - rollout_history, available_rollout_steps)
            if available_rollout_steps < requested_total_frames - rollout_history:
                print(
                    f"WARNING: only {available_rollout_steps} rollout steps are available from the stored trajectory; "
                    f"requested {requested_total_frames - rollout_history}."
                )
            print(
                f"Using history={rollout_history} and {long_rollout_steps} autoregressive steps "
                f"to evaluate {rollout_history + long_rollout_steps} total frames."
            )

            old_inference_time = trainer.inference_time
            old_eval_bs = trainer.args.per_device_eval_batch_size
            old_compute_metrics = trainer.compute_metrics
            old_ar_steps = trainer.ar_steps
            old_output_all_steps = trainer.output_all_steps
            trainer.inference_time = 0.0
            trainer.args.per_device_eval_batch_size = 1
            trainer.compute_metrics = None
            trainer.set_ar_steps([1] * long_rollout_steps, output_all_steps=True)
            torch.cuda.empty_cache()
            long_rollout_ts = time.time()
            long_rollout_predictions = trainer.predict(long_rollout_dataset, metric_key_prefix="")
            long_rollout_te = time.time()

            rollout_preds_norm = np.asarray(long_rollout_predictions.predictions)
            if rollout_preds_norm.ndim != 5:
                raise ValueError(
                    f"Long-rollout predictions must be 5D (N,T,C,H,W) when output_all_steps=True, got {rollout_preds_norm.shape}"
                )
            print(f"Long-rollout prediction tensor shape: {rollout_preds_norm.shape}")
            rollout_preds = _denormalize_multi_obstacle_rollout(
                rollout_preds_norm,
                long_rollout_dataset.constants["mean"].cpu().numpy(),
                long_rollout_dataset.constants["std"].cpu().numpy(),
            )
            rollout_refs = _extract_multi_obstacle_reference_rollout(
                long_rollout_dataset,
                long_rollout_steps,
            )
            rollout_physical_scores = (
                _channelwise_physical_nrmse_multi_obstacle(
                    rollout_preds,
                    rollout_refs,
                    _multi_obstacle_fluid_masks(
                        long_rollout_dataset,
                        rollout_preds.shape[0],
                    ),
                )
            )
            rollout_physical_nrmse = rollout_physical_scores.mean(axis=0)
            rollout_physical_nrmse_macro = float(
                rollout_physical_nrmse.mean()
            )
            print(
                "Channelwise physical-unit nRMSE | "
                f"vx={rollout_physical_nrmse[0]:.6f}, "
                f"vy={rollout_physical_nrmse[1]:.6f}, "
                "pressure(gauge-centered)="
                f"{rollout_physical_nrmse[2]:.6f}, "
                f"macro={rollout_physical_nrmse_macro:.6f}"
            )
            mean = long_rollout_dataset.constants["mean"].cpu().numpy().reshape(1, 1, -1, 1, 1)
            std = long_rollout_dataset.constants["std"].cpu().numpy().reshape(1, 1, -1, 1, 1)
            rollout_refs_norm = (rollout_refs - mean) / std
            rollout_diagnostics_norm = _compute_rollout_temporal_diagnostics(
                rollout_preds_norm,
                rollout_refs_norm,
            )

            rollout_stats = _compute_multi_obstacle_time_avg_stats(
                rollout_preds,
                rollout_refs,
                last_n=160,
            )
            rollout_diagnostics = _compute_rollout_temporal_diagnostics(
                rollout_preds,
                rollout_refs,
            )

            if RANK == 0 or RANK == -1:
                _log_multi_obstacle_rollout_step_slider(
                    rollout_preds_norm,
                    rollout_refs_norm,
                    wandb_prefix="test/long_rollout",
                    max_steps=requested_total_frames - rollout_history,
                )

                short_rollout_steps = min(14, rollout_preds.shape[1], rollout_refs.shape[1])
                if short_rollout_steps > 0:
                    _log_multi_obstacle_rollout_step_slider(
                        rollout_preds_norm,
                        rollout_refs_norm,
                        wandb_prefix="test/short_rollout",
                        max_steps=short_rollout_steps,
                    )

            print("Multi-obstacle 200-step rollout time-average statistics:")
            for quantity in ["velocity_x", "velocity_y", "pressure"]:
                print(
                    f"  {quantity:>10s} | flip: {rollout_stats[f'tavg_flip_{quantity}']:.6f} | "
                    f"noflip: {rollout_stats[f'tavg_noflip_{quantity}']:.6f} | "
                    f"worst_noflip: {rollout_stats[f'worst_tavg_noflip_{quantity}']:.6f} "
                    f"(idx={rollout_stats[f'worst_tavg_noflip_{quantity}_traj_idx']})"
                )
            print(
                f"  {'velocity':>10s} | flip: {rollout_stats['tavg_flip_velocity']:.6f} | "
                f"noflip: {rollout_stats['tavg_noflip_velocity']:.6f} | "
                f"worst_noflip: {rollout_stats['worst_tavg_noflip_velocity']:.6f} "
                f"(idx={rollout_stats['worst_tavg_noflip_velocity_traj_idx']})"
            )
            print(
                f"  {'all_fields':>10s} | flip: {rollout_stats['tavg_flip_all_fields']:.6f} | "
                f"noflip: {rollout_stats['tavg_noflip_all_fields']:.6f} | "
                f"worst_noflip: {rollout_stats['worst_tavg_noflip_all_fields']:.6f} "
                f"(idx={rollout_stats['worst_tavg_noflip_all_fields_traj_idx']})"
            )
            print(
                f"Multi-obstacle 200-step rollout inference time (total): {long_rollout_te - long_rollout_ts}"
            )
            print(
                f"Multi-obstacle 200-step rollout inference time per trajectory: "
                f"{(long_rollout_te - long_rollout_ts) / long_rollout_dataset._num_traj_total}"
            )
            print(
                "Rollout temporal diagnostics | "
                f"pred_step_delta={rollout_diagnostics['pred_mean_step_delta']:.6e}, "
                f"ref_step_delta={rollout_diagnostics['ref_mean_step_delta']:.6e}, "
                f"pred/ref ratio={rollout_diagnostics['pred_ref_step_delta_ratio']:.6f}"
            )
            print(
                "Rollout temporal diagnostics (normalized) | "
                f"pred_step_delta={rollout_diagnostics_norm['pred_mean_step_delta']:.6e}, "
                f"ref_step_delta={rollout_diagnostics_norm['ref_mean_step_delta']:.6e}, "
                f"pred/ref ratio={rollout_diagnostics_norm['pred_ref_step_delta_ratio']:.6f}"
            )

            if RANK == 0 or RANK == -1:
                rollout_metrics = {
                    "test/long_rollout_infer_time": long_rollout_te - long_rollout_ts,
                    "test/long_rollout_infer_time_per_traj": (
                        (long_rollout_te - long_rollout_ts) / long_rollout_dataset._num_traj_total
                    ),
                }
                for quantity in ["velocity_x", "velocity_y", "pressure"]:
                    rollout_metrics[f"test/{quantity}/tavg_flip_nrmse"] = rollout_stats[
                        f"tavg_flip_{quantity}"
                    ]
                    rollout_metrics[f"test/{quantity}/tavg_noflip_nrmse"] = rollout_stats[
                        f"tavg_noflip_{quantity}"
                    ]
                    rollout_metrics[f"test/{quantity}/worst_tavg_flip_nrmse"] = rollout_stats[
                        f"worst_tavg_flip_{quantity}"
                    ]
                    rollout_metrics[f"test/{quantity}/worst_tavg_noflip_nrmse"] = rollout_stats[
                        f"worst_tavg_noflip_{quantity}"
                    ]
                    rollout_metrics[f"test/{quantity}/worst_tavg_flip_traj_idx"] = rollout_stats[
                        f"worst_tavg_flip_{quantity}_traj_idx"
                    ]
                    rollout_metrics[f"test/{quantity}/worst_tavg_noflip_traj_idx"] = rollout_stats[
                        f"worst_tavg_noflip_{quantity}_traj_idx"
                    ]
                rollout_metrics["test/velocity/tavg_flip_nrmse"] = rollout_stats[
                    "tavg_flip_velocity"
                ]
                rollout_metrics["test/velocity/tavg_noflip_nrmse"] = rollout_stats[
                    "tavg_noflip_velocity"
                ]
                rollout_metrics["test/velocity/worst_tavg_flip_nrmse"] = rollout_stats[
                    "worst_tavg_flip_velocity"
                ]
                rollout_metrics["test/velocity/worst_tavg_noflip_nrmse"] = rollout_stats[
                    "worst_tavg_noflip_velocity"
                ]
                rollout_metrics["test/velocity/worst_tavg_flip_traj_idx"] = rollout_stats[
                    "worst_tavg_flip_velocity_traj_idx"
                ]
                rollout_metrics["test/velocity/worst_tavg_noflip_traj_idx"] = rollout_stats[
                    "worst_tavg_noflip_velocity_traj_idx"
                ]
                rollout_metrics["test/all_fields/tavg_flip_nrmse"] = rollout_stats[
                    "tavg_flip_all_fields"
                ]
                rollout_metrics["test/all_fields/tavg_noflip_nrmse"] = rollout_stats[
                    "tavg_noflip_all_fields"
                ]
                rollout_metrics["test/all_fields/worst_tavg_flip_nrmse"] = rollout_stats[
                    "worst_tavg_flip_all_fields"
                ]
                rollout_metrics["test/all_fields/worst_tavg_noflip_nrmse"] = rollout_stats[
                    "worst_tavg_noflip_all_fields"
                ]
                rollout_metrics["test/all_fields/worst_tavg_flip_traj_idx"] = rollout_stats[
                    "worst_tavg_flip_all_fields_traj_idx"
                ]
                rollout_metrics["test/all_fields/worst_tavg_noflip_traj_idx"] = rollout_stats[
                    "worst_tavg_noflip_all_fields_traj_idx"
                ]
                rollout_metrics["test/long_rollout/pred_mean_step_delta"] = rollout_diagnostics[
                    "pred_mean_step_delta"
                ]
                rollout_metrics["test/long_rollout/ref_mean_step_delta"] = rollout_diagnostics[
                    "ref_mean_step_delta"
                ]
                rollout_metrics["test/long_rollout/pred_first_last_delta"] = rollout_diagnostics[
                    "pred_first_last_delta"
                ]
                rollout_metrics["test/long_rollout/ref_first_last_delta"] = rollout_diagnostics[
                    "ref_first_last_delta"
                ]
                rollout_metrics["test/long_rollout/pred_ref_step_delta_ratio"] = rollout_diagnostics[
                    "pred_ref_step_delta_ratio"
                ]
                rollout_metrics["test/long_rollout_norm/pred_mean_step_delta"] = rollout_diagnostics_norm[
                    "pred_mean_step_delta"
                ]
                rollout_metrics["test/long_rollout_norm/ref_mean_step_delta"] = rollout_diagnostics_norm[
                    "ref_mean_step_delta"
                ]
                rollout_metrics["test/long_rollout_norm/pred_ref_step_delta_ratio"] = rollout_diagnostics_norm[
                    "pred_ref_step_delta_ratio"
                ]
                rollout_metrics[
                    "test/channelwise_physical_nrmse/velocity_x"
                ] = float(rollout_physical_nrmse[0])
                rollout_metrics[
                    "test/channelwise_physical_nrmse/velocity_y"
                ] = float(rollout_physical_nrmse[1])
                rollout_metrics[
                    "test/channelwise_physical_nrmse/pressure_gauge_centered"
                ] = float(rollout_physical_nrmse[2])
                rollout_metrics[
                    "test/channelwise_physical_nrmse_macro"
                ] = rollout_physical_nrmse_macro
                rollout_metrics["test/nrmse"] = rollout_physical_nrmse_macro
                wandb.log(rollout_metrics)

            trainer.inference_time = old_inference_time
            trainer.args.per_device_eval_batch_size = old_eval_bs
            trainer.compute_metrics = old_compute_metrics
            trainer.ar_steps = old_ar_steps
            trainer.output_all_steps = old_output_all_steps

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

    # tee = time.time()

    # # print("train time:", te-ts)
    # print("test time:", tee-te)
