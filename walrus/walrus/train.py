import logging
import os
import pathlib
from typing import Dict, Optional, cast

import hydra
import torch
import wandb
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torchinfo import summary

from walrus.data import MixedWellDataModule
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.optim.optim_utils import build_param_groups
from walrus.trainer.checkpoints import CheckPointLoader
from walrus.trainer.training import Trainer
from walrus.utils.distribution_utils import configure_distribution, distribute_model
from walrus.utils.experiment_utils import (
    align_checkpoint_with_field_to_index_map,
    configure_experiment,
)

logger = logging.getLogger("walrus")
# logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = pathlib.Path(__file__).parent / "configs"
CONFIG_NAME = "config"
CONFIG_PATH = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
assert CONFIG_PATH.is_file(), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


# -------------------------
# Budget / W&B helpers
# -------------------------
def _cfg_get(cfg: DictConfig, path: str, default=None):
    """
    Safe nested access for DictConfig like "trainer.time_budget_s".
    Returns default if missing.
    """
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, DictConfig):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            return default
    return cur


def _format_num(x) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if xf.is_integer():
        return str(int(xf))
    return f"{xf:g}"


def build_wandb_name(cfg: DictConfig):
    """
    Returns (group, name)
      group = B{budget}
      name  = N{Ndata}
    Falls back safely if missing.
    """
    time_budget_s = _cfg_get(cfg, "trainer.time_budget_s", None)
    Ndata_generated = _cfg_get(cfg, "trainer.Ndata_generated", None)

    if time_budget_s is None:
        group = str(_cfg_get(cfg, "data.wandb_data_name", "default"))
    else:
        group = f"B{_format_num(time_budget_s)}"

    if Ndata_generated is None:
        name = str(_cfg_get(cfg, "name", "run"))
    else:
        name = f"N{int(Ndata_generated)}_B{_format_num(time_budget_s)}"

    return group, name

def load_from_coalesced_checkpoint(
    model: torch.nn.Module,
    coalesced_checkpoint_path: str,
    field_to_index_map: Dict,
    old_field_index_map: Optional[Dict] = None,
    align_fields: bool = True,
):
    """Load model weights from a coalesced checkpoint, aligning field indices if necessary."""
    logger.info(f"Loading coalesced checkpoint {coalesced_checkpoint_path}")
    checkpoint = torch.load(coalesced_checkpoint_path, map_location="cpu")
    # Load the model weights
    model_checkpoint = checkpoint["app"]["model"]
    if align_fields and field_to_index_map != old_field_index_map:
        # NOTE (MM) - "model" n_fields should be strictly larger than "old" n_fields to loading
        # the old field index map values into the new override, but indices may not line
        # up exactly, so it's safer to align regardless
        model_checkpoint = align_checkpoint_with_field_to_index_map(
            checkpoint_state_dict=model_checkpoint,
            model_state_dict=model.state_dict(),
            checkpoint_field_to_index_map=old_field_index_map,
            model_field_to_index_map=field_to_index_map,
        )
    model.load_state_dict(model_checkpoint, strict=True)
    return model


def train(
    cfg: DictConfig,
    experiment_name: str,
    experiment_folder: str,
    viz_folder: str,
    old_field_index_map: Optional[Dict] = None,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    device_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
):
    """Instantiate the different objects required for training and run the training loop."""
    logger.info(f"Instantiate datamodule {cfg.data.wandb_data_name}")
    datamodule: MixedWellDataModule = instantiate(
        cfg.data.module_parameters,
        world_size=world_size,
        rank=rank,
        data_workers=cfg.data_workers,
        well_base_path=cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )
    field_to_index_map = datamodule.train_dataset.field_to_index_map
    total_input_fields = max(field_to_index_map.values()) + 1

    logger.info(f"Instantiate model {cfg.model._target_}")
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_states=total_input_fields,
    )

    # --------------- checkpoint loading logic ---------------
    load_coalesced_chkpt_cond = (
        hasattr(cfg.checkpoint, "coalesced_checkpoint_path")
        and cfg.checkpoint.coalesced_checkpoint_path is not None
        and (cfg.finetune or cfg.validation_mode)
    )
    if load_coalesced_chkpt_cond and not cfg.checkpoint.get(
        "load_chkpt_after_finetuning_expansion", False
    ):
        model = load_from_coalesced_checkpoint(
            model=model,
            coalesced_checkpoint_path=cfg.checkpoint.coalesced_checkpoint_path,
            field_to_index_map=field_to_index_map,
            old_field_index_map=old_field_index_map,
            align_fields=cfg.checkpoint.get("align_fields", True),
        )

    if hasattr(cfg, "finetuning_mods"):
        if hasattr(model, "add_ft_options"):
            model.add_ft_options(cfg.finetuning_mods)

    if load_coalesced_chkpt_cond and cfg.checkpoint.get(
        "load_chkpt_after_finetuning_expansion", False
    ):
        model = load_from_coalesced_checkpoint(
            model=model,
            coalesced_checkpoint_path=cfg.checkpoint.coalesced_checkpoint_path,
            field_to_index_map=field_to_index_map,
            old_field_index_map=old_field_index_map,
            align_fields=cfg.checkpoint.get("align_fields", True),
        )

    if rank == 0:
        summary(model, depth=5)

    logger.info(f"Assigning distribution strategy: {cfg.distribution.distribution_type}")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(local_rank)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model = distribute_model(model, cfg, device_mesh)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    if hasattr(cfg.optimizer, "param_groups"):
        with open_dict(cfg):
            param_groups_cfg = OmegaConf.to_container(
                cfg.optimizer.pop("param_groups"), resolve=True
            )
    else:
        param_groups_cfg = None

    if param_groups_cfg is not None:
        param_groups = build_param_groups(model, param_groups_cfg=param_groups_cfg)
        optimizer = cast(
            torch.optim.Optimizer,
            instantiate(
                cfg.optimizer,
                params=param_groups,
                lr=cfg.optimizer.lr,
                _convert_="all",
            ),
        )
    else:
        optimizer = cast(
            torch.optim.Optimizer,
            instantiate(
                cfg.optimizer,
                params=model.parameters(),
                lr=cfg.optimizer.lr,
                _convert_="all",
            ),
        )

    start_epoch = 1
    last_epoch = -1
    val_loss = torch.tensor(float("inf"))

    logger.info(f"Instantiate checkpointer {cfg.checkpoint._target_}")
    checkpointer: CheckPointLoader = instantiate(cfg.checkpoint, rank=rank)
    if hasattr(checkpointer, "load_checkpoint_path"):
        load_checkpoint_path = checkpointer.load_checkpoint_path
        if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
            if (
                cfg.finetune or cfg.validation_mode
            ) and load_checkpoint_path != checkpointer.last_checkpoint:
                logger.info(f"Finetuning from checkpoint {load_checkpoint_path}")
                checkpointer.load(
                    model,
                    local=cfg.distribution.distribution_type.upper() in ["LOCAL", "DDP"],
                )
            else:
                logger.info(f"Resume from checkpoint {load_checkpoint_path}")
                epoch, val_loss = checkpointer.load(
                    model,
                    optimizer,
                    local=cfg.distribution.distribution_type.upper() in ["LOCAL", "DDP"],
                )

                for i, param_group in enumerate(optimizer.param_groups):
                    if "initial_lr" not in param_group:
                        if param_groups_cfg is not None:
                            group_cfg = param_groups_cfg[i]
                            if "lr" in group_cfg:
                                param_group["initial_lr"] = group_cfg["lr"]
                            else:
                                param_group["initial_lr"] = cfg.optimizer.lr
                        else:
                            param_group["initial_lr"] = cfg.optimizer.lr

                logger.info(f"Resume from epoch {epoch} with validation loss {val_loss}")
                start_epoch = 1 if epoch is None else epoch + 1
                last_epoch = start_epoch - 1

    if hasattr(cfg, "lr_scheduler"):
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        if cfg.trainer.lr_scheduler_per_step:
            step_mult_factor = (
                cfg.data.module_parameters.max_samples / cfg.trainer.grad_acc_steps
            )
        else:
            step_mult_factor = 1

        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.max_epoch,
            step_mult_factor=step_mult_factor,
            last_epoch=max(-1, last_epoch - 1),
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    with open_dict(cfg):
        cfg.data.field_index_map_override = field_to_index_map

    if rank == 0:
        logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")

    logger.info(f"Instantiate trainer {cfg.trainer._target_}")

    batch_aggregation_fns = [
        get_method(name)
        for name in cfg.trainer.get("batch_aggregation_fns", ["torch.mean"])
    ]

    # -------------------------
    # Budget knobs (Hydra CLI: ++trainer.time_budget_s=..., ++trainer.Ndata_generated=..., ++trainer.gen_time_s=...)
    # -------------------------
    time_budget_s = _cfg_get(cfg, "trainer.time_budget_s", None)
    gen_time_s = _cfg_get(cfg, "trainer.gen_time_s", 0.0)
    Ndata_generated = _cfg_get(cfg, "trainer.Ndata_generated", None)
    save_path = _cfg_get(cfg, "trainer.save_path", None)

    if rank == 0:
        logger.info(
            "[BUDGET CFG] time_budget_s=%s, Ndata_generated=%s, gen_time_s=%s",
            str(time_budget_s),
            str(Ndata_generated),
            str(gen_time_s),
        )

    import inspect
    import walrus.trainer.training as training_mod
    logger.info(f"[DEBUG] walrus package file: {__import__('walrus').__file__}")
    logger.info(f"[DEBUG] training.py file: {training_mod.__file__}")
    logger.info(f"[DEBUG] Trainer class: {training_mod.Trainer}")
    logger.info(f"[DEBUG] Trainer.__init__ sig: {inspect.signature(training_mod.Trainer.__init__)}")

    trainer: Trainer = instantiate(
        cfg.trainer,
        experiment_name=experiment_name,
        batch_aggregation_fns=batch_aggregation_fns,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
        device=device,
        device_mesh=device_mesh,
        distribution_type=cfg.distribution.distribution_type,
        rank=rank,
        world_size=world_size,
        formatter=ChannelsFirstWithTimeFormatter,
        wandb_logging=cfg.logger.wandb,
        start_epoch=start_epoch,
        start_val_loss=val_loss,
        # NEW: compute-only budget discounted by one-time data generation cost
        time_budget_s=time_budget_s,
        gen_time_s=gen_time_s,
        Ndata_generated=Ndata_generated,
        save_path=save_path,
        single_frame_initialization=bool(
            cfg.data.module_parameters.get("single_frame_initialization", False)
        )
    )

    if cfg.validation_mode:
        if rank == 0 and not os.path.exists(
            pathlib.Path(experiment_folder) / "extended_config.yaml"
        ):
            with open(pathlib.Path(experiment_folder) / "extended_config.yaml", "w") as f:
                OmegaConf.save(cfg, f)
        trainer.validate()
    else:
        if rank == 0:
            with open(pathlib.Path(experiment_folder) / "extended_config.yaml", "w") as f:
                OmegaConf.save(cfg, f)
        trainer.train()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name=str(CONFIG_NAME))
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = (
        cfg.distribution.distribution_type.upper() != "LOCAL" and world_size > 1
    )

    device_mesh = configure_distribution(cfg)
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
        old_field_index_map,
    ) = configure_experiment(cfg, rank, is_distributed)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    config_for_wandb = cast(Dict, OmegaConf.to_container(cfg, resolve=True))
    config_for_wandb["world_size"] = world_size
    config_for_wandb["global_batch_size"] = (
        cfg.data.module_parameters.batch_size * world_size
    ) * cfg.trainer.grad_acc_steps

    if rank == 0 and cfg.logger.wandb:
        group, name = build_wandb_name(cfg)
        wandb.init(
            project=cfg.logger.wandb_project_name,
            group=group,
            config=config_for_wandb,
            name=name,
        )

    train(
        cfg,
        experiment_name,
        experiment_folder,
        viz_folder,
        old_field_index_map,
        world_size,
        rank,
        local_rank,
        device_mesh=device_mesh,
    )

    if rank == 0 and cfg.logger.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
