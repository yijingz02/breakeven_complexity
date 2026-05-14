import logging
import os
import os.path as osp
from typing import Tuple, cast

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)


def align_checkpoint_with_field_to_index_map(
    checkpoint_state_dict,
    model_state_dict,
    checkpoint_field_to_index_map,
    model_field_to_index_map,
    embed_string="embed",
    embed_weight_name="proj1.weight",
    debed_string="debed",
    debed_weight_name="proj2.weight",
    debed_bias_name="proj2.bias",
):
    """Manually copies field-aware weights into alignment with a new field_to_index map.

    Default architecture must do this for the weight of the first input projection and
    weight and bias of the last output projection"""
    checkpoint_num_dims = max(checkpoint_field_to_index_map.values()) + 1
    model_num_dims = max(model_field_to_index_map.values()) + 1
    scale_factor = (checkpoint_num_dims / model_num_dims) ** 0.5
    for param_name in model_state_dict:
        # Look for weights requiring reshape
        if (
            (embed_string in param_name and embed_weight_name in param_name)
            or (debed_string in param_name and debed_weight_name in param_name)
            or (debed_string in param_name and debed_bias_name in param_name)
        ):
            replacement_param = model_state_dict[param_name].clone()
            checkpoint_param = checkpoint_state_dict[param_name]
            # Loop through aligning
            for field in model_field_to_index_map:
                if field in checkpoint_field_to_index_map:
                    if debed_bias_name in param_name:
                        replacement_param[model_field_to_index_map[field]] = (
                            checkpoint_param[checkpoint_field_to_index_map[field]]
                        )
                    # Embedding weights need to be rescaled to match their new size
                    elif embed_weight_name in param_name:
                        replacement_param[:, model_field_to_index_map[field]] = (
                            checkpoint_param[:, checkpoint_field_to_index_map[field]]
                            * scale_factor
                        )
                    # Otherwise, last case is debedding weight
                    else:
                        replacement_param[:, model_field_to_index_map[field]] = (
                            checkpoint_param[:, checkpoint_field_to_index_map[field]]
                        )
            checkpoint_state_dict[param_name] = replacement_param.clone()
    return checkpoint_state_dict


def configure_paths(experiment_folder, rank=0):
    """Configure the paths for the experiment with the given experiment folder."""
    # Make ____ directory as experiment_folder/______
    if rank == 0:
        os.makedirs(osp.join(experiment_folder, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(experiment_folder, "artifacts"), exist_ok=True)
        os.makedirs(osp.join(experiment_folder, "viz"), exist_ok=True)
    # Return corresponding paths
    checkpoint_folder = osp.join(experiment_folder, "checkpoints")
    artifact_folder = osp.join(experiment_folder, "artifacts")
    viz_folder = osp.join(experiment_folder, "viz")
    return checkpoint_folder, artifact_folder, viz_folder


def get_experiment_name(cfg: DictConfig) -> str:
    """
    Get the experiment name based on the configuration model, data, and optimizer.

    Used to set default save path if not overridden.

    This is a messy hardcoded process that is likely a good candidate for refactoring.
    """
    # Data section
    data_name = cfg.data.wandb_data_name.replace("_", "")[:5]
    # Model section
    model_name = cfg.model._target_.split(".")[-1].replace("_", "")[:5]
    if hasattr(cfg.model, "encoder"):
        encoder_name = cfg.model.encoder._target_.split(".")[-1].replace("_", "")[:5]
    else:
        encoder_name = ""
    if hasattr(cfg.model, "decoder"):
        decoder_name = cfg.model.decoder._target_.split(".")[-1].replace("_", "")[:5]
    else:
        decoder_name = ""
    # TODO - this is sloppy but get's what I need for now (getting space/time names). Maybe recursive search for _target_s?
    if hasattr("cfg.model", "processor"):
        processor_name = cfg.model.processor._target_.split(".")[-1].replace("_", "")[
            :5
        ]
        if hasattr(cfg.model.processor, "space_mixing"):
            space_name = cfg.model.processor.space_mixing._target_.split(".")[
                -1
            ].replace("_", "")[:5]
            processor_name += f"-{space_name}"
        if hasattr(cfg.model.processor, "time_mixing"):
            time_name = cfg.model.processor.time_mixing._target_.split(".")[-1].replace(
                "_", ""
            )[:5]
            processor_name += f"-{time_name}"
    else:
        processor_name = ""
    # Optimizer section
    optimizer_name = cfg.optimizer._target_.split(".")[-1].replace("_", "")[:5]
    # Training type section
    prediction_type = cfg.trainer.prediction_type.replace("_", "")[:5]
    aggregate_name = f"{cfg.name}-{data_name}-{prediction_type}-{model_name}[{encoder_name}-{decoder_name}-{processor_name}]-{optimizer_name}-{cfg.optimizer.lr}"
    return aggregate_name


def configure_experiment(
    cfg: DictConfig, rank: int = 0, is_distributed: bool = False
) -> Tuple[DictConfig, str, str, str, str, str]:
    """Works through resume logic to figure out where to save the current experiment
    and where to look to resume or validate previous experiments.

    If the user provides overrides for the folder/checkpoint/config, use them.

    If folder isn't provided, construct default. If autoresume or validation_mode is enabled,
    look for the most recent run under that directory and take the config and weights from it.

    If checkpoint is provided, use it to override any weights obtained until now. If
    any checkpoint is available either in the folder or checkpoint override, this
    is considered a resume run.

    If it's in validation mode but no checkpoint is found, throw an error.

    If config override is provided, use it (with the weights and current output folder).
    Otherwise start search over hierarchy.
      - If checkpoint is being used, look to see if it has an associated config file
      - If no checkpoint but folder, look in folder
      - If not, just use the default config (whatever is currently set)

    Parameters
    ----------
    cfg : DictConfig
        The yaml configuration object being modified/read
    rank : int, optional
        The rank of the current torch process, by default 0
    is_distributed : bool, optional
        Whether the current process is distributed, by default False
    """
    # Sort out default names and folders
    if not cfg.automatic_setup:
        return cfg, cfg.name, ".", "./checkpoints", "./artifacts", "./viz", {}
    experiment_name = get_experiment_name(cfg)
    if hasattr(cfg, "experiment_dir"):
        base_experiment_folder = cfg.experiment_dir
    else:
        base_experiment_folder = os.getcwd()
    base_experiment_folder = osp.join(base_experiment_folder, experiment_name)
    if cfg.finetune:
        base_experiment_folder = osp.join(base_experiment_folder, "finetune")
    experiment_folder = cfg.folder_override  # Default is ""
    checkpoint_folder_override = cfg.checkpoint_override  # Default is ""
    config_file = cfg.config_override  # Default is ""
    validation_mode = cfg.validation_mode
    # If using default naming, check for auto-resume, otherwise make a new folder with default name
    if len(experiment_folder) == 0:
        if osp.exists(base_experiment_folder):
            prev_runs = sorted(
                [f for f in os.listdir(base_experiment_folder) if f.isnumeric()],
                key=lambda x: int(x),
            )
        else:
            prev_runs = []
        # If distributed, barrier here to ensure all processes have same list of previous runs
        if is_distributed:
            torch.distributed.barrier()
        if (validation_mode or cfg.auto_resume) and len(prev_runs) > 0:
            experiment_folder = osp.join(base_experiment_folder, prev_runs[-1])
        elif (
            validation_mode
            and cfg.checkpoint.coalesced_checkpoint_path is None
            and len(checkpoint_folder_override) == 0
        ):
            raise ValueError(
                f"Validation mode enabled but no previous runs found in {base_experiment_folder}."
            )
        else:
            experiment_folder = osp.join(base_experiment_folder, str(len(prev_runs)))
        logger.info(
            f"No override experiment folder detected. Using default experiment folder {experiment_folder}"
        )
    else:
        logger.info(f"Using override experiment folder {experiment_folder}")
    # Barrier around this to ensure all processes choose same folder.
    if is_distributed:
        torch.distributed.barrier()
    if (
        len(config_file) == 0
    ):  # If no config override, check for config file in experiment folder
        config_file = osp.join(experiment_folder, "extended_config.yaml")
        if not osp.isfile(config_file):
            config_file = ""

    if len(config_file) > 0:
        old_cfg = cast(DictConfig, OmegaConf.load(config_file))
    else:
        # If no config override, use the current config
        old_cfg = cfg

    # Create experiment folder if it doesn't already exist
    if rank == 0:
        os.makedirs(experiment_folder, exist_ok=True)
    checkpoint_folder, artifact_folder, viz_folder = configure_paths(
        experiment_folder, rank=rank
    )
    if len(checkpoint_folder_override) > 0 and validation_mode:
        checkpoint_folder = checkpoint_folder_override
    elif len(checkpoint_folder_override) > 0:
        logger.info(
            "Checkpoint override outside of validation mode not currently supported. Ignoring checkpoint override."
        )
    with open_dict(cfg):
        # Overwrite the new checkpoint with frozen details - usually model shapes
        if "all" in cfg.frozen_components:
            logger.info(
                "Using exact previous config since `all` in frozen components list."
            )
            cfg = old_cfg
        cfg.checkpoint.save_dir = checkpoint_folder
        cfg.name = experiment_name
        # Merge field_index_maps
        old_field_index_map = old_cfg.data.get("field_index_map_override", {})
        if len(old_field_index_map) > 0:
            logger.info(
                "Combining current field_index_map_override with previous config."
            )
            current_field_index_map = cfg.data.get("field_index_map_override", {})
            if len(current_field_index_map) == 0:
                field_index = 0
            else:
                field_index = max(current_field_index_map.values()) + 1
            for field in old_field_index_map:
                if field not in current_field_index_map:
                    current_field_index_map[field] = field_index
                    field_index += 1
            cfg.data.field_index_map_override = current_field_index_map
    return (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
        old_field_index_map,
    )
