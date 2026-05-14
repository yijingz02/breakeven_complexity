import torch


def build_param_groups(model: torch.nn.Module, param_groups_cfg):
    """Build parameter groups for the optimizer with different learning rates.

    Currently very hardcoded, but factored into separate function to make this easier to change down the line.
    """
    param_groups = []
    for group_cfg in param_groups_cfg:
        layer_params = getattr(model, group_cfg["params"])  # .parameters()
        if isinstance(layer_params, torch.nn.Parameter):
            layer_params = [layer_params]
        elif isinstance(layer_params, torch.nn.Module):
            layer_params = layer_params.parameters()
        elif isinstance(layer_params, (list, tuple)):
            # Assume list of parameters
            pass
        else:
            raise ValueError(
                f"Unknown type {type(layer_params)} for param group {group_cfg.params}"
            )
        options = {k: v for k, v in group_cfg.items() if k != "params"}
        param_groups.append(
            {"params": layer_params, "name": group_cfg["params"], **options}
        )
    return param_groups
