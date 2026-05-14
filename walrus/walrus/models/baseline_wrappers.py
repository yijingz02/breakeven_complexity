"""
Wrappers for comparison baseline models to make it easier to execute in our environment

Code for baselines largely left unchanged, just wrapped in modules that perform data
transforms to match their expected formats.
"""

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from .external_models.dpot import DPOTNet
from .external_models.dpot3d import DPOTNet3D
from .external_models.mpp_avit import (
    AttentionBlock,
    AViT,
    AxialAttentionBlock,
    SpaceTimeBlock,
)
from .external_models.scot import ScOT, ScOTConfig


def apply_checkpointing(module):
    # Store the original forward method
    original_forward = module.forward

    # Define a new forward method that uses checkpointing
    def checkpointed_forward(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)

    # Monkey patch the module's forward method
    module.forward = checkpointed_forward


def dpot_load_3d_components_from_2d(model, state_dict, components="all"):
    """
    :model: the model
    :state_dict: state_dict of source model
    :components: 'all' or list from 'patch', 'pos', 'blocks','time_agg','cls_head', 'scale_feats', 'out'
    """

    if next(iter(state_dict.keys())).startswith("module."):
        new_state_dict = OrderedDict()
        for key, item in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = item
        del state_dict
        state_dict = new_state_dict
    if (components == "all") or ("all" in components):
        model.load_state_dict(state_dict)
        return
    else:
        for name in components:
            if name == "blocks" and hasattr(model, "blocks"):
                for i, block in enumerate(model.blocks):
                    block_state_dict = OrderedDict(
                        {
                            k.replace(f"blocks.{i}.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith(f"blocks.{i}.")
                        }
                    )
                    ## reshape 2d conv param to 3d conv param
                    for k, v in block_state_dict.items():
                        if "mlp" in k and "weight" in k:
                            block_state_dict[k] = v.unsqueeze(-1)
                    block.load_state_dict(block_state_dict)

            elif name == "time_agg" and hasattr(model, "time_agg_layer"):
                model.time_agg_layer.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("time_agg_layer.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("time_agg_layer.")
                        }
                    )
                )
            else:
                print(f"Submodule does not exists：{name}")
        return


def dim_pad(x, max_d):
    """
    Assume T B C are first channels, then see how many spatial dims we need to append/
    """
    squeeze = 0
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    return x, squeeze


class ScOTWrapper(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=11,
        num_out_channels=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        skip_connections=[True, True, True],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,  # for loss: 1 for l1, 2 for l2
        channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
        residual_model="convnext",  # "convnext" or "resnet"
        use_conditioning=False,
        learn_residual=False,
        use_mask_token=False,
        from_pretrained="",
        gradient_checkpointing_freq=0,
        **kwargs,
    ):
        super().__init__()
        config = ScOTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_out_channels=num_out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            skip_connections=skip_connections,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_absolute_embeddings=use_absolute_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            p=p,  # for loss: 1 for l1, 2 for l2
            channel_slice_list_normalized_loss=channel_slice_list_normalized_loss,  # if None will fall back to absolute loss otherwise normalized loss with split channels
            residual_model=residual_model,  # "convnext" or "resnet"
            use_conditioning=use_conditioning,
            learn_residual=learn_residual,  # learn the residual for time-dependent problems
            # **kwargs,
        )
        if len(from_pretrained) > 0:
            if from_pretrained not in {"T", "B", "L"}:
                raise ValueError(
                    f"From pretrained value of {from_pretrained} not a valid poseidon model size. Use T, B, or L"
                )
            self.inner_model = ScOT.from_pretrained(
                f"camlab-ethz/Poseidon-{from_pretrained}",
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            self.inner_model = ScOT(config)
        if gradient_checkpointing_freq > 0:
            for i, layer in enumerate(self.inner_model.decoder.layers):
                if (i + 1) % gradient_checkpointing_freq == 0:
                    apply_checkpointing(layer)
                    print(f"Using gradient checkpointing for ScOT decoder layer {i}")
            for i, layer in enumerate(self.inner_model.encoder.layers):
                if (i + 1) % gradient_checkpointing_freq == 0:
                    apply_checkpointing(layer)
                    print(f"Using gradient checkpointing for ScOT encoder layer {i}")
            for i, layer in enumerate(self.inner_model.residual_blocks):
                if (i + 1) % gradient_checkpointing_freq == 0:
                    apply_checkpointing(layer)
                    print(f"Using gradient checkpointing for ScOT residual block {i}")
        self.causal_in_time = False

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
    ):
        # Reshape inputs to match poseidon
        # Well inputs - T x B x C x H x W x D
        x = x[-1]  # eliminate singleton time dim
        # RUN MODEL
        preds = self.inner_model(
            pixel_values=x,
            time=torch.tensor([0.05], device=x.device),  # dummy time conditioning
        ).output
        preds = preds.unsqueeze(0)
        return preds


class DPOTWrapper(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        mixing_type="afno",
        in_channels=1,
        out_channels=4,
        in_timesteps=1,
        out_timesteps=1,
        n_blocks=4,
        embed_dim=768,
        out_layer_dim=32,
        depth=12,
        modes=32,
        mlp_ratio=1.0,
        n_cls=12,
        normalize=False,
        act="gelu",
        time_agg="exp_mlp",
        dim=2,
        load_checkpoint_path="",
        gradient_checkpointing_freq=0,
        **kwargs,
    ):
        """ """
        super().__init__()
        self.img_size = img_size
        if dim == 2:
            self.inner_model = DPOTNet(
                img_size=img_size,
                patch_size=patch_size,
                mixing_type=mixing_type,
                in_channels=in_channels,
                out_channels=out_channels,
                in_timesteps=in_timesteps,
                out_timesteps=out_timesteps,
                n_blocks=n_blocks,
                embed_dim=embed_dim,
                out_layer_dim=out_layer_dim,
                depth=depth,
                modes=modes,
                mlp_ratio=mlp_ratio,
                n_cls=n_cls,
                normalize=normalize,
                act=act,
                time_agg=time_agg,
            )
        elif dim == 3:
            self.inner_model = DPOTNet3D(
                img_size=img_size,
                patch_size=patch_size,
                mixing_type=mixing_type,
                in_channels=in_channels,
                out_channels=out_channels,
                in_timesteps=in_timesteps,
                out_timesteps=out_timesteps,
                n_blocks=n_blocks,
                embed_dim=embed_dim,
                out_layer_dim=out_layer_dim,
                depth=depth,
                modes=modes,
                mlp_ratio=mlp_ratio,
                n_cls=n_cls,
                normalize=normalize,
                act=act,
                time_agg=time_agg,
            )
        self.causal_in_time = False
        if len(load_checkpoint_path) > 0:
            print(f"Loading DPOT checkpoint from {load_checkpoint_path}")
            state_dict = torch.load(load_checkpoint_path, map_location="cpu")
            if dim == 2:
                current_model_state_dict = self.inner_model.state_dict()
                for k in state_dict["model"].keys():
                    if k in current_model_state_dict:
                        if (
                            state_dict["model"][k].shape
                            != current_model_state_dict[k].shape
                        ):
                            if k == "pos_embed":
                                pos_embed = state_dict["model"][k]
                                reshaped_pos_embed = F.interpolate(
                                    pos_embed,
                                    size=current_model_state_dict[k].shape[2:],
                                    mode="bilinear",
                                )
                                state_dict["model"][k] = reshaped_pos_embed
                            else:
                                print(
                                    f"  - Skipping {k} due to size mismatch, loaded: {state_dict['model'][k].shape}, expected: {current_model_state_dict[k].shape}"
                                )
                                state_dict["model"][k] = current_model_state_dict[k]
                    else:
                        print(f"  - Skipping {k} since not in current model")
                        # state_dict["model"].pop(k)
                # Strict is false her since encoder may be different size
                self.inner_model.load_state_dict(state_dict["model"], strict=False)
            elif dim == 3:
                dpot_load_3d_components_from_2d(
                    self.inner_model,
                    state_dict["model"],
                    components=["blocks", "time_agg"],
                )

        if gradient_checkpointing_freq > 0:
            for i, block in enumerate(self.inner_model.blocks):
                if (i + 1) % gradient_checkpointing_freq == 0:
                    apply_checkpointing(block)
                    print(f"Using gradient checkpointing for DPOT block {i}")
            apply_checkpointing(self.inner_model.time_agg_layer)
            # apply_checkpointing(self.inner_model.patch_embed)
            print(
                "Using gradient checkpointing for DPOT time_agg_layer and patch_embed"
            )

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
    ):
        # DPOT looks for B, X, Y, [z], T, C
        # Well inputs - T x B x C x H x W x D
        orig_len = len(x.shape)
        original_shape = None

        # print("DPOTWrapper input x shape", x.shape)
        if orig_len == 5:
            x = rearrange(x, "t b c h w -> b h w t c")
        elif orig_len == 6:
            if tuple(x.shape[-3:]) != tuple(self.img_size):
                T = x.shape[0]
                original_shape = x.shape[-3:]
                x = rearrange(x, "t b c h w d -> (t b) c h w d")
                x = F.interpolate(
                    x,
                    size=tuple(self.img_size),
                    mode="trilinear",
                    align_corners=False,
                )
                x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
            x = rearrange(x, "t b c h w d -> b h w d t c")
        # RUN MODEL
        # print("DPOTWrapper reshaped x shape", x.shape)
        preds = self.inner_model(x)[0]
        # # RESHAPE OUTPUTS
        if orig_len == 5:
            preds = rearrange(preds, "b h w t c -> t b c h w")
        elif orig_len == 6:
            preds = rearrange(preds, "b h w d t c -> t b c h w d")
            if original_shape is not None:
                T = preds.shape[0]
                preds = rearrange(preds, "t b c h w d -> (t b) c h w d")
                preds = F.interpolate(
                    preds,
                    size=tuple(original_shape),
                    mode="trilinear",
                    align_corners=False,
                )
                preds = rearrange(preds, "(t b) c h w d -> t b c h w d", t=T)
        return preds


class MPPWrapper(nn.Module):
    def __init__(
        self,
        patch_size=16,
        embed_dim=1024,
        processor_blocks=24,
        n_states=64,
        num_heads=16,
        drop_path=0.1,
        load_checkpoint_path="",
        gradient_checkpointing_freq=0,
        **kwargs,
    ):
        """ """
        super().__init__()
        space_block = partial(
            AxialAttentionBlock, embed_dim, num_heads, bias_type="rel"
        )
        time_block = partial(AttentionBlock, embed_dim, num_heads, bias_type="rel")
        run_block = partial(
            SpaceTimeBlock,
            embed_dim,
            num_heads,
            space_override=space_block,
            time_override=time_block,
            gradient_checkpointing=False,
        )
        self.inner_model = AViT(
            patch_size=patch_size,
            embed_dim=embed_dim,
            processor_blocks=processor_blocks,
            n_states=n_states,
            override_block=run_block,
            drop_path=drop_path,
        )

        self.causal_in_time = False
        if len(load_checkpoint_path) > 0:
            print(f"Loading MPP checkpoint from {load_checkpoint_path}")
            state_dict = torch.load(load_checkpoint_path, map_location="cpu")
            # Iterate through state dict and load parameters if there is a size match
            current_model_state_dict = self.inner_model.state_dict()
            for k in state_dict.keys():
                if k in current_model_state_dict:
                    if state_dict[k].shape != current_model_state_dict[k].shape:
                        print(
                            f"  - Skipping {k} due to size mismatch, loaded: {state_dict[k].shape}, expected: {current_model_state_dict[k].shape}"
                        )
                        state_dict[k] = current_model_state_dict[k]
                else:
                    print(f"  - Skipping {k} since not in current model")
            # Now load the state dict
            self.inner_model.load_state_dict(state_dict, strict=True)

        if gradient_checkpointing_freq > 0:
            for i, block in enumerate(self.inner_model.blocks):
                if (i + 1) % gradient_checkpointing_freq == 0:
                    apply_checkpointing(block)
                    print(f"Using gradient checkpointing for DPOT block {i}")

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
    ):
        # MPP looks for T, x B x C x H x W
        # Well inputs - T x B x C x H x W [x D]
        # RUN MODEL
        bcs = (torch.tensor(bcs[0]) == 2).int()
        preds = self.inner_model(x, [state_labels], bcs)
        preds = preds.unsqueeze(0)
        return preds
