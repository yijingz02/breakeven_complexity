"""DISCO: train a hypernetwork (hpnn) to output the operator network (opnn) 
which is then evolved to solve the PDE."""
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from torch._functorch.apis import vmap
from torch._functorch.functional_call import functional_call
from src.torchdiffeq import odeint_adjoint

from src.models.attention import SpaceTimeBlock, RMSGroupNorm
from src.utils import standardize
from src.utils.data_utils import DATASET_SPECS


def make_conv(
    dim: int, c_in: int, c_out: int, ksize: int, stride: int, padding: int, padding_mode: str, groups: int
) -> nn.Module:
    if dim == 1:
        return nn.Conv1d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False)
    elif dim == 2:
        return nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False)
    elif dim == 3:
        return nn.Conv3d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False)
    else:
        raise ValueError(f"Unsupported dim={dim}")
    

def make_conv_transpose(
    dim: int, c_in: int, c_out: int, ksize: int, stride: int, padding: int, groups: int
) -> nn.Module:
    if dim == 1:
        return nn.ConvTranspose1d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, groups=groups, bias=False)
    elif dim == 2:
        return nn.ConvTranspose2d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, groups=groups, bias=False)
    elif dim == 3:
        return nn.ConvTranspose3d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, groups=groups, bias=False)
    else:
        raise ValueError(f"Unsupported dim={dim}")

# ========= Hypernetwork (hpnn) =========

class SubsampledInLinear(nn.Module):
    """A linear layer with varying input channels but same output channels."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        dim_in = self.linear.in_features
        label_size = len(labels)
        weight, bias = self.linear.weight, self.linear.bias
        scale = torch.tensor((dim_in / label_size) ** .5, dtype=x.dtype, device=x.device)
        x = scale * F.linear(x, weight[:, labels], bias)
        return x
    

class SubsampledOutLinear(nn.Module):
    """ A linear layer with varying output channels but same input channels."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        weight, bias = self.linear.weight, self.linear.bias
        x = F.linear(x, weight[labels, :], bias[labels])
        return x


class Encoder(nn.Module):
    """A standard CNN encoder."""
    def __init__(self, 
        patch_size: int,
        embed_dim: int, 
        spatial_ndims: int, 
        padding_mode: str, 
        groups: int
    ):
        super().__init__()
        if patch_size not in [8, 16]:
            raise ValueError("Patch size must be one of 8, 16")
        ksize = patch_size // 4  # Will be 2 or 4
        self.encoder = nn.Sequential(*[
            make_conv(spatial_ndims, embed_dim//4, embed_dim//4, ksize=ksize, stride=ksize, padding=0, padding_mode=padding_mode, groups=1),
            RMSGroupNorm(groups, embed_dim//4, affine=True),
            nn.GELU(),
            make_conv(spatial_ndims, embed_dim//4, embed_dim//4, ksize=2, stride=2, padding=0, padding_mode=padding_mode, groups=1),
            RMSGroupNorm(groups, embed_dim//4, affine=True),
            nn.GELU(),
            make_conv(spatial_ndims, embed_dim//4, embed_dim, ksize=2, stride=2, padding=0, padding_mode=padding_mode, groups=1),
            RMSGroupNorm(groups, embed_dim, affine=True),
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Hypernetwork(nn.Module):
    """A hypernetwork that outputs the operator network latent representation.
    It projects the input (a short trajectory) into a low-dimensional space 
    (the operator space)."""
    def __init__(self, 
        patch_size: int,
        n_states: int,
        ndims: List[int], 
        embed_dim: int, 
        groups: int,
        processor_blocks: int,
        drop_path: float,
        num_heads: int,
        bias_type: str,
    ) -> None:
        super().__init__()

        # Adapters to varying number of input channels (e.g. pressure, velocity ...)
        self.adapter = SubsampledInLinear(dim_in=n_states, dim_out=embed_dim//4)

        # Encoder (one for each 1d, 2d, 3d)
        self.encoder = nn.ModuleDict({
            str(ndim): Encoder(patch_size, embed_dim, ndim, "reflect", groups)
            for ndim in ndims
        })

        # Processor (attention layers common to all trajectories)
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(dim=embed_dim, num_heads=num_heads, bias_type=bias_type, drop_path=dp)
            for dp in self.dp
        ])

    def forward(self, x: Tensor, state_labels: Tensor) -> Tensor:
        """ x is B T C H (W) (D) """
        # Dimensions
        B = x.shape[0]
        spatial_ndims = x.ndim - 3
        spatial_dims = tuple(range(3,x.ndim))
        x = x[(...,) + (None,) * (6 - x.ndim)]  # now x is B T C H W D

        # Preprocess
        x = standardize(x, dims=(1,*spatial_dims), return_stats=False)

        # Adapt varying input channels to a fixed number
        x = rearrange(x, 'b t c ... -> (b t) ... c')
        x = self.adapter(x, state_labels)
        x = rearrange(x, 'bt ... c -> bt c ...')

        # Encode
        x = x.squeeze((-2,-1))
        x = self.encoder[str(spatial_ndims)](x)
        x = rearrange(x, '(b t) c ... ->  b t c ...', b=B)

        # Attention layers
        all_att_maps = []
        for blk in self.blocks:
            x, att_maps = blk(x, return_att=False)
            all_att_maps += att_maps

        return x

# ========= Operator network (opnn) =========

class Down(nn.Module):
    """A downsampling block."""
    def __init__(self, 
        dim: int, 
        in_channels: int, 
        mid_channels: int,
        out_channels: int, 
        groups: int,
        norm_groups: int,
        padding_mode: str,
    ):
        super().__init__()
        self.conv1 = make_conv(dim, in_channels, mid_channels, ksize=2, stride=2, padding=0, padding_mode=padding_mode, groups=1)
        self.norm1 = nn.GroupNorm(norm_groups, mid_channels, affine=False)
        self.gelu1 = nn.GELU()
        self.conv2 = make_conv(dim, mid_channels, out_channels, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=groups)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels, affine=False)
        self.gelu2 = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu2(x)
        return x


class Up(nn.Module):
    """An upsampling block."""
    def __init__(self, 
        dim: int, 
        in_channels: int, 
        out_channels: int, 
        groups: int,
        norm_groups: int,
        padding_mode: str,
    ):
        super().__init__()
        self.up = make_conv_transpose(dim, in_channels, in_channels//2, ksize=2, stride=2, padding=0, groups=1)
        self.norm = nn.GroupNorm(norm_groups, in_channels//2, affine=False)
        self.gelu = nn.GELU()
        self.conv = make_conv(dim, in_channels, out_channels, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=groups)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.up(x1)
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x


def create_frontier_mask(x: Tensor) -> Tensor:
    """General frontier mask creator for arbitrary spatial dimensions (1D, 2D, 3D, ...).
    Assumes shape = (B, *spatial)."""
    ndim = x.ndim - 1
    mask = torch.zeros_like(x)
    
    for dim in range(ndim):
        slice_down = [...] + [0 if i == dim else slice(None) for i in range(ndim)]
        slice_up = [...] + [-1 if i == dim else slice(None) for i in range(ndim)]
        
        mask[tuple(slice_down)] = 1
        mask[tuple(slice_up)] = 1

    return mask


class OperatorNetwork(nn.Module):
    """A standard U-Net architecture."""
    def __init__(self, in_channels, start, spatial_ndims, boundary_conditions, norm_groups, num_heads=0):
        super().__init__()
        self.in_channels = in_channels
        self.start = start
        self.spatial_ndims = spatial_ndims
        self.boundary_conditions = boundary_conditions
        self.norm_groups = norm_groups
        self.num_heads = num_heads

        if boundary_conditions == 'periodic':
            padding_mode = 'circular'
        else:
            padding_mode = 'reflect'
        self.add_mask = boundary_conditions != 'periodic'
        bc_offset = int(self.add_mask)

        # Layers 
        self.adapter_in = make_conv(spatial_ndims, in_channels+bc_offset, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1)
        self.conv_in = nn.Sequential(
            make_conv(spatial_ndims, start, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1),
            nn.GroupNorm(norm_groups, start, affine=False),
            nn.GELU(),
        )
        self.downs = nn.ModuleList([
            Down(spatial_ndims, start * 1, start * 2, start * 2, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*2),
            Down(spatial_ndims, start * 2, start * 4, start * 4, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*4),
            Down(spatial_ndims, start * 4, start * 8, start * 8, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*8),
            Down(spatial_ndims, start * 8, start * 16, start * 16, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*16),
        ])  
        self.ups = nn.ModuleList([
            Up(spatial_ndims, start * 16, start * 8, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*8),
            Up(spatial_ndims, start * 8, start * 4, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*4),
            Up(spatial_ndims, start * 4, start * 2, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*2),
            Up(spatial_ndims, start * 2, start * 1, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*1),
        ])
        self.bottleneck = nn.Sequential(
            make_conv(spatial_ndims, start * 16, start * 32, ksize=1, stride=1, padding=0, padding_mode=padding_mode, groups=1),
            nn.GroupNorm(norm_groups, start * 32, affine=False),
            nn.GELU(),
            make_conv(spatial_ndims, start * 32, start * 16, ksize=1, stride=1, padding=0, padding_mode=padding_mode, groups=1),
        )
        self.convs_out = nn.Sequential(
            make_conv(spatial_ndims, start, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1),
            nn.GroupNorm(norm_groups, start, affine=False),
            nn.GELU(),
        )
        self.adapter_out = make_conv(spatial_ndims, start, in_channels, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1)

    def forward(self, x: Tensor) -> Tensor:

        # Early diagnostics: verify input channels match adapter expectation
        # (helps debug mismatches seen in CHTC logs)
        try:
            input_channels = x.shape[1]
        except Exception:
            input_channels = None
        conv_in_channels = getattr(self.adapter_in, 'in_channels', None)
        conv_groups = getattr(self.adapter_in, 'groups', None)
        if conv_in_channels is not None and input_channels is not None:
            expected = conv_in_channels - int(self.add_mask)
            if input_channels != expected:
                print(f"DEBUG OperatorNetwork input mismatch: adapter_in.in_channels={conv_in_channels}, add_mask={int(self.add_mask)}, adapter_in.groups={conv_groups}, adapter_in.weight.shape={tuple(getattr(self.adapter_in, 'weight').shape) if hasattr(self.adapter_in, 'weight') else None}, got input channels={input_channels}")
                raise RuntimeError(f"OperatorNetwork input channels mismatch: expected {expected} channels (adapter expects {conv_in_channels} incl. mask), but got {input_channels} channels")

        # Deal with boundary conditions
        if self.add_mask:
            mask = create_frontier_mask(x[:,0,...]).unsqueeze(1)
            x = torch.cat([x, mask], dim=1)

        # Adapt to varying number of channels
        x = self.adapter_in(x)

        # First conv
        x1 = self.conv_in(x)
        
        # Downsampling
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)

        # Bottleneck 
        x5 = x5 + self.bottleneck(x5)

        # Upsampling
        x = self.ups[0](x5, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        x = self.convs_out(x)

        # Adapt to varying number of channels
        x = self.adapter_out(x)

        return x

# ========= DISCO =========

def vectors_to_parameters(vec: Tensor, parameters_dict: Dict) -> Dict:
    """Slice a vector into parameters given by a dictionary of parameters."""
    pointer = 0
    new_parameters_dict = {}
    B = vec.size(0)
    for name, param in parameters_dict.items():
        num_param = param.numel()
        param_chunk = vec[:, pointer : pointer + num_param]
        new_parameters_dict[name] = param_chunk.view(B, *param.shape)
        pointer += num_param
    return new_parameters_dict


def functional_call_batched(model: nn.Module, parameter_vec_batch: Tensor, x_batch: Tensor) -> Tensor:
    """Call a model with a batch of parameters and a batch of inputs.
    Args:
        model: The model architecture (its parameters won't be used).
        parameter_vec_batch: The model batch parameters (B x dim_parameter)
        x_batch: The model batch inputs (B x ...)
    """
    param_dict = dict(model.named_parameters())
    batched_params_dict = vectors_to_parameters(parameter_vec_batch, param_dict)
    # functional_call will be vmapped over the batch dimension. Each per-sample
    # call expects the model input to include a batch dimension (B, C, H, W).
    # When vmap supplies a single-sample input it will have shape (C, H, W),
    # so ensure a singleton batch dimension is present for each sample and
    # remove it from the outputs before returning.
    x_batched = x_batch.unsqueeze(1)
    outs = vmap(functional_call, in_dims=(None, 0, 0))(model, batched_params_dict, x_batched)
    return outs.squeeze(1)


class DISCO(nn.Module):
    """DISCO model: a hypernetwork that outputs the weights of an operator network,
    which is then evolved via neuralPDE to predict the next state."""
    def __init__(self, 
        n_states: int,
        hidden_dim: int,
        patch_size: int,
        ndims: List[int],
        groups: int,
        processor_blocks: int,
        drop_path: float,
        num_heads: int,
        bias_type: str,
        hpnn_head_hidden_dim: int,
        dataset_names: List[str],
        max_steps: int = 32,
        atol: float = 1e-9,  # same as in torchdiffeq
        rtol: float = 5e-6,
        integration_library: str = "torchdiffeq",
    ):
        super().__init__()
        
        # Operator network (opnn): network evolved via neuralPDE solver
        self.opnns = nn.ModuleDict({
            dname: OperatorNetwork(
                in_channels=DATASET_SPECS[dname]['in_channels'], 
                start=8, 
                spatial_ndims=DATASET_SPECS[dname]['spatial_ndims'],
                boundary_conditions=DATASET_SPECS[dname]['boundary_conditions'], 
                norm_groups=4
            )
            for dname in dataset_names
        })
        self.numel_opnn_parameters = {dname: parameters_to_vector(self.opnns[dname].parameters()).numel() for dname in dataset_names}
        self.param_masks = self.make_weights_masks()  # masks indicating which weights are adimensional, dimensional, or varying-channels
        
        # Solver specifics (neuralPDE)
        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol
        self.integration_library = integration_library

        # Hypernetwork (hpnn): from input, estimate the operator network latent representation
        self.hpnn = Hypernetwork(
            patch_size=patch_size,
            n_states=n_states,
            ndims=ndims,
            embed_dim=hidden_dim,
            groups=groups,
            processor_blocks=processor_blocks,
            drop_path=drop_path,
            num_heads=num_heads,
            bias_type=bias_type,
        )

        # Parameter generator: generate the weights of the operator network from its predicted latent representation
        param_counts = {
            dname: {
                'adim': (self.param_masks[dname] == 0).sum().item(),  # weights that don't vary in shape with different inputs (e.g. fully connected layers)
                'dim': (self.param_masks[dname] == 1).sum().item(),  # weights that vary in the spatial dimension of the input (e.g. conv layers)
                'varying_channels': (self.param_masks[dname] == 2).sum().item(),  # weights that vary in the input channels (adapter in and out conv layers)
            }
            for dname in dataset_names
        }
        count_dim = {}
        for dname in dataset_names:
            dim = DATASET_SPECS[dname]['spatial_ndims']
            count_adim = param_counts[dname]['adim']
            if dim not in count_dim:
                count_dim[str(dim)] = param_counts[dname]['dim']
        self.param_gen_common = nn.Sequential(
            nn.Linear(hidden_dim, hpnn_head_hidden_dim),
            nn.GELU(),
            nn.Linear(hpnn_head_hidden_dim, hpnn_head_hidden_dim),
            nn.GELU(),
        )
        self.param_gen_adim = nn.Linear(hpnn_head_hidden_dim, count_adim)
        self.param_gen_dim = nn.ModuleDict({
            dim: nn.Linear(hpnn_head_hidden_dim, numel)
            for (dim, numel) in count_dim.items()
        })
        self.param_gen_channels = nn.ModuleDict({
            dname: nn.Linear(hpnn_head_hidden_dim, param_counts[dname]['varying_channels'])
            for dname in dataset_names
        })
        theta_norms = self.init_normalizations()  # normalization to apply to predicted weights
        for k, norm in theta_norms.items():
            self.register_buffer(f"theta_norm_{k}", norm)

    def init_normalizations(self) -> Dict[str, Tensor]:
        """ Determine the normalizations of the opnns. """
        norms = {}
        for k, opnn in self.opnns.items():
            params = dict(opnn.named_parameters())
            params_max = {k: torch.full_like(v, v.abs().max().item()) for k, v in params.items()}
            norm = parameters_to_vector(params_max.values())
            norms[k] = norm
        return norms
    
    def make_weights_masks(self) -> Dict[str, Tensor]:
        """Obtain the 'type' of each weight in the operator network.
        The type is either:
        - 0: adimensional weights (same shape for all inputs), e.g. fully connected layers
        - 1: dimensional weights (vary in the spatial dimension of the input), e.g. conv layers
        - 2: varying-channels weights (vary in the number of input channels), e.g. adapter in and out conv layers
        """
        masks = {}
        for name, opnn in self.opnns.items():
            params = dict(opnn.named_parameters())
            mask_this_name = {k: torch.ones_like(v, dtype=torch.int32) for k, v in params.items()}
            for i_layer, (k, v) in enumerate(params.items()):
                if v.shape[-1] == 1:  # indicates 1x1 convolution, so fully connected
                    mask_this_name[k].fill_(0)
                if i_layer == 0 or i_layer == len(params)-1:
                    mask_this_name[k].fill_(2)
            mask_this_name = parameters_to_vector(mask_this_name.values())
            masks[name] = mask_this_name
        return masks
        
    def forward(self, 
        x: Tensor, 
        state_labels: Tensor, 
        dset_name: str, 
        predict_normed: bool = False,
    ) -> Tensor:
        """ x is B T C H (W) (D) """
        B, _, _, *spatial = x.shape
        dim = len(spatial)
        state_labels = state_labels.unsqueeze(0).repeat(B, 1)

        # Preprocess
        spatial_dims = tuple(range(3,x.ndim))
        x, mean, std = standardize(x, dims=(1,*spatial_dims), return_stats=True)  # b t c h w
        metadata = {'mean': mean, 'std': std}

        # Keep as initial state for the solver later
        x_input = x[:,-1,...]

        # Estimate the operator latent parameters -> enforece a bottleneck on estimating the dynamics
        theta_latent = self.hpnn(x, state_labels[0])
        theta_latent = theta_latent.mean((1,3,4,5))  # average on time x space

        # Generate the operator parameters
        theta = self.param_gen_common(theta_latent)
        theta_adim = self.param_gen_adim(theta)
        theta_dim = self.param_gen_dim[str(dim)](theta)
        theta_dset = self.param_gen_channels[dset_name](theta)
        theta = torch.zeros(B, self.numel_opnn_parameters[dset_name], device=x.device, dtype=x.dtype)
        theta[:, self.param_masks[dset_name] == 0] = theta_adim
        theta[:, self.param_masks[dset_name] == 1] = theta_dim
        theta[:, self.param_masks[dset_name] == 2] = theta_dset

        # Normalize weights per layer accounting for the size of each layer
        theta_norm = getattr(self, f"theta_norm_{dset_name}")
        def signed_sigmoid(x, factor=2.0):
            return factor * (2 * torch.sigmoid(2 * x / factor) - 1)
        theta = theta * theta_norm / theta_norm.pow(2.0).mean().pow(0.5)
        theta = theta_norm * signed_sigmoid(theta / theta_norm)

        # Preprocess last step
        spatial_dims = tuple(range(2, x_input.ndim))
        x_input, mean_t, std_t = standardize(x_input, dims=spatial_dims, return_stats=True)  # b c h (w)

        # Solve
        def opnn(t, x):  # the operator that is integrated by the solver
            # x has shape [B, C, H, W] coming from the solver; OperatorNetwork
            # expects a 4D input (B, C, H, W). Avoid inserting a time dim here
            # which would cause downstream channel-count mismatches.
            x = functional_call_batched(self.opnns[dset_name], theta, x)
            return x
        options = {'min_step': 1/self.max_steps}
        t = torch.linspace(0, 1, 2, device=x.device)
        n_steps, x = odeint_adjoint(opnn, x_input, t=t, rtol=self.rtol, method="bosh3", options=options, adjoint_params=(theta,))

        # Post-process
        x = x[-1,...]
        x = x * std_t + mean_t
        x = x.unsqueeze(1)

        if predict_normed:
            x = x * metadata['std'] + metadata['mean']

        metadata['n_steps'] = n_steps
        metadata['theta'] = theta

        return x, metadata
