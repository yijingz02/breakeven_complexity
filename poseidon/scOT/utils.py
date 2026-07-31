"""Utility functions."""

import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

import copy
import math

import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm

def read_cli(parser):
    """Reads command line arguments."""

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file or JSON string",
    )
    parser.add_argument(
        "--json_config",
        action="store_true",
        help="Whether the config is a JSON string",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        required=False,
        default=None,
        help="Name of the run in wandb",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="scOT",
        help="Name of the wandb project",
    )
    parser.add_argument(
        "--max_num_train_time_steps",
        type=int,
        default=None,
        help="Maximum number of time steps to use for training and validation.",
    )
    parser.add_argument(
        "--train_time_step_size",
        type=int,
        default=None,
        help="Time step size to use for training and validation.",
    )
    parser.add_argument(
        "--train_small_time_transition",
        action="store_true",
        help="Whether to train only for next step prediction.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base path to data.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory. Will be prepended by wandb project and run name.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Whether to disable tqdm progress bar",
    )
    parser.add_argument(
        "--push_to_hf_hub",
        type=str,
        default=None,
        help="Whether to push the model to Huggingface Hub. Specify the model repository name.",
    )
    parser.add_argument(
        "--just_velocities",
        action="store_true",
        help="Whether to only use velocities as input. Only relevant for incompressible flow datasets.",
    )
    parser.add_argument(
        "--move_data",
        type=str,
        default=None,
        help="If set, moves the data to this directory and trains from there.",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        required=True,
        help="Training time budget in seconds.",
    )
    parser.add_argument(
        "--ndata",
        type=int,
        required=True,
        help="Number of trajactories for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Training batch size.",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        required=False,
        help="Batch size for validation and inference.",
    )
    parser.add_argument(
        "--do_inference",
        type=bool,
        default=False,
        required=False,
        help="Whether to only do inference.",
    )
    return parser


def get_num_parameters(model):
    """Returns the number of trainable parameters in a model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_parameters_no_embed(model):
    """Returns the number of trainable parameters in a scOT model without embedding and recovery."""
    out = 0
    for name, p in model.named_parameters():
        if not ("embeddings" in name or "patch_recovery" in name) and p.requires_grad:
            out += p.numel()
    return out

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
