""" Main training script, inspired from: 
https://github.com/anonymous/multiple_physics_pretraining/blob/main/train_basic.py """
import argparse
import math
import os
import sys
sys.path.append("..")
import time
from pathlib import Path
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict
import wandb
import gc
from torchinfo import summary

from src.models import build_model
from src.utils.data_utils import get_data_objects, DATASET_SPECS
from src.utils import is_debug, YParams, logging_utils, TimeTracker


def nrmse(y_true, y_pred, dims=(-1,)):
    """ Normalized root mean squared error. """
    residual = y_true - y_pred
    mse = residual.pow(2).mean(dims, keepdim=True)
    norm = 1e-7 + y_true.pow(2).mean(dims, keepdim=True)
    return (mse / norm).sqrt()


class LpLoss(object):
    """Lp-norm loss with proper size averaging.
    
    Calculates Lp norms with optional mesh scaling and supports both absolute and relative metrics.
    Follows the pattern from: https://github.com/zongyi-li/fourier_neural_operator
    
    Args:
        d (int): Spatial dimension for mesh scaling h^(d/p). Default: 2
        p (int or float): Lp norm type (1, 2, inf, etc). Default: 2 (L2)
        size_average (bool): If True, returns mean over batch; else sum. Default: True
        reduction (bool): If True, reduces over batch; else returns per-sample. Default: True
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """Absolute Lp norm with mesh scaling h^(d/p)."""
        num_examples = x.size()[0]
        # Assume uniform mesh: h = 1 / (grid_size - 1)
        h = 1.0 / (x.size()[1] - 1.0)
        # Flatten spatial/temporal dims and compute Lp norm per sample
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        """Relative Lp norm: ||x-y||_p / ||y||_p, with size averaging."""
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        relative_norms = diff_norms / (y_norms.clamp_min(1e-7))
        
        if self.reduction:
            if self.size_average:
                return torch.mean(relative_norms)
            else:
                return torch.sum(relative_norms)
        return relative_norms


def add_weight_decay(params, weight_decay=1e-5, skip_list=()):
    """ From Ross Wightman at:
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3 
    
    Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale) 
    then don't apply weight decay. 
    """
    decay = []
    no_decay = []
    for name, param in params:
        if not param.requires_grad:
            continue
        if (len(param.squeeze().shape) <= 1 or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.,},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**.5


def grad_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.pow(2).sum().item()
        return total_norm**.5


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB


def model_rollout(
    model: nn.Module,
    x: torch.Tensor,
    predict_normed: bool = False,
    n_future_steps: int = 1,
    state_labels: torch.Tensor | None = None,
    dset_name: str | None = None,
):
    """ x is B T C H W """
    # first iteration 
    out, metadata = model(x, predict_normed=predict_normed, state_labels=state_labels, dset_name=dset_name)
    if n_future_steps > 1:
        # more iterations: rollouts
        context = x.clone()
        outputs = [out]
        for _ in range(n_future_steps-1):
            context = torch.cat([context[:,1:,...], out], dim=1)
            out, _ = model(context, predict_normed=predict_normed, state_labels=state_labels, dset_name=dset_name)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)
    return out, metadata


class Trainer:

    def __init__(self, params, global_rank, local_rank, device, use_autoregressive=False):
        self.device = device
        self.params = params
        self.use_autoregressive = use_autoregressive
        self.base_dtype = torch.float
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.log_to_screen = params.log_to_screen

        # Basic setup
        self.start_epoch = 0
        self.epoch = 0
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.mp_type = torch.bfloat16
        else:
            self.mp_type = torch.half
        self.true_time = params.true_time
        self.inference_batch_size = int(getattr(params, 'inference_batch_size', params.batch_size) or params.batch_size)
        test_max_batches_cfg = getattr(params, 'test_max_batches', None)
        self.test_max_batches = int(test_max_batches_cfg) if test_max_batches_cfg is not None else None
        test_max_samples_gs_cfg = getattr(params, 'test_max_samples_grayscott', None)
        self.test_max_samples_grayscott = int(test_max_samples_gs_cfg) if test_max_samples_gs_cfg is not None else None
        self._amp_eval_fallback_warned = False

        self.iters = 0
        self.train_compute_time_total = 0.0
        self.train_compute_time_budget = 0.0
        self.optimizer_steps_completed = 0
        self.scheduler_reestimated = False
        self.scheduler_total_steps = None
        self._scheduler_state_to_restore = None
        self.warmup_steps_budget = int(getattr(params, 'timing_warmup_steps', 5) or 0)
        avg_steps_cfg = int(getattr(params, 'timing_avg_steps', 3) or 3)
        self.timing_avg_steps = min(5, max(3, avg_steps_cfg))
        self.post_warmup_step_times = []
        self.should_stop_training = False
        self.stop_reason = None
        self.run_signature = self._build_run_signature()
        self.initialize_data(self.params)
        self.initialize_model(self.params)
        self.initialize_optimizer(self.params)
        if params.resuming:
            print("Loading checkpoint %s"%params.checkpoint_path)
            print('LOADING CHECKPOINTTTTTT')
            self.restore_checkpoint(params.checkpoint_path)  # no finetuning in that case
        if not params.resuming and params.pretrained:  # finetuning
            print("Starting from pretrained model at %s"%params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.start_epoch = 0
        self.initialize_scheduler(self.params)

    def _build_run_signature(self):
        """Signature used to ensure checkpoint resume matches current run configuration."""
        train_datasets = tuple(getattr(self.params, 'train_datasets', []))
        valid_datasets = tuple(getattr(self.params, 'valid_datasets', []))
        test_datasets = tuple(getattr(self.params, 'test_datasets', []) or [])
        return {
            'run_name': str(getattr(self.params, 'run_name', '')),
            'yaml_config': str(getattr(self.params, 'yaml_config', '')),
            'train_datasets': train_datasets,
            'valid_datasets': valid_datasets,
            'test_datasets': test_datasets,
            'ntrain': int(getattr(self.params, 'ntrain', -1) or -1),
            'time_budget': float(getattr(self.params, 'time_budget', -1.0) or -1.0),
            'checkpoint_dir': str(getattr(self.params, 'checkpoint_dir', '')),
        }

    def initialize_data(self, params):

        if len(set(DATASET_SPECS[dname]['group'] for dname in params.train_datasets + params.valid_datasets)) > 1:
            # this would require adapting the way the field labels are determined
            raise ValueError("Cannot mix datasets from PDEBench and the_well")
                             
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")
        self.train_dataset, self.train_sampler, self.train_data_loader = get_data_objects(
            params.train_datasets, params.batch_size, params.epoch_size, params.train_val_test,
            params.n_past, params.n_future, dist.is_initialized(), params.num_data_workers,
            rank=self.global_rank, split='train', 
        )
        self.valid_dataset, _, _ = get_data_objects(
            params.valid_datasets, self.inference_batch_size, params.epoch_size, params.train_val_test,
            params.n_past, params.val_rollout, dist.is_initialized(), params.num_data_workers,
            rank=self.global_rank, split='valid', 
        )
        if hasattr(params, 'test_datasets') and params.test_datasets:
            test_rollout = getattr(params, 'test_rollout', params.val_rollout)
            self.test_dataset, _, _ = get_data_objects(
                params.test_datasets, self.inference_batch_size, params.epoch_size, params.train_val_test,
                params.n_past, test_rollout, dist.is_initialized(), params.num_data_workers,
                rank=self.global_rank, split='test',
            )
        else:
            self.test_dataset = None
        self.train_sampler.set_epoch(0)

    def initialize_model(self, params):

        print(f"Initializing model on rank {self.global_rank}")
        
        self.model = build_model(params).to(device=self.device)

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank], 
                output_device=[self.local_rank], find_unused_parameters=True,
            )

        n_params = sum([p.numel() for p in self.model.parameters()])
        
        self.single_print(f'Model parameter count: {n_params:,}')
        if self.params.model == "metaop" and ('finetune' not in self.params or not self.params.finetune):
            if dist.is_initialized():
                params_count = {k: opnn.weight_count() for k,opnn in self.model.module.opnn.items()}
            else:
                params_count = {k: opnn.weight_count() for k,opnn in self.model.opnn.items()}
            self.single_print(f"Operator class params: {params_count}")

    def initialize_optimizer(self, params): 

        parameters_standard = self.model.named_parameters()
        parameters = add_weight_decay(parameters_standard, self.params.weight_decay) # Dont use weight decay on bias/scaling terms
        if params.optimizer == 'adam':
            if self.params.learning_rate < 0:
                self.optimizer =  DAdaptAdam(parameters, lr=1., growth_rate=1.05, log_every=100, decouple=True)
            else:
                self.optimizer = optim.AdamW(parameters, lr=params.learning_rate)
        elif params.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=params.learning_rate, momentum=0.9)
        else: 
            raise ValueError(f"Optimizer {params.optimizer} not supported")
        self.gscaler = torch.amp.GradScaler(enabled=(self.mp_type == torch.half and params.enable_amp))

    def initialize_scheduler(self, params):

        self.scheduler = None

        # If resuming and scheduler was already re-estimated, recreate it with proper state
        if self.params.resuming and self.scheduler_reestimated and self.scheduler_total_steps is not None:
            if self.log_to_screen:
                print(f"Recreating re-estimated scheduler: total_steps={self.scheduler_total_steps}, completed_steps={self.optimizer_steps_completed}")
            remaining_steps = max(1, self.scheduler_total_steps - self.optimizer_steps_completed)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                eta_min=max(0, params.learning_rate/100),
                T_max=remaining_steps,
                last_epoch=-1
            )
            if self._scheduler_state_to_restore is not None:
                self.scheduler.load_state_dict(self._scheduler_state_to_restore)
            return

        if params.scheduler == 'cosine':
            
            last_step = (self.start_epoch * params.epoch_size) // params.accum_grad - 1
            warmup_steps = (params.warmup_epochs * params.epoch_size) // params.accum_grad
            total_steps = (params.max_epochs * params.epoch_size) // params.accum_grad

            if warmup_steps > 0 and params.learning_rate > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=warmup_steps)
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=params.learning_rate/100, T_max=total_steps-warmup_steps)
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, decay], [warmup_steps])
                for _ in range(last_step+1):  # sequentialLR: need to manually step through the last_step
                    self.scheduler.step()
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=max(0,params.learning_rate/100), T_max=total_steps, last_epoch=last_step)

        if self.scheduler is not None and self._scheduler_state_to_restore is not None:
            self.scheduler.load_state_dict(self._scheduler_state_to_restore)

    def reset_scheduler_with_estimated_steps(self, estimated_total_steps):

        if self.scheduler is None or self.params.scheduler != 'cosine':
            return

        estimated_total_steps = max(1, int(estimated_total_steps))
        remaining_steps = max(1, estimated_total_steps - self.optimizer_steps_completed)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            eta_min=max(0, self.params.learning_rate/100),
            T_max=remaining_steps,
            last_epoch=-1
        )

        self.scheduler_total_steps = estimated_total_steps
        self.scheduler_reestimated = True

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(' '.join([str(t) for t in text]))

    def _safe_eval_rollout(self, inp, state_labels, dset_name, n_future_steps):
        try:
            return model_rollout(
                self.model, inp, predict_normed=True,
                n_future_steps=n_future_steps,
                state_labels=state_labels,
                dset_name=dset_name,
            )
        except (RuntimeError, AssertionError) as err:
            err_msg = str(err)
            if (not self.params.enable_amp) or ('non-finite values in state `y`' not in err_msg):
                raise
            if not self._amp_eval_fallback_warned:
                self.single_print('AMP eval produced non-finite ODE state; retrying failed eval batches in FP32.')
                self._amp_eval_fallback_warned = True
            with torch.amp.autocast("cuda", enabled=False, dtype=self.mp_type):
                return model_rollout(
                    self.model, inp.float(), predict_normed=True,
                    n_future_steps=n_future_steps,
                    state_labels=state_labels,
                    dset_name=dset_name,
                )

    def restore_checkpoint(self, checkpoint_path):
        """ Load model/opt from path """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        if self.params.resuming:
            ckpt_sig = checkpoint.get('run_signature', None)
            if ckpt_sig is not None and ckpt_sig != self.run_signature:
                raise RuntimeError(
                    "Checkpoint/run mismatch while resuming.\n"
                    f"Current run signature: {self.run_signature}\n"
                    f"Checkpoint signature: {ckpt_sig}\n"
                    "Use a separate --checkpoint_dir for this run or remove old ckpt.tar."
                )
        new_state_dict = OrderedDict()
        new_state_dict_place_holder = self.model.state_dict()
        pretrained = self.params.pretrained
        model_type = self.params.model
        skipped_ckpt_keys = 0
        for key, val in checkpoint['model_state'].items():
            current_name = new_name = key
            if not self.params.use_ddp:
                if key.startswith('module.'):
                    current_name = new_name = key[7:]  # remove the "module." prefix
                else:
                    current_name = new_name = key
            if pretrained:  
                # for pretarined models, some useless parameters must be removed and some must be kept fresh
                # remove: basically the weights related to 1d contexts (finetuning is done on 2d only)
                # keep fresh: the weights related to the fields-specific weights in the finetuning dataset
                if model_type == "disco":
                    filter_remove = ['burgers', 'diffre', 'swe', 'compNS', 'hpnn.encoder.1', 'proj_dim_variant_param.1']
                    filter_keep_fresh = ['space_bag.bias', 'space_bag.weight']
                if any([f in key for f in filter_remove]):
                    continue
                elif any([f in key for f in filter_keep_fresh]):
                    new_state_dict[new_name] = new_state_dict_place_holder[new_name]
                    continue
            if current_name not in new_state_dict_place_holder:
                skipped_ckpt_keys += 1
                continue
            if new_state_dict_place_holder[current_name].shape != val.shape:
                assert new_state_dict_place_holder[current_name].shape[:-1] == val.shape
                new_state_dict[new_name] = val.unsqueeze(-1)  # correct an incompatibility issue when the code was extended to 3d
            else:
                new_state_dict[new_name] = val
        if skipped_ckpt_keys > 0 and self.log_to_screen and self.global_rank == 0:
            print(f"WARNING: Skipped {skipped_ckpt_keys} checkpoint keys not present in current model.")
        if pretrained and model_type in ["disco"]:
            for key, val in new_state_dict_place_holder.items():
                if 'euler' in key:  # for fietuning, keep the fresh fields-specific weights
                    new_state_dict[key] = val
        self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint.get('iters', checkpoint.get('epoch', 0) * self.params.epoch_size)
        if self.params.resuming:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
            legacy_timing_missing = ('train_compute_time_budget' not in checkpoint)
            self._scheduler_state_to_restore = checkpoint.get('scheduler_state_dict', None)
            # Restore training state variables
            self.optimizer_steps_completed = checkpoint.get('optimizer_steps_completed', 0)
            self.train_compute_time_total = checkpoint.get('train_compute_time_total', 0.0)
            self.train_compute_time_budget = checkpoint.get('train_compute_time_budget', 0.0)
            self.scheduler_reestimated = checkpoint.get('scheduler_reestimated', False)
            self.scheduler_total_steps = checkpoint.get('scheduler_total_steps', None)
            self.post_warmup_step_times = checkpoint.get('post_warmup_step_times', [])
            self.warmup_steps_budget = checkpoint.get('warmup_steps_budget', self.warmup_steps_budget)
            self.timing_avg_steps = checkpoint.get('timing_avg_steps', self.timing_avg_steps)
            if self.log_to_screen:
                print(f"Resumed from checkpoint: epoch={self.epoch}, opt_steps={self.optimizer_steps_completed}, "
                      f"compute_budget={self.train_compute_time_budget:.1f}s, scheduler_reestimated={self.scheduler_reestimated}")
                if legacy_timing_missing:
                    print("WARNING: Legacy checkpoint missing compute-time state; time/compute_total may restart from 0 after resume.")
                if self._scheduler_state_to_restore is None:
                    print("WARNING: Legacy checkpoint missing scheduler state; LR may show a one-time jump after resume.")
        checkpoint = None

    def save_checkpoint(self, checkpoint_path, model=None):
        """ Save model, optimizer, and training state to checkpoint """
        if not model:
            model = self.model
        d = {
            'iters': self.iters,
            'epoch': self.epoch, 
            'model_state': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'run_signature': self.run_signature,
            'optimizer_steps_completed': self.optimizer_steps_completed,
            'train_compute_time_total': self.train_compute_time_total,
            'train_compute_time_budget': self.train_compute_time_budget,
            'scheduler_reestimated': self.scheduler_reestimated,
            'scheduler_total_steps': self.scheduler_total_steps,
            'post_warmup_step_times': self.post_warmup_step_times,
            'warmup_steps_budget': self.warmup_steps_budget,
            'timing_avg_steps': self.timing_avg_steps
        }
        torch.save(d, checkpoint_path)

    @staticmethod
    def mse_loss(y_ref, y):
        """ y and y_ref are B T C H (W) (D) """
        spatial_dims = tuple(range(3,y.ndim))
        var = 1e-7 + y_ref.var(spatial_dims, keepdim=True)
        loss = F.gaussian_nll_loss(y, y_ref, torch.ones_like(y)*var, eps=1e-8, reduction='mean')
        with torch.no_grad():
            residual = y - y_ref
            norm_ref = 1e-7 + y_ref.pow(2).mean(spatial_dims, keepdim=True)
            raw_loss = residual.pow(2.0).mean(spatial_dims, keepdims=True) / norm_ref
        return loss, raw_loss

    def train_one_epoch(self):
        tt = TimeTracker()
        tt.track("data", "training")
        self.epoch += 1
        self.model.train()
        logs = {
            'train_nrmse': torch.zeros(1).to(self.device, dtype=self.base_dtype),
            'train_l1': torch.zeros(1).to(self.device, dtype=self.base_dtype),
        }
        steps = 0
        grad_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        grad_counts = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        theta_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        steps_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        loss_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        loss_counts = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        dset_counts = {}
        epoch_samples_seen = 0
        cycle_compute_time = 0.0
        budget_enabled = hasattr(self.params, 'time_budget') and self.params.time_budget is not None
        self.single_print("--------")
        if hasattr(self.params, 'ntrain') and self.params.ntrain is not None:
            effective_samples = min(len(self.train_dataset), int(self.params.ntrain))
            self.single_print('train_loader_size', len(self.train_data_loader), len(self.train_dataset),
                              'effective_ntrain_samples', effective_samples)
        else:
            self.single_print('train_loader_size', len(self.train_data_loader), len(self.train_dataset))

        for batch_idx, batch in enumerate(self.train_data_loader):
            if batch_idx >= self.params.epoch_size:  # certain dataloaders are not restricted in length
                break
            if budget_enabled and self.train_compute_time_budget >= self.params.time_budget:
                self.should_stop_training = True
                self.stop_reason = 'time_budget'
                break
            steps += 1
            tt.track("data", "training", "data_batch")
            if self.true_time:
                torch.cuda.synchronize()

            inp, tar = batch['input_fields'].to(device=self.device), batch['output_fields'].to(device=self.device)
            mask = None
            if 'mask' in batch:
                mask = batch['mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.to(device=self.device).float()
                else:
                    mask = torch.from_numpy(mask).to(device=self.device).float()
                # mask expected shape [B, H, W] -> add channel and time dims for broadcasting
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.ndim == 4:
                    mask = mask.unsqueeze(1)
            dset_name = batch['name'][0]
            state_labels = batch['field_labels'].to(device=self.device)

            if hasattr(self.params, 'ntrain') and self.params.ntrain is not None:
                samples_remaining = int(self.params.ntrain) - epoch_samples_seen
                if samples_remaining <= 0:
                    break
                per_rank_remaining = max(1, int(math.ceil(samples_remaining / max(1, self.world_size))))
                if inp.shape[0] > per_rank_remaining:
                    inp = inp[:per_rank_remaining]
                    tar = tar[:per_rank_remaining]
                    state_labels = state_labels[:per_rank_remaining]

            epoch_samples_seen += inp.shape[0] * max(1, self.world_size)

            dset_counts[dset_name] = dset_counts.get(dset_name, 0) + 1
            loss_counts[dset_name] += 1

            # whether the model weights should be updated this batch
            self.model.require_backward_grad_sync = ((1+batch_idx) % self.params.accum_grad == 0)
            if budget_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=self.params.enable_amp, dtype=self.mp_type):
                
                # forward
                tt.track("forward", "training", "forw_batch")
                if self.use_autoregressive:
                    # Use autoregressive rollout for training
                    output, metadata = model_rollout(
                        self.model,
                        inp, predict_normed=False,
                        n_future_steps=self.params.n_future,
                        state_labels=state_labels[0],
                        dset_name=dset_name,
                    )
                else:
                    # Standard teacher forcing (single forward pass)
                    output, metadata = self.model(
                        inp, predict_normed=False, 
                        # n_future_steps=self.params.n_future, 
                        state_labels=state_labels[0],
                        dset_name=dset_name,
                    )
                tar = (tar - metadata['mean']) / metadata['std']  # normalize tar

                # Apply mask if present (mask shape [B,1,1,H,W])
                if mask is not None:
                    tar = tar * mask
                    output = output * mask

                # loss
                tt.track("loss", "training", "loss_batch")
                loss, loss_raw = self.mse_loss(tar, output)
                loss = loss / self.params.accum_grad

                # logs
                tt.track("logs", "training")
                with torch.no_grad():
                    log_nrmse = loss_raw.sqrt().mean()
                    logs['train_nrmse'] += log_nrmse
                    loss_logs[dset_name] += log_nrmse
                    loss_print = log_nrmse.item()
                    if 'theta' in metadata:
                        theta_logs[dset_name] += metadata['theta'].abs().mean()
                    if 'n_steps' in metadata:
                        steps_logs[dset_name] += metadata['n_steps']
                
                # backward
                tt.track("backward", "training", "back_batch")
                loss.backward()

                if self.true_time:
                    torch.cuda.synchronize()

                # gradient step
                tt.track("gradient_step", "training", "optim_batch")
                if self.model.require_backward_grad_sync: # Only take step once per accumulation cycle
                    grad_logs[dset_name] += grad_norm(self.model.parameters())
                    grad_counts[dset_name] += 1
                    # clip the gradients 
                    if self.params.gnorm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.params.gnorm)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        if self.scheduler_total_steps is None or self.optimizer_steps_completed < self.scheduler_total_steps:
                            self.scheduler.step()
                    if self.true_time:
                        torch.cuda.synchronize()

                # logs
                if self.true_time:
                    torch.cuda.synchronize()
                if self.log_to_screen and batch_idx % self.params.log_interval == 0 and self.global_rank == 0:
                    current_mem_GB = torch.cuda.memory_allocated() / 1024**3
                    max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
                    reserved_mem_GB = torch.cuda.memory_reserved() / 1024**3
                    current_CPU_mem_GB = get_memory_usage()
                    print(f"Epoch {self.epoch} Batch {batch_idx} Train Loss {loss_print:.3f}")
                    print('Total Times. Batch: {}, Rank: {}, Data Shape: {}, Data time: {:.2f}, Forward: {:.2f}, Backward: {:.2f}, Optimizer: {:.2f}'.format(
                        batch_idx, self.global_rank, list(inp.shape), tt.get("data_batch"), tt.get("forw_batch"), tt.get("back_batch"), tt.get("optim_batch"))
                    )
                    print(f"Memory: CPU {current_CPU_mem_GB:5.2f} GB, Current {current_mem_GB:5.2f} GB, Max {max_mem_GB:5.2f} GB, Reserved {reserved_mem_GB:5.2f} GB")
                tt.reset("data_batch", "forw_batch", "back_batch", "loss_batch", "optim_batch")

            if budget_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_elapsed = time.perf_counter() - compute_start
            self.train_compute_time_total += compute_elapsed
            cycle_compute_time += compute_elapsed

            if self.global_rank == 0 and getattr(self.params, 'save_checkpoint', False):
                global_step = self.iters + steps
                if global_step > 0 and global_step % 500 == 0:
                    step_ckpt = Path(self.params.checkpoint_dir) / f'ckpt_step{global_step}.tar'
                    self.save_checkpoint(str(step_ckpt))

            if self.model.require_backward_grad_sync:
                self.optimizer_steps_completed += 1

                if self.optimizer_steps_completed > self.warmup_steps_budget:
                    self.train_compute_time_budget += cycle_compute_time

                    if budget_enabled and (not self.scheduler_reestimated):
                        self.post_warmup_step_times.append(cycle_compute_time)
                        if len(self.post_warmup_step_times) >= self.timing_avg_steps:
                            avg_step_time = sum(self.post_warmup_step_times) / len(self.post_warmup_step_times)
                            avg_step_time = max(avg_step_time, 1e-8)
                            estimated_total_steps = max(
                                self.optimizer_steps_completed + 1,
                                int(self.params.time_budget / avg_step_time)
                            )
                            self.reset_scheduler_with_estimated_steps(estimated_total_steps)
                            self.single_print(
                                f"Scheduler re-estimated: warmup_steps={self.warmup_steps_budget}, "
                                f"avg_step_time={avg_step_time:.4f}s (over {len(self.post_warmup_step_times)} steps), "
                                f"estimated_total_steps={estimated_total_steps}"
                            )

                cycle_compute_time = 0.0

                if budget_enabled and self.train_compute_time_budget >= self.params.time_budget:
                    self.should_stop_training = True
                    self.stop_reason = 'time_budget'
                    break

        # Flush leftover micro-batches when steps are not divisible by accum_grad
        if steps > 0 and (steps % self.params.accum_grad) != 0:
            if self.params.gnorm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.params.gnorm)
            self.gscaler.step(self.optimizer)
            self.gscaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                if self.scheduler_total_steps is None or self.optimizer_steps_completed < self.scheduler_total_steps:
                    self.scheduler.step()

            self.optimizer_steps_completed += 1

            if self.optimizer_steps_completed > self.warmup_steps_budget:
                self.train_compute_time_budget += cycle_compute_time

                if budget_enabled and (not self.scheduler_reestimated):
                    self.post_warmup_step_times.append(cycle_compute_time)
                    if len(self.post_warmup_step_times) >= self.timing_avg_steps:
                        avg_step_time = sum(self.post_warmup_step_times) / len(self.post_warmup_step_times)
                        avg_step_time = max(avg_step_time, 1e-8)
                        estimated_total_steps = max(
                            self.optimizer_steps_completed + 1,
                            int(self.params.time_budget / avg_step_time)
                        )
                        self.reset_scheduler_with_estimated_steps(estimated_total_steps)
                        self.single_print(
                            f"Scheduler re-estimated: warmup_steps={self.warmup_steps_budget}, "
                            f"avg_step_time={avg_step_time:.4f}s (over {len(self.post_warmup_step_times)} steps), "
                            f"estimated_total_steps={estimated_total_steps}"
                        )

            cycle_compute_time = 0.0

            if budget_enabled and self.train_compute_time_budget >= self.params.time_budget:
                self.should_stop_training = True
                self.stop_reason = 'time_budget'

        # logs
        if steps == 0:
            logs = {
                'train_nrmse': torch.zeros(1).to(self.device, dtype=self.base_dtype),
                'train_l1': torch.zeros(1).to(self.device, dtype=self.base_dtype),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'time/compute_total': self.train_compute_time_budget,
                'time/compute_total_with_warmup': self.train_compute_time_total,
                'time/compute_budget_used': self.train_compute_time_budget,
                'optimizer_steps_completed': float(self.optimizer_steps_completed),
                'epoch_steps': 0.0,
            }
            if self.scheduler_total_steps is not None:
                logs['scheduler/estimated_total_steps'] = float(self.scheduler_total_steps)
            if self.global_rank == 0:
                logs['iters'] = self.iters
            return 0.0, 0.0, logs

        logs = {k: v/steps for k, v in logs.items()}
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach()) # Mike: there was a bug with means when I originally implemented this - dont know if fixed
                logs[key] = float(logs[key]/dist.get_world_size())
            for key in sorted(loss_logs.keys()):
                dist.all_reduce(loss_logs[key].detach())
            for key in sorted(grad_logs.keys()):
                dist.all_reduce(grad_logs[key].detach())
            for key in sorted(theta_logs.keys()):
                dist.all_reduce(theta_logs[key].detach())
            for key in sorted(steps_logs.keys()):
                dist.all_reduce(steps_logs[key].detach())
            for key in sorted(loss_counts.keys()):
                dist.all_reduce(loss_counts[key].detach())
            for key in sorted(grad_counts.keys()):
                dist.all_reduce(grad_counts[key].detach())
        logs['learning_rate'] = self.optimizer.param_groups[0]['lr']

        times = tt.stop()
        for k in times:
            logs[f'time/{k}'] = times[k] / steps

        for key in loss_logs.keys():
            logs[f'{key}/train_nrmse'] = loss_logs[key] / loss_counts[key]
        for key in grad_logs.keys():
            logs[f'{key}/train_grad_norm'] = grad_logs[key] / grad_counts[key]
        for key in theta_logs.keys():
            logs[f'{key}/train_theta_norm'] = theta_logs[key] / loss_counts[key]
        for key in steps_logs.keys():
            logs[f'{key}/train_steps'] = steps_logs[key] / loss_counts[key]
        self.iters += steps
        logs['time/compute_total'] = self.train_compute_time_budget
        logs['time/compute_total_with_warmup'] = self.train_compute_time_total
        logs['time/compute_budget_used'] = self.train_compute_time_budget
        logs['optimizer_steps_completed'] = float(self.optimizer_steps_completed)
        logs['epoch_steps'] = float(steps)
        if self.scheduler_total_steps is not None:
            logs['scheduler/estimated_total_steps'] = float(self.scheduler_total_steps)
        if self.global_rank == 0:
            logs['iters'] = self.iters
            logs['parameter norm'] = param_norm(self.model.parameters())
        self.single_print('all reduces executed!')

        return times["training"], times["data"], logs

    def validate_one_epoch(self, cutoff):
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD. 
        """
        # Don't bother with full validation set between epochs
        self.model.eval()
        self.single_print('STARTING VALIDATION')
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.params.enable_amp, dtype=self.mp_type):
                sub_dsets = self.valid_dataset.sub_dsets
                logs = {}
                distinct_dsets = [dset.dataset_name for dset in sub_dsets]
                counts = {dset: 0 for dset in distinct_dsets}
                # iterate over the validation datasets
                for subset in sub_dsets:
                    dset_name = subset.dataset_name
                    # validation dataloser: technically shuffled but the shuffling order will always be the same
                    if self.params.use_ddp:
                        val_loader = torch.utils.data.DataLoader(
                            subset, batch_size=self.inference_batch_size,
                            num_workers=self.params.num_data_workers,
                            sampler=torch.utils.data.distributed.DistributedSampler(subset, drop_last=True)
                        )
                    else:
                        val_loader = torch.utils.data.DataLoader(
                            subset, batch_size=self.inference_batch_size,
                            num_workers=self.params.num_data_workers, 
                            shuffle=True, generator=torch.Generator().manual_seed(0), drop_last=True
                        )
                    count = 0
                    for batch_idx, batch in enumerate(val_loader):
                        # Only do a few batches of each dataset if not doing full validation
                        if count >= cutoff:  # validating on burgers equations is extremely long
                            del(val_loader)
                            break
                        count += 1
                        counts[dset_name] += 1

                        inp, tar = batch['input_fields'].to(self.device), batch['output_fields'].to(self.device)
                        n_past, n_future_val = self.params.n_past, self.params.val_rollout
                        if inp.shape[1] > n_past:
                            # indicates we need to split the input 
                            tar = torch.cat([inp[:,n_past:,...], tar], dim=1)
                            inp = inp[:,:n_past,...]

                        state_labels = torch.tensor(
                            self.train_dataset.subset_dict.get(subset.get_name(), [-1]*len(self.valid_dataset.subset_dict[subset.get_name()])),
                            device=self.device
                        ).unsqueeze(0)

                        # forward
                        output, _ = self._safe_eval_rollout(
                            inp,
                            state_labels[0],
                            dset_name,
                            min(self.params.val_rollout, tar.shape[1]),
                        )

                        # loss
                        residuals = output - tar
                        spatial_dims = tuple(range(residuals.ndim))[3:] # Assume 0, 1 are B, C
                        val_nrmse = nrmse(tar, output, dims=spatial_dims)
                        for step in [1,2,4,8]:
                            if step <= tar.shape[1]:
                                logs[f'{dset_name}/valid_nrmse_t{step}'] = logs.get(f'{dset_name}/valid_nrmse_t{step}',0) + val_nrmse[:,step-1,...].mean()

            self.single_print('DONE VALIDATING - NOW SYNCING')
            
            # divide by number of batches
            for k, v in logs.items():
                dset_name = k.split('/')[0]
                logs[k] = v / counts[dset_name]

            # # replace keys <>
            # average nrmse across datasets
            logs['valid_nrmse'] = 0
            for dset_name in distinct_dsets:
                logs['valid_nrmse'] += logs[f'{dset_name}/valid_nrmse_t1']/len(distinct_dsets)

            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach()) # Mike: There was a bug with means when I implemented this - dont know if fixed
                    logs[key] = float(logs[key].item()/dist.get_world_size())
                    if 'rmse' in key:
                        logs[key] = logs[key]
            self.single_print('DONE SYNCING - NOW LOGGING')
        return logs

    def test_one_epoch(self):
        """Evaluate on test split and report aggregate + worst-case nRMSE."""
        if self.test_dataset is None:
            return {}

        self.model.eval()
        self.single_print('STARTING TEST EVALUATION')
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.params.enable_amp, dtype=self.mp_type):
                sub_dsets = self.test_dataset.sub_dsets
                logs = {}
                distinct_dsets = [dset.dataset_name for dset in sub_dsets]
                counts = {dset: torch.zeros(1, device=self.device, dtype=self.base_dtype) for dset in distinct_dsets}
                total_inference_wallclock = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                total_inference_samples = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                local_worst_nrmse = torch.full((1,), -float('inf'), device=self.device, dtype=self.base_dtype)
                local_worst_idx = torch.full((1,), -1.0, device=self.device, dtype=self.base_dtype)

                # Pre-initialize keys so all ranks execute identical collective ops
                eval_steps = [1, 2, 4, 8, 16, 32, 64, 128]
                sample_sum_full = {dset_name: torch.zeros(1, device=self.device, dtype=self.base_dtype) for dset_name in distinct_dsets}
                sample_sum_step = {dset_name: torch.zeros(1, device=self.device, dtype=self.base_dtype) for dset_name in distinct_dsets}
                sample_counts = {dset_name: torch.zeros(1, device=self.device, dtype=self.base_dtype) for dset_name in distinct_dsets}
                for dset_name in distinct_dsets:
                    logs[f'{dset_name}/test_nrmse_rollout'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                    logs[f'{dset_name}/test_nrmse_step'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                    logs[f'{dset_name}/test_relative_l2_rollout'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                    logs[f'{dset_name}/test_nrmse_tlast'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                    for step in eval_steps:
                        logs[f'{dset_name}/test_nrmse_t{step}'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)

                for subset in sub_dsets:
                    dset_name = subset.dataset_name
                    max_samples_for_subset = self.test_max_samples_grayscott if dset_name == 'grayscott' else None
                    processed_samples = 0
                    if self.params.use_ddp:
                        test_loader = torch.utils.data.DataLoader(
                            subset, batch_size=self.inference_batch_size,
                            num_workers=self.params.num_data_workers,
                            sampler=torch.utils.data.distributed.DistributedSampler(subset, drop_last=False)
                        )
                    else:
                        test_loader = torch.utils.data.DataLoader(
                            subset, batch_size=self.inference_batch_size,
                            num_workers=self.params.num_data_workers,
                            shuffle=False, drop_last=False
                        )

                    test_start = time.perf_counter()
                    self.single_print(
                        f"TEST {dset_name}: batches={len(test_loader)}, "
                        f"max_batches={self.test_max_batches if self.test_max_batches is not None else 'all'}, "
                        f"max_samples={max_samples_for_subset if max_samples_for_subset is not None else 'all'}"
                    )

                    for batch_idx, batch in enumerate(test_loader):
                        if self.test_max_batches is not None and batch_idx >= self.test_max_batches:
                            break
                        if max_samples_for_subset is not None and processed_samples >= max_samples_for_subset:
                            break
                        counts[dset_name] += 1
                        inp, tar = batch['input_fields'].to(self.device), batch['output_fields'].to(self.device)
                        if max_samples_for_subset is not None:
                            remaining_samples = max_samples_for_subset - processed_samples
                            if remaining_samples <= 0:
                                break
                            if inp.shape[0] > remaining_samples:
                                inp = inp[:remaining_samples, ...]
                                tar = tar[:remaining_samples, ...]
                        n_past = self.params.n_past
                        if inp.shape[1] > n_past:
                            tar = torch.cat([inp[:, n_past:, ...], tar], dim=1)
                            inp = inp[:, :n_past, ...]

                        state_labels = torch.tensor(
                            self.train_dataset.subset_dict.get(subset.get_name(), [-1]*len(self.test_dataset.subset_dict.get(subset.get_name(), ['x']))),
                            device=self.device
                        ).unsqueeze(0)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        infer_start = time.perf_counter()
                        output, _ = self._safe_eval_rollout(
                            inp,
                            state_labels[0],
                            dset_name,
                            tar.shape[1],
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        infer_elapsed = time.perf_counter() - infer_start
                        total_inference_wallclock += infer_elapsed
                        total_inference_samples += float(inp.shape[0])
                        processed_samples += int(inp.shape[0])

                        metric_tar = tar.float()
                        metric_output = output.float()

                        spatial_dims = tuple(range(metric_output.ndim))[3:]
                        test_nrmse = nrmse(metric_tar, metric_output, dims=spatial_dims)
                        
                        # Whole-rollout relative L2: ||pred-target||_2 / ||target||_2 per sample
                        lp_loss = LpLoss(d=2, p=2, size_average=False, reduction=False)
                        rollout_rel_per_sample = lp_loss.rel(metric_output, metric_tar)  # Shape: [batch]
                        
                        # Step-averaged: mean relative L2 over all timesteps per sample
                        # Reshape to [batch*time, channels*H*W] for per-timestep relative L2
                        batch_size, num_steps = metric_output.shape[0], metric_output.shape[1]
                        output_flat = metric_output.reshape(batch_size * num_steps, -1)
                        target_flat = metric_tar.reshape(batch_size * num_steps, -1)
                        
                        # Compute per-step relative L2 for each (sample, timestep) pair
                        step_errors = (output_flat - target_flat).pow(2).sum(dim=1).sqrt()
                        step_norms = target_flat.pow(2).sum(dim=1).sqrt().clamp_min(1e-7)
                        step_rel_all = step_errors / step_norms  # Shape: [batch*time]
                        step_rel_all = step_rel_all.reshape(batch_size, num_steps)
                        step_rel_per_sample = step_rel_all.mean(dim=1)  # Average over timesteps per sample
                        
                        # Accumulate per-sample sums and counts for sample-weighted averaging
                        sample_sum_full[dset_name] += rollout_rel_per_sample.sum()
                        sample_sum_step[dset_name] += step_rel_per_sample.sum()
                        sample_counts[dset_name] += float(metric_tar.shape[0])
                        
                        sample_scores = test_nrmse.mean(dim=tuple(range(1, test_nrmse.ndim)))
                        batch_worst_nrmse, batch_worst_pos = torch.max(sample_scores, dim=0)
                        batch_indices = batch['index'].to(self.device).float()
                        batch_worst_idx = batch_indices[batch_worst_pos]
                        if batch_worst_nrmse > local_worst_nrmse[0]:
                            local_worst_nrmse[0] = batch_worst_nrmse
                            local_worst_idx[0] = batch_worst_idx

                        for step in eval_steps:
                            if step <= tar.shape[1]:
                                key = f'{dset_name}/test_nrmse_t{step}'
                                logs[key] += test_nrmse[:, step-1, ...].mean()
                        logs[f'{dset_name}/test_nrmse_tlast'] += test_nrmse[:, -1, ...].mean()

                        if (batch_idx + 1) % 10 == 0:
                            elapsed = time.perf_counter() - test_start
                            self.single_print(
                                f"TEST {dset_name}: processed {batch_idx + 1} batches in {elapsed:.1f}s"
                            )

            if dist.is_initialized():
                for dset_name in distinct_dsets:
                    dist.all_reduce(counts[dset_name])
                    dist.all_reduce(sample_sum_full[dset_name])
                    dist.all_reduce(sample_sum_step[dset_name])
                    dist.all_reduce(sample_counts[dset_name])
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key])
                dist.all_reduce(total_inference_wallclock)
                dist.all_reduce(total_inference_samples)

            # First aggregate per-dataset metrics using proper sample weighting
            for dset_name in distinct_dsets:
                if sample_counts[dset_name] > 0:
                    logs[f'{dset_name}/test_nrmse_rollout'] = sample_sum_full[dset_name] / sample_counts[dset_name]
                    logs[f'{dset_name}/test_nrmse_step'] = sample_sum_step[dset_name] / sample_counts[dset_name]
                    logs[f'{dset_name}/test_relative_l2_rollout'] = sample_sum_full[dset_name] / sample_counts[dset_name]
                else:
                    logs[f'{dset_name}/test_nrmse_rollout'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                    logs[f'{dset_name}/test_nrmse_step'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                    logs[f'{dset_name}/test_relative_l2_rollout'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)

            # Aggregate per-step metrics using batch counting (as before)
            for k, v in logs.items():
                if 'test_nrmse_t' in k:  # Per-step metrics
                    dset_name = k.split('/')[0]
                    denom = torch.clamp(counts[dset_name], min=1.0)
                    logs[k] = v / denom

            logs['test_nrmse'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
            logs['test_nrmse_rollout'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
            logs['test_nrmse_step'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
            logs['test_relative_l2_rollout'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
            for dset_name in distinct_dsets:
                nrmse_key = f'{dset_name}/test_nrmse_rollout'
                step_key = f'{dset_name}/test_nrmse_step'
                rel_l2_key = f'{dset_name}/test_relative_l2_rollout'
                if nrmse_key in logs:
                    logs['test_nrmse'] += logs[nrmse_key] / len(distinct_dsets)
                    logs['test_nrmse_rollout'] += logs[nrmse_key] / len(distinct_dsets)
                if step_key in logs:
                    logs['test_nrmse_step'] += logs[step_key] / len(distinct_dsets)
                if rel_l2_key in logs:
                    logs['test_relative_l2_rollout'] += logs[rel_l2_key] / len(distinct_dsets)

            if float(total_inference_samples.item()) > 0:
                logs['time/inference_wallclock_total'] = total_inference_wallclock
                logs['time/inference_wallclock_per_data'] = total_inference_wallclock / total_inference_samples
            else:
                logs['time/inference_wallclock_total'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)
                logs['time/inference_wallclock_per_data'] = torch.zeros(1, device=self.device, dtype=self.base_dtype)

            if dist.is_initialized():
                gathered_worst = [torch.zeros_like(local_worst_nrmse) for _ in range(dist.get_world_size())]
                gathered_idx = [torch.zeros_like(local_worst_idx) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_worst, local_worst_nrmse)
                dist.all_gather(gathered_idx, local_worst_idx)
                all_worst = torch.stack(gathered_worst).view(-1)
                all_idx = torch.stack(gathered_idx).view(-1)
                max_pos = torch.argmax(all_worst)
                logs['test_worst_nrmse'] = all_worst[max_pos]
                logs['test_worst_idx'] = all_idx[max_pos]
            else:
                logs['test_worst_nrmse'] = local_worst_nrmse[0]
                logs['test_worst_idx'] = local_worst_idx[0]

            for key in list(logs.keys()):
                if isinstance(logs[key], torch.Tensor):
                    logs[key] = float(logs[key].item())

        self.single_print('DONE TEST EVALUATION')
        self.single_print(
            f"Test inference wallclock: total={logs.get('time/inference_wallclock_total', 0.0):.4f}s, "
            f"per_data={logs.get('time/inference_wallclock_per_data', 0.0):.6f}s"
        )
        return logs

    def train(self):

        if self.params.log_to_wandb:
            wandb.init(
                dir=self.params.experiment_dir, config=self.params, 
                name=self.params.name, group=self.params.group,
                project=self.params.project,
                resume="never"
            )
            wandb.define_metric('optimizer_steps_completed')
            wandb.define_metric('*', step_metric='optimizer_steps_completed')
        if self.global_rank == 0:
            summary(self.model)
        if self.params.log_to_wandb:
            wandb.watch(self.model, log="parameters")
        self.single_print("Starting Training Loop...")
    
        # Actually train now, saving checkpoints, logging time, and logging to wandb
        best_valid_loss = 1.e6
        budget_enabled = hasattr(self.params, 'time_budget') and self.params.time_budget is not None
        epoch = self.start_epoch
        while True:
            if (not budget_enabled) and epoch >= self.params.max_epochs:
                break
            # if dist.is_initialized():
            if 'overfit' in self.params and self.params.overfit:
                self.train_sampler.set_epoch(0)
            elif hasattr(self.params, 'ntrain') and self.params.ntrain is not None:
                self.train_sampler.set_epoch(0)
            else:
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            tr_time, data_time, train_logs = self.train_one_epoch()

            post_start = time.time()
            train_logs['time/train_time'] = post_start - start
            train_logs['time/train_data_time'] = data_time
            train_logs['time/train_compute_time'] = tr_time
            train_logs['time/valid_time'] = 0.0

            should_log_train_epoch = True
            if self.should_stop_training and self.stop_reason == 'time_budget':
                epoch_steps = int(train_logs.get('epoch_steps', 0))
                if epoch_steps < int(self.params.epoch_size):
                    should_log_train_epoch = False

            if self.params.log_to_wandb:
                if should_log_train_epoch:
                    wandb.log(train_logs, step=int(self.optimizer_steps_completed))
                else:
                    self.single_print(
                        f"Skipping final partial-epoch train log (epoch_steps={int(train_logs.get('epoch_steps', 0))}) "
                        "to avoid tail noise."
                    )

            if self.global_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                if epoch % self.params.checkpoint_save_interval == 0:
                    self.save_checkpoint(self.params.checkpoint_path + f'_epoch{epoch}')
                
                cur_time = time.time()
                self.single_print(f'Time for train {post_start-start:.2f}. For postprocessing:{cur_time-post_start:.2f}')
                self.single_print('Time taken for epoch {} is {:.2f} sec'.format(1+epoch, time.time()-start))
                self.single_print('Train loss: {}'.format(train_logs['train_nrmse']))

            # Clear references to large tensors after we're done using them
            train_logs = None
            
            # More aggressive memory cleanup
            gc.collect()
            
            # Force CUDA to synchronize and clear any pending operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.should_stop_training:
                self.single_print(
                    f"Stopping training due to {self.stop_reason}. "
                    f"compute_total={self.train_compute_time_total:.2f}s, "
                    f"compute_budget_used={self.train_compute_time_budget:.2f}s, "
                    f"adjusted_budget={self.params.time_budget:.2f}s"
                )
                # Save final checkpoint when training stops
                if self.global_rank == 0 and self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                    self.single_print(f"Saved final checkpoint to {self.params.checkpoint_path}")
                break

            epoch += 1

        # Final evaluation once training finishes/stops
        if self.test_dataset is not None:
            test_logs = self.test_one_epoch()
            if self.params.log_to_wandb:
                for key, value in test_logs.items():
                    wandb.run.summary[key] = value
                final_test_logs = {f'final/{key}': value for key, value in test_logs.items()}
                wandb.log(final_test_logs, step=int(self.optimizer_steps_completed))
        else:
            val_cutoff = 999
            if self.params.debug:
                val_cutoff = 1
            valid_logs = self.validate_one_epoch(val_cutoff)
            if self.params.log_to_wandb:
                for key, value in valid_logs.items():
                    wandb.run.summary[key] = value
                final_valid_logs = {f'final/{key}': value for key, value in valid_logs.items()}
                wandb.log(final_valid_logs, step=int(self.optimizer_steps_completed))
            if self.global_rank == 0 and 'valid_nrmse' in valid_logs and valid_logs['valid_nrmse'] <= best_valid_loss:
                self.save_checkpoint(self.params.best_checkpoint_path)


if __name__ == '__main__':

    print(f"DEBUG : {is_debug()}")

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--yaml_config", "--yaml-config", dest="yaml_config", default='_debug.yaml', type=str)
    parser.add_argument("--run_name", type=str, default=None, help='Override run_name from config')
    parser.add_argument("--wandb_project", type=str, default=None, help='WandB project name override')
    parser.add_argument("--time_budget", type=float, default=None, help='Time budget in seconds (placeholder for future use)')
    parser.add_argument("--ntrain", type=int, default=None, help='Number of training samples to load/use per epoch')
    parser.add_argument("--per_data_gen_cost", type=float, default=0.0, help='Per-sample data generation cost in seconds')
    parser.add_argument("--checkpoint_dir", type=str, default=None, help='Directory for checkpoints; resume if ckpt.tar exists')
    parser.add_argument("--timing_warmup_steps", type=int, default=3, help='Number of optimizer warmup steps excluded from budget')
    parser.add_argument("--timing_avg_steps", type=int, default=3, help='Number of post-warmup optimizer steps (3-5) for avg step time')
    parser.add_argument("--test_max_batches", type=int, default=None, help='Optional cap on batches for final test evaluation (None = all)')
    parser.add_argument("--test_max_samples_grayscott", type=int, default=None, help='Optional cap on Gray-Scott test samples (None = all)')
    parser.add_argument("--use_autoregressive", action='store_true', help='Use autoregressive rollout during training')
    args = parser.parse_args()

    # Config 
    CONFIG_PATH = Path(__file__).resolve().parent / "config"
    params = YParams(CONFIG_PATH/"_base.yaml")
    refined_params = YParams(CONFIG_PATH/args.yaml_config)
    params.update_params(refined_params.params)
    if is_debug():
        debug_params = YParams(CONFIG_PATH/"_debug.yaml")
        params.update_params(debug_params.params)
    params['debug'] = is_debug()
    params['yaml_config'] = args.yaml_config
    params['use_ddp'] = args.use_ddp
    if args.run_name is not None:
        params['run_name'] = args.run_name
    if args.ntrain is not None:
        params['ntrain'] = int(args.ntrain)
    else:
        params['ntrain'] = None
    params['per_data_gen_cost'] = float(args.per_data_gen_cost)
    params['timing_warmup_steps'] = int(args.timing_warmup_steps)
    params['timing_avg_steps'] = int(args.timing_avg_steps)
    params['test_max_batches'] = int(args.test_max_batches) if args.test_max_batches is not None else None
    params['test_max_samples_grayscott'] = int(args.test_max_samples_grayscott) if args.test_max_samples_grayscott is not None else (params['test_max_samples_grayscott'] if 'test_max_samples_grayscott' in params else None)
    params['time_budget'] = float(args.time_budget) if args.time_budget is not None else None
    if params['time_budget'] is not None and params['ntrain'] is not None:
        adjusted_budget = params['time_budget'] - params['per_data_gen_cost'] * params['ntrain']
        params['time_budget'] = adjusted_budget

    # Set up distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.use_ddp:
        dist.init_process_group("nccl")  # backend for nvidia gpus, multi-node, multi-gpu
    torch.cuda.set_device(local_rank)
        
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Modify params
    params['batch_size'] = int(params.batch_size//world_size)
    params['start_epoch'] = 0
    if 'exp_dir' not in params:
        params['exp_dir'] = str((Path(__file__).resolve().parent / 'experiments').resolve())
    exp_dir = Path(params.exp_dir) / params.run_name
    params['experiment_dir'] = str(exp_dir)
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir is not None else (exp_dir / 'training_checkpoints')
    params['checkpoint_dir'] = str(checkpoint_dir)
    params['checkpoint_path'] = str(checkpoint_dir / 'ckpt.tar')
    params['best_checkpoint_path'] = str(checkpoint_dir / 'best_ckpt.tar')

    # Have rank 0 check for and/or make directory
    if global_rank == 0:
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    params['resuming'] = Path(params.checkpoint_path).is_file()

    # Wandb setup
    if args.wandb_project is not None:
        params['project'] = args.wandb_project
    else:
        params['project'] = params.run_name
    if args.time_budget is not None and params.ntrain is not None:
        params['group'] = f"B{int(args.time_budget)}"
        params['name'] = f"n{int(params.ntrain)}_B{int(args.time_budget)}"
    else:
        params['group'] = None
        params['name'] = params.run_name
    if global_rank==0:
        logging_utils.log_to_file(logger_name=None, log_filename=exp_dir/'out.log')
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (global_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (global_rank==0) and params['log_to_screen']

    if global_rank == 0:  # save config for this run
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = value
        with open(exp_dir/'hyperparams.yaml', 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    # Start training
    trainer = Trainer(params, global_rank, local_rank, device, use_autoregressive=args.use_autoregressive)
    trainer.train()

    if params.log_to_wandb:
        wandb.finish()

    # Explicit DDP teardown to avoid post-training hangs
    if args.use_ddp and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()

    # Print from all ranks for easier debugging on clusters
    print('DONE ---- rank %d' % global_rank, flush=True)