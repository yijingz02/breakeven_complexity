import sys
import os
# sys.path.append(['.','./../'])
# os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import wandb


from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.dpot import DPOTNet
from models.dpot_res import CDPOTNet
import pickle




################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='DPOT')
parser.add_argument('--dataset',type=str, default='ns2d')

parser.add_argument('--train_paths',nargs='+', type=str, default=['ns2d_fno_1e-5','ns2d_pdb_M1_eta1e-1_zeta1e-1'])
parser.add_argument('--test_paths',nargs='+',type=str, default=['ns2d_pdb_M1_eta1e-1_zeta1e-1'])
parser.add_argument('--resume_path',type=str, default='')
parser.add_argument('--ntrain_list', nargs='+', type=int, default=[1000, 5000])
parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
parser.add_argument('--use_writer', action='store_true',default=False)

parser.add_argument('--res', type=int, default=128)
parser.add_argument('--noise_scale',type=float, default=0.0)
# parser.add_argument('--n_channels',type=int,default=-1)

### shared params
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--n_layers',type=int, default=4)
parser.add_argument('--act',type=str, default='gelu')


### FNO params
parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--normalize',type=int, default=0)


### DPOT
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--out_layer_dim', type=int, default=32)

parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cycle')
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=100)
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--gpu', type=str, default="5")
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='')
parser.add_argument('--ndata', type=int, default=-1, help='Number of training data to use; -1 uses all')
parser.add_argument('--budget', type=float, default=-1, help='Time budget in seconds; -1 means no budget')
parser.add_argument('--per_data_gen_time', type=float, default=0.0, help='Time to generate per training data in seconds')
parser.add_argument('--project_name', type=str, default='DPOT', help='Weights & Biases project name')
parser.add_argument('--checkpoint_dir', type=str,
                    default=os.path.join(os.environ.get("MODEL_STORAGE_ROOT", "checkpoints"), "DPOT"),
                    help='Persistent directory for DPOT checkpoints')
parser.add_argument('--sched_warmup_steps', type=int, default=10, help='Number of burn-in optimization steps before timing profile')
parser.add_argument('--sched_profile_steps', type=int, default=5, help='Number of profiled optimization steps for average step-time estimation')
parser.add_argument('--sched_time_multiplier', type=float, default=1.2, help='Conservative multiplier for profiled step time when estimating total steps')
args = parser.parse_args()


def _time_avg_nrmse_per_sample(pred, ref, eps=1e-12):
    batch_size = pred.shape[0]
    numerator = (pred - ref).reshape(batch_size, -1).pow(2).sum(dim=1)
    denominator = ref.reshape(batch_size, -1).pow(2).sum(dim=1).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


def _flip_multi_obstacle_fields(fields, names):
    """Flip B,H,W,C fields using Poseidon's physical-field parity."""
    flipped = torch.flip(fields, dims=[2])
    for channel_idx, name in enumerate(names):
        if name in {'vorticity', 'velocity_x'}:
            flipped[..., channel_idx] *= -1.0
    return flipped


def multi_obstacle_time_avg_scores(predictions, references, mask=None, last_n=160):
    """Return per-trajectory flip/no-flip NRMSE of the final-frame time average."""
    # DPOT tensors are B,H,W,T,C.
    predictions = predictions.float()
    references = references.float()
    if mask is not None:
        fluid = mask.to(device=predictions.device, dtype=predictions.dtype)
        predictions = predictions * fluid
        references = references * fluid

    k = min(int(last_n), predictions.shape[-2], references.shape[-2])
    if k <= 0:
        return {}

    pred_avg = predictions[..., -k:, :].mean(dim=-2)
    ref_avg = references[..., -k:, :].mean(dim=-2)
    if pred_avg.shape[-1] >= 3 and ref_avg.shape[-1] >= 3:
        channel_names = ['velocity_x', 'velocity_y', 'pressure']
        groups = {
            'velocity_x': [0],
            'velocity_y': [1],
            'pressure': [2],
            'velocity': [0, 1],
            'vxvyp': [0, 1, 2],
            'all_fields': [0, 1, 2],
        }
    elif pred_avg.shape[-1] == 1 and ref_avg.shape[-1] == 1:
        channel_names = ['vorticity']
        groups = {'vorticity': [0], 'all_fields': [0]}
    else:
        return {}

    scores = {}
    for group_name, indices in groups.items():
        names = [channel_names[idx] for idx in indices]
        pred_fields = pred_avg[..., indices]
        ref_fields = ref_avg[..., indices]
        scores[f'tavg_flip_{group_name}'] = _time_avg_nrmse_per_sample(
            pred_fields + _flip_multi_obstacle_fields(pred_fields, names),
            ref_fields + _flip_multi_obstacle_fields(ref_fields, names),
        )
        scores[f'tavg_noflip_{group_name}'] = _time_avg_nrmse_per_sample(
            pred_fields, ref_fields
        )
    return scores


def channelwise_physical_nrmse_breakflow(
    predictions, references, mask=None, eps=1e-12
):
    """Per-trajectory physical-unit nRMSE for DPOT B,H,W,T,C tensors."""
    pred = predictions[..., :3].float().clone()
    ref = references[..., :3].float().clone()
    if pred.shape[-1] < 3 or ref.shape[-1] < 3:
        raise ValueError("BreakFlow physical nRMSE requires vx, vy, pressure channels")

    if mask is None:
        weight = torch.ones(
            *pred.shape[:3], 1, 3,
            device=pred.device,
            dtype=pred.dtype,
        )
    else:
        weight = mask.to(device=pred.device, dtype=pred.dtype)[..., :3]
        weight = (weight > 0.5).to(pred.dtype)

    pressure_weight = weight[..., 2]
    pressure_count = pressure_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    pred[..., 2] -= (
        (pred[..., 2] * pressure_weight).sum(dim=(1, 2), keepdim=True)
        / pressure_count
    )
    ref[..., 2] -= (
        (ref[..., 2] * pressure_weight).sum(dim=(1, 2), keepdim=True)
        / pressure_count
    )

    numerator = ((pred - ref).square() * weight).sum(dim=(1, 2, 3))
    denominator = (ref.square() * weight).sum(dim=(1, 2, 3)).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


if isinstance(args.gpu, str) and args.gpu.lower() == 'cpu':
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args.gpu))
else:
    device = torch.device('cpu')
    print('Warning: CUDA is not available or PyTorch is CPU-only. Falling back to CPU.')

print(f"Current working directory: {os.getcwd()}")

################################################################
# load data and dataloader
################################################################
train_paths = args.train_paths
test_paths = args.test_paths
args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
print('args',args)

# Cap number of training samples if ndata is specified
if args.ndata > 0:
    ntrain_list = [min(n, args.ndata) for n in args.ntrain_list]
else:
    ntrain_list = args.ntrain_list

train_dataset = MixedTemporalDataset(args.train_paths, ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar, normalize=False,train=True, data_weights=args.data_weights)
test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False) for i, test_path in enumerate(test_paths)]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]
ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]
print('Train num {} test num {}'.format(train_dataset.n_sizes, ntests))
################################################################
# load model
################################################################
if args.model == "FNO":
    model = FNO2d(args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels, n_cls=len(args.train_paths)).to(device)
elif args.model == 'DPOT':
    model = DPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset.n_channels, normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=len(args.train_paths)).to(device)
elif args.model == 'CDPOT':
    model = CDPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset.n_channels, normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=len(args.train_paths)).to(device)

else:
    raise NotImplementedError

if args.resume_path:
    print('Loading models and fine tune from {}'.format(args.resume_path))
    args.resume_path = args.resume_path
    load_model_from_checkpoint(model, torch.load(args.resume_path, map_location=device)['model'])


#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
else:
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)

default_total_steps = max(1, args.epochs * len(train_loader))
default_total_epochs = max(1, args.epochs)


def build_scheduler(total_epochs):
    total_epochs = max(1, int(total_epochs))
    total_steps = max(1, total_epochs * len(train_loader))
    if args.lr_method == 'cycle':
        return OneCycleLR(
            optimizer,
            max_lr=args.lr,
            div_factor=1e4,
            pct_start=0.08,
            final_div_factor=1e4,
            steps_per_epoch=len(train_loader),
            epochs=total_epochs,
        )
    if args.lr_method == 'step':
        print('Using step learning rate schedule')
        scaled_step_size = max(1, int(args.step_size * total_steps / max(1, default_total_steps)))
        return StepLR(optimizer, step_size=scaled_step_size, gamma=args.step_gamma)
    if args.lr_method == 'warmup':
        print('Using warmup learning rate schedule')
        warmup_total_steps = max(1, int(args.warmup_epochs / max(1, args.epochs) * total_steps))
        return LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_total_steps, np.power(warmup_total_steps / float(steps + 1), 0.5)))
    if args.lr_method == 'linear':
        print('Using linear learning rate schedule')
        return LambdaLR(optimizer, lambda steps: max(0.0, 1 - steps / float(total_steps)))
    if args.lr_method == 'restart':
        print('Using cos anneal restart')
        lr_step_size = getattr(args, 'lr_step_size', args.step_size)
        restart_period = max(1, int(total_steps / max(1, lr_step_size)))
        return CosineAnnealingWarmRestarts(optimizer, T_0=restart_period, eta_min=0.)
    if args.lr_method == 'cyclic':
        lr_step_size = getattr(args, 'lr_step_size', args.step_size)
        step_size_up = max(1, int(lr_step_size * total_steps / max(1, default_total_steps)))
        return CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=step_size_up, mode='triangular2', cycle_momentum=False)
    raise NotImplementedError

comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
model_path = log_path + '/model.pth'
safe_project_name = ''.join(
    ch if ch.isalnum() or ch in '._-' else '_' for ch in args.project_name
)
persistent_checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
os.makedirs(persistent_checkpoint_dir, exist_ok=True)
persistent_checkpoint_path = os.path.join(
    persistent_checkpoint_dir,
    (
        f'{safe_project_name}_{args.dataset}_N{ntrain}_B{int(args.budget)}'
        f'_p{args.patch_size}_w{args.width}_l{args.n_layers}_b{args.n_blocks}.pth'
    ),
)

# Calculate actual training budget after accounting for data generation time
if args.budget > 0:
    data_gen_overhead = ntrain * args.per_data_gen_time
    actual_budget = args.budget - data_gen_overhead
    print(f'Total budget: {args.budget}s, Data generation overhead: {data_gen_overhead}s, Actual training budget: {actual_budget}s')
else:
    actual_budget = -1
    print(f'No budget limit (budget={args.budget})')

# Initialize wandb
wandb.init(
    project=f"{args.project_name}",
    group=f"B{int(args.budget)}",
    name=f"n{ntrain}_B{int(args.budget)}",
    # config=vars(args)
    settings=wandb.Settings(disable_job_creation=True)
)

if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)
    fp = open(log_path + '/logs.txt', 'w+',buffering=1)
    json.dump({**vars(args), 'actual_budget': actual_budget}, open(log_path + '/params.json', 'w'),indent=4)
    sys.stdout = fp

else:
    writer = None

print(model)
count_parameters(model)

################################################################
# Main function for pretraining
################################################################
myloss = SimpleLpLoss(size_average=False)
clsloss = torch.nn.CrossEntropyLoss(reduction='sum')
iter = 0
total_train_compute_time = 0.0
stopped_by_budget = False
scheduler_total_steps = default_total_steps
scheduler_total_epochs = default_total_epochs
scheduler = build_scheduler(scheduler_total_epochs)
scheduler_max_steps = max(1, scheduler_total_epochs * len(train_loader))
scheduler_steps_taken = 0
scheduler_exhausted_logged = False


def save_persistent_checkpoint(epoch):
    torch.save(
        {
            'args': vars(args),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': int(epoch),
            'global_step': int(iter),
            'train_compute_time': float(total_train_compute_time),
            'stopped_by_budget': bool(stopped_by_budget),
        },
        persistent_checkpoint_path,
    )
    print(f'[SAVE] Saved checkpoint to {persistent_checkpoint_path}')


if args.budget > 0 and actual_budget <= 0:
    print(f'Actual training budget is {actual_budget:.5f}s (<=0). Skipping training loop.')
    stopped_by_budget = True

if (not stopped_by_budget) and actual_budget > 0:
    warmup_steps = max(0, args.sched_warmup_steps)
    profile_steps = max(1, args.sched_profile_steps)
    total_profile_steps = warmup_steps + profile_steps
    profiled_step_times = []
    warmup_l2_step = 0.0
    warmup_l2_full = 0.0
    warmup_samples_seen = 0
    warmup_rollout_steps_per_sample = None
    print(f'Profiling scheduler with {warmup_steps} warmup steps + {profile_steps} profiled steps...')

    model.train()
    profile_loader_iter = train_loader.__iter__()
    completed_profile_steps = 0
    while completed_profile_steps < total_profile_steps:
        try:
            xx, yy, msk, cls = next(profile_loader_iter)
        except StopIteration:
            profile_loader_iter = train_loader.__iter__()
            xx, yy, msk, cls = next(profile_loader_iter)

        loss = 0.
        xx = xx.to(device)
        yy = yy.to(device)
        msk = msk.to(device)
        warmup_samples_seen += xx.shape[0]
        if warmup_rollout_steps_per_sample is None:
            warmup_rollout_steps_per_sample = yy.shape[-2] / args.T_bundle

        compute_t0 = default_timer()
        for t in range(0, yy.shape[-2], args.T_bundle):
            y = yy[..., t:t + args.T_bundle, :]
            xx = xx + args.noise_scale * torch.sum(xx**2, dim=(1, 2, 3), keepdim=True)**0.5 * torch.randn_like(xx)
            im, _ = model(xx)
            loss += myloss(im, y, mask=msk)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-2)
            xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

        warmup_l2_step += loss.item()
        l2_full_warmup = myloss(pred, yy, mask=msk)
        warmup_l2_full += l2_full_warmup.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step_time = default_timer() - compute_t0
        if completed_profile_steps >= warmup_steps:
            profiled_step_times.append(step_time)
            print(f"[warmup-profile] step={completed_profile_steps - warmup_steps + 1}/{profile_steps}, step_time={step_time:.5f}s")
        else:
            print(f"[warmup-burn] step={completed_profile_steps + 1}/{warmup_steps}, step_time={step_time:.5f}s")
        completed_profile_steps += 1

    avg_step_time = float(np.mean(profiled_step_times)) if len(profiled_step_times) > 0 else 0.0
    # conservative_step_time = avg_step_time * args.sched_time_multiplier
    conservative_step_time = avg_step_time
    estimated_total_steps = int(actual_budget / max(conservative_step_time, 1e-8))
    estimated_total_epochs = int(np.ceil(estimated_total_steps / max(1, len(train_loader))))
    scheduler_total_epochs = max(1, estimated_total_epochs)
    scheduler = build_scheduler(scheduler_total_epochs)
    scheduler_max_steps = max(1, scheduler_total_epochs * len(train_loader))
    scheduler_steps_taken = 0
    scheduler_exhausted_logged = False
    warmup_l2_step_avg = warmup_l2_step / max(1, warmup_samples_seen) / max(1e-8, warmup_rollout_steps_per_sample)
    warmup_l2_full_avg = warmup_l2_full / max(1, warmup_samples_seen)
    print(
        f'Warmup/profile done: estimated_total_epochs={scheduler_total_epochs}, '
        f'warmup_l2_step={warmup_l2_step_avg:.5f}, warmup_l2_full={warmup_l2_full_avg:.5f}'
    )
    wandb.log({
        'sched/profile_warmup_steps': warmup_steps,
        'sched/profile_steps': profile_steps,
        'sched/estimated_total_steps': estimated_total_steps,
        'sched/estimated_total_epochs': scheduler_total_epochs,
        'sched/warmup_l2_step': warmup_l2_step_avg,
        'sched/warmup_l2_full': warmup_l2_full_avg,
        'global_step': iter,
    })

# for ep in range(args.epochs):
ep = 0
while True:
    ep += 1
    
    if stopped_by_budget:
        break

    model.train()

    t1 = t_1 = default_timer()
    t_load, t_train = 0., 0.
    train_l2_step = 0
    train_l2_full = 0
    cls_total, cls_correct, cls_acc = 0, 0, 0.
    loss_previous = np.inf
    epoch_samples_seen = 0
    epoch_batches_seen = 0
    rollout_steps_per_sample = None

    for xx, yy, msk, cls in train_loader:
        t_load += default_timer() - t_1
        t_1 = default_timer()

        epoch_samples_seen += xx.shape[0]
        epoch_batches_seen += 1
        if rollout_steps_per_sample is None:
            rollout_steps_per_sample = yy.shape[-2] / args.T_bundle

        loss, cls_loss = 0. , 0.
        xx = xx.to(device)  ## B, n, n, T_in, C
        yy = yy.to(device)  ## B, n, n, T_ar, C
        msk = msk.to(device)
        cls = cls.to(device)

        compute_t0 = default_timer()


        ## auto-regressive training loop, support 1. noise injection, 2. long rollout backward, 3. temporal bundling prediction
        for t in range(0, yy.shape[-2], args.T_bundle):
            y = yy[..., t:t + args.T_bundle, :]

            ### auto-regressive training
            xx = xx + args.noise_scale *torch.sum(xx**2, dim=(1,2,3),keepdim=True)**0.5 * torch.randn_like(xx)
            im, cls_pred = model(xx)
            loss += myloss(im, y, mask=msk)

            ### classification
            pred_labels = torch.argmax(cls_pred,dim=1)
            cls_loss += clsloss(cls_pred, cls.squeeze())
            cls_correct += (pred_labels == cls.squeeze()).sum().item()
            cls_total += cls.shape[0]

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-2)
            xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

        train_l2_step += loss.item()
        l2_full = myloss(pred, yy, mask=msk)
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        total_loss = loss  # + 1.0 * cls_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler_steps_taken < scheduler_max_steps:
            scheduler.step()
            scheduler_steps_taken += 1
        elif not scheduler_exhausted_logged:
            scheduler_exhausted_logged = True
            print(f"Scheduler exhausted at step {scheduler_steps_taken}; keeping lr fixed at {optimizer.param_groups[0]['lr']:.6e}")

        batch_compute_time = default_timer() - compute_t0
        t_train += batch_compute_time
        total_train_compute_time += batch_compute_time
        if (iter < 10) or (iter % 100 == 0):
            print(f"[train-step] global_step={iter + 1}, batch_compute_time={batch_compute_time:.5f}s")

        cls_acc = cls_correct / cls_total
        iter +=1
        wandb.log({
            'train/loss_step_batch': loss.item()/(xx.shape[0] * yy.shape[-2] / args.T_bundle),
            'train/loss_full_batch': l2_full.item() / xx.shape[0],
            'train/cls_acc': cls_acc,
            'train/lr': optimizer.param_groups[0]['lr'],
            'train/epoch': ep,
            'global_step': iter
        })
        if args.use_writer:
            writer.add_scalar("train_loss_step", loss.item()/(xx.shape[0] * yy.shape[-2] / args.T_bundle), iter)
            writer.add_scalar("train_loss_full", l2_full.item() / xx.shape[0], iter)

            ## reset model
            if loss.item() > 10 * loss_previous : # or (ep > 50 and l2_full / xx.shape[0] > 0.9):
                print('loss explodes, loading model from previous epoch')
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint["optimizer"])
                loss_previous = loss.item()

        if actual_budget > 0 and total_train_compute_time >= actual_budget:
            stopped_by_budget = True
            print(f"Stopping early due to budget: compute_time={total_train_compute_time:.5f}s >= actual_budget={actual_budget:.5f}s")
            break

        t_1 = default_timer()

    if epoch_batches_seen == 0:
        print('No training batches were processed in this epoch.')
        break

    # Compute epoch-level averages (after all batches are processed)
    train_l2_step_avg = train_l2_step / epoch_samples_seen / rollout_steps_per_sample
    train_l2_full_avg = train_l2_full / epoch_samples_seen
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0

    if not stopped_by_budget and epoch_samples_seen != ntrain:
        raise RuntimeError(f"Expected exactly ntrain={ntrain} samples per epoch, but got {epoch_samples_seen}.")

    if args.use_writer:
        torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)
    save_persistent_checkpoint(ep)

    t2 = default_timer()
    lr = optimizer.param_groups[0]['lr']
    epoch_log = {
        'epoch/lr': lr,
        'epoch/train_l2_step': train_l2_step_avg,
        'epoch/train_l2_full': train_l2_full_avg,
        'epoch/cls_acc': cls_acc,
        'epoch/time_total': t2 - t1,
        'epoch/time_train_avg': t_train / epoch_batches_seen,
        'epoch/time_load_avg': t_load / epoch_batches_seen,
        'epoch/compute_time_cum': total_train_compute_time,
        'epoch/samples_seen': epoch_samples_seen,
        'epoch/index': ep
    }
    wandb.log(epoch_log)
    print('epoch {}, time {:.5f}, lr {:.2e}, train l2 step {:.5f} train l2 full {:.5f}, time train avg {:.5f} load avg {:.5f}, samples {}, compute_time_cum {:.5f}'.format(ep, t2 - t1, lr, train_l2_step_avg, train_l2_full_avg, t_train / epoch_batches_seen, t_load / epoch_batches_seen, epoch_samples_seen, total_train_compute_time))

    if stopped_by_budget:
        break

# Also save when training is skipped or exits between epoch-level saves.
save_persistent_checkpoint(ep)
wandb.run.summary['checkpoint_path'] = persistent_checkpoint_path

test_l2_fulls, test_l2_steps = [], []
test_rollout_time_per_trajs = []
test_time_avg_stats = []
test_physical_nrmse_stats = []
worst_traj_l2 = -float('inf')
worst_traj_idx = -1
worst_traj_dataset = -1
test_t0 = default_timer()
with torch.no_grad():
    model.eval()
    for id, test_loader in enumerate(test_loaders):
        test_l2_full, test_l2_step = 0, 0
        test_rollout_time = 0.0
        test_rollout_trajs = 0
        time_avg_sums = {}
        time_avg_counts = {}
        physical_nrmse_sum = torch.zeros(3, dtype=torch.float64, device=device)
        physical_nrmse_count = 0
        dataset_base_idx = 0
        for xx, yy, msk, _ in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            msk = msk.to(device)
            batch_size_eval = xx.shape[0]
            rollout_t0 = default_timer()

            for t in range(0, yy.shape[-2], args.T_bundle):
                y = yy[..., t:t + args.T_bundle, :]
                im, _ = model(xx)
                loss += myloss(im, y, mask=msk).item()

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)

                xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)

            test_rollout_time += default_timer() - rollout_t0
            test_rollout_trajs += batch_size_eval

            test_l2_step += loss
            per_traj_full = myloss(pred, yy, mask=msk)
            test_l2_full += per_traj_full.item()

            if 'custom_bf' in test_paths[id]:
                physical_batch_scores = channelwise_physical_nrmse_breakflow(
                    pred,
                    yy,
                    mask=msk,
                )
                physical_nrmse_sum += physical_batch_scores.double().sum(dim=0)
                physical_nrmse_count += int(physical_batch_scores.shape[0])
                batch_time_avg_scores = multi_obstacle_time_avg_scores(
                    pred,
                    yy,
                    mask=msk,
                    last_n=160,
                )
                for key, values in batch_time_avg_scores.items():
                    time_avg_sums[key] = time_avg_sums.get(key, 0.0) + float(values.sum().item())
                    time_avg_counts[key] = time_avg_counts.get(key, 0) + int(values.shape[0])

            if 'custom_bf' in test_paths[id]:
                per_traj_vec = physical_batch_scores.mean(dim=1)
            else:
                per_traj_vec = torch.sum(
                    torch.norm(pred.reshape(pred.shape[0], -1, pred.shape[-1]) - yy.reshape(yy.shape[0], -1, yy.shape[-1]), 2, dim=1)
                    /
                    (torch.norm(yy.reshape(yy.shape[0], -1, yy.shape[-1]), 2, dim=1) + 1e-8),
                    dim=-1,
                ) / pred.shape[-1]
            batch_worst_loss, batch_worst_j = torch.max(per_traj_vec, dim=0)
            if float(batch_worst_loss.item()) > worst_traj_l2:
                worst_traj_l2 = float(batch_worst_loss.item())
                worst_traj_idx = int(dataset_base_idx + batch_worst_j.item())
                worst_traj_dataset = id
            dataset_base_idx += batch_size_eval

        test_l2_step_avg, test_l2_full_avg = test_l2_step / ntests[id] / (yy.shape[-2] / args.T_bundle), test_l2_full / ntests[id]
        rollout_time_per_traj = test_rollout_time / max(1, test_rollout_trajs)
        test_l2_steps.append(test_l2_step_avg)
        test_l2_fulls.append(test_l2_full_avg)
        test_rollout_time_per_trajs.append(rollout_time_per_traj)
        dataset_time_avg_stats = {
            key: time_avg_sums[key] / max(1, time_avg_counts[key])
            for key in time_avg_sums
        }
        test_time_avg_stats.append(dataset_time_avg_stats)
        dataset_physical_stats = {}
        if physical_nrmse_count > 0:
            channel_scores = physical_nrmse_sum / physical_nrmse_count
            dataset_physical_stats = {
                'velocity_x': float(channel_scores[0]),
                'velocity_y': float(channel_scores[1]),
                'pressure_gauge_centered': float(channel_scores[2]),
                'macro': float(channel_scores.mean()),
            }
            print(
                f"{test_paths[id]} channelwise physical-unit nRMSE | "
                f"vx={channel_scores[0]:.6f}, vy={channel_scores[1]:.6f}, "
                f"pressure(gauge-centered)={channel_scores[2]:.6f}, "
                f"macro={channel_scores.mean():.6f}"
            )
        test_physical_nrmse_stats.append(dataset_physical_stats)
        if dataset_time_avg_stats:
            print(f'{test_paths[id]} time-average statistics (last 160 rollout frames):')
            for quantity in ['velocity_x', 'velocity_y', 'pressure', 'velocity', 'vxvyp']:
                if f'tavg_flip_{quantity}' in dataset_time_avg_stats:
                    print(
                        f"  {quantity:>10s} | "
                        f"flip: {dataset_time_avg_stats[f'tavg_flip_{quantity}']:.6f} | "
                        f"noflip: {dataset_time_avg_stats[f'tavg_noflip_{quantity}']:.6f}"
                    )
        if args.use_writer:
            writer.add_scalar("test_loss_step_{}".format(test_paths[id]), test_l2_step_avg, args.epochs)
            writer.add_scalar("test_loss_full_{}".format(test_paths[id]), test_l2_full_avg, args.epochs)
            writer.add_scalar("test_rollout_time_per_traj_{}".format(test_paths[id]), rollout_time_per_traj, args.epochs)

test_time = default_timer() - test_t0
final_log = {
    'final/time_test': test_time,
    'final/epoch': args.epochs
}
for dataset_name, step_val, full_val in zip(test_paths, test_l2_steps, test_l2_fulls):
    safe_name = dataset_name.replace('/', '_')
    final_log[f'final/test/{safe_name}/l2_step'] = step_val
    final_log[f'final/test/{safe_name}/l2_full'] = full_val
for dataset_name, rollout_time_val in zip(test_paths, test_rollout_time_per_trajs):
    safe_name = dataset_name.replace('/', '_')
    final_log[f'final/test/{safe_name}/rollout_time_per_traj'] = rollout_time_val
for dataset_name, dataset_stats in zip(test_paths, test_time_avg_stats):
    safe_name = dataset_name.replace('/', '_')
    for key, value in dataset_stats.items():
        mode, quantity = key[len('tavg_'):].split('_', 1)
        final_log[f'final/test/{safe_name}/{quantity}/tavg_{mode}_nrmse'] = value
for dataset_name, dataset_stats in zip(test_paths, test_physical_nrmse_stats):
    safe_name = dataset_name.replace('/', '_')
    for channel_name, value in dataset_stats.items():
        if channel_name == 'macro':
            final_log[f'final/test/{safe_name}/channelwise_physical_nrmse_macro'] = value
            final_log[f'final/test/{safe_name}/nrmse'] = value
        else:
            final_log[
                f'final/test/{safe_name}/channelwise_physical_nrmse/{channel_name}'
            ] = value
if len(test_physical_nrmse_stats) == 1 and test_physical_nrmse_stats[0]:
    final_log['final/test/nrmse'] = test_physical_nrmse_stats[0]['macro']
final_log['final/rollout_time_per_traj'] = float(np.mean(test_rollout_time_per_trajs)) if len(test_rollout_time_per_trajs) > 0 else 0.0
final_log['final/worst_traj_l2'] = worst_traj_l2 if worst_traj_l2 > -float('inf') else 0.0
final_log['final/worst_traj_idx'] = worst_traj_idx
final_log['final/worst_traj_dataset_id'] = worst_traj_dataset
wandb.log(final_log)
print('final test, l2 step {}, l2 full {}, test time {:.5f}'.format(', '.join(['{:.5f}'.format(val) for val in test_l2_steps]), ', '.join(['{:.5f}'.format(val) for val in test_l2_fulls]), test_time))
print('final rollout inference time per trajectory {}, mean {:.5f}'.format(', '.join(['{:.5f}s'.format(val) for val in test_rollout_time_per_trajs]), final_log['final/rollout_time_per_traj']))
print(f"[TEST] worst trajectory idx = {worst_traj_idx}, dataset_id = {worst_traj_dataset}, worst full-trajectory L2 = {final_log['final/worst_traj_l2']:.5f}")

wandb.finish()
