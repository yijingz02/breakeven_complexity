from nsm.typing import *

from nsm.flow import Flow
from nsm.model import Model
from nsm.writer import Writer

from tqdm import tqdm
import wandb

import time
import numpy as np
import functools as F
from collections import defaultdict as dd
import logging
import itertools
import math

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import lax
import optax

import matplotlib.pyplot as plt

# Sharding (only used if trainer.batch_sharding=True)
try:
  from jax.sharding import Mesh, PartitionSpec as P
  from jax.experimental.shard_map import shard_map
except Exception:
  Mesh = None
  P = None
  shard_map = None


def _to_float(x):
  try:
    return float(x)
  except Exception:
    return float(np.asarray(x))


# ---------------------------------------------------------------------------- #
#                        WARMUP + STEP-TIME ESTIMATION                         #
# ---------------------------------------------------------------------------- #

def _block_until_ready(x):
  """Block on scalar/array/pytree."""
  try:
    return jax.block_until_ready(x)
  except Exception:
    return jax.tree.map(jax.block_until_ready, x)


def estimate_warmed_step_time(step_fn, state, loader, *, n_discard: int = 2, n_timing: int = 8):
  """
  Runs:
    - n_discard steps (compile/cache), not timed
    - n_timing steps, timed (compute only; data loading excluded)
  Returns:
    (new_state, mean_s, median_s, times_list)
  """
  it = iter(loader)

  for _ in range(max(0, int(n_discard))):
    batch = next(it)
    state, stats, _ = step_fn(state, batch)
    _block_until_ready(stats["loss"])

  times = []
  for _ in range(max(1, int(n_timing))):
    batch = next(it)
    t0 = time.perf_counter()
    state, stats, _ = step_fn(state, batch)
    _block_until_ready(stats["loss"])
    t1 = time.perf_counter()
    times.append(t1 - t0)

  mean_s = float(np.mean(times))
  median_s = float(np.median(times))
  return state, mean_s, median_s, times


def make_plateau_then_decay_schedule(
  base_lr: float,
  total_steps: int,
  *,
  plateau_frac: float = 0.10,
  decay: str = "cosine",
  final_lr: float = 0.0,
) -> optax.Schedule:
  """
  LR is constant for first plateau_frac of total_steps, then decays to final_lr.
  """
  total_steps = max(1, int(total_steps))
  plateau_steps = max(1, int(round(total_steps * float(plateau_frac))))
  decay_steps = max(1, total_steps - plateau_steps)

  if decay.lower() == "linear":
    decay_sched = optax.linear_schedule(
      init_value=float(base_lr),
      end_value=float(final_lr),
      transition_steps=decay_steps,
    )
  else:
    alpha = (float(final_lr) / float(base_lr)) if float(base_lr) > 0 else 0.0
    decay_sched = optax.cosine_decay_schedule(
      init_value=float(base_lr),
      decay_steps=decay_steps,
      alpha=float(alpha),
    )

  return optax.join_schedules(
    schedules=[optax.constant_schedule(float(base_lr)), decay_sched],
    boundaries=[plateau_steps],
  )


# ---------------------------------------------------------------------------- #
#                           DATA CAP (TRAIN USAGE)                              #
# ---------------------------------------------------------------------------- #

def _as_array(x):
  # Grid/SEM/etc. -> jnp array
  if hasattr(x, "value"):
    return x.value
  return x


def _get_batch_B(batch_data, fallback: int):
  """
  Infer actual batch size B from batch_data.
  Works for window==1 and window>1 (list of Flow.Data_t).
  """
  try:
    x0 = batch_data[0].ic  # Grid-like
    return int(_as_array(x0).shape[0])
  except Exception:
    return int(fallback)


def initialize_single_frame_history(flow: Flow, ic: Grid, enabled: bool) -> Grid:
  """Repeat the first temporal channel block for flows with bundled history."""
  history_length = int(getattr(flow, "n_steps_input", 1))
  if not enabled or history_length <= 1:
    return ic
  channels = ic.value.shape[-1]
  if channels % history_length:
    raise ValueError(
      f"Cannot split {channels} input channels into {history_length} history frames"
    )
  channels_per_frame = channels // history_length
  first = ic.value[..., :channels_per_frame]
  return Grid(jnp.concatenate([first] * history_length, axis=-1), ic.size)


def cap_train_loader(dataloader, trainer, cfg):
  """
  Supports either:
    cfg.train.max_train_batches: cap batches/epoch
    cfg.train.max_train_examples: cap examples/epoch (converted to batches)
  If both set, uses the stricter one.
  """
  max_batches = getattr(cfg.train, "max_train_batches", None)
  max_examples = getattr(cfg.train, "max_train_examples", None)

  if getattr(trainer, "datacnt", None) is not None:
    max_examples = int(trainer.datacnt) if max_examples is None else min(int(max_examples), int(trainer.datacnt))
    print(f"Capping train data usage per epoch to datacnt={trainer.datacnt} examples.")

  if max_examples is not None:
    max_examples = int(max_examples)
    b_from_examples = max(1, math.ceil(max_examples / int(trainer.batch_size)))
    max_batches = b_from_examples if max_batches is None else min(int(max_batches), b_from_examples)

  if max_batches is not None:
    max_batches = int(max_batches)
    dataloader = itertools.islice(dataloader, max_batches)

  return dataloader, max_batches, max_examples


# ---------------------------------------------------------------------------- #
#                               EVAL L2 METRICS                                 #
# ---------------------------------------------------------------------------- #

def _align_pred_targ(pred, targ):
  """
  Align pred/target for L2 metrics.
  - If same ndim and both look like (..., C) channels, crop to min(C)
  - Then flatten and crop to min(total_elems) as a last resort
  Returns: (pred_flat, targ_flat)
  """
  p = _as_array(pred)
  t = _as_array(targ)

  # First try: align channel dim if it exists (last dim differs)
  if p.ndim == t.ndim and p.ndim >= 3 and p.shape[-1] != t.shape[-1]:
    c = min(p.shape[-1], t.shape[-1])
    p = p[..., :c]
    t = t[..., :c]

  # Last resort: flatten and crop to min length
  pf = jnp.ravel(p)
  tf = jnp.ravel(t)
  n = jnp.minimum(pf.shape[0], tf.shape[0])
  return pf[:n], tf[:n]

def l2_rmse(pred, targ, eps: float = 1e-12):
  pf, tf = _align_pred_targ(pred, targ)
  return jnp.sqrt(jnp.mean(jnp.square(pf - tf)) + eps)

def l2_rel(pred, targ, eps: float = 1e-12):
  pf, tf = _align_pred_targ(pred, targ)
  num = jnp.linalg.norm(pf - tf)
  den = jnp.linalg.norm(tf) + eps
  return num / den

def per_traj_l2_rel_distribution(
  forward,
  variable,
  flow,
  *,
  single_frame_initialization: bool = False,
  eps: float = 1e-12,
  max_traj: int | None = None,
):
  """
  Returns:
    losses: np.ndarray shape (num_traj,), per-trajectory relative L2 over the whole rollout
  Definition:
    rel = ||pred - target||_2 / (||target||_2 + eps), aggregated across time+space(+channels)
  """
  dataset = flow.dataset("test")
  losses = []

  for j, data in enumerate(dataset):
    if max_traj is not None and j >= int(max_traj):
      break

    u = initialize_single_frame_history(
      flow, data[0].ic, single_frame_initialization
    )
    err2 = 0.0
    tar2 = 0.0

    print(len(data), "timesteps in trajectory???")

    for i in range(len(data)):
      # same branching as your eval: handle possible channel mismatch in ic
      if u.value.shape[-1] == data[i].ic.value.shape[-1]:
        out = forward(variable, ic=u)
      else:
        out = forward(variable, ic=data[i].ic)

      u_pred = out.u if isinstance(out.u, Grid) else out.u.eval(data[i].ut.resolution)
      ut = data[i].ut

      pf, tf = _align_pred_targ(u_pred, ut)  # uses your align helper
      pf = np.asarray(jax.device_get(pf))
      tf = np.asarray(jax.device_get(tf))

      diff = pf - tf
      err2 += float(np.sum(diff * diff))
      tar2 += float(np.sum(tf * tf))

      u = u_pred  # autoregressive rollout

    losses.append(np.sqrt(err2 / (tar2 + eps)))

  return np.asarray(losses, dtype=np.float64)


# def l2_rmse(pred, targ, eps: float = 1e-12):
#   """Scalar L2 as RMSE: sqrt(mean((pred - targ)^2))"""
#   p = _as_array(pred)
#   t = _as_array(targ)
#   return jnp.sqrt(jnp.mean(jnp.square(p - t)) + eps)


# def l2_rel(pred, targ, eps: float = 1e-12):
#   """Relative L2: ||pred - targ||_2 / (||targ||_2 + eps)"""
#   p = jnp.ravel(_as_array(pred))
#   t = jnp.ravel(_as_array(targ))
#   num = jnp.linalg.norm(p - t)
#   den = jnp.linalg.norm(t) + eps
#   return num / den


def _flatten_for_wandb(d, prefix=""):
  """Flatten nested dicts to a flat dict suitable for wandb.log."""
  out = {}
  if isinstance(d, dict):
    for k, v in d.items():
      key = f"{prefix}/{k}" if prefix else str(k)
      out.update(_flatten_for_wandb(v, key))
  else:
    try:
      out[prefix] = _to_float(d)
    except Exception:
      try:
        out[prefix] = float(np.asarray(d))
      except Exception:
        pass
  return out


# ---------------------------------------------------------------------------- #
#                                   PROTOCOL                                   #
# ---------------------------------------------------------------------------- #

class Forward(Protocol):
  def __call__(self, params: PyTree, data: DataType) -> Model.Output:
    """Forward pass of the model."""


class GradFn(Protocol):
  def __call__(self, params: PyTree, batch_data: DataType) -> Tuple[PyTree, ...]:
    """Gradient step of the model."""


class TrainFn(Protocol):
  def __call__(self, flow: Flow, model: Model, writer: Writer):
    """Training function."""


class EvalFn(Protocol):
  def __call__(self, params: PyTree) -> PyTree:
    """Evaluation function."""


# ---------------------------------------------------------------------------- #
#                                    TRAINER                                   #
# ---------------------------------------------------------------------------- #

@dataclass
class Trainer:
  epoch: int
  iteration: Maybe[int]

  learning_rate: float
  scheduler: optax.Schedule

  # Rebuilt after warm step-time inference
  optimizer: optax.GradientTransformation

  batch_size: int
  datacnt: int
  vmap_batch: Maybe[int]
  batch_sharding: bool
  num_cpu_proc: Maybe[int]

  window: Maybe[int]
  gradient_clip: Maybe[float]

  inference_batch_size: int = 16
  use_autoregressive: bool = False
  single_frame_initialization: bool = False
  optimizer_name: str = "adamw"
  optimizer_kwargs: dict = field(default_factory=dict)

  @struct.dataclass
  class State:
    prng: Array
    variable: PyTree
    optim: optax.OptState

  # ----------------------- optimizer builder ----------------------- #

  def build_optimizer(self, schedule: optax.Schedule) -> optax.GradientTransformation:
    """
    Build optimizer that captures the given schedule via optax.scale_by_schedule.
    Adjust this stack to match your existing setup if needed.
    """
    name = (self.optimizer_name or "adamw").lower()
    kw = dict(self.optimizer_kwargs or {})

    if name == "adamw":
      b1 = kw.get("b1", 0.9)
      b2 = kw.get("b2", 0.999)
      eps = kw.get("eps", 1e-8)
      wd  = kw.get("weight_decay", 0.0)

      return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.add_decayed_weights(wd) if wd and wd > 0 else optax.identity(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
      )

    if name == "adam":
      b1 = kw.get("b1", 0.9)
      b2 = kw.get("b2", 0.999)
      eps = kw.get("eps", 1e-8)
      return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
      )

    raise ValueError(f"Unknown optimizer_name={self.optimizer_name}")

  # ---------------------------------------------------------------- #

  def init(self, flow: Flow, model: Model, prng: Array) -> State:
    prng, subkey = jrand.split(prng)
    u, *_ = next(iter(flow.dataset("test")))

    ic = initialize_single_frame_history(
      flow, u.ic, self.single_frame_initialization
    )
    print(model.tabulate(subkey, flow, ic, None))
    variable = model.init(subkey, flow, ic, None)

    optim = self.optimizer.init(variable["params"])
    return self.State(prng, variable, optim)

  def loss(self, flow: Flow, model: Model, params: PyTree, ic: DataType, ut: Grid):
    t0 = time.time()
    out = model.apply({ "params": params }, flow, ic, out=ut, return_aux=config.debug)
    t1 = time.time()
    print(f"model applying time: {t1-t0}")

    u = out.u.eval(ut.resolution) if isinstance(out.u, SEM) else out.u

    metric = { "loss": flow.loss_data(u, ut) }
    for key, value in utils.flatten(out.aux).items():
      metric[f"aux/{key}"] = value

    return metric["loss"], (out, metric)

  def step(self, flow: Flow, model: Model, state: State, batch_data: Flow.Data) -> Tuple[PyTree, ...]:
    variable = state.variable
    params = variable["params"]

    prng, subkey = jrand.split(state.prng)
    del subkey  # unused

    t0 = time.time()
    grad, metric = self.grad(flow, model)(params, batch_data)
    t1 = time.time()

    print(f"train step time: {t1 - t0}")

    stats = {
      "loss": jnp.mean(metric["loss"]),
      "norm": (norm := utils.norm(grad)),
    }
    for key, values in metric.items():
      metric[key] = { f"t={t * flow.t}": x for t, x in enumerate(values, 1) }

    if self.gradient_clip is not None:
      clip = jnp.array(self.gradient_clip, norm.dtype)
      grad = jax.tree.map(
        F.partial(lax.cond, norm < clip, lambda x: x, lambda x: x / norm * clip),
        grad,
      )

    if config.debug:
      aux = dd(lambda: {})
      for name, values in grad.items():
        cat = name.split("_", 1)[0]
        for key, value in utils.flatten(values).items():
          aux[cat][f"{name}/{key}"] = jnp.linalg.norm(jnp.ravel(value))

      metric["norm"] = {}
      for cat, norms in aux.items():
        xs = jnp.array([x for x in norms.values()])
        metric["norm"][cat] = jnp.linalg.norm(xs)
        metric[f"norm/{cat}"] = norms

    with jax.numpy_dtype_promotion("standard"):
      updates, optim = self.optimizer.update(grad, state.optim, params)
      variable["params"] = optax.apply_updates(params, updates)

    return self.State(prng, variable, optim), stats, metric

  # --------------------------------- GRAD --------------------------------- #

  def grad(self, flow: Flow, model: Model) -> GradFn:
    if self.window == 1:

      def batch_grad(params: PyTree, batch_data: Flow.Data):
        grad_t = jax.grad(F.partial(self.loss, flow, model), has_aux=True)
        batch_ic = initialize_single_frame_history(
          flow, batch_data[0].ic, self.single_frame_initialization
        )
        grad, (_, metric) = utils.vmap(F.partial(grad_t, params), self.vmap_batch)(
          batch_ic, batch_data[0].ut
        )
        grad, metric = jax.tree.map(F.partial(jnp.mean, axis=0), (grad, metric))
        if self.batch_sharding:
          grad, metric = lax.pmean((grad, metric), axis)
        return grad, { key: value[None] for key, value in metric.items() }

    else:

      def batch_loss(params: PyTree, batch_ic: Grid, batch_ut: Grid) -> PyTree:
        loss = jax.checkpoint(F.partial(self.loss, flow, model, params))
        return utils.vmap(loss, self.vmap_batch)(batch_ic, batch_ut)

      @F.partial(jax.grad, has_aux=True)
      def batch_grad(params: PyTree, batch_data: List[Flow.Data]):
        device_batch = self.batch_size // jax.local_device_count()
        logging.info(f"Device batch size: {device_batch}")

        @struct.dataclass
        class ScanState:
          u: DataType
          loss: PyTree

        @jax.checkpoint
        def next(carry: ScanState, batch_data_t: Flow.Data_t) -> Tuple[ScanState, PyTree]:
          loss, (out, metric) = batch_loss(params, carry.u, batch_data_t.ut)
          return ScanState(out.u, carry.loss + jnp.mean(loss)), jax.tree.map(jnp.mean, metric)

        xs = jax.tree.map(lambda *xs: jnp.stack(xs), *batch_data)
        initial_ic = initialize_single_frame_history(
          flow, batch_data[0].ic, self.single_frame_initialization
        )
        scan, metric = lax.scan(next, ScanState(initial_ic, 0.), xs)

        loss = scan.loss / len(batch_data)
        result = (loss, metric)
        if self.batch_sharding:
          result = lax.pmean(result, axis)
        return result

    if self.batch_sharding:
      if Mesh is None or shard_map is None or P is None:
        raise RuntimeError("batch_sharding=True but shard_map/Mesh/PartitionSpec imports failed.")
      mesh = Mesh(jax.devices("gpu"), axis := "batch")
      batch_grad = shard_map(
        batch_grad,
        mesh=mesh,
        in_specs=(P(None), P(axis)),
        out_specs=P(None),
      )
      logging.info(f"Sharded over {jax.local_device_count()} device(s): {jax.local_devices()}")

    return batch_grad

  # ---------------------------------------------------------------------------- #
  #                                     TRAIN                                    #
  # ---------------------------------------------------------------------------- #

  @classmethod
  def make_train(cls, cfg: ConfigDict) -> TrainFn:
    def train(flow: Flow, model: Model, writer: Writer):

      train_cfg = cfg.train.to_dict()
      _discard_budget_steps_cfg = int(train_cfg.pop("discard_budget_steps", 0) or 0)
      _time_budget_s_cfg = train_cfg.pop("time_budget_s", None)
      _data_gen_time = train_cfg.pop("data_gen_time", None)
      # _datacnt = train_cfg.pop("datacnt", None)

      trainer = cls(**train_cfg)
      setattr(
        flow,
        "single_frame_initialization",
        trainer.single_frame_initialization,
      )
      trainer.discard_budget_steps = _discard_budget_steps_cfg

      time_budget_s_for_name = _time_budget_s_cfg

      _time_budget_s_cfg -= float(_data_gen_time) * float(trainer.datacnt)

      trainer.time_budget_s = _time_budget_s_cfg
      trainer.data_gen_time = _data_gen_time
      
      # trainer.datacnt = _datacnt

      prng = jrand.PRNGKey(cfg.seed)

      datacnt_for_name = int(trainer.datacnt)

      run_name   = f"N{datacnt_for_name}_B{int(time_budget_s_for_name)}"
      group_name = f"B{int(time_budget_s_for_name)}"
      wandb_project = getattr(cfg.log, "wandb_project", "eddyformer")
      wandb_dir = cfg.log.save_path

      wb = False
      try:
        wandb.init(
          project=wandb_project,
          name=run_name,
          group=group_name,
          dir=wandb_dir,
        )
        wb = True
      except Exception:
        wb = False

      try:
        wandb.config.update(cfg.to_dict(), allow_val_change=True)
      except Exception:
        pass

      # ------------------ Load / init state ------------------ #
      if cfg.log.load_ckpt_it is not None:
        try:
          it, state = writer.load_checkpoint(cfg.log.load_ckpt_it)
        except FileNotFoundError:
          prng, subkey = jrand.split(prng)
          it, state = 0, trainer.init(flow, model, subkey)
      else:
        prng, subkey = jrand.split(prng)
        it, state = 0, trainer.init(flow, model, subkey)

      # ------------------ Build step/forward ------------------ #
      def make_step_and_forward():
        step_fn = F.partial(trainer.step, flow, model)
        forward_fn = F.partial(model.apply, flow=flow)
        if cfg.compile:
          step_fn = utils.jit(step_fn, donate_argnums=0)
          forward_fn = jax.jit(forward_fn, donate_argnames="ic")
        if config.debug:
          from nvtx import annotate
          step_fn = annotate("step", "blue")(step_fn)
          forward_fn = annotate("forward", "red")(forward_fn)
        return step_fn, forward_fn

      # ------------------------------------------------------------------ #
      # Phase A: warm + measure warmed step time (NOT counted against budget)
      # ------------------------------------------------------------------ #
      orig_batch_sharding = trainer.batch_sharding
      trainer.batch_sharding = False
      
      step_tmp, forward_tmp = make_step_and_forward()

      warm_timing_steps = int(getattr(cfg.train, "warm_timing_steps", 8) or 8)
      warm_discard_steps = int(getattr(cfg.train, "warm_discard_steps", (3 if cfg.compile else 1)) or 0)

      prng, subkey = jrand.split(prng)
      warm_dataset = flow.dataset("train", subkey, trainer.window, 64, trainer.num_cpu_proc)
      warm_loader = utils.batched(warm_dataset, trainer.batch_size)
      step_tmp, forward_tmp = make_step_and_forward()

      state, warmed_mean_s, warmed_median_s, warm_times = estimate_warmed_step_time(
        step_tmp, state, warm_loader,
        n_discard=warm_discard_steps,
        n_timing=warm_timing_steps,
      )

      # trainer.batch_sharding = orig_batch_sharding

      # ------------------------------------------------------------------ #
      # Infer total steps and rebuild schedule so decay starts after 10%
      # ------------------------------------------------------------------ #
      time_budget_s = getattr(cfg.train, "time_budget_s", None)
      discard_budget_steps = int(getattr(cfg.train, "discard_budget_steps", 0) or 0)

      if cfg.compile:
        discard_budget_steps = max(discard_budget_steps, 1)

      planned_total_steps = None
      if trainer.iteration is not None:
        planned_total_steps = int(trainer.iteration)
      elif time_budget_s is not None:
        per_step = max(1e-6, float(warmed_median_s))
        planned_total_steps = max(1, int(int(it) + max(1, int(float(time_budget_s) // per_step))))

      if planned_total_steps is not None:
        base_lr = float(trainer.learning_rate)
        decay_kind = getattr(cfg.train, "lr_decay", "cosine")
        final_lr = float(getattr(cfg.train, "final_lr", 0.0) or 0.0)

        core_sched = make_plateau_then_decay_schedule(
          base_lr=base_lr,
          total_steps=int(planned_total_steps),
          plateau_frac=0.10,  # <-- start decay after 10%
          decay=decay_kind,
          final_lr=final_lr,
        )

        it_offset = int(it)

        def sched_with_offset(step_i):
          # keep JAX-traceable
          step_i = step_i + jnp.asarray(it_offset, dtype=step_i.dtype)
          return core_sched(step_i)

        trainer.scheduler = sched_with_offset

        # Rebuild optimizer to capture schedule
        new_optimizer = trainer.build_optimizer(trainer.scheduler)

        # Keep old opt state if structure matches (useful for resume)
        params = state.variable["params"]
        new_opt_state = new_optimizer.init(params)
        try:
          old_struct = jax.tree_util.tree_structure(state.optim)
          new_struct = jax.tree_util.tree_structure(new_opt_state)
          optim_state = state.optim if old_struct == new_struct else new_opt_state
        except Exception:
          optim_state = new_opt_state

        trainer.optimizer = new_optimizer
        state = trainer.State(state.prng, state.variable, optim_state)

      logging.info(
        f"Warmup done: mean_step={warmed_mean_s:.4f}s median_step={warmed_median_s:.4f}s "
        f"planned_total_steps={planned_total_steps} it={int(it)}"
      )

      # Log warm diagnostics
      try:
        writer.write(
          int(it),
          dict(
            time=dict(
              warm_mean_s=float(warmed_mean_s),
              warm_median_s=float(warmed_median_s),
              warm_timing_steps=int(warm_timing_steps),
              warm_discard_steps=int(warm_discard_steps),
              planned_total_steps=(int(planned_total_steps) if planned_total_steps is not None else None),
            )
          ),
        )
      except Exception:
        pass

      if wb:
        try:
          wandb.log(
            {
              "time/warm_mean_s": float(warmed_mean_s),
              "time/warm_median_s": float(warmed_median_s),
              "time/planned_total_steps": (int(planned_total_steps) if planned_total_steps is not None else None),
            },
            step=int(it),
          )
        except Exception:
          pass

      # ------------------------------------------------------------------ #
      # Phase B: rebuild step/forward so JIT captures updated optimizer/sched
      # ------------------------------------------------------------------ #
      step, forward = make_step_and_forward()
      eval_fn = cls.make_eval(
        forward,
        flow,
        trainer.inference_batch_size,
        trainer.single_frame_initialization,
      )

      # ----------------------------------- RUN ----------------------------------- #

      compute_spent_s = 0.0
      counted_steps = 0
      stop_reason = None

      used_train_examples_total = 0
      used_train_batches_total = 0

      # One extra compile/cache step that is NOT counted against budget.
      if cfg.compile and discard_budget_steps > 0:
        prng, subkey = jrand.split(prng)
        warm_dataset2 = flow.dataset("train", subkey, trainer.window, 64, trainer.num_cpu_proc)
        warm_loader2 = utils.batched(warm_dataset2, trainer.batch_size)
        warm_batch2 = next(iter(warm_loader2))
        state, stats, _ = step(state, warm_batch2)
        _block_until_ready(stats["loss"])
        discard_budget_steps -= 1

      budget_flag = False

      last_save = time.time()
      for epoch in range(trainer.epoch):

        prng, subkey = jrand.split(prng)
        dataset = flow.dataset("train", subkey, trainer.window, 64, trainer.num_cpu_proc)
        dataloader = utils.batched(dataset, trainer.batch_size)

        max_batches = int(math.ceil(trainer.datacnt / trainer.batch_size))
        dataloader = itertools.islice(dataloader, max_batches)

        if config.tqdm:
          dataloader = tqdm(
            dataloader,
            f"Train epoch {epoch}",
            initial=it,
            total=max_batches,   # <- use the cap you just computed
          )

        # dataloader, capped_batches, capped_examples = cap_train_loader(dataloader, trainer, cfg)

        # if config.tqdm:
        #   dataloader = tqdm(
        #     dataloader,
        #     f"Train epoch {epoch}",
        #     initial=it,
        #     total=capped_batches if capped_batches is not None else None,
        #   )

        for it, batch_data in enumerate(dataloader, it):

          if trainer.iteration and it >= trainer.iteration:
            stop_reason = f"iteration cap reached (it={it} >= {trainer.iteration})"
            break

          if time_budget_s is not None and compute_spent_s >= float(time_budget_s):
            stop_reason = (
              f"time budget reached (spent={compute_spent_s:.2f}s >= {float(time_budget_s):.2f}s)"
            )
            budget_flag = True
            break

          # --------- eval + log (includes L2) ---------
          if cfg.log.test_period and it % cfg.log.test_period == 0:
            metric = eval_fn(state.variable)
            metric = jax.device_get(metric)
            writer.write(it, metric)
            logging.info(f"Iter {it}: {metric}")

            if wb:
              wb_eval = _flatten_for_wandb(metric, prefix="eval")
              wandb.log(wb_eval, step=int(it))

          now = time.time()
          if cfg.log.ckpt_period and (now - last_save >= cfg.log.ckpt_period):
            writer.save_checkpoint(it, state)
            last_save = now

          # track used data
          B_actual = _get_batch_B(batch_data, fallback=trainer.batch_size)
          used_train_examples_total += int(B_actual)
          used_train_batches_total += 1

          # train step timing (compute only)
          t0 = time.perf_counter()
          state, stats, metric = step(state, batch_data)
          _block_until_ready(stats["loss"])
          t1 = time.perf_counter()

          print(f"training step time: {t1-t0}")

          step_compute_s = t1 - t0

          if discard_budget_steps > 0:
            discard_budget_steps -= 1
          else:
            compute_spent_s += step_compute_s
            counted_steps += 1

          stats_host, metric_host = jax.device_get((stats, metric))

          writer.write(
            it,
            dict(
              stats=stats_host,
              time=dict(
                step_compute_s=float(step_compute_s),
                compute_spent_s=float(compute_spent_s),
                counted_steps=int(counted_steps),
                budget_s=(float(time_budget_s) if time_budget_s is not None else None),
                warm_mean_s=float(warmed_mean_s),
                warm_median_s=float(warmed_median_s),
                planned_total_steps=(int(planned_total_steps) if planned_total_steps is not None else None),
              ),
              data=dict(
                used_train_examples_total=int(used_train_examples_total),
                used_train_batches_total=int(used_train_batches_total),
                # max_train_examples=(int(capped_examples) if capped_examples is not None else None),
                # max_train_batches=(int(capped_batches) if capped_batches is not None else None),
              ),
            )
            | metric_host
          )

          if wb:
            wb_dict = {
              "train/loss": _to_float(stats_host["loss"]),
              "train/grad_norm": _to_float(stats_host["norm"]),
              "time/step_compute_s": float(step_compute_s),
              "time/compute_spent_s": float(compute_spent_s),
              "epoch": int(epoch),
              "data/used_train_examples_total": int(used_train_examples_total),
              "data/used_train_batches_total": int(used_train_batches_total),
              # "data/max_train_examples": (int(capped_examples) if capped_examples is not None else None),
              # "data/max_train_batches": (int(capped_batches) if capped_batches is not None else None),
            }
            wandb.log(wb_dict, step=int(it))

          if config.tqdm:
            dataloader.set_postfix(stats)
          else:
            logging.info(
              f"Epoch {epoch}: {it}it - {stats_host} (compute_spent_s={compute_spent_s:.2f})"
            )

        else:
          it += 1
          continue
        break

      if stop_reason:
        logging.info(f"Training stopped: {stop_reason}")

      final_eval = jax.device_get(eval_fn(state.variable))
      writer.write(int(it), {"eval": final_eval} if isinstance(final_eval, dict) else final_eval)
      logging.info(f"Final eval @ it={int(it)}: {final_eval}")

      if wb:
        final_wandb_metrics = _flatten_for_wandb(final_eval, prefix="eval")
        final_wandb_metrics["test/dnrmse"] = float(final_eval["dnrmse"])
        wandb.log(final_wandb_metrics, step=int(it))

      # ---- Per-trajectory L2 distribution + worst-case logging ----
      if wb:
        # optionally cap how many test traj to evaluate (None = all)
        max_eval_traj = getattr(cfg.log, "eval_max_traj", None)

        losses = per_traj_l2_rel_distribution(
          forward,
          state.variable,
          flow,
          single_frame_initialization=trainer.single_frame_initialization,
          max_traj=max_eval_traj,
        )
        if losses.size > 0:
          worst_idx = int(np.argmax(losses))
          worst_val = float(losses[worst_idx])

          # 1) Scalars
          wandb.log(
            {
              "eval/worst_traj_l2_rel": worst_val,
              "eval/worst_traj_idx": worst_idx,
              "eval/num_eval_traj": int(losses.size),
            },
            step=int(it),
          )

          # 2) Histogram (W&B-native)
          wandb.log(
            {
              "eval/l2_rel_hist": wandb.Histogram(losses, num_bins=60),
            },
            step=int(it),
          )

          # 3) Matplotlib figure
          fig = plt.figure()
          lo = max(losses.min(), 1e-12)
          hi = max(losses.max(), lo * 1.01)
          bins = np.logspace(np.log10(lo), np.log10(hi), 60)
          plt.hist(losses, bins=bins)
          plt.xscale("log")
          plt.xlabel("Per-trajectory relative L2 loss (log x)")
          plt.ylabel("Count")
          plt.title("GS loss dist (log x)")

          try:
            from scipy.stats import gaussian_kde

            # KDE in log10 space
            z = np.log10(losses)
            kde = gaussian_kde(z)

            # Evaluate on a dense grid in log space
            z_grid = np.linspace(np.log10(lo), np.log10(hi), 400)
            x_grid = 10 ** z_grid

            # Convert density in z-space to expected counts per (log) bin
            z_bins = np.log10(bins)
            dz = float(z_bins[1] - z_bins[0])  # constant because bins are log-spaced
            y_counts = losses.size * kde(z_grid) * dz

            plt.plot(x_grid, y_counts)  # default color; if you want orange, set color="orange"
          except Exception:
            # Fallback: smooth the histogram counts (no scipy)
            centers = np.sqrt(bins[:-1] * bins[1:])  # geometric centers for log bins
            kernel = np.ones(7) / 7.0
            smooth = np.convolve(counts, kernel, mode="same")
            plt.plot(centers, smooth)

          wandb.log({"eval/l2_rel_dist_plot": wandb.Image(fig)}, step=int(it))
          plt.close(fig)

          # 4) Optional: top-K worst table
          topk = min(10, losses.size)
          idxs = np.argsort(-losses)[:topk]
          table = wandb.Table(columns=["traj_idx", "l2_rel"])
          for k in idxs:
            table.add_data(int(k), float(losses[k]))
          wandb.log({"eval/worst_traj_table": table}, step=int(it))

      writer.save_checkpoint(it, state)
      writer.finish()

      if wb:
        wandb.finish()

    return train

  # ---------------------------------------------------------------------------- #
  #                                     EVAL                                     #
  # ---------------------------------------------------------------------------- #

  @classmethod
  def make_eval(
    cls,
    f: Forward,
    flow: Flow,
    batch_size: int = 1,
    single_frame_initialization: bool = False,
  ) -> EvalFn:
    def roll(variable: PyTree, data: Flow.Data) -> PyTree:
      u = initialize_single_frame_history(
        flow, data[0].ic, single_frame_initialization
      )
      metric = dd(lambda: [])
      dnrmse_numerator = None
      dnrmse_denominator = None

      print(len(data), "timesteps in eval rollout???")

      for i in range(len(data)):
        if u.value.shape[-1] == data[i].ic.value.shape[-1]:
          out = f(variable, ic=u)
        else:
          out = f(variable, ic=data[i].ic)

        u = out.u if isinstance(out.u, Grid) else out.u.eval(data[i].ut.resolution)
        ut = data[i].ut

        channels = min(u.value.shape[-1], ut.value.shape[-1])
        pred_value = u.value[..., :channels]
        ref_value = ut.value[..., :channels]
        reduction_dims = tuple(range(1, ref_value.ndim - 1))
        channel_diff2 = jnp.square(pred_value - ref_value).sum(axis=reduction_dims)
        channel_ref2 = jnp.square(ref_value).sum(axis=reduction_dims)
        if dnrmse_numerator is None:
          dnrmse_numerator = channel_diff2
          dnrmse_denominator = channel_ref2
        else:
          dnrmse_numerator += channel_diff2
          dnrmse_denominator += channel_ref2

        metric_t = flow.metric(u, ut)
        # --- NEW: L2 metrics ---
        metric_t["l2_rmse"] = l2_rmse(u, ut)
        metric_t["l2_rel"]  = l2_rel(u, ut)

        for key, value in metric_t.items():
          metric[key].append(jax.tree.map(float, value))

      result = dict(metric)
      result["dnrmse_samples"] = jnp.sqrt(
        dnrmse_numerator / jnp.maximum(dnrmse_denominator, 1e-12)
      ).mean(axis=-1)
      return result

    def eval(variable: PyTree) -> PyTree:
      dataset = flow.dataset("test")
      dataset = utils.batched(dataset, batch_size, drop_last=False)
      if config.tqdm:
        dataset = tqdm(dataset, "Test")

      metrics = dd(lambda: [])
      dnrmse_samples = []
      for data in dataset:
        rolled = roll(variable, data)
        dnrmse_samples.extend(
          np.asarray(jax.device_get(rolled.pop("dnrmse_samples"))).reshape(-1)
        )
        for key, values in rolled.items():
          for t, value in enumerate(values):
            metrics[f"{key}_avg"].append(value)
            metrics[f"{key}/{t}"].append(value)

      mean = lambda *x: np.mean(x).item()
      metrics_out = { key: jax.tree.map(mean, *value) for key, value in metrics.items() }
      metrics_out["dnrmse"] = float(np.mean(dnrmse_samples))
      return metrics_out

    return eval
