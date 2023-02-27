# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library to training NeRFs."""
import functools
from typing import Any, Callable, Dict

from absl import logging
import flax
from flax import struct
from flax import traverse_util
from flax.training import checkpoints
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import vmap

from hypernerf import model_utils
from hypernerf import models
from hypernerf import utils


@struct.dataclass
class ScalarParams:
  """Scalar parameters for training."""
  learning_rate: float
  elastic_loss_weight: float = 0.0
  warp_reg_loss_weight: float = 0.0
  warp_reg_loss_alpha: float = -2.0
  warp_reg_loss_scale: float = 0.001
  background_loss_weight: float = 0.0
  background_noise_std: float = 0.001
  hyper_reg_loss_weight: float = 0.0

  # eventHyperNerf loss hyperparams
  event_loss_weight: float = 1.0
  rgb_loss_weight: float = 1.0

def save_checkpoint(path, state, keep=2):
  """Save the state to a checkpoint."""
  state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
  step = state_to_save.optimizer.state.step
  checkpoint_path = checkpoints.save_checkpoint(
      path, state_to_save, step, keep=keep)
  logging.info('Saved checkpoint: step=%d, path=%s', int(step), checkpoint_path)
  return checkpoint_path


def zero_adam_param_states(state: flax.optim.OptimizerState, selector: str):
  """Applies a gradient for a set of parameters.

  Args:
    state: a named tuple containing the state of the optimizer
    selector: a path string defining which parameters to freeze.

  Returns:
    A tuple containing the new parameters and the new optimizer state.
  """
  step = state.step
  params = flax.core.unfreeze(state.param_states)
  flat_params = {'/'.join(k): v
                 for k, v in traverse_util.flatten_dict(params).items()}
  for k in flat_params:
    if k.startswith(selector):
      v = flat_params[k]
      # pylint: disable=protected-access
      flat_params[k] = flax.optim.adam._AdamParamState(
          jnp.zeros_like(v.grad_ema), jnp.zeros_like(v.grad_sq_ema))

  new_param_states = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_params.items()})
  new_param_states = dict(flax.core.freeze(new_param_states))
  new_state = flax.optim.OptimizerState(step, new_param_states)
  return new_state


@jax.jit
def nearest_rotation_svd(matrix, eps=1e-6):
  """Computes the nearest rotation using SVD."""
  # TODO(keunhong): Currently this produces NaNs for some reason.
  u, _, vh = jnp.linalg.svd(matrix + eps, compute_uv=True, full_matrices=False)
  # Handle the case when there is a flip.
  # M will be the identity matrix except when det(UV^T) = -1
  # in which case the last diagonal of M will be -1.
  det = jnp.linalg.det(utils.matmul(u, vh))
  m = jnp.stack([jnp.ones_like(det), jnp.ones_like(det), det], axis=-1)
  m = jnp.diag(m)
  r = utils.matmul(u, utils.matmul(m, vh))
  return r


def compute_elastic_loss(jacobian, eps=1e-6, loss_type='log_svals'):
  """Compute the elastic regularization loss.

  The loss is given by sum(log(S)^2). This penalizes the singular values
  when they deviate from the identity since log(1) = 0.0,
  where D is the diagonal matrix containing the singular values.

  Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.

  Returns:
    The elastic regularization loss.
  """
  if loss_type == 'log_svals':
    svals = jnp.linalg.svd(jacobian, compute_uv=False)
    log_svals = jnp.log(jnp.maximum(svals, eps))
    sq_residual = jnp.sum(log_svals**2, axis=-1)
  elif loss_type == 'svals':
    svals = jnp.linalg.svd(jacobian, compute_uv=False)
    sq_residual = jnp.sum((svals - 1.0)**2, axis=-1)
  elif loss_type == 'jtj':
    jtj = utils.matmul(jacobian, jacobian.T)
    sq_residual = ((jtj - jnp.eye(3)) ** 2).sum() / 4.0
  elif loss_type == 'div':
    div = utils.jacobian_to_div(jacobian)
    sq_residual = div ** 2
  elif loss_type == 'det':
    det = jnp.linalg.det(jacobian)
    sq_residual = (det - 1.0) ** 2
  elif loss_type == 'log_det':
    det = jnp.linalg.det(jacobian)
    sq_residual = jnp.log(jnp.maximum(det, eps)) ** 2
  elif loss_type == 'nr':
    rot = nearest_rotation_svd(jacobian)
    sq_residual = jnp.sum((jacobian - rot) ** 2)
  else:
    raise NotImplementedError(
        f'Unknown elastic loss type {loss_type!r}')
  residual = jnp.sqrt(sq_residual)
  loss = utils.general_loss_with_squared_residual(
      sq_residual, alpha=-2.0, scale=0.03)
  return loss, residual


@functools.partial(jax.jit, static_argnums=0)
def compute_background_loss(model, state, params, key, points, noise_std,
                            alpha=-2, scale=0.001):
  """Compute the background regularization loss."""
  metadata = random.choice(key, model.warp_embeds, shape=(points.shape[0], 1))
  point_noise = noise_std * random.normal(key, points.shape)
  points = points + point_noise
  warp_fn = functools.partial(model.apply, method=model.apply_warp)
  warp_fn = jax.vmap(warp_fn, in_axes=(None, 0, 0, None))
  warp_out = warp_fn({'params': params}, points, metadata, state.extra_params)
  warped_points = warp_out['warped_points'][..., :3]
  sq_residual = jnp.sum((warped_points - points)**2, axis=-1)
  loss = utils.general_loss_with_squared_residual(
      sq_residual, alpha=alpha, scale=scale)
  return loss


@functools.partial(jax.jit,
                   static_argnums=0,
                   static_argnames=('disable_hyper_grads',
                                    'grad_max_val',
                                    'grad_max_norm',
                                    'use_elastic_loss',
                                    'elastic_reduce_method',
                                    'elastic_loss_type',
                                    'use_background_loss',
                                    'use_warp_reg_loss',
                                    'use_hyper_reg_loss'))
def train_step(model: models.NerfModel,
               rng_key: Callable[[int], jnp.ndarray],
               state: model_utils.TrainState,
               batch: Dict[str, Any],
               scalar_params: ScalarParams,
               disable_hyper_grads: bool = False,
               grad_max_val: float = 0.0,
               grad_max_norm: float = 0.0,
               use_elastic_loss: bool = False,
               elastic_reduce_method: str = 'median',
               elastic_loss_type: str = 'log_svals',
               use_background_loss: bool = False,
               use_warp_reg_loss: bool = False,
               use_hyper_reg_loss: bool = False):
  """One optimization step.

  Args:
    model: the model module to evaluate.
    rng_key: The random number generator.
    state: model_utils.TrainState, state of model and optimizer.
    batch: dict. A mini-batch of data for training.
    scalar_params: scalar-valued parameters.
    disable_hyper_grads: if True disable gradients to the hyper coordinate
      branches.
    grad_max_val: The gradient clipping value (disabled if == 0).
    grad_max_norm: The gradient clipping magnitude (disabled if == 0).
    use_elastic_loss: is True use the elastic regularization loss.
    elastic_reduce_method: which method to use to reduce the samples for the
      elastic loss. 'median' selects the median depth point sample while
      'weight' computes a weighted sum using the density weights.
    elastic_loss_type: which method to use for the elastic loss.
    use_background_loss: if True use the background regularization loss.
    use_warp_reg_loss: if True use the warp regularization loss.
    use_hyper_reg_loss: if True regularize the hyper points.

  Returns:
    new_state: model_utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
  """
  # rng_key, fine_key, coarse_key, reg_key = random.split(rng_key, 4)
  rng_key, fk_prev, ck_prev, \
           fk_next, ck_next, \
           fk_col, ck_col, reg_key = random.split(rng_key, 8)

  # pylint: disable=unused-argument
  def _compute_loss_and_stats(
      params, prev_out, next_out, rgb_out,
      level,
      use_elastic_loss=False,
      use_hyper_reg_loss=False):
    
    def to_gray(img):
      c2g_vec = jnp.array([0.2989 , 0.5870 , 0.1140 ]).reshape(-1,1)
      grey_img = img@c2g_vec
      return grey_img

    stats = {}
    loss = 0

    if not "None" in prev_out:
      rgb_next = next_out['rgb'][..., :3]
      rgb_prev = prev_out['rgb'][..., :3]

      if rgb_next.shape[-1] == 3:
        rgb_prev, rgb_next = to_gray(rgb_prev), to_gray(rgb_next)
      
      # delta_log = jnp.log(rgb_next) - jnp.log(rgb_prev)
      # e_t = batch["events"][..., :3]
      # c_thresh = 0.075
      # event_loss = jax.nn.relu(delta_log - e_t - c_thresh)**2 + \
      #              jax.nn.relu(e_t - delta_log - c_thresh)**2

      # event_loss = scalar_params.event_loss_weight * event_loss.mean()

      event_loss = scalar_params.event_loss_weight*\
                      ((jnp.log(rgb_next + 1e-7) - \
                      jnp.log(rgb_prev + 1e-7) - batch['evs_data']['evs'][..., :3])**2).mean()

      stats["loss/event"] = event_loss
      loss = loss + event_loss

    if not "None" in rgb_out:
      rgb_target = batch['col_data']['rgb'][..., :3]
      rgb_curr = rgb_out['rgb'][..., :3]
      assert rgb_curr.shape[-1] == rgb_target.shape[-1], \
              "model and image channel different; model:%s, img:%s"%(rgb_curr.shape[-1], rgb_target.shape[-1])
      rgb_loss_raw = ((rgb_curr - rgb_target)**2).mean()
      rgb_curr_loss = scalar_params.rgb_loss_weight*rgb_loss_raw
      stats["loss/rgb"] = rgb_curr_loss
      loss = loss + rgb_curr_loss

    if use_elastic_loss:
      elastic_fn = functools.partial(compute_elastic_loss,
                                     loss_type=elastic_loss_type)
      v_elastic_fn = jax.jit(vmap(vmap(jax.jit(elastic_fn))))
      weights = lax.stop_gradient(model_out['weights'])
      jacobian = model_out['warp_jacobian']
      # Pick the median point Jacobian.
      if elastic_reduce_method == 'median':
        depth_indices = model_utils.compute_depth_index(weights)
        jacobian = jnp.take_along_axis(
            # Unsqueeze axes: sample axis, Jacobian row, Jacobian col.
            jacobian, depth_indices[..., None, None, None], axis=-3)
      # Compute loss using Jacobian.
      elastic_loss, elastic_residual = v_elastic_fn(jacobian)
      # Multiply weight if weighting by density.
      if elastic_reduce_method == 'weight':
        elastic_loss = weights * elastic_loss
      elastic_loss = elastic_loss.sum(axis=-1).mean()
      stats['loss/elastic'] = elastic_loss
      stats['residual/elastic'] = jnp.mean(elastic_residual)
      loss += scalar_params.elastic_loss_weight * elastic_loss

    if use_warp_reg_loss:
      def calc_warp_reg_loss(model_out):
        weights = lax.stop_gradient(model_out['weights'])
        depth_indices = model_utils.compute_depth_index(weights)
        warp_mag = ((model_out['points']
                    - model_out['warped_points'][..., :3]) ** 2).sum(axis=-1)
        warp_reg_residual = jnp.take_along_axis(
            warp_mag, depth_indices[..., None], axis=-1)
        warp_reg_loss = utils.general_loss_with_squared_residual(
            warp_reg_residual,
            alpha=scalar_params.warp_reg_loss_alpha,
            scale=scalar_params.warp_reg_loss_scale).mean()
        stats['loss/warp_reg'] = warp_reg_loss
        stats['residual/warp_reg'] = jnp.mean(jnp.sqrt(warp_reg_residual))
        return warp_reg_loss
      # loss += scalar_params.warp_reg_loss_weight * warp_reg_loss
      loss += scalar_params.warp_reg_loss_weight * (calc_warp_reg_loss(prev_out) + calc_warp_reg_loss(next_out))/2

    if use_hyper_reg_loss:
      def calc_hyper_reg_loss(model_out):
        weights = lax.stop_gradient(model_out['weights'])
        hyper_points = model_out['warped_points'][..., 3:]
        hyper_reg_residual = (hyper_points ** 2).sum(axis=-1)
        hyper_reg_loss = utils.general_loss_with_squared_residual(
            hyper_reg_residual, alpha=0.0, scale=0.05)
        assert weights.shape == hyper_reg_loss.shape
        hyper_reg_loss = (weights * hyper_reg_loss).sum(axis=1).mean()
        stats['loss/hyper_reg'] = hyper_reg_loss
        stats['residual/hyper_reg'] = jnp.mean(jnp.sqrt(hyper_reg_residual))
        return hyper_reg_loss
      # loss += scalar_params.hyper_reg_loss_weight * hyper_reg_loss
      loss += scalar_params.hyper_reg_loss_weight * (calc_hyper_reg_loss(prev_out) + calc_hyper_reg_loss(next_out))/2

    if 'warp_jacobian' in prev_out:
      jacobian = prev_out['warp_jacobian']
      jacobian_det = jnp.linalg.det(jacobian)
      jacobian_div = utils.jacobian_to_div(jacobian)
      jacobian_curl = utils.jacobian_to_curl(jacobian)
      stats['metric/jacobian_det'] = jnp.mean(jacobian_det)
      stats['metric/jacobian_div'] = jnp.mean(jacobian_div)
      stats['metric/jacobian_curl'] = jnp.mean(
          jnp.linalg.norm(jacobian_curl, axis=-1))

    stats['loss/total'] = loss
    if not "None" in rgb_out:
      stats['metric/psnr'] = utils.compute_psnr(rgb_loss_raw)
      
    return loss, stats
  
  def _loss_fn(params):
    col_batch, evs_batch = batch.get("col_data"), batch.get("evs_data")
    ret_col = ret_prev = ret_next = {"fine":{"None" : None},
                                     "coarse":{"None" : None}}
    forward_fnc = lambda d_batch, fk, ck :model.apply({'params': params['model']},
                                              d_batch,
                                              extra_params=state.extra_params,
                                              return_points=(use_warp_reg_loss or use_hyper_reg_loss),
                                              return_weights=(use_warp_reg_loss or use_elastic_loss),
                                              return_warp_jacobian=use_elastic_loss,
                                              rngs={
                                                  'fine': fk,
                                                  'coarse': ck
                                              })

    if col_batch is not None:
      ret_col = forward_fnc(col_batch, fk_col, ck_col)
    
    if evs_batch is not None:
      ret_prev = forward_fnc(evs_batch["prev_data"], fk_prev, ck_prev)
      ret_next = forward_fnc(evs_batch["next_data"], fk_next, ck_next)

    losses = {}
    stats = {}
    cond_fn = lambda key : (key in ret_col) or (key in ret_prev)
    if cond_fn("fine"):
      losses['fine'], stats['fine'] = _compute_loss_and_stats(
          params, ret_prev['fine'], ret_next['fine'], ret_col['fine'], 'fine')
    if cond_fn('coarse'):
      losses['coarse'], stats['coarse'] = _compute_loss_and_stats(
          params, ret_prev['coarse'],ret_next['coarse'],ret_col['coarse'], 'coarse',
          use_elastic_loss=use_elastic_loss,
          use_hyper_reg_loss=use_hyper_reg_loss)

    if use_background_loss:
      background_loss = compute_background_loss(
          model,
          state=state,
          params=params['model'],
          key=reg_key,
          points=batch['background_points'],
          noise_std=scalar_params.background_noise_std)
      background_loss = background_loss.mean()
      losses['background'] = (
          scalar_params.background_loss_weight * background_loss)
      stats['background_loss'] = background_loss

    ret = ret_next if evs_batch is not None else ret_col
    return sum(losses.values()), (stats, ret)

  optimizer = state.optimizer
  if disable_hyper_grads:
    optimizer = optimizer.replace(
        state=zero_adam_param_states(optimizer.state, 'model/hyper_sheet_mlp'))

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (_, (stats, model_out)), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, axis_name='batch')
  if grad_max_val > 0.0 or grad_max_norm > 0.0:
    grad = utils.clip_gradients(grad, grad_max_val, grad_max_norm)
  stats = jax.lax.pmean(stats, axis_name='batch')
  model_out = jax.lax.pmean(model_out, axis_name='batch')
  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=scalar_params.learning_rate)
  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng_key, model_out
