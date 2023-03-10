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

"""Modules for NeRF models."""
import functools
from typing import Any, Optional, Tuple

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import types
from transformer import TransformerEncoder, PositionalEncoding

def get_norm_layer(norm_type):
  """Translates a norm type to a norm constructor."""
  if norm_type is None or norm_type == 'none':
    return None
  elif norm_type == 'layer':
    return functools.partial(nn.LayerNorm, use_scale=False, use_bias=False)
  elif norm_type == 'group':
    return functools.partial(nn.GroupNorm, use_scale=False, use_bias=False)
  elif norm_type == 'batch':
    return functools.partial(nn.BatchNorm, use_scale=False, use_bias=False)
  else:
    raise ValueError(f'Unknown norm type {norm_type}')


class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  hidden_activation: types.Activation = nn.relu
  hidden_norm: Optional[types.Normalizer] = None
  output_init: Optional[types.Initializer] = None
  output_channels: int = 0
  output_activation: Optional[types.Activation] = lambda x: x
  use_bias: bool = True
  skips: Tuple[int] = tuple()

  @nn.compact
  def __call__(self, x):
    inputs = x
    for i in range(self.depth):
      layer = nn.Dense(
          self.width,
          use_bias=self.use_bias,
          kernel_init=self.hidden_init,
          name=f'hidden_{i}')
      if i in self.skips:
        x = jnp.concatenate([x, inputs], axis=-1)
      x = layer(x)
      if self.hidden_norm is not None:
        x = self.hidden_norm()(x)  # pylint: disable=not-callable
      x = self.hidden_activation(x)

    if self.output_channels > 0:
      logit_layer = nn.Dense(
          self.output_channels,
          use_bias=self.use_bias,
          kernel_init=self.output_init,
          name='logit')
      x = logit_layer(x)
      if self.output_activation is not None:
        x = self.output_activation(x)

    return x


class NerfMLP(nn.Module):
  """A simple MLP.

  Attributes:
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    activation: function, the activation function used in the MLP.
    skips: which layers to add skip layers to.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    condition_density: if True put the condition at the begining which
      conditions the density of the field.
  """
  trunk_depth: int = 8
  trunk_width: int = 256

  rgb_branch_depth: int = 1
  rgb_branch_width: int = 128
  rgb_channels: int = 3

  alpha_branch_depth: int = 0
  alpha_branch_width: int = 128
  alpha_channels: int = 1

  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  skips: Tuple[int] = (4,)

  @nn.compact
  def __call__(self, x, alpha_condition, rgb_condition):
    """Multi-layer perception for nerf.

    Args:
      x: sample points with shape [batch, num_coarse_samples, feature].
      alpha_condition: a condition array provided to the alpha branch.
      rgb_condition: a condition array provided in the RGB branch.

    Returns:
      raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
    """
    dense = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])

    def broadcast_condition(c):
      # Broadcast condition from [batch, feature] to
      # [batch, num_coarse_samples, feature] since all the samples along the
      # same ray has the same viewdir.
      c = jnp.tile(c[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_coarse_samples, feature] tensor to
      # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
      c = c.reshape([-1, c.shape[-1]])
      return c

    trunk_mlp = MLP(depth=self.trunk_depth,
                    width=self.trunk_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    skips=self.skips)
    rgb_mlp = MLP(depth=self.rgb_branch_depth,
                  width=self.rgb_branch_width,
                  hidden_activation=self.activation,
                  hidden_norm=self.norm,
                  hidden_init=jax.nn.initializers.glorot_uniform(),
                  output_init=jax.nn.initializers.glorot_uniform(),
                  output_channels=self.rgb_channels)
    alpha_mlp = MLP(depth=self.alpha_branch_depth,
                    width=self.alpha_branch_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    output_init=jax.nn.initializers.glorot_uniform(),
                    output_channels=self.alpha_channels)

    if self.trunk_depth > 0:
      x = trunk_mlp(x)

    if (alpha_condition is not None) or (rgb_condition is not None):
      bottleneck = dense(self.trunk_width, name='bottleneck')(x)

    if alpha_condition is not None:
      alpha_condition = broadcast_condition(alpha_condition)
      alpha_input = jnp.concatenate([bottleneck, alpha_condition], axis=-1)
    else:
      alpha_input = x
    alpha = alpha_mlp(alpha_input)

    if rgb_condition is not None:
      rgb_condition = broadcast_condition(rgb_condition)
      rgb_input = jnp.concatenate([bottleneck, rgb_condition], axis=-1)
    else:
      rgb_input = x
    rgb = rgb_mlp(rgb_input)

    return {
        'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
        'alpha': alpha.reshape((-1, num_samples, self.alpha_channels)),
    }


@gin.configurable(denylist=['name'])
class GLOEmbed(nn.Module):
  """A GLO encoder module, which is just a thin wrapper around nn.Embed.

  Attributes:
    num_embeddings: The number of embeddings.
    features: The dimensions of each embedding.
    embedding_init: The initializer to use for each.
  """

  num_embeddings: int = gin.REQUIRED
  num_dims: int = gin.REQUIRED
  embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

  def setup(self):
    self.embed = nn.Embed(
        num_embeddings=self.num_embeddings,
        features=self.num_dims,
        embedding_init=self.embedding_init)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Method to get embeddings for specified indices.

    Args:
      inputs: The indices to fetch embeddings for.

    Returns:
      The embeddings corresponding to the indices provided.
    """
    if inputs.shape[-1] == 1:
      inputs = jnp.squeeze(inputs, axis=-1)

    return self.embed(inputs)


@gin.configurable(denylist=['name'])
class PadEmbed(GLOEmbed):
  """
  GLO embedding with padding so that target frame embed is in the center
  """
  n_emb_per_frame: int = 7  # number of embeding use per frame
  num_embeddings: int = gin.REQUIRED
  num_dims: int = 16
  embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

  def setup(self):
    assert self.n_emb_per_frame%2 == 1, "frame latent will not be at center"
    n_embd = (self.n_emb_per_frame//2)*2 + self.num_embeddings + 3  
    self.embed = nn.Embed(
        num_embeddings=n_embd,
        features=self.num_dims,
        embedding_init=self.embedding_init)
    
    self.index_shift = self.n_emb_per_frame//2 + 2 # add extra 1 so index = 0 is mask token
    self.n_embd = n_embd

  def __call__(self, inputs: jnp.ndarray, do_query:bool =False) -> jnp.ndarray:
    
    if do_query:
      return self.query_mask(inputs)
    else:
      return self.get_embd(inputs)

  def get_embd(self, inputs):
    """Method to get embeddings for specified indices.

    Args:
      inputs: The indices to fetch embeddings for.

    Returns:
      The embeddings corresponding to the indices provided.
    """
    if inputs.shape[-1] != 1:
      inputs = inputs[...,None]

    inputs = inputs + self.index_shift
    add_dims = [1 for _ in range(len(inputs.shape[:-1]))]
    shift_seq = jnp.arange(- self.n_emb_per_frame//2, 
                                 self.n_emb_per_frame//2).reshape(*add_dims, -1)
    inputs = inputs + shift_seq

    # assert not (0 in inputs).all(), "mask should not be in call, fix the indexing!!"
    if inputs.shape[-1] == 1:
      inputs = jnp.squeeze(inputs, axis=-1)
    
    embds = self.embed(inputs) 
    self.sow("intermediates", "latents", embds)

    return embds


  def query_mask(self, inputs:jnp.ndarray) -> jnp.ndarray:
    """Method to get embeddings for specified indices.

    Args:
      inputs: The indices to be masked

    Returns:
      index is masked in random intervals
    """
    if inputs.shape[-1] != 1:
      inputs = inputs[...,None]

    inputs = inputs + self.index_shift
    add_dims = [1 for _ in range(len(inputs.shape[:-1]))]
    shift_seq = jnp.arange(-self.n_emb_per_frame//2, 
                                 self.n_emb_per_frame//2).reshape(*add_dims, -1)
    input_seq = inputs + shift_seq

    msk_inputs = input_seq + jax.random.randint(self.make_rng('coarse'), inputs.shape, minval=-self.n_emb_per_frame//2, 
                                                                                       maxval= self.n_emb_per_frame//2)
    
    # make sure minimum is at least 1
    msk_min = msk_inputs.min(axis= 1, keepdims=True)
    inval_cond = (msk_min <= 0) #.squeeze()
    # msk_inputs = msk_inputs.at[inval_cond].set(msk_inputs[inval_cond] - msk_min[inval_cond] + 1)
    msk_inputs = jnp.where(inval_cond, msk_inputs - msk_min + 1, msk_inputs)

    # msk_inputs[inval_cond] = msk_inputs[inval_cond] - msk_min[inval_cond] + 1

    msk_cond = msk_inputs == inputs
    # msk_inputs = msk_inputs.at[msk_cond].set(0) # mask embd index
    msk_inputs = jnp.where(msk_cond, 0, msk_inputs)

    if msk_inputs.shape[-1] == 1:
      msk_inputs = jnp.squeeze(msk_inputs, axis=-1)
    
    embds = self.embed(msk_inputs) 

    return embds, msk_cond


@gin.configurable(denylist=['name'])
class HyperSheetMLP(nn.Module):
  """An MLP that defines a bendy slicing surface through hyper space."""
  output_channels: int = gin.REQUIRED
  min_deg: int = 0
  max_deg: int = 1

  depth: int = 6
  width: int = 64
  skips: Tuple[int] = (4,)
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  output_init: types.Initializer = jax.nn.initializers.normal(1e-5)
  # output_init: types.Initializer = jax.nn.initializers.glorot_uniform()

  use_residual: bool = False

  @nn.compact
  def __call__(self, points, embed, alpha=None):
    points_feat = model_utils.posenc(
        points, self.min_deg, self.max_deg, alpha=alpha)
    inputs = jnp.concatenate([points_feat, embed], axis=-1)
    mlp = MLP(depth=self.depth,
              width=self.width,
              skips=self.skips,
              hidden_init=self.hidden_init,
              output_channels=self.output_channels,
              output_init=self.output_init)
    if self.use_residual:
      return mlp(inputs) + embed
    else:
      return mlp(inputs)

@gin.configurable(denylist=['name'])
class SelectFuser(nn.Module):
  """
  aggregate
  """
  def setup(self):
    pass

  def __call__(self, latents, metadata=None, do_query=False):
    latent_idx = latents.shape[-2]//2
    return latents[:, latent_idx]

@gin.configurable(denylist=['name'])  
class MeanFuser(nn.Module):
  def setup(self):
    pass

  def __call__(self, latents, metadata=None, do_query=False):
    return jnp.mean(latents, axis=-2)

@gin.configurable(denylist=['name'])
class TransformerFuser(nn.Module):
    model_dim : int = 32                  # Hidden dimensionality to use inside the Transformer
    num_heads : int = 4                 # Number of heads to use in the Multi-Head Attention blocks
    num_layers : int = 3                # Number of encoder blocks to use

    # NOTE: turn dropout prob to 0 during evaluation, passing in "train" is not easy
    dropout_prob : float = 0.0        # Dropout to apply inside the model
    input_dropout_prob : float = 0.0  # Dropout to apply on the input features;

    def setup(self):
        # Input dim -> Model dim
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        # self.input_layer = nn.Dense(self.model_dim)
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(self.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.num_layers,
                                              input_dim=self.model_dim,
                                              dim_feedforward=2*self.model_dim,
                                              num_heads=self.num_heads,
                                              dropout_prob=self.dropout_prob)


    def __call__(self, x, select_msk = None, do_query=False, mask=None, add_positional_encoding=True, train=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
            train - If True, dropout is stochastic
        """
        x = self.input_dropout(x, deterministic=not train)
        # x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask, train=train)

        if (select_msk is None) or (not do_query):
          # return the center one
          latent_idx = x.shape[-2]//2
          return x[:, latent_idx]
        else:
          # return the ones enlisted by metadata
          # return x[select_msk]
          d1_idx = jnp.arange(len(x))
          d2_idx = select_msk.argmax(axis=1)
          return x[d1_idx, d2_idx]

    def get_attention_maps(self, x, mask=None, add_positional_encoding=True, train=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask, train=train)
        return attention_maps