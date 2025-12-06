import jax
import jax.numpy as jnp

from functools import partial
from typing import Any, Collection, Sequence

import flax.linen as nn
import optax

# Dtype = jnp.float32

class UpSample(nn.Module):
    dim: int
    kernel_size: int
    dtype: jnp.float32 #Dtype

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(2, 2),
            dtype=self.dtype,
        )(x)
        return x


class DownSample(nn.Module):
    dim: int
    kernel_size: int
    dtype: jnp.float32 #Dtype

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(2, 2),
            dtype=self.dtype,
        )(x)
        return x


class SinusoidalPosEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, pos):
        """Refer to https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5"""
        batch_size = pos.shape[0]

        assert self.dim % 2 == 0, self.dim
        assert pos.shape == (batch_size, 1), pos.shape

        d_model = self.dim // 2
        i = jnp.arange(d_model)[None, :]

        pos_embedding = pos * jnp.exp(-(2 * i / d_model) * jnp.log(10000))
        pos_embedding = jnp.concatenate(
            (jnp.sin(pos_embedding), jnp.cos(pos_embedding)), axis=-1
        )

        assert pos_embedding.shape == (batch_size, self.dim), pos_embedding.shape

        return pos_embedding


class TimeEmbedding(nn.Module):
    dim: int
    sinusoidal_embed_dim: int
    dtype: jnp.float32 #Dtype

    @nn.compact
    def __call__(self, time):
        x = SinusoidalPosEmbedding(self.sinusoidal_embed_dim)(time)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int
    dropout: float
    dtype: jnp.float32 #Dtype

    @nn.compact
    def __call__(self, x, deterministic: bool, *, scale_shift=None):
        x = nn.GroupNorm(self.num_groups, dtype=self.dtype)(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.Conv(
            self.dim, kernel_size=(self.kernel_size, self.kernel_size), dtype=self.dtype
        )(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift

        return x


class ResnetBlock(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int
    dropout: float
    dtype: jnp.float32 # Dtype

    @nn.compact
    def __call__(self, x, deterministic: bool, *, time_emb=None):
        """
        Args:
            x: array of shape `[batch_size, width, height, d]`
        """
        h = Block(self.dim, self.kernel_size, self.num_groups, 0.0, self.dtype)(
            x, deterministic
        )

        scale_shift = None
        if time_emb is not None:
            time_emb = nn.silu(time_emb)

            scale = nn.DenseGeneral(self.dim, dtype=self.dtype)(time_emb)
            scale = jnp.expand_dims(scale, axis=(1, 2))

            shift = nn.DenseGeneral(self.dim, dtype=self.dtype)(time_emb)
            shift = jnp.expand_dims(shift, axis=(1, 2))

            scale_shift = (scale, shift)

        h = Block(
            self.dim, self.kernel_size, self.num_groups, self.dropout, self.dtype
        )(h, deterministic, scale_shift=scale_shift)

        if x.shape[-1] != self.dim:
            x = nn.Conv(self.dim, kernel_size=(1, 1))(x)

        x = x + h
        return x


class ResidualAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    num_groups: int
    dtype: jnp.float32 #Dtype

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: array of shape `[batch_size, width, height, dim]`
        """

        res = x
        b, w, h, d = res.shape
        res = nn.GroupNorm(self.num_groups, dtype=self.dtype)(res)
        res = jnp.reshape(res, (b, w * h, d))
        res = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype)(res)
        res = jnp.reshape(res, (b, w, h, self.dim))

        x = x + res
        return x

class Encoder(nn.Module):
    dim_init: int
    kernel_size: int
    dim_mults: Sequence[int]
    num_groups: int
    latent_dim: int
    dropout: float
    dtype: jnp.float32

    @nn.compact
    def __call__(self, x, train=False):
        # Initial convolution
        x = nn.Conv(self.dim_init, (self.kernel_size, self.kernel_size), dtype=self.dtype)(x)

        # Downsample blocks
        # for dim_mult in self.dim_mults:
        for i, dim_mult in enumerate(self.dim_mults):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            # ResNet block
            x = ResnetBlock(
                dim=dim,
                kernel_size=self.kernel_size,
                num_groups=self.num_groups,
                dropout=self.dropout,
                dtype=self.dtype
            )(x, not train)

            if not is_last:
                x = DownSample(dim, self.kernel_size, self.dtype)(x)

        # Final layers to produce means and log variances
        h = nn.GroupNorm(self.num_groups, dtype=self.dtype)(x)
        h = nn.silu(h)

        # Flatten and project to 8-dimensional latent space
        h = jnp.mean(h, axis=(1, 2))  # Global average pooling

        # Split into means and log variances
        means = nn.Dense(self.latent_dim, dtype=self.dtype)(h)
        log_vars = nn.Dense(self.latent_dim, dtype=self.dtype)(h)

        return means, log_vars


class Decoder(nn.Module):
    dim_init: int
    kernel_size: int
    dim_mults: Sequence[int]
    num_groups: int
    dropout: float
    dtype: jnp.float32

    @nn.compact
    def __call__(self, z, train=False):

        # Reshape the input z to have spatial dimensions
        x = z[:, None, None, :]  # Shape: [batch_size, 1, 1, 8]

        bottleneck_dim = 4  # The spatial size after all encoder downsampling
        channels = self.dim_init * self.dim_mults[-1] # Channels at the bottleneck

        x = nn.Dense(bottleneck_dim * bottleneck_dim * channels, dtype=self.dtype)(z)
        x = nn.relu(x) # Add a non-linearity
        x = jnp.reshape(x, (-1, bottleneck_dim, bottleneck_dim, channels))

        # Upsample blocks
        for i, dim_mult in enumerate(reversed(self.dim_mults)):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            # ResNet block
            x = ResnetBlock(
                dim=dim,
                kernel_size=self.kernel_size,
                num_groups=self.num_groups,
                dropout=self.dropout,
                dtype=self.dtype
            )(x, not train)

            if not is_last:
                x = UpSample(dim, self.kernel_size, self.dtype)(x)

        # Final layers
        x = nn.GroupNorm(self.num_groups, dtype=self.dtype)(x)
        x = nn.silu(x)
        x = nn.Conv(1, kernel_size=(1, 1), dtype=self.dtype)(x)

        x = nn.sigmoid(x)

        return x


class VAE(nn.Module):
    dim_init: int
    kernel_size: int
    dim_mults: Sequence[int]
    num_groups: int
    latent_dim: int
    dropout: float

    def setup(self):
        self.encoder = Encoder(
            dim_init=self.dim_init,
            kernel_size=self.kernel_size,
            dim_mults=self.dim_mults,
            num_groups=self.num_groups,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            dtype=jnp.float32

        )

        self.decoder = Decoder(
            dim_init=self.dim_init,
            kernel_size=self.kernel_size,
            dim_mults=self.dim_mults,
            num_groups=self.num_groups,
            dropout=self.dropout,
            dtype=jnp.float32
        )

    def __call__(self, x, train=False, rng=None):
        means, log_vars = self.encoder(x, train)

        if train and rng is not None:
            # Reparameterization trick
            std = jnp.exp(0.5 * log_vars)
            eps = jax.random.normal(rng, means.shape)
            z = means + eps * std
        else:
            z = means

        reconstruction = self.decoder(z, train)

        return reconstruction, means, log_vars

    def encode(self, x):
        return self.encoder(x, train=False)

    def decode(self, z):
        return self.decoder(z, train=False)
    

    # training

    def vae_loss(self, params, rng, xs):
        reconstruction, means, log_vars = self.apply(params, xs, train=True, rngs={'dropout': rng}) # (B, 32, 32, 1), (B, 8), (B, 8)
        
        # Reconstruction loss
        recon_losses = jnp.sum((xs - reconstruction) ** 2, axis=1) # (B,)
        recon_loss = jnp.mean(recon_losses)
        
        # KL divergence
        kl_loss = -0.5 * jnp.mean(1 + log_vars - means ** 2 - jnp.exp(log_vars))

        loss = recon_loss + kl_loss
        
        return loss, (recon_loss, kl_loss)

    def update(self, params, rng, opt_state, xs):
        (loss, (recon_loss, kl_loss)), grads = jax.value_and_grad(self.vae_loss, has_aux=True)(params, rng, xs)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, recon_loss, kl_loss

    @partial(jax.jit, static_argnums=(0, 4))
    def train_step(self, rng, params, opt_state, batch_size, train_images):
        
        batch_rng, update_rng, rng = jax.random.split(rng, 3)
        idx = jax.random.randint(batch_rng, (batch_size,), 0, train_images.shape[0])
        xs = train_images[idx]

        params, opt_state, loss,  recon_loss, kl_loss = self.update(params, update_rng, opt_state, xs)

        return params, opt_state, loss, recon_loss, kl_loss
    

    def train(self, rng, params, train_images, train_config):
        learning_rate = train_config['learning_rate']
        batch_size = train_config['batch_size']
        num_steps = train_config['num_steps']
        num_checkpoints = train_config['num_checkpoints']

        self.optimizer = optax.adam(learning_rate)
        opt_state = self.optimizer.init(params)

        checkpoint_freq = num_steps // num_checkpoints

        losses = []
        params_lst = []

        for step in range(num_steps):
            rng, rng_ = jax.random.split(rng)
            params, opt_state, loss, recon_loss, kl_loss = self.train_step(rng_, params, opt_state, batch_size, train_images)
            losses.append(loss)
            if step % 10 == 0:
                print(f'Step {step}: loss {loss}, reconstruction loss {recon_loss}, kl loss {kl_loss}')

            if step % checkpoint_freq == 0:
                params_lst.append(params)

        return params, losses, params_lst


def create_vae(rng, model_config):
    vae = VAE(
        dim_init=model_config['dim_init'],        # Initial dimension (scaled by dim_mults)
        kernel_size=model_config['kernel_size'],        # Kernel size for convolutions
        dim_mults=model_config['dim_mults'],  # Dimension multipliers for each level
        num_groups=model_config['num_groups'],         # Number of groups for GroupNorm
        latent_dim=model_config['latent_dim'],         # Dimension of the latent space
        dropout=model_config['dropout']           # Dropout rate
    )
    
    # Initialize with dummy input
    dummy_input = jnp.zeros((1, 32, 32, 1))
    variables = vae.init(rng, dummy_input)
    
    return vae, variables
