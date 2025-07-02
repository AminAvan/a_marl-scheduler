# mava/networks/job_shop_network.py
import chex
import jax.numpy as jnp
from flax import linen as nn
import tensorflow_probability.substrates.jax.distributions as tfd

from mava.types import Observation
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from mava.networks.heads import CategoricalHead

class JobShopEncoder(nn.Module):
    """Flax encoder for Jumanji JobShop observation."""
    @nn.compact
    def __call__(self, j_obs: JumanjiObservation) -> chex.Array:
        # Flatten and embed machine IDs and durations
        m_flat = j_obs.ops_machine_ids.flatten()
        d_flat = j_obs.ops_durations.flatten()
        m_emb  = nn.Dense(64)(m_flat)
        d_emb  = nn.Dense(64)(d_flat)
        x = jnp.concatenate([m_emb, d_emb], axis=-1)
        # Two-layer MLP
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(64)(x))
        return x

class JobShopActor(nn.Module):
    """Actor for JobShop using custom encoder + CategoricalHead."""
    encoder: nn.Module = JobShopEncoder()
    dist_head: nn.Module = CategoricalHead()

    @nn.compact
    def __call__(self, observation: Observation) -> tfd.Distribution:
        # Expect the wrapper to supply .jumanji_obs and .action_mask
        j_obs = observation.jumanji_obs
        emb   = self.encoder(j_obs)
        return self.dist_head(emb, observation.action_mask)

class JobShopCritic(nn.Module):
    """Critic for JobShop using the same encoder + a value head."""
    encoder: nn.Module = JobShopEncoder()

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        j_obs = observation.jumanji_obs
        emb   = self.encoder(j_obs)
        v     = nn.Dense(1)(emb)
        return jnp.squeeze(v, axis=-1)
