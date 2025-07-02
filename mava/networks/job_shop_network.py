import chex
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from mava.types import Observation
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from mava.networks.heads import DiscreteActionHead  # ← correct import


class JobShopEncoder(nn.Module):
    """Flax encoder for Jumanji JobShop observation."""
    @nn.compact
    def __call__(self, j_obs: JumanjiObservation) -> chex.Array:
        m_flat = j_obs.ops_machine_ids.flatten()
        d_flat = j_obs.ops_durations.flatten()
        m_emb  = nn.Dense(64)(m_flat)
        d_emb  = nn.Dense(64)(d_flat)
        x = jnp.concatenate([m_emb, d_emb], axis=-1)
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(64)(x))
        return x


class JobShopActor(nn.Module):
    """Actor for JobShop: encoder + masked Categorical head."""
    @nn.compact
    def __call__(self, observation: Observation) -> tfd.Distribution:
        # Grab the raw Jumanji observation & mask
        j_obs       = observation.jumanji_obs
        action_mask = observation.action_mask

        # 1) Encode
        emb = JobShopEncoder()(j_obs)

        # 2) Build a DiscreteActionHead on the fly with the correct action_dim
        action_dim = int(action_mask.shape[-1])
        dist_head  = DiscreteActionHead(action_dim=action_dim)

        # 3) Return a masked categorical distribution
        return dist_head(emb, action_mask)


class JobShopCritic(nn.Module):
    """Critic for JobShop: same encoder + single‐head value net."""
    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        j_obs = observation.jumanji_obs
        emb   = JobShopEncoder()(j_obs)
        value = nn.Dense(1)(emb)
        return jnp.squeeze(value, axis=-1)