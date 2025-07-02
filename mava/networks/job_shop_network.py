# mava/networks/job_shop_network.py
import chex, jax.numpy as jnp, tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from mava.types import Observation
from mava.networks.heads import DiscreteActionHead
from jumanji.environments.packing.job_shop.types import Observation as JObs

class JobShopEncoder(nn.Module):
    @nn.compact
    def __call__(self, j_obs: JObs) -> chex.Array:
        m_flat = j_obs.ops_machine_ids.flatten()
        d_flat = j_obs.ops_durations.flatten()
        m_emb  = nn.Dense(64)(m_flat)
        d_emb  = nn.Dense(64)(d_flat)
        x = jnp.concatenate([m_emb, d_emb], -1)
        x = nn.relu(nn.Dense(128)(x))
        return nn.relu(nn.Dense(64)(x))

class JobShopActor(nn.Module):
    @nn.compact
    def __call__(self, obs: Observation) -> tfd.Distribution:
        emb = JobShopEncoder()(obs.jumanji_obs)
        head = DiscreteActionHead(action_dim=int(obs.action_mask.shape[-1]))
        return head(emb, obs.action_mask)

class JobShopCritic(nn.Module):
    @nn.compact
    def __call__(self, obs: Observation) -> chex.Array:
        emb = JobShopEncoder()(obs.jumanji_obs)
        v   = nn.Dense(1)(emb)
        return jnp.squeeze(v, -1)
