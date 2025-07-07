import jax
import jax.numpy as jnp
import flax.linen as nn
import tensorflow_probability.substrates.jax.distributions as tfd
from mava.types import Observation
from jumanji.environments.packing.job_shop.types import Observation as JObs

class JobShopEncoder(nn.Module):
    num_jobs: int
    max_num_ops: int
    num_machines: int
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, j_obs: JObs) -> jnp.ndarray:
        # Operation Encoder: Process ops_machine_ids, ops_durations, and ops_mask
        ops_machine_emb = nn.Embed(num_embeddings=self.num_machines, features=self.embedding_dim)(j_obs.ops_machine_ids)
        ops_duration_emb = nn.Dense(self.embedding_dim)(j_obs.ops_durations[..., None])
        ops_emb = ops_machine_emb + ops_duration_emb  # (num_jobs, max_num_ops, embedding_dim)
        ops_emb = ops_emb * j_obs.ops_mask[..., None]  # mask invalid ops
        job_emb = jnp.sum(ops_emb, axis=1) / jnp.sum(j_obs.ops_mask, axis=1, keepdims=True)  # average pooling

        # Machine Encoder: Process machines_job_ids and machines_remaining_times
        machines_job_emb = nn.Embed(num_embeddings=self.num_jobs + 1, features=self.embedding_dim)(j_obs.machines_job_ids)
        machines_time_emb = nn.Dense(self.embedding_dim)(j_obs.machines_remaining_times[..., None])
        machine_emb = machines_job_emb + machines_time_emb  # (num_machines, embedding_dim)
        machine_emb = jnp.mean(machine_emb, axis=0)  # global average pooling across machines

        # Combine job and machine embeddings
        combined_emb = jnp.concatenate([job_emb.flatten(), machine_emb[None, :]], axis=-1)
        agents_view = nn.Dense(128)(combined_emb)  # project to fixed-size vector
        return agents_view[None, :]  # shape: (1, 128)

class JobShopActor(nn.Module):
    num_actions: int  # num_machines * (num_jobs + 1)

    @nn.compact
    def __call__(self, obs: Observation) -> tfd.Distribution:
        # agents_view may be (batch, feat) or (batch, agents, feat)
        emb = obs.agents_view
        mask = obs.action_mask  # may be (batch, actions) or (batch, agents, actions)
        # Align embedding dims to mask dims
        if emb.ndim + 1 == mask.ndim:
            emb = jnp.expand_dims(emb, axis=-2)
            emb = jnp.repeat(emb, repeats=mask.shape[-2], axis=-2)
        # Compute logits and mask invalid actions
        logits = nn.Dense(self.num_actions)(emb)
        masked_logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
        return tfd.Categorical(logits=masked_logits)

class JobShopCritic(nn.Module):
    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        # value per agent or per batch
        emb = obs.agents_view
        value = nn.Dense(1)(emb)
        return jnp.squeeze(value, axis=-1)

# Helper to create networks outside Hydra if needed
def create_networks(num_jobs: int, max_num_ops: int, num_machines: int):
    encoder = JobShopEncoder(num_jobs=num_jobs, max_num_ops=max_num_ops, num_machines=num_machines)
    num_actions = num_machines * (num_jobs + 1)
    actor = JobShopActor(num_actions=num_actions)
    critic = JobShopCritic()
    return encoder, actor, critic

# Wrap raw Jumanji obs into Mava Observation
def process_observation(j_obs: JObs, encoder: JobShopEncoder) -> Observation:
    agents_view = encoder(j_obs)  # (1, 128)
    action_mask = jnp.reshape(j_obs.action_mask, (1, -1))
    return Observation(agents_view=agents_view, action_mask=action_mask)
