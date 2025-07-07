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
        ops_machine_emb = nn.Embed(
            num_embeddings=self.num_machines,
            features=self.embedding_dim
        )(j_obs.ops_machine_ids)
        ops_duration_emb = nn.Dense(self.embedding_dim)(
            j_obs.ops_durations[..., None]
        )
        ops_emb = ops_machine_emb + ops_duration_emb  # (num_jobs, max_num_ops, embedding_dim)
        ops_emb = ops_emb * j_obs.ops_mask[..., None]  # mask invalid ops
        job_emb = (
            jnp.sum(ops_emb, axis=1)
            / jnp.sum(j_obs.ops_mask, axis=1, keepdims=True)
        )  # average pooling per job

        # Machine Encoder: Process machines_job_ids and machines_remaining_times
        machines_job_emb = nn.Embed(
            num_embeddings=self.num_jobs + 1,
            features=self.embedding_dim
        )(j_obs.machines_job_ids)
        machines_time_emb = nn.Dense(self.embedding_dim)(
            j_obs.machines_remaining_times[..., None]
        )
        machine_emb = machines_job_emb + machines_time_emb  # (num_machines, embedding_dim)
        machine_emb = jnp.mean(machine_emb, axis=0)  # global avg pooling over machines

        # Combine job and machine embeddings
        combined_emb = jnp.concatenate(
            [job_emb.flatten(), machine_emb[None, :]], axis=-1
        )
        agents_view = nn.Dense(128)(combined_emb)  # project to fixed-size vector
        return agents_view[None, :]  # shape: (1, 128)

class JobShopActor(nn.Module):
    num_actions: int  # num_machines * (num_jobs + 1)

    @nn.compact
    def __call__(self, obs: Observation) -> tfd.Distribution:
        emb  = obs.agents_view                        # (B, M, feat)  or (B, feat)
        mask = obs.action_mask                        # (B, M, A)

        # --- align embedding dims to mask dims (you already have this) ---
        if emb.ndim + 1 == mask.ndim:                 # emb=(B,feat), mask=(B,M,A)
            emb = jnp.expand_dims(emb, -2)            # → (B,1,feat)
            emb = jnp.repeat(emb, mask.shape[-2], -2) # → (B,M,feat)

        # --- logits ------------------------------------------------------
        logits = nn.Dense(self.num_actions)(emb)      # (B,M,A)  *or* (B,A)

        # >>> new part: align logits to mask if needed  <<<
        if logits.ndim + 1 == mask.ndim:              # logits=(B,A), mask=(B,M,A)
            logits = jnp.expand_dims(logits, -2)      # → (B,1,A)
            logits = jnp.repeat(logits, mask.shape[-2], -2)  # → (B,M,A)

        # --- mask invalid actions ---------------------------------------
        masked_logits = jnp.where(
            mask,
            logits,
            jnp.finfo(jnp.float32).min,
        )
        return tfd.Categorical(logits=masked_logits)

class JobShopCritic(nn.Module):
    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        # Produce a scalar (batch, agents) or (batch,) value
        emb = obs.agents_view
        value = nn.Dense(1)(emb)
        return jnp.squeeze(value, axis=-1)

# Helpers if used outside Hydra

def create_networks(num_jobs: int, max_num_ops: int, num_machines: int):
    encoder = JobShopEncoder(
        num_jobs=num_jobs,
        max_num_ops=max_num_ops,
        num_machines=num_machines
    )
    num_actions = num_machines * (num_jobs + 1)
    actor = JobShopActor(num_actions=num_actions)
    critic = JobShopCritic()
    return encoder, actor, critic


def process_observation(j_obs: JObs, encoder: JobShopEncoder) -> Observation:
    agents_view = encoder(j_obs)  # (1, 128)
    action_mask = jnp.reshape(j_obs.action_mask, (1, -1))
    return Observation(agents_view=agents_view, action_mask=action_mask)
