import jax
import jax.numpy as jnp
import flax.linen as nn
import tensorflow_probability.substrates.jax as tfd
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
        ops_emb = ops_machine_emb + ops_duration_emb  # Shape: (num_jobs, max_num_ops, embedding_dim)
        ops_emb = ops_emb * j_obs.ops_mask[..., None]  # Mask invalid operations
        job_emb = jnp.sum(ops_emb, axis=1) / jnp.sum(j_obs.ops_mask, axis=1, keepdims=True)  # Average pooling per job

        # Machine Encoder: Process machines_job_ids and machines_remaining_times
        machines_job_emb = nn.Embed(num_embeddings=self.num_jobs + 1, features=self.embedding_dim)(j_obs.machines_job_ids)
        machines_time_emb = nn.Dense(self.embedding_dim)(j_obs.machines_remaining_times[..., None])
        machine_emb = machines_job_emb + machines_time_emb  # Shape: (num_machines, embedding_dim)
        machine_emb = jnp.mean(machine_emb, axis=0)  # Global average pooling across machines

        # Combine job and machine embeddings
        combined_emb = jnp.concatenate([job_emb.flatten(), machine_emb[None, :]], axis=-1)
        agents_view = nn.Dense(128)(combined_emb)  # Project to a fixed-size feature vector
        return agents_view[None, :]  # Shape: (1, 128) for single agent

class JobShopActor(nn.Module):
    num_actions: int  # Flattened action space: num_machines * (num_jobs + 1)

    @nn.compact
    def __call__(self, obs: Observation) -> tfd.Distribution:
        agents_view = obs.agents_view  # Shape: (1, num_obs_features)
        action_mask = obs.action_mask  # Shape: (1, num_actions)
        logits = nn.Dense(self.num_actions)(agents_view)
        masked_logits = jnp.where(action_mask, logits, -1e9)  # Mask invalid actions
        return tfd.Categorical(logits=masked_logits)

class JobShopCritic(nn.Module):
    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        agents_view = obs.agents_view  # Shape: (1, num_obs_features)
        value = nn.Dense(1)(agents_view)
        return jnp.squeeze(value, -1)  # Scalar value

def create_networks(num_jobs: int, max_num_ops: int, num_machines: int):
    """Creates the encoder, actor, and critic networks for JobShop in Mava."""
    encoder = JobShopEncoder(num_jobs=num_jobs, max_num_ops=max_num_ops, num_machines=num_machines)
    num_actions = num_machines * (num_jobs + 1)  # Flattened action space
    actor = JobShopActor(num_actions=num_actions)
    critic = JobShopCritic()
    return encoder, actor, critic

# Example usage within Mava
def process_observation(j_obs: JObs, encoder: JobShopEncoder) -> Observation:
    """Wraps JobShop observation into Mava's Observation format."""
    agents_view = encoder(j_obs)  # Shape: (1, 128)
    action_mask = jnp.reshape(j_obs.action_mask, (1, -1))  # Flatten to (1, num_machines * (num_jobs + 1))
    return Observation(agents_view=agents_view, action_mask=action_mask)