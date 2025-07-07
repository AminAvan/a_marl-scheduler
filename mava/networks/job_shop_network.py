"""
JobShop networks for Mava that work with the existing Mava architecture.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Tuple
from mava.types import Observation
from jumanji.environments.packing.job_shop.types import Observation as JObs


class JobShopEncoder(nn.Module):
    """Encoder for JobShop observations."""
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

        # Average pooling per job
        job_emb = (
                jnp.sum(ops_emb, axis=1)
                / jnp.maximum(jnp.sum(j_obs.ops_mask, axis=1, keepdims=True), 1)
        )  # Avoid division by zero

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
            [job_emb.flatten(), machine_emb], axis=-1
        )
        agents_view = nn.Dense(128)(combined_emb)  # project to fixed-size vector
        return agents_view[None, :]  # shape: (1, 128)


class JobShopActor(nn.Module):
    """Actor network for JobShop that outputs action logits."""
    num_actions: int  # num_machines * (num_jobs + 1)

    @nn.compact
    def __call__(self, observation: chex.ArrayTree) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process observation and return action logits and value.

        Args:
            observation: Mava Observation with agents_view and action_mask

        Returns:
            Tuple of (action_logits, value) where:
            - action_logits: Array of shape (batch, num_agents, num_actions)
            - value: Array of shape (batch, num_agents)
        """
        # Handle both Observation type and raw arrays
        if hasattr(observation, 'agents_view'):
            agents_view = observation.agents_view
            action_mask = observation.action_mask
        else:
            # Assume it's already the agents_view
            agents_view = observation
            action_mask = None

        # Process agents view
        x = agents_view  # Expected shape: (batch, num_agents, features)

        # If single agent view, expand dimensions
        if x.ndim == 2:  # (batch, features)
            x = jnp.expand_dims(x, 1)  # (batch, 1, features)

        # Apply layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Output heads
        logits = nn.Dense(self.num_actions)(x)  # (batch, num_agents, num_actions)
        value = nn.Dense(1)(x)  # (batch, num_agents, 1)
        value = jnp.squeeze(value, axis=-1)  # (batch, num_agents)

        # Apply action mask if provided
        if action_mask is not None:
            # Ensure mask has same shape as logits
            if action_mask.shape != logits.shape:
                if action_mask.ndim == 2 and logits.ndim == 3:
                    # Expand mask from (batch, actions) to (batch, 1, actions)
                    action_mask = jnp.expand_dims(action_mask, 1)
                    # Repeat for all agents
                    action_mask = jnp.repeat(action_mask, logits.shape[1], axis=1)

            # Apply mask
            logits = jnp.where(
                action_mask,
                logits,
                jnp.finfo(jnp.float32).min
            )

        return logits, value


class JobShopCritic(nn.Module):
    """Critic network for JobShop."""

    @nn.compact
    def __call__(self, observation: chex.ArrayTree) -> jnp.ndarray:
        """
        Process observation and return value estimates.

        Args:
            observation: Mava Observation with agents_view

        Returns:
            Value estimates of shape (batch, num_agents)
        """
        # Handle both Observation type and raw arrays
        if hasattr(observation, 'agents_view'):
            x = observation.agents_view
        else:
            x = observation

        # If single agent view, expand dimensions
        if x.ndim == 2:  # (batch, features)
            x = jnp.expand_dims(x, 1)  # (batch, 1, features)

        # Apply layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Output value
        value = nn.Dense(1)(x)  # (batch, num_agents, 1)
        value = jnp.squeeze(value, axis=-1)  # (batch, num_agents)

        return value


# For compatibility with Mava's FeedForwardActor interface
class JobShopFFActor(nn.Module):
    """FeedForward Actor wrapper for JobShop."""
    action_dim: int

    @nn.compact
    def __call__(self, observation: chex.ArrayTree) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning pi logits and value."""
        actor = JobShopActor(num_actions=self.action_dim)
        return actor(observation)