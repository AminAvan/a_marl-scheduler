"""
JobShop networks for Mava - includes both torsos for Mava integration
and direct networks for potential future use.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import orthogonal
from typing import NamedTuple, Tuple

from jumanji.environments.packing.job_shop.types import Observation as JobShopObs


# ============================================================================
# Torsos for Mava Integration (work with agents_view)
# ============================================================================

class JobShopTorso(nn.Module):
    """
    A torso that processes flattened JobShop observations.
    This works with Mava's standard architecture.
    """
    num_jobs: int
    num_machines: int
    max_num_ops: int
    hidden_dim: int = 256
    num_layers: int = 3

    @nn.compact
    def __call__(self, agents_view: jnp.ndarray) -> jnp.ndarray:
        """
        Process the flattened JobShop features.

        Args:
            agents_view: Shape (batch, num_agents, feature_dim) or (num_agents, feature_dim)

        Returns:
            Processed features of shape (..., hidden_dim)
        """
        x = agents_view

        # First layer to project from flat features to hidden dimension
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)

        # Hidden layers with residual connections
        for _ in range(self.num_layers - 1):
            # Residual connection
            residual = x
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
            x = x + residual
            x = nn.relu(x)

        return x


class JobShopAttentionTorso(nn.Module):
    """
    A more sophisticated torso that tries to reconstruct structure from flat features.
    """
    num_jobs: int
    num_machines: int
    max_num_ops: int
    hidden_dim: int = 256
    num_heads: int = 4

    def setup(self):
        # Calculate indices for unpacking flat features
        ops_size = self.num_jobs * self.max_num_ops
        self.ops_machine_ids_idx = (0, ops_size)
        self.ops_durations_idx = (ops_size, 2 * ops_size)
        self.ops_mask_idx = (2 * ops_size, 3 * ops_size)
        self.machines_job_ids_idx = (3 * ops_size, 3 * ops_size + self.num_machines)
        self.machines_times_idx = (3 * ops_size + self.num_machines,
                                   3 * ops_size + 2 * self.num_machines)

    @nn.compact
    def __call__(self, agents_view: jnp.ndarray) -> jnp.ndarray:
        """
        Process flattened features by reconstructing some structure.
        """
        # Save original shape
        original_shape = agents_view.shape[:-1]
        feature_dim = agents_view.shape[-1]

        # Flatten batch and agent dimensions if needed
        if agents_view.ndim > 2:
            agents_view = agents_view.reshape(-1, feature_dim)

        # Unpack different components (approximately)
        # Note: This is a heuristic since we normalized in the wrapper
        ops_features = agents_view[:, :3 * self.num_jobs * self.max_num_ops]
        machine_features = agents_view[:, 3 * self.num_jobs * self.max_num_ops:]

        # Process operations features
        ops_encoded = nn.Dense(self.hidden_dim)(ops_features)
        ops_encoded = nn.relu(ops_encoded)

        # Process machine features
        machine_encoded = nn.Dense(self.hidden_dim)(machine_features)
        machine_encoded = nn.relu(machine_encoded)

        # Self-attention on combined features
        combined = ops_encoded + machine_encoded

        # Multi-head attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2))
        )(combined)

        # Final projection
        output = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(attn_output)
        output = nn.relu(output)

        # Reshape back to original shape
        output = output.reshape(*original_shape, self.hidden_dim)

        return output


# ============================================================================
# Direct JobShop Networks (for potential future use with raw observations)
# ============================================================================

class JobShopActor(nn.Module):
    """Actor network that directly processes JobShop observations."""
    num_jobs: int
    max_num_ops: int
    num_machines: int
    num_actions: int  # num_machines * (num_jobs + 1)
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, observation: JobShopObs) -> jnp.ndarray:
        """
        Process JobShop observation directly and return action logits.

        Args:
            observation: JobShop observation with ops_machine_ids, ops_durations, etc.

        Returns:
            Action logits of shape (batch, num_machines, num_actions_per_machine)
        """
        # Process operations information
        ops_machine_emb = nn.Embed(
            num_embeddings=self.num_machines,
            features=self.hidden_dim // 4
        )(observation.ops_machine_ids)

        ops_duration_emb = nn.Dense(self.hidden_dim // 4)(
            observation.ops_durations[..., None]
        )

        ops_features = jnp.concatenate([ops_machine_emb, ops_duration_emb], axis=-1)
        ops_features = ops_features * observation.ops_mask[..., None]

        # Aggregate operation features per job
        job_features = jnp.sum(ops_features, axis=-2) / jnp.maximum(
            jnp.sum(observation.ops_mask, axis=-1, keepdims=True), 1
        )

        # Process machine information
        machine_job_emb = nn.Embed(
            num_embeddings=self.num_jobs + 1,
            features=self.hidden_dim // 2
        )(observation.machines_job_ids)

        machine_time_emb = nn.Dense(self.hidden_dim // 2)(
            observation.machines_remaining_times[..., None]
        )

        machine_features = jnp.concatenate([machine_job_emb, machine_time_emb], axis=-1)

        # Combine job and machine features
        # Create cross-attention between machines and jobs
        machine_features = nn.Dense(self.hidden_dim)(machine_features)
        job_features = nn.Dense(self.hidden_dim)(job_features)

        # Simple attention mechanism
        machine_query = nn.Dense(self.hidden_dim)(machine_features)
        job_keys = nn.Dense(self.hidden_dim)(job_features)
        job_values = nn.Dense(self.hidden_dim)(job_features)

        # Compute attention scores
        scores = jnp.einsum('...md,...jd->...mj', machine_query, job_keys)
        scores = scores / jnp.sqrt(self.hidden_dim)
        attention_weights = nn.softmax(scores, axis=-1)

        # Apply attention
        machine_context = jnp.einsum('...mj,...jd->...md', attention_weights, job_values)

        # Combine with original machine features
        combined_features = machine_features + machine_context
        combined_features = nn.relu(combined_features)
        combined_features = nn.Dense(self.hidden_dim)(combined_features)
        combined_features = nn.relu(combined_features)

        # Output logits for each machine
        # Each machine can choose from (num_jobs + 1) actions
        logits = nn.Dense(self.num_jobs + 1)(combined_features)

        # Apply action mask
        masked_logits = jnp.where(
            observation.action_mask,
            logits,
            jnp.finfo(jnp.float32).min
        )

        return masked_logits


class JobShopCritic(nn.Module):
    """Critic network that directly processes JobShop observations."""
    num_jobs: int
    max_num_ops: int
    num_machines: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, observation: JobShopObs) -> jnp.ndarray:
        """
        Process JobShop observation and return value estimates.

        Args:
            observation: JobShop observation

        Returns:
            Value estimates of shape (batch, num_machines)
        """
        # Similar encoding as actor
        ops_machine_emb = nn.Embed(
            num_embeddings=self.num_machines,
            features=self.hidden_dim // 4
        )(observation.ops_machine_ids)

        ops_duration_emb = nn.Dense(self.hidden_dim // 4)(
            observation.ops_durations[..., None]
        )

        ops_features = jnp.concatenate([ops_machine_emb, ops_duration_emb], axis=-1)
        ops_features = ops_features * observation.ops_mask[..., None]

        # Global aggregation of all operations
        global_ops_features = jnp.mean(
            jnp.sum(ops_features, axis=-2) / jnp.maximum(
                jnp.sum(observation.ops_mask, axis=-1, keepdims=True), 1
            ),
            axis=-2
        )

        # Machine features
        machine_job_emb = nn.Embed(
            num_embeddings=self.num_jobs + 1,
            features=self.hidden_dim // 2
        )(observation.machines_job_ids)

        machine_time_emb = nn.Dense(self.hidden_dim // 2)(
            observation.machines_remaining_times[..., None]
        )

        machine_features = jnp.concatenate([machine_job_emb, machine_time_emb], axis=-1)

        # Combine global and per-machine features
        global_features_expanded = jnp.expand_dims(global_ops_features, axis=-2)
        global_features_expanded = jnp.tile(
            global_features_expanded,
            (1,) * (global_features_expanded.ndim - 2) + (self.num_machines, 1)
        )

        combined_features = jnp.concatenate(
            [machine_features, global_features_expanded],
            axis=-1
        )

        # Process through MLP
        x = nn.Dense(self.hidden_dim)(combined_features)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)

        # Output one value per machine
        values = nn.Dense(1)(x)
        values = jnp.squeeze(values, axis=-1)

        return values


# ============================================================================
# Helper Classes and Functions
# ============================================================================

class JobShopObservationWrapper(NamedTuple):
    """Wrapper to pass both JobShop observation and any additional info."""
    jobshop_obs: JobShopObs
    step_count: jnp.ndarray = None