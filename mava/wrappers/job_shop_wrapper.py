"""
job_shop_wrapper.py - Wrapper that converts JobShop to Mava format
"""
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import chex
from functools import partial
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.types import Observation as JobShopObs
from jumanji.types import TimeStep
from mava.types import Observation
from mava.wrappers.jumanji import JumanjiMarlWrapper


class JobShopWrapper(JumanjiMarlWrapper):
    """
    Wrapper that converts JobShop observations to Mava format.
    """
    def __init__(self, env: JobShop, add_global_state: bool = False):
        # Set required attributes
        env.num_agents = env.num_machines
        env.time_limit = env.num_jobs * env.max_num_ops * env.max_op_duration

        # Store environment parameters
        self.num_jobs = env.num_jobs
        self.max_num_ops = env.max_num_ops
        self.num_machines = env.num_machines
        self.num_actions = env.num_jobs + 1

        # Calculate feature dimension for agents_view
        # This will be the flattened size of all JobShop observation components
        self.feature_dim = self._calculate_feature_dim()

        # Initialize base wrapper
        super().__init__(env, add_global_state)

    def _calculate_feature_dim(self) -> int:
        """Calculate the total feature dimension when flattening JobShop obs."""
        # ops_machine_ids: (num_jobs, max_num_ops)
        # ops_durations: (num_jobs, max_num_ops)
        # ops_mask: (num_jobs, max_num_ops)
        # machines_job_ids: (num_machines,)
        # machines_remaining_times: (num_machines,)
        # Total: 3 * num_jobs * max_num_ops + 2 * num_machines
        return 3 * self.num_jobs * self.max_num_ops + 2 * self.num_machines

    def _jobshop_obs_to_agents_view(self, obs: JobShopObs) -> jnp.ndarray:
        """Convert JobShop observation to a flat feature vector."""
        # Flatten all components of the observation
        features = []

        # Add operation features
        features.append(obs.ops_machine_ids.flatten())
        features.append(obs.ops_durations.flatten())
        features.append(obs.ops_mask.flatten().astype(jnp.float32))

        # Add machine features
        features.append(obs.machines_job_ids.flatten())
        features.append(obs.machines_remaining_times.flatten())

        # Concatenate all features
        flat_features = jnp.concatenate(features)

        # Normalize features to reasonable range
        # This helps with training stability
        flat_features = flat_features / jnp.maximum(
            jnp.abs(flat_features).max(), 1.0
        )

        return flat_features

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """Convert JobShop timestep to Mava format."""
        obs = timestep.observation

        # Convert JobShop observation to flat features
        flat_features = self._jobshop_obs_to_agents_view(obs)

        # Create agents_view - each agent sees the same global state
        # Shape: (num_agents, feature_dim)
        agents_view = jnp.tile(flat_features[None, :], (self.num_agents, 1))

        # Handle action mask - JobShop already has it in the right format
        # Shape should be (num_machines, num_jobs + 1)
        action_mask = obs.action_mask

        # Create step count
        step_count = jnp.arange(self.num_agents)

        # Build Mava Observation
        mava_obs = Observation(
            agents_view=agents_view.astype(jnp.float32),
            action_mask=action_mask,
            step_count=step_count
        )

        # Handle rewards and discounts
        reward = timestep.reward
        discount = timestep.discount

        # Replicate for all agents
        if jnp.isscalar(reward) or reward.shape == ():
            reward = jnp.full((self.num_agents,), reward)
        if jnp.isscalar(discount) or discount.shape == ():
            discount = jnp.full((self.num_agents,), discount)

        # Store original JobShop observation in extras for networks that need it
        extras = timestep.extras if hasattr(timestep, "extras") else {}
        extras["jobshop_observation"] = obs

        return timestep.replace(
            observation=mava_obs,
            reward=reward,
            discount=discount,
            extras=extras
        )