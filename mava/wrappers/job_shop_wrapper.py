"""
job_shop_wrapper.py

A Mava multi-agent wrapper for Jumanji's JobShop environment.
"""
import math
from functools import cached_property
from typing import Dict

import jax.numpy as jnp
import chex
import jumanji.specs as specs
from jumanji.environments.packing.job_shop import JobShop
from jumanji.types import TimeStep
import logging

from mava.types import Observation  # Mava Observation NamedTuple
from mava.wrappers.jumanji import JumanjiMarlWrapper

logging.basicConfig(level=logging.INFO)


class MavaObservationSpec(specs.Spec):
    """Custom spec to generate Mava Observation objects."""

    def __init__(self, agents_view_shape, action_mask_shape, step_count_shape):
        self.agents_view_spec = specs.Array(shape=agents_view_shape, dtype=float)
        self.action_mask_spec = specs.Array(shape=action_mask_shape, dtype=bool)
        self.step_count_spec = specs.Array(shape=step_count_shape, dtype=int)

    def generate_value(self):
        return Observation(
            agents_view=self.agents_view_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
            step_count=self.step_count_spec.generate_value(),
        )

    def validate(self, value):
        # Basic validation can be implemented if needed
        pass


class JobShopWrapper(JumanjiMarlWrapper):
    """
    Multi-agent wrapper around Jumanji's JobShop,
    creating one agent per machine.
    """

    def __init__(
            self,
            env: JobShop,
            add_global_state: bool = False,
    ):
        # Determine number of agents (machines) and episode length
        num_agents = env.num_machines
        max_episode_steps = env.num_jobs * env.max_num_ops * env.max_op_duration
        # Inject into env for base wrapper
        env.num_agents = num_agents
        env.time_limit = max_episode_steps
        # Compute per-agent feature dimension by flattening ops_mask
        j_spec = env.observation_spec
        self.obs_feature_dim = math.prod(j_spec.ops_mask.shape)
        # Define number of actions per agent (jobs + no-op)
        self.num_actions = env.num_jobs + 1
        # Initialize base wrapper (sets self._env, self.num_agents, self.time_limit)
        super().__init__(env, add_global_state)

    def reset(self, key) -> TimeStep:
        """
        Reset the environment and ensure the initial timestep is formatted
        as a Mava multi-agent timestep.
        """
        state, timestep = self._env.reset(key)  # Unpack the tuple
        modified_timestep = self.modify_timestep(timestep)  # Pass the TimeStep object
        return state, modified_timestep  # Return tuple as expected

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """
        Convert a batched single-agent Jumanji timestep into a batched multi-agent timestep,
        wrapping the raw Jumanji observation into Mava's Observation.
        """
        batch_size = timestep.observation.ops_mask.shape[0]

        # Flatten ops_mask for each batch
        feature_dim = math.prod(timestep.observation.ops_mask.shape[1:])
        flat_obs = timestep.observation.ops_mask.reshape(batch_size, feature_dim)

        # Repeat the flattened observation for each agent
        agents_view = jnp.repeat(
            flat_obs[:, None, :], self.num_agents, axis=1
        )  # (batch_size, num_agents, feature_dim)

        # Create step_count: (batch_size, num_agents)
        step_count = jnp.tile(
            jnp.arange(self.num_agents)[None, :], (batch_size, 1)
        )

        # Handle action_mask shape
        action_mask = timestep.observation.action_mask
        expected_shape = (batch_size, self.num_agents, self.num_actions)
        if action_mask.shape == expected_shape:
            # Shape is already correct: (batch_size, num_agents, num_actions)
            pass
        elif action_mask.shape == (batch_size, self.num_actions):
            # Expand from (batch_size, num_actions) to (batch_size, num_agents, num_actions)
            action_mask = jnp.repeat(action_mask[:, None, :], self.num_agents, axis=1)
        else:
            raise ValueError(
                f"Unexpected action_mask shape: {action_mask.shape}, "
                f"expected {expected_shape} or {(batch_size, self.num_actions)}"
            )

        # Build Mava Observation
        observation = Observation(
            agents_view=agents_view.astype(float),
            action_mask=action_mask,
            step_count=step_count,
        )

        # Replicate reward and discount across agents
        reward = jnp.repeat(timestep.reward[:, None], self.num_agents, axis=1)  # (batch_size, num_agents)
        discount = jnp.repeat(timestep.discount[:, None], self.num_agents, axis=1)  # (batch_size, num_agents)

        # Preserve existing extras, if any
        extras: Dict[str, chex.Array] = timestep.extras if hasattr(timestep, "extras") else {}

        # Return a new TimeStep with Mava Observation
        return timestep.replace(
            observation=observation,
            reward=reward,
            discount=discount,
            extras=extras,
        )

    @property
    def observation_spec(self):
        """Override observation spec to match Mava's Observation structure."""
        return MavaObservationSpec(
            agents_view_shape=(self.num_agents, self.obs_feature_dim),
            action_mask_shape=(self.num_agents, self.num_actions),
            step_count_shape=(self.num_agents,),
        )