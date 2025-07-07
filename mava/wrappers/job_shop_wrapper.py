"""
job_shop_wrapper.py

A Mava multi-agent wrapper for Jumanji's JobShop environment.
"""
import math
from functools import cached_property
from typing import Dict, Tuple

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

        # Store environment parameters
        self.num_jobs = env.num_jobs
        self.max_num_ops = env.max_num_ops
        self.num_machines = env.num_machines

        # Define number of actions per agent (jobs + no-op)
        self.num_actions = env.num_jobs + 1

        # Initialize base wrapper (sets self._env, self.num_agents, self.time_limit)
        super().__init__(env, add_global_state)

        # Create the encoder once during initialization
        from mava.networks.job_shop_network import JobShopEncoder
        self.encoder = JobShopEncoder(
            num_jobs=self.num_jobs,
            max_num_ops=self.max_num_ops,
            num_machines=self.num_machines,
            embedding_dim=64
        )

    def reset(self, key) -> Tuple[chex.ArrayTree, TimeStep]:
        """
        Reset the environment and ensure the initial timestep is formatted
        as a Mava multi-agent timestep.
        """
        state, timestep = self._env.reset(key)  # Unpack the tuple
        modified_timestep = self.modify_timestep(timestep, state)  # Pass both
        return state, modified_timestep  # Return tuple as expected

    def step(self, state, actions) -> Tuple[chex.ArrayTree, TimeStep]:
        """Step the environment with multi-agent actions."""
        # Convert multi-agent actions to single-agent if needed
        if actions.ndim == 2:  # (batch, num_agents)
            # For JobShop, we might need to select one agent's action
            # or combine them somehow. For now, let's use the first agent
            actions = actions[:, 0]

        state, timestep = self._env.step(state, actions)
        modified_timestep = self.modify_timestep(timestep, state)
        return state, modified_timestep

    def modify_timestep(self, timestep: TimeStep, state=None) -> TimeStep:
        """
        Convert a single-agent Jumanji timestep into a multi-agent timestep,
        wrapping the raw Jumanji observation into Mava's Observation.
        """
        # Use the encoder to process the JobShop observation
        # Note: The encoder expects a single observation, not batched
        obs = timestep.observation

        # Initialize the encoder parameters if not done yet
        if not hasattr(self, '_encoder_params'):
            import jax
            dummy_obs = self._env.observation_spec.generate_value()
            self._encoder_params = self.encoder.init(
                jax.random.PRNGKey(0), dummy_obs
            )

        # Apply the encoder to get agents_view
        agents_view = self.encoder.apply(self._encoder_params, obs)

        # Ensure agents_view has the right shape (num_agents, feature_dim)
        if agents_view.ndim == 2:  # (1, feature_dim)
            # Repeat for all agents
            agents_view = jnp.repeat(agents_view, self.num_agents, axis=0)

        # Handle action mask - ensure it's (num_agents, num_actions)
        action_mask = obs.action_mask
        if action_mask.ndim == 2:  # Already (num_machines, num_jobs+1)
            # Good, this is what we expect
            pass
        elif action_mask.ndim == 1:  # Single agent mask
            # Expand to multi-agent
            action_mask = jnp.expand_dims(action_mask, 0)
            action_mask = jnp.repeat(action_mask, self.num_agents, axis=0)

        # Step count - use the timestep's step counter
        step_count = jnp.full((self.num_agents,), timestep.step_count if hasattr(timestep, 'step_count') else 0)

        # Build Mava Observation
        observation = Observation(
            agents_view=agents_view.astype(jnp.float32),
            action_mask=action_mask,
            step_count=step_count,
        )

        # Replicate reward and discount across agents
        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)

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
        # The encoder outputs 128-dimensional features
        return MavaObservationSpec(
            agents_view_shape=(self.num_agents, 128),
            action_mask_shape=(self.num_agents, self.num_actions),
            step_count_shape=(self.num_agents,),
        )

    @property
    def action_spec(self):
        """Override action spec for multi-agent."""
        # Return specs for multi-agent actions
        return specs.Array(
            shape=(self.num_agents,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_actions - 1
        )