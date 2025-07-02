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
        # Initialize base wrapper (sets self._env, self.num_agents, self.time_limit)
        super().__init__(env, add_global_state)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """
        Convert a single-agent Jumanji timestep into a multi-agent timestep,
        wrapping the raw Jumanji observation into Mava's Observation.
        """
        # Flatten the ops_mask into a vector per agent
        flat_obs = timestep.observation.ops_mask.reshape(-1)
        agents_view = jnp.repeat(
            jnp.expand_dims(flat_obs, 0), self.num_agents, axis=0
        )
        # Step count per agent
        step_count = jnp.arange(self.num_agents)

        # Build Mava Observation
        observation = Observation(
            agents_view=agents_view.astype(float),
            action_mask=timestep.observation.action_mask,
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
