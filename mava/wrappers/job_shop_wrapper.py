from typing import Dict
import chex
import jax.numpy as jnp
from jumanji.types import TimeStep
from mava.types import Observation

class JobShopWrapper:
    def __init__(self, env, num_agents, num_actions):
        self._env = env
        self.num_agents = num_agents
        self.num_actions = num_actions

    def reset(self, key) -> tuple:
        state, timestep = self._env.reset(key)
        modified_timestep = self.modify_timestep(timestep)
        return state, modified_timestep

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """
        Convert a single-agent Jumanji timestep into a multi-agent timestep,
        wrapping the raw Jumanji observation into Mava's Observation.
        """
        # Flatten the ops_mask into a vector for agents_view
        flat_obs = timestep.observation.ops_mask.reshape(-1)
        agents_view = jnp.repeat(
            jnp.expand_dims(flat_obs, 0), self.num_agents, axis=0
        )  # Shape: (num_agents, feature_dim)

        # Step count per agent
        step_count = jnp.arange(self.num_agents)  # Shape: (num_agents,)

        # Use the action_mask directly, assuming it's (num_agents, num_actions)
        action_mask = timestep.observation.action_mask  # Shape: (num_agents, num_actions)

        # Build Mava Observation
        observation = Observation(
            agents_view=agents_view.astype(float),
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

    def step(self, key, state, action):
        return self._env.step(key, state, action)