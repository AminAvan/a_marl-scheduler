# a draft by Amin Avan for writting a wrapper for JobShop in jumanji
# that the source is located in
# https://github.com/instadeepai/jumanji/tree/main/jumanji/environments/packing/job_shop
"""
It seems that the environment which defines in jumanji.py are already implemented multi-agent in jumanji
inherently; thus, their wrappers functions and classes does not work for my JobShop environment.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment

from jumanji.environments.packing.job_shop import JobShop

from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State


def aggregate_rewards(reward: chex.Array, num_agents: int) -> chex.Array:
    """Aggregate individual rewards across agents."""
    """copied from mava/wrappers/jumanji.py"""
    team_reward = jnp.sum(reward)
    return jnp.repeat(team_reward, num_agents)


class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool):
        self.add_global_state = add_global_state
        super().__init__(env)
        # either read an existing .num_agents, or assume “one agent per machine” for JobShop: ## added by amin
        if hasattr(self._env, "num_agents"):    ## added by amin
            self.num_agents = self._env.num_agents  ## added by amin
        else:   ## added by amin
            # JobShop stores its instance generator on `.generator`, which has .num_machines    ## added by amin
            self.num_agents = self._env.generator.num_machines  ## added by amin
        # time_limit may not exist on JobShop — default to None ## added by amin
        self.time_limit = getattr(self._env, "time_limit", None)    ## added by amin

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        """The default way to create a global state for an environment if it has no
        available global state - concatenate all observations.
        """
        global_state = jnp.concatenate(obs.agents_view, axis=0)
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))
        return global_state

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep)
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return state, timestep.replace(observation=observation)

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        timestep = self.modify_timestep(timestep)
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return state, timestep.replace(observation=observation)

        return state, timestep

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )

        obs_spec = self._env.observation_spec
        obs_data = {
            "agents_view": obs_spec.agents_view,
            "action_mask": obs_spec.action_mask,
            "step_count": step_count,
        }

        if self.add_global_state:
            num_obs_features = obs_spec.agents_view.shape[-1]
            global_state = specs.Array(
                (self._env.num_agents, self._env.num_agents * num_obs_features),
                obs_spec.agents_view.dtype,
                "global_state",
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        return int(self._env.action_spec.num_values[0])


class JobShopWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the JobShop environment inspired by
    'CleanerWrapper(JumanjiMarlWrapper)' in mava/wrappers/jumanji.py ."""

    def __init__(self, env: JobShop, add_global_state: bool = False):
        """inspired by 'def __init__(self, env: Cleaner,...' in mava/wrappers/jumanji.py"""
        super().__init__(env, add_global_state)
        self._env: JobShop

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        # pull out the raw, single-agent observation
        s = timestep.observation

        # flatten each field of the JobShop state and concatenate
        flat = jnp.concatenate([
            s.ops_machine_ids.ravel().astype(float),
            s.ops_durations.ravel().astype(float),
            s.ops_mask.astype(float).ravel(),
            s.machines_job_ids.ravel().astype(float),
            s.machines_remaining_times.ravel().astype(float),
            s.scheduled_times.ravel().astype(float),
        ], axis=0)

        # tile so each “agent” (machine) sees the same flattened state
        agents_view = jnp.tile(flat[None, :], (self.num_agents, 1))

        # use the env’s own action_mask and step count
        action_mask = s.action_mask
        step_count = jnp.repeat(s.step_count, self.num_agents)

        # build the Mava Observation
        observation = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        # replicate reward and discount across agents
        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)

        extras: Dict[str, Any] = {"env_metrics": {}}

        return timestep.replace(
            observation=observation,
            reward=reward,
            discount=discount,
            extras=extras,
        )

    @cached_property
    def observation_spec(
            self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        # ## need to cast the agents view and global state to floats as we do in modify timestep
        # inner_spec = super().observation_spec
        # spec = inner_spec.replace(agents_view=inner_spec.agents_view.replace(dtype=float))
        # if self.add_global_state:
        #     spec = spec.replace(global_state=inner_spec.global_state.replace(dtype=float))
        #
        # return spec
        # Pull the original Jumanji spec so we can read shapes
        env_spec = self._env.observation_spec
        # compute feature dim: sum of all flattened state fields
        num_jobs, max_ops = env_spec.ops_machine_ids.shape
        num_machines = self.num_agents
        feature_dim = (
                num_jobs * max_ops  # ops_machine_ids
              + num_jobs * max_ops  # ops_durations
              + num_jobs * max_ops  # ops_mask
              + num_machines  # machines_job_ids
              + num_machines  # machines_remaining_times
              + num_jobs * max_ops  # scheduled_times
        )
        # define new specs
        agents_view_spec = specs.Array(
            (num_machines, feature_dim),
            float,
            "agents_view",
        )
        action_mask_spec = specs.BoundedArray(
            (num_machines, env_spec.action_mask.shape[-1]),
            bool,
            False,
            True,
            "action_mask",
        )
        # step_count_spec = specs.BoundedArray(  # commented amin
        #     (num_machines,),  # commented amin
        #     int,  # commented amin
        #     jnp.zeros(num_machines, int), # commented amin
        #     jnp.repeat(self.time_limit or -1, num_machines),  # commented amin
        #     "step_count", # commented amin
        # )
        if self.time_limit is not None:     # added amin
            step_count_spec = specs.BoundedArray(   # added amin
                (num_machines,),    # added amin
                int,    # added amin
                jnp.zeros(num_machines, int),   # added amin
                jnp.repeat(self.time_limit, num_machines),  # added amin
                "step_count",   # added amin
            )   # added amin
        else:   # added amin
            # no fixed upper bound on episode length in JobShop # added amin
            step_count_spec = specs.Array(      # added amin
                (num_machines,),        # added amin
                int,    # added amin
                "step_count",   # added amin
            )   # added amin
        obs_data = {
            "agents_view": agents_view_spec,
            "action_mask": action_mask_spec,
            "step_count": step_count_spec,
        }
        if self.add_global_state:
            # fallback to concatenating agents_view across machines
            global_dim = num_machines * feature_dim
            global_state_spec = specs.Array(
                (num_machines, global_dim),
                float,
                "global_state",
            )
            obs_data["global_state"] = global_state_spec
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)