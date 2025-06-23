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
    team_reward = jnp.sum(reward)
    return jnp.repeat(team_reward, num_agents)

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool):
        self.add_global_state = add_global_state
        super().__init__(env)
        if hasattr(self._env, "num_agents"):
            self.num_agents = self._env.num_agents
        else:
            self.num_agents = self._env.generator.num_machines
        self.time_limit = getattr(self._env, "time_limit", None)

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep, state) -> TimeStep[Observation]:
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        global_state = jnp.concatenate(obs.agents_view, axis=0)
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))
        return global_state

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep, state)
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
        state, timestep = self._env.step(state, action)
        timestep = self.modify_timestep(timestep, state)
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
        return int(self._env.action_spec.num_values[0])

# class JobShopWrapper(JumanjiMarlWrapper):
#     def __init__(self, env: JobShop, add_global_state: bool = False):
#         super().__init__(env, add_global_state)
#         self._env: JobShop
#         # Set a fallback time_limit if not provided by the environment
#         if self.time_limit is None:
#             num_jobs = self._env.generator.num_jobs
#             max_num_ops = self._env.generator.max_num_ops
#             max_op_duration = self._env.generator.max_op_duration
#             self.time_limit = num_jobs * max_num_ops * max_op_duration  # e.g., 5 * 4 * 4 = 80
#
#     def modify_timestep(self, timestep: TimeStep, state) -> TimeStep[Observation]:
#         obs = timestep.observation
#         if not hasattr(obs, "ops_machine_ids"):
#             return timestep
#         raw = obs
#
#         # Calculate and log makespan
#         makespan = jnp.max(state.scheduled_times + raw.ops_durations, where=raw.ops_mask, initial=0)
#         print("Step reward:", timestep.reward, "Is terminal:", timestep.step_type == jnp.array(2), "Makespan:",
#               makespan)
#
#         flat = jnp.concatenate([
#             raw.ops_machine_ids.ravel().astype(float),
#             raw.ops_durations.ravel().astype(float),
#             raw.ops_mask.astype(float).ravel(),
#             raw.machines_job_ids.ravel().astype(float),
#             raw.machines_remaining_times.ravel().astype(float),
#             state.scheduled_times.ravel().astype(float),
#         ], axis=0)
#
#         agents_view = jnp.tile(flat[None, :], (self.num_agents, 1))
#         action_mask = raw.action_mask
#         step_count = jnp.repeat(state.step_count, self.num_agents)  # Use state.step_count
#
#         obs = Observation(
#             agents_view=agents_view,
#             action_mask=action_mask,
#             step_count=step_count,
#         )
#
#         reward = jnp.repeat(timestep.reward, self.num_agents)
#         discount = jnp.repeat(timestep.discount, self.num_agents)
#         extras: Dict[str, Any] = {"env_metrics": {}}
#
#         return timestep.replace(
#             observation=obs,
#             reward=reward,
#             discount=discount,
#             extras=extras,
#         )
class JobShopWrapper(Wrapper):
    def __init__(self, env, num_agents: int = 4):
        super().__init__(env)
        self.num_agents = num_agents

    def reset(self, key):
        state, timestep = super().reset(key)
        return state, self.modify_timestep(timestep, state)

    def step(self, state, action):
        state, timestep = super().step(state, action)
        return state, self.modify_timestep(timestep, state)

    def modify_timestep(self, timestep: TimeStep, state) -> TimeStep[Observation]:
        obs = timestep.observation
        if not hasattr(obs, "ops_machine_ids"):
            return timestep
        raw = obs

        # Calculate makespan and number of active operations
        makespan = jnp.max(state.scheduled_times + raw.ops_durations, where=raw.ops_mask, initial=0)
        num_ops = jnp.sum(raw.ops_mask)

        # Convert to numpy for concrete logging to avoid tracing
        reward = np.array(timestep.reward[0])  # First agent's reward
        is_terminal = np.array(timestep.step_type == jnp.array(2))
        makespan = np.array(makespan)
        num_ops = np.array(num_ops)
        print(f"Step: Reward={reward}, Is terminal={is_terminal}, Makespan={makespan}, Num ops={num_ops}")

        flat = jnp.concatenate([
            raw.ops_machine_ids.ravel().astype(float),
            raw.ops_durations.ravel().astype(float),
            raw.ops_mask.astype(float).ravel(),
            raw.machines_job_ids.ravel().astype(float),
            raw.machines_remaining_times.ravel().astype(float),
            state.scheduled_times.ravel().astype(float),
        ], axis=0)

        agents_view = jnp.tile(flat[None, :], (self.num_agents, 1))
        action_mask = raw.action_mask
        step_count = jnp.repeat(state.step_count, self.num_agents)

        obs = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        extras = {"env_metrics": {"makespan": makespan}}

        return timestep.replace(
            observation=obs,
            reward=reward,
            discount=discount,
            extras=extras,
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        env_spec = self._env.observation_spec
        num_jobs, max_ops = env_spec.ops_machine_ids.shape
        num_machines = self.num_agents
        feature_dim = (
            num_jobs * max_ops
            + num_jobs * max_ops
            + num_jobs * max_ops
            + num_machines
            + num_machines
            + num_jobs * max_ops
        )
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
        if self.time_limit is not None:
            step_count_spec = specs.BoundedArray(
                (num_machines,),
                int,
                jnp.zeros(num_machines, int),
                jnp.repeat(self.time_limit, num_machines),
                "step_count",
            )
        else:
            step_count_spec = specs.Array(
                (num_machines,),
                int,
                "step_count",
            )
        obs_data = {
            "agents_view": agents_view_spec,
            "action_mask": action_mask_spec,
            "step_count": step_count_spec,
        }
        if self.add_global_state:
            global_dim = num_machines * feature_dim
            global_state_spec = specs.Array(
                (num_machines, global_dim),
                float,
                "global_state",
            )
            obs_data["global_state"] = global_state_spec
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)
        return specs.Spec(Observation, "ObservationSpec", **obs_data)