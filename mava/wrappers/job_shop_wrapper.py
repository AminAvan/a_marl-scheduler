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

class RewardWrapper(JobShop):
    def step(self, state, action):
        next_state, timestep = super().step(state, action)
        reward = timestep.reward
        if action != self.num_jobs * self.max_num_ops and jnp.any(state.ops_mask):
            reward += 2.0
        return next_state, timestep._replace(reward=reward)

class NoOpPenaltyWrapper(JobShop):
    def step(self, state, action):
        if action == self.num_jobs * self.max_num_ops and jnp.any(state.ops_mask):
            return state, -10.0, False, {}
        return super().step(state, action)

class ExtendedEpisodeWrapper(JobShop):
    def step(self, state, action):
        next_state, timestep = super().step(state, action)
        if jnp.any(state.ops_mask):
            timestep = timestep._replace(done=False)
        return next_state, timestep

class ActionAggregationWrapper(JobShop):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self._env: JobShop
        self.num_agents = self._env.generator.num_machines

    def step(self, state, actions: chex.Array) -> Tuple[State, TimeStep]:
        no_op = self._env.num_jobs * self._env.max_num_ops
        valid_actions = jnp.where(actions != no_op)[0]
        selected_action = actions[valid_actions[0]] if valid_actions.size > 0 else no_op
        return self._env.step(state, selected_action)

class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: JobShop, add_global_state: bool = False):
        super().__init__(env, add_global_state)
        self._env: JobShop
        if self.time_limit is None:
            num_jobs = self._env.generator.num_jobs
            max_num_ops = self._env.generator.max_num_ops
            max_op_duration = self._env.generator.max_op_duration
            self.time_limit = num_jobs * max_num_ops * max_op_duration

    def modify_timestep(self, timestep: TimeStep, state) -> TimeStep[Observation]:
        obs = timestep.observation
        if not hasattr(obs, "ops_machine_ids"):
            return timestep
        raw = obs

        makespan = jnp.max(state.scheduled_times + raw.ops_durations, where=raw.ops_mask, initial=0)
        num_ops = jnp.sum(raw.ops_mask)

        reward = timestep.reward[0] if hasattr(timestep.reward,
                                               'ndim') and timestep.reward.ndim > 0 else timestep.reward
        is_terminal = (timestep.step_type == jnp.array(2) if not hasattr(timestep.step_type,
                                                                         'ndim') or timestep.step_type.ndim == 0 else
                       timestep.step_type[0] == jnp.array(2))
        makespan_log = makespan[0] if hasattr(makespan, 'ndim') and makespan.ndim > 0 else makespan
        num_ops_log = num_ops[0] if hasattr(num_ops, 'ndim') and num_ops.ndim > 0 else num_ops

        extras = {
            "env_metrics": {
                "makespan": makespan,
                "reward": reward,
                "is_terminal": is_terminal,
                "num_ops": num_ops_log
            }
        }

        num_jobs = self._env.generator.num_jobs
        max_num_ops = self._env.generator.max_num_ops
        max_ops_size = num_jobs * max_num_ops
        agents_view = []
        for machine_id in range(self.num_agents):
            machine_ops_mask = (raw.ops_machine_ids == machine_id) & raw.ops_mask
            op_indices = jnp.where(machine_ops_mask.ravel(), size=max_ops_size, fill_value=-1)[0]
            # Get indices where op_indices >= 0
            valid_mask = op_indices >= 0
            valid_op_indices = jnp.where(valid_mask, op_indices, -1)[:jnp.sum(valid_mask)]
            machine_ops_ids = jnp.zeros(max_ops_size, dtype=float)
            machine_ops_durations = jnp.zeros(max_ops_size, dtype=float)
            machine_ops_mask_array = jnp.zeros(max_ops_size, dtype=float)

            def update_arrays(idx, arrays):
                ids, durations, mask = arrays
                ids = ids.at[idx].set(raw.ops_machine_ids.ravel()[idx])
                durations = durations.at[idx].set(raw.ops_durations.ravel()[idx])
                mask = mask.at[idx].set(raw.ops_mask.ravel()[idx])
                return ids, durations, mask

            machine_ops_ids, machine_ops_durations, machine_ops_mask_array = jax.lax.fori_loop(
                0, jnp.minimum(valid_op_indices.size, max_ops_size),
                lambda i, arrays: update_arrays(valid_op_indices[i], arrays),
                (machine_ops_ids, machine_ops_durations, machine_ops_mask_array)
            )
            machine_ops_features = jnp.concatenate([
                machine_ops_ids,
                machine_ops_durations,
                machine_ops_mask_array,
            ])
            machine_state = jnp.array([
                raw.machines_job_ids[machine_id].astype(float),
                raw.machines_remaining_times[machine_id].astype(float),
            ])
            global_features = jnp.concatenate([
                raw.ops_machine_ids.ravel().astype(float),
                raw.ops_durations.ravel().astype(float),
                raw.ops_mask.ravel().astype(float),
                state.scheduled_times.ravel().astype(float),
            ])
            agent_view = jnp.concatenate([machine_ops_features, machine_state, global_features])
            agents_view.append(agent_view)
        agents_view = jnp.stack(agents_view)

        action_mask = jnp.zeros((self.num_agents, raw.action_mask.shape[-1]), dtype=bool)
        for machine_id in range(self.num_agents):
            machine_ops = (raw.ops_machine_ids == machine_id) & raw.ops_mask
            op_indices = jnp.where(machine_ops.ravel(), size=max_ops_size, fill_value=-1)[0]
            # Get indices where op_indices >= 0
            valid_mask = op_indices >= 0
            valid_op_indices = jnp.where(valid_mask, op_indices, -1)[:jnp.sum(valid_mask)]
            action_mask = action_mask.at[machine_id, valid_op_indices].set(True)
            if not jnp.any(machine_ops):
                action_mask = action_mask.at[machine_id, self._env.num_jobs * self._env.max_num_ops].set(True)

        step_count = jnp.repeat(state.step_count, self.num_agents)

        obs = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)

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
        max_ops_size = num_jobs * max_ops
        feature_dim = (
            max_ops_size * 3
            + 2
            + max_ops_size * 4
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