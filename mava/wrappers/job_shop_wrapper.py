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
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_rewards(reward: chex.Array, num_agents: int, num_envs: int = 1) -> chex.Array:
    """
    Aggregate rewards across agents, handling scalar or array rewards.
    Args:
        reward: Scalar or array of shape [num_envs] or [num_envs, num_agents].
        num_agents: Number of agents (machines).
        num_envs: Number of environments (default 1 for single env).
    Returns:
        Array of shape [num_envs, num_agents] with aggregated rewards.
    """
    if reward.ndim == 0:
        return jnp.zeros((num_envs, num_agents), dtype=reward.dtype)
    if reward.ndim == 1:
        team_reward = reward
        return jnp.repeat(team_reward[:, None], num_agents, axis=-1)
    team_reward = jnp.sum(reward, axis=-1)
    return jnp.repeat(team_reward[:, None], num_agents, axis=-1)


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
        global_state = jnp.concatenate(obs.agents_view, axis=-1)
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


class RewardWrapper(Wrapper):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self.num_jobs = self._env.generator.num_jobs
        self.max_num_ops = self._env.generator.max_num_ops

    def step(self, state, action):
        next_state, timestep = self._env.step(state, action)
        reward = timestep.reward
        no_op = self.num_jobs * self.max_num_ops
        reward = jnp.where(
            (action != no_op) & jnp.any(state.ops_mask, axis=(1, 2) if state.ops_mask.ndim == 3 else (0, 1)),
            reward + 2.0,
            reward
        )
        return next_state, timestep._replace(reward=reward)


class NoOpPenaltyWrapper(Wrapper):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self.num_jobs = self._env.generator.num_jobs
        self.max_num_ops = self._env.generator.max_num_ops

    def step(self, state, action):
        no_op = self.num_jobs * self.max_num_ops
        is_no_op = action == no_op
        has_ops = jnp.any(state.ops_mask, axis=(1, 2) if state.ops_mask.ndim == 3 else (0, 1))
        is_batched = state.ops_mask.ndim == 3

        def single_step(s, a):
            return self._env.step(s, a)

        if is_batched:
            penalty_case = is_no_op & has_ops[:, None]  # Shape: [num_envs, num_machines]
            next_state, next_reward, done, info = jax.vmap(single_step)(state, action)
            reward = jnp.where(penalty_case, -10.0, next_reward)
            done = jnp.where(penalty_case, False, done)
            return next_state, reward, done, info
        else:
            penalty_case = is_no_op & has_ops
            if penalty_case.any():
                return state, jnp.full_like(action, -10.0), False, {}
            return single_step(state, action)


class ExtendedEpisodeWrapper(Wrapper):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self.num_jobs = self._env.generator.num_jobs
        self.max_num_ops = self._env.generator.max_num_ops

    def step(self, state, action):
        next_state, timestep = self._env.step(state, action)
        has_ops = jnp.any(state.ops_mask, axis=(1, 2) if state.ops_mask.ndim == 3 else (0, 1))
        done = jnp.where(has_ops, False, timestep.done)
        return next_state, timestep._replace(done=done)


class MultiAgentActionWrapper(Wrapper):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self.num_agents = self._env.generator.num_machines
        self.num_jobs = self._env.generator.num_jobs
        self.max_num_ops = self._env.generator.max_num_ops
        self.no_op = self.num_jobs * self.max_num_ops

    def step(self, state, actions: chex.Array) -> Tuple[State, TimeStep]:
        """
        Process simultaneous actions from all machine-agents.
        Args:
            state: Current environment state (batched or single).
            actions: Array of actions, shape: [num_envs, num_machines] or [num_machines].
        Returns:
            next_state: Updated state.
            timestep: Updated timestep with observations, rewards, and termination.
        """
        is_batched = actions.ndim == 2
        num_envs = actions.shape[0] if is_batched else 1
        actions = actions if is_batched else actions[None, :]
        state = jax.tree_map(lambda x: x if x.ndim == state.ops_mask.ndim else x[None], state)

        valid_actions_mask, per_agent_rewards = self._validate_and_reward_actions(state, actions)

        def step_single_env(s, a, valid_mask):
            new_state = s
            for machine_id, (action, valid) in enumerate(zip(a, valid_mask)):
                if valid:
                    logger.info(f"Machine {machine_id} scheduling action {action} in env")
                    new_state, _ = self._env.step(new_state, action)
            return new_state

        new_state = jax.vmap(step_single_env)(state, actions, valid_actions_mask) if is_batched else step_single_env(
            state[0], actions[0], valid_actions_mask[0])

        next_event_time = self._get_next_event_time(new_state)
        new_state = self._advance_time(new_state, next_event_time)

        _, timestep = jax.vmap(self._env.step)(new_state,
                                               jnp.full_like(actions, self.no_op)) if is_batched else self._env.step(
            new_state, self.no_op)
        has_ops = jnp.any(new_state.ops_mask, axis=(1, 2) if new_state.ops_mask.ndim == 3 else (0, 1))
        timestep = timestep._replace(
            reward=per_agent_rewards,
            done=~has_ops
        )

        logger.info(f"Step: Done={timestep.done}")
        return new_state[0] if not is_batched else new_state, timestep

    def _validate_and_reward_actions(self, state: State, actions: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Validate actions and compute per-agent rewards.
        Args:
            state: Batched or single state.
            actions: Shape: [num_envs, num_machines] or [num_machines].
        Returns:
            valid_actions_mask: Shape: [num_envs, num_machines] or [num_machines].
            per_agent_rewards: Shape: [num_envs, num_machines] or [num_machines].
        """
        is_batched = actions.ndim == 2

        def validate_single_env(s, a):
            valid_actions = []
            rewards = []
            for machine_id, action in enumerate(a):
                is_valid = self._is_action_valid(s, machine_id, action)
                valid_actions.append(is_valid)
                reward = jnp.where(
                    is_valid & (action != self.no_op),
                    -1.0 + 2.0 * jnp.any(s.ops_mask),
                    jnp.where((action == self.no_op) & jnp.any(s.ops_mask), -10.0, 0.0)
                )
                rewards.append(reward)
            return jnp.array(valid_actions), jnp.array(rewards)

        if is_batched:
            valid_actions_mask, per_agent_rewards = jax.vmap(validate_single_env)(state, actions)
        else:
            valid_actions_mask, per_agent_rewards = validate_single_env(state[0], actions[0])

        return valid_actions_mask, per_agent_rewards

    def _is_action_valid(self, state: State, machine_id: int, action: int) -> bool:
        """
        Check if the action is valid for the machine.
        Args:
            state: Single environment state.
            machine_id: Machine ID.
            action: Action (operation index or no-op).
        Returns:
            Boolean indicating if the action is valid.
        """
        if action == self.no_op:
            return True

        job_id = action // self.max_num_ops
        op_id = action % self.max_num_ops

        if not state.ops_mask[job_id, op_id]:
            return False

        if state.ops_machine_ids[job_id, op_id] != machine_id:
            return False

        if op_id > 0 and jnp.any(state.ops_mask[job_id, :op_id]):
            return False

        return True

    def _get_next_event_time(self, state: State) -> float:
        """
        Determine the time of the next event.
        Args:
            state: Batched or single state.
        Returns:
            Next event time (float or batched).
        """
        completion_times = state.scheduled_times + state.ops_durations
        active_ops = completion_times * state.ops_mask
        return jnp.min(active_ops, axis=(1, 2) if active_ops.ndim == 3 else (0, 1), initial=jnp.inf,
                       where=state.ops_mask)

    def _advance_time(self, state: State, next_event_time: float) -> State:
        """
        Advance the environment time and update state.
        Args:
            state: Batched or single state.
            next_event_time: Time to advance to (float or batched).
        Returns:
            Updated state.
        """
        completion_times = state.scheduled_times + state.ops_durations
        completed = (completion_times <= next_event_time[..., None, None]) & state.ops_mask
        new_ops_mask = state.ops_mask & ~completed
        new_scheduled_times = jnp.where(
            state.ops_mask,
            jnp.maximum(state.scheduled_times, next_event_time[..., None, None]),
            state.scheduled_times
        )
        return state._replace(
            ops_mask=new_ops_mask,
            scheduled_times=new_scheduled_times,
            step_count=state.step_count + 1
        )


class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: JobShop, add_global_state: bool = False):
        env = MultiAgentActionWrapper(env)
        env = ExtendedEpisodeWrapper(env)
        env = NoOpPenaltyWrapper(env)
        env = RewardWrapper(env)
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

        is_batched = state.ops_mask.ndim == 3
        num_envs = state.ops_mask.shape[0] if is_batched else 1
        makespan = jnp.max(state.scheduled_times + raw.ops_durations, axis=(1, 2) if is_batched else (0, 1),
                           where=raw.ops_mask, initial=0)
        num_ops = jnp.sum(raw.ops_mask, axis=(1, 2) if is_batched else (0, 1))

        reward = aggregate_rewards(timestep.reward, self.num_agents, num_envs)
        is_terminal = ~jnp.any(raw.ops_mask, axis=(1, 2) if is_batched else (0, 1))
        step_type = jnp.where(
            is_terminal,
            jnp.array(2, dtype=jnp.int32),
            jnp.array(1, dtype=jnp.int32)
        )
        logger.info(f"Modify timestep: Num_ops={num_ops}, Is_terminal={is_terminal}")
        makespan_log = makespan[0] if is_batched else makespan
        num_ops_log = num_ops[0] if is_batched else num_ops

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
            op_indices = \
            jnp.where(machine_ops_mask.reshape(*machine_ops_mask.shape[:-2], -1), size=max_ops_size, fill_value=-1)[0]
            machine_ops_ids = jnp.zeros(max_ops_size, dtype=float)
            machine_ops_durations = jnp.zeros(max_ops_size, dtype=float)
            machine_ops_mask_array = jnp.zeros(max_ops_size, dtype=float)

            def update_arrays(i, arrays):
                ids, durations, mask = arrays
                valid = op_indices[i] >= 0
                idx = jnp.where(valid, op_indices[i], 0)
                ids = jnp.where(valid, ids.at[i].set(
                    raw.ops_machine_ids.reshape(*raw.ops_machine_ids.shape[:-2], -1)[..., idx]), ids)
                durations = jnp.where(valid, durations.at[i].set(
                    raw.ops_durations.reshape(*raw.ops_durations.shape[:-2], -1)[..., idx]), durations)
                mask = jnp.where(valid, mask.at[i].set(raw.ops_mask.reshape(*raw.ops_mask.shape[:-2], -1)[..., idx]),
                                 mask)
                return ids, durations, mask

            machine_ops_ids, machine_ops_durations, machine_ops_mask_array = jax.lax.fori_loop(
                0, max_ops_size,
                update_arrays,
                (machine_ops_ids, machine_ops_durations, machine_ops_mask_array)
            )
            machine_ops_features = jnp.concatenate([
                machine_ops_ids,
                machine_ops_durations,
                machine_ops_mask_array,
            ])
            machine_state = jnp.array([
                raw.machines_job_ids[..., machine_id].astype(float),
                raw.machines_remaining_times[..., machine_id].astype(float),
            ])
            global_features = jnp.concatenate([
                raw.ops_machine_ids.reshape(*raw.ops_machine_ids.shape[:-2], -1).astype(float),
                raw.ops_durations.reshape(*raw.ops_durations.shape[:-2], -1).astype(float),
                raw.ops_mask.reshape(*raw.ops_mask.shape[:-2], -1).astype(float),
                state.scheduled_times.reshape(*state.scheduled_times.shape[:-2], -1).astype(float),
            ])
            agent_view = jnp.concatenate([machine_ops_features, machine_state, global_features], axis=-1)
            agents_view.append(agent_view)
        agents_view = jnp.stack(agents_view, axis=-2)  # Shape: [num_envs, num_agents, feature_dim]

        action_mask = jnp.zeros((*raw.action_mask.shape[:-2], self.num_agents, raw.action_mask.shape[-1]), dtype=bool)
        for machine_id in range(self.num_agents):
            machine_ops = (raw.ops_machine_ids == machine_id) & raw.ops_mask
            op_indices = jnp.where(machine_ops.reshape(*machine_ops.shape[:-2], -1), size=max_ops_size, fill_value=-1)[
                0]

            def set_action_mask(i, mask):
                valid = op_indices[i] >= 0
                idx = op_indices[i]
                return jnp.where(valid, mask.at[..., machine_id, idx].set(True), mask)

            action_mask = jax.lax.fori_loop(0, max_ops_size, set_action_mask, action_mask)

        no_op_idx = self._env.num_jobs * self._env.max_num_ops
        has_ops = jnp.any(raw.ops_mask, axis=(-2, -1))
        action_mask = action_mask.at[..., no_op_idx].set(~has_ops)

        step_count = jnp.repeat(state.step_count[..., None], self.num_agents, axis=-1)

        obs = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        discount = jnp.where(is_terminal, 0.0, 1.0)

        return timestep.replace(
            observation=obs,
            reward=reward,
            discount=discount,
            step_type=step_type,
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