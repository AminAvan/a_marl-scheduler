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
        logger.info(f"Reset: Ops_mask={state.ops_mask}")
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
            state: Current environment state.
            actions: Array of actions, one per machine (shape: [num_machines]).
        Returns:
            next_state: Updated state after processing all valid actions.
            timestep: Updated timestep with observations, rewards, and termination.
        """
        logger.info(f"Step: Actions={actions}, Ops_mask before={state.ops_mask}")

        # Validate actions and compute per-agent rewards
        valid_actions_mask, per_agent_rewards = self._validate_and_reward_actions(state, actions)

        # Schedule valid operations
        new_state = state
        for machine_id, action in enumerate(actions):
            if valid_actions_mask[machine_id]:
                new_state, single_timestep = self._env.step(new_state, action)
                logger.info(f"Machine {machine_id} action {action}: New ops_mask={new_state.ops_mask}")

        # Advance time to the next event (placeholder)
        next_event_time = self._get_next_event_time(new_state)
        new_state = self._advance_time(new_state, next_event_time)

        # Compute timestep (use no-op to get updated observations)
        _, timestep = self._env.step(new_state, self.no_op)
        timestep = timestep._replace(
            reward=per_agent_rewards,
            done=~jnp.any(new_state.ops_mask)
        )

        logger.info(f"Step: Ops_mask after={new_state.ops_mask}, Done={timestep.done}")
        return new_state, timestep

    def _validate_and_reward_actions(self, state: State, actions: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Validate actions and compute per-agent rewards.
        Returns:
            valid_actions_mask: Boolean array indicating valid actions (shape: [num_machines]).
            per_agent_rewards: Rewards for each agent (shape: [num_machines]).
        """
        valid_actions_mask = []
        per_agent_rewards = []

        for machine_id, action in enumerate(actions):
            is_valid = self._is_action_valid(state, machine_id, action)
            valid_actions_mask.append(is_valid)

            # Compute reward based on action
            if is_valid and action != self.no_op:
                reward = -1.0  # Base reward for scheduling
                if jnp.any(state.ops_mask):
                    reward += 2.0  # RewardWrapper logic
            elif action == self.no_op and jnp.any(state.ops_mask):
                reward = -10.0  # NoOpPenaltyWrapper logic
            else:
                reward = 0.0
            per_agent_rewards.append(reward)

        return jnp.array(valid_actions_mask), jnp.array(per_agent_rewards)

    def _is_action_valid(self, state: State, machine_id: int, action: int) -> bool:
        """
        Check if the action is valid for the machine.
        Args:
            state: Current environment state.
            machine_id: ID of the machine-agent.
            action: Selected action (operation index or no-op).
        Returns:
            Boolean indicating if the action is valid.
        """
        if action == self.no_op:
            return True

        job_id = action // self.max_num_ops
        op_id = action % self.max_num_ops

        # Check if operation is available
        if not state.ops_mask[job_id, op_id]:
            return False

        # Check if operation is assigned to this machine
        if state.ops_machine_ids[job_id, op_id] != machine_id:
            return False

        # Check if predecessor operations are completed
        if op_id > 0 and jnp.any(state.ops_mask[job_id, :op_id]):
            return False

        # Check if machine is idle (simplified; adjust based on JobShop state)
        return True

    def _get_next_event_time(self, state: State) -> float:
        """
        Determine the time of the next event (e.g., operation completion).
        Returns:
            Time of the next event (float).
        """
        # Compute completion times for active operations
        completion_times = state.scheduled_times + state.ops_durations
        active_ops = completion_times * state.ops_mask
        return jnp.min(active_ops, initial=jnp.inf, where=state.ops_mask)

    def _advance_time(self, state: State, next_event_time: float) -> State:
        """
        Advance the environment time and update state.
        Args:
            state: Current state.
            next_event_time: Time to advance to.
        Returns:
            Updated state.
        """
        # Update scheduled_times and ops_mask for completed operations
        completion_times = state.scheduled_times + state.ops_durations
        completed = (completion_times <= next_event_time) & state.ops_mask
        new_ops_mask = state.ops_mask & ~completed

        # Update scheduled_times for remaining operations
        new_scheduled_times = jnp.where(
            state.ops_mask,
            jnp.maximum(state.scheduled_times, next_event_time),
            state.scheduled_times
        )

        return state._replace(
            ops_mask=new_ops_mask,
            scheduled_times=new_scheduled_times,
            step_count=state.step_count + 1
        )


class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: JobShop, add_global_state: bool = False):
        # Apply wrapper stack
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

        makespan = jnp.max(state.scheduled_times + raw.ops_durations, where=raw.ops_mask, initial=0)
        num_ops = jnp.sum(raw.ops_mask)

        reward = timestep.reward  # Per-agent rewards from MultiAgentActionWrapper
        is_terminal = ~jnp.any(raw.ops_mask)
        step_type = jnp.where(
            is_terminal,
            jnp.array(2, dtype=jnp.int32),
            jnp.array(1, dtype=jnp.int32)
        )
        logger.info(f"Modify timestep: Num_ops={num_ops}, Is_terminal={is_terminal}, Ops_mask={raw.ops_mask}")
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
            machine_ops_ids = jnp.zeros(max_ops_size, dtype=float)
            machine_ops_durations = jnp.zeros(max_ops_size, dtype=float)
            machine_ops_mask_array = jnp.zeros(max_ops_size, dtype=float)

            def update_arrays(i, arrays):
                ids, durations, mask = arrays
                valid = op_indices[i] >= 0
                idx = jnp.where(valid, op_indices[i], 0)
                ids = jnp.where(valid, ids.at[i].set(raw.ops_machine_ids.ravel()[idx]), ids)
                durations = jnp.where(valid, durations.at[i].set(raw.ops_durations.ravel()[idx]), durations)
                mask = jnp.where(valid, mask.at[i].set(raw.ops_mask.ravel()[idx]), mask)
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

            def set_action_mask(i, mask):
                valid = op_indices[i] >= 0
                return jnp.where(valid, mask.at[machine_id, op_indices[i]].set(True), mask)

            action_mask = jax.lax.fori_loop(
                0, max_ops_size,
                set_action_mask,
                action_mask
            )
            no_op_idx = self._env.num_jobs * self._env.max_num_ops
            has_ops = jnp.any(machine_ops)
            action_mask = jnp.where(
                ~has_ops,
                action_mask.at[machine_id, no_op_idx].set(True),
                action_mask
            )

        step_count = jnp.repeat(state.step_count, self.num_agents)

        obs = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        reward = aggregate_rewards(timestep.reward, self.num_agents)
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