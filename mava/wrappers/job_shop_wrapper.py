from abc import ABC, abstractmethod
from functools import cached_property
from typing import Tuple, Union

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

# Configure logging
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
        return jnp.repeat(reward[:, None], num_agents, axis=-1)
    return reward

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env)
        self.add_global_state = add_global_state
        self.num_agents = env.generator.num_machines
        self.time_limit = getattr(env, "time_limit", None)

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep, state) -> TimeStep[Observation]:
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        global_state = jnp.concatenate(obs.agents_view, axis=-1)
        return jnp.tile(global_state, (self.num_agents, 1))

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
            (self.num_agents,), int, 0, self.time_limit, "step_count"
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
                (self.num_agents, self.num_agents * num_obs_features),
                obs_spec.agents_view.dtype,
                "global_state",
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)
        return specs.Spec(Observation, "ObservationSpec", **obs_data)

class MultiAgentActionWrapper(Wrapper):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self.num_agents = env.generator.num_machines
        self.num_jobs = env.generator.num_jobs
        self.max_num_ops = env.generator.max_num_ops
        self.no_op = self.num_jobs * self.max_num_ops
        self.action_dim = self.no_op + 1

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep]:
        """
        Process simultaneous actions from all machine-agents.
        Args:
            state: Batched environment state (from jax.vmap).
            actions: Array of actions, shape: [num_envs, num_machines].
        Returns:
            next_state: Updated state.
            timestep: Updated timestep.
        """
        num_envs = actions.shape[0]
        logger.info(f"Step: num_envs={num_envs}, Actions shape={actions.shape}, Ops_mask shape={state.ops_mask.shape}")

        valid_actions_mask, per_agent_rewards = self._validate_and_reward_actions(state, actions)

        def step_single_env(s, a, valid_mask):
            new_state = s
            for machine_id, (action, valid) in enumerate(zip(a, valid_mask)):
                if valid and action != self.no_op:
                    logger.info(f"Machine {machine_id} scheduling action {action}")
                    new_state, _ = self._env.step(new_state, action)
            return new_state

        new_state = jax.vmap(step_single_env)(state, actions, valid_actions_mask)

        next_event_time = self._get_next_event_time(new_state)
        new_state = self._advance_time(new_state, next_event_time)

        _, timestep = jax.vmap(self._env.step)(new_state, jnp.full_like(actions, self.no_op))
        has_ops = jnp.any(new_state.ops_mask, axis=(1, 2))
        timestep = timestep._replace(reward=per_agent_rewards, done=~has_ops)

        logger.info(f"Step completed: Done={timestep.done}, Num_ops={jnp.sum(new_state.ops_mask, axis=(-2, -1))}")
        return new_state, timestep

    def _validate_and_reward_actions(self, state: State, actions: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Validate actions and compute per-agent rewards.
        Args:
            state: Batched state with fields [num_envs, ...].
            actions: Shape: [num_envs, num_machines].
        Returns:
            valid_actions_mask: Shape: [num_envs, num_machines].
            per_agent_rewards: Shape: [num_envs, num_machines].
        """
        def validate_single_env(ops_mask, ops_machine_ids, a):
            valid_actions = []
            rewards = []
            for machine_id, action in enumerate(a):
                is_valid = self._is_action_valid(ops_mask, ops_machine_ids, machine_id, action)
                valid_actions.append(is_valid)
                reward = jnp.where(
                    is_valid & (action != self.no_op),
                    -1.0,
                    jnp.where((action == self.no_op) & jnp.any(ops_mask), -10.0, 0.0)
                )
                rewards.append(reward)
            return jnp.array(valid_actions), jnp.array(rewards)

        valid_actions_mask, per_agent_rewards = jax.vmap(
            validate_single_env,
            in_axes=(0, 0, 0)
        )(state.ops_mask, state.ops_machine_ids, actions)
        return valid_actions_mask, per_agent_rewards

    def _is_action_valid(self, ops_mask: chex.Array, ops_machine_ids: chex.Array, machine_id: int, action: int) -> bool:
        is_no_op = action == self.no_op
        job_id = action // self.max_num_ops
        op_id = action % self.max_num_ops
        job_valid = (job_id >= 0) & (job_id < self.num_jobs)
        op_valid = (op_id >= 0) & (op_id < self.max_num_ops)
        valid_indices = job_valid & op_valid
        op_pending = jnp.where(valid_indices, ops_mask[job_id, op_id], False)
        machine_correct = jnp.where(valid_indices, ops_machine_ids[job_id, op_id] == machine_id, False)
        ops_before = jnp.arange(self.max_num_ops) < op_id
        preceding_pending = jnp.where(job_valid, jnp.any(ops_mask[job_id] & ops_before), True)
        valid_op = valid_indices & op_pending & machine_correct & ~preceding_pending
        return jnp.where(is_no_op, True, valid_op)

    def _get_next_event_time(self, state: State) -> chex.Array:
        completion_times = state.scheduled_times + state.ops_durations
        active_ops = completion_times * state.ops_mask
        return jnp.min(active_ops, axis=(1, 2), where=state.ops_mask, initial=jnp.inf)

    def _advance_time(self, state: State, next_event_time: chex.Array) -> State:
        completion_times = state.scheduled_times + state.ops_durations
        completed = (completion_times <= next_event_time[..., None, None]) & state.ops_mask
        new_ops_mask = state.ops_mask & ~completed
        new_scheduled_times = jnp.where(
            state.ops_mask,
            jnp.maximum(state.scheduled_times, next_event_time[..., None, None]),
            state.scheduled_times
        )
        logger.info(f"Advance time: Num_ops before={jnp.sum(state.ops_mask, axis=(-2, -1))}, after={jnp.sum(new_ops_mask, axis=(-2, -1))}")
        return state._replace(ops_mask=new_ops_mask, scheduled_times=new_scheduled_times, step_count=state.step_count + 1)

class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: JobShop, add_global_state: bool = False):
        env = MultiAgentActionWrapper(env)
        super().__init__(env, add_global_state)
        self.num_jobs = env.generator.num_jobs
        self.max_num_ops = env.generator.max_num_ops
        self.action_dim = self.num_jobs * self.max_num_ops + 1
        self.time_limit = self.num_jobs * self.max_num_ops * env.generator.max_op_duration

    def modify_timestep(self, timestep: TimeStep, state: State) -> TimeStep:
        """
        Modify the timestep to be compatible with Mava's multi-agent framework.
        This includes creating per-agent observations, action masks, and step counts.

        Args:
            timestep: The original timestep from the environment.
            state: The current state of the environment.

        Returns:
            A modified timestep with per-agent observations, rewards, and extras.
        """
        # Determine if the state is batched
        is_batched = state.ops_mask.ndim == 3
        num_envs = state.ops_mask.shape[0] if is_batched else 1

        # Extract raw observation from the state (not timestep.observation)
        # In Jumanji JobShop, the state contains the full observation data
        ops_durations = state.ops_durations
        ops_mask = state.ops_mask
        ops_machine_ids = state.ops_machine_ids

        # Calculate makespan and number of operations
        makespan = jnp.max(
            state.scheduled_times + ops_durations,
            axis=(1, 2) if is_batched else (0, 1),
            where=ops_mask,
            initial=0
        )
        num_ops = jnp.sum(ops_mask, axis=(1, 2) if is_batched else (0, 1))

        # Aggregate rewards for each agent
        reward = aggregate_rewards(timestep.reward, self.num_agents, num_envs)

        # Check if the episode is terminal
        is_terminal = ~jnp.any(ops_mask, axis=(1, 2) if is_batched else (0, 1))

        # Log for debugging
        logger.info(f"Num_ops={num_ops}, Is_terminal={is_terminal}")

        # Extras for logging or metrics
        extras = {"env_metrics": {"makespan": makespan, "num_ops": num_ops}}

        # Create action mask for each agent
        action_mask = jnp.zeros((num_envs, self.num_agents, self.action_dim), dtype=bool)
        max_ops_size = self.num_jobs * self.max_num_ops
        env_indices = jnp.arange(num_envs)[:, None]  # Shape (num_envs, 1)

        for machine_id in range(self.num_agents):
            # Identify operations for this machine
            machine_ops = (ops_machine_ids == machine_id) & ops_mask
            # Get valid operation indices
            op_indices = jnp.where(
                machine_ops.reshape(num_envs, -1) if is_batched else machine_ops.reshape(-1),
                size=max_ops_size,
                fill_value=-1
            )[1]  # Shape (num_envs, max_ops_size) or (max_ops_size,)

            # Set valid actions in action_mask
            if is_batched:
                action_mask = action_mask.at[env_indices, machine_id, op_indices].set(True)
            else:
                action_mask = action_mask.at[0, machine_id, op_indices].set(True)

        # Set no-op action (last index) as always valid
        action_mask = action_mask.at[:, :, -1].set(True)

        # Create per-agent observations
        # Combine relevant features into a flat array per agent
        # For each agent, provide a view of ops_durations and ops_mask filtered by machine
        if is_batched:
            feature_dim = self.num_jobs * self.max_num_ops * 2  # ops_durations + ops_mask
            agents_view = jnp.zeros((num_envs, self.num_agents, feature_dim), dtype=float)
            for machine_id in range(self.num_agents):
                machine_mask = (ops_machine_ids == machine_id)  # Shape [num_envs, num_jobs, max_num_ops]
                durations = jnp.where(machine_mask, ops_durations, 0.0)
                mask = machine_mask.astype(float)
                agent_features = jnp.concatenate(
                    [durations.reshape(num_envs, -1), mask.reshape(num_envs, -1)],
                    axis=-1
                )
                agents_view = agents_view.at[:, machine_id, :].set(agent_features)
        else:
            feature_dim = self.num_jobs * self.max_num_ops * 2
            agents_view = jnp.zeros((self.num_agents, feature_dim), dtype=float)
            for machine_id in range(self.num_agents):
                machine_mask = (ops_machine_ids == machine_id)
                durations = jnp.where(machine_mask, ops_durations, 0.0)
                mask = machine_mask.astype(float)
                agent_features = jnp.concatenate(
                    [durations.reshape(-1), mask.reshape(-1)],
                    axis=-1
                )
                agents_view = agents_view.at[machine_id, :].set(agent_features)
            agents_view = agents_view[None, ...]  # Add batch dimension

        # Step count for each agent
        step_count = jnp.repeat(
            state.step_count[..., None] if is_batched else state.step_count[None, None],
            self.num_agents,
            axis=-1
        )

        # Create Observation object
        observation = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        # Replace timestep observation and reward
        return timestep.replace(observation=observation, reward=reward, extras=extras)

        # Replace timestep observation and reward
        return timestep.replace(observation=observation, reward=reward, extras=extras)

    @cached_property
    def observation_spec(self):
        max_ops_size = self.num_jobs * self.max_num_ops
        feature_dim = max_ops_size * 3
        agents_view_spec = specs.Array((self.num_agents, feature_dim), float, "agents_view")
        action_mask_spec = specs.BoundedArray((self.num_agents, self.action_dim), bool, False, True, "action_mask")
        step_count_spec = specs.BoundedArray((self.num_agents,), int, 0, self.time_limit, "step_count")
        obs_data = {"agents_view": agents_view_spec, "action_mask": action_mask_spec, "step_count": step_count_spec}
        if self.add_global_state:
            global_state_spec = specs.Array((self.num_agents, self.num_agents * feature_dim), float, "global_state")
            obs_data["global_state"] = global_state_spec
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)
        return specs.Spec(Observation, "ObservationSpec", **obs_data)