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
    team_reward = jnp.sum(reward, axis=-1)  # Sum across agents, preserve batch dim
    return jnp.repeat(team_reward[:, None], num_agents, axis=-1)  # Shape: [num_envs, num_agents]


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
        global_state = jnp.concatenate(obs.agents_view, axis=-1)  # Concatenate along feature dim
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))  # Repeat for each agent
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


class RewardWrapper(Wrapper):
    def __init__(self, env: JobShop):
        super().__init__(env)
        self.num_jobs = self._env.generator.num_jobs
        self.max_num_ops = self._env.generator.max_num_ops

    def step(self, state, action):
        next_state, timestep = self._env.step(state, action)
        reward = timestep.reward
        no_op = self.num_jobs * self.max_num_ops
        # Handle batched or single actions
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
        # Handle batched or single actions
        is_no_op = action == no_op
        has_ops = jnp.any(state.ops_mask, axis=(1, 2) if state.ops_mask.ndim == 3 else (0, 1))
        if state.ops_mask.ndim == 3:  # Batched case
            penalty_case = is_no_op & has_ops
            return jax.tree_map(
                lambda s, r: jax.lax.cond(
                    penalty_case,
                    lambda: (s, -10.0, False, {}),
                    lambda: self._env.step(s, action),
                ),
                state,
                action
            )
        else:  # Single environment case
            if is_no_op and has_ops:
                return state, -10.0, False, {}
            return self._env.step(state, action)


class ExtendedEpisodeWrapper(Wrapper):
    def step(self, state, action):
        next_state, timestep = self._env.step(state, action)
        # Handle batched or single state
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
        logger.info(f"Step: Actions={actions}, Ops_mask before={state.ops_mask}")

        # Handle batched or single actions
        is_batched = actions.ndim == 2
        actions = actions if is_batched else actions[None, :]  # Add batch dim if needed
        state = jax.tree_map(lambda x: x if x.ndim == state.ops_mask.ndim else x[None], state)

        # Validate actions and compute rewards
        valid_actions_mask, per_agent_rewards = self._validate_and_reward_actions(state, actions)

        # Schedule valid operations
        def step_single_env(s, a, valid_mask):
            new_state = s
            for machine_id, (action, valid) in enumerate(zip(a, valid_mask)):
                if valid:
                    logger.info(f"Machine {machine_id} scheduling action {action}")
                    new_state, _ = self._env.step(new_state, action)
            return new_state

        new_state = jax.vmap(step_single_env)(state, actions, valid_actions_mask) if is_batched else step_single_env(
            state[0], actions[0], valid_actions_mask[0])

        # Advance time
        next_event_time = self._get_next_event_time(new_state)
        new_state = self._advance_time(new_state, next_event_time)

        # Compute timestep
        _, timestep = jax.vmap(self._env.step)(new_state,
                                               jnp.full_like(actions, self.no_op)) if is_batched else self._env.step(
            new_state, self.no_op)
        has_ops = jnp.any(new_state.ops_mask, axis=(1, 2) if new_state.ops_mask.ndim == 3 else (0, 1))
        timestep = timestep._replace(
            reward=per_agent_rewards,
            done=~has_ops
        )

        logger.info(f"Step: Ops_mask after={new_state.ops_mask}, Done={timestep.done}")
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
                    jnp.where((action == self.no_op) & jnp.any(s.ops_mask),