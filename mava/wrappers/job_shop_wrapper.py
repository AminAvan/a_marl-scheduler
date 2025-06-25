from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Tuple, Union  # Added Any import
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
import chex
import logging
from mava.types import Observation, ObservationGlobalState
from dm_env import specs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_rewards(reward: chex.Array, num_agents: int, num_envs: int = 1) -> chex.Array:
    """Aggregate environment reward across agents."""
    if reward.ndim == 0:  # Scalar reward
        return jnp.full((num_envs, num_agents), reward / num_agents)
    if reward.ndim == 1:  # Batched reward [num_envs]
        return jnp.repeat(reward[:, None], num_agents, axis=-1)
    return reward  # Already [num_envs, num_agents]

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env)
        self.add_global_state = add_global_state
        self.num_agents = env.generator.num_machines
        self.time_limit = getattr(env, "time_limit", None)

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep, state: Any) -> TimeStep:
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        global_state = jnp.concatenate(obs.agents_view, axis=-1)
        return jnp.tile(global_state, (self.num_agents, 1))

    def reset(self, key: chex.PRNGKey) -> Tuple[Any, TimeStep]:
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

    def step(self, state: Any, action: chex.Array) -> Tuple[Any, TimeStep]:
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

class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: JobShop, add_global_state: bool = False):
        super().__init__(env, add_global_state)
        self.num_jobs = env.num_jobs
        self.max_num_ops = env.max_num_ops
        self.action_dim = self.num_jobs * self.max_num_ops + 1

    def modify_timestep(self, timestep: TimeStep, state: Any) -> TimeStep:
        is_batched = state.ops_mask.ndim == 3
        num_envs = state.ops_mask.shape[0] if is_batched else 1
        obs = timestep.observation

        makespan = jnp.max(
            state.scheduled_times + obs.ops_durations,
            axis=(1, 2) if is_batched else (0, 1),
            where=obs.ops_mask,
            initial=0
        )
        num_ops = jnp.sum(obs.ops_mask, axis=(1, 2) if is_batched else (0, 1))
        is_terminal = ~jnp.any(obs.ops_mask, axis=(1, 2) if is_batched else (0, 1))
        logger.info(f"Num_ops={num_ops}, Is_terminal={is_terminal}")
        extras = {"env_metrics": {"makespan": makespan, "num_ops": num_ops}}

        reward = aggregate_rewards(timestep.reward, self.num_agents, num_envs)

        action_mask = jnp.zeros((num_envs, self.num_agents, self.action_dim), dtype=bool)
        max_ops_size = self.num_jobs * self.max_num_ops
        env_indices = jnp.arange(num_envs)[:, None]

        for machine_id in range(self.num_agents):
            machine_ops = (obs.ops_machine_ids == machine_id) & obs.ops_mask
            op_indices = jnp.where(
                machine_ops.reshape(num_envs, -1) if is_batched else machine_ops.reshape(-1),
                size=max_ops_size,
                fill_value=-1
            )[1] if is_batched else jnp.where(machine_ops.reshape(-1), size=max_ops_size, fill_value=-1)[0]
            if is_batched:
                action_mask = action_mask.at[env_indices, machine_id, op_indices].set(True)
            else:
                action_mask = action_mask.at[0, machine_id, op_indices].set(True)
        action_mask = action_mask.at[:, :, -1].set(True)

        feature_dim = self.num_jobs * self.max_num_ops * 3
        if is_batched:
            agents_view = jnp.concatenate([
                obs.ops_durations.reshape(num_envs, -1),
                obs.ops_mask.reshape(num_envs, -1),
                obs.ops_machine_ids.reshape(num_envs, -1),
            ], axis=-1)
            agents_view = jnp.repeat(agents_view[:, None, :], self.num_agents, axis=1)
        else:
            agents_view = jnp.concatenate([
                obs.ops_durations.reshape(-1),
                obs.ops_mask.reshape(-1),
                obs.ops_machine_ids.reshape(-1),
            ], axis=-1)
            agents_view = jnp.repeat(agents_view[None, :], self.num_agents, axis=0)
            agents_view = agents_view[None, ...]
        logger.info(f"Agents_view shape={agents_view.shape}, Action_mask shape={action_mask.shape}")

        step_count = jnp.repeat(
            state.step_count[..., None] if is_batched else state.step_count[None, None],
            self.num_agents,
            axis=-1
        )

        observation = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        return timestep.replace(observation=observation, reward=reward, extras=extras)

    @cached_property
    def observation_spec(self):
        from dm_env import specs
        feature_dim = self.num_jobs * self.max_num_ops * 3
        agents_view_spec = specs.Array((self.num_agents, feature_dim), float, "agents_view")
        action_mask_spec = specs.BoundedArray((self.num_agents, self.action_dim), bool, False, True, "action_mask")
        step_count_spec = specs.BoundedArray((self.num_agents,), int, 0, self.time_limit, "step_count")
        obs_data = {"agents_view": agents_view_spec, "action_mask": action_mask_spec, "step_count": step_count_spec}
        if self.add_global_state:
            global_state_spec = specs.Array((self.num_agents, self.num_agents * feature_dim), float, "global_state")
            obs_data["global_state"] = global_state_spec
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)
        return specs.Spec(Observation, "ObservationSpec", **obs_data)

    @cached_property
    def action_spec(self):
        from dm_env import specs
        return specs.DiscreteArray(num_values=self.action_dim, name="action")