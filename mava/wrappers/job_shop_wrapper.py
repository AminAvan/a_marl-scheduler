from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Tuple, Union, Dict, NamedTuple
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop, State
from jumanji.environments.packing.job_shop.generator import Generator
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

class ObservationSpec(NamedTuple):
    """Custom observation spec for Mava compatibility."""
    specs: Dict[str, specs.Array]

    def generate_value(self):
        """Generate a sample observation based on the specs."""
        return Observation(
            **{key: spec.generate_value() for key, spec in self.specs.items() if spec is not None}
        )

class JobShopPatched(JobShop):
    def __init__(self, num_jobs: int, num_machines: int, max_num_ops: int, max_op_duration: int):
        generator = Generator(
            num_jobs=num_jobs,
            num_machines=num_machines,
            max_num_ops=max_num_ops,
            max_op_duration=max_op_duration
        )
        super().__init__(generator=generator)

    def step(self, state: State, action: jnp.ndarray) -> Tuple[State, TimeStep]:
        """Step the environment with JAX-compatible action validation."""
        indices = action[:, None]  # Shape: [num_machines, 1]
        selected_mask = jnp.take_along_axis(state.action_mask, indices, axis=1).squeeze(axis=1)  # Shape: [num_machines]
        invalid = ~jnp.all(selected_mask)

        if invalid:
            return state, TimeStep(
                step_type=state.step_type,
                reward=jnp.array(-1.0, dtype=jnp.float32),
                discount=state.discount,
                observation=self._state_to_observation(state),
                extras=state.extras,
            )
        return super().step(state, action)

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env)
        self.add_global_state = add_global_state
        self.num_agents = env.generator.num_machines
        self.time_limit = getattr(env, "time_limit", None)
        if self.time_limit is None:
            num_jobs = getattr(env.generator, "num_jobs", 5)
            max_num_ops = getattr(env.generator, "max_num_ops", 4)
            max_op_duration = getattr(env.generator, "max_op_duration", 4)
            self.time_limit = num_jobs * max_num_ops * max_op_duration

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
        logger.info(f"Step: Actions={action}, Num_ops={jnp.sum(state.ops_mask, axis=(-2, -1))}")
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
    def observation_spec(self) -> ObservationSpec:
        feature_dim = self._env.observation_spec().agents_view.shape[-1]
        obs_specs = {
            "agents_view": specs.Array(
                shape=(self.num_agents, feature_dim),
                dtype=jnp.float32,
                name="agents_view"
            ),
            "action_mask": specs.BoundedArray(
                shape=(self.num_agents, self._env.action_spec().num_values),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask"
            ),
            "step_count": specs.BoundedArray(
                shape=(self.num_agents,),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.time_limit,
                name="step_count"
            ),
        }
        if self.add_global_state:
            obs_specs["global_state"] = specs.Array(
                shape=(self.num_agents, self.num_agents * feature_dim),
                dtype=jnp.float32,
                name="global_state"
            )
        return ObservationSpec(specs=obs_specs)

class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, num_jobs: int = 5, num_machines: int = 4, max_num_ops: int = 4, max_op_duration: int = 4, add_global_state: bool = False):
        env = JobShopPatched(
            num_jobs=num_jobs,
            num_machines=num_machines,
            max_num_ops=max_num_ops,
            max_op_duration=max_op_duration
        )
        super().__init__(env, add_global_state)
        self.num_jobs = num_jobs
        self.max_num_ops = max_num_ops
        self.action_dim = self.num_jobs * self.max_num_ops + 1

    def modify_timestep(self, timestep: TimeStep, state: Any) -> TimeStep:
        is_batched = state.ops_mask.ndim == 3
        num_envs = state.ops_mask.shape[0] if is_batched else 1
        obs_durations = state.ops_durations
        obs_mask = state.ops_mask
        obs_machine_ids = state.ops_machine_ids

        makespan = jnp.max(
            state.scheduled_times + obs_durations,
            axis=(1, 2) if is_batched else (0, 1),
            where=obs_mask,
            initial=0
        )
        num_ops = jnp.sum(obs_mask, axis=(1, 2) if is_batched else (0, 1))
        is_terminal = ~jnp.any(obs_mask, axis=(1, 2) if is_batched else (0, 1))
        logger.info(f"Num_ops={num_ops}, Is_terminal={is_terminal}")
        extras = {"env_metrics": {"makespan": makespan, "num_ops": num_ops}}

        reward = aggregate_rewards(timestep.reward, self.num_agents, num_envs)

        action_mask = jnp.zeros((num_envs, self.num_agents, self.action_dim), dtype=bool)
        max_ops_size = self.num_jobs * self.max_num_ops
        env_indices = jnp.arange(num_envs)[:, None]

        for machine_id in range(self.num_agents):
            machine_ops = (obs_machine_ids == machine_id) & obs_mask
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
                obs_durations.reshape(num_envs, -1),
                obs_mask.reshape(num_envs, -1),
                obs_machine_ids.reshape(num_envs, -1),
            ], axis=-1)
            agents_view = jnp.repeat(agents_view[:, None, :], self.num_agents, axis=1)
        else:
            agents_view = jnp.concatenate([
                obs_durations.reshape(-1),
                obs_mask.reshape(-1),
                obs_machine_ids.reshape(-1),
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
    def observation_spec(self) -> ObservationSpec:
        feature_dim = self.num_jobs * self.max_num_ops * 3
        obs_specs = {
            "agents_view": specs.Array(
                shape=(self.num_agents, feature_dim),
                dtype=jnp.float32,
                name="agents_view"
            ),
            "action_mask": specs.BoundedArray(
                shape=(self.num_agents, self.action_dim),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask"
            ),
            "step_count": specs.BoundedArray(
                shape=(self.num_agents,),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.time_limit,
                name="step_count"
            ),
        }
        if self.add_global_state:
            obs_specs["global_state"] = specs.Array(
                shape=(self.num_agents, self.num_agents * feature_dim),
                dtype=jnp.float32,
                name="global_state"
            )
        return ObservationSpec(specs=obs_specs)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.action_dim, name="action")