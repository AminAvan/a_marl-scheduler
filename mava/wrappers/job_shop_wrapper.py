from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Tuple, Dict, NamedTuple
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop, State
from jumanji.environments.packing.job_shop.generator import RandomGenerator
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
import logging
from mava.types import Observation, ObservationGlobalState
from dm_env import specs

# Configure logging
logging.basicConfig(level=logging.INFO)

class ObservationSpec(NamedTuple):
    specs: Dict[str, specs.Array]

    def generate_value(self) -> Observation:
        return Observation(
            **{key: spec.generate_value() for key, spec in self.specs.items() if spec is not None}
        )

class JobShopPatched(JobShop):
    def __init__(self, num_jobs: int, num_machines: int, max_num_ops: int, max_op_duration: int):
        # Set integer attributes before parent init
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_num_ops = max_num_ops
        self.max_op_duration = max_op_duration
        generator = RandomGenerator(
            num_jobs=num_jobs,
            num_machines=num_machines,
            max_num_ops=max_num_ops,
            max_op_duration=max_op_duration,
        )
        super().__init__(generator=generator)

    def step(self, state: State, action: jnp.ndarray) -> Tuple[State, TimeStep]:
        return super().step(state, action)

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env)
        self.add_global_state = add_global_state
        self.num_agents = env.num_machines
        self.action_dim = self.num_agents
        self.time_limit = getattr(env, "time_limit", None)
        if self.time_limit is None:
            nj = getattr(env, "num_jobs", 5)
            mo = getattr(env, "max_num_ops", 4)
            md = getattr(env, "max_op_duration", 4)
            self.time_limit = nj * mo * md

    @abstractmethod
    def reset(self) -> Tuple[Any, Observation]:
        ...

    @abstractmethod
    def step(self, state: Any, action: Any) -> Tuple[Any, TimeStep]:
        ...

class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env, add_global_state)

    def reset(self) -> Tuple[State, TimeStep]:
        state, ts = self._env.reset()
        mava_obs = self._to_mava_obs(ts.observation)
        ts = ts.replace(observation=mava_obs)
        return state, ts

    def step(self, state: State, action: jnp.ndarray) -> Tuple[State, TimeStep]:
        state, ts = super().step(state, action)
        mava_obs = self._to_mava_obs(ts.observation)
        ts = ts.replace(observation=mava_obs)
        return state, ts

    def _to_mava_obs(self, jumanji_obs) -> Observation:
        # Example conversion: adapt based on JobShop's observation structure
        # Jumanji's Observation has 'machines', 'jobs', 'action_mask', etc.
        agents_view = jnp.concatenate([jumanji_obs.machines, jumanji_obs.jobs], axis=-1)
        action_mask = jumanji_obs.action_mask
        step_count = jumanji_obs.step_count
        if self.add_global_state:
            gs = self.get_global_state(Observation(
                agents_view=agents_view,
                action_mask=action_mask,
                step_count=step_count
            ))
            return ObservationGlobalState(
                global_state=gs,
                agents_view=agents_view,
                action_mask=action_mask,
                step_count=step_count
            )
        return Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        # Define based on actual Jumanji observation structure
        machine_shape = self._env.observation_spec().machines.shape
        job_shape = self._env.observation_spec().jobs.shape
        feat = machine_shape[-1] + job_shape[-1]  # Concatenated features
        specs_map = {
            "agents_view": specs.BoundedArray(
                shape=(self.num_agents, feat),
                dtype=jnp.float32,
                minimum=0.0,
                maximum=float('inf'),  # Adjust based on data
                name="agents_view",
            ),
            "action_mask": self._env.observation_spec().action_mask,
            "step_count": self._env.observation_spec().step_count,
        }
        if self.add_global_state:
            specs_map["global_state"] = specs.Array(
                shape=(self.num_agents, feat * self.num_agents),
                dtype=jnp.float32,
                name="global_state",
            )
        return ObservationSpec(specs=specs_map)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.num_agents, name="action")
