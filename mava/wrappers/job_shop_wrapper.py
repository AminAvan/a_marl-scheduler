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
import chex
import logging
from mava.types import Observation, ObservationGlobalState
from dm_env import specs

# Configure logging
logging.basicConfig(level=logging.INFO)

class ObservationSpec(NamedTuple):
    specs: Dict[str, specs.Array]

    def generate_value(self):
        # Produce a sample matching our Mava Observation fields
        return Observation(
            **{key: spec.generate_value() for key, spec in self.specs.items() if spec is not None}
        )

class JobShopPatched(JobShop):
    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
    ):
        # 1. Set integer attrs before calling super()
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

        # 2. Call JobShop.__init__, which will now see our ints
        super().__init__(generator=generator)

    def step(self, state: State, action: jnp.ndarray) -> Tuple[State, TimeStep]:
        return super().step(state, action)

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env)
        self.add_global_state = add_global_state
        # multi-agent count is number of machines
        self.num_agents = env.num_machines
        # expose action_dim for other wrappers (e.g. AutoResetWrapper)
        self.action_dim = self.num_agents
        # compute a default time_limit if absent
        self.time_limit = getattr(env, "time_limit", None)
        if self.time_limit is None:
            nj = getattr(env, "num_jobs", 5)
            mo = getattr(env, "max_num_ops", 4)
            md = getattr(env, "max_op_duration", 4)
            self.time_limit = nj * mo * md

    @abstractmethod
    def reset(self) -> Tuple[Any, TimeStep]:
        ...

    @abstractmethod
    def step(self, state: Any, action: Any) -> Tuple[Any, TimeStep]:
        ...

class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: Environment, add_global_state: bool = False):
        super().__init__(env, add_global_state)

    def reset(self) -> Tuple[State, TimeStep]:
        # reset underlying env state
        state = self._env.reset()
        # use our wrapper spec to generate a sample Observation with .agents_view
        obs = self.observation_spec.generate_value()
        timestep = TimeStep(
            observation=obs,
            reward=0.0,
            discount=1.0,
            step_type=None,
        )
        if self.add_global_state:
            gs = self.get_global_state(obs)
            wrapped = ObservationGlobalState(
                global_state=gs,
                agents_view=obs.agents_view,
                action_mask=obs.action_mask,
                step_count=obs.step_count,
            )
            return state, timestep.replace(observation=wrapped)
        return state, timestep

    def step(self, state: State, action: jnp.ndarray) -> Tuple[State, TimeStep]:
        # step underlying env
        state, jtime = self._env.step(state, action)
        # again generate a dummy Observation matching spec
        obs = self.observation_spec.generate_value()
        timestep = TimeStep(
            observation=obs,
            reward=jtime.reward,
            discount=jtime.discount,
            step_type=jtime.step_type,
        )
        if self.add_global_state:
            gs = self.get_global_state(obs)
            wrapped = ObservationGlobalState(
                global_state=gs,
                agents_view=obs.agents_view,
                action_mask=obs.action_mask,
                step_count=obs.step_count,
            )
            return state, timestep.replace(observation=wrapped)
        return state, timestep

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        # derive feature dim from underlying env's spec tree
        feat = self._env.observation_spec.agents_view.shape[-1]
        specs_map = {
            "agents_view": specs.BoundedArray(
                shape=(self.num_agents, feat),
                dtype=jnp.float32,
                minimum=0.0,
                maximum=1.0,
                name="agents_view",
            ),
            "action_mask": specs.BoundedArray(
                shape=(self.num_agents, self.num_agents),
                dtype=jnp.bool_,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            "step_count": specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.time_limit,
                name="step_count",
            ),
        }
        if self.add_global_state:
            specs_map["global_state"] = specs.Array(
                shape=(self.num_agents, self.num_agents * feat),
                dtype=jnp.float32,
                name="global_state",
            )
        return ObservationSpec(specs=specs_map)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.num_agents, name="action")
