from functools import cached_property
from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop, State
from jumanji.environments.packing.job_shop.generator import RandomGenerator
from jumanji.types import TimeStep
import logging

from mava.types import Observation, ObservationGlobalState
from mava.wrappers.jumanji import JumanjiMarlWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)

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


class JobShopWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the JobShop environment."""

    def __init__(self, env: Environment, add_global_state: bool = False):
        # The common wrapper expects `num_agents` and `time_limit` on the env.
        if not hasattr(env, "num_agents"):
            env.num_agents = getattr(env, "num_machines", 1)
        if not hasattr(env, "time_limit"):
            nj = getattr(env, "num_jobs", 5)
            mo = getattr(env, "max_num_ops", 4)
            md = getattr(env, "max_op_duration", 4)
            env.time_limit = nj * mo * md

        super().__init__(env, add_global_state)
        self._env: JobShop

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Convert Jumanji observation to a Mava observation."""

        obs = timestep.observation
        agents_view = jnp.concatenate(
            [obs.machines.astype(float), obs.jobs.astype(float)],
            axis=-1,
        )

        observation = Observation(
            agents_view=agents_view,
            action_mask=obs.action_mask,
            step_count=jnp.repeat(obs.step_count, self.num_agents),
        )

        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        metrics: Dict[str, Any] = {"env_metrics": {}}

        return timestep.replace(
            observation=observation, reward=reward, discount=discount, extras=metrics
        )

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Specification of the environment observations."""

        spec = self._env.observation_spec
        machine_feat = spec.machines.shape[-1]
        job_feat = spec.jobs.shape[-1]
        feat = machine_feat + job_feat

        agents_view = specs.Array(
            (self.num_agents, feat), spec.machines.dtype, "agents_view"
        )

        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )

        obs_data = {
            "agents_view": agents_view,
            "action_mask": spec.action_mask,
            "step_count": step_count,
        }

        if self.add_global_state:
            obs_data["global_state"] = specs.Array(
                (self.num_agents, self.num_agents * feat),
                spec.machines.dtype,
                "global_state",
            )
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.num_agents, name="action")