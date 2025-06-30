from functools import cached_property
from typing import Any, Dict, Tuple

from dataclasses import dataclass
from jumanji.specs import Spec

import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop, State
from jumanji.environments.packing.job_shop.generator import RandomGenerator
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from jumanji.types import TimeStep
import logging

from mava.types import Observation
from mava.wrappers.jumanji import JumanjiMarlWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)


# Define the CustomObservation class
@dataclass
class CustomObservation:
    jumanji_obs: JumanjiObservation

# Custom Observation Class
@dataclass
class CustomObservation:
    jumanji_obs: JumanjiObservation

# Environment Wrapper
class JobShopWrapper(JumanjiMarlWrapper):
    def modify_timestep(self, timestep: TimeStep) -> TimeStep[CustomObservation]:
        """Wrap the observation in CustomObservation."""
        observation = CustomObservation(jumanji_obs=timestep.observation)
        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        metrics = {"env_metrics": {}}
        return timestep.replace(
            observation=observation,
            reward=reward,
            discount=discount,
            extras=metrics
        )

    @cached_property
    def observation_spec(self) -> Spec:
        """Return a spec for CustomObservation."""
        jumanji_spec = self._env.observation_spec  # Jumanji observation spec
        return Spec(
            CustomObservation,
            "CustomObservationSpec",
            jumanji_obs=jumanji_spec
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

class CustomJobShopObservation(Observation):
    """Custom observation class that includes the full Jumanji observation."""
    def __init__(self, jumanji_obs: JumanjiObservation, **kwargs):
        super().__init__(**kwargs)
        self.jumanji_obs = jumanji_obs

class JobShopWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the JobShop environment with custom observation handling."""

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

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[CustomJobShopObservation]:
        """Convert Jumanji observation to a custom Mava observation with full Jumanji obs."""
        obs = timestep.observation  # This is the JumanjiObservation

        # Create a custom observation that includes the full Jumanji observation
        observation = CustomJobShopObservation(
            jumanji_obs=obs,
            agents_view=None,  # Not used, but required by Mava's Observation
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
        """Specification of the custom environment observations."""
        jumanji_spec = self._env.observation_spec

        # Define specs for the custom observation fields
        obs_data = {
            "jumanji_obs": jumanji_spec,  # Full Jumanji observation spec
            "agents_view": specs.Array((1,), float, "agents_view"),  # Placeholder, not used
            "action_mask": jumanji_spec.action_mask,
            "step_count": specs.BoundedArray(
                (self.num_agents,),
                int,
                jnp.zeros(self.num_agents, dtype=int),
                jnp.repeat(self.time_limit, self.num_agents),
                "step_count",
            ),
        }

        return specs.Spec(CustomJobShopObservation, "CustomObservationSpec", **obs_data)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.num_agents, name="action")