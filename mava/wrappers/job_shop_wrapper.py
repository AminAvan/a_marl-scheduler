from functools import cached_property
from typing import Any, Dict

import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from jumanji.types import TimeStep
import logging
from dataclasses import dataclass

from mava.types import Observation
from mava.wrappers.jumanji import JumanjiMarlWrapper

logging.basicConfig(level=logging.INFO)

@dataclass
class CustomJobShopObservation(Observation):
    jumanji_obs: JumanjiObservation

class CustomJobShopObservation(Observation):
    """Custom observation class including full Jumanji observation."""
    def __init__(self, jumanji_obs: JumanjiObservation, **kwargs):
        super().__init__(**kwargs)
        self.jumanji_obs = jumanji_obs

class JobShopWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for JobShop with custom observation handling."""
    def __init__(self, env: Environment, add_global_state: bool = False):
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
        """Convert Jumanji observation to custom Mava observation."""
        obs = timestep.observation
        observation = CustomJobShopObservation(
            jumanji_obs=obs,
            agents_view=None,  # Placeholder, not used
            action_mask=obs.action_mask,
            step_count=jnp.repeat(obs.step_count, self.num_agents),
        )
        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        metrics: Dict[str, Any] = {"env_metrics": {}}
        return timestep.replace(observation=observation, reward=reward, discount=discount, extras=metrics)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Specification of custom environment observations."""
        jumanji_spec = self._env.observation_spec
        obs_data = {
            "jumanji_obs": jumanji_spec,
            "agents_view": specs.Array((1,), float, "agents_view"),
            "action_mask": jumanji_spec.action_mask,
            "step_count": specs.BoundedArray(
                (self.num_agents,), int, jnp.zeros(self.num_agents, dtype=int),
                jnp.repeat(self.time_limit, self.num_agents), "step_count",
            ),
        }
        return specs.Spec(CustomJobShopObservation, "CustomObservationSpec", **obs_data)

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.num_agents, name="action")