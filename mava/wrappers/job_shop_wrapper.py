from functools import cached_property
from typing import NamedTuple, Optional, Dict

import jax.numpy as jnp
import chex
from jumanji import specs
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from jumanji.types import TimeStep
import logging

from mava.wrappers.jumanji import JumanjiMarlWrapper

logging.basicConfig(level=logging.INFO)

class CustomJobShopObservation(NamedTuple):
    """
    Carries the original Jumanji JobShop observation plus
    per-agent views, action mask, and optional step counter.
    """
    jumanji_obs: JumanjiObservation                  # full Jumanji observation
    agents_view: chex.Array                          # shape (num_agents, obs_dim)
    action_mask: chex.Array                          # shape (num_agents, num_actions)
    step_count: Optional[chex.Array] = None          # optional per-agent step counter

class JobShopWrapper(JumanjiMarlWrapper):
    """
    Multi-agent wrapper around Jumanji's JobShop, splitting
    the single-agent environment into one agent per machine.
    """

    def __init__(
        self,
        env: JobShop,
        add_global_state: bool = False,
    ):
        # Underlying JobShop spec
        j_spec = env.observation_spec()
        # Number of machines == number of agents
        self.num_agents = int(j_spec.machines_job_ids.shape[0])
        # Time limit from the env
        self.time_limit = env.time_limit
        # Flatten mask to get per-agent feature dimension
        self.obs_feature_dim = int(jnp.prod(j_spec.ops_mask.shape))
        # Initialize parent (sets up wrappers, etc.)
        super().__init__(env, add_global_state)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Specification of the custom multi-agent observation."""
        jumanji_spec = self._env.observation_spec()
        obs_data: Dict[str, specs.Array] = {
            "jumanji_obs": jumanji_spec,
            "agents_view": specs.Array(
                shape=(self.num_agents, self.obs_feature_dim),
                dtype=jumanji_spec.dtype,
                name="agents_view",
            ),
            "action_mask": jumanji_spec.action_mask,
            "step_count": specs.BoundedArray(
                shape=(self.num_agents,),
                dtype=int,
                minimum=jnp.zeros(self.num_agents, dtype=int),
                maximum=jnp.repeat(self.time_limit, self.num_agents),
                name="step_count",
            ),
        }
        return specs.Spec(
            constructor=CustomJobShopObservation,
            name="CustomJobShopObservationSpec",
            **obs_data,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        """Each agent (machine) chooses a job to process."""
        return specs.DiscreteArray(num_values=self.num_agents, name="action")

    def reset(self, seed: int) -> TimeStep:
        """Reset underlying env and convert to multi-agent timestep."""
        ts = self._env.reset(seed)
        return self.modify_timestep(ts)

    def step(self, actions: chex.Array) -> TimeStep:
        """
        Step the underlying single-agent env using the first agent's pick,
        then convert to a multi-agent timestep.
        """
        sa_action = int(actions[0])
        ts = self._env.step(sa_action)
        return self.modify_timestep(ts)

    def modify_timestep(self, ts: TimeStep) -> TimeStep:
        """
        Convert a single-agent TimeStep into a multi-agent TimeStep.
        """
        # Build per-agent observation by flattening ops_mask
        flat_obs = ts.observation.ops_mask.reshape(-1)
        agents_view = jnp.repeat(
            jnp.expand_dims(flat_obs, 0), self.num_agents, axis=0
        )
        # Step count for each agent (optional)
        step_count = jnp.arange(self.num_agents)

        return TimeStep(
            observation=CustomJobShopObservation(
                jumanji_obs=ts.observation,
                agents_view=agents_view,
                action_mask=ts.observation.action_mask,
                step_count=step_count,
            ),
            reward=jnp.repeat(ts.reward, self.num_agents),
            discount=jnp.repeat(ts.discount, self.num_agents),
            step_type=jnp.repeat(ts.step_type, self.num_agents),
        )
