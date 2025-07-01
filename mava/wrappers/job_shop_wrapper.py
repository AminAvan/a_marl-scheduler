from functools import cached_property
from typing import NamedTuple, Optional, Dict

import jax.numpy as jnp
import chex
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from jumanji.types import TimeStep
import logging

from mava.types import Observation
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
    the single-agent environment into `num_agents` agents.
    """

    def __init__(
        self,
        env: JobShop,
        num_agents: int,
        add_global_state: bool = False,
    ):
        # Store the number of agents and environment time limit
        self.num_agents = num_agents
        self.time_limit = env.time_limit
        # Infer per-agent observation feature dimension from Jumanji spec
        j_spec = env.observation_spec()
        # Example: flatten the ops_mask to get feature size
        self.obs_feature_dim = int(jnp.prod(j_spec.ops_mask.shape))
        super().__init__(env, add_global_state)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Specification of the custom multi-agent observation."""
        # Grab the original single-agent spec
        jumanji_spec = self._env.observation_spec()

        # Build a mapping from field name to its Spec
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

        # Return a Spec whose constructor is our NamedTuple
        return specs.Spec(
            constructor=CustomJobShopObservation,
            name="CustomJobShopObservationSpec",
            **obs_data,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        # Each agent has num_agents possible actions
        return specs.DiscreteArray(num_values=self.num_agents, name="action")

    def reset(self, seed: int) -> TimeStep:
        # Reset the underlying single-agent environment
        ts: TimeStep = self._env.reset(seed)
        # Convert to multi-agent TimeStep
        return self._to_multi_agent_ts(ts)

    def step(self, actions: chex.Array) -> TimeStep:
        # Here, `actions` is an array of shape (num_agents,)
        # Flatten back to single-agent action if needed
        sa_action = int(actions[0])  # or your custom aggregation logic
        ts = self._env.step(sa_action)
        return self._to_multi_agent_ts(ts)

    def _to_multi_agent_ts(self, ts: TimeStep) -> TimeStep:
        """
        Convert a single-agent TimeStep into a multi-agent one,
        by wrapping the Jumanji observation and splitting/replicating
        into per-agent arrays.
        """
        # Example: replicate a flattened mask as each agent's view
        flat_obs = ts.observation.ops_mask.reshape(-1)
        agents_view = jnp.repeat(
            jnp.expand_dims(flat_obs, 0), self.num_agents, axis=0
        )
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
