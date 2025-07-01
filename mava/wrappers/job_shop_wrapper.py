"""
job_shop_wrapper.py

A Mava multi-agent wrapper for Jumanji's JobShop environment.
"""
from functools import cached_property
from typing import NamedTuple, Optional, Dict
import math

import jax.numpy as jnp
import chex
import jumanji.specs as specs
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
    jumanji_obs: JumanjiObservation                  # the raw Jumanji observation
    agents_view: chex.Array                          # shape (num_agents, obs_dim)
    action_mask: chex.Array                          # shape (num_agents, num_actions)
    step_count: Optional[chex.Array] = None          # optional per-agent step counter

class JobShopWrapper(JumanjiMarlWrapper):
    """
    Multi-agent wrapper around Jumanji's JobShop,
    splitting the single-agent env into one agent per machine.
    """
    def __init__(
        self,
        env: JobShop,
        add_global_state: bool = False,
    ):
        # Underlying environment spec is a property
        j_spec = env.observation_spec
        # Number of agents equals number of machines
        num_agents = env.num_machines
        # Inject into env so JumanjiMarlWrapper can see it
        env.num_agents = num_agents
        # Store locally as well
        self.num_agents = num_agents
        # Compute per-agent observation feature dimension by flattening ops_mask shape
        self.obs_feature_dim = math.prod(j_spec.ops_mask.shape)
        # Compute an upper-bound on episode length (makespan)
        self.max_episode_steps = (
            env.num_jobs * env.max_num_ops * env.max_op_duration
        )
        # Initialize the base wrapper (sets up self._env, self.num_agents, etc.)
        super().__init__(env, add_global_state)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Return a Spec for CustomJobShopObservation."""
        jumanji_spec = self._env.observation_spec
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
                maximum=jnp.repeat(self.max_episode_steps, self.num_agents),
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
        """Each agent (machine) picks a job id or no-op."""
        # Action values are in [0, num_jobs] (last index is no-op)
        return specs.DiscreteArray(
            num_values=self._env.num_jobs + 1,
            name="action",
        )

    def reset(self, seed: int) -> TimeStep:
        """Reset underlying env and wrap into a multi-agent timestep."""
        ts = self._env.reset(seed)
        return self.modify_timestep(ts)

    def step(self, actions: chex.Array) -> TimeStep:
        """Step underlying env with the first agent's action, then wrap."""
        sa_action = int(actions[0])
        ts = self._env.step(sa_action)
        return self.modify_timestep(ts)

    def modify_timestep(self, ts: TimeStep) -> TimeStep:
        """
        Convert a single-agent TimeStep into a multi-agent TimeStep.
        """
        # Flatten ops_mask into per-agent observation
        flat_obs = ts.observation.ops_mask.reshape(-1)
        agents_view = jnp.repeat(
            jnp.expand_dims(flat_obs, 0), self.num_agents, axis=0
        )
        # Simple per-agent step counter
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
