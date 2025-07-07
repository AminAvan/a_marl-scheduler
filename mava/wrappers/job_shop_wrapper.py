"""
job_shop_wrapper.py

A Mava multi-agent wrapper for Jumanji's JobShop environment.
"""
import math
from functools import cached_property
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import chex
from jumanji import specs
from jumanji.environments.packing.job_shop import JobShop
from jumanji.types import TimeStep
import logging

from mava.types import Observation  # Mava Observation NamedTuple
from mava.wrappers.jumanji import JumanjiMarlWrapper

logging.basicConfig(level=logging.INFO)


class JobShopWrapper(JumanjiMarlWrapper):
    """
    Multi-agent wrapper around Jumanji's JobShop,
    creating one agent per machine.
    """
    def __init__(
        self,
        env: JobShop,
        add_global_state: bool = False,
    ):
        # Store original environment
        self._original_env = env

        # Determine number of agents (machines) and episode length
        num_agents = env.num_machines
        max_episode_steps = env.num_jobs * env.max_num_ops * env.max_op_duration

        # Inject attributes expected by base wrapper
        env.num_agents = num_agents
        env.time_limit = max_episode_steps

        # Store environment parameters
        self.num_jobs = env.num_jobs
        self.max_num_ops = env.max_num_ops
        self.num_machines = env.num_machines

        # Define number of actions per agent (jobs + no-op)
        self.num_actions = env.num_jobs + 1

        # Initialize base wrapper (sets self._env, self.num_agents, self.time_limit)
        super().__init__(env, add_global_state)

        # Create the encoder once during initialization
        from mava.networks.job_shop_network import JobShopEncoder
        self.encoder = JobShopEncoder(
            num_jobs=self.num_jobs,
            max_num_ops=self.max_num_ops,
            num_machines=self.num_machines,
            embedding_dim=64
        )
        self._encoder_params = None

        print(f"JobShop environment created with:")
        print(f"  num_jobs: {self.num_jobs}")
        print(f"  num_machines: {self.num_machines}")
        print(f"  max_num_ops: {self.max_num_ops}")
        print(f"  num_agents: {self.num_agents}")
        print(f"  num_actions: {self.num_actions}")

    def _init_encoder_params(self, dummy_obs):
        """Initialize encoder parameters if not already done."""
        if self._encoder_params is None:
            self._encoder_params = self.encoder.init(
                jax.random.PRNGKey(0), dummy_obs
            )

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, TimeStep]:
        """
        Reset the environment and ensure the initial timestep is formatted
        as a Mava multi-agent timestep.
        """
        state, timestep = self._env.reset(key)

        # Initialize encoder params with the first observation
        self._init_encoder_params(timestep.observation)

        modified_timestep = self.modify_timestep(timestep, state)
        return state, modified_timestep

    def step(self, state: chex.ArrayTree, actions: chex.Array) -> Tuple[chex.ArrayTree, TimeStep]:
        """Step the environment with multi-agent actions."""
        # Convert multi-agent actions to single-agent format for JobShop
        # JobShop expects a single action, not per-agent actions
        if actions.ndim == 2:  # (batch, num_agents)
            # For now, use the first agent's action
            # In a real implementation, you might want a different strategy
            single_actions = actions[:, 0]
        elif actions.ndim == 1 and len(actions) == self.num_agents:
            # Single environment, multiple agents
            single_actions = actions[0]
        else:
            single_actions = actions

        # Step the environment
        state, timestep = self._env.step(state, single_actions)

        # Convert to multi-agent format
        modified_timestep = self.modify_timestep(timestep, state)
        return state, modified_timestep

    def modify_timestep(self, timestep: TimeStep, state: Optional[chex.ArrayTree] = None) -> TimeStep:
        """
        Convert a single-agent Jumanji timestep into a multi-agent timestep,
        wrapping the raw Jumanji observation into Mava's Observation.
        """
        obs = timestep.observation

        # Initialize encoder if needed
        self._init_encoder_params(obs)

        # Apply the encoder to get agents_view
        agents_view = self.encoder.apply(self._encoder_params, obs)

        # Handle batch dimension
        if agents_view.ndim == 2 and agents_view.shape[0] == 1:
            # Remove unnecessary batch dimension
            agents_view = agents_view[0]  # Now shape: (feature_dim,)

        # For multi-agent, we need (num_agents, feature_dim)
        # Repeat the same observation for all agents
        if agents_view.ndim == 1:
            agents_view = jnp.tile(agents_view[None, :], (self.num_agents, 1))
        elif agents_view.ndim == 2 and agents_view.shape[0] != self.num_agents:
            # If we have a batch dimension but wrong number of agents
            agents_view = jnp.tile(agents_view[:1], (self.num_agents, 1))

        # Handle action mask
        action_mask = obs.action_mask

        # JobShop action mask is typically (num_machines, num_jobs+1)
        # which matches our expected (num_agents, num_actions) format
        if action_mask.shape != (self.num_agents, self.num_actions):
            # If shape doesn't match, try to reshape or tile
            if action_mask.ndim == 1:
                # Single flat mask - reshape and tile
                action_mask = jnp.tile(action_mask[None, :], (self.num_agents, 1))
            elif action_mask.shape[0] != self.num_agents:
                # Wrong number of agents - tile the first one
                action_mask = jnp.tile(action_mask[:1], (self.num_agents, 1))

        # Step count
        step_count = jnp.full((self.num_agents,), timestep.step_count if hasattr(timestep, 'step_count') else 0)

        # Build Mava Observation
        observation = Observation(
            agents_view=agents_view.astype(jnp.float32),
            action_mask=action_mask,
            step_count=step_count,
        )

        # Replicate rewards and discounts for all agents
        reward = timestep.reward
        discount = timestep.discount

        # Ensure rewards and discounts are arrays and have the right shape
        if jnp.isscalar(reward) or reward.shape == ():
            reward = jnp.full((self.num_agents,), reward)
        elif reward.shape != (self.num_agents,):
            reward = jnp.tile(reward, self.num_agents)[:self.num_agents]

        if jnp.isscalar(discount) or discount.shape == ():
            discount = jnp.full((self.num_agents,), discount)
        elif discount.shape != (self.num_agents,):
            discount = jnp.tile(discount, self.num_agents)[:self.num_agents]

        # Preserve existing extras
        extras: Dict[str, chex.Array] = timestep.extras if hasattr(timestep, "extras") else {}

        # Return new TimeStep
        return TimeStep(
            observation=observation,
            reward=reward,
            discount=discount,
            step_type=timestep.step_type,
            extras=extras,
        )

    @cached_property
    def observation_spec(self):
        """Override observation spec to match Mava's Observation structure."""
        # Create a custom spec that generates Mava Observations
        class MavaObservationSpec(specs.Spec):
            def __init__(self, num_agents, feature_dim, num_actions):
                self.num_agents = num_agents
                self.feature_dim = feature_dim
                self.num_actions = num_actions
                self._name = "MavaObservation"

            def generate_value(self) -> Observation:
                return Observation(
                    agents_view=jnp.zeros((self.num_agents, self.feature_dim), dtype=jnp.float32),
                    action_mask=jnp.ones((self.num_agents, self.num_actions), dtype=bool),
                    step_count=jnp.zeros((self.num_agents,), dtype=jnp.int32),
                )

            def validate(self, value: Observation) -> Observation:
                # Basic validation
                assert hasattr(value, 'agents_view')
                assert hasattr(value, 'action_mask')
                assert hasattr(value, 'step_count')
                return value

            def replace(self, **kwargs):
                # For compatibility
                return self

            @property
            def name(self) -> str:
                return self._name

        return MavaObservationSpec(
            num_agents=self.num_agents,
            feature_dim=128,  # From encoder output
            num_actions=self.num_actions
        )

    @cached_property
    def action_spec(self):
        """Override action spec for multi-agent."""
        # Return DiscreteArray spec for multi-agent actions
        return specs.DiscreteArray(
            num_values=self.num_actions,
            shape=(self.num_agents,),
            dtype=jnp.int32
        )