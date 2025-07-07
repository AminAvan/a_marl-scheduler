"""
Minimal job_shop_wrapper.py

A simpler Mava multi-agent wrapper for Jumanji's JobShop environment.
"""
from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import chex
from jumanji.environments.packing.job_shop import JobShop
from jumanji.types import TimeStep
from mava.types import Observation
from mava.wrappers.jumanji import JumanjiMarlWrapper


class JobShopWrapper(JumanjiMarlWrapper):
    """
    Multi-agent wrapper around Jumanji's JobShop.
    """
    def __init__(self, env: JobShop, add_global_state: bool = False):
        # Set required attributes before calling super().__init__
        env.num_agents = env.num_machines
        env.time_limit = env.num_jobs * env.max_num_ops * env.max_op_duration

        # Store environment parameters
        self.num_jobs = env.num_jobs
        self.max_num_ops = env.max_num_ops
        self.num_machines = env.num_machines
        self.num_actions = env.num_jobs + 1

        # Initialize base wrapper
        super().__init__(env, add_global_state)

        # Initialize encoder
        self._init_encoder()

    def _init_encoder(self):
        """Initialize the JobShop encoder."""
        from mava.networks.job_shop_network import JobShopEncoder
        self.encoder = JobShopEncoder(
            num_jobs=self.num_jobs,
            max_num_ops=self.max_num_ops,
            num_machines=self.num_machines,
            embedding_dim=64
        )
        # We'll initialize params lazily on first use
        self._encoder_params = None

    def _get_encoder_params(self, dummy_obs):
        """Get or initialize encoder parameters."""
        if self._encoder_params is None:
            self._encoder_params = self.encoder.init(
                jax.random.PRNGKey(0), dummy_obs
            )
        return self._encoder_params

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """Convert JobShop observation to multi-agent format."""
        obs = timestep.observation

        # Get encoder params
        encoder_params = self._get_encoder_params(obs)

        # Encode observation to get agents_view
        agents_view = self.encoder.apply(encoder_params, obs)

        # Ensure correct shape for agents_view
        if agents_view.shape[0] == 1:
            agents_view = agents_view[0]  # Remove batch dim

        # Repeat for all agents (they all see the same encoded state)
        if agents_view.ndim == 1:
            agents_view = jnp.tile(agents_view[None, :], (self.num_agents, 1))

        # Handle action mask
        # JobShop's action_mask should already be (num_machines, num_jobs+1)
        action_mask = obs.action_mask

        # Ensure action mask has correct shape
        if action_mask.shape != (self.num_agents, self.num_actions):
            if action_mask.ndim == 1:
                # Reshape if needed
                action_mask = action_mask.reshape(self.num_agents, -1)
            elif action_mask.shape[0] != self.num_agents:
                # This shouldn't happen with JobShop, but handle it
                action_mask = jnp.tile(action_mask[:1], (self.num_agents, 1))

        # Create step count
        step_count = jnp.arange(self.num_agents)

        # Build Mava Observation
        mava_obs = Observation(
            agents_view=agents_view.astype(jnp.float32),
            action_mask=action_mask,
            step_count=step_count
        )

        # Handle rewards and discounts
        reward = timestep.reward
        discount = timestep.discount

        # Ensure they are arrays with correct shape
        if jnp.isscalar(reward) or reward.shape == ():
            reward = jnp.full((self.num_agents,), reward)
        if jnp.isscalar(discount) or discount.shape == ():
            discount = jnp.full((self.num_agents,), discount)

        # Return modified timestep
        return timestep.replace(
            observation=mava_obs,
            reward=reward,
            discount=discount
        )