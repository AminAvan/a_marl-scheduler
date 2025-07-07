"""
job_shop_wrapper.py - Robust wrapper that ensures proper Mava observation format
"""
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import chex
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.types import Observation as JobShopObs
from jumanji.types import TimeStep
from jumanji import specs
from mava.types import Observation as MavaObservation
from mava.wrappers.jumanji import JumanjiMarlWrapper


class JobShopWrapper(JumanjiMarlWrapper):
    """
    Wrapper that converts JobShop observations to Mava format.
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

        # Calculate feature dimension for agents_view
        self.feature_dim = self._calculate_feature_dim()

        # Initialize base wrapper
        super().__init__(env, add_global_state)

    def _calculate_feature_dim(self) -> int:
        """Calculate the total feature dimension when flattening JobShop obs."""
        # Total features: 3 * num_jobs * max_num_ops + 2 * num_machines
        return 3 * self.num_jobs * self.max_num_ops + 2 * self.num_machines

    def _jobshop_obs_to_agents_view(self, obs: JobShopObs) -> jnp.ndarray:
        """Convert JobShop observation to a flat feature vector."""
        # Flatten and normalize all components
        features = []

        # Normalize to [0, 1] range for each component
        # Operations machine IDs (normalize by num_machines)
        ops_machine_normalized = obs.ops_machine_ids.astype(jnp.float32) / max(self.num_machines - 1, 1)
        features.append(ops_machine_normalized.flatten())

        # Operations durations (normalize by max_op_duration)
        ops_duration_normalized = obs.ops_durations.astype(jnp.float32) / max(self._env.max_op_duration, 1)
        features.append(ops_duration_normalized.flatten())

        # Operations mask (already 0/1)
        features.append(obs.ops_mask.flatten().astype(jnp.float32))

        # Machines job IDs (normalize by num_jobs)
        machines_job_normalized = obs.machines_job_ids.astype(jnp.float32) / max(self.num_jobs, 1)
        features.append(machines_job_normalized.flatten())

        # Machines remaining times (normalize by max possible time)
        max_time = self.num_jobs * self.max_num_ops * self._env.max_op_duration
        machines_time_normalized = obs.machines_remaining_times.astype(jnp.float32) / max(max_time, 1)
        features.append(machines_time_normalized.flatten())

        # Concatenate all features
        flat_features = jnp.concatenate(features)

        return flat_features

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """Convert JobShop timestep to Mava format."""
        obs = timestep.observation

        # Convert JobShop observation to flat features
        flat_features = self._jobshop_obs_to_agents_view(obs)

        # Create agents_view - each agent sees the same global state
        agents_view = jnp.tile(flat_features[None, :], (self.num_agents, 1))

        # Handle action mask
        action_mask = obs.action_mask

        # Create step count
        if hasattr(obs, 'step_count') and obs.step_count is not None:
            step_count = jnp.full((self.num_agents,), obs.step_count)
        else:
            # Use a default value
            step_count = jnp.zeros((self.num_agents,), dtype=jnp.int32)

        # Create Mava Observation
        mava_obs = MavaObservation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count
        )

        # Handle rewards and discounts
        reward = timestep.reward
        discount = timestep.discount

        # Ensure proper shape for multi-agent
        if jnp.ndim(reward) == 0 or reward.shape == ():
            reward = jnp.full((self.num_agents,), reward)
        if jnp.ndim(discount) == 0 or discount.shape == ():
            discount = jnp.full((self.num_agents,), discount)

        # Handle extras - ensure all required keys are present
        extras = dict(timestep.extras) if hasattr(timestep, "extras") and timestep.extras else {}

        # Ensure both episode_metrics and env_metrics exist (required by ff_ippo.py line 99)
        if "episode_metrics" not in extras:
            extras["episode_metrics"] = {}

        if "env_metrics" not in extras:
            extras["env_metrics"] = {}

        # Store original JobShop observation for debugging
        extras["jobshop_observation"] = obs

        # Create new timestep
        return TimeStep(
            step_type=timestep.step_type,
            reward=reward,
            discount=discount,
            observation=mava_obs,
            extras=extras
        )

    @property
    def observation_spec(self) -> specs.Spec:
        """Return the specification of the observation."""

        class MavaObservationSpec(specs.Spec):
            """Custom spec for Mava observations."""

            def __init__(self, num_agents: int, feature_dim: int, num_actions: int):
                self.num_agents = num_agents
                self.feature_dim = feature_dim
                self.num_actions = num_actions
                self._name = "MavaObservation"

            def generate_value(self) -> MavaObservation:
                """Generate a dummy observation."""
                return MavaObservation(
                    agents_view=jnp.zeros((self.num_agents, self.feature_dim), dtype=jnp.float32),
                    action_mask=jnp.ones((self.num_agents, self.num_actions), dtype=bool),
                    step_count=jnp.zeros((self.num_agents,), dtype=jnp.int32)
                )

            def validate(self, value: MavaObservation) -> MavaObservation:
                """Validate the observation."""
                assert isinstance(value, MavaObservation), f"Expected MavaObservation, got {type(value)}"
                assert hasattr(value, 'agents_view'), "Observation missing agents_view"
                assert hasattr(value, 'action_mask'), "Observation missing action_mask"
                assert hasattr(value, 'step_count'), "Observation missing step_count"
                return value

            def replace(self, **kwargs) -> specs.Spec:
                """Return a new spec with replaced values."""
                return self

            @property
            def name(self) -> str:
                """Return the name of the spec."""
                return self._name

        return MavaObservationSpec(
            num_agents=self.num_agents,
            feature_dim=self.feature_dim,
            num_actions=self.num_actions
        )