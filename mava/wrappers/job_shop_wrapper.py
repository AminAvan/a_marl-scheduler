import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.environments.packing.job_shop import JobShopState
from jumanji.types import TimeStep
from typing import Tuple, Dict, Any
from mava.types import Observation


def aggregate_rewards(reward: jnp.ndarray, num_agents: int, num_envs: int) -> jnp.ndarray:
    """Aggregate environment reward across agents."""
    if reward.ndim == 0:  # Non-batched case
        return jnp.full((1, num_agents), reward / num_agents)
    return jnp.full((num_envs, num_agents), reward / num_agents)


class JobShopWrapper(Environment):
    """Wrapper to adapt Jumanji JobShop environment for Mava multi-agent RL."""

    def __init__(self, env: Environment, num_agents: int):
        """
        Args:
            env: Jumanji JobShop environment instance.
            num_agents: Number of agents (machines).
        """
        super().__init__()
        self._env = env
        self.num_agents = num_agents
        self.num_jobs = env.num_jobs
        self.max_num_ops = env.max_num_ops
        self.action_dim = self.num_jobs * self.max_num_ops + 1  # Operations + no-op

    def reset(self, key: jax.random.PRNGKey) -> Tuple[JobShopState, TimeStep]:
        """Reset the environment and return a Mava-compatible timestep."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep, state)
        return state, timestepEXPORT

    def step(self, state: JobShopState, action: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[JobShopState, TimeStep]:
        """Step the environment with multi-agent actions."""
        state, timestep = self._env.step(state, action)
        timestep = self.modify_timestep(timestep, state)
        return state, timestep

    def modify_timestep(self, timestep: TimeStep, state: JobShopState) -> TimeStep:
        """
        Modify timestep for Mava compatibility.

        Args:
            timestep: Original timestep from Jumanji.
            state: Current environment state.

        Returns:
            Modified timestep with per-agent observations and rewards.
        """
        is_batched = state.ops_mask.ndim == 3
        num_envs = state.ops_mask.shape[0] if is_batched else 1
        obs = timestep.observation

        # Calculate metrics
        makespan = jnp.max(
            state.scheduled_times + obs.ops_durations,
            axis=(1, 2) if is_batched else (0, 1),
            where=obs.ops_mask,
            initial=0
        )
        num_ops = jnp.sum(obs.ops_mask, axis=(1, 2) if is_batched else (0, 1))
        is_terminal = ~jnp.any(obs.ops_mask, axis=(1, 2) if is_batched else (0, 1))
        extras = {"env_metrics": {"makespan": makespan, "num_ops": num_ops}}

        # Aggregate rewards
        reward = aggregate_rewards(timestep.reward, self.num_agents, num_envs)

        # Construct action mask
        action_mask = jnp.zeros((num_envs, self.num_agents, self.action_dim), dtype=bool)
        for machine_id in range(self.num_agents):
            machine_ops = (obs.ops_machine_ids == machine_id) & obs.ops_mask
            op_indices = jnp.where(
                machine_ops.reshape(num_envs, -1) if is_batched else machine_ops.reshape(-1),
                size=self.num_jobs * self.max_num_ops,
                fill_value=-1
            )[1] if is_batched else \
                jnp.where(machine_ops.reshape(-1), size=self.num_jobs * self.max_num_ops, fill_value=-1)[0]

            if is_batched:
                for env in range(num_envs):
                    valid_ops = op_indices[env]
                    valid_ops = valid_ops[valid_ops != -1]
                    action_mask = action_mask.at[env, machine_id, valid_ops].set(True)
            else:
                valid_ops = op_indices[op_indices != -1]
                action_mask = action_mask.at[0, machine_id, valid_ops].set(True)
        action_mask = action_mask.at[:, :, -1].set(True)  # No-op is always valid

        # Construct agents_view
        feature_dim = self.num_jobs * self.max_num_ops * 3
        if is_batched:
            agents_view = jnp.concatenate([
                obs.ops_durations.reshape(num_envs, -1),
                obs.ops_mask.reshape(num_envs, -1),
                obs.ops_machine_ids.reshape(num_envs, -1),
            ], axis=-1)
            agents_view = jnp.repeat(agents_view[:, None, :], self.num_agents, axis=1)
        else:
            agents_view = jnp.concatenate([
                obs.ops_durations.reshape(-1),
                obs.ops_mask.reshape(-1),
                obs.ops_machine_ids.reshape(-1),
            ], axis=-1)
            agents_view = jnp.repeat(agents_view[None, :], self.num_agents, axis=0)
            agents_view = agents_view[None, ...]

        # Step count
        step_count = state.step_count
        step_count = step_count[:, None] if is_batched else step_count[None, None]
        step_count = jnp.repeat(step_count, self.num_agents, axis=1)

        # Create Mava Observation
        observation = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        return timestep.replace(observation=observation, reward=reward, extras=extras)

    def observation_spec(self):
        """Return the observation spec for Mava."""
        from dm_env import specs
        feature_dim = self.num_jobs * self.max_num_ops * 3
        return specs.Array(
            shape=(self.num_agents, feature_dim),
            dtype=jnp.float32,
            name="observation"
        )

    def action_spec(self):
        """Return the action spec for Mava."""
        from dm_env import specs
        return specs.DiscreteArray(
            num_values=self.action_dim,
            name="action"
        )