from jumanji.env import Environment
from jumanji.environments.routing.job_shop.types import JobShop
import math
from mava.wrappers.jumanji import JumanjiMarlWrapper

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
        # Determine number of agents (machines) and episode length
        num_agents = env.num_machines
        max_episode_steps = env.num_jobs * env.max_num_ops * env.max_op_duration
        # Inject into env for base wrapper
        env.num_agents = num_agents
        env.time_limit = max_episode_steps
        # Compute per-agent feature dimension by flattening ops_mask
        j_spec = env.observation_spec
        self.obs_feature_dim = math.prod(j_spec.ops_mask.shape)
        # Define number of actions per agent (jobs + no-op)
        self.num_actions = env.num_jobs + 1
        # Initialize base wrapper (sets self._env, self.num_agents, self.time_limit)
        super().__init__(env, add_global_state)

    def reset(self, key):
        timestep = self._env.reset(key)
        return self.modify_timestep(timestep)

    def modify_timestep(self, timestep):
        """Convert Jumanji timestep to Mava-compatible timestep."""
        if timestep.step_type.first():
            # Initial observation: return just the per-agent obs
            obs = {
                f"agent_{i}": timestep.observation.obs[i].flatten()
                for i in range(self.num_agents)
            }
            return types.TimeStep(
                observation=obs,
                reward=None,
                discount=None,
                step_type=timestep.step_type,
            )
        else:
            # Subsequent timesteps: include reward, discount, and per-agent obs
            obs = {
                f"agent_{i}": timestep.observation.obs[i].flatten()
                for i in range(self.num_agents)
            }
            reward = {f"agent_{i}": timestep.reward[i] for i in range(self.num_agents)}
            discount = {
                f"agent_{i}": timestep.discount for i in range(self.num_agents)
            }
            return types.TimeStep(
                observation=obs,
                reward=reward,
                discount=discount,
                step_type=timestep.step_type,
            )

    def action_spec(self):
        import tree
        from jumanji import specs

        j_spec = self._env.action_spec
        per_agent_action_spec = specs.DiscreteArray(
            num_values=self.num_actions, dtype=j_spec.dtype
        )
        return tree.map_structure(
            lambda _: per_agent_action_spec,
            [0] * self.num_agents,
        )

    def observation_spec(self):
        import tree
        from jumanji import specs

        return tree.map_structure(
            lambda _: specs.Array(
                shape=(self.obs_feature_dim,), dtype="float32", name="observation"
            ),
            [0] * self.num_agents,
        )