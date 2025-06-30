from jumanji.wrappers import Wrapper
import jax.numpy as jnp

class CustomAgentIDWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents

    def modify_timestep(self, timestep):
        # Customize how agent IDs are added to the timestep for JobShop
        observation = timestep.observation
        agent_ids = jnp.arange(self.num_agents)
        # Example: Modify observation to include agent IDs if required
        # Adjust this based on your system's needs
        return timestep