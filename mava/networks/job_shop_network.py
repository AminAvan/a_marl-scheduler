import chex
import haiku as hk
from mava.networks.base import FeedForwardActor, FeedForwardValueNet
from mava.types import Observation

class JobShopActor(FeedForwardActor):
    """
    Actor network for JobShop: consumes the per-agent flattened ops_mask view
    provided in Observation.agents_view.
    """
    def __call__(self, observation: Observation) -> chex.Array:
        # observation.agents_view: shape (num_agents, obs_dim)
        embedding = self.torso(observation.agents_view)
        return self.action_head(embedding)

class JobShopCritic(FeedForwardValueNet):
    """
    Critic network for JobShop: consumes the per-agent flattened ops_mask view
    provided in Observation.agents_view.
    """
    def __call__(self, observation: Observation) -> chex.Array:
        embedding = self.torso(observation.agents_view)
        return self.value_head(embedding)
