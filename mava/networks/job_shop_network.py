import jax
from mava.networks.base import FeedForwardActor, FeedForwardValueNet
from jumanji.training.networks.job_shop.actor_critic import JobShopTorso

class JobShopActor(FeedForwardActor):
    """Actor using Jumanji's JobShop GNN torso."""
    torso: JobShopTorso

class JobShopCritic(FeedForwardValueNet):
    """Critic using Jumanji's JobShop GNN torso."""
    torso: JobShopTorso
    centralised_critic: bool = False
