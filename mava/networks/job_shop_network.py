import chex
import haiku as hk
import jax.numpy as jnp
from mava.networks.base import FeedForwardActor, FeedForwardValueNet
from jumanji.environments.packing.job_shop.types import Observation as JumanjiObservation
from mava.wrappers.job_shop_wrapper import CustomJobShopObservation  # Correct import

class CustomJobShopEncoder(hk.Module):
    """Custom encoder for JobShop structured observation."""
    def __call__(self, observation: JumanjiObservation) -> chex.Array:
        ops_machines_emb = hk.Linear(64)(observation.ops_machine_ids.flatten())
        ops_durations_emb = hk.Linear(64)(observation.ops_durations.flatten())
        combined = jnp.concatenate([ops_machines_emb, ops_durations_emb], axis=-1)
        return hk.nets.MLP([128, 64])(combined)

class JobShopActor(FeedForwardActor):
    """Actor using a custom JobShop encoder."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torso = CustomJobShopEncoder()

    def __call__(self, observation: CustomJobShopObservation) -> chex.Array:
        jumanji_obs = observation.jumanji_obs
        embedding = self.torso(jumanji_obs)
        return self.action_head(embedding)

class JobShopCritic(FeedForwardValueNet):
    """Critic using a custom JobShop encoder."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torso = CustomJobShopEncoder()
        self.centralised_critic = False

    def __call__(self, observation: CustomJobShopObservation) -> chex.Array:
        jumanji_obs = observation.jumanji_obs
        embedding = self.torso(jumanji_obs)
        return self.value_head(embedding)