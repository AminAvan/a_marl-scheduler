### add amin
import jax
import haiku as hk
from mava.networks.base import Network
from jumanji.training.networks.job_shop.actor_critic import (
    make_actor_network_job_shop,
    make_critic_network_job_shop,
)

class JobShopNetwork(Network):
    """Wraps Jumanji’s JobShop GNN as a Mava Network."""

    def __init__(self, signature, key: jax.random.PRNGKey, config):
        super().__init__(signature, key)

        # Build the actor + critic Haiku fns using Jumanji’s factories
        actor_init, actor_apply = make_actor_network_job_shop(
            **config.job_shop.actor  # your hyperparams
        )
        critic_init, critic_apply = make_critic_network_job_shop(
            **config.job_shop.critic  # your hyperparams
        )

        # Store for Mava’s base.train and base.eval calls
        self.actor_init = actor_init
        self.actor_apply = actor_apply
        self.critic_init = critic_init
        self.critic_apply = critic_apply

    def __call__(self, params, rng, observation):
        # returns (logits, values) per Mava’s API
        logits = self.actor_apply(params.actor, rng, observation)
        values = self.critic_apply(params.critic, rng, observation)
        return logits, values