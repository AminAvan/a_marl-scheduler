import jax
import chex
from mava.networks.base import FeedForwardActor, FeedForwardValueNet
from jumanji.training.networks.job_shop.actor_critic import JobShopTorso
from mava.types import Observation

class JobShopActor(FeedForwardActor):
    """Actor using Jumanji's JobShop GNN torso."""
    torso: JobShopTorso

    def __call__(self, observation: Observation) -> chex.Array:
        # Convert Mava Observation to a structure compatible with JobShopTorso
        jumanji_obs = self._to_jumanji_obs(observation)
        obs_embedding = self.torso(jumanji_obs)
        return self.action_head(obs_embedding)

    def _to_jumanji_obs(self, obs: Observation):
        # Convert Mava Observation to Jumanji Observation
        from jumanji.environments.packing.job_shop.types import Observation as JumanjiObs
        feat_per_agent = obs.agents_view.shape[-1] // 2  # Split features between machines and jobs
        return JumanjiObs(
            machines=obs.agents_view[:, :feat_per_agent],
            jobs=obs.agents_view[:, feat_per_agent:],
            action_mask=obs.action_mask,
            step_count=obs.step_count
        )

class JobShopCritic(FeedForwardValueNet):
    """Critic using Jumanji's JobShop GNN torso."""
    torso: JobShopTorso
    centralised_critic: bool = False

    def __call__(self, observation: Observation) -> chex.Array:
        # Convert Mava Observation to a structure compatible with JobShopTorso
        jumanji_obs = self._to_jumanji_obs(observation)
        obs_embedding = self.torso(jumanji_obs)
        return self.value_head(obs_embedding)

    def _to_jumanji_obs(self, obs: Observation):
        # Convert Mava Observation to Jumanji Observation (same as in JobShopActor)
        from jumanji.environments.packing.job_shop.types import Observation as JumanjiObs
        feat_per_agent = obs.agents_view.shape[-1] // 2  # Split features between machines and jobs
        return JumanjiObs(
            machines=obs.agents_view[:, :feat_per_agent],
            jobs=obs.agents_view[:, feat_per_agent:],
            action_mask=obs.action_mask,
            step_count=obs.step_count
        )