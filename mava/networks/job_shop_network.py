import jax
import jax.numpy as jnp
import haiku as hk
from typing import Sequence, Optional
from mava.networks.base import BaseNetwork
from mava.networks.heads import CategoricalActionHead
from mava.networks.torsos import MLPTorso, TransformerTorso


class JobShopEncoder(hk.Module):
    """Custom encoder for JobShop observations."""

    def __init__(
            self,
            output_dim: int = 128,
            num_layers: int = 2,
            layer_sizes: Optional[Sequence[int]] = None,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes or [256, 256]

    def __call__(self, observation):
        """Encode JobShop observation into a fixed-size representation."""
        # Instead of using agents_view, we'll process the raw JobShop observation
        # Check if we have agents_view (from wrapper) or raw JobShop observation
        if hasattr(observation, 'agents_view'):
            # If wrapped, use agents_view
            x = observation.agents_view
        else:
            # Otherwise, manually concatenate JobShop features
            features = []

            # Flatten and concatenate all observation components
            if hasattr(observation, 'ops_machine_ids'):
                features.append(observation.ops_machine_ids.reshape(*observation.ops_machine_ids.shape[:-2], -1))
            if hasattr(observation, 'ops_durations'):
                features.append(observation.ops_durations.reshape(*observation.ops_durations.shape[:-2], -1))
            if hasattr(observation, 'ops_mask'):
                features.append(observation.ops_mask.reshape(*observation.ops_mask.shape[:-2], -1))
            if hasattr(observation, 'machines_job_ids'):
                features.append(observation.machines_job_ids)
            if hasattr(observation, 'machines_remaining_times'):
                features.append(observation.machines_remaining_times)

            if features:
                x = jnp.concatenate(features, axis=-1)
            else:
                raise ValueError("No recognized JobShop observation attributes found")

        # Pass through MLP layers
        for i, layer_size in enumerate(self.layer_sizes):
            x = hk.Linear(layer_size)(x)
            x = jax.nn.relu(x)

        # Final projection to output dimension
        x = hk.Linear(self.output_dim)(x)

        return x


class JobShopActor(BaseNetwork):
    """Actor network for JobShop environment."""

    def __init__(
            self,
            num_actions: int,
            pre_torso: Optional[hk.Module] = None,
            post_torso: Optional[hk.Module] = None,
            encoder: Optional[hk.Module] = None,
    ):
        self.num_actions = num_actions
        self.encoder = encoder or JobShopEncoder()
        self.pre_torso = pre_torso
        self.post_torso = post_torso
        self.action_head = CategoricalActionHead(num_actions)

    def __call__(self, observation):
        # Encode observation
        x = self.encoder(observation)

        # Apply pre-torso if provided
        if self.pre_torso:
            x = self.pre_torso(x)

        # Apply post-torso if provided
        if self.post_torso:
            x = self.post_torso(x)

        # Get action distribution
        return self.action_head(x, observation.action_mask)


class JobShopCritic(BaseNetwork):
    """Critic network for JobShop environment."""

    def __init__(
            self,
            pre_torso: Optional[hk.Module] = None,
            post_torso: Optional[hk.Module] = None,
            encoder: Optional[hk.Module] = None,
    ):
        self.encoder = encoder or JobShopEncoder()
        self.pre_torso = pre_torso
        self.post_torso = post_torso

    def __call__(self, observation):
        # Encode observation
        x = self.encoder(observation)

        # Apply pre-torso if provided
        if self.pre_torso:
            x = self.pre_torso(x)

        # Apply post-torso if provided
        if self.post_torso:
            x = self.post_torso(x)

        # Output value estimate
        value = hk.Linear(1)(x)
        return jnp.squeeze(value, axis=-1)