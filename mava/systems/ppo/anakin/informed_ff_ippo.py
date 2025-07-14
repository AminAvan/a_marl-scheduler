# The code of Agile Multi Agent Reinforcement Learning (A-MARL)
import copy
import time
from typing import Any, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


# === SPT Heuristic Function ===
def get_spt_action_jobshop(observation, num_machines, key):
    """
    SPT (Shortest Processing Time) heuristic for JobShop.
    Returns action array of shape (batch_size, num_machines) with job assignments.

    Based on Jumanji's JobShop observation structure from types.py:
    - ops_durations: (num_jobs, max_num_ops) - processing time of each operation
    - ops_mask: (num_jobs, max_num_ops) - True for operations to be scheduled
    - ops_machine_ids: (num_jobs, max_num_ops) - machine required for each op
    - action_mask: (num_machines, num_jobs + 1) - legal actions per machine
    """
    # Check if this is a JobShop observation by looking for the expected fields
    if not (hasattr(observation, 'ops_durations') or
            (isinstance(observation, dict) and 'ops_durations' in observation)):
        return None

    # Extract fields based on observation type
    if hasattr(observation, 'ops_durations'):
        # Direct attribute access (NamedTuple style)
        ops_durations = observation.ops_durations
        ops_mask = observation.ops_mask
        ops_machine_ids = observation.ops_machine_ids
        action_mask = observation.action_mask
    else:
        # Dictionary access
        ops_durations = observation['ops_durations']
        ops_mask = observation['ops_mask']
        ops_machine_ids = observation['ops_machine_ids']
        action_mask = observation['action_mask']

    # Handle batch dimension - Mava uses batched environments
    # If ops_durations has shape (batch, num_jobs, max_ops)
    if len(ops_durations.shape) == 3:
        batch_size = ops_durations.shape[0]
        # For simplicity, compute SPT for first environment and broadcast
        # This is reasonable since parallel environments often start from same state
        ops_durations_single = ops_durations[0]
        ops_mask_single = ops_mask[0]
        ops_machine_ids_single = ops_machine_ids[0]
        action_mask_single = action_mask[0] if action_mask is not None else None

        # Compute SPT for single environment
        actions_single = compute_spt_single(
            ops_durations_single,
            ops_mask_single,
            ops_machine_ids_single,
            action_mask_single,
            num_machines
        )

        # Broadcast to all environments in batch
        # Shape: (batch_size, num_machines)
        actions = jnp.broadcast_to(actions_single[None, :], (batch_size, num_machines))
        return actions
    else:
        # Single environment case
        actions = compute_spt_single(
            ops_durations,
            ops_mask,
            ops_machine_ids,
            action_mask,
            num_machines
        )
        # Add batch dimension for consistency
        return actions[None, :]  # Shape: (1, num_machines)


def compute_spt_single(ops_durations, ops_mask, ops_machine_ids, action_mask, num_machines):
    """Compute SPT for a single environment."""
    num_jobs = ops_durations.shape[0]
    actions = jnp.full(num_machines, num_jobs, dtype=jnp.int32)  # Initialize with no-op

    # For each machine, find the job with shortest processing time for its next operation
    for machine_id in range(num_machines):
        # Find the next operation for each job (first True in ops_mask)
        next_op_idx = jnp.argmax(ops_mask, axis=1)  # Shape: (num_jobs,)

        # Get the machine required for each job's next operation
        batch_indices = jnp.arange(num_jobs)
        next_op_machine = ops_machine_ids[batch_indices, next_op_idx]  # Shape: (num_jobs,)

        # Get the duration of each job's next operation
        next_op_duration = ops_durations[batch_indices, next_op_idx]  # Shape: (num_jobs,)

        # Check which jobs:
        # 1. Have operations remaining (any True in ops_mask)
        # 2. Need the current machine for their next operation
        job_has_ops = jnp.any(ops_mask, axis=1)  # Shape: (num_jobs,)
        job_needs_machine = (next_op_machine == machine_id)  # Shape: (num_jobs,)

        # Combine conditions
        can_schedule = job_has_ops & job_needs_machine  # Shape: (num_jobs,)

        # Apply action mask if available
        if action_mask is not None:
            legal_actions = action_mask[machine_id, :num_jobs]  # Shape: (num_jobs,)
            can_schedule = can_schedule & legal_actions

        # Find job with minimum duration among those that can be scheduled
        masked_durations = jnp.where(can_schedule, next_op_duration, jnp.inf)

        # Select the job with minimum duration
        best_job = jnp.argmin(masked_durations)

        # Only update if there's at least one schedulable job
        has_schedulable = jnp.any(can_schedule)
        actions = jnp.where(
            has_schedulable,
            actions.at[machine_id].set(best_job),
            actions  # Keep no-op if no job can be scheduled
        )

    return actions


def get_learner_fn(
        env: MarlEnv,
        apply_fns: Tuple[ActorApply, CriticApply],
        update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
        config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    # === Entropy threshold for switching ===
    # When entropy drops below this threshold, switch from SPT to policy
    # This threshold should be tuned based on your action space size
    # For discrete actions with N choices, max entropy = log(N)
    # We use a fraction of max entropy as threshold
    entropy_threshold_fraction = 0.8  # Use policy when (entropy < 80% of max) and use SPT when (entropy > 80% of max)

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.

        """

        def _env_step(
                carry: Tuple[LearnerState, Tuple], _: Any
        ) -> Tuple[Tuple[LearnerState, Tuple], Tuple[PPOTransition, Metrics]]:
            """Step the environment."""
            learner_state, (spt_usage_count, total_steps, entropy_sum) = carry
            params, opt_states, key, env_state, last_timestep, last_done = learner_state

            # Select action
            key, policy_key, spt_key, entropy_key = jax.random.split(key, 4)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            value = critic_apply_fn(params.critic_params, last_timestep.observation)

            # === ENTROPY-BASED EXPLORATION DECISION ===
            # Calculate current policy entropy
            policy_entropy = actor_policy.entropy(seed=entropy_key).mean()

            # Estimate max entropy based on action space
            # For discrete actions: max_entropy = log(num_actions)
            if hasattr(actor_policy, 'logits'):
                # For categorical distribution
                num_actions = actor_policy.logits.shape[-1]
                max_entropy = jnp.log(num_actions)
            else:
                # This is an approximation - actual max entropy depends on distribution
                max_entropy = jnp.log(6.0)  # log(num_jobs + 1)

            # Decide whether to use SPT based on entropy
            entropy_threshold = entropy_threshold_fraction * max_entropy
            use_spt = policy_entropy > entropy_threshold

            # Sample from policy
            policy_action = actor_policy.sample(seed=policy_key)

            # Try to get SPT action (only works for JobShop)
            # Check if JobShop observation is in extras (common in Mava)
            jobshop_obs = None
            if hasattr(last_timestep, 'extras') and 'jobshop_observation' in last_timestep.extras:
                jobshop_obs = last_timestep.extras['jobshop_observation']
            else:
                jobshop_obs = last_timestep.observation

            spt_action = get_spt_action_jobshop(
                jobshop_obs,
                getattr(config.env, 'num_machines', 4),  # Default to 4 if not specified
                spt_key
            )

            # Select action based on entropy threshold
            # We need to handle the case where spt_action might be None
            # First, check if we got a valid SPT action
            has_spt_action = spt_action is not None

            if has_spt_action:
                # Use JAX control flow to select between SPT and policy
                action = jax.lax.select(use_spt, spt_action, policy_action)
                # Update usage count conditionally
                spt_usage_count = jax.lax.select(
                    use_spt,
                    spt_usage_count + config.arch.num_envs,
                    spt_usage_count
                )
            else:
                # No SPT action available, use policy action
                action = policy_action

            # Always update statistics
            total_steps = total_steps + config.arch.num_envs
            entropy_sum = entropy_sum + policy_entropy * config.arch.num_envs

            log_prob = actor_policy.log_prob(action)
            # === END OF ENTROPY-BASED EXPLORATION ===

            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            transition = PPOTransition(
                last_done, action, value, timestep.reward, log_prob, last_timestep.observation
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, done)
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]

            # Return updated carry
            new_carry = (spt_usage_count, total_steps, entropy_sum)
            return (learner_state, new_carry), (transition, metrics)

        # Step environment for rollout length
        initial_carry = (0, 0, 0.0)  # spt_usage_count, total_steps, entropy_sum
        (learner_state, final_carry), (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, (learner_state, initial_carry), None, config.system.rollout_length
        )

        # Extract final statistics
        spt_usage_count, total_steps, entropy_sum = final_carry


        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done = learner_state
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                        actor_params: FrozenDict,
                        traj_batch: PPOTransition,
                        gae: chex.Array,
                        key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.system.clip_eps,
                                1.0 + config.system.clip_eps,
                            )
                            * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_actor_loss = actor_loss - config.system.ent_coef * entropy
                    return total_actor_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                        critic_params: FrozenDict,
                        traj_batch: PPOTransition,
                        targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_value_loss = config.system.vf_coef * value_loss
                    return total_value_loss, value_loss

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params, traj_batch, advantages, entropy_key
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (_, entropy) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info

                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # Shuffle data and create minibatches
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, last_done)
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
        env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso)

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation with obs of all agents.
    obs = env.observation_spec.generate_value()
    init_x = tree.map(lambda x: x[jnp.newaxis, ...], obs)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, config.system.num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, step_keys, dones = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps, dones)

    # Inside learner_setup or wherever init_x is created
    print(f"init_x type: {type(init_x)}")  #
    print(f"Has agents_view: {hasattr(init_x, 'agents_view')}")  #
    print(f"init_x attributes: {dir(init_x)}")  #

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "informed_ff_ippo"  # Changed to distinguish from standard ff_ippo
    config = copy.deepcopy(_config)
    # print("Full config:")  #
    # print(OmegaConf.to_yaml(config))  #
    # print("Pre-torso config:")  #
    # print(OmegaConf.to_yaml(config.network.actor_network.pre_torso))  #
    # print("================================")  #

    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
            config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    assert (
            config.arch.num_envs % config.system.num_minibatches == 0
    ), "Number of envs must be divisibile by number of minibatches."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
            n_devices
            * config.system.num_updates_per_eval
            * config.system.rollout_length
            * config.system.update_batch_size
            * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # === MINIMAL ADDITION: Log that we're using entropy-based exploration ===
    print(f"\n{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Using ENTROPY-BASED SPT informed exploration{Style.RESET_ALL}")
    print(f"{Fore.CYAN}SPT is used when (policy-entropy > entropy) threshold{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Policy takes over when (policy-entropy < entropy) threshold{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}\n")

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_ippo.yaml", ## A-MARL (informed_ff_ippo.py) has the same configuration as MARL (ff_ippo.py)
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    # # right after you get `cfg` from Hydra
    # print("=== Effective network config ===")   #
    # print(OmegaConf.to_yaml(cfg.network))   #
    # print("================================")   #

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()