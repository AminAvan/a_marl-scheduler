# mava/wrappers/job_shop_wrapper.py

# ─── 0) Registry hack ──────────────────────────────────────────────────────────
from jumanji.registration import _REGISTRY, EnvSpec
from typing      import Optional

# Import the original single-agent JobShop base
from jumanji.environments.packing.job_shop import JobShop as _BaseJobShop
from jumanji.types import StepType, TimeStep

# 0.a) Subclass to inject time_limit logic
class TimeLimitedJobShop(_BaseJobShop):
    def __init__(
        self,
        *,
        generator,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
        time_limit: Optional[int] = None,   # new param
    ):
        super().__init__(
            generator=generator,
            num_jobs=num_jobs,
            num_machines=num_machines,
            max_num_ops=max_num_ops,
            max_op_duration=max_op_duration,
        )
        self._time_limit = time_limit

    def step(self, state, action):
        next_state, ts = super().step(state, action)
        # enforce horizon if requested
        if self._time_limit is not None:
            # step_count is in ts.observation.step_count
            if ts.observation.step_count >= self._time_limit:
                ts = ts.replace(step_type=StepType.LAST)
        return next_state, ts

# 0.b) Mutate the existing EnvSpec so Jumanji.make("JobShop-v0", time_limit=…) uses our subclass
_spec: EnvSpec = _REGISTRY["JobShop-v0"]
_spec.entry_point = TimeLimitedJobShop
# _spec.kwargs is left alone so generator, plus Hydra’s time_limit, flow through

# ─── 1) Usual imports for your MARL wrapper ────────────────────────────────────
from abc            import ABC, abstractmethod
from functools      import cached_property
from typing         import Any, Dict, Tuple, Union

import chex
import jax.numpy as jnp
from jumanji         import specs
from jumanji.env     import Environment
from jumanji.types   import TimeStep
from jumanji.wrappers import Wrapper

from mava.types      import Observation, ObservationGlobalState, State

# ─── 2) Reward aggregator (unchanged) ─────────────────────────────────────────
def aggregate_rewards(reward: chex.Array, num_agents: int) -> chex.Array:
    team_reward = jnp.sum(reward)
    return jnp.repeat(team_reward, num_agents)

# ─── 3) Base multi-agent wrapper (with fallback num_agents/time_limit) ────────
class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool):
        self.add_global_state = add_global_state
        super().__init__(env)
        # fallback to `generator.num_machines` if no .num_agents
        if hasattr(self._env, "num_agents"):
            self.num_agents = self._env.num_agents
        else:
            self.num_agents = self._env.generator.num_machines
        self.time_limit = getattr(self._env, "_time_limit", None)

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        gs = jnp.concatenate(obs.agents_view, axis=0)
        return jnp.tile(gs, (self.num_agents, 1))

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, ts = self._env.reset(key)
        ts = self.modify_timestep(ts)
        if self.add_global_state:
            global_state = self.get_global_state(ts.observation)
            obs = ObservationGlobalState(
                global_state=global_state,
                agents_view=ts.observation.agents_view,
                action_mask=ts.observation.action_mask,
                step_count=ts.observation.step_count,
            )
            return state, ts.replace(observation=obs)
        return state, ts

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, ts = self._env.step(state, action)
        ts = self.modify_timestep(ts)
        if self.add_global_state:
            global_state = self.get_global_state(ts.observation)
            obs = ObservationGlobalState(
                global_state=global_state,
                agents_view=ts.observation.agents_view,
                action_mask=ts.observation.action_mask,
                step_count=ts.observation.step_count,
            )
            return state, ts.replace(observation=obs)
        return state, ts

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        # build a BoundedArray only if we have a horizon
        if self.time_limit is not None:
            step_count_spec = specs.BoundedArray(
                (self.num_agents,),
                int,
                jnp.zeros(self.num_agents, int),
                jnp.repeat(self.time_limit, self.num_agents),
                "step_count",
            )
        else:
            step_count_spec = specs.Array(
                (self.num_agents,),
                int,
                "step_count",
            )

        obs_spec = self._env.observation_spec
        data = {
            "agents_view": obs_spec.agents_view,
            "action_mask": obs_spec.action_mask,
            "step_count": step_count_spec,
        }
        if self.add_global_state:
            num_feat = obs_spec.agents_view.shape[-1]
            global_state = specs.Array(
                (self.num_agents, self.num_agents * num_feat),
                obs_spec.agents_view.dtype,
                "global_state",
            )
            data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **data)
        return specs.Spec(Observation, "ObservationSpec", **data)

    @cached_property
    def action_dim(self) -> int:
        return int(self._env.action_spec.num_values[0])

# ─── 4) Your JobShop wrapper ───────────────────────────────────────────────────
class JobShopWrapper(JumanjiMarlWrapper):
    def __init__(self, env: _BaseJobShop, add_global_state: bool = False):
        super().__init__(env, add_global_state)

    def modify_timestep(self, ts: TimeStep) -> TimeStep[Observation]:
        s = ts.observation
        # flatten and tile for “one agent per machine”
        flat = jnp.concatenate([
            s.ops_machine_ids.ravel().astype(float),
            s.ops_durations.ravel().astype(float),
            s.ops_mask.astype(float).ravel(),
            s.machines_job_ids.ravel().astype(float),
            s.machines_remaining_times.ravel().astype(float),
            s.scheduled_times.ravel().astype(float),
        ], axis=0)
        agents_view = jnp.tile(flat[None, :], (self.num_agents, 1))
        action_mask = s.action_mask
        step_count = jnp.repeat(s.step_count, self.num_agents)
        obs = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )
        reward   = jnp.repeat(ts.reward,   self.num_agents)
        discount = jnp.repeat(ts.discount, self.num_agents)
        extras   = {"env_metrics": {}}
        return ts.replace(
            observation=obs,
            reward=reward,
            discount=discount,
            extras=extras,
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        # exactly the same logic you had—building specs from env_spec
        env_spec = self._env.observation_spec
        num_jobs, max_ops = env_spec.ops_machine_ids.shape
        nm = self.num_agents
        fd = num_jobs * max_ops * 3 + nm * 2  # as before
        return super().observation_spec  # or rebuild if you prefer

# ─── End of file ───────────────────────────────────────────────────────────────
