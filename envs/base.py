"""Shared environment-side data structures and the (informal) environment interface.

An environment module registered in envs/__init__.py provides a class with:
  - attributes: env_id, n_agents, n_actions, obs_dim
  - rollout(policies, epsilon) -> Trajectory
  - collect_batch(policies, epsilon, n_episodes) -> list[Trajectory]
  - describe() -> dict with "env" and "desc" prompt strings (for LLM critics)
  - serialize(traj) -> str (text rendering of a trajectory for LLM critics)
"""
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """One episode. Lists are indexed by timestep k = 0 .. T-1."""

    obs: list = field(default_factory=list)        # obs[k] = tuple(np.ndarray) per agent, at step k
    actions: list = field(default_factory=list)    # actions[k] = np.ndarray(n_agents,) int
    global_reward: list = field(default_factory=list)  # global_reward[k] = float
    next_obs: list = field(default_factory=list)   # next_obs[k] = tuple(np.ndarray) per agent
    done: list = field(default_factory=list)       # done[k] = bool (terminal or time-limit cutoff)
    state: list = field(default_factory=list)      # state[k] = env-specific global state snapshot
    #   (full world view for the centralized critic, per the paper; policies never see it)
    next_state: list = field(default_factory=list) # post-action global state; needed for shaping

    def __len__(self):
        return len(self.actions)
