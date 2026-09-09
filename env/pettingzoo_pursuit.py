"""pymarl wrapper for PettingZoo SISL Pursuit (pursuit_v5).

Cooperative pursuit: 8 pursuers, 30 random-walking evaders, 16x16 grid,
5 discrete actions (up/down/left/right/stay), shared team reward.
Obs = flattened 7x7x3 local map per agent; state = flattened global 16x16x3.
Terminates early when every evader is caught (counted as battle_won).
"""
import numpy as np

from .multiagentenv import MultiAgentEnv

N_ACTIONS = 5


class PursuitEnv(MultiAgentEnv):

    def __init__(self, episode_limit=500, n_pursuers=8, n_evaders=30,
                 seed=None, **kwargs):
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from pettingzoo.sisl import pursuit_v5
        self.n_agents = n_pursuers
        self.episode_limit = episode_limit
        self._env = pursuit_v5.parallel_env(
            max_cycles=episode_limit, shared_reward=True,
            n_pursuers=n_pursuers, n_evaders=n_evaders, **kwargs)
        self._seed = seed
        self._next_seed = seed
        self._obs = None
        self._t = 0
        self._ep_reward = 0.0
        self.n_evaders0 = n_evaders
        self.last_grid = None   # global 16x16x3 grid for the semantic interface

    # -------------------------------------------------------------- pymarl
    def reset(self):
        obs, _ = self._env.reset(seed=self._next_seed)
        self._next_seed = None  # vary episodes after the seeded first reset
        self._agents = list(self._env.agents)
        self._obs = obs
        self._t = 0
        self._ep_reward = 0.0
        self.last_grid = np.asarray(self._env.state())
        return self.get_obs(), self.get_state()

    def step(self, actions):
        acts = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs, rew, term, trunc, _ = self._env.step(acts)
        self._obs = obs or self._obs  # keep last obs if env cleared agents
        self._t += 1
        try:
            self.last_grid = np.asarray(self._env.state())
        except Exception:
            pass  # terminal state after all evaders caught may be unavailable
        reward = float(np.mean(list(rew.values())))
        self._ep_reward += reward
        all_term = all(term.values()) if term else True
        all_trunc = all(trunc.values()) if trunc else False
        terminated = all_term or all_trunc
        env_info = {}
        if not terminated and self._t >= self.episode_limit:
            terminated = True
            all_trunc = True
        if all_trunc:
            env_info["episode_limit"] = True
        if terminated:
            # early (non-truncated) termination means every evader was caught
            env_info["battle_won"] = bool(all_term and not all_trunc)
            env_info["ep_reward"] = self._ep_reward
        return reward, terminated, env_info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, i):
        return np.asarray(self._obs[self._agents[i]], dtype=np.float32).reshape(-1)

    def get_obs_size(self):
        if self._obs is None:
            self.reset()
        return len(self.get_obs_agent(0))

    def get_state(self):
        return np.asarray(self._env.state(), dtype=np.float32).reshape(-1)

    def get_state_size(self):
        if self._obs is None:
            self.reset()
        return len(self.get_state())

    @property
    def t_now(self):
        return self._t

    def get_avail_actions(self):
        return [[1] * N_ACTIONS for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, i):
        return [1] * N_ACTIONS

    def get_total_actions(self):
        return N_ACTIONS

    def render(self):
        pass

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
