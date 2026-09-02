"""pymarl wrapper for Google Research Football (left team, N field players).

Cooperative, discrete (19 actions, always available). Team reward = mean of
the per-player rewards (scoring is shared, checkpoints are team-level).
`last_raw` keeps the most recent raw observation of controlled player 0 for the
semantic interface (env/semantic/grf.py).
"""
import numpy as np

from .multiagentenv import MultiAgentEnv

N_ACTIONS = 19
GAME_MODES = 7
ROLES = 10


class GFootballEnv(MultiAgentEnv):

    def __init__(self, scenario="5_vs_5", n_agents=4, episode_limit=1000,
                 rewards="scoring,checkpoints", right_difficulty=None,
                 seed=None, logdir="", **kwargs):
        import gfootball.env as football_env
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        other = {}
        if right_difficulty is not None:
            other["right_team_difficulty"] = float(right_difficulty)
        self._env = football_env.create_environment(
            env_name=scenario, representation="raw", stacked=False,
            rewards=rewards, logdir=logdir, write_goal_dumps=False,
            write_full_episode_dumps=False, render=False,
            number_of_left_players_agent_controls=n_agents,
            other_config_options=other)
        self._seed = seed
        if seed is not None:
            self._env.seed(seed)
        self.last_raw = None
        self._t = 0
        self.n_left = None
        self.n_right = None

    # ------------------------------------------------------------ helpers
    def _raw0(self):
        return self.last_raw[0]

    @property
    def controlled(self):
        """left_team indices of the controlled players, agent order."""
        return [o["active"] for o in self.last_raw]

    def _feat_state(self, o):
        f = [o["left_team"].reshape(-1), o["left_team_direction"].reshape(-1),
             o["right_team"].reshape(-1), o["right_team_direction"].reshape(-1),
             np.asarray(o["ball"]), np.asarray(o["ball_direction"]),
             np.eye(3)[o["ball_owned_team"] + 1], np.eye(GAME_MODES)[o["game_mode"]],
             np.array([o["score"][0] - o["score"][1],
                       o["steps_left"] / max(1.0, float(self._steps_total))])]
        return np.concatenate(f).astype(np.float32)

    # -------------------------------------------------------------- pymarl
    def reset(self):
        self.last_raw = self._env.reset()
        self._t = 0
        o = self._raw0()
        self.n_left, self.n_right = len(o["left_team"]), len(o["right_team"])
        self._steps_total = o["steps_left"]
        return self.get_obs(), self.get_state()

    def step(self, actions):
        acts = [int(a) for a in actions]
        raw, rew, done, info = self._env.step(acts)
        self.last_raw = raw
        self._t += 1
        reward = float(np.mean(rew))
        o = self._raw0()
        terminated = bool(done)
        env_info = {"score_left": int(o["score"][0]), "score_right": int(o["score"][1])}
        if not terminated and self._t >= self.episode_limit:
            terminated = True
            env_info["episode_limit"] = True
        if terminated:
            env_info["battle_won"] = o["score"][0] > o["score"][1]
            env_info["goal_diff"] = int(o["score"][0] - o["score"][1])
        return reward, terminated, env_info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, i):
        o = self.last_raw[i]
        me = o["active"]
        f = [self._feat_state(o),
             np.eye(self.n_left)[me],
             np.eye(ROLES)[o["left_team_roles"][me]],
             np.asarray(o["sticky_actions"], dtype=np.float32)]
        return np.concatenate(f).astype(np.float32)

    def get_obs_size(self):
        if self.last_raw is None:
            self.reset()
        return len(self.get_obs_agent(0))

    def get_state(self):
        return self._feat_state(self._raw0())

    def get_state_size(self):
        if self.last_raw is None:
            self.reset()
        return len(self.get_state())

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
