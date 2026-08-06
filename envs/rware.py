"""Robotic Warehouse (RWARE) environment wrapper (paper Fig. 6b).

Partially observable, VERY sparse: agents must navigate to a requested shelf, load it, carry it
to a goal (delivery), then return it. A single global reward = sum of per-agent delivery rewards.
Same interface as envs/lbf.py: rollout / collect_batch / describe / serialize / get_example, plus
a global-state snapshot (traj.state) for the centralized LLM critic.
"""
import gymnasium as gym
import numpy as np
import rware  # noqa: F401  (registers rware-* envs)
from rware.warehouse import Action

from .base import Trajectory

ACTION_MEANINGS = ("do nothing", "move forward", "turn left", "turn right",
                   "load/unload a shelf (toggle)")
AGENT_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank")


class RWAREEnv:
    def __init__(self, env_id="rware-tiny-2ag-v2", seed=0, max_episode_steps=None):
        kw = {"max_steps": max_episode_steps} if max_episode_steps else {}
        self.env = gym.make(env_id, **kw)
        self.env_id = env_id
        self.n_agents = len(self.env.action_space)
        self.n_actions = self.env.action_space[0].n
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.agent_names = list(AGENT_NAMES[: self.n_agents])
        g = np.array(self.env.unwrapped.grid).shape
        self.grid_h, self.grid_w = g[1], g[2]
        self._scale = float(max(self.grid_h, self.grid_w))
        self._seed = seed
        self._ep = 0

    def _snapshot_state(self):
        u = self.env.unwrapped
        agents = [(int(a.x), int(a.y), a.dir.name,
                   (int(a.carrying_shelf.id) if a.carrying_shelf is not None else None))
                  for a in u.agents]
        goals = [(int(x), int(y)) for (x, y) in u.goals]
        requested = [(int(s.x), int(s.y), int(s.id)) for s in u.request_queue]
        return {"agents": agents, "goals": goals, "requested": requested}

    def _norm(self, obs):
        return tuple(np.asarray(o, dtype=np.float32) / self._scale for o in obs)

    def rollout(self, policies, epsilon):
        traj = Trajectory()
        raw_obs, _ = self.env.reset(seed=self._seed + self._ep)
        self._ep += 1
        obs = self._norm(raw_obs)
        for pol in policies:
            if hasattr(pol, "reset"):
                pol.reset()
        done = False
        while not done:
            state = self._snapshot_state()
            actions = np.array(
                [policies[i].act(obs[i], epsilon) for i in range(self.n_agents)], dtype=np.int64
            )
            raw_next, rewards, term, trunc, _ = self.env.step(tuple(int(a) for a in actions))
            next_obs = self._norm(raw_next)
            traj.obs.append(obs)
            traj.actions.append(actions)
            traj.global_reward.append(float(np.sum(rewards)))
            traj.next_obs.append(next_obs)
            traj.done.append(bool(term or trunc))
            traj.state.append(state)
            traj.next_state.append(self._snapshot_state())
            obs = next_obs
            done = term or trunc
        return traj

    def collect_batch(self, policies, epsilon, n_episodes):
        return [self.rollout(policies, epsilon) for _ in range(n_episodes)]

    def describe(self, traj=None):
        names = self.agent_names
        name_list = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
        start = ""
        if traj is not None and traj.state:
            s0 = traj.state[0]
            starts = ". ".join(
                f"{names[i]} starts at ({x}, {y}) facing {d}" + (f" carrying shelf {c}" if c is not None else "")
                for i, (x, y, d, c) in enumerate(s0["agents"]))
            goals = ", ".join(f"({x}, {y})" for (x, y) in s0["goals"])
            reqs = ", ".join(f"shelf {i} at ({x}, {y})" for (x, y, i) in s0["requested"])
            start = f" {starts}. Delivery goals are at {goals}. Currently requested shelves: {reqs}."
        p_env = (
            f"There are {self.n_agents} robots named {name_list} operating in a "
            f"{self.grid_h}x{self.grid_w} robotic warehouse.{start} Each robot can do one of five "
            f"actions: do nothing, move forward, turn left, turn right, or toggle load/unload a "
            f"shelf. The team's goal is to deliver REQUESTED shelves to a delivery goal square: a "
            f"robot must move onto a requested shelf, load it, carry it to a goal square, and "
            f"deliver it. The team receives a reward each time a requested shelf is delivered; the "
            f"reward is sparse and only occurs on successful deliveries."
        )
        obs_form = ("Each robot only sees a 3x3 area around itself, so the global state (positions, "
                    "directions, carried shelves, goals, requested shelves) is provided to you.")
        return {"env": p_env, "desc": obs_form}

    def serialize(self, traj, precision=0):
        lines = []
        for k in range(len(traj)):
            st = traj.state[k]
            ag = ", ".join(f"{self.agent_names[i]}@({x},{y}){d[0]}" + (f"[shelf{c}]" if c is not None else "")
                           for i, (x, y, d, c) in enumerate(st["agents"]))
            acts = tuple(int(a) for a in traj.actions[k])
            lines.append(f"t={k}: {ag}, actions={acts}, global reward={traj.global_reward[k]:.0f}")
        return "\n".join(lines)

    def get_example(self, kind):
        return ("Example: a robot moves onto a requested shelf, toggles load, carries it to a goal "
                "square, and toggles unload to deliver it -- earning the team its sparse reward.")


ACTION_INTS = {a.name: int(a.value) for a in Action}  # NOOP0 FORWARD1 LEFT2 RIGHT3 TOGGLE_LOAD4
