"""Level-Based Foraging (LBF) environment: rollouts, trajectory serialization, and the
LBF-specific prompt text the LLM critic needs (scenario rules + input description + examples).

Following the paper (Fig. 3), the centralized critic receives a FULL observation of the world
as named positions/levels (Alice, Bob, apples), not the agents' partial observation vectors.
We capture that global state from the underlying env at every step (traj.state); the
decentralized policies still only ever see their own partial observation vectors.

The team receives a single collective (global) reward at each timestep, obtained by summing
the per-agent rewards LBF returns. The per-agent LBF rewards are never shown to the critic or
the policies -- credit is what the LLM must infer.
"""
import re

import gymnasium as gym
import lbforaging  # noqa: F401  (registers Foraging-* envs)
import numpy as np

from .base import Trajectory

# LBF discrete action semantics (lbforaging.foraging.environment.Action)
ACTION_MEANINGS = ("doing nothing", "moving up", "moving down", "moving left", "moving right",
                   "picking up an apple (load)")

AGENT_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank")


class LBFEnv:
    def __init__(self, env_id="Foraging-8x8-2p-2f-coop-v3", seed=0, max_episode_steps=None):
        kw = {"max_episode_steps": max_episode_steps} if max_episode_steps else {}
        self.env = gym.make(env_id, **kw)
        self.env_id = env_id
        self.n_agents = len(self.env.action_space)
        self.n_actions = self.env.action_space[0].n
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.agent_names = list(AGENT_NAMES[: self.n_agents])
        grid = re.search(r"(\d+)x(\d+)", env_id)
        self.grid_size = int(grid.group(1)) if grid else 8
        self._seed = seed
        self._ep = 0
        self._initial_foods = []   # [(row, col, level)] at reset of the last episode

    def _snapshot_state(self):
        """Global world state from the underlying env: agents and apples as (row, col, level).

        Harvested apples keep their slot with position (-1, -1) and level 0, so the critic can
        see WHICH apple disappeared and when.
        """
        u = self.env.unwrapped
        agents = [(int(p.position[0]), int(p.position[1]), int(p.level)) for p in u.players]
        foods = []
        for (r, c, lvl) in self._initial_foods:
            if u.field[r, c] > 0:
                foods.append((r, c, lvl))
            else:
                foods.append((-1, -1, 0))
        return {"agents": agents, "foods": foods}

    def _norm(self, obs):
        """Normalize the policy's observation to ~[0,1] by dividing by the grid size. LBF's raw
        observation is unnormalized absolute coordinates (0..grid_size); feeding those into an
        MLP alongside small reward targets slows learning a lot. The critic is unaffected (it
        reads traj.state, not traj.obs)."""
        return tuple(np.asarray(o, dtype=np.float32) / self.grid_size for o in obs)

    def rollout(self, policies, epsilon):
        """Run one episode with per-agent epsilon-greedy `policies`. Returns a Trajectory.

        `policies[i].act(obs_i, epsilon)` must return an int action.
        """
        traj = Trajectory()
        raw_obs, _ = self.env.reset(seed=self._seed + self._ep)
        self._ep += 1
        field = self.env.unwrapped.field
        self._initial_foods = [(int(r), int(c), int(field[r, c]))
                               for r, c in zip(*np.nonzero(field))]
        obs = self._norm(raw_obs)
        for pol in policies:                       # recurrent policies reset hidden state per episode
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
            g = float(np.sum(rewards))
            traj.obs.append(obs)
            traj.actions.append(actions)
            traj.global_reward.append(g)
            traj.next_obs.append(next_obs)
            # A configured max_steps is the effective task horizon. Mark its cutoff terminal so
            # value-based learners do not bootstrap from a state that is never continued.
            traj.done.append(bool(term or trunc))
            traj.state.append(state)
            traj.next_state.append(self._snapshot_state())
            obs = next_obs
            done = term or trunc
        return traj

    def collect_batch(self, policies, epsilon, n_episodes):
        return [self.rollout(policies, epsilon) for _ in range(n_episodes)]

    def describe(self, traj=None, include_progress=True):
        """LBF-specific prompt pieces (paper Fig. 3): p_env (scenario rules, with this episode's
        starting locations when `traj` is given) and p_desc (input format description)."""
        names = self.agent_names
        name_list = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]

        start = ""
        if traj is not None and traj.state:
            s0 = traj.state[0]
            starts = ". ".join(
                f"The starting location of {names[i]} is ({r}, {c}) and their level is {lvl}"
                for i, (r, c, lvl) in enumerate(s0["agents"])
            )
            apples = ". ".join(
                f"Apple {j + 1} is located at ({r}, {c}) and its level is {lvl}"
                for j, (r, c, lvl) in enumerate(s0["foods"])
            )
            start = f" {starts}. {apples}."

        p_env = (
            f"There are {self.n_agents} robots named {name_list}. "
            f"{name_list} are foraging robots on a {self.grid_size}x{self.grid_size} grid tasked with "
            f"collecting apples. Each robot and each apple has an integer level.{start} "
            f"At each timestep, each robot can do one of six actions: do nothing, move up, move down, "
            f"move left, move right, or load (pick up an apple). An apple can only be picked up if the "
            f"combined levels of the robots standing on cells adjacent to it and simultaneously "
            f"performing the load action is greater than or equal to the level of the apple. "
            f"When an apple is picked up, the team receives a reward proportional to the level of that "
            f"apple and the apple disappears from the grid. If the robots are at the boundary of the "
            f"grid and attempt an action that would take them off the grid, they stay in place. "
            f"The objective is to collect all the apples in as few timesteps as possible, which "
            f"sometimes requires the robots to cooperate on the same apple."
        )

        obs_form = ", ".join([f"position of {n}" for n in names] +
                             [f"position of apple {j + 1}" for j in range(len(self._initial_foods))])
        lvl_form = ", ".join([f"level of {n}" for n in names] +
                             [f"level of apple {j + 1}" for j in range(len(self._initial_foods))])
        act_legend = ", ".join(f"{a} denotes {m}" for a, m in enumerate(ACTION_MEANINGS))
        progress_desc = (
            " For arithmetic reliability, progress_to_apples contains one tuple per "
            "robot and one signed entry per apple. It is the decrease in Manhattan distance to an "
            "adjacent loading cell caused by that transition: +1 means closer, -1 means farther, "
            "and 0 means unchanged. This is derived from the displayed positions, not an extra "
            "reward."
            if include_progress else ""
        )
        p_desc = (
            f"At each timestep, I will give you the global reward and a full observation of the world "
            f"using a numpy array called observations of the form ({obs_form}) where each position is a "
            f"collection of two numbers (row, column). Once an apple has been picked up, its position is "
            f"written as (-1, -1). I also give a numpy array called levels of the form ({lvl_form}). "
            f"I also give you a numpy array called actions of {self.n_agents} numbers which describes "
            f"the action each of {name_list} took. Each robot can pick one of 6 actions where "
            f"{act_legend}.{progress_desc} The full trajectory of an episode is given as a sequence "
            f"of these lines."
        )
        return {"env": p_env, "desc": p_desc}

    def get_example(self, kind):
        """Short example episodes for the definition prompts (the paper's utils.get_example)."""
        examples = {
            "temporal": (
                "t=0: Alice at (4, 1) moves right toward the apple at (4, 4). global reward = 0.\n"
                "t=1: Alice at (4, 2) moves right. global reward = 0.\n"
                "t=2: Alice at (4, 3) performs load next to the apple. global reward = 2.\n"
                "The reward only appears at t=2, but the moves at t=0 and t=1 made it possible, so "
                "they should also receive credit. A suitable dense annotation is "
                "alice_credit = [1, 1, 2] and bob_credit = [0, 0, 0]."
            ),
            "structural": (
                "t=0: Alice moves toward the apple while Bob walks away from every apple.\n"
                "t=1: Alice loads the apple alone. global reward = 2.\n"
                "The team reward is 2, but it was earned entirely by Alice; Bob contributed nothing "
                "and should receive no credit for it. A suitable annotation is "
                "alice_credit = [1, 2] and bob_credit = [0, 0]."
            ),
            "under-collaboration": (
                "A level-2 apple requires two level-1 robots to load it together. Alice stands next "
                "to it and performs load while Bob stays on the other side of the grid. The load "
                "fails and the team receives no reward: this is under-collaboration, since only one "
                "robot attempted a task that requires two. A suitable one-step annotation is "
                "alice_credit = [0.5] and bob_credit = [-1]."
            ),
            "over-collaboration": (
                "A level-1 apple can be picked up by a single level-1 robot, but both Alice and Bob "
                "walk to it and load it together while another level-1 apple is left uncollected. "
                "The team wastes timesteps: this is over-collaboration, since more robots than "
                "necessary committed to the same task. A suitable annotation for the final approach "
                "and load is alice_credit = [1, 2] and bob_credit = [-1, -1]."
            ),
            "agreement": (
                "The grid has a level-2 apple and a level-1 apple. Alice and Bob (both level 1) "
                "first meet at the level-2 apple and load it together, then Bob alone collects the "
                "level-1 apple while Alice stays out of his way. Exactly the right number of robots "
                "cooperated on each apple: they have solved the Agreement Problem. A suitable "
                "two-stage annotation is alice_credit = [2, 0] and bob_credit = [2, 2]."
            ),
        }
        return examples[kind]

    def serialize(self, traj, precision=2, include_progress=True):
        """Render a trajectory as the paper-style text: full world observation (positions +
        levels), numeric actions, and global reward, one line per timestep."""
        lines = []
        for k in range(len(traj)):
            s = traj.state[k]
            ns = traj.next_state[k]
            positions = [(r, c) for (r, c, _) in s["agents"]] + [(r, c) for (r, c, _) in s["foods"]]
            levels = [lvl for (_, _, lvl) in s["agents"]] + [lvl for (_, _, lvl) in s["foods"]]
            obs_str = ", ".join(f"({r}, {c})" for r, c in positions)
            lvl_str = ", ".join(str(v) for v in levels)
            act_str = ", ".join(str(int(a)) for a in traj.actions[k])
            line = (f"t={k}: observations = ({obs_str}), levels = ({lvl_str}), "
                    f"actions = ({act_str})")
            if include_progress:
                progress = []
                for i in range(self.n_agents):
                    pos = s["agents"][i][:2]
                    next_pos = ns["agents"][i][:2]
                    per_apple = []
                    for r, c, _ in s["foods"]:
                        if r < 0:
                            per_apple.append(0)
                            continue
                        before = max(0, abs(pos[0] - r) + abs(pos[1] - c) - 1)
                        after = max(0, abs(next_pos[0] - r) + abs(next_pos[1] - c) - 1)
                        per_apple.append(int(before - after))
                    progress.append(tuple(per_apple))
                line += ", progress_to_apples = (" + ", ".join(str(x) for x in progress) + ")"
            lines.append(line + f", global reward = {traj.global_reward[k]:.{precision}f}")
        return "\n".join(lines)
