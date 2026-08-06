"""Cooperative Climbing Matrix Game (paper Fig. 5): two agents repeatedly play a 3x3
common-payoff matrix for a fixed horizon. Dense reward, constant observation, no navigation --
the challenge is purely coordination + credit assignment, with no sparse-reward / partial-
observability pathologies. Ideal for isolating whether the LLM credit actually helps.

Payoff matrix M[a_Alice][a_Bob] gives the shared reward. The global optimum M[2][0]=11 is
flanked by miscoordination penalties (-30), so myopic learners settle for the safe M[1][1]=7.
"""
import numpy as np

from .base import Trajectory

MATRIX = np.array([[0, 6, 5],
                   [-30, 7, 0],
                   [11, -30, 0]], dtype=np.float32)

AGENT_NAMES = ("Alice", "Bob")


class ClimbingEnv:
    def __init__(self, env_id="Climbing", seed=0, max_episode_steps=25):
        self.env_id = env_id
        self.n_agents = 2
        self.n_actions = 3
        self.obs_dim = 1                      # constant observation
        self.horizon = max_episode_steps or 25
        self.agent_names = list(AGENT_NAMES)
        self.matrix = MATRIX
        self._seed = seed
        self._ep = 0

    def _obs(self):
        return tuple(np.ones(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents))

    def rollout(self, policies, epsilon):
        traj = Trajectory()
        self._ep += 1
        obs = self._obs()
        for pol in policies:
            if hasattr(pol, "reset"):
                pol.reset()
        for k in range(self.horizon):
            actions = np.array(
                [policies[i].act(obs[i], epsilon) for i in range(self.n_agents)], dtype=np.int64
            )
            r = float(self.matrix[int(actions[0]), int(actions[1])])
            next_obs = self._obs()
            traj.obs.append(obs)
            traj.actions.append(actions)
            traj.global_reward.append(r)
            traj.next_obs.append(next_obs)
            traj.done.append(k == self.horizon - 1)     # episodic: last step terminal
            traj.state.append({"actions": (int(actions[0]), int(actions[1])), "reward": r})
            obs = next_obs
        return traj

    def collect_batch(self, policies, epsilon, n_episodes):
        return [self.rollout(policies, epsilon) for _ in range(n_episodes)]

    def describe(self, traj=None):
        rows = "\n".join(
            "  " + "  ".join(f"{int(self.matrix[i, j]):>4d}" for j in range(self.n_actions))
            for i in range(self.n_actions)
        )
        p_env = (
            f"There are two robots named Alice and Bob playing a cooperative game for "
            f"{self.horizon} timesteps. At every timestep each robot simultaneously chooses one of "
            f"three actions (0, 1, or 2). Both robots then receive the SAME shared reward, read "
            f"from this common payoff matrix where the row is Alice's action and the column is "
            f"Bob's action:\n{rows}\n"
            f"The best possible reward is 11, obtained only when Alice plays 2 and Bob plays 0, but "
            f"the entries next to it are -30, so if the robots miscoordinate near it they are "
            f"heavily punished. A safe but suboptimal choice is Alice=1, Bob=1 giving 7. The "
            f"objective is to maximize the total shared reward over the episode, which requires the "
            f"two robots to agree on the risky but optimal action pair."
        )
        p_desc = (
            "At each timestep I give you a numpy array called actions of two numbers -- the action "
            "Alice took and the action Bob took -- and the shared reward that pair produced. The "
            "full trajectory of an episode is given as a sequence of these lines."
        )
        return {"env": p_env, "desc": p_desc}

    def serialize(self, traj, precision=0):
        lines = []
        for k in range(len(traj)):
            a0, a1 = traj.state[k]["actions"]
            lines.append(f"t={k}: actions = ({a0}, {a1}), shared reward = {traj.global_reward[k]:.0f}")
        return "\n".join(lines)

    def get_example(self, kind):
        examples = {
            "temporal": (
                "t=0: actions = (2, 0), shared reward = 11.\n"
                "The reward is immediate here, but choosing action 2 (Alice) and 0 (Bob) is what "
                "earned it, so those actions deserve the credit."
            ),
            "structural": (
                "t=0: actions = (1, 1), shared reward = 7.\n"
                "The reward 7 was produced by BOTH robots agreeing on action 1; each contributed "
                "equally, so the credit should be shared between them."
            ),
            "under-collaboration": (
                "t=0: actions = (2, 1), shared reward = -30.\n"
                "Alice reached for the optimal action 2 but Bob did not play the matching action 0, "
                "so they were punished. This is under-collaboration: one robot committed to the "
                "risky optimum while the other did not."
            ),
            "over-collaboration": (
                "Both robots keep playing the safe action 1 for reward 7 every timestep, never "
                "risking the coordinated jump to the optimal (2, 0) = 11. They are being overly "
                "cautious and leaving reward on the table."
            ),
            "agreement": (
                "t=0: actions = (2, 0), shared reward = 11.\n"
                "Alice and Bob correctly agreed on the risky optimal action pair (2, 0), solving "
                "the Agreement Problem and earning the maximum reward."
            ),
        }
        return examples[kind]
