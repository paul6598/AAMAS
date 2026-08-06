"""Baseline algorithm: independent Double DQN agents trained on an equal split of the global
reward (credit_i[k] = global_reward[k] / n_agents). No LLM -- fast, deterministic; useful both
as a MARL baseline and for verifying the shared training loop.
"""
import numpy as np

from ..common.trainer import train_with_critic


class EqualSplitCritic:
    def _one(self, traj, env):
        n, T = env.n_agents, len(traj)
        credits = np.zeros((n, T), dtype=np.float32)
        for k in range(T):
            credits[:, k] = traj.global_reward[k] / n
        return credits

    def assign_credits_batch(self, trajs, env):
        return [self._one(traj, env) for traj in trajs], "equal-split"


def train(env, eval_env, args):
    return train_with_critic(env, eval_env, EqualSplitCritic(), args)


def train_oracle(env, eval_env, args):
    """ddqn_oracle: MLP Double DQN trained on the oracle DENSE credit (isolates the MLP backbone
    from credit quality -- the same-backbone counterpart to rnn_oracle)."""
    from ..common.oracle import OracleDenseCritic
    return train_with_critic(env, eval_env, OracleDenseCritic(gamma=args.gamma), args)
