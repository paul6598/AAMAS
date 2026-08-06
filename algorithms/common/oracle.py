"""Oracle dense credit (diagnostic only -- NOT a submitted model).

Computes a perfect, ground-truth DENSE per-agent credit from the global state, to answer one
question: if densification were done perfectly, does dense credit + the RNN backbone reach the
true-reward ceiling (~0.85 on 5x5-2p-2f)? If yes, the densification IDEA is right and the
LLM-MCA gap is purely the LLM's credit *quality/consistency*; if no, density alone is not enough.

Credit_i[k] = load_base_i[k] + SHAPE * potential_shaping_i[k]
  load_base:   the actual collection reward at step k, given to the adjacent agent(s) that loaded.
  shaping:     F = gamma*Phi(s') - Phi(s), with Phi_i = -(distance to nearest active apple).
               Potential-based shaping provably preserves the optimal policy -- it only makes the
               signal dense/directional, it does not change what "optimal" is. This is exactly the
               densification the LLM prompt asks for, computed perfectly from ground truth.
"""
import numpy as np

SHAPE = 0.1  # weight on the dense shaping term (collection reward stays the true objective)


def _active_apples(state):
    return [(r, c) for (r, c, lvl) in state["foods"] if r >= 0]


def _min_dist(pos, apples):
    if not apples:
        return 0.0
    return float(min(abs(pos[0] - r) + abs(pos[1] - c) for (r, c) in apples))


class OracleDenseCritic:
    def __init__(self, gamma=0.99):
        self.gamma = gamma

    def _one(self, traj, env):
        n, T = env.n_agents, len(traj)
        credits = np.zeros((n, T), dtype=np.float32)
        for k in range(T):
            st = traj.state[k]
            apples = _active_apples(st)
            nst = traj.state[k + 1] if k + 1 < T else st
            for i in range(n):
                pos = st["agents"][i][:2]
                npos = nst["agents"][i][:2]
                phi = -_min_dist(pos, apples)
                phi_next = -_min_dist(npos, apples)      # same apple set -> directional shaping
                credits[i, k] += SHAPE * (self.gamma * phi_next - phi)
            # load_base: distribute the step's collection reward to adjacent loaders
            g = traj.global_reward[k]
            if g > 0.0:
                loaders = [i for i in range(n)
                           if int(traj.actions[k][i]) == 5 and _min_dist(st["agents"][i][:2], apples) <= 1.0]
                if not loaders:
                    loaders = [i for i in range(n) if int(traj.actions[k][i]) == 5] or list(range(n))
                for i in loaders:
                    credits[i, k] += g / len(loaders)
        return credits

    def assign_credits_batch(self, trajs, env):
        return [self._one(traj, env) for traj in trajs], "oracle-dense"
