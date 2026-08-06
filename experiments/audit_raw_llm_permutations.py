"""Adversarial reserialization audit for a raw sequential LLM-MCA critic."""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common.permutation_data import inverse_agent_axis, permuted_lbf_view
from algorithms.common.rnn import RIQL
from algorithms.llm_mca.critic import LLMCritic
from envs.base import Trajectory
from envs import make_env


def score(critic, traj, env):
    credit, _ = critic.assign_credits(
        traj, env, env.serialize(traj, include_progress=True),
    )
    return np.asarray(credit, dtype=np.float32)


def cooperative_success():
    """A grounded nonzero-credit probe; avoids conflating order drift with all-zero parsing."""
    state = {
        "agents": [(1, 0, 1), (3, 0, 1), (4, 4, 1)],
        "foods": [(2, 0, 2), (4, 3, 1)],
    }
    next_state = {
        "agents": [(1, 0, 1), (3, 0, 1), (4, 4, 1)],
        "foods": [(-1, -1, 0), (4, 3, 1)],
    }
    return Trajectory(
        actions=[np.asarray([5, 5, 0])], global_reward=[2.0], done=[True],
        state=[state], next_state=[next_state],
    )


def delayed_cooperation():
    """Three-step success: two robots make distinct preparatory moves, then co-load.

    This avoids the degenerate one-line prompt where a critic can simply copy the two adjacent
    loaders.  The third robot is deliberately irrelevant to the successful level-2 apple.
    """
    states = [
        {"agents": [(0, 1, 1), (4, 1, 1), (4, 4, 1)], "foods": [(2, 2, 2), (4, 3, 1)]},
        {"agents": [(1, 1, 1), (3, 1, 1), (4, 4, 1)], "foods": [(2, 2, 2), (4, 3, 1)]},
        {"agents": [(2, 1, 1), (2, 1, 1), (4, 4, 1)], "foods": [(2, 2, 2), (4, 3, 1)]},
    ]
    next_states = [states[1], states[2], {
        "agents": [(2, 1, 1), (2, 1, 1), (4, 4, 1)],
        "foods": [(-1, -1, 0), (4, 3, 1)],
    }]
    return Trajectory(
        actions=[np.asarray([2, 1, 0]), np.asarray([2, 1, 0]), np.asarray([5, 5, 0])],
        global_reward=[0.0, 0.0, 2.0], done=[False, False, True], state=states, next_state=next_states,
    )


def two_object_sequence():
    """A cooperative collection followed by a separate single-agent collection."""
    states = [
        {"agents": [(1, 0, 1), (3, 0, 1), (4, 4, 1)], "foods": [(2, 0, 2), (4, 3, 1)]},
        {"agents": [(1, 0, 1), (3, 0, 1), (4, 4, 1)], "foods": [(-1, -1, 0), (4, 3, 1)]},
    ]
    next_states = [states[1], {
        "agents": [(1, 0, 1), (3, 0, 1), (4, 4, 1)],
        "foods": [(-1, -1, 0), (-1, -1, 0)],
    }]
    return Trajectory(
        actions=[np.asarray([5, 5, 0]), np.asarray([0, 0, 5])],
        global_reward=[2.0, 1.0], done=[False, True], state=states, next_state=next_states,
    )


CASES = {
    "one_step": cooperative_success,
    "delayed": delayed_cooperation,
    "two_object": two_object_sequence,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Foraging-5x5-3p-2f-coop-v3")
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--case", choices=sorted(CASES), default="one_step")
    parser.add_argument("--all_agent_permutations", action="store_true",
                        help="Evaluate every agent ordering (small-N diagnostic).")
    parser.add_argument("--all_object_permutations", action="store_true",
                        help="Evaluate every object ordering (small-M diagnostic).")
    parser.add_argument("--max_permutations", type=int, default=0,
                        help="Optional cap after the identity ordering is removed; 0 means no cap.")
    args = parser.parse_args()
    env = make_env(args.env, seed=0, max_episode_steps=args.max_steps)
    trajectory = CASES[args.case]()
    critic = LLMCritic(model_name=args.model, max_new_tokens=args.max_new_tokens, device="cuda")
    baseline = score(critic, trajectory, env)
    agent_identity = tuple(range(env.n_agents))
    if args.all_agent_permutations:
        agent_permutations = [np.asarray(p) for p in itertools.permutations(range(env.n_agents))
                              if p != agent_identity]
    else:
        agent_permutations = [np.roll(np.arange(env.n_agents), shift) for shift in range(1, env.n_agents)]
    object_count = len(trajectory.state[0]["foods"])
    if args.all_object_permutations:
        object_permutations = [np.asarray(p) for p in itertools.permutations(range(object_count))]
    else:
        # Agent-order robustness is the primary claim.  Keep object serialization fixed unless
        # an explicit object-order ablation is requested.
        object_permutations = [np.arange(object_count)]
    permutations = [(ap, op) for ap in agent_permutations for op in object_permutations]
    if args.max_permutations:
        permutations = permutations[:args.max_permutations]
    print("CASE", args.case, "BASELINE_CREDIT", baseline.tolist(), flush=True)
    for ap, op in permutations:
        view, view_env = permuted_lbf_view(trajectory, env, ap, op)
        raw = score(critic, view, view_env)
        restored = inverse_agent_axis(raw, ap)
        drift = float(np.max(np.abs(baseline - restored)))
        team_drift = float(np.max(np.abs(baseline.sum(axis=0) - restored.sum(axis=0))))
        print({
            "agent_permutation": ap.tolist(), "object_permutation": op.tolist(),
            "credit_max_drift": drift, "team_credit_max_drift": team_drift,
            "raw_credit": raw.tolist(),
        }, flush=True)


if __name__ == "__main__":
    main()
