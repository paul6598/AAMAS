"""Finite-precision permutation audit for the Qwen PEEL event encoder.

Uses the same hand-constructed LBF probes as the raw LLM-MCA audit.  Outputs are not semantic
credits yet: this checks whether structured parallel outputs stay stable under the exact input
reserializations that make the raw text critic change its credits.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.peel.permutation import audit_event_equivariance
from algorithms.peel.qwen import QwenPEELEventEncoder
from algorithms.peel.transition import lbf_transition
from envs import make_env
from experiments.audit_raw_llm_permutations import CASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--max_permutations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = make_env("Foraging-5x5-3p-2f-coop-v3", seed=args.seed, max_episode_steps=6)
    trajectory = CASES[args.case]()
    records = [lbf_transition(trajectory, t) for t in range(len(trajectory))]
    model = QwenPEELEventEncoder().cuda().to(dtype=torch.bfloat16).eval()
    # Materialize rule tokens once so this audit has the same rule context in every permutation.
    model.rule_tokens(env.describe()["env"])
    permutations = [
        (np.asarray(ap), np.asarray(op))
        for ap in itertools.permutations(range(env.n_agents))
        if ap != tuple(range(env.n_agents))
        for op in itertools.permutations(range(len(records[0].object_features)))
    ]
    if args.max_permutations:
        permutations = permutations[:args.max_permutations]
    rows = []
    with torch.no_grad():
        for timestep, record in enumerate(records):
            for ap, op in permutations:
                drift = audit_event_equivariance(model, record, ap, op)
                rows.append(drift)
                print({"case": args.case, "timestep": timestep,
                       "agent_permutation": ap.tolist(), "object_permutation": op.tolist(), **drift}, flush=True)
    print({
        "SUMMARY": args.case, "n_audits": len(rows),
        "max_pointer_probability_drift": max(
            max(row["agent_pointer_prob_drift"], row["object_pointer_prob_drift"]) for row in rows
        ),
        "max_reward_drift": max(row["reward_drift"] for row in rows),
    }, flush=True)


if __name__ == "__main__":
    main()
