"""Small integrated LoRA training smoke for the Qwen relational event encoder."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.peel.permutation import audit_event_equivariance
from algorithms.peel.qwen import QwenPEELEventEncoder
from algorithms.peel.transition import lbf_transition
from envs import make_env


class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self, obs, epsilon):
        return random.randrange(self.n_actions)


def records_for(env, episodes):
    policies = [RandomPolicy(env.n_actions) for _ in range(env.n_agents)]
    records = []
    for _ in range(episodes):
        trajectory = env.rollout(policies, epsilon=1.0)
        records.extend(lbf_transition(trajectory, step) for step in range(len(trajectory)))
    # Fixed-cardinality first smoke. Padded variable entity sets are the next adapter extension.
    buckets = {}
    for record in records:
        buckets.setdefault((len(record.agent_features), len(record.object_features)), []).append(record)
    return max(buckets.values(), key=len)


def loss(model, record, rule):
    agent = torch.as_tensor(record.agent_features, device="cuda", dtype=torch.bfloat16)[None]
    obj = torch.as_tensor(record.object_features, device="cuda", dtype=torch.bfloat16)[None]
    output = model(agent, obj, rule)
    agent_target = torch.as_tensor(record.agent_delta, device="cuda")[None]
    object_target = torch.as_tensor(record.object_delta, device="cuda")[None]
    reward_target = torch.tensor([record.reward], device="cuda")
    return (
        F.mse_loss(output["agent_delta"].float(), agent_target)
        + F.mse_loss(output["object_delta"].float(), object_target)
        + F.mse_loss(output["reward"].float(), reward_target)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Foraging-5x5-3p-2f-coop-v3")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = make_env(args.env, seed=args.seed, max_episode_steps=25)
    records = records_for(env, args.episodes)
    model = QwenPEELEventEncoder(lora_rank=args.lora_rank).cuda().to(dtype=torch.bfloat16)
    rule = model.rule_tokens(env.describe()["env"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4,
    )
    model.train()
    for step in range(args.steps):
        value = loss(model, records[step % len(records)], rule)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        if step % 8 == 0 or step == args.steps - 1:
            model.eval()
            drift = audit_event_equivariance(
                model, records[(step + 1) % len(records)],
                np.random.permutation(env.n_agents), np.random.permutation(len(records[0].object_features)),
            )
            model.train()
            print(
                f"step={step:03d} loss={float(value.detach()):.6g} "
                f"prob_drift={max(drift['agent_pointer_prob_drift'], drift['object_pointer_prob_drift']):.3g} "
                f"reward_drift={drift['reward_drift']:.3g}", flush=True,
            )


if __name__ == "__main__":
    main()
