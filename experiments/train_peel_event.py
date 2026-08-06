"""Stage-1 event representation experiment on generic transition deltas.

This intentionally trains no policy and contains no environment-specific event label.  It checks
whether a set-structured event model can ground entity transitions while retaining exact
permutation behavior.  Use this before the expensive pretrained-LLM adaptation.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ``python experiments/train_relational_event.py`` sets sys.path to experiments/, unlike
# ``python -m``.  Add the repository root so this runner is directly usable in tmux.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.peel.model import PEELEventEncoder
from algorithms.peel.permutation import audit_event_equivariance
from algorithms.peel.transition import lbf_transition
from envs import make_env


class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self, obs, epsilon):
        return random.randrange(self.n_actions)


def collect(env, episodes):
    policies = [RandomPolicy(env.n_actions) for _ in range(env.n_agents)]
    records = []
    for _ in range(episodes):
        traj = env.rollout(policies, epsilon=1.0)
        records.extend(lbf_transition(traj, t) for t in range(len(traj)))
    return records


def fixed_cardinality_bucket(records):
    """Temporary Stage-1 bucket; the next encoder revision uses padded variable sets.

    Some LBF variants randomize the active object count across resets.  Bucketing avoids
    treating padding as a real apple while preserving the experiment's within-layout event and
    permutation diagnostic.
    """
    counts = {}
    for record in records:
        key = (record.agent_features.shape[0], record.object_features.shape[0])
        counts.setdefault(key, []).append(record)
    key, bucket = max(counts.items(), key=lambda item: len(item[1]))
    print(f"cardinality_bucket={key} retained={len(bucket)}/{len(records)}", flush=True)
    return bucket


def batch(records, indices, device):
    # LBF has fixed N/M per selected environment.  Variable-cardinality padding is the next
    # adapter step for RWARE; keeping this first experiment fixed makes delta diagnostics clear.
    selected = [records[i] for i in indices]
    return {
        "agent": torch.as_tensor(np.stack([r.agent_features for r in selected]), device=device),
        "object": torch.as_tensor(np.stack([r.object_features for r in selected]), device=device),
        "agent_delta": torch.as_tensor(np.stack([r.agent_delta for r in selected]), device=device),
        "object_delta": torch.as_tensor(np.stack([r.object_delta for r in selected]), device=device),
        "reward": torch.as_tensor([r.reward for r in selected], device=device),
    }


def loss_and_metrics(model, data):
    out = model(data["agent"], data["object"])
    agent_loss = nn.functional.mse_loss(out["agent_delta"], data["agent_delta"])
    object_loss = nn.functional.mse_loss(out["object_delta"], data["object_delta"])
    reward_loss = nn.functional.mse_loss(out["reward"], data["reward"])
    loss = agent_loss + object_loss + reward_loss
    return loss, {
        "agent_delta_mse": float(agent_loss.detach()),
        "object_delta_mse": float(object_loss.detach()),
        "reward_mse": float(reward_loss.detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Foraging-5x5-3p-2f-coop-v3")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--event_slots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    env = make_env(args.env, seed=args.seed, max_episode_steps=25)
    records = fixed_cardinality_bucket(collect(env, args.episodes))
    split = int(len(records) * 0.8)
    train, validation = records[:split], records[split:]
    model = PEELEventEncoder(
        agent_dim=train[0].agent_features.shape[-1], object_dim=train[0].object_features.shape[-1],
        dim=args.dim, event_slots=args.event_slots,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train))
        for start in range(0, len(order), args.batch_size):
            data = batch(train, order[start:start + args.batch_size], device)
            loss, _ = loss_and_metrics(model, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            data = batch(validation, np.arange(len(validation)), device)
            _, metrics = loss_and_metrics(model, data)
        record = validation[epoch % len(validation)]
        ap = np.random.permutation(len(record.agent_features))
        op = np.random.permutation(len(record.object_features))
        drift = audit_event_equivariance(model, record, ap, op)
        print(
            f"epoch={epoch:03d} "
            + " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
            + f" perm_max_drift={max(drift.values()):.3g}", flush=True,
        )


if __name__ == "__main__":
    main()
