"""Print a human-readable Stage-0 event-extractor input/target/output example.

This is an inspection aid, not an event-quality benchmark.  Unless a trained checkpoint is added,
the numeric heads are randomly initialized; the script makes the tensor interface observable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.peel.model import PEELEventEncoder
from algorithms.peel.qwen import QwenPEELEventEncoder
from algorithms.peel.transition import lbf_transition
from envs import make_env
from experiments.audit_raw_llm_permutations import CASES


def _top(probabilities, prefix):
    index = int(np.argmax(probabilities))
    return {"index": index, "tag": f"{prefix}{index}", "probability": round(float(probabilities[index]), 6)}


def inspect(model, record, rule=None):
    device = next(model.parameters()).device
    agent = torch.as_tensor(record.agent_features, dtype=torch.float32, device=device)[None]
    obj = torch.as_tensor(record.object_features, dtype=torch.float32, device=device)[None]
    if isinstance(model, QwenPEELEventEncoder):
        agent, obj = agent.to(torch.bfloat16), obj.to(torch.bfloat16)
        output = model(agent, obj, rule)
    else:
        output = model(agent, obj)
    agent_prob = torch.softmax(output["agent_pointer_logits"], dim=-1)[0].float().cpu().numpy()
    object_prob = torch.softmax(output["object_pointer_logits"], dim=-1)[0].float().cpu().numpy()
    events = []
    for slot in range(agent_prob.shape[0]):
        events.append({
            "slot": slot,
            "embedding_l2_norm": round(float(output["event_slots"][0, slot].float().norm().cpu()), 6),
            "impact": round(float(output["event_impact"][0, slot].float().cpu()), 6),
            "top_agent": _top(agent_prob[slot], "agent_"),
            "top_object": _top(object_prob[slot], "object_"),
        })
    return {
        "predicted_team_reward": round(float(output["reward"][0].float().cpu()), 6),
        "event_slots": events,
        "predicted_agent_delta": np.round(output["agent_delta"][0].float().cpu().numpy(), 5).tolist(),
        "predicted_object_delta": np.round(output["object_delta"][0].float().cpu().numpy(), 5).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("native", "qwen"), default="native")
    parser.add_argument("--case", choices=sorted(CASES), default="delayed")
    parser.add_argument("--timestep", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trajectory = CASES[args.case]()
    timestep = len(trajectory) - 1 if args.timestep < 0 else args.timestep
    record = lbf_transition(trajectory, timestep)
    env = make_env("Foraging-5x5-3p-2f-coop-v3", seed=0, max_episode_steps=6)
    if args.backbone == "qwen":
        model = QwenPEELEventEncoder().cuda().to(dtype=torch.bfloat16).eval()
        rule = model.rule_tokens(env.describe()["env"])
    else:
        torch.manual_seed(0)
        model = PEELEventEncoder(
            agent_dim=record.agent_features.shape[-1], object_dim=record.object_features.shape[-1],
        ).eval()
        rule = None
    with torch.no_grad():
        output = inspect(model, record, rule)
    report = {
        "warning": "No trained semantic-event checkpoint is loaded. Predictions show the interface, not event quality.",
        "case": args.case,
        "timestep": timestep,
        "rule_text": env.describe()["env"] if args.backbone == "qwen" else None,
        "model_input": {
            "agent_features_[row,col,level,action]": record.agent_features.tolist(),
            "object_features_[row,col,level,active]": record.object_features.tolist(),
        },
        "supervision_target": {
            "agent_delta": record.agent_delta.tolist(),
            "object_delta": record.object_delta.tolist(),
            "global_reward": record.reward,
            "done": record.done,
        },
        "model_output": output,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
