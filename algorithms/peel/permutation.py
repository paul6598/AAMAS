"""Permutation audits shared by event and later LLM critic models."""
import numpy as np
import torch

from .transition import permute_record


def _tensor(record, device):
    return (
        torch.as_tensor(record.agent_features, dtype=torch.float32, device=device)[None],
        torch.as_tensor(record.object_features, dtype=torch.float32, device=device)[None],
    )


@torch.no_grad()
def audit_event_equivariance(model, record, agent_permutation, object_permutation):
    """Return numerical invariant/equivariant drift after inverse-mapping output axes."""
    device = next(model.parameters()).device
    model.eval()
    original = model(*_tensor(record, device))
    permuted = model(*_tensor(permute_record(record, agent_permutation, object_permutation), device))
    inverse_agents = torch.as_tensor(np.argsort(agent_permutation), device=device)
    inverse_objects = torch.as_tensor(np.argsort(object_permutation), device=device)
    agent_delta = permuted["agent_delta"][:, inverse_agents]
    object_delta = permuted["object_delta"][:, inverse_objects]
    agent_pointer = permuted["agent_pointer_logits"][:, :, inverse_agents]
    object_pointer = permuted["object_pointer_logits"][:, :, inverse_objects]
    agent_pointer_prob = torch.softmax(permuted["agent_pointer_logits"], dim=-1)[:, :, inverse_agents]
    object_pointer_prob = torch.softmax(permuted["object_pointer_logits"], dim=-1)[:, :, inverse_objects]
    original_agent_prob = torch.softmax(original["agent_pointer_logits"], dim=-1)
    original_object_prob = torch.softmax(original["object_pointer_logits"], dim=-1)
    raw_pointer_scale = max(
        float(original["agent_pointer_logits"].abs().max().cpu()),
        float(original["object_pointer_logits"].abs().max().cpu()),
        1e-12,
    )
    return {
        "reward_drift": float((original["reward"] - permuted["reward"]).abs().max().cpu()),
        "event_impact_drift": float((original["event_impact"] - permuted["event_impact"]).abs().max().cpu()),
        "agent_delta_drift": float((original["agent_delta"] - agent_delta).abs().max().cpu()),
        "object_delta_drift": float((original["object_delta"] - object_delta).abs().max().cpu()),
        "agent_pointer_drift": float((original["agent_pointer_logits"] - agent_pointer).abs().max().cpu()),
        "object_pointer_drift": float((original["object_pointer_logits"] - object_pointer).abs().max().cpu()),
        "agent_pointer_prob_drift": float((original_agent_prob - agent_pointer_prob).abs().max().cpu()),
        "object_pointer_prob_drift": float((original_object_prob - object_pointer_prob).abs().max().cpu()),
        "pointer_logit_relative_drift": max(
            float((original["agent_pointer_logits"] - agent_pointer).abs().max().cpu()),
            float((original["object_pointer_logits"] - object_pointer).abs().max().cpu()),
        ) / raw_pointer_scale,
    }
