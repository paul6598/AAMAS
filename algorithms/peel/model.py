"""A compact exact-symmetry event encoder.

This is the architecture reference implementation for Stage 1, not yet the pretrained LLM
adapter.  It establishes the structured interface, relational mask semantics, parallel event
slots, and numerical equivariance tests before changing a large causal LLM's RoPE/attention.
"""
import math

import torch
from torch import nn


class PEELAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim))

    def forward(self, x, allowed):
        """Attention with a batch-specific structural adjacency ``allowed`` [B,L,L]."""
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        score = torch.einsum("blhd,bmhd->bhlm", query, key) / math.sqrt(self.head_dim)
        score = score.masked_fill(~allowed[:, None], torch.finfo(score.dtype).min)
        weights = torch.softmax(score, dim=-1)
        attended = torch.einsum("bhlm,bmhd->blhd", weights, value).reshape(batch, length, dim)
        x = x + self.out(attended)
        return x + self.ff(self.norm2(x))


class PEELEventEncoder(nn.Module):
    """Event-slot encoder with invariant summaries and equivariant entity pointers.

    No input position or ID embedding exists.  Relation masks are made only from entity class
    and valid-set membership.  The Stage-0 default uses complete structured-set attention: every
    agent, object, and event slot can directly read every other structured node.  This is the
    simplest useful symmetry-safe baseline; sparse environment-relation masks are ablations.
    """
    def __init__(self, agent_dim=4, object_dim=4, dim=96, event_slots=4, layers=3, heads=4):
        super().__init__()
        self.agent_proj = nn.Linear(agent_dim, dim)
        self.object_proj = nn.Linear(object_dim, dim)
        self.type_embedding = nn.Embedding(3, dim)  # agent, object, event slot only
        self.event_queries = nn.Parameter(torch.empty(event_slots, dim))
        nn.init.normal_(self.event_queries, std=0.02)
        self.blocks = nn.ModuleList([PEELAttentionBlock(dim, heads) for _ in range(layers)])
        self.event_norm = nn.LayerNorm(dim)
        self.agent_pointer = nn.Linear(dim, dim, bias=False)
        self.object_pointer = nn.Linear(dim, dim, bias=False)
        self.impact_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.reward_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.agent_delta_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, agent_dim))
        self.object_delta_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, object_dim))

    def _allowed_mask(self, agent_features, object_features, n_events):
        """Complete attention over the unordered structured set.

        There is deliberately no geometric radius, active-object gate, or entity-index rule here.
        A full graph commutes with every agent/object reindexing and avoids imposing an untested
        locality bottleneck on the global critic.
        """
        batch, n_agents, _ = agent_features.shape
        n_objects = object_features.shape[1]
        length = n_agents + n_objects + n_events
        return torch.ones((batch, length, length), dtype=torch.bool, device=agent_features.device)

    def forward(self, agent_features, object_features):
        """Inputs [B,N,D_a], [B,M,D_o]; output agent/object pointers preserve their axes."""
        batch, n_agents, _ = agent_features.shape
        _, n_objects, _ = object_features.shape
        agent = self.agent_proj(agent_features) + self.type_embedding.weight[0]
        obj = self.object_proj(object_features) + self.type_embedding.weight[1]
        event = self.event_queries[None].expand(batch, -1, -1) + self.type_embedding.weight[2]
        x = torch.cat([agent, obj, event], dim=1)
        allowed = self._allowed_mask(agent_features, object_features, event.shape[1])
        for block in self.blocks:
            x = block(x, allowed)
        agents = x[:, :n_agents]
        objects = x[:, n_agents:n_agents + n_objects]
        events = self.event_norm(x[:, n_agents + n_objects:])
        # [B,K,N] and [B,K,M]; a reindexing only reorders the final axes.
        agent_logits = torch.einsum("bkd,bnd->bkn", self.agent_pointer(events), agents)
        object_logits = torch.einsum("bkd,bmd->bkm", self.object_pointer(events), objects)
        global_event = events.mean(dim=1)  # invariant over exchangeable event slots
        global_to_agent = global_event[:, None].expand(-1, n_agents, -1)
        global_to_object = global_event[:, None].expand(-1, n_objects, -1)
        return {
            "event_slots": events,
            "agent_pointer_logits": agent_logits,
            "object_pointer_logits": object_logits,
            "event_impact": self.impact_head(events).squeeze(-1),
            "reward": self.reward_head(global_event).squeeze(-1),
            "agent_delta": self.agent_delta_head(torch.cat([agents, global_to_agent], dim=-1)),
            "object_delta": self.object_delta_head(torch.cat([objects, global_to_object], dim=-1)),
        }
