"""Qwen backbone adapter for the PEEL event interface.

Unlike prompt-and-generate MCA, this calls the pretrained causal model's forward pass with
structured embeddings, custom RoPE positions, and a relation mask.  Rule text retains ordinary
language order; homogeneous agent/object nodes do not receive serialization-order positions.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Minimal dependency-free LoRA wrapper for Qwen attention projections."""
    def __init__(self, base, rank=4, alpha=8.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA can wrap nn.Linear projections only")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_a), self.lora_b)


class QwenPEELEventEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", agent_dim=4, object_dim=4,
                 event_slots=4, freeze_backbone=True, lora_rank=0, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, dtype=dtype)
        hidden = self.backbone.config.hidden_size
        self.agent_proj = nn.Linear(agent_dim, hidden, bias=False)
        self.object_proj = nn.Linear(object_dim, hidden, bias=False)
        self.type_embedding = nn.Embedding(3, hidden)
        self.event_queries = nn.Parameter(torch.empty(event_slots, hidden))
        nn.init.normal_(self.event_queries, std=0.02)
        self.norm = nn.LayerNorm(hidden)
        self.agent_pointer = nn.Linear(hidden, hidden, bias=False)
        self.object_pointer = nn.Linear(hidden, hidden, bias=False)
        # Bounded cosine pointer logits prevent bf16 accumulation noise in a 3584-dimensional
        # raw dot product from being amplified into a discontinuous participant assignment.
        self.pointer_temperature = 0.2
        self.impact_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.reward_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.agent_delta_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                              nn.Linear(hidden, agent_dim))
        self.object_delta_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                               nn.Linear(hidden, object_dim))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        if lora_rank:
            self.enable_lora(lora_rank)

    def enable_lora(self, rank=4, alpha=8.0):
        """Adapt Q/K/V/O in every attention layer while retaining frozen pretrained weights."""
        replaced = 0
        for layer in self.backbone.layers:
            attention = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(attention, name)
                if not isinstance(module, LoRALinear):
                    setattr(attention, name, LoRALinear(module, rank=rank, alpha=alpha))
                    replaced += 1
        return replaced

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def device(self):
        return self.backbone.device

    def rule_tokens(self, text, device=None):
        device = device or self.device
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    @staticmethod
    def _mask(agent_features, object_features, text_len, n_events, dtype):
        """Return the Qwen ``full_attention`` mask as a dict, bypassing causal-mask creation.

        Text-text edges remain causal (valid language order).  Every structured entity/event
        node reads the full rule text and the complete unordered structured set.  No edge refers
        to an agent/object serialized index or to a geometric cutoff: locality is an ablation,
        not a prerequisite for permutation symmetry.
        """
        batch, n_agents, _ = agent_features.shape
        n_objects = object_features.shape[1]
        device = agent_features.device
        length = text_len + n_agents + n_objects + n_events
        allowed = torch.zeros((batch, length, length), dtype=torch.bool, device=device)
        if text_len:
            text_causal = torch.tril(torch.ones((text_len, text_len), dtype=torch.bool, device=device))
            allowed[:, :text_len, :text_len] = text_causal
        structured = text_len
        # Structured nodes read all rule tokens and all structured nodes.  The full structured
        # graph is invariant to any reserialization of agent/object rows.
        allowed[:, structured:, :text_len] = True
        allowed[:, structured:, structured:] = True
        additive = torch.zeros((batch, 1, length, length), dtype=dtype, device=device)
        additive.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        return {"full_attention": additive}

    def forward(self, agent_features, object_features, rule_input_ids=None):
        batch, n_agents, _ = agent_features.shape
        _, n_objects, _ = object_features.shape
        device = self.device
        agent_features = agent_features.to(device=device, dtype=self.agent_proj.weight.dtype)
        object_features = object_features.to(device=device, dtype=self.object_proj.weight.dtype)
        if rule_input_ids is None:
            rule_input_ids = torch.empty((batch, 0), dtype=torch.long, device=device)
        else:
            rule_input_ids = rule_input_ids.to(device)
            if rule_input_ids.shape[0] == 1 and batch > 1:
                rule_input_ids = rule_input_ids.expand(batch, -1)
        text_len = rule_input_ids.shape[1]
        text = (self.backbone.embed_tokens(rule_input_ids) if text_len else
                torch.empty((batch, 0, self.backbone.config.hidden_size), dtype=self.agent_proj.weight.dtype,
                            device=device))
        agent = self.agent_proj(agent_features) + self.type_embedding.weight[0].to(agent_features.dtype)
        obj = self.object_proj(object_features) + self.type_embedding.weight[1].to(agent_features.dtype)
        event = self.event_queries[None].expand(batch, -1, -1).to(agent_features.dtype)
        event = event + self.type_embedding.weight[2].to(agent_features.dtype)
        embeds = torch.cat([text, agent, obj, event], dim=1)
        # Rule text occupies 0..L-1; all same-time entities share L and event queries share L+1.
        positions = torch.cat([
            torch.arange(text_len, device=device),
            torch.full((n_agents + n_objects,), text_len, device=device),
            torch.full((event.shape[1],), text_len + 1, device=device),
        ])[None].expand(batch, -1)
        hidden = self.backbone(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=self._mask(agent_features, object_features, text_len, event.shape[1], embeds.dtype),
            use_cache=False,
        ).last_hidden_state
        agents = hidden[:, text_len:text_len + n_agents]
        objects = hidden[:, text_len + n_agents:text_len + n_agents + n_objects]
        events = self.norm(hidden[:, text_len + n_agents + n_objects:])
        event_agent = F.normalize(self.agent_pointer(events), dim=-1)
        event_object = F.normalize(self.object_pointer(events), dim=-1)

        agent_logits = torch.einsum("bkd,bnd->bkn", event_agent, F.normalize(agents, dim=-1))
        object_logits = torch.einsum("bkd,bmd->bkm", event_object, F.normalize(objects, dim=-1))

        agent_logits = agent_logits / self.pointer_temperature
        object_logits = object_logits / self.pointer_temperature
        
        global_event = events.mean(dim=1)
        agent_global = global_event[:, None].expand(-1, n_agents, -1)
        object_global = global_event[:, None].expand(-1, n_objects, -1)
        return {
            "event_slots": events,
            "agent_pointer_logits": agent_logits,
            "object_pointer_logits": object_logits,
            "event_impact": self.impact_head(events).squeeze(-1),
            "reward": self.reward_head(global_event).squeeze(-1),
            "agent_delta": self.agent_delta_head(torch.cat([agents, agent_global], dim=-1)),
            "object_delta": self.object_delta_head(torch.cat([objects, object_global], dim=-1)),
        }
