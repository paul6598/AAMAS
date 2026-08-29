"""Compile Commander action rules into per-agent hard masks and soft weights.

Rules are symbolic, e.g.
    {"applies_to": "type:Marine", "forbid": ["move_north"],
     "prefer": ["attack_type:Medivac"], "prefer_weight": 2.0}
and are re-grounded against the current snapshot every step (Eq. 7), so a
token like 'attack_lowest_health' always resolves to the current target.

Returns:
    hard : (n_agents, n_actions) float32 in {0,1}, 1 = allowed
    soft : (n_agents, n_actions) float32 > 0, preference weight W_soft
"""
import numpy as np

MIN_W, MAX_W = 1.0, 5.0


def build_masks(rules, snap, iface, n_agents, n_actions):
    hard = np.ones((n_agents, n_actions), dtype=np.float32)
    soft = np.ones((n_agents, n_actions), dtype=np.float32)
    if not rules:
        return hard, soft

    for rule in rules:
        sel = rule.get("applies_to", "all")
        w = float(rule.get("prefer_weight", 2.0))
        w = max(MIN_W + 0.1, min(MAX_W, w))
        for i in range(n_agents):
            if not snap["allies"][i]["alive"]:
                continue
            if not iface.agent_matches(sel, i, snap):
                continue
            for token in rule.get("forbid", []) or []:
                for a in iface.resolve_action_token(token, i, snap):
                    hard[i, a] = 0.0
            for token in rule.get("prefer", []) or []:
                for a in iface.resolve_action_token(token, i, snap):
                    soft[i, a] = max(soft[i, a], w)
    return hard, soft
