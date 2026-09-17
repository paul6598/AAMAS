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


# 의미적 행동 규칙을 현재 상태의 행동 인덱스에 대응시켜 제약과 선호를 만든다.
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

    # Semantic safety net.  Even individually valid rules can jointly remove
    # every attack (e.g. attack_type:Marine on a homogeneous map) or every
    # movement action.  Such category-wide prohibitions contradict the
    # intended coarse guidance role and do not trigger the controller's
    # all-actions-empty fallback.  A preference/prohibition collision is
    # resolved in favour of the non-binding preference.
    hard[(hard == 0.0) & (soft > 1.0)] = 1.0
    for i in range(n_agents):
        if not snap["allies"][i]["alive"]:
            continue
        attack_actions = list(range(6, n_actions))
        if attack_actions and not hard[i, attack_actions].any():
            hard[i, attack_actions] = 1.0
        move_actions = [a for a in range(2, min(6, n_actions))]
        if move_actions and not hard[i, move_actions].any():
            hard[i, move_actions] = 1.0
    return hard, soft
