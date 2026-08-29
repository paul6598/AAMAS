"""Rule-Commander: handcrafted tactical heuristics producing sub-goals and
masks from the same semantic snapshot (the 'Rule-Commander + QMIX'
LLM-isolation control in the paper). Also used as the t=0 bootstrap and
offline fallback when no LLM server is reachable."""
from .base import Commander

# Enemy types worth prioritizing, in rough threat order.
PRIORITY_TARGETS = ["Medivac", "Marauder", "Baneling", "Colossus",
                    "Hydralisk", "Marine", "Stalker", "Zealot", "Zergling"]


class RuleCommander(Commander):
    def __call__(self, summary, cache_key, iface):
        snap = iface.snapshot()
        enemy_types = {e["type"] for e in snap["enemies"] if e["alive"]}
        ally_types = {a["type"] for a in snap["allies"] if a["alive"]}

        subgoals = [
            {"predicate": "enemy_kill", "weight": 0.8},
            {"predicate": "enemy_damage", "weight": 0.6},
            {"predicate": "focus_fire", "weight": 0.7},
            {"predicate": "ally_survive", "weight": 0.5},
        ]
        rules = [{"applies_to": "all", "forbid": [],
                  "prefer": ["attack_lowest_health"], "prefer_weight": 2.0}]

        # Prioritize the highest-threat enemy type present (if >1 type).
        if len(enemy_types) > 1:
            for t in PRIORITY_TARGETS:
                if t in enemy_types:
                    subgoals.append({"predicate": "kill_type", "weight": 0.6,
                                     "unit_type": t})
                    rules.append({"applies_to": "all", "forbid": [],
                                  "prefer": ["attack_type:%s" % t],
                                  "prefer_weight": 1.5})
                    break
        if "Medivac" in ally_types:
            subgoals.append({"predicate": "protect_type", "weight": 0.6,
                             "unit_type": "Medivac"})

        return {"strategy": "heuristic: focus fire, finish low-health targets, "
                            "prioritize threats, preserve allies",
                "subgoals": subgoals[:6], "action_rules": rules}
