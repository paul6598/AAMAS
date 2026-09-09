"""Simple positive-direction shaping control for SMAC diagnostics."""
from .base import Commander


class AlignedCommander(Commander):
    """Reward enemy damage and kills only; no masks or state-dependent text."""
    def __call__(self, summary, cache_key, iface):
        return {
            "strategy": "diagnostic: reinforce enemy damage and kills",
            "subgoals": [
                {"predicate": "enemy_damage", "weight": 1.0},
                {"predicate": "enemy_kill", "weight": 0.5},
            ],
            "action_rules": [],
        }
