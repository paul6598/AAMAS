#!/usr/bin/env python3
"""Small counterfactual audit of LEHCA's executable shaping objective.

This is not an environment simulation.  It feeds controlled state transitions
to the same predicate implementation used in training, making reward-design
trade-offs explicit without consuming GPU or LLM capacity.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm.lehca.shaping.predicates import compute_shaping


GUIDANCE = [
    {"predicate": "enemy_kill", "weight": 0.9},
    {"predicate": "enemy_damage", "weight": 0.7},
    {"predicate": "ally_survive", "weight": 0.8},
    {"predicate": "focus_fire", "weight": 0.6},
    {"predicate": "retreat_low_health", "weight": 0.5},
]


def unit(x: float, y: float, hp: float = 45.0) -> dict:
    return {
        "alive": hp > 0,
        "hp": max(hp, 0.0),
        "hp_max": 45.0,
        "type": "Marine",
        "x": x,
        "y": y,
    }


def base_state() -> dict:
    return {
        "allies": [unit(float(i), 0.0) for i in range(5)],
        "enemies": [unit(10.0 + float(i), 0.0) for i in range(6)],
    }


def score(pre: dict, post: dict, actions: list[int]) -> float:
    return compute_shaping(GUIDANCE, pre, post, actions)


def scenarios() -> dict[str, float]:
    pre = base_state()
    attack_same_target = [6, 6, 6, 6, 6]
    idle = [1, 1, 1, 1, 1]
    out = {}

    out["idle_no_transition"] = score(pre, copy.deepcopy(pre), idle)
    out["focus_command_without_damage"] = score(
        pre, copy.deepcopy(pre), attack_same_target
    )

    post = copy.deepcopy(pre)
    post["enemies"][0]["hp"] -= 30.0
    out["focus_plus_30_enemy_damage"] = score(pre, post, attack_same_target)

    post = copy.deepcopy(pre)
    post["enemies"][0]["hp"] = 0.0
    post["enemies"][0]["alive"] = False
    out["focus_plus_one_enemy_kill"] = score(pre, post, attack_same_target)

    post = copy.deepcopy(pre)
    post["allies"][0]["hp"] = 0.0
    post["allies"][0]["alive"] = False
    out["focus_while_one_ally_dies"] = score(pre, post, attack_same_target)

    low_pre = copy.deepcopy(pre)
    low_pre["allies"][0]["hp"] = 10.0
    low_post = copy.deepcopy(low_pre)
    low_post["allies"][0]["x"] -= 2.0
    out["one_low_health_ally_retreats"] = score(
        low_pre, low_post, [5, 1, 1, 1, 1]
    )
    return out


def main() -> None:
    values = scenarios()
    values["retreat_steps_to_offset_one_ally_death"] = (
        abs(values["focus_while_one_ally_dies"] - values["focus_command_without_damage"])
        / values["one_low_health_ally_retreats"]
    )
    print(json.dumps({"guidance": GUIDANCE, "shaping": values}, indent=2))


if __name__ == "__main__":
    main()
