"""Environment dispatch for the predicate library used by the scheduler critic.

Provides, per env:
  build_library(env_name, snap) -> list of sub-goal keys [(predicate, unit_type)]
  f_vector(env_name, lib, pre, post, actions) -> np.array over the library
  features(env_name, snap) -> np.array state features x(s) (critic input)
  shaping(env_name, subgoals, pre, post, actions, clip) -> weighted F_t
  head_index(lib, subgoal) -> index of a guidance sub-goal in the library (or None)
"""
import math

import numpy as np

from algorithm.lehca.shaping.predicates import (SIMPLE_PREDICATES, TYPED_PREDICATES,
                                                evaluate_predicate as sc2_pred,
                                                compute_shaping as sc2_shaping)
from algorithm.vigil.shaping.grf import (ALL_PREDICATES as GRF_PREDICATES,
                                              evaluate_predicate as grf_pred,
                                              compute_shaping as grf_shaping)


def build_library(env_name, snap):
    if env_name == "gfootball":
        return [(p, None) for p in GRF_PREDICATES]
    ally_types = sorted({u["type"] for u in snap["allies"]})
    enemy_types = sorted({u["type"] for u in snap["enemies"]})
    lib = [(p, None) for p in SIMPLE_PREDICATES]
    for p in ("kill_type", "damage_type"):
        lib += [(p, t) for t in enemy_types]
    lib += [("protect_type", t) for t in ally_types]
    return lib


def head_index(lib, sg):
    key = (sg.get("predicate"), sg.get("unit_type"))
    try:
        return lib.index(key)
    except ValueError:
        return None


def f_vector(env_name, lib, pre, post, actions):
    if env_name == "gfootball":
        return np.array([grf_pred(p, pre, post, actions) for p, _ in lib], dtype=np.float32)
    return np.array([sc2_pred(p, t, pre, post, actions or []) for p, t in lib], dtype=np.float32)


def shaping(env_name, subgoals, pre, post, actions, clip=3.0):
    if env_name == "gfootball":
        return grf_shaping(subgoals, pre, post, actions, clip)
    return sc2_shaping(subgoals or [], pre, post, actions, clip)


# ------------------------------------------------------------------ features
def _grf_features(snap):
    poss = {"ours": 0, "loose": 1, "theirs": 2}[snap["possession"]]
    onehot = [0.0, 0.0, 0.0]
    onehot[poss] = 1.0
    sh = snap["shape"]
    n_field = max(1, len(snap.get("controlled", [])) or 4)
    n_opp = max(1, len(snap["enemies"]) - 1)
    return np.array(onehot + [
        snap["ball"]["x"], snap["ball"]["y"], snap["ball"]["dx"] * 50, snap["ball"]["dy"] * 50,
        sh["ours_behind_ball"] / n_field, sh["theirs_in_our_third"] / n_opp,
        min(1.0, sh["nearest_enemy_to_ball"] * 5), min(1.0, sh["nearest_ally_to_ball"] * 5),
        1.0 if snap["game_mode"] != "normal" else 0.0, snap["time_frac"],
    ], dtype=np.float32)


def _profile(units, types, visible_only=False):
    out = []
    for t in types:
        sel = [u for u in units if u["type"] == t and (not visible_only or u.get("visible", True))]
        alive = [u for u in sel if u["alive"]]
        n = max(1, len(sel))
        hp = sum(u["hp"] for u in alive) / max(1e-6, sum(u["hp_max"] for u in sel))
        out += [len(alive) / n, hp]
    return out


def _sc2_features(snap, ally_types, enemy_types):
    a = _profile(snap["allies"], ally_types)
    e = _profile(snap["enemies"], enemy_types, visible_only=True)
    ca = [(u["x"], u["y"]) for u in snap["allies"] if u["alive"]]
    ce = [(u["x"], u["y"]) for u in snap["enemies"] if u["alive"] and u.get("visible", True)]
    if ca and ce:
        ax = sum(p[0] for p in ca) / len(ca)
        ay = sum(p[1] for p in ca) / len(ca)
        ex = sum(p[0] for p in ce) / len(ce)
        ey = sum(p[1] for p in ce) / len(ce)
        d = math.hypot(ax - ex, ay - ey)
        dist, engaged, seen = min(1.0, d / 20.0), 1.0 if d < 8 else 0.0, 1.0
    else:
        dist, engaged, seen = 1.0, 0.0, 0.0
    return np.array(a + e + [dist, engaged, seen], dtype=np.float32)


class FeatureExtractor:
    def __init__(self, env_name, snap):
        self.env_name = env_name
        if env_name != "gfootball":
            self.ally_types = sorted({u["type"] for u in snap["allies"]})
            self.enemy_types = sorted({u["type"] for u in snap["enemies"]})

    def __call__(self, snap):
        if self.env_name == "gfootball":
            return _grf_features(snap)
        return _sc2_features(snap, self.ally_types, self.enemy_types)
