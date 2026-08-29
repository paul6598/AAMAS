"""Grounded reward predicates f_j(s_t, a_t, s_{t+1}) for semantic reward shaping.

The Commander outputs sub-goals as (predicate, weight[, unit_type]) tuples;
this module maps them onto measurable events between two snapshots:
    F_t = sum_j w_j * f_j(pre, post, actions)          (Eq. 1 in the paper)

All predicate values are scaled so that typical per-step magnitudes are <= 1,
keeping F_t comparable to SMAC's (scaled) environment reward.
"""

N_BASE_ACTIONS = 6
MOVE_ACTIONS = (2, 3, 4, 5)

# Predicates that take a 'unit_type' parameter.
TYPED_PREDICATES = ("kill_type", "damage_type", "protect_type")
SIMPLE_PREDICATES = ("enemy_kill", "enemy_damage", "ally_survive",
                     "focus_fire", "retreat_low_health")
ALL_PREDICATES = SIMPLE_PREDICATES + TYPED_PREDICATES


def _new_deaths(pre_units, post_units, type_name=None):
    n = 0
    for pu, qu in zip(pre_units, post_units):
        if pu["alive"] and not qu["alive"]:
            if type_name is None or pu["type"].lower() == type_name:
                n += 1
    return n


def _damage(pre_units, post_units, type_name=None):
    """(damage dealt, total max hp) over the matching units."""
    dmg, tot = 0.0, 0.0
    for pu, qu in zip(pre_units, post_units):
        if type_name is not None and pu["type"].lower() != type_name:
            continue
        tot += pu["hp_max"]
        dmg += max(0.0, pu["hp"] - qu["hp"])
    return dmg, max(tot, 1e-6)


def evaluate_predicate(pred, unit_type, pre, post, actions):
    tname = unit_type.lower() if unit_type else None

    if pred == "enemy_kill":
        return 1.0 * _new_deaths(pre["enemies"], post["enemies"])

    if pred == "enemy_damage":
        dmg, tot = _damage(pre["enemies"], post["enemies"])
        return 10.0 * dmg / tot  # killing the whole enemy army ~ 10

    if pred == "ally_survive":
        return -1.0 * _new_deaths(pre["allies"], post["allies"])

    if pred == "kill_type":
        return 1.5 * _new_deaths(pre["enemies"], post["enemies"], tname)

    if pred == "damage_type":
        dmg, tot = _damage(pre["enemies"], post["enemies"], tname)
        return 5.0 * dmg / tot

    if pred == "protect_type":
        deaths = _new_deaths(pre["allies"], post["allies"], tname)
        dmg, tot = _damage(pre["allies"], post["allies"], tname)
        return -1.5 * deaths - 3.0 * dmg / tot

    if pred == "focus_fire":
        # Fraction of attacking agents that hit the most-targeted enemy.
        targets = []
        for i, a in enumerate(actions):
            if a >= N_BASE_ACTIONS and pre["allies"][i]["alive"] \
                    and pre["allies"][i]["type"] != "Medivac":
                targets.append(a - N_BASE_ACTIONS)
        if len(targets) < 2:
            return 0.0
        best = max(targets.count(t) for t in set(targets))
        frac = best / len(targets)
        return 0.3 * frac if frac > 0.5 else 0.0

    if pred == "retreat_low_health":
        # Reward only movement that increases distance to the enemy centroid.
        alive_e = [e for e in pre["enemies"] if e["alive"]]
        if not alive_e:
            return 0.0
        cx = sum(e["x"] for e in alive_e) / len(alive_e)
        cy = sum(e["y"] for e in alive_e) / len(alive_e)
        v = 0.0
        for i, a in enumerate(actions):
            u, q = pre["allies"][i], post["allies"][i]
            if u["alive"] and u["hp"] / u["hp_max"] < 0.3 and a in MOVE_ACTIONS:
                d_pre = (u["x"] - cx) ** 2 + (u["y"] - cy) ** 2
                d_post = (q["x"] - cx) ** 2 + (q["y"] - cy) ** 2
                if d_post > d_pre:
                    v += 0.1
        return v

    return 0.0


def compute_shaping(subgoals, pre, post, actions, clip=3.0):
    """Weighted sum of active sub-goal predicates, clipped for stability."""
    total = 0.0
    for sg in subgoals:
        pred = sg.get("predicate")
        w = float(sg.get("weight", 0.0))
        if w <= 0.0:
            continue
        val = evaluate_predicate(pred, sg.get("unit_type"), pre, post, actions)
        total += min(1.0, w) * val
    return max(-clip, min(clip, total))
