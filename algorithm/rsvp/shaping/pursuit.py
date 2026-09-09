"""Pursuit predicate library for the RSVP critic and shaping channel.

Sub-goals are (predicate, sector) pairs; the sector rides in the guidance's
"unit_type" field so the shared library/head plumbing (predlib.head_index,
multi-head critic) works unchanged.

Delta predicates ("approach", "encircle") are SIGNED potential differences
Phi(post) - Phi(pre), so any closed loop of moves telescopes to zero and
oscillating back-and-forth cannot harvest shaping reward (Q-003 P3 check).
Potentials are only piecewise-telescoping: catches and sector migration of
evaders re-anchor the potential (documented limitation).
"""
import numpy as np

SIZE = 16
SECTORS = ("NW", "NE", "SW", "SE")
SECTOR_PREDICATES = ("approach", "encircle", "blockade")
GLOBAL_PREDICATES = ("catch", "tag_pressure")
ALL_PREDICATES = SECTOR_PREDICATES + GLOBAL_PREDICATES


def sector_of(cell):
    return ("N" if cell[0] < SIZE // 2 else "S") + ("W" if cell[1] < SIZE // 2 else "E")


def _sector_evaders(snap, sector):
    return [e for e in snap["evaders"] if sector_of(e) == sector]


def _min_cheb(cell, others):
    if not others:
        return SIZE
    return min(max(abs(cell[0] - o[0]), abs(cell[1] - o[1])) for o in others)


def _phi_approach(snap, sector):
    """-(mean over sector evaders of Chebyshev distance to nearest pursuer)/8."""
    evs = _sector_evaders(snap, sector)
    if not evs or not snap["pursuers"]:
        return 0.0
    d = [_min_cheb(e, snap["pursuers"]) for e in evs]
    return -float(np.mean(d)) / 8.0


def _phi_encircle(snap, sector):
    """Mean over sector evaders of the fraction of 4 quadrant directions
    around the evader that contain a pursuer within Chebyshev 3."""
    evs = _sector_evaders(snap, sector)
    if not evs or not snap["pursuers"]:
        return 0.0
    fracs = []
    for e in evs:
        dirs = set()
        for p in snap["pursuers"]:
            dx, dy = p[0] - e[0], p[1] - e[1]
            if max(abs(dx), abs(dy)) <= 3 and (dx, dy) != (0, 0):
                dirs.add(("N" if dx < 0 else "S") + ("W" if dy < 0 else "E"))
        fracs.append(len(dirs) / 4.0)
    return float(np.mean(fracs))


def evaluate_predicate(pred, sector, pre, post):
    if pred == "approach":
        return _phi_approach(post, sector) - _phi_approach(pre, sector)
    if pred == "encircle":
        return _phi_encircle(post, sector) - _phi_encircle(pre, sector)
    if pred == "blockade":
        return 0.3 if any(sector_of(p) == sector for p in post["pursuers"]) else 0.0
    if pred == "catch":
        return float(max(0, pre["remaining"] - post["remaining"]))
    if pred == "tag_pressure":
        evs = post["evaders"]
        if not evs:
            return 0.0
        return sum(1 for e in evs if _min_cheb(e, post["pursuers"]) <= 1) / len(evs)
    return 0.0


def compute_shaping(subgoals, pre, post, actions=None, clip=3.0):
    total = 0.0
    for sg in subgoals or []:
        w = float(sg.get("weight", 0.5))
        if w <= 0.0:
            continue
        total += min(1.0, w) * evaluate_predicate(
            sg.get("predicate"), sg.get("unit_type"), pre, post)
    return max(-clip, min(clip, total))
