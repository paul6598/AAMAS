"""Standalone unit tests for the Pursuit predicate library (Q-003 P3).

Run: python -m algorithm.rsvp.analysis.test_pursuit_pred
Covers sign correctness, the telescoping (anti-reward-hacking) property of
delta predicates, sanitisation, and library head resolution.
"""
import sys

from algorithm.rsvp.shaping.pursuit import evaluate_predicate, compute_shaping
from algorithm.rsvp.commander.pursuit import sanitize_pursuit
from algorithm.rsvp import predlib

FAIL = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        FAIL.append(name)


def snap(purs, evs):
    return {"pursuers": purs, "evaders": evs,
            "remaining": len(evs), "caught": 30 - len(evs)}


# --- approach: sign and telescoping -------------------------------------
A = snap([(2, 2)], [(6, 6)])          # NW evader, pursuer far
B = snap([(4, 4)], [(6, 6)])          # pursuer closer
fwd = evaluate_predicate("approach", "NW", A, B)
back = evaluate_predicate("approach", "NW", B, A)
check("approach closer > 0", fwd > 0)
check("approach away < 0", back < 0)
check("approach round-trip telescopes to 0", abs(fwd + back) < 1e-9)
loop = (evaluate_predicate("approach", "NW", A, B)
        + evaluate_predicate("approach", "NW", B, A)
        + evaluate_predicate("approach", "NW", A, B)
        + evaluate_predicate("approach", "NW", B, A))
check("approach 2-cycle oscillation harvests 0", abs(loop) < 1e-9)

# --- encircle ------------------------------------------------------------
one = snap([(4, 6)], [(6, 6)])                       # one quadrant covered
three = snap([(4, 4), (4, 8), (8, 4)], [(6, 6)])     # three quadrants
d = evaluate_predicate("encircle", "NW", one, three)
check("encircle more quadrants > 0", d > 0)
check("encircle round-trip telescopes",
      abs(d + evaluate_predicate("encircle", "NW", three, one)) < 1e-9)

# --- blockade / catch / tag_pressure ------------------------------------
check("blockade present = 0.3",
      evaluate_predicate("blockade", "SE", A, snap([(12, 12)], [(6, 6)])) == 0.3)
check("blockade absent = 0",
      evaluate_predicate("blockade", "SE", A, snap([(2, 2)], [(6, 6)])) == 0.0)
check("catch counts removals",
      evaluate_predicate("catch", None, snap([], [(1, 1), (2, 2)]), snap([], [(1, 1)])) == 1.0)
check("tag_pressure adjacency fraction",
      abs(evaluate_predicate("tag_pressure", None, A,
                             snap([(6, 7)], [(6, 6), (12, 12)])) - 0.5) < 1e-9)

# --- shaping composition -------------------------------------------------
sgs = [{"predicate": "approach", "unit_type": "NW", "weight": 1.0},
       {"predicate": "catch", "unit_type": None, "weight": 0.5}]
val = compute_shaping(sgs, A, B)
check("compute_shaping = weighted sum", abs(val - fwd) < 1e-9)

# --- sanitisation --------------------------------------------------------
g = sanitize_pursuit({"strategy": "s", "subgoals": [
    {"predicate": "encircle", "sector": "NW", "weight": 0.9},
    {"predicate": "catch", "sector": "NE", "weight": 0.7},      # sector ignored
    {"predicate": "approach", "weight": 0.5},                    # missing sector -> drop
    {"predicate": "bogus", "sector": "NW", "weight": 0.5}]})     # unknown -> drop
check("sanitize keeps 2 valid subgoals", g is not None and len(g["subgoals"]) == 2)
check("sanitize maps sector -> unit_type", g["subgoals"][0]["unit_type"] == "NW")
check("sanitize global predicate sector None", g["subgoals"][1]["unit_type"] is None)

# --- Q-005 regression: stacking must not fake a catch --------------------
import numpy as np
from env.semantic.pursuit import grid_entities

grid = np.zeros((16, 16, 3), dtype=np.float32)
grid[6, 6, 2] = 2.0   # two evaders stacked on one cell
grid[2, 2, 2] = 1.0
grid[4, 4, 1] = 1.0
check("grid_entities counts stacked entities", len(grid_entities(grid, 2)) == 3)
pre_stack = snap([(4, 4)], grid_entities(grid, 2))
g2 = grid.copy(); g2[6, 6, 2] = 1.0; g2[7, 6, 2] = 1.0  # unstack, same 3 entities
post_stack = snap([(4, 4)], grid_entities(g2, 2))
check("stack/unstack fakes no catch",
      evaluate_predicate("catch", None, pre_stack, post_stack) == 0.0
      and evaluate_predicate("catch", None, post_stack, pre_stack) == 0.0)

# --- library head resolution --------------------------------------------
lib = predlib.build_library("pursuit", None)
check("library size 14", len(lib) == 14)
idxs = [predlib.head_index(lib, sg) for sg in g["subgoals"]]
check("sanitized subgoals resolve to heads", all(i is not None for i in idxs))

print("\n%d failures" % len(FAIL))
sys.exit(1 if FAIL else 0)
