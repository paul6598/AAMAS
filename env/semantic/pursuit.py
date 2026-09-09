"""Semantic interface for PettingZoo Pursuit (16x16 grid, 8 pursuers, evaders).

Reads the wrapper's last global state grid (env.pettingzoo_pursuit.PursuitEnv
.last_grid, channels: 0=obstacles, 1=pursuers, 2=evaders) and summarises it
by 2x2 quadrant sectors (NW/NE/SW/SE). No masking vocabulary: the pursuit
arms are shaping-only.
"""
import numpy as np

from .base import SemanticInterface

# duplicated from algorithm.rsvp.shaping.pursuit to avoid a circular import
# (env.semantic must not depend on the algorithm package)
SIZE = 16
SECTORS = ("NW", "NE", "SW", "SE")


def sector_of(cell):
    return ("N" if cell[0] < SIZE // 2 else "S") + ("W" if cell[1] < SIZE // 2 else "E")

_BUCKETS = (0, 1, 5, 10)  # evader-count buckets per sector


def _bucket(n):
    b = 0
    for i, lo in enumerate(_BUCKETS):
        if n >= lo:
            b = i
    return b


def grid_entities(grid, ch):
    """Cell positions expanded by per-cell ENTITY COUNT (PettingZoo state cells
    hold counts; stacked entities must not be collapsed into one — Q-005 fix:
    occupied-cell counting made overlaps look like captures / fake `catch`)."""
    out = []
    for c in np.argwhere(grid[:, :, ch] > 0):
        out.extend([tuple(c)] * int(round(float(grid[c[0], c[1], ch]))))
    return out


class PursuitSemanticInterface(SemanticInterface):

    def snapshot(self):
        g = np.asarray(self.env.last_grid)
        pursuers = grid_entities(g, 1)
        evaders = grid_entities(g, 2)
        sectors = {s: {"evaders": 0, "pursuers": 0} for s in SECTORS}
        for e in evaders:
            sectors[sector_of(e)]["evaders"] += 1
        for p in pursuers:
            sectors[sector_of(p)]["pursuers"] += 1
        n0 = getattr(self.env, "n_evaders0", 30)
        return {"pursuers": pursuers, "evaders": evaders, "sectors": sectors,
                "remaining": len(evaders), "caught": n0 - len(evaders),
                "t_frac": min(1.0, self.env.t_now / max(1, self.env.episode_limit)),
                "n_actions": 5}

    def summary(self, snap, extra_stats=None):
        sec = ", ".join(f"{s}: {v['evaders']} evaders / {v['pursuers']} pursuers"
                        for s, v in snap["sectors"].items())
        lines = [f"Grid 16x16 pursuit. Evaders remaining: {snap['remaining']}, "
                 f"caught so far: {snap['caught']}.",
                 f"Sector occupancy - {sec}.",
                 f"Episode progress: {int(100 * snap['t_frac'])}%."]
        if extra_stats and "rolling_win_rate" in extra_stats:
            lines.append(f"Recent capture-all rate: {extra_stats['rolling_win_rate']:.2f}.")
        return "\n".join(lines)

    def cache_key(self, snap):
        ev = "".join(str(_bucket(snap["sectors"][s]["evaders"])) for s in SECTORS)
        pu = "".join(str(min(3, snap["sectors"][s]["pursuers"])) for s in SECTORS)
        return f"pursuit|c{snap['caught'] // 5}|E{ev}|P{pu}"

    def prompt_context(self):
        return ("The arena is a 16x16 grid split into quadrant sectors NW, NE, SW, SE. "
                "8 pursuers (your team) chase randomly moving evaders; an evader is "
                "caught when pursuers surround it. Agents move one cell per step "
                "(up/down/left/right/stay). Catching all evaders ends the episode.")
