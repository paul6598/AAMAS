"""Semantic interface for SMAC (StarCraft II) environments.

Builds the structured natural-language bundle d_t from observable unit
information (aggregated ally observations / visible enemy profile), and
grounds symbolic action tokens and reward predicates into SMAC's discrete
action space:  0 no-op, 1 stop, 2..5 move N/S/E/W, 6+i attack enemy i
(heal ally i for Medivacs).
"""
import math

from .base import SemanticInterface

# Standard SC2 unit type ids (used by SMAC for enemy units).
STANDARD_TYPE_NAMES = {
    4: "Colossus",
    9: "Baneling",
    48: "Marine",
    51: "Marauder",
    54: "Medivac",
    73: "Zealot",
    74: "Stalker",
    98: "SpineCrawler",
    105: "Zergling",
    107: "Hydralisk",
}

# SMAC assigns custom type ids to ally units, exposed as attributes.
ALLY_TYPE_ATTRS = [
    ("marine_id", "Marine"),
    ("marauder_id", "Marauder"),
    ("medivac_id", "Medivac"),
    ("stalker_id", "Stalker"),
    ("colossus_id", "Colossus"),
    ("zealot_id", "Zealot"),
    ("zergling_id", "Zergling"),
    ("baneling_id", "Baneling"),
    ("hydralisk_id", "Hydralisk"),
    ("scv_id", "SCV"),
]

N_BASE_ACTIONS = 6
MOVE_ACTIONS = [2, 3, 4, 5]

UNIT_ROLE_NOTES = {
    "Marine": "light ranged infantry: low health, high sustained damage",
    "Marauder": "heavy ranged infantry: high durability, strong vs armored",
    "Medivac": "non-combat flying healer: heals biological allies, cannot attack",
    "Stalker": "ranged mechanical: mobile, shields regenerate",
    "Zealot": "melee: high damage up close, must reach targets",
    "Colossus": "massive long-range splash damage, vulnerable when focused",
    "Zergling": "fast fragile melee swarm unit",
    "Baneling": "suicide splash unit: lethal on contact",
    "Hydralisk": "ranged attacker with moderate health",
    "SpineCrawler": "static defensive structure",
}


class SC2SemanticInterface(SemanticInterface):
    def __init__(self, env, args):
        super().__init__(env, args)
        self._ally_type_names = None

    # ------------------------------------------------------------- helpers
    def _ally_types(self):
        # SMAC sets *_id attributes during launch; build the map lazily.
        if self._ally_type_names is None:
            m = {}
            for attr, name in ALLY_TYPE_ATTRS:
                if hasattr(self.env, attr):
                    m[getattr(self.env, attr)] = name
            self._ally_type_names = m
        return self._ally_type_names

    def _type_name(self, unit, ally):
        if ally:
            name = self._ally_types().get(unit.unit_type)
            if name is not None:
                return name
        return STANDARD_TYPE_NAMES.get(unit.unit_type, "Unit%d" % unit.unit_type)

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        allies, enemies = [], []
        for aid in range(self.env.n_agents):
            u = self.env.agents[aid]
            allies.append(self._unit_info(u, ally=True))
        for eid in range(self.env.n_enemies):
            u = self.env.enemies[eid]
            enemies.append(self._unit_info(u, ally=False))
        return {"allies": allies, "enemies": enemies,
                "n_actions": N_BASE_ACTIONS + self.env.n_enemies}

    def _unit_info(self, u, ally):
        hp_max = float(u.health_max) + float(getattr(u, "shield_max", 0.0))
        hp = float(u.health) + float(getattr(u, "shield", 0.0))
        return {
            "type": self._type_name(u, ally),
            "hp": hp,
            "hp_max": max(hp_max, 1e-6),
            "x": float(u.pos.x),
            "y": float(u.pos.y),
            "alive": u.health > 0,
        }

    @staticmethod
    def _force_profile(units):
        prof = {}
        for u in units:
            p = prof.setdefault(u["type"], {"total": 0, "alive": 0, "hp": 0.0,
                                            "hp_max": 0.0, "low": 0})
            p["total"] += 1
            if u["alive"]:
                p["alive"] += 1
                p["hp"] += u["hp"]
                p["hp_max"] += u["hp_max"]
                if u["hp"] / u["hp_max"] < 0.35:
                    p["low"] += 1
        return prof

    @staticmethod
    def _centroid(units):
        pts = [(u["x"], u["y"]) for u in units if u["alive"]]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    # ------------------------------------------------------------- summary
    def summary(self, snap, extra_stats=None):
        lines = []
        lines.append("Scenario: SMAC map '%s' (%d allied vs %d enemy units)."
                     % (self.env.map_name, self.env.n_agents, self.env.n_enemies))

        for side, units in (("Allied", snap["allies"]), ("Enemy", snap["enemies"])):
            prof = self._force_profile(units)
            n_alive = sum(1 for u in units if u["alive"])
            lines.append("%s force: %d/%d alive." % (side, n_alive, len(units)))
            for tname, p in sorted(prof.items()):
                if p["alive"] > 0:
                    frac = 100.0 * p["hp"] / max(p["hp_max"], 1e-6)
                    s = ("  - %s: %d/%d alive, avg health %.0f%%"
                         % (tname, p["alive"], p["total"], frac))
                    if p["low"] > 0:
                        s += ", %d critically low" % p["low"]
                else:
                    s = "  - %s: 0/%d alive (eliminated)" % (tname, p["total"])
                note = UNIT_ROLE_NOTES.get(tname)
                if note:
                    s += " [%s]" % note
                lines.append(s)

        ca, ce = self._centroid(snap["allies"]), self._centroid(snap["enemies"])
        if ca and ce:
            dist = math.hypot(ca[0] - ce[0], ca[1] - ce[1])
            phase = "engaged (within typical attack range)" if dist < 8 else \
                    "approaching (out of attack range)"
            lines.append("Spatial: army separation %.1f map units; phase: %s."
                         % (dist, phase))
            dx, dy = ce[0] - ca[0], ce[1] - ca[1]
            ew = "east" if dx > 0 else "west"
            ns = "north" if dy > 0 else "south"
            major = ew if abs(dx) >= abs(dy) else ns
            lines.append("Enemy centroid lies to the %s of the allied force." % major)

        if extra_stats:
            if "rolling_win_rate" in extra_stats:
                lines.append("Training context: rolling win rate %.2f over recent "
                             "episodes, %d environment steps elapsed."
                             % (extra_stats["rolling_win_rate"],
                                extra_stats.get("t_env", 0)))
        return "\n".join(lines)

    def phase(self, snap):
        ca, ce = self._centroid(snap["allies"]), self._centroid(snap["enemies"])
        if not (ca and ce):
            return None
        return "engaged" if math.hypot(ca[0] - ce[0], ca[1] - ce[1]) < 8 else "approaching"

    def cache_key(self, snap):
        # Coarse state: per-type alive counts + bucketed avg health + phase.
        parts = [self.env.map_name]
        for side, units in (("A", snap["allies"]), ("E", snap["enemies"])):
            prof = self._force_profile(units)
            for tname, p in sorted(prof.items()):
                if p["alive"]:
                    bucket = int(4 * p["hp"] / max(p["hp_max"], 1e-6))
                else:
                    bucket = -1
                parts.append("%s:%s:%d:%d" % (side, tname, p["alive"], bucket))
        ca, ce = self._centroid(snap["allies"]), self._centroid(snap["enemies"])
        if ca and ce:
            parts.append("d%d" % int(math.hypot(ca[0] - ce[0], ca[1] - ce[1]) // 4))
        return "|".join(parts)

    # ------------------------------------------------------- prompt context
    def prompt_context(self):
        types_ally = sorted({self._type_name(self.env.agents[a], True)
                             for a in range(self.env.n_agents)})
        types_enemy = sorted({self._type_name(self.env.enemies[e], False)
                              for e in range(self.env.n_enemies)})
        return (
            "Each allied agent has this discrete action space: "
            "noop (only when dead), stop, move_north, move_south, move_east, "
            "move_west, and attack_enemy_i for each enemy unit i "
            "(Medivacs heal allies instead of attacking).\n"
            "Allied unit types: %s. Enemy unit types: %s.\n"
            "Victory requires eliminating all enemy units; the environment "
            "reward is sparse and mainly given for damage, kills and winning."
            % (", ".join(types_ally), ", ".join(types_enemy))
        )

    # ------------------------------------------------------------ grounding
    def resolve_action_token(self, token, agent_idx, snap):
        """Return concrete action indices for a symbolic token; [] if empty."""
        n_actions = snap["n_actions"]
        enemies = snap["enemies"]
        is_medivac = snap["allies"][agent_idx]["type"] == "Medivac"

        if token == "noop":
            return [0]
        if token == "stop":
            return [1]
        if token == "move_north":
            return [2]
        if token == "move_south":
            return [3]
        if token == "move_east":
            return [4]
        if token == "move_west":
            return [5]
        if token == "move_all":
            return list(MOVE_ACTIONS)

        # Attack-style tokens: not meaningful for Medivacs (their target
        # actions are heals on allies) -> no-op mapping.
        if is_medivac:
            return []
        if token == "attack_all":
            return [N_BASE_ACTIONS + i for i, e in enumerate(enemies) if e["alive"]]
        if token.startswith("attack_type:"):
            tname = token.split(":", 1)[1].strip().lower()
            return [N_BASE_ACTIONS + i for i, e in enumerate(enemies)
                    if e["alive"] and e["type"].lower() == tname]
        if token == "attack_lowest_health":
            alive = [(e["hp"], i) for i, e in enumerate(enemies) if e["alive"]]
            if not alive:
                return []
            return [N_BASE_ACTIONS + min(alive)[1]]
        if token == "attack_nearest":
            me = snap["allies"][agent_idx]
            alive = [(math.hypot(e["x"] - me["x"], e["y"] - me["y"]), i)
                     for i, e in enumerate(enemies) if e["alive"]]
            if not alive:
                return []
            return [N_BASE_ACTIONS + min(alive)[1]]
        return []

    def agent_matches(self, selector, agent_idx, snap):
        if selector in (None, "all", "*"):
            return True
        if selector.startswith("type:"):
            tname = selector.split(":", 1)[1].strip().lower()
            return snap["allies"][agent_idx]["type"].lower() == tname
        if selector.startswith("agent:"):
            try:
                return int(selector.split(":", 1)[1]) == agent_idx
            except ValueError:
                return False
        return False
