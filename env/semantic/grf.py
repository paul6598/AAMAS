"""Semantic interface for Google Research Football (left team).

snapshot()  : possession / ball / both teams / shape summary (numeric)
summary()   : d_t text for the Commander (deterministic function of snapshot)
cache_key() : possession x ball zone x game mode x score sign x time tercile
grounding   : GRF tokens -> action indices (see docs/grf-semantic-design.md §6)

Action indices: 0 idle, 1 left, 2 top_left, 3 top, 4 top_right, 5 right,
6 bottom_right, 7 bottom, 8 bottom_left, 9 long_pass, 10 high_pass,
11 short_pass, 12 shot, 13 sprint, 14 release_direction, 15 release_sprint,
16 slide, 17 dribble, 18 release_dribble.  Left team attacks toward +x;
"top" is -y.
"""
import math

from .base import SemanticInterface

ROLE_NAMES = ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF"]
GAME_MODE_NAMES = ["normal", "kickoff", "goalkick", "freekick", "corner",
                   "throwin", "penalty"]
# unit vectors of the 8 move actions (index -> (dx, dy))
DIRS = {1: (-1, 0), 2: (-1, -1), 3: (0, -1), 4: (1, -1), 5: (1, 0),
        6: (1, 1), 7: (0, 1), 8: (-1, 1)}
FORWARD, BACK = [5, 4, 6], [1, 2, 8]
UP, DOWN = [3, 2, 4], [7, 6, 8]

SIMPLE_TOKENS = {
    "idle": [0], "move_forward": FORWARD, "move_back": BACK,
    "move_up": UP, "move_down": DOWN, "sprint": [13], "release_sprint": [15],
    "long_pass": [9], "high_pass": [10], "short_pass": [11], "shot": [12],
    "dribble": [17], "release_dribble": [18], "slide": [16],
}
CARRIER_ONLY = ("long_pass", "high_pass", "short_pass", "shot", "dribble",
                "release_dribble")
OFF_BALL_ONLY = ("slide",)
VALID_TOKENS = tuple(SIMPLE_TOKENS) + ("move_toward_ball", "move_toward_goal",
                                       "move_toward_own_goal")
SELECTORS = ("all", "carrier", "off_ball", "nearest_to_ball", "deepest") + \
    tuple("role:" + r for r in ROLE_NAMES)


def _zone(x, y):
    zx = "def" if x < -1 / 3 else ("mid" if x < 1 / 3 else "att")
    zy = "L" if y < -0.14 else ("C" if y < 0.14 else "R")
    return zx + "-" + zy


def _quantize(dx, dy):
    best, bd = 0, -2.0
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    for a, (ux, uy) in DIRS.items():
        un = math.hypot(ux, uy)
        d = (dx * ux + dy * uy) / (n * un)
        if d > bd:
            best, bd = a, d
    return best


class GRFSemanticInterface(SemanticInterface):

    def __init__(self, env, args):
        super().__init__(env, args)
        self._last_possession = None
        self._last_change_t = None
        self._t = 0

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        o = self.env.last_raw[0]
        ctrl = self.env.controlled
        bx, by, bz = [float(v) for v in o["ball"]]
        bdx, bdy = float(o["ball_direction"][0]), float(o["ball_direction"][1])
        own = int(o["ball_owned_team"])
        possession = {-1: "loose", 0: "ours", 1: "theirs"}[own]

        allies, enemies = [], []
        for i in range(len(o["left_team"])):
            x, y = [float(v) for v in o["left_team"][i]]
            allies.append({
                "idx": i, "role": ROLE_NAMES[int(o["left_team_roles"][i])],
                "x": x, "y": y, "dx": float(o["left_team_direction"][i][0]),
                "dy": float(o["left_team_direction"][i][1]),
                "tired": float(o["left_team_tired_factor"][i]),
                "has_ball": own == 0 and int(o["ball_owned_player"]) == i,
                "controlled": i in ctrl, "dist_ball": math.hypot(x - bx, y - by),
                "alive": True, "type": ROLE_NAMES[int(o["left_team_roles"][i])]})
        for i in range(len(o["right_team"])):
            x, y = [float(v) for v in o["right_team"][i]]
            enemies.append({
                "idx": i, "role": ROLE_NAMES[int(o["right_team_roles"][i])],
                "x": x, "y": y, "has_ball": own == 1 and int(o["ball_owned_player"]) == i,
                "dist_ball": math.hypot(x - bx, y - by), "alive": True})

        carrier = None
        if own == 0:
            c = allies[int(o["ball_owned_player"])]
            carrier = {"idx": c["idx"], "role": c["role"], "x": c["x"], "y": c["y"],
                       "pressure": min(e["dist_ball"] for e in enemies)}
        elif own == 1:
            c = enemies[int(o["ball_owned_player"])]
            carrier = {"idx": c["idx"], "role": c["role"], "x": c["x"], "y": c["y"],
                       "pressure": min(a["dist_ball"] for a in allies)}

        field = [a for a in allies if a["role"] != "GK"]
        shape = {
            "ours_behind_ball": sum(1 for a in field if a["x"] < bx),
            "theirs_behind_ball": sum(1 for e in enemies if e["role"] != "GK" and e["x"] > bx),
            "ours_in_att_third": sum(1 for a in field if a["x"] > 1 / 3),
            "theirs_in_our_third": sum(1 for e in enemies if e["role"] != "GK" and e["x"] < -1 / 3),
            "nearest_enemy_to_ball": min(e["dist_ball"] for e in enemies),
            "nearest_ally_to_ball": min(a["dist_ball"] for a in field),
        }

        return {"possession": possession, "carrier": carrier,
                "ball": {"x": bx, "y": by, "z": bz, "dx": bdx, "dy": bdy,
                         "zone": _zone(bx, by)},
                "game_mode": GAME_MODE_NAMES[int(o["game_mode"])],
                "score": [int(o["score"][0]), int(o["score"][1])],
                "time_frac": float(o["steps_left"]) / max(1.0, float(self.env._steps_total)),
                "allies": allies, "enemies": enemies, "shape": shape,
                "controlled": list(ctrl),
                "since_change": (None if self._last_change_t is None
                                 else self._t - self._last_change_t),
                "n_actions": 19}

    def reset_episode(self):
        self._last_possession, self._last_change_t, self._t = None, None, 0

    def tick(self, snap):
        """Advance the per-episode clock ONCE per env step (kept out of
        snapshot(), which may be called several times per step)."""
        possession = snap["possession"]
        if possession != "loose" and possession != self._last_possession:
            if self._last_possession is not None:
                self._last_change_t = self._t
            self._last_possession = possession
        self._t += 1

    # ------------------------------------------------------------- summary
    def summary(self, snap, extra_stats=None):
        s = snap["shape"]
        sc = snap["score"]
        lines = ["Scenario: %dv%d football (GRF). We are the LEFT team attacking to the RIGHT; "
                 "you guide our %d field players (GK is automatic). %d%% of the match remaining. "
                 "Score us %d - them %d."
                 % (len(snap["allies"]), len(snap["enemies"]), len(snap["controlled"]),
                    int(100 * snap["time_frac"]), sc[0], sc[1])]
        p, c = snap["possession"], snap["carrier"]
        if p == "ours":
            lines.append("Possession: OURS (our %s in the %s zone; nearest opponent %.2f away)."
                         % (c["role"], _zone(c["x"], c["y"]), c["pressure"]))
        elif p == "theirs":
            lines.append("Possession: THEIRS (their %s in the %s zone; our nearest player %.2f away)."
                         % (c["role"], _zone(c["x"], c["y"]), c["pressure"]))
        else:
            lines.append("Possession: LOOSE ball in the %s zone (nearest: ours %.2f, theirs %.2f)."
                         % (snap["ball"]["zone"], s["nearest_ally_to_ball"], s["nearest_enemy_to_ball"]))
        b = snap["ball"]
        motion = "toward THEIR goal" if b["dx"] > 0.002 else ("toward OUR goal" if b["dx"] < -0.002 else "roughly still")
        lines.append("Ball: %s zone, moving %s." % (b["zone"], motion))
        nf = len(snap["controlled"])
        lines.append("Our shape: %d of %d field players behind the ball; %d in their attacking third."
                     % (s["ours_behind_ball"], nf, s["ours_in_att_third"]))
        lines.append("Their shape: %d players in our defensive third; %d behind the ball."
                     % (s["theirs_in_our_third"], s["theirs_behind_ball"]))
        lines.append("Set piece: %s." % ("none (normal play)" if snap["game_mode"] == "normal" else snap["game_mode"]))
        if snap["since_change"] is not None:
            lines.append("Recent: possession changed %d steps ago." % snap["since_change"])
        if extra_stats and "rolling_win_rate" in extra_stats:
            lines.append("Training context: rolling win rate %.2f, %d environment steps elapsed."
                         % (extra_stats["rolling_win_rate"], extra_stats.get("t_env", 0)))
        return "\n".join(lines)

    def phase(self, snap):
        return snap["possession"]

    def cache_key(self, snap):
        sc = snap["score"]
        sign = (sc[0] > sc[1]) - (sc[0] < sc[1])
        tt = min(2, int(3 * (1 - snap["time_frac"])))
        ob = snap["shape"]["ours_behind_ball"]
        obb = 0 if ob <= 1 else (1 if ob == 2 else 2)
        return "grf|%s|%s|%s|%+d|t%d|b%d" % (snap["possession"], snap["ball"]["zone"],
                                            snap["game_mode"], sign, tt, obb)

    # ------------------------------------------------------- prompt context
    def prompt_context(self):
        roles = [a for a in self.snapshot()["allies"] if a["controlled"]]
        return (
            "We are the LEFT team and attack to the RIGHT (+x). You guide our field "
            "players: %s. The goalkeeper is automatic.\n"
            "Each controlled player has 19 discrete actions: idle; move in 8 directions; "
            "long_pass, high_pass, short_pass, shot (only meaningful for the ball carrier); "
            "sprint / release_sprint; dribble / release_dribble (carrier only); slide (tackle, "
            "foul risk).\n"
            "The environment already rewards goals and forward ball progress."
            % ", ".join("%s(#%d)" % (a["role"], a["idx"]) for a in roles))

    # ------------------------------------------------------------ grounding
    def _agent(self, agent_idx, snap):
        return snap["allies"][snap["controlled"][agent_idx]]

    def resolve_action_token(self, token, agent_idx, snap):
        me = self._agent(agent_idx, snap)
        is_carrier = me["has_ball"]
        if token in CARRIER_ONLY and not is_carrier:
            return []
        if token in OFF_BALL_ONLY and is_carrier:
            return []
        if token == "shot" and me["x"] < 0.5:
            return []
        if token in SIMPLE_TOKENS:
            return list(SIMPLE_TOKENS[token])
        b = snap["ball"]
        if token == "move_toward_ball":
            a = _quantize(b["x"] - me["x"], b["y"] - me["y"])
        elif token == "move_toward_goal":
            a = _quantize(1.0 - me["x"], 0.0 - me["y"])
        elif token == "move_toward_own_goal":
            a = _quantize(-1.0 - me["x"], 0.0 - me["y"])
        else:
            return []
        return [] if a is None else [a]

    def agent_matches(self, selector, agent_idx, snap):
        me = self._agent(agent_idx, snap)
        if selector in (None, "all", "*"):
            return True
        if selector == "carrier":
            return me["has_ball"]
        if selector == "off_ball":
            return not me["has_ball"]
        ctrl = [snap["allies"][i] for i in snap["controlled"]]
        if selector == "nearest_to_ball":
            return me["idx"] == min(ctrl, key=lambda a: a["dist_ball"])["idx"]
        if selector == "deepest":
            return me["idx"] == min(ctrl, key=lambda a: a["x"])["idx"]
        if selector.startswith("role:"):
            return me["role"].lower() == selector.split(":", 1)[1].strip().lower()
        return False
