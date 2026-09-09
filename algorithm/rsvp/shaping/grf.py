"""GRF reward predicates f_j(pre, post, actions) for semantic shaping and for
sub-goal progress monitoring. Per-step magnitudes are scaled to <= 1.

Actions are agent-ordered (controlled players); index 19 (builtin_ai) or None
means the action is unobserved and is treated as a plain move.
"""
import numpy as np

PASS_ACTIONS = (9, 10, 11)
SHOT, SLIDE = 12, 16
SIMPLE_PREDICATES = ("ball_progress", "keep_possession", "shot_in_box", "pass_forward",
                     "regain_possession", "defensive_shape", "press_carrier",
                     "compactness", "no_slide_foul")
ALL_PREDICATES = SIMPLE_PREDICATES


def _controlled(snap):
    return [snap["allies"][i] for i in snap["controlled"]]


def _spread(snap):
    pts = np.array([[a["x"], a["y"]] for a in _controlled(snap)])
    return float(pts.std(axis=0).sum()) if len(pts) > 1 else 0.0


def _carrier_action(pre, actions):
    for k, a in enumerate(_controlled(pre)):
        if a["has_ball"]:
            act = actions[k] if actions else None
            return act if (act is not None and act < 19) else None
    return None


def evaluate_predicate(pred, pre, post, actions):
    p0, p1 = pre["possession"], post["possession"]
    if pred == "ball_progress":
        if p0 == "ours" and p1 == "ours":
            return float(np.clip((post["ball"]["x"] - pre["ball"]["x"]) / 0.02, -1.0, 1.0))
        return 0.0
    if pred == "keep_possession":
        # event-scaled: only a true turnover is penalised; a ball in flight
        # (ours -> loose, e.g. a pass) is not taxed; holding gives a small
        # maintenance signal instead of a dominant constant stream.
        if p0 == "ours":
            if p1 == "theirs":
                return -1.0
            return 0.1 if p1 == "ours" else 0.0
        return 0.0
    if pred == "shot_in_box":
        act = _carrier_action(pre, actions)
        return 1.0 if (act == SHOT and pre["ball"]["x"] > 0.7) else 0.0
    if pred == "pass_forward":
        act = _carrier_action(pre, actions)
        if act in PASS_ACTIONS and post["ball"]["x"] > pre["ball"]["x"] and p1 != "theirs":
            return 1.0
        return 0.0
    if pred == "regain_possession":
        return 1.0 if (p0 != "ours" and p1 == "ours") else 0.0
    if pred == "defensive_shape":
        if p1 == "theirs":
            d = (post["shape"]["ours_behind_ball"] - pre["shape"]["ours_behind_ball"]) / 4.0
            return float(np.clip(d + (0.25 if post["shape"]["ours_behind_ball"] >= 3 else 0.0), -1.0, 1.0))
        return 0.0
    if pred == "press_carrier":
        if p0 == "theirs" and p1 == "theirs":
            d = pre["shape"]["nearest_ally_to_ball"] - post["shape"]["nearest_ally_to_ball"]
            return float(np.clip(d / 0.02, -1.0, 1.0))
        return 0.0
    if pred == "compactness":
        d = _spread(post) - _spread(pre)
        sign = -1.0 if p1 == "theirs" else (1.0 if p1 == "ours" else 0.0)
        return float(np.clip(sign * d / 0.02, -1.0, 1.0))
    if pred == "no_slide_foul":
        return -1.0 if any(a == SLIDE for a in (actions or [])) else 0.0
    return 0.0


def applicable(pred, snap):
    """Precondition of the predicate in the current state (can it produce reward?)."""
    poss = snap["possession"]
    if pred in ("ball_progress", "keep_possession", "shot_in_box", "pass_forward"):
        return poss == "ours"
    if pred == "regain_possession":
        return poss != "ours"
    if pred == "press_carrier":
        return poss == "theirs"
    if pred == "defensive_shape":
        return poss != "ours"
    return True  # compactness, no_slide_foul


def applicable_fraction(subgoals, snap):
    """Weight share of sub-goals whose precondition holds (1 = fully applicable)."""
    tot = sum(float(sg.get("weight", 0.5)) for sg in subgoals or [])
    if tot <= 0:
        return 1.0
    return sum(float(sg.get("weight", 0.5)) for sg in subgoals or [] if applicable(sg["predicate"], snap)) / tot


def compute_shaping(subgoals, pre, post, actions, clip=3.0):
    total = 0.0
    for sg in subgoals or []:
        w = min(1.0, float(sg.get("weight", 0.5)))   # belt-and-suspenders, as in LEHCA
        total += w * evaluate_predicate(sg["predicate"], pre, post, actions)
    return float(np.clip(total, -clip, clip))


def progress_vector(subgoals, pre, post, actions):
    """Per-sub-goal values (weighted) — the monitoring stream."""
    return [float(sg.get("weight", 0.5)) * evaluate_predicate(sg["predicate"], pre, post, actions)
            for sg in subgoals or []]
