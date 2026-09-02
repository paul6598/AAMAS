"""Is the FRESH LLM guidance appropriate for the current state? (GRF probe logs)

Cross-tabulates state conditions (possession, ball zone, carrier pressure,
set piece, loose ball) against (a) the phase word in `strategy`, (b) sub-goal
predicates, (c) rule prefer/forbid tokens per selector. Also reports guidance
diversity (does the LLM collapse to a few templates?) and prints examples.

Usage: python algorithm/vigil/analyze_guidance_quality.py results/vigil/probe_grf_*.jsonl
"""
import json
import sys
from collections import Counter, defaultdict

ATT_PRED = {"ball_progress", "keep_possession", "pass_forward", "shot_in_box"}
DEF_PRED = {"regain_possession", "defensive_shape", "press_carrier", "no_slide_foul"}
ATT_TOK = {"move_forward", "move_toward_goal", "shot", "dribble", "short_pass", "long_pass", "high_pass"}
DEF_TOK = {"move_back", "move_toward_own_goal", "move_toward_ball", "slide"}


def phase_word(strategy):
    s = strategy.upper()
    for w in ("ATTACK", "DEFEN", "TRANSITION", "SET PIECE", "SET-PIECE", "LOOSE", "PRESS", "COUNTER"):
        if w in s:
            return w
    return "other"


def load_points(paths):
    pts = []
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            if d.get("end") or not d.get("shadow"):
                continue
            g = d["shadow"]["guidances"][0]
            if g is None:
                pts.append((d, None))
                continue
            pts.append((d, g))
    return pts


def cond(d):
    s = d["snap"]
    poss = s["possession"]
    zone = s["ball"]["zone"].split("-")[0]
    c = s["carrier"]
    press = None
    if poss == "ours" and c is not None:
        press = "pressed" if c["pressure"] < 0.06 else "free"
    return poss, zone, press, s["game_mode"] != "normal"


def pct(c, n):
    return "%3d%%" % (100.0 * c / n) if n else "  - "


def main(paths):
    pts = load_points(paths)
    n_fail = sum(1 for _, g in pts if g is None)
    pts = [(d, g) for d, g in pts if g is not None]
    print("shadow points: %d  (LLM failures/None: %d)" % (len(pts), n_fail))

    # ---- (a) phase word vs possession
    tab = defaultdict(Counter)
    for d, g in pts:
        poss, zone, press, sp = cond(d)
        tab[poss][phase_word(g["strategy"])] += 1
    print("\n[a] strategy phase word by possession")
    words = sorted({w for c in tab.values() for w in c})
    print("  %-7s" % "poss" + "".join("%12s" % w for w in words) + "     n")
    for poss in ("ours", "loose", "theirs"):
        n = sum(tab[poss].values())
        print("  %-7s" % poss + "".join("%12s" % pct(tab[poss][w], n) for w in words) + "  %4d" % n)

    # ---- (b) sub-goals by possession x zone
    print("\n[b] sub-goal predicates by possession (share of guidances containing it)")
    preds = sorted(ATT_PRED | DEF_PRED | {"compactness"})
    sg = defaultdict(Counter)
    nn = Counter()
    for d, g in pts:
        poss, zone, press, sp = cond(d)
        key = poss
        nn[key] += 1
        for x in g["subgoals"]:
            sg[key][x["predicate"]] += 1
    print("  %-7s" % "poss" + "".join("%17s" % p for p in preds))
    for key in ("ours", "loose", "theirs"):
        print("  %-7s" % key + "".join("%17s" % pct(sg[key][p], nn[key]) for p in preds))
    # attack-only vs defend-only mix
    print("\n  sub-goal set is phase-consistent (ours: no DEF preds; theirs: no ATT preds):")
    for key, bad in (("ours", DEF_PRED), ("theirs", ATT_PRED)):
        n = ok = 0
        for d, g in pts:
            if cond(d)[0] != key:
                continue
            n += 1
            ok += not ({x["predicate"] for x in g["subgoals"]} & bad)
        print("    %-7s %s (n=%d)" % (key, pct(ok, n), n))

    # ---- (c) rule tokens by possession and selector
    print("\n[c] prefer tokens by possession x selector (share of guidances)")
    tok = defaultdict(Counter)
    nn = Counter()
    for d, g in pts:
        poss = cond(d)[0]
        nn[poss] += 1
        seen = set()
        for r in g["action_rules"]:
            for t in r["prefer"]:
                seen.add((r["applies_to"], t))
        for k in seen:
            tok[poss][k] += 1
    for poss in ("ours", "loose", "theirs"):
        print("  -- %s (n=%d)" % (poss, nn[poss]))
        for (sel, t), c in tok[poss].most_common(10):
            print("     %-16s %-22s %s" % (sel, t, pct(c, nn[poss])))
    print("\n  forbid tokens (all states):", Counter(t for _, g in pts for r in g["action_rules"] for t in r["forbid"]).most_common(8))

    # ---- (d) finer appropriateness checks
    print("\n[d] situation-specific checks")
    checks = {
        "ours & att zone -> shot_in_box or 'shot' prefer": lambda d, g: (cond(d)[0] == "ours" and cond(d)[1] == "att",
            any(x["predicate"] == "shot_in_box" for x in g["subgoals"]) or any("shot" in r["prefer"] for r in g["action_rules"])),
        "ours & carrier pressed -> short_pass prefer": lambda d, g: (cond(d)[2] == "pressed",
            any("short_pass" in r["prefer"] for r in g["action_rules"])),
        "ours & carrier free -> dribble/move_forward prefer": lambda d, g: (cond(d)[2] == "free",
            any({"dribble", "move_forward"} & set(r["prefer"]) for r in g["action_rules"])),
        "theirs & ball in our def zone -> defensive_shape/press/move_toward_ball": lambda d, g: (cond(d)[0] == "theirs" and cond(d)[1] == "def",
            any(x["predicate"] in ("defensive_shape", "press_carrier") for x in g["subgoals"]) or
            any({"move_toward_ball", "move_back", "move_toward_own_goal"} & set(r["prefer"]) for r in g["action_rules"])),
        "theirs -> no ATT tokens for off_ball/all": lambda d, g: (cond(d)[0] == "theirs",
            not any(r["applies_to"] in ("all", "off_ball") and ATT_TOK & set(r["prefer"]) for r in g["action_rules"])),
        "loose -> move_toward_ball prefer (any selector)": lambda d, g: (cond(d)[0] == "loose",
            any("move_toward_ball" in r["prefer"] for r in g["action_rules"])),
        "set piece -> strategy mentions set piece/kick/corner/throw": lambda d, g: (cond(d)[3],
            any(w in g["strategy"].lower() for w in ("set", "kick", "corner", "throw", "penalty"))),
    }
    for name, fn in checks.items():
        n = ok = 0
        for d, g in pts:
            applies, good = fn(d, g)
            if applies:
                n += 1
                ok += bool(good)
        print("  %-70s %s (n=%d)" % (name, pct(ok, n), n))

    # ---- (e) diversity / template collapse
    print("\n[e] diversity")
    def sig(g):
        return json.dumps({"sg": sorted(x["predicate"] for x in g["subgoals"]),
                           "rules": sorted((r["applies_to"], tuple(sorted(r["prefer"])), tuple(sorted(r["forbid"]))) for r in g["action_rules"])}, sort_keys=True)
    by_poss = defaultdict(Counter)
    for d, g in pts:
        by_poss[cond(d)[0]][sig(g)] += 1
    for poss in ("ours", "loose", "theirs"):
        c = by_poss[poss]
        n = sum(c.values())
        top = c.most_common(3)
        print("  %-7s distinct guidances %3d / %3d points; top-3 cover %s" % (
            poss, len(c), n, pct(sum(v for _, v in top), n)))
    keys = Counter(d["cache_key"] for d, _ in pts)
    print("  distinct cache_keys: %d / %d points" % (len(keys), len(pts)))

    # ---- (f) examples
    print("\n[f] examples")
    shown = set()
    for d, g in pts:
        poss, zone, press, sp = cond(d)
        tag = (poss, zone, press, sp)
        if tag in shown or len(shown) >= 6:
            continue
        shown.add(tag)
        print("  --- state: possession=%s zone=%s pressure=%s set_piece=%s" % tag)
        print("      " + d["shadow"]["summary"].split("\n")[1])
        print("      strategy:", g["strategy"][:110])
        print("      subgoals:", [(x["predicate"], x["weight"]) for x in g["subgoals"]])
        for r in g["action_rules"]:
            print("      rule:", r["applies_to"], "forbid", r["forbid"], "prefer", r["prefer"], r["prefer_weight"])


if __name__ == "__main__":
    main(sys.argv[1:])
