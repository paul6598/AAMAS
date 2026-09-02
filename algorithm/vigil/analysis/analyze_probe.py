"""Offline analysis of probe_phase.py logs.

Reports
  1. noise floor D_noise vs staleness D_stale(delta) in effect space
  2. guidance change across event vs non-event windows
  3. backfire on the heuristic policy: held-at-episode-start mask forbids the
     action actually taken, vs. the freshest shadow guidance
  4. trigger replay: fixed F / per-episode / event / CUSUM(phi) -> calls per
     episode vs. mean staleness exposure (Pareto table)

Refreshes in the replay are only allowed at shadow points (where an LLM
sample exists); the guidance adopted at a refresh is the recorded sample.

Usage: python algorithm/vigil/analyze_probe.py results/vigil/probe_*.jsonl
"""
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from env.semantic.sc2 import SC2SemanticInterface  # noqa: E402
from algorithm.vigil.analysis.effect import mask_distance  # noqa: E402
from algorithm.lehca.masking import build_masks  # noqa: E402
from algorithm.vigil.shaping.grf import compute_shaping as grf_shaping, applicable_fraction, evaluate_predicate, ALL_PREDICATES  # noqa: E402
from algorithm.lehca.shaping.predicates import evaluate_predicate as sc2_predicate  # noqa: E402

from env.semantic.grf import GRFSemanticInterface  # noqa: E402

_SC2 = SC2SemanticInterface(None, SimpleNamespace(dt_observable=True))
_GRF = GRFSemanticInterface(None, SimpleNamespace())


def iface_for(snap):
    return _GRF if "controlled" in snap else _SC2


def n_agents_of(snap):
    return len(snap["controlled"]) if "controlled" in snap else len(snap["allies"])


def load(paths):
    eps = defaultdict(list)
    ends = {}
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            if d.get("end"):
                ends[(p, d["ep"])] = d
            else:
                eps[(p, d["ep"])].append(d)
    return eps, ends


def dist(g1, g2, rec):
    snap = rec["snap"]
    return mask_distance(g1, g2, snap, rec["avail"], iface_for(snap), n_agents_of(snap))


def shadow_points(steps):
    return [r for r in steps if r["shadow"] and r["shadow"]["guidances"][0] is not None]


def mean_of(ds, key):
    v = [d[key] for d in ds if d is not None]
    return float(np.mean(v)) if v else float("nan")


def phi_vec(ph, types):
    """Normalised feature vector for CUSUM (SMAC: alive/hp/dist; GRF: possession,
    ball, shape)."""
    if "possession" in ph:  # GRF
        poss = {"ours": (1, 0, 0), "loose": (0, 1, 0), "theirs": (0, 0, 1)}[ph["possession"]]
        return np.array(list(poss) + [ph["ball_x"], ph["ball_y"], ph["ours_behind"] / 4.0,
                                      ph["theirs_in_our_third"] / 4.0,
                                      min(1.0, ph["nearest_enemy_to_ball"] * 5), float(ph["set_piece"])])
    v = []
    for t in types:
        v.append(ph.get(t + "_alive", 0) / 5.0)
        v.append(ph.get(t + "_hp", 0.0))
    v.append(ph["n_enemy_vis"] / 5.0)
    v.append((ph["dist"] or 20.0) / 20.0)
    return np.array(v)


MAJOR = ("ally_death", "enemy_death", "eliminated", "force_flip", "phase", "first_sight",
         "possession", "set_piece")


def major_events(evs):
    return [e for e in evs if e.split(":")[0] in MAJOR]


def drift_vec(ph):
    return np.array([ph["ball_x"], ph["ball_y"], ph["ours_behind"] / 4.0,
                     ph["theirs_in_our_third"] / 4.0, min(1.0, ph["nearest_enemy_to_ball"] * 5)])


def is_grf(snap):
    return "controlled" in snap


def sg_key(sg):
    return (sg["predicate"], sg.get("unit_type"))


def pred_value(sg, pre, post, actions):
    if is_grf(pre):
        return evaluate_predicate(sg["predicate"], pre, post, actions)
    return sc2_predicate(sg["predicate"], sg.get("unit_type"), pre, post, actions or [])


def context(rec):
    snap = rec["snap"]
    if is_grf(snap):
        return (snap["possession"], snap["ball"]["zone"].split("-")[0])
    a = tuple(sorted({u["type"] for u in snap["allies"] if u["alive"]}))
    e = tuple(sorted({u["type"] for u in snap["enemies"] if u["alive"] and u.get("visible", True)}))
    return (a, e, rec["phi"].get("phase"))


def progress_tables(all_eps, exclude_key):
    """Cross-spell reference for guidance progress: per sub-goal key and context,
    mean predicate value m and support probability s = P(f != 0)."""
    keys = set()
    for key, steps in all_eps.items():
        for r in steps:
            if r["shadow"]:
                for g in r["shadow"]["guidances"]:
                    for sg in (g or {}).get("subgoals", []):
                        keys.add(sg_key(sg))
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))  # key -> ctx -> [sum f, n nonzero, n]
    for ek, steps in all_eps.items():
        if ek == exclude_key:
            continue
        for a, b in zip(steps, steps[1:]):
            c = context(a)
            for k in keys:
                sg = {"predicate": k[0], "unit_type": k[1], "weight": 1.0}
                v = pred_value(sg, a["snap"], b["snap"], a.get("actions"))
                cell = acc[k][c]
                cell[0] += v
                cell[1] += (v != 0)
                cell[2] += 1
    tabs = {}
    for k, ctxs in acc.items():
        tabs[k] = {c: (cell[0] / cell[2], cell[1] / cell[2]) for c, cell in ctxs.items() if cell[2] > 0}
    return tabs


def guidance_progress(held, hist, ctx, tabs, W):
    """v_t in [0,1]: weight-average of normalised progress ratios over the last W steps.
    hist: list of (pre, post, actions) since refresh."""
    num = tot = 0.0
    window = hist[-W:]
    for sg in held.get("subgoals") or []:
        w = float(sg.get("weight", 0.5))
        tot += w
        m, sup = tabs.get(sg_key(sg), {}).get(ctx, (0.0, 0.0))
        if sup < 0.02 or m <= 0:
            continue  # terminated / unsupported here -> rho = 0
        fbar = sum(pred_value(sg, a, b, acts) for a, b, acts in window) / max(1, len(window))
        num += w * min(1.0, max(0.0, fbar) / max(m, 1e-3))
    return num / tot if tot > 0 else 1.0


def firing_tables(all_eps, exclude_key):
    """P(f_j != 0 | possession, zone_x) per predicate from all episodes except one."""
    cnt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, steps in all_eps.items():
        if key == exclude_key:
            continue
        for a, b in zip(steps, steps[1:]):
            ctx = (a["snap"]["possession"], a["snap"]["ball"]["zone"].split("-")[0])
            for pr in ALL_PREDICATES:
                v = evaluate_predicate(pr, a["snap"], b["snap"], a.get("actions"))
                c = cnt[pr][ctx]
                c[1] += 1
                c[0] += (v != 0)
    tabs = {}
    for pr in ALL_PREDICATES:
        rates = {ctx: c[0] / max(1, c[1]) for ctx, c in cnt[pr].items()}
        m = max(rates.values()) if rates else 0.0
        # never-fired predicates carry no information -> treat as applicable everywhere
        tabs[pr] = ({ctx: r / m for ctx, r in rates.items()} if m > 0.002 else None)
    return tabs


def learned_fraction(subgoals, snap, tabs):
    ctx = (snap["possession"], snap["ball"]["zone"].split("-")[0])
    tot = num = 0.0
    for sg in subgoals or []:
        w = float(sg.get("weight", 0.5))
        tab = tabs.get(sg["predicate"])
        pr = 1.0 if tab is None else tab.get(ctx, 0.0)
        tot += w
        num += w * pr
    return num / tot if tot > 0 else 1.0


def replay(steps, schedule, **kw):
    """Return (n_calls, mean hard_diff, mean prefer_diff, mean forbid_conflict)
    of held guidance vs fresh sample over shadow points. An alarm raised between
    shadow points is served at the next shadow point."""
    sp = shadow_points(steps)
    if not sp:
        return None
    types = sorted({k[:-6] for r in steps for k in r["phi"] if k.endswith("_alive")})
    held, last_t, calls, S, phi0, pending = None, None, 0, 0.0, None, True
    ds, refresh_ts = [], []
    last_side, d0 = None, None
    prev_rec, n_since, ref_sum = None, 0, 0.0
    hist = []
    for r in steps:
        t = r["t"]
        if schedule in ("event2", "hybrid") and held is None and r["phi"].get("possession") in ("ours", "theirs"):
            last_side = r["phi"]["possession"]
        is_sp = bool(r["shadow"] and r["shadow"]["guidances"][0] is not None)
        if held is not None:
            if schedule == "fixed":
                fire = t - last_t >= kw["F"]
            elif schedule == "ep":
                fire = False
            elif schedule == "event":
                fire = bool(major_events(r["events"]))
            elif schedule == "event2":  # GRF: only when possession settles on the other team
                poss = r["phi"].get("possession")
                fire = poss in ("ours", "theirs") and poss != last_side
                if poss in ("ours", "theirs"):
                    last_side = poss
            elif schedule == "hybrid":  # event2 OR drift-CUSUM within a possession spell
                poss = r["phi"].get("possession")
                ev = poss in ("ours", "theirs") and poss != last_side
                if poss in ("ours", "theirs"):
                    last_side = poss
                z = float(np.abs(drift_vec(r["phi"]) - d0).sum())
                S = max(0.0, S + z - kw["k"])
                fire = ev or S >= kw["h"]
            elif schedule == "gp":  # guidance-progress CUSUM (shaping signal vs cross-spell reference)
                fire = False
                if prev_rec is not None:
                    hist.append((prev_rec["snap"], r["snap"], prev_rec.get("actions")))
                    if len(hist) >= kw["W"]:
                        v = guidance_progress(held, hist, context(r), kw["tabs"], kw["W"])
                        S = max(0.0, S + kw["k"] - v)
                        fire = S >= kw["h"]
            elif schedule == "sgapp_learned":  # applicability from data-estimated firing rates
                frac = learned_fraction(held.get("subgoals"), r["snap"], kw["tabs"])
                if "h" in kw:  # budgeted: accumulate inapplicability, debounced by k
                    S = max(0.0, S + (1.0 - frac) - kw["k"])
                    fire = S >= kw["h"]
                else:
                    fire = frac < kw["theta"]
            elif schedule == "sgapp":  # applicability of HELD sub-goals (precondition share)
                frac = applicable_fraction(held.get("subgoals"), r["snap"])
                fire = frac < kw["theta"]
            elif schedule == "sgappcusum":  # accumulated inapplicability, debounced by k
                frac = applicable_fraction(held.get("subgoals"), r["snap"])
                S = max(0.0, S + (1.0 - frac) - kw["k"])
                fire = S >= kw["h"]
            elif schedule == "sgcusum":  # sub-goal progress under HELD guidance
                fire = False
                if prev_rec is not None:
                    F = grf_shaping(held.get("subgoals"), prev_rec["snap"], r["snap"], prev_rec.get("actions"))
                    n_since += 1
                    if n_since <= kw["W"]:
                        ref_sum += F
                    else:
                        ref = ref_sum / kw["W"]
                        # one-sided: progress drops below the post-issuance reference (or turns negative)
                        S = max(0.0, S + (ref - F) - kw["k"])
                        fire = S >= kw["h"]
            elif schedule in ("cusum", "threshold"):
                z = float(np.abs(phi_vec(r["phi"], types) - phi0).sum())
                if schedule == "cusum":
                    S = max(0.0, S + z - kw["k"])
                    fire = S >= kw["h"]
                else:  # ETC-style single-shot threshold, no accumulation
                    fire = z >= kw["h"]
            pending = pending or fire
        if is_sp and pending:
            held, last_t, calls, pending = r["shadow"]["guidances"][0], t, calls + 1, False
            S, phi0 = 0.0, phi_vec(r["phi"], types)
            d0 = drift_vec(r["phi"]) if "possession" in r["phi"] else None
            n_since, ref_sum = 0, 0.0
            hist = []
            refresh_ts.append(t)
        if is_sp and held is not None:
            ds.append(dist(held, r["shadow"]["guidances"][0], r))
        prev_rec = r
    nus = nu_times(steps)
    delays = []
    for nu in nus:
        after = [rt for rt in refresh_ts if rt >= nu]
        delays.append((after[0] - nu) if after else (steps[-1]["t"] - nu))
    fa = sum(1 for rt in refresh_ts if not any(0 <= rt - nu <= 20 for nu in nus))
    return (calls, mean_of(ds, "hard_diff"), mean_of(ds, "prefer_diff"), mean_of(ds, "forbid_conflict"),
            float(np.mean(delays)) if delays else float("nan"), fa)


def nu_times(steps):
    """GRF: start of each possession transition = first step the previous settled
    side lost the ball, counted only if the ball then settles on the other side.
    SMAC: major events."""
    out, last, left_at = [], None, None
    for r in steps:
        poss = r["phi"].get("possession")
        if poss is None:
            if major_events(r["events"]):
                out.append(r["t"])
            continue
        if last is None:
            if poss in ("ours", "theirs"):
                last = poss
            continue
        if poss == last:
            left_at = None
        elif left_at is None:
            left_at = r["t"]
        if poss in ("ours", "theirs") and poss != last:
            out.append(left_at if left_at is not None else r["t"])
            last, left_at = poss, None
    return out


def main(paths):
    eps, ends = load(paths)
    print("episodes:", len(eps), " won:", sum(e["battle_won"] for e in ends.values()),
          " mean len: %.1f" % np.mean([e["length"] for e in ends.values()]))

    # 1. noise floor vs staleness
    noise, stale = [], defaultdict(list)
    for steps in eps.values():
        sp = shadow_points(steps)
        for r in sp:
            if r["shadow"].get("d_noise"):
                noise.append(r["shadow"]["d_noise"])
        for i, ri in enumerate(sp):
            for rj in sp[i + 1:]:
                dl = rj["t"] - ri["t"]
                stale[dl].append(dist(ri["shadow"]["guidances"][0], rj["shadow"]["guidances"][0], rj))
    same_rules, sg_j, n_pairs = 0, [], 0
    for steps in eps.values():
        for r in shadow_points(steps):
            gs = r["shadow"]["guidances"]
            if len(gs) >= 2 and all(gs):
                n_pairs += 1
                same_rules += json.dumps(gs[0]["action_rules"], sort_keys=True) == json.dumps(gs[1]["action_rules"], sort_keys=True)
                a = {(x["predicate"], x.get("unit_type")) for x in gs[0]["subgoals"]}
                b = {(x["predicate"], x.get("unit_type")) for x in gs[1]["subgoals"]}
                sg_j.append(len(a & b) / max(1, len(a | b)))
    print("\n[0] same-state pairs: identical action_rules %.2f, subgoal Jaccard %.2f  (n=%d)" % (
        same_rules / max(1, n_pairs), float(np.mean(sg_j)) if sg_j else float("nan"), n_pairs))
    print("\n[1] effect-space distance (hard_diff / prefer_diff / forbid_conflict)")
    print("  D_noise (same state, 2 samples): %.3f / %.3f / %.3f  (n=%d)" % (
        mean_of(noise, "hard_diff"), mean_of(noise, "prefer_diff"), mean_of(noise, "forbid_conflict"), len(noise)))
    for dl in sorted(stale):
        ds = stale[dl]
        if dl > 300 or (dl % 50 and dl not in (5, 10, 20)):
            continue
        print("  D_stale(delta=%3d): %.3f / %.3f / %.3f  (n=%d)" % (
            dl, mean_of(ds, "hard_diff"), mean_of(ds, "prefer_diff"), mean_of(ds, "forbid_conflict"), len(ds)))

    # 2. event vs non-event windows (consecutive shadow points)
    ev_d, nev_d = [], []
    ev_types = defaultdict(list)
    for steps in eps.values():
        sp = shadow_points(steps)
        by_t = {r["t"]: r for r in steps}
        for a, b in zip(sp, sp[1:]):
            window_events = major_events([e for t in range(a["t"] + 1, b["t"] + 1) for e in by_t[t]["events"]])
            d = dist(a["shadow"]["guidances"][0], b["shadow"]["guidances"][0], b)
            (ev_d if window_events else nev_d).append(d)
            for e in set(x.split(":")[0] for x in window_events):
                ev_types[e].append(d)
    print("\n[2] consecutive-shadow change, windows WITH major events: %.3f / %.3f  (n=%d)" % (
        mean_of(ev_d, "hard_diff"), mean_of(ev_d, "prefer_diff"), len(ev_d)))
    print("                                 windows WITHOUT:    %.3f / %.3f  (n=%d)" % (
        mean_of(nev_d, "hard_diff"), mean_of(nev_d, "prefer_diff"), len(nev_d)))
    for e, ds in sorted(ev_types.items()):
        print("    event %-14s %.3f / %.3f  (n=%d)" % (e, mean_of(ds, "hard_diff"), mean_of(ds, "prefer_diff"), len(ds)))

    # 3. backfire on heuristic actions
    forb_held, forb_fresh, n = 0, 0, 0
    for steps in eps.values():
        sp = shadow_points(steps)
        if not sp:
            continue
        g0 = sp[0]["shadow"]["guidances"][0]
        fresh = None
        for r in steps:
            if r["shadow"] and r["shadow"]["guidances"][0] is not None:
                fresh = r["shadow"]["guidances"][0]
            na, ifc = n_agents_of(r["snap"]), iface_for(r["snap"])
            h0, _ = build_masks(g0.get("action_rules"), r["snap"], ifc, na, r["snap"]["n_actions"])
            hf, _ = build_masks(fresh.get("action_rules"), r["snap"], ifc, na, r["snap"]["n_actions"])
            ctrl = r["snap"].get("controlled")
            for i, a in enumerate(r["actions"] or []):
                unit = r["snap"]["allies"][ctrl[i] if ctrl else i]
                if not unit["alive"] or a is None or a >= r["snap"]["n_actions"]:
                    continue
                n += 1
                forb_held += h0[i, a] == 0
                forb_fresh += hf[i, a] == 0
    print("\n[3] heuristic action forbidden by held(ep-start) mask: %.3f | by freshest mask: %.3f  (n=%d)" % (
        forb_held / max(n, 1), forb_fresh / max(n, 1), n))

    # 5. GRF: phase-contradiction rate of held vs fresh guidance
    ATT = {"move_forward", "sprint", "move_toward_goal"}
    DEF = {"move_back", "move_toward_own_goal"}

    def contradicts(g, possession):
        for rule in g.get("action_rules", []):
            sel = rule.get("applies_to", "all")
            pref = set(rule.get("prefer", []))
            if possession == "theirs" and sel in ("all", "off_ball") and pref & ATT:
                return True
            if possession == "ours" and sel in ("all", "off_ball", "carrier") and pref & DEF:
                return True
        return False

    if any("possession" in r["phi"] for steps in eps.values() for r in steps[:1]):
        by_delta = defaultdict(lambda: [0, 0])
        fresh_c, fresh_n = 0, 0
        for steps in eps.values():
            sp = shadow_points(steps)
            for i, ri in enumerate(sp):
                poss = ri["phi"]["possession"]
                if poss == "loose":
                    continue
                fresh_n += 1
                fresh_c += contradicts(ri["shadow"]["guidances"][0], poss)
                for rj in sp[:i]:
                    dl = ri["t"] - rj["t"]
                    by_delta[dl][1] += 1
                    by_delta[dl][0] += contradicts(rj["shadow"]["guidances"][0], poss)
        print("\n[5] GRF phase-contradiction rate (guidance prefers attack tokens while THEY have the ball, or retreat while WE do)")
        print("  fresh guidance: %.3f  (n=%d)" % (fresh_c / max(1, fresh_n), fresh_n))
        for dl in sorted(by_delta):
            c, m = by_delta[dl]
            if dl <= 200:
                print("  held for delta=%3d: %.3f  (n=%d)" % (dl, c / max(1, m), m))

    # 4. trigger replay
    n_nu = float(np.mean([len(nu_times(steps)) for steps in eps.values()]))
    print("\n[4] trigger replay (mean nu/ep = %.1f): calls/ep | stale hard | prefer | forbid_conflict | delay after nu | false alarms/ep" % n_nu)
    grf_log = any("possession" in r["phi"] for steps in eps.values() for r in steps[:1])
    schedules = [("ep", {}), ("fixed", {"F": 200}), ("fixed", {"F": 100}), ("fixed", {"F": 50}),
                 ("fixed", {"F": 25}), ("fixed", {"F": 10}), ("event", {})]
    if grf_log:
        schedules.append(("event2", {}))
    for k in (0.1, 0.3):
        for h in (2.0, 4.0, 8.0, 16.0, 32.0):
            schedules.append(("cusum", {"k": k, "h": h}))
    for h in (1.0, 2.0, 3.0):
        schedules.append(("threshold", {"h": h}))
    if any("possession" in r["phi"] for steps in eps.values() for r in steps[:1]):
        for k in (0.05, 0.1):
            for h in (2.0, 4.0, 8.0):
                schedules.append(("hybrid", {"k": k, "h": h}))
        pass
        for th in (0.3, 0.5, 0.7):
            schedules.append(("sgapp", {"theta": th}))
        for k in (0.2, 0.5):
            for h in (1.0, 3.0, 6.0):
                schedules.append(("sgappcusum", {"k": k, "h": h}))
    if any("possession" in r["phi"] for steps in eps.values() for r in steps[:1]):
        for th in (0.3, 0.5):
            schedules.append(("sgapp_learned", {"theta": th}))
        for k in (0.3, 0.5, 0.7):
            for h in (2.0, 4.0):
                schedules.append(("sgapp_learned", {"k": k, "h": h}))
    for W in (10,):
        for k in (0.3, 0.5, 0.7):
            for h in (2.0, 4.0, 8.0):
                schedules.append(("gp", {"W": W, "k": k, "h": h}))
    tabs_cache = {key: None for key in eps}
    gp_cache = {key: None for key in eps}
    for name, kw in schedules:
        if name == "gp":
            rows = []
            for key, steps in eps.items():
                if gp_cache[key] is None:
                    gp_cache[key] = progress_tables(eps, key)
                rows.append(replay(steps, name, tabs=gp_cache[key], **kw))
        elif name == "sgapp_learned":
            rows = []
            for key, steps in eps.items():
                if tabs_cache[key] is None:
                    tabs_cache[key] = firing_tables(eps, key)
                rows.append(replay(steps, name, tabs=tabs_cache[key], **kw))
        else:
            rows = [replay(steps, name, **kw) for steps in eps.values()]
        rows = [r for r in rows if r]
        c, hd, pd, fc, dl, fa = (np.mean([r[i] for r in rows]) for i in range(6))
        kws = {k: v for k, v in kw.items() if k != "tabs"}
        print("  %-13s %-20s %6.1f | %.3f | %.3f | %.3f | %6.1f | %5.1f" % (name, json.dumps(kws), c, hd, pd, fc, dl, fa))


if __name__ == "__main__":
    main(sys.argv[1:])
