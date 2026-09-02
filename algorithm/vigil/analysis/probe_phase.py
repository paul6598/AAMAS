"""E1-lite: policy-free probe of premise P1 (does LLM guidance depend on the
in-episode phase?) and of the raw material for CUSUM replay.

Runs SMAC with the built-in heuristic ally AI (attack nearest), so no trained
policy is needed. Every `--shadow-every` steps the LLM Commander is called
TWICE on the current summary (cache off) and the guidance is logged but never
applied. Every step logs the snapshot, avail actions, heuristic actions and
phase events, so any refresh trigger (fixed F, event, CUSUM) can be replayed
offline against the recorded LLM outputs.

Usage (repo root, aamas env):
  python algorithm/vigil/probe_phase.py --map 2s3z --episodes 20 \
      --api http://n020:8356/v1 --out results/vigil/probe_2s3z.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from smac.env import StarCraft2Env  # noqa: E402
from env.semantic.sc2 import SC2SemanticInterface  # noqa: E402
from algorithm.lehca.commander.base import set_vocab_mode  # noqa: E402
from algorithm.lehca.commander.llm_commander import LLMCommander  # noqa: E402
from algorithm.vigil.analysis.effect import mask_distance  # noqa: E402


def force_profile(units):
    prof = {}
    for u in units:
        p = prof.setdefault(u["type"], {"alive": 0, "hp": 0.0, "hp_max": 0.0})
        if u["alive"]:
            p["alive"] += 1
            p["hp"] += u["hp"]
            p["hp_max"] += u["hp_max"]
    return prof


def centroid(units):
    pts = [(u["x"], u["y"]) for u in units if u["alive"]]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def phi(snap):
    """Low-dim feature vector the LLM summary is a deterministic function of."""
    vis = [e for e in snap["enemies"] if e.get("visible", True)]
    out = {"n_ally": sum(u["alive"] for u in snap["allies"]),
           "n_enemy": sum(u["alive"] for u in snap["enemies"]),
           "n_enemy_vis": sum(u["alive"] for u in vis)}
    for side, units in (("A", snap["allies"]), ("E", snap["enemies"]), ("V", vis)):
        for t, p in sorted(force_profile(units).items()):
            out["%s_%s_alive" % (side, t)] = p["alive"]
            out["%s_%s_hp" % (side, t)] = p["hp"] / p["hp_max"] if p["hp_max"] > 0 else 0.0
            out["%s_%s_bucket" % (side, t)] = int(4 * p["hp"] / p["hp_max"]) if p["hp_max"] > 0 else -1
    ca, cv = centroid(snap["allies"]), centroid(vis)
    out["dist"] = math.hypot(ca[0] - cv[0], ca[1] - cv[1]) if (ca and cv) else None
    out["phase"] = (None if out["dist"] is None else ("engaged" if out["dist"] < 8 else "approaching"))
    return out


def events(prev, cur):
    """Phase-transition markers between consecutive steps (phi dicts)."""
    ev = []
    if prev is None:
        return ev
    if cur["n_ally"] < prev["n_ally"]:
        ev.append("ally_death")
    if cur["n_enemy"] < prev["n_enemy"]:
        ev.append("enemy_death")
    for k in cur:
        if k.endswith("_alive") and prev.get(k, 0) > 0 and cur[k] == 0:
            ev.append("eliminated:" + k[:-6])
        if k.endswith("_bucket") and k in prev and prev[k] != cur[k]:
            ev.append("bucket:" + k[:-7])
    if prev["n_ally"] >= prev["n_enemy"] and cur["n_ally"] < cur["n_enemy"]:
        ev.append("force_flip")
    if prev["phase"] != cur["phase"]:
        ev.append("phase:%s->%s" % (prev["phase"], cur["phase"]))
    if prev["n_enemy_vis"] == 0 and cur["n_enemy_vis"] > 0:
        ev.append("first_sight")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="2s3z")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--shadow-every", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--api", default="http://n020:8356/v1")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(os.path.join(REPO, "config", "envs", "sc2.yaml")) as f:
        env_args = yaml.safe_load(f)["env_args"]
    env_args.update(map_name=a.map, heuristic_ai=True, seed=a.seed)
    env = StarCraft2Env(**env_args)
    info = env.get_env_info()
    n_agents = info["n_agents"]

    largs = SimpleNamespace(
        dt_observable=True, mask_vocab="full", llm_api_base=a.api, llm_model=a.model,
        llm_temperature=a.temperature, llm_max_tokens=3072, llm_timeout=90,
        llm_cache=False, llm_reasoning_effort="low", prompt_style="default")
    set_vocab_mode("full")
    iface = SC2SemanticInterface(env, largs)
    commanders = [LLMCommander(largs, iface) for _ in range(a.repeats)]
    pool = ThreadPoolExecutor(max_workers=a.repeats)

    out = a.out or os.path.join(REPO, "results", "vigil",
                                "probe_%s_%s.jsonl" % (a.map, time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fo = open(out, "a")
    print("logging to", out, flush=True)

    for ep in range(a.episodes):
        env.reset()
        t, term = 0, False
        prev_phi, g_ep0 = None, None
        n_calls = 0
        while not term:
            snap = iface.snapshot()
            avail = env.get_avail_actions()
            ph = phi(snap)
            ev = events(prev_phi, ph)
            rec = {"ep": ep, "t": t, "phi": ph, "events": ev, "snap": snap,
                   "avail": avail, "cache_key": iface.cache_key(snap), "shadow": None}

            if t % a.shadow_every == 0:
                summary = iface.summary(snap)
                key = iface.cache_key(snap)
                t0 = time.time()
                futs = [pool.submit(c, summary, key, iface) for c in commanders]
                gs = [f.result() for f in futs]
                lat = time.time() - t0
                n_calls += len(gs)
                if g_ep0 is None and gs[0] is not None:
                    g_ep0 = gs[0]
                sh = {"summary": summary, "guidances": gs, "latency": lat}
                if all(g is not None for g in gs) and len(gs) >= 2:
                    sh["d_noise"] = mask_distance(gs[0], gs[1], snap, avail, iface, n_agents)
                if g_ep0 is not None and gs[0] is not None:
                    sh["d_stale_ep0"] = mask_distance(g_ep0, gs[0], snap, avail, iface, n_agents)
                rec["shadow"] = sh

            actions = [0] * n_agents  # ignored: heuristic_ai overwrites in place
            reward, term, env_info = env.step(actions)
            rec["actions"] = [int(x) for x in actions]
            rec["reward"] = float(reward)
            fo.write(json.dumps(rec) + "\n")
            prev_phi = ph
            t += 1
        fo.write(json.dumps({"ep": ep, "end": True, "length": t,
                             "battle_won": bool(env_info.get("battle_won", False)),
                             "llm_calls": n_calls}) + "\n")
        fo.flush()
        print("ep %d len %d won %s calls %d" % (ep, t, env_info.get("battle_won"), n_calls), flush=True)

    env.close()
    fo.close()


if __name__ == "__main__":
    main()
