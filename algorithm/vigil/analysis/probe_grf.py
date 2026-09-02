"""E1-lite for GRF: bot-vs-bot trajectories (our 4 field players are driven by
the built-in AI via the `builtin_ai` action of action_set v2), LLM shadow
calls every k steps (cache off), no guidance applied. Logs per step:
snapshot, avail (all 19), bot actions taken by the built-in AI are not
observable, so backfire is measured against the guidance pair only.

Usage (grf env, repo root):
  python algorithm/vigil/probe_grf.py --episodes 10 --api http://n020:8356/v1
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from env.gfootball import GFootballEnv  # noqa: E402
from env.semantic.grf import GRFSemanticInterface  # noqa: E402
from algorithm.vigil.commander.grf import GRFLLMCommander  # noqa: E402
from algorithm.vigil.analysis.effect import mask_distance  # noqa: E402

BUILTIN_AI = 19  # action_set v2: index 19 = builtin_ai


class BotEnv(GFootballEnv):
    """GFootballEnv with action_set v2 so the built-in AI can drive our players."""

    def __init__(self, **kw):
        import gfootball.env as football_env
        self.n_agents = kw.get("n_agents", 4)
        self.episode_limit = kw.get("episode_limit", 1000)
        self._env = football_env.create_environment(
            env_name=kw.get("scenario", "5_vs_5"), representation="raw", stacked=False,
            rewards="scoring", logdir="", write_goal_dumps=False,
            write_full_episode_dumps=False, render=False,
            number_of_left_players_agent_controls=self.n_agents,
            other_config_options={"action_set": "v2",
                                  "right_team_difficulty": float(kw.get("difficulty", 0.05))})
        self._seed = kw.get("seed")
        if self._seed is not None:
            self._env.seed(self._seed)
        self.last_raw = None
        self._t = 0


def bot_actions(snap, n_agents):
    """Built-in AI for everyone, except: when we do not have the ball, the
    controlled player nearest to the ball presses it (move toward ball + sprint).
    When we have it, our controlled carrier drives forward and shoots in the box.
    Avoids bot-vs-bot deadlocks (carrier waits, nobody presses)."""
    from env.semantic.grf import _quantize
    acts = [BUILTIN_AI] * n_agents
    ctrl = [snap["allies"][i] for i in snap["controlled"]]
    if snap["possession"] == "ours":
        # our carrier (if controlled) drives toward their goal and shoots in the box
        for k, a in enumerate(ctrl):
            if a["has_ball"]:
                acts[k] = 12 if a["x"] > 0.7 else (_quantize(1.0 - a["x"], 0.0 - a["y"]) or BUILTIN_AI)
        return acts
    j = min(range(n_agents), key=lambda k: ctrl[k]["dist_ball"])
    me, b = ctrl[j], snap["ball"]
    a = _quantize(b["x"] - me["x"], b["y"] - me["y"])
    acts[j] = a if a is not None else BUILTIN_AI
    return acts


def phi(snap):
    s = snap["shape"]
    return {"possession": snap["possession"], "ball_x": snap["ball"]["x"], "ball_y": snap["ball"]["y"],
            "ours_behind": s["ours_behind_ball"], "theirs_in_our_third": s["theirs_in_our_third"],
            "nearest_enemy_to_ball": s["nearest_enemy_to_ball"], "set_piece": snap["game_mode"] != "normal",
            "zone": snap["ball"]["zone"]}


def events(prev, cur):
    ev = []
    if prev is None:
        return ev
    if prev["possession"] != cur["possession"]:
        ev.append("possession:%s->%s" % (prev["possession"], cur["possession"]))
    if prev["set_piece"] != cur["set_piece"]:
        ev.append("set_piece:%s" % cur["set_piece"])
    if prev["zone"] != cur["zone"]:
        ev.append("zone:%s->%s" % (prev["zone"], cur["zone"]))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="5_vs_5")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-limit", type=int, default=1000)
    ap.add_argument("--shadow-every", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--api", default="http://n020:8356/v1")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=0.05, help="right (bot) team difficulty")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env = BotEnv(scenario=a.scenario, n_agents=4, episode_limit=a.episode_limit, seed=a.seed,
                 difficulty=a.difficulty)
    largs = SimpleNamespace(llm_api_base=a.api, llm_model=a.model, llm_temperature=a.temperature,
                            llm_max_tokens=3072, llm_timeout=90, llm_cache=False,
                            llm_reasoning_effort="low", prompt_style="default")
    iface = GRFSemanticInterface(env, largs)
    commanders = [GRFLLMCommander(largs, iface) for _ in range(a.repeats)]
    pool = ThreadPoolExecutor(max_workers=a.repeats)

    out = a.out or os.path.join(REPO, "results", "vigil",
                                "probe_grf_%s_%s.jsonl" % (a.scenario, time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fo = open(out, "a")
    print("logging to", out, flush=True)

    for ep in range(a.episodes):
        env.reset()
        iface.reset_episode()
        t, term, prev, g0, n_calls = 0, False, None, None, 0
        while not term:
            snap = iface.snapshot()
            avail = [[1] * 19 for _ in range(env.n_agents)]
            ph = phi(snap)
            ev = events(prev, ph)
            rec = {"ep": ep, "t": t, "phi": ph, "events": ev, "snap": snap, "avail": avail,
                   "cache_key": iface.cache_key(snap), "shadow": None}
            if t % a.shadow_every == 0:
                summary = iface.summary(snap)
                key = iface.cache_key(snap)
                t0 = time.time()
                gs = [f.result() for f in [pool.submit(c, summary, key, iface) for c in commanders]]
                n_calls += len(gs)
                if g0 is None and gs[0] is not None:
                    g0 = gs[0]
                sh = {"summary": summary, "guidances": gs, "latency": time.time() - t0}
                if len(gs) >= 2 and all(gs):
                    sh["d_noise"] = mask_distance(gs[0], gs[1], snap, avail, iface, env.n_agents)
                if g0 is not None and gs[0] is not None:
                    sh["d_stale_ep0"] = mask_distance(g0, gs[0], snap, avail, iface, env.n_agents)
                rec["shadow"] = sh
            acts = bot_actions(snap, env.n_agents)
            reward, term, info = env.step(acts)
            iface.tick(snap)
            rec["reward"] = reward
            rec["actions"] = acts
            fo.write(json.dumps(rec) + "\n")
            prev = ph
            t += 1
        fo.write(json.dumps({"ep": ep, "end": True, "length": t, "battle_won": bool(info.get("battle_won")),
                             "goal_diff": info.get("goal_diff"), "llm_calls": n_calls}) + "\n")
        fo.flush()
        print("ep %d len %d goal_diff %s calls %d" % (ep, t, info.get("goal_diff"), n_calls), flush=True)
    env.close()
    fo.close()


if __name__ == "__main__":
    main()
