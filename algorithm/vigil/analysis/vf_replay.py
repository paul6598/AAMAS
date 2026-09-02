"""Offline validation of the shaping remaining-value critic on GRF probe logs.

V_F(s; G) = sum_j w_j V_j(s), where V_j(s) is the discounted future sum of
predicate f_j along the (bot-policy) trajectory — guidance-agnostic heads,
exact composition by linearity of F in the sub-goal weights.

Per leave-one-episode-out fold: train a small MLP s -> (V_1..V_9) on Monte
Carlo targets from the other episodes, then replay the held-out episode with
the trigger  v_t = V_F(s_t;G)/max(V_F(s_r;G), eps)  and one-sided CUSUM
S += k - min(v,1), firing at S >= h (guard: undecidable -> fixed F_max).

Usage (grf env): python algorithm/vigil/vf_replay.py results/vigil/probe_grf_5v5_d06_t0.jsonl
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(2)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from algorithm.vigil.analysis.analyze_probe import load, dist, nu_times, mean_of, shadow_points  # noqa: E402
from algorithm.vigil.shaping.grf import evaluate_predicate, ALL_PREDICATES  # noqa: E402

GAMMA = float(os.environ.get("VF_GAMMA", "0.97"))
EPS_DEN = 0.5
FMAX = 100


def feats(rec):
    s = rec["snap"]
    ph = rec["phi"]
    poss = {"ours": 0, "loose": 1, "theirs": 2}[s["possession"]]
    onehot = [0.0, 0.0, 0.0]
    onehot[poss] = 1.0
    return onehot + [
        s["ball"]["x"], s["ball"]["y"], s["ball"]["dx"] * 50, s["ball"]["dy"] * 50,
        ph["ours_behind"] / 4.0, ph["theirs_in_our_third"] / 4.0,
        min(1.0, s["shape"]["nearest_enemy_to_ball"] * 5),
        min(1.0, s["shape"]["nearest_ally_to_ball"] * 5),
        1.0 if s["game_mode"] != "normal" else 0.0,
        s["time_frac"],
    ]


def episode_arrays(steps):
    X = np.array([feats(r) for r in steps[:-1]], dtype=np.float32)
    Fj = np.array([[evaluate_predicate(p, a["snap"], b["snap"], a.get("actions"))
                    for p in ALL_PREDICATES] for a, b in zip(steps, steps[1:])], dtype=np.float32)
    # discounted future sums per predicate
    Y = np.zeros_like(Fj)
    run = np.zeros(Fj.shape[1], dtype=np.float32)
    for t in range(len(Fj) - 1, -1, -1):
        run = Fj[t] + GAMMA * run
        Y[t] = run
    return X, Y


def train_heads(Xs, Ys, epochs=800):
    X = torch.tensor(np.concatenate(Xs))
    Y = torch.tensor(np.concatenate(Ys))
    mu, sd = Y.mean(0), Y.std(0).clamp(min=1e-3)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(epochs):
        idx = torch.randint(0, len(X), (2048,))
        loss = ((net(X[idx]) - (Y[idx] - mu) / sd) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    def predict(x):
        with torch.no_grad():
            return (net(torch.tensor(x)) * sd + mu).numpy()
    return predict


def vF(pred_row, guidance):
    idx = {p: i for i, p in enumerate(ALL_PREDICATES)}
    return sum(float(sg.get("weight", 0.5)) * pred_row[idx[sg["predicate"]]]
               for sg in guidance.get("subgoals") or [])


def replay_vf(steps, V, k, h):
    sp = shadow_points(steps)
    if not sp:
        return None
    held, S, v_ref, last_t, calls, pending = None, 0.0, None, None, 0, True
    ds, refresh_ts = [], []
    for r, vrow in zip(steps, V):
        t = r["t"]
        is_sp = bool(r["shadow"] and r["shadow"]["guidances"][0])
        if held is not None:
            den = max(v_ref, EPS_DEN) if v_ref is not None else None
            if v_ref is not None and v_ref >= EPS_DEN:
                v = min(1.0, max(0.0, vF(vrow, held)) / den)
                S = max(0.0, S + k - v)
                fire = S >= h
            else:  # undecidable at issuance -> fixed-period fallback
                fire = t - last_t >= FMAX
            pending = pending or fire
        if is_sp and pending:
            held = r["shadow"]["guidances"][0]
            v_ref = vF(vrow, held)
            last_t, calls, pending, S = t, calls + 1, False, 0.0
            refresh_ts.append(t)
        if is_sp and held is not None:
            ds.append(dist(held, r["shadow"]["guidances"][0], r))
    nus = nu_times(steps)
    delays = [((min((rt for rt in refresh_ts if rt >= nu), default=steps[-1]["t"])) - nu) for nu in nus]
    fa = sum(1 for rt in refresh_ts if not any(0 <= rt - nu <= 20 for nu in nus))
    return (calls, mean_of(ds, "hard_diff"), mean_of(ds, "prefer_diff"), mean_of(ds, "forbid_conflict"),
            float(np.mean(delays)) if delays else float("nan"), fa)


def main(paths):
    eps, _ = load(paths)
    keys = list(eps)
    arrays = {}
    for kk, steps in eps.items():
        arrays[kk] = episode_arrays(steps)
        print("arrays", kk[1], flush=True)
    # critic quality: LOO R^2 per predicate (pooled)
    r2n = r2d = 0.0
    hn = hd_ = None
    rows = {kk: None for kk in keys}
    import time
    for kk in keys:
        t0 = time.time()
        Vhat = train_heads([arrays[o][0] for o in keys if o != kk],
                           [arrays[o][1] for o in keys if o != kk])
        X, Y = arrays[kk]
        P = Vhat(X)
        r2n += ((Y - P) ** 2).sum()
        r2d += ((Y - Y.mean(0)) ** 2).sum()
        e2 = ((Y - P) ** 2).sum(0); v2 = ((Y - Y.mean(0)) ** 2).sum(0)
        hn = e2 if hn is None else hn + e2
        hd_ = v2 if hd_ is None else hd_ + v2
        rows[kk] = P
        print("fold ep%s trained %.0fs" % (kk[1], time.time() - t0), flush=True)
    print("critic LOO R^2 (pooled): %.3f   gamma=%s" % (1 - r2n / r2d, GAMMA))
    for i, pr in enumerate(ALL_PREDICATES):
        print("   head %-18s R^2 %.3f" % (pr, 1 - hn[i] / max(hd_[i], 1e-9)))
    print("\nvf replay: calls/ep | stale hard | prefer | forbid | delay | FA")
    for k in (0.3, 0.5, 0.7):
        for h in (2.0, 4.0, 8.0):
            out = []
            for kk in keys:
                steps = eps[kk]
                res = replay_vf(steps[:-1], rows[kk], k, h)
                if res:
                    out.append(res)
            c, hd, pd, fc, dl, fa = (np.mean([r[i] for r in out]) for i in range(6))
            print("  vf k%.1f h%.0f   %6.1f | %.3f | %.3f | %.3f | %6.1f | %5.1f" % (k, h, c, hd, pd, fc, dl, fa))


if __name__ == "__main__":
    main(sys.argv[1:])
