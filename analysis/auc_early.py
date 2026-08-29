"""Compute the paper's evaluation metrics from wandb runs.

AUC_early (Eq. 9): trapezoidal integral of test win rate over the first 20%
of training steps, normalized by T_early -> value in [0, 1].

Usage:
    python tools/auc_early.py <group> [--project AAMAS-LEHCA]
                              [--entity joonhuk6598-university-of-seoul]
                              [--metric test_battle_won_mean] [--frac 0.2]
"""
import argparse

import numpy as np
import wandb


def auc_early(ts, ys, t_max, frac=0.2):
    t_early = frac * t_max
    pts = [(0.0, 0.0)] + [(t, y) for t, y in zip(ts, ys) if t <= t_early]
    ts_, ys_ = zip(*pts)
    return float(np.trapz(ys_, ts_) / t_early)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("group")
    p.add_argument("--project", default="AAMAS-LEHCA")
    p.add_argument("--entity", default="joonhuk6598-university-of-seoul")
    p.add_argument("--metric", default="test_battle_won_mean")
    p.add_argument("--frac", type=float, default=0.2)
    args = p.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}",
                    filters={"group": args.group})
    aucs, finals = [], []
    for run in runs:
        t_max = run.config.get("t_max")
        rows = [r for r in run.scan_history(keys=[args.metric, "t_env"])
                if r.get(args.metric) is not None and r.get("t_env") is not None]
        if not rows or not t_max:
            print(f"  {run.name}: no data, skipped")
            continue
        rows.sort(key=lambda r: r["t_env"])
        ts = np.array([r["t_env"] for r in rows], dtype=float)
        ys = np.array([r[args.metric] for r in rows], dtype=float)
        a = auc_early(ts, ys, t_max, args.frac)
        final = float(np.mean(ys[ts >= 0.9 * ts[-1]]))  # mean over last 10%
        aucs.append(a)
        finals.append(final)
        print(f"  {run.name}: AUC_early={a:.4f}  final_win_rate={final:.4f}")

    if aucs:
        print(f"\n{args.group} (n={len(aucs)} runs)")
        print(f"  AUC_early      mean {np.mean(aucs):.4f}  std {np.std(aucs):.4f}")
        print(f"  final win rate mean {np.mean(finals):.4f}  std {np.std(finals):.4f}")


if __name__ == "__main__":
    main()
