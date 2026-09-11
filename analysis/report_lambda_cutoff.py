"""Report the finite-lambda experiment from a small, explicit W&B run set."""
from pathlib import Path
import json

import numpy as np
import wandb

ENTITY = "joonhuk6598-university-of-seoul"
PROJECT = "AAMAS_SMAC_5m_vs_6m"
RUNS = {
    "fixed_zero": {0: "dyzy1yhz", 1: "bs85arnh"},
    "rsvp_zero": {0: "7gdchfle", 1: "ryeb44k9"},
    "qmix_paper": {0: "bjsa2n1e", 1: "h8ytjf5q"},
}


def points(history, key):
    return [(int(row["t_env"]), float(row[key])) for row in history
            if row.get("t_env") is not None and row.get(key) is not None]


api = wandb.Api(timeout=120)
results = {}
for arm, seeds in RUNS.items():
    results[arm] = {}
    for seed, run_id in seeds.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        history = run.history(samples=10000, pandas=False)
        record = {"id": run_id, "state": run.state}
        for key in ("test/battle_won_mean", "test/return_mean",
                    "test/dead_enemies_mean", "test/dead_allies_mean",
                    "train/lehca_lambda", "train/sched_refresh_per_ep",
                    "train/sched_early_per_ep", "train/sched_fallback_per_ep",
                    "train/lambda_shaping_to_env_abs"):
            pts = points(history, key)
            if not pts:
                continue
            t = np.asarray([x[0] for x in pts])
            y = np.asarray([x[1] for x in pts])
            n_final = max(1, len(y) // 10)
            record[key] = {
                "n": len(y), "last_t": int(t[-1]), "mean": float(y.mean()),
                "final10": float(y[-n_final:].mean()), "last": float(y[-1]),
                "pre300": float(y[t <= 300000].mean()) if np.any(t <= 300000) else None,
                "post300": float(y[t > 300000].mean()) if np.any(t > 300000) else None,
                "post300_max": float(y[t > 300000].max()) if np.any(t > 300000) else None,
            }

        matches = list(Path("results/guidance").glob(
            f"rsvp__*lambda_cutoff_20260910_5m_vs_6m_{arm}_s{seed}_*.jsonl"))
        calls = []
        for path in matches:
            with path.open() as handle:
                calls.extend(json.loads(line) for line in handle if line.strip())
        record["guidance_lines"] = len(calls)
        record["guidance_early"] = sum(bool(row.get("early")) for row in calls)
        record["guidance_after_300k"] = sum(
            int(row.get("t_global", row.get("t_env", 0))) > 300000 for row in calls)
        record["max_guidance_t"] = max(
            (int(row.get("t_global", row.get("t_env", 0))) for row in calls), default=None)
        results[arm][seed] = record

print(json.dumps(results, indent=2, sort_keys=True))
