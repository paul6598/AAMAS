"""Export selected Sacred curves and explicit completion evidence without W&B.

Example: python analysis/summarize_experiments.py --runs 357 356 379 380 \
    --jobs 976091 976094 982780 982781 --output results/summary.json
Evaluation progress alone never proves that training completed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = (
    "name", "env", "seed", "t_max", "wandb_group", "wandb_run", "scheduler",
    "f_update", "commander", "epsilon_anneal_time", "use_reward_shaping",
    "use_action_masking", "use_masking_at_test", "soft_guidance_only", "beta",
    "shaping_in_learner", "lambda_start", "lambda_min", "lambda_decay",
    "lambda_floor_frac", "lambda_zero_t", "deduplicate_subgoals", "prompt_style",
    "commander_include_training_stats", "llm_model", "llm_temperature", "llm_cache",
    "sched_gamma", "sched_k", "sched_h", "sched_min_interval", "sched_gate_interval",
    "sched_target_early_per_ep", "rsvp_validation_every", "rsvp_refresh_trial",
    "rsvp_trial_start_t", "rsvp_trial_window_episodes", "rsvp_trial_max_per_kind",
    "test_interval", "test_nepisode",
)


def scalar(value):
    return float(value["value"] if isinstance(value, dict) else value)


def series(info, key):
    times, values = info.get(key + "_T", []), info.get(key, [])
    if len(times) != len(values):
        raise ValueError(f"unaligned series: {key}")
    points = sorted({int(t): scalar(v) for t, v in zip(times, values)}.items())
    if any(not np.isfinite(v) for _, v in points):
        raise ValueError(f"non-finite series: {key}")
    return points


def curve_metrics(points, horizon, complete):
    # 관측 구간 적분과 명목 지평의 tail 유지 근사를 서로 구분한다.
    if not points:
        return {}
    observed = min(horizon, points[-1][0])
    inner = [(t, v) for t, v in points if 0 < t < observed]
    end = float(np.interp(observed, *zip(*points)))
    x, y = map(np.asarray, zip(*([(0, 0.0)] + inner + [(observed, end)])))
    integral = float(np.trapezoid(y, x))
    tail = [v for t, v in points if max(0, horizon - 100000) <= t <= horizon]
    tail10 = [v for t, v in points if .9 * horizon <= t <= horizon]
    values = [v for t, v in points if t <= horizon]
    return dict(
        observed_horizon=observed,
        evaluation_mean=float(np.mean(values)) if values else None,
        auc_observed=integral / observed if observed else None,
        auc_nominal_hold_last=(integral + end * (horizon - observed)) / horizon
            if complete else None,
        final100k=float(np.mean(tail)) if complete and tail else None,
        final10pct=float(np.mean(tail10)) if complete and tail10 else None,
        peak=max(values) if values else None,
        last=points[-1][1],
    )


def slurm_states(job_ids):
    output = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(map(str, job_ids)),
        "--format=JobID,State,ExitCode,Elapsed,End",
    ], text=True)
    states = {}
    for line in output.splitlines():
        job, state, code, elapsed, end = line.split("|")[:5]
        states[int(job)] = dict(state=state, exit_code=code, elapsed=elapsed, ended_at=end)
    return states


def summarize_run(directory, job_id=None, slurm=None):
    config_bytes = (directory / "config.json").read_bytes()
    info_bytes = (directory / "info.json").read_bytes()
    config, info = json.loads(config_bytes), json.loads(info_bytes)
    run = json.loads((directory / "run.json").read_text())
    complete = (slurm.get("state") == "COMPLETED" and slurm.get("exit_code") == "0:0") \
        if slurm is not None else run.get("status") == "COMPLETED"
    wins = series(info, "test_battle_won_mean")
    selected = {k: config[k] for k in CONFIG_KEYS if k in config}
    selected["env_args"] = config.get("env_args", {})
    return dict(
        sacred=int(directory.name), job_id=job_id, slurm=slurm,
        sacred_status=run.get("status"), complete=complete,
        config=selected,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        info_sha256=hashlib.sha256(info_bytes).hexdigest(),
        metrics=curve_metrics(wins, int(config["t_max"]), complete),
        last_train_llm_calls=scalar(info["llm_calls"][-1]) if info.get("llm_calls") else 0,
        last_train_llm_failures=scalar(info["llm_failures"][-1]) if info.get("llm_failures") else 0,
        evaluation_curve=wins,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=int, required=True)
    parser.add_argument("--jobs", nargs="+", type=int,
                        help="Slurm IDs in the same order as --runs; requires sacct")
    parser.add_argument("--sacred-root", type=Path, default=ROOT / "results/sacred")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.jobs and len(args.jobs) != len(args.runs):
        parser.error("--jobs must have one ID for each --runs entry")
    states = slurm_states(args.jobs) if args.jobs else {}
    result = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        metric_notes=dict(
            auc_observed="trapezoidal y(0)=0, normalized by observed horizon; no tail extrapolation",
            auc_nominal_hold_last="completed runs only; hold last evaluation to nominal t_max",
            final100k="arithmetic mean of recorded evaluations in [t_max-100000, t_max]",
            final10pct="arithmetic mean in [0.9*t_max, t_max]",
            calls="last periodic training counter; excludes unlogged tail, evaluation, retries and tokens",
            completion="Slurm COMPLETED/0:0 when provided; otherwise explicit Sacred COMPLETED",
        ),
        runs=[summarize_run(args.sacred_root / str(n),
                            args.jobs[i] if args.jobs else None,
                            states.get(args.jobs[i], {}) if args.jobs else None)
              for i, n in enumerate(args.runs)],
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(args.output)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
