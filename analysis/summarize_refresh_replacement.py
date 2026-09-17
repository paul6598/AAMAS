"""Measure whether an RSVP early refresh replaces stale guidance with a better fit.

For shaping-only runs, guidance cannot affect actions before the episode ends.
The same realized transition suffix can therefore score both the old and the new
guidance without another environment rollout.  This is a mechanism diagnostic,
not an estimate of downstream win-rate treatment effect.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _cluster_ci(events, key, n_boot, seed):
    if not events:
        return None
    clusters = {}
    for event in events:
        clusters.setdefault(event["episode"], []).append(event[key])
    ids = list(clusters)
    if len(ids) < 2 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        values = [v for episode in sampled for v in clusters[episode]]
        means[i] = np.mean(values)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def _metric(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_rate": float(np.mean(values > 0)),
        "zero_rate": float(np.mean(np.isclose(values, 0))),
    }


def summarize(path, horizon=5, gamma=.8, start=0, stop=None, bootstrap=10000,
              bootstrap_seed=20260913):
    path = Path(path)
    discount = gamma ** np.arange(horizon, dtype=float)
    discount /= discount.sum()
    metadata, events, partial = None, [], False
    with gzip.open(path, "rt") as handle:
        try:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    partial = True
                    break
                if row.get("kind") == "metadata":
                    metadata = row
                    continue
                if row.get("kind") != "episode":
                    continue
                records = row.get("records") or []
                if not records:
                    continue
                predicates = np.asarray([r["predicates"] for r in records], dtype=float)
                for index, record in enumerate(records):
                    decision = record.get("decision") or {}
                    t = int(record["t"])
                    if not (decision.get("early") and decision.get("refresh_success")):
                        continue
                    if t < start or (stop is not None and t >= stop):
                        continue
                    if index < horizon or index + horizon > len(records):
                        continue
                    old_weights = np.asarray(records[index - 1]["weights"], dtype=float)
                    new_weights = np.asarray(record["weights"], dtype=float)
                    if old_weights.shape != new_weights.shape:
                        raise ValueError("old/new guidance head dimensions differ")

                    def score(window, weights, reverse=False):
                        values = np.clip(predicates[window] @ weights, -3., 3.)
                        if reverse:
                            values = values[::-1]
                        return float(values @ discount)

                    old_pre = score(slice(index - horizon, index), old_weights, True)
                    old_post = score(slice(index, index + horizon), old_weights)
                    new_post = score(slice(index, index + horizon), new_weights)
                    old_mass = max(float(old_weights.sum()), 1e-9)
                    new_mass = max(float(new_weights.sum()), 1e-9)
                    events.append({
                        "episode": int(row["episode"]), "t": t,
                        "score": decision.get("S"), "threshold": decision.get("h"),
                        "old_pre": old_pre, "old_post": old_post, "new_post": new_post,
                        "staleness_drop": old_pre - old_post,
                        "replacement_gain": new_post - old_post,
                        "normalized_replacement_gain": new_post / new_mass - old_post / old_mass,
                        "same_guidance_weights": bool(np.array_equal(old_weights, new_weights)),
                    })
        except EOFError:
            partial = True

    if metadata is None:
        raise ValueError(f"missing metadata in {path}")
    config = metadata.get("config") or {}
    if config.get("use_action_masking") is not False:
        raise ValueError("replacement diagnostic requires shaping-only use_action_masking=false")
    if not events:
        return {"file": str(path), "partial_file": partial, "config": config,
                "horizon": horizon, "gamma": gamma, "events": 0}

    staleness = [e["staleness_drop"] for e in events]
    gain = [e["replacement_gain"] for e in events]
    normalized = [e["normalized_replacement_gain"] for e in events]
    usable_scores = [e for e in events if e["score"] is not None]
    result = {
        "file": str(path), "partial_file": partial, "seed": config.get("seed"),
        "wandb_run": config.get("wandb_run"), "horizon": horizon, "gamma": gamma,
        "events": len(events), "episode_clusters": len({e["episode"] for e in events}),
        "staleness_drop": _metric(staleness),
        "replacement_gain": _metric(gain),
        "normalized_replacement_gain": _metric(normalized),
        "dynamic_success_rate": float(np.mean([
            e["staleness_drop"] > 0 and e["replacement_gain"] > 0 for e in events])),
        "same_guidance_weight_rate": float(np.mean([
            e["same_guidance_weights"] for e in events])),
        "score_staleness_spearman": _spearman(
            [e["score"] for e in usable_scores],
            [e["staleness_drop"] for e in usable_scores]),
        "score_gain_spearman": _spearman(
            [e["score"] for e in usable_scores],
            [e["replacement_gain"] for e in usable_scores]),
    }
    result["staleness_drop"]["cluster_bootstrap_95ci"] = _cluster_ci(
        events, "staleness_drop", bootstrap, bootstrap_seed)
    result["replacement_gain"]["cluster_bootstrap_95ci"] = _cluster_ci(
        events, "replacement_gain", bootstrap, bootstrap_seed + 1)
    result["normalized_replacement_gain"]["cluster_bootstrap_95ci"] = _cluster_ci(
        events, "normalized_replacement_gain", bootstrap, bootstrap_seed + 2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=.8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.horizon <= 0 or not 0 <= args.gamma <= 1:
        parser.error("horizon must be positive and gamma must be in [0, 1]")
    output = [summarize(path, args.horizon, args.gamma, args.start, args.stop,
                        args.bootstrap) for path in args.paths]
    text = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
