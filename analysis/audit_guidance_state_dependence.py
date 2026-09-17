"""Audit state dependence and action-rule prevalence in guidance JSONL logs.

Duplicate subgoals are collapsed by semantic key with maximum weight so the
reported variation is not an artifact of the legacy duplicate-reward bug.
The state R^2 is descriptive: it is computed only over cache keys observed at
least ``--min-count`` times and must not be interpreted as causal evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json

import numpy as np


def canonical_subgoals(guidance):
    result = {}
    for subgoal in guidance.get("subgoals") or []:
        key = (subgoal.get("predicate"),
               (subgoal.get("unit_type") or "").strip().casefold())
        result[key] = max(result.get(key, 0.0),
                          float(subgoal.get("weight", 0.0)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="guidance JSONL globs")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--valid-unit-types", default="",
                        help="comma-separated ally/enemy type names")
    args = parser.parse_args()
    paths = sorted({path for pattern in args.patterns for path in glob.glob(pattern)})

    rows = []
    for path in paths:
        with open(path) as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    guidance = item.get("guidance") or {}
                except (json.JSONDecodeError, AttributeError):
                    continue
                rows.append((item.get("cache_key", ""),
                             canonical_subgoals(guidance), guidance))

    vocabulary = sorted({key for _, vector, _ in rows for key in vector})
    vectors = np.asarray([[vector.get(key, 0.0) for key in vocabulary]
                          for _, vector, _ in rows])
    by_state = defaultdict(list)
    for index, (state, _, _) in enumerate(rows):
        by_state[state].append(index)
    eligible = [indices for indices in by_state.values()
                if len(indices) >= args.min_count]
    selected = np.asarray([index for indices in eligible for index in indices],
                          dtype=int)

    state_r2 = float("nan")
    if len(selected) and vectors.shape[1]:
        subset = vectors[selected]
        grand = subset.mean(axis=0)
        total = np.mean(np.sum((subset - grand) ** 2, axis=1))
        within = sum(np.sum((vectors[indices] -
                             vectors[indices].mean(axis=0)) ** 2)
                     for indices in eligible) / len(selected)
        if total > 0:
            state_r2 = 1 - within / total

    predicates = Counter(key[0] for _, vector, _ in rows for key in vector)
    forbids, prefers = Counter(), Counter()
    records_with_forbid = 0
    valid_types = {item.strip().casefold()
                   for item in args.valid_unit_types.split(",") if item.strip()}
    grounding = Counter()
    for _, _, guidance in rows:
        current_forbids = [token for rule in guidance.get("action_rules") or []
                           for token in rule.get("forbid") or []]
        current_prefers = [token for rule in guidance.get("action_rules") or []
                           for token in rule.get("prefer") or []]
        records_with_forbid += bool(current_forbids)
        forbids.update(current_forbids)
        prefers.update(current_prefers)
        if valid_types:
            for subgoal in guidance.get("subgoals") or []:
                if subgoal.get("predicate") in {
                        "kill_type", "damage_type", "protect_type"}:
                    grounding["typed_subgoals"] += 1
                    unit_type = str(subgoal.get("unit_type", "")).strip().casefold()
                    grounding["invalid_typed_subgoals"] += unit_type not in valid_types
            for rule in guidance.get("action_rules") or []:
                for kind in ("forbid", "prefer"):
                    for token in rule.get(kind) or []:
                        if token.startswith("attack_type:"):
                            grounding["attack_type_tokens"] += 1
                            unit_type = token.split(":", 1)[1].strip().casefold()
                            grounding["invalid_attack_type_tokens"] += unit_type not in valid_types

    unique_vectors = len({tuple(vector) for vector in vectors}) if len(rows) else 0
    print(f"files={len(paths)} records={len(rows)} states={len(by_state)}")
    print(f"eligible_states={len(eligible)} eligible_calls={len(selected)} "
          f"state_r2={state_r2:.4f} unique_vectors={unique_vectors}")
    print(f"records_with_forbid={records_with_forbid}/{max(len(rows), 1)} "
          f"({records_with_forbid/max(len(rows), 1):.1%})")
    print("predicates=" + repr(predicates.most_common()))
    print("forbids=" + repr(forbids.most_common()))
    print("prefers=" + repr(prefers.most_common()))
    if valid_types:
        print("grounding=" + repr(dict(grounding)))


if __name__ == "__main__":
    main()
