"""Quantify duplicate semantic sub-goals in Commander JSONL logs."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json


def semantic_key(subgoal):
    unit_type = (subgoal.get("unit_type") or "").strip().casefold()
    return subgoal.get("predicate"), unit_type


def audit(paths):
    stats = Counter()
    duplicate_keys = Counter()
    for path in paths:
        with open(path) as stream:
            for line in stream:
                try:
                    subgoals = (json.loads(line).get("guidance", {})
                                .get("subgoals") or [])
                except (json.JSONDecodeError, AttributeError):
                    continue
                stats["records"] += 1
                if not subgoals:
                    continue
                stats["with_subgoals"] += 1
                stats["entries"] += len(subgoals)
                grouped = {}
                for subgoal in subgoals:
                    grouped.setdefault(semantic_key(subgoal), []).append(
                        float(subgoal.get("weight", 0.0)))
                duplicate_count = sum(len(values) - 1
                                      for values in grouped.values())
                if duplicate_count:
                    stats["duplicate_records"] += 1
                stats["duplicate_entries"] += duplicate_count
                stats["raw_weight_milli"] += round(
                    1000 * sum(sum(values) for values in grouped.values()))
                stats["dedup_weight_milli"] += round(
                    1000 * sum(max(values) for values in grouped.values()))
                for key, values in grouped.items():
                    if len(values) > 1:
                        duplicate_keys[key] += 1
    return stats, duplicate_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="guidance JSONL globs")
    args = parser.parse_args()
    paths = sorted({path for pattern in args.patterns for path in glob.glob(pattern)})
    stats, keys = audit(paths)
    with_subgoals = max(stats["with_subgoals"], 1)
    entries = max(stats["entries"], 1)
    dedup_weight = stats["dedup_weight_milli"] / 1000
    print(f"files={len(paths)} records={stats['records']} "
          f"with_subgoals={stats['with_subgoals']}")
    print(f"duplicate_guidance={stats['duplicate_records']}/"
          f"{stats['with_subgoals']} "
          f"({stats['duplicate_records']/with_subgoals:.1%})")
    print(f"duplicate_entries={stats['duplicate_entries']}/"
          f"{stats['entries']} ({stats['duplicate_entries']/entries:.1%})")
    print(f"raw/deduplicated_max_weight="
          f"{stats['raw_weight_milli']/1000:.1f}/{dedup_weight:.1f} "
          f"({stats['raw_weight_milli']/1000/max(dedup_weight, 1e-9):.3f}x)")
    print("duplicate_keys=" + ", ".join(
        f"{predicate}:{unit_type or '*'}={count}"
        for (predicate, unit_type), count in keys.most_common()))


if __name__ == "__main__":
    main()
