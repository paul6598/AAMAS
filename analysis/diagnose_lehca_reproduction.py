"""Read-only reproduction audit; no SC2 launch, LLM requests, or training.

Run with the aamas Python. Outputs JSON to stdout. Uses observed evaluation
points with interpolation at a shared horizon; never extrapolates a short run.
"""
import bisect
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_run(run_id):
    path = ROOT / "results/sacred" / str(run_id)
    return (json.loads((path / "config.json").read_text()),
            json.loads((path / "info.json").read_text()))


def curve(info, key="test_battle_won_mean"):
    return info[key + "_T"], info[key]


def auc(times, values, horizon):
    if times[-1] < horizon:
        return None
    assert all(a < b for a, b in zip(times, times[1:]))
    # Initial value is assumed zero, as in the repository's existing metric.
    points = [(0, 0)] + [(t, y) for t, y in zip(times, values) if 0 < t < horizon]
    i = bisect.bisect_left(times, horizon)
    if times[i] == horizon:
        end = values[i]
    elif i == 0:
        end = values[0] * horizon / times[0]
    else:
        end = values[i-1] + (values[i] - values[i-1]) * (
            horizon - times[i-1]) / (times[i] - times[i-1])
    points.append((horizon, end))
    return sum((b[0]-a[0]) * (a[1]+b[1]) / 2
               for a, b in zip(points, points[1:])) / horizon


def interval_mean(info, key, lo, hi):
    if key not in info:
        return None
    times, values = curve(info, key)
    selected = [v.get("value") if isinstance(v, dict) else v
                for t, v in zip(times, values) if lo <= t <= hi]
    return statistics.mean(selected) if selected else None


def main():
    report = {"metric_notes": {
        "auc": "trapezoidal, y(0)=0, interpolated endpoint, no extrapolation",
        "short_comparison": "0--290000, common observed coverage",
        "tail": "arithmetic mean of evaluations in [0.9*T, T]",
        "paper_5m_early": "T=1000000 (20% of paper 5M budget)",
        "warning": "Sacred RUNNING status can remain stale after Finished Training",
    }}
    run_ids = [142, 153, 176, 163, 188, 191, 242, 244, 249, 250,
               256, 262, 263, 264, 265, 266, 267, 268,
               275, 276, 279, 280, 281, 282, 283, 284, 285, 286]
    report["runs"] = []
    for run_id in run_ids:
        config, info = read_run(run_id)
        times, values = curve(info)
        budget = config["t_max"]
        entry = {
            "sacred": run_id, "seed": config["seed"],
            "name": config.get("wandb_run") or config.get("wandb_group"),
            "map": config["env_args"]["map_name"], "budget": budget,
            "last_eval": times[-1], "auc_290k": auc(times, values, 290000),
            "auc_1m": auc(times, values, 1000000),
            "auc_200k": auc(times, values, 200000),
            "tail_290k": interval_mean(info, "test_battle_won_mean", 261000, 290000),
            "tail_budget": interval_mean(info, "test_battle_won_mean", .9*budget, budget),
            "tail_stats": {},
        }
        for key in ["battle_won_mean", "test_battle_won_mean", "return_mean",
                    "test_return_mean", "ep_length_mean", "test_ep_length_mean",
                    "mask_forbid_frac", "mask_override_rate", "q_gap_mean",
                    "hard_mask_override_rate", "attack_category_removed_rate",
                    "move_category_removed_rate",
                    "shaped_return_mean", "llm_failures"]:
            entry["tail_stats"][key] = interval_mean(info, key, .9*budget, budget)
        report["runs"].append(entry)
    report["identical_curve_pairs"] = []
    for left, right in [(142, 256), (153, 264), (142, 283), (153, 284)]:
        t, y = curve(read_run(left)[1])
        s, z = curve(read_run(right)[1])
        report["identical_curve_pairs"].append({
            "left": left, "right": right, "points": len(t),
            "identical_prefix": t == s[:len(t)] and y == z[:len(y)]})
    report["epsilon_anneal_counts"] = dict(Counter(
        json.loads(p.read_text()).get("epsilon_anneal_time")
        for p in (ROOT / "results/sacred").glob("*/config.json")))

    # Exercise the actual compiler and runner methods without creating an env.
    import numpy as np
    from algorithm.lehca.runner import LehcaRunner
    from algorithm.lehca.masking.compiler import build_masks
    from env.semantic.sc2 import SC2SemanticInterface

    patterns = ["lehca__2026-09-07_14-20*", "lehca__2026-09-08_14-24*",
                "lehca__2026-09-09_16-55*"]
    report["guidance"] = []
    attack_example = None
    for pattern in patterns:
        for path in sorted((ROOT / "results/guidance").glob(pattern)):
            counts = Counter()
            for line in path.open():
                row = json.loads(line)
                g = row.get("guidance")
                if not g:
                    continue
                counts["valid"] += 1
                # All allies on these maps are Marines. Only count rules that
                # apply to every live ally, avoiding cross-agent false conflicts.
                rules = [r for r in g["action_rules"]
                         if r["applies_to"] in ("all", "*", "type:Marine")]
                forbid = {t for r in rules for t in r.get("forbid", [])}
                prefer = {t for r in rules for t in r.get("prefer", [])}
                if "attack_all" in forbid:
                    counts["attack_all_forbid"] += 1
                    if attack_example is None:
                        attack_example = row
                if "move_all" in forbid:
                    counts["move_all_forbid"] += 1
                    if prefer & {"move_north", "move_south", "move_east", "move_west"}:
                        counts["move_all_but_prefer_move"] += 1
            report["guidance"].append({"path": str(path.relative_to(ROOT)),
                                       "counts": dict(counts)})
    snap = json.loads(next((ROOT / "results/diagnostics/grounding_20260907_1232/transitions.jsonl").open()))["pre"]
    iface = object.__new__(SC2SemanticInterface)
    hard, soft = build_masks(attack_example["guidance"]["action_rules"],
                             snap, iface, 5, 12)
    # Synthetic availability: five live Marines with movement and all attacks.
    avail = np.ones((5, 12), dtype=np.float32)
    avail[:, 0] = 0
    allowed = hard * avail
    report["attack_forbid_probe"] = {
        "guidance_source": attack_example,
        "all_attacks_removed": bool((allowed[:, 6:] == 0).all()),
        "fallback_would_fire": bool((allowed.sum(-1) == 0).any()),
        "historical_rule_contains_attack_all_forbid": True,
        "scope": "real logged guidance with current safety compiler, synthetic availability; not a rollout",
    }
    # Regression guard: the historical rule used to erase every attack while
    # leaving movement available, so the controller fallback did not fire.
    # The compiler must now preserve at least one tactical action category.
    assert not report["attack_forbid_probe"]["all_attacks_removed"]
    assert not report["attack_forbid_probe"]["fallback_would_fire"]

    class Spy:
        calls = 0

        def __call__(self, *args):
            self.calls += 1
            return {"strategy": "refreshed", "subgoals": [], "action_rules": []}

    spy = Spy()
    runner = object.__new__(LehcaRunner)
    runner.commander = spy
    runner.eval_commander = spy
    runner.test_guidance_mode = "fresh"
    runner._test_t_env = 0
    runner._test_last_refresh_t = None
    runner._test_guidance = None
    runner.include_training_stats = False
    runner.state = SimpleNamespace(guidance={"strategy": "old training guidance"})
    runner.refresh_at_episode_start = False
    runner.last_refresh_t = None
    runner.f_update = 25
    runner.t_env = 10000
    runner.recent_wins = []
    runner._guidance_log = None
    runner.iface = SimpleNamespace(summary=lambda *a: "summary", cache_key=lambda *a: "key")
    for _ in range(32):
        for step in range(70):
            runner.t = step
            runner._maybe_refresh_commander(snap, test_mode=True)
    report["test_refresh_probe"] = {"test_calls": spy.calls,
                                     "training_guidance": runner.state.guidance,
                                     "evaluation_guidance": runner._test_guidance}
    assert spy.calls > 0
    assert runner.state.guidance == {"strategy": "old training guidance"}
    before_train = spy.calls
    runner.t = 0
    runner._maybe_refresh_commander(snap, test_mode=False)
    report["test_refresh_probe"]["training_positive_control_calls"] = spy.calls - before_train
    assert spy.calls == before_train + 1

    sources = ["config/algs/qmix_paper.yaml", "config/algs/lehca.yaml",
               "config/envs/sc2.yaml", "run.py", "algorithm/lehca/runner.py",
               "algorithm/lehca/controller.py", "algorithm/lehca/commander/base.py",
               "algorithm/lehca/commander/llm_commander.py",
               "algorithm/lehca/shaping/predicates.py", "env/semantic/sc2.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in sources}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
