"""Fast regression tests for paper-alignment fixes; no env/LLM launch."""
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithm.lehca.commander import base as commander_base
from algorithm.lehca.commander.llm_commander import (
    GROUNDING_SYSTEM_PROMPT, PAPER_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from algorithm.lehca.masking.compiler import build_masks
from algorithm.lehca.runner import LehcaRunner
from algorithm.lehca.state import LehcaState
from algorithm.rsvp.runner import RSVPRunner
from env.semantic.grf import GRFSemanticInterface
from env.semantic.sc2 import SC2SemanticInterface


class PaperAlignmentTests(unittest.TestCase):
    def tearDown(self):
        commander_base.set_vocab_mode("full")

    def test_prompts_do_not_disclose_true_environment_reward(self):
        prompts = "\n".join((SYSTEM_PROMPT, PAPER_SYSTEM_PROMPT,
                              PLAN_SYSTEM_PROMPT, GROUNDING_SYSTEM_PROMPT))
        self.assertNotIn("environment already", prompts.lower())
        self.assertIn("true reward function", prompts.lower())
        grf_source = inspect.getsource(GRFSemanticInterface.prompt_context)
        self.assertNotIn("already rewards", grf_source.lower())

    def test_sanitizer_removes_unsafe_and_contradictory_forbids(self):
        raw = {
            "strategy": "focus fire",
            "subgoals": [],
            "action_rules": [{
                "applies_to": "all",
                "forbid": ["noop", "stop", "move_all", "attack_all",
                           "attack_type:Marine"],
                "prefer": ["attack_type:Marine", "attack_lowest_health"],
                "prefer_weight": 2.0,
            }],
        }
        clean = commander_base.sanitize_guidance(raw)
        self.assertIsNotNone(clean)
        self.assertEqual(clean["action_rules"][0]["forbid"], [])

    def test_sanitizer_deduplicates_semantically_identical_subgoals(self):
        raw = {
            "strategy": "focus fire",
            "subgoals": [
                {"predicate": "enemy_kill", "weight": 0.4},
                {"predicate": "enemy_kill", "weight": 0.9},
                {"predicate": "kill_type", "unit_type": " Stalker ",
                 "weight": 0.6},
                {"predicate": "kill_type", "unit_type": "stalker",
                 "weight": 0.8},
                {"predicate": "enemy_damage", "weight": 0.7},
            ],
            "action_rules": [],
        }
        clean = commander_base.sanitize_guidance(raw)
        self.assertEqual(clean["subgoals"], [
            {"predicate": "enemy_kill", "weight": 0.9},
            {"predicate": "kill_type", "unit_type": "Stalker",
             "weight": 0.8},
            {"predicate": "enemy_damage", "weight": 0.7},
        ])

    def test_duplicate_subgoals_do_not_consume_unique_goal_limit(self):
        repeated = [{"predicate": "enemy_kill", "weight": 0.5}] * 6
        raw = {
            "strategy": "retain later unique goal",
            "subgoals": repeated + [
                {"predicate": "enemy_damage", "weight": 0.7}],
            "action_rules": [],
        }
        clean = commander_base.sanitize_guidance(raw)
        self.assertEqual([x["predicate"] for x in clean["subgoals"]],
                         ["enemy_kill", "enemy_damage"])

    def test_sanitizer_can_replay_legacy_duplicate_semantics(self):
        raw = {
            "strategy": "legacy control",
            "subgoals": [
                {"predicate": "enemy_kill", "weight": 0.4},
                {"predicate": "enemy_kill", "weight": 0.9},
            ],
            "action_rules": [],
        }
        clean = commander_base.sanitize_guidance(
            raw, deduplicate_subgoals=False)
        self.assertEqual(len(clean["subgoals"]), 2)

    def test_compiler_cannot_erase_entire_tactical_category(self):
        class Iface:
            @staticmethod
            def agent_matches(selector, i, snap):
                return True

            @staticmethod
            def resolve_action_token(token, i, snap):
                if token == "attack_type:Marine":
                    return [6, 7]
                if token == "move_all":
                    return [2, 3, 4, 5]
                return []

        snap = {"allies": [{"alive": True}], "enemies": [], "n_actions": 8}
        rules = [{"applies_to": "all",
                  "forbid": ["attack_type:Marine", "move_all"],
                  "prefer": [], "prefer_weight": 2.0}]
        hard, _ = build_masks(rules, snap, Iface(), 1, 8)
        self.assertTrue(hard[0, 6:].any())
        self.assertTrue(hard[0, 2:6].any())

    def test_action_grounding_uses_visible_targets_only(self):
        iface = object.__new__(SC2SemanticInterface)
        snap = {
            "allies": [{"alive": True, "type": "Marine", "x": 0.0, "y": 0.0}],
            "enemies": [
                {"alive": True, "visible": False, "type": "Marine",
                 "hp": 1.0, "x": 0.1, "y": 0.1},
                {"alive": True, "visible": True, "type": "Marine",
                 "hp": 5.0, "x": 4.0, "y": 4.0},
            ],
            "n_actions": 8,
        }
        self.assertEqual(iface.resolve_action_token("attack_all", 0, snap), [7])
        self.assertEqual(iface.resolve_action_token("attack_lowest_health", 0, snap), [7])
        self.assertEqual(iface.resolve_action_token("attack_nearest", 0, snap), [7])

    def test_eval_refresh_is_isolated_and_receives_no_training_stats(self):
        class Spy:
            def __init__(self):
                self.calls = []

            def __call__(self, summary, key, iface):
                self.calls.append((summary, key))
                return {"strategy": "eval", "subgoals": [], "action_rules": []}

        train, evaluate = Spy(), Spy()
        summaries = []
        runner = object.__new__(LehcaRunner)
        runner.commander = train
        runner.eval_commander = evaluate
        runner.test_guidance_mode = "fresh"
        runner._test_t_env = 0
        runner._test_last_refresh_t = None
        runner._test_guidance = None
        runner.include_training_stats = False
        runner.state = SimpleNamespace(guidance={"strategy": "train"})
        runner.refresh_at_episode_start = False
        runner.last_refresh_t = None
        runner.f_update = 25
        runner.t_env = 1000
        runner.t = 0
        runner.recent_wins = [1.0]
        runner._guidance_log = None
        runner.iface = SimpleNamespace(
            summary=lambda snap, stats: summaries.append(stats) or "observable",
            cache_key=lambda snap: "state",
        )

        runner._maybe_refresh_commander({}, test_mode=True)
        self.assertEqual(len(evaluate.calls), 1)
        self.assertEqual(len(train.calls), 0)
        self.assertEqual(runner.state.guidance, {"strategy": "train"})
        self.assertEqual(runner._test_guidance["strategy"], "eval")
        self.assertEqual(summaries, [None])

    def test_cosine_lambda_reaches_exact_zero(self):
        state = LehcaState()
        state.configure(SimpleNamespace(
            lambda_start=0.5, lambda_min=0.05, lambda_decay=0.9995))
        state.set_lambda_cosine_zero(0, 300_000)
        self.assertEqual(state.lambda_val, 0.5)
        state.set_lambda_cosine_zero(150_000, 300_000)
        self.assertAlmostEqual(state.lambda_val, 0.25)
        state.set_lambda_cosine_zero(300_000, 300_000)
        self.assertEqual(state.lambda_val, 0.0)
        state.set_lambda_cosine_zero(400_000, 300_000)
        self.assertEqual(state.lambda_val, 0.0)

    def test_shaping_only_zero_lambda_skips_train_and_eval_guidance(self):
        class Spy:
            def __init__(self):
                self.calls = 0

            def __call__(self, *args):
                self.calls += 1
                return {"strategy": "unused", "subgoals": [], "action_rules": []}

        for runner_class, method_name in (
                (LehcaRunner, "_maybe_refresh_commander"),
                (RSVPRunner, "_maybe_refresh")):
            with self.subTest(runner=runner_class.__name__):
                runner = object.__new__(runner_class)
                runner.use_shaping = True
                runner.use_masking = False
                runner.mask_at_test = False
                runner.state = SimpleNamespace(lambda_val=0.0, guidance=None)
                runner.args = SimpleNamespace(lambda_zero_t=300_000)
                runner.t_env = 300_000
                runner.t = 0
                runner.commander = Spy()
                if runner_class is LehcaRunner:
                    runner.eval_commander = Spy()
                    runner.test_guidance_mode = "fresh"
                    runner.mask_anneal_t = 0
                    runner._maybe_refresh_commander({}, test_mode=False)
                    runner._maybe_refresh_commander({}, test_mode=True)
                    self.assertEqual(runner.eval_commander.calls, 0)
                else:
                    runner._maybe_refresh({}, None, test_mode=False)
                    runner._maybe_refresh({}, None, test_mode=True)
                self.assertEqual(runner.commander.calls, 0)

    def test_cutoff_boundary_skips_guidance_before_learner_updates_lambda(self):
        for runner_class in (LehcaRunner, RSVPRunner):
            with self.subTest(runner=runner_class.__name__):
                runner = object.__new__(runner_class)
                runner.use_shaping = True
                runner.use_masking = False
                runner.state = SimpleNamespace(lambda_val=0.01)
                runner.args = SimpleNamespace(lambda_zero_t=300_000)
                runner.t_env = 299_950
                runner.t = 49
                self.assertTrue(runner._guidance_needed(False))
                runner.t = 50
                self.assertFalse(runner._guidance_needed(False))


if __name__ == "__main__":
    unittest.main()
