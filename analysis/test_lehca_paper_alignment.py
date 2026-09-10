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


if __name__ == "__main__":
    unittest.main()
