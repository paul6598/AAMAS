"""Offline prompt routing/contract checks; these do not prove LLM compliance."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from algorithm.lehca.commander.llm_commander import (
    LLMCommander, PAPER_SYSTEM_PROMPT,
)
from algorithm.lehca.commander.paper_prompt_v2 import PAPER_V2_SYSTEM_PROMPT
from algorithm.lehca.shaping.predicates import ALL_PREDICATES


class PaperPromptV2Tests(unittest.TestCase):
    def commander(self, style):
        args = SimpleNamespace(
            llm_api_base="http://unused/v1", llm_model="test",
            llm_temperature=0.2, llm_max_tokens=3072, llm_timeout=1,
            llm_cache=False, prompt_style=style, f_update=200,
        )
        iface = SimpleNamespace(prompt_context=lambda: "Observable interface.")
        commander = LLMCommander(args, iface)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content":
            '{"strategy":"One ally remains; preserve combat ability.",'
            '"subgoals":[{"predicate":"ally_survive","weight":0.5}],'
            '"action_rules":[]}'}}]}
        commander._session = Mock()
        commander._session.post.return_value = response
        return commander, iface

    def test_new_style_routes_formats_and_parses_without_schema_change(self):
        commander, iface = self.commander("paper_v2")
        result = commander("One living ally.", "key", iface)
        self.assertEqual(result["subgoals"][0]["predicate"], "ally_survive")
        payload = commander._session.post.call_args.kwargs["json"]
        self.assertIn("200 environment steps", payload["messages"][0]["content"])
        self.assertIn("One living ally.", payload["messages"][1]["content"])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(commander.n_calls, 1)

    def test_legacy_paper_is_unchanged(self):
        commander, iface = self.commander("paper")
        commander("summary", "key", iface)
        self.assertEqual(commander.system_prompt,
                         PAPER_SYSTEM_PROMPT.format(env_context=iface.prompt_context()))

    def test_contract_covers_predicates_and_known_grounding_limits(self):
        for predicate in ALL_PREDICATES:
            self.assertIn("- " + predicate + ":", PAPER_V2_SYSTEM_PROMPT)
        for text in ("true reward function", "fewer than two combat allies",
                     "below 30% HP", "NO health", "not a reasoning transcript",
                     "maximum weight, not addition", "not necessarily dead"):
            self.assertIn(text, PAPER_V2_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
