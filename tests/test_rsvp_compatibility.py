"""RSVP의 옛 설정·import 호환과 grounding 회귀를 검사한다. 환경·LLM 호출은 없다."""
import importlib
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CompatibilityTests(unittest.TestCase):
    def test_registry_and_legacy_identity(self):
        import algorithm.rsvp as new
        import algorithm.vigil as old
        from algorithm.src.runners import REGISTRY
        self.assertIs(new.RSVPRunner, old.SchedRunner)
        self.assertIs(REGISTRY['rsvp'], REGISTRY['vigil'])
        self.assertEqual(new.RSVPRunner.__module__, 'algorithm.rsvp.runner')
        self.assertEqual(new.RSVPRunner.__name__, 'RSVPRunner')
        for name in ['runner','critic','predlib','commander.pursuit','shaping.pursuit']:
            self.assertIs(importlib.import_module('algorithm.rsvp.'+name),
                          importlib.import_module('algorithm.vigil.'+name))
        self.assertIs(importlib.import_module('algorithm.vigil.runner').SchedRunner,
                      new.RSVPRunner)

    def test_config_alias(self):
        new = yaml.safe_load((ROOT/'config/algs/rsvp.yaml').read_text())
        old = yaml.safe_load((ROOT/'config/algs/vigil.yaml').read_text())
        self.assertEqual(new, old)
        self.assertEqual(new['name'], 'rsvp')
        self.assertEqual(new['runner'], 'rsvp')
        self.assertEqual(new['llm_model'], 'openai/gpt-oss-20b')

    def test_grounding_regressions(self):
        from analysis.audits.audit_lehca_grounding import synthetic_tests
        result = synthetic_tests()
        self.assertTrue(result['strategy_only_change_same_shaping'])
        self.assertEqual(result['focus_fire_without_damage'], .3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
