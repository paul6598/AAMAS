"""RSVP naming/compatibility regression tests (no environment or LLM calls).

Optional RSVP_PRE_RENAME_ARCHIVE points to the migration's before.tar to
check that the executable runner/critic/predicate AST is unchanged by naming.
"""
import ast
import importlib
import os
from pathlib import Path
import sys
import tarfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MigrationTests(unittest.TestCase):
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
        from analysis.audit_lehca_grounding import synthetic_tests
        result = synthetic_tests()
        self.assertTrue(result['strategy_only_change_same_shaping'])
        self.assertEqual(result['focus_fire_without_damage'], .3)

    def test_pre_rename_executable_ast(self):
        archive = os.environ.get('RSVP_PRE_RENAME_ARCHIVE')
        if not archive:
            self.skipTest('Set RSVP_PRE_RENAME_ARCHIVE for pre/post implementation equivalence')

        class Normalize(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value,str):
                    return None  # documentation, not executable logic
                return self.generic_visit(node)

        def normalized(source):
            source = source.replace('algorithm.rsvp','algorithm.vigil').replace('RSVPRunner','SchedRunner')
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if not (
                isinstance(n,ast.Assign) and len(n.targets)==1 and
                isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SchedRunner' and
                isinstance(n.value,ast.Name) and n.value.id=='SchedRunner')]
            return ast.dump(Normalize().visit(tree),include_attributes=False)

        with tarfile.open(archive) as tf:
            for rel in ['runner.py','critic.py','predlib.py','shaping/grf.py','shaping/pursuit.py',
                        'commander/grf.py','commander/pursuit.py']:
                with self.subTest(file=rel):
                    before = tf.extractfile('algorithm/vigil/'+rel).read().decode()
                    after = (ROOT/'algorithm/rsvp'/rel).read_text()
                    self.assertEqual(normalized(before),normalized(after))


if __name__ == '__main__':
    unittest.main(verbosity=2)
