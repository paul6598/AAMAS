import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_refresh_replacement import summarize


class RefreshReplacementTests(unittest.TestCase):
    def test_old_drop_and_new_replacement_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.gz"
            records = []
            predicates = [[1, 0]] * 5 + [[0, 2]] * 5
            for t, value in enumerate(predicates):
                old = t < 5
                records.append({
                    "t": t, "predicates": value,
                    "weights": [1, 0] if old else [0, 1],
                    "decision": ({"early": True, "refresh_success": True,
                                  "S": 4., "h": 3.} if t == 5 else {}),
                })
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps({"kind": "metadata", "config": {
                    "seed": 0, "use_action_masking": False}}) + "\n")
                handle.write(json.dumps({"kind": "episode", "episode": 7,
                                         "records": records}) + "\n")
            result = summarize(path, horizon=5, gamma=1, bootstrap=0)
            self.assertEqual(result["events"], 1)
            self.assertAlmostEqual(result["staleness_drop"]["mean"], 1.)
            self.assertAlmostEqual(result["replacement_gain"]["mean"], 2.)
            self.assertEqual(result["dynamic_success_rate"], 1.)

    def test_action_guidance_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps({"kind": "metadata", "config": {
                    "seed": 0, "use_action_masking": True}}) + "\n")
            with self.assertRaises(ValueError):
                summarize(path, bootstrap=0)


if __name__ == "__main__":
    unittest.main()
