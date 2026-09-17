"""Check metric boundaries and explicit completion evidence using small Sacred fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.summarize_experiments import curve_metrics, summarize_run


class ExperimentSummaryTests(unittest.TestCase):
    def test_incomplete_curve_has_no_final_or_extrapolated_auc(self):
        result = curve_metrics([(10, 0.0), (50, 1.0)], 100, False)
        self.assertAlmostEqual(result["auc_observed"], .4)
        self.assertIsNone(result["auc_nominal_hold_last"])
        self.assertIsNone(result["final100k"])

    def test_endpoint_interpolation_and_completed_tail_are_explicit(self):
        result = curve_metrics([(10, 0.0), (60, 1.0), (110, 0.0)], 100, True)
        self.assertAlmostEqual(result["auc_observed"], .49)
        self.assertAlmostEqual(result["auc_nominal_hold_last"], .49)
        self.assertAlmostEqual(result["evaluation_mean"], .5)
        self.assertAlmostEqual(result["final100k"], .5)

    def test_cancelled_job_is_not_completed_despite_full_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "1"
            directory.mkdir()
            for name, data in {
                "config": {"seed": 0, "t_max": 100},
                "info": {"test_battle_won_mean_T": [100], "test_battle_won_mean": [1.]},
                "run": {"status": "RUNNING"},
            }.items():
                (directory / (name + ".json")).write_text(json.dumps(data))
            cancelled = summarize_run(directory, 42, {"state": "CANCELLED", "exit_code": "0:15"})
            self.assertFalse(cancelled["complete"])
            completed = summarize_run(directory, 42, {"state": "COMPLETED", "exit_code": "0:0"})
            self.assertTrue(completed["complete"])
            self.assertEqual(completed["sacred_status"], "RUNNING")


if __name__ == "__main__":
    unittest.main()
