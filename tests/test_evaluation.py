import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.evaluate import evaluate, omim_id, score_stage


class EvaluationTests(unittest.TestCase):
    def test_omim_matching_is_namespace_safe(self):
        self.assertEqual(omim_id("omim: 123456"), "OMIM:123456")
        self.assertIsNone(omim_id("MONDO:123456"))
        self.assertIsNone(omim_id("disease 123456"))

    def test_preserves_declared_rank_and_scores_top_k(self):
        row = score_stage(
            {"ans": [{"rank": 7, "OMIM_id": "123456"}, {"rank": 2, "OMIM_id": None}]},
            "OMIM:123456",
            [1, 5, 10],
            "zeroShotResult",
        )
        self.assertEqual(row["correct_rank"], 7)
        self.assertEqual(row["hits"], {"1": False, "5": False, "10": True})

    def test_merged_candidate_stage_uses_best_rank(self):
        row = score_stage(
            {"mergedDiseaseCandidates": [
                {"OMIM_id": "123456", "best_rank": 2},
                {"OMIM_id": "654321", "best_rank": 1},
            ]},
            "OMIM:123456",
            [1, 3],
            "mergedDiseaseCandidates",
        )
        self.assertEqual(row["correct_rank"], 2)
        self.assertTrue(row["hits"]["3"])

    def test_report_keeps_missing_cases_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                {"case_id": "a", "expected_output": {"omim_id": "123456"}},
                {"case_id": "b", "expected_output": {"omim_id": "123456"}},
            ]
            (root / "a.json").write_text(json.dumps({
                "zeroShotResult": {"ans": [{"rank": 1, "OMIM_id": "123456"}]}
            }))
            report = evaluate(cases, root, ["zeroShotResult"], [1])
            self.assertEqual(report["summary"]["zeroShotResult"]["top_k"]["1"]["rate"], 0.5)
            self.assertEqual(report["summary"]["zeroShotResult"]["status_counts"]["missing_file"], 1)


if __name__ == "__main__":
    unittest.main()
