import json
import tempfile
import unittest
from pathlib import Path

from project_ops.evaluation import analyze_run, compare_reports
from project_ops.run_management import initialize_run


def report(mrr: float, top1: float, hits: bool) -> dict:
    return {
        "metadata": {"model": "fixture"},
        "summary": {
            "finalDiagnosis": {
                "mrr": mrr,
                "top_k": {"1": {"rate": top1}},
            }
        },
        "cases": [{
            "case_id": "case-1",
            "stages": {
                "finalDiagnosis": {"hits": {"1": hits}}
            },
        }],
    }


class EvaluationManagementTests(unittest.TestCase):
    def test_compare_reports_records_metric_delta_and_case_changes(self):
        comparison = compare_reports(report(0.5, 0.5, True), report(0.0, 0.0, False))
        stage = comparison["stages"]["finalDiagnosis"]
        self.assertEqual(stage["mrr_delta"], 0.5)
        self.assertEqual(stage["top_k"]["1"]["rate_delta"], 0.5)
        self.assertEqual(comparison["case_changes"]["finalDiagnosis"]["gained"], ["case-1@1"])

    def test_analyze_run_writes_metrics_comparison_and_manifest_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_run(root=root, run_id="analysis_001", kind="evaluation", command=["runner"])
            candidate_path = root / "candidate.json"
            baseline_path = root / "baseline.json"
            candidate_path.write_text(json.dumps(report(0.5, 0.5, True)), encoding="utf-8")
            baseline_path.write_text(json.dumps(report(0.0, 0.0, False)), encoding="utf-8")
            result = analyze_run(
                root / "artifacts/runs/analysis_001",
                candidate_path,
                baseline_report_path=baseline_path,
            )
            run_dir = root / "artifacts/runs/analysis_001"
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertTrue((run_dir / "analysis/metrics.json").is_file())
            self.assertTrue((run_dir / "analysis/comparison.json").is_file())
            self.assertEqual(result["comparison"]["stages"]["finalDiagnosis"]["mrr_delta"], 0.5)
            self.assertEqual(manifest["analysis"][0]["kind"], "evaluation_metrics")


if __name__ == "__main__":
    unittest.main()

