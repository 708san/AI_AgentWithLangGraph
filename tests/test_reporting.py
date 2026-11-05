import json
import tempfile
import unittest
from pathlib import Path

from project_ops.reporting import generate_project_report


class ReportingTests(unittest.TestCase):
    def test_project_report_generates_json_html_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.json"
            registry = root / "registry.json"
            output_dir = root / "report"
            catalog.write_text(
                json.dumps({"repository_root": str(root), "summary": {"file_count": 3, "script_count": 1}}),
                encoding="utf-8",
            )
            (root / "configs").mkdir()
            (root / "configs/execution_profiles.json").write_text(
                json.dumps({"profiles": [{"profile_id": "fixture", "purpose": "Fixture profile"}]}),
                encoding="utf-8",
            )
            metrics_path = root / "metrics.json"
            metrics_path.write_text(
                json.dumps({
                    "summary": {
                        "finalDiagnosis": {
                            "mrr": 0.75,
                            "top_k": {"1": {"rate": 0.5}},
                        }
                    }
                }),
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({
                    "runs": [{
                        "run_id": "run-1",
                        "status": "completed",
                        "kind": "evaluation",
                        "model": {"name": "fixture"},
                        "dataset": {"dataset_id": "bench"},
                        "analysis": [{"kind": "evaluation_metrics", "path": str(metrics_path)}],
                    }]
                }),
                encoding="utf-8",
            )
            result = generate_project_report(
                catalog_path=catalog,
                registry_path=registry,
                output_dir=output_dir,
            )
            self.assertEqual(result["run_count"], 1)
            self.assertEqual(result["stage_metric_count"], 1)
            self.assertTrue((output_dir / "project_overview.html").is_file())
            self.assertTrue((output_dir / "runs.csv").is_file())
            overview = json.loads((output_dir / "project_overview.json").read_text())
            self.assertEqual(overview["run_summary"]["status_counts"]["completed"], 1)
            self.assertEqual(overview["profiles"][0]["profile_id"], "fixture")
            self.assertIn("run-1", (output_dir / "project_overview.html").read_text())


if __name__ == "__main__":
    unittest.main()

