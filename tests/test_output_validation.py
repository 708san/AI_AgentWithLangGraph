import json
import tempfile
import unittest
from pathlib import Path

from project_ops.output_validation import validate_external_response
from project_ops.run_management import attach_output_validation, initialize_run


class OutputValidationTests(unittest.TestCase):
    def test_current_node_wrapped_shape_can_pass_with_declared_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external_response.json"
            path.write_text(
                json.dumps({
                    "external_response": {
                        "schema_version": "zebra-v2",
                        "finalDiagnosis": {"ans": [{"rank": 1}]},
                    },
                    "finalDiagnosis": {
                        "finalDiagnosis": {"ans": [{"rank": 1}]}
                    },
                }),
                encoding="utf-8",
            )
            report = validate_external_response(path, schema_version="zebra-v2")
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["shape_profile"], "node_wrapped")
            self.assertEqual(len(report["sha256"]), 64)

    def test_malformed_rank_is_failed_and_unversioned_output_is_partial_or_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external_response.json"
            path.write_text(
                json.dumps({"finalDiagnosis": {"ans": [{"rank": 0}, {"rank": 1}]}}),
                encoding="utf-8",
            )
            report = validate_external_response(path)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("positive integer" in error for error in report["errors"]))
            self.assertIn("No output schema version", " ".join(report["warnings"]))

    def test_validation_record_is_attached_to_manifest_and_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_run(root=root, run_id="output_run", kind="single", command=["runner"])
            manifest = attach_output_validation(
                root / "artifacts/runs/output_run",
                {
                    "kind": "external_response",
                    "path": str(root / "external_response.json"),
                    "schema_version": "zebra-v2",
                    "validation_status": "passed",
                },
            )
            self.assertEqual(manifest["outputs"][0]["schema_version"], "zebra-v2")
            registry = json.loads(
                (root / "local_artifacts/catalog/run_registry.json").read_text()
            )
            self.assertEqual(registry["runs"][0]["outputs"][0]["kind"], "external_response")


if __name__ == "__main__":
    unittest.main()

