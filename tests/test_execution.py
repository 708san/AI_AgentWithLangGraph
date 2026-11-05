import json
import sys
import tempfile
import unittest
from pathlib import Path

from project_ops.execution import parse_output_spec, run_managed_command


class ManagedExecutionTests(unittest.TestCase):
    def test_managed_command_records_process_and_validates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "external_response.json"
            output.write_text(
                json.dumps({"schema_version": "fixture-v1", "finalDiagnosis": {"ans": [{"rank": 1}]}}),
                encoding="utf-8",
            )
            result = run_managed_command(
                root=root,
                run_id="managed_001",
                kind="smoke",
                command=[sys.executable, "-c", "print('managed')"],
                output_specs=[("external_response", str(output))],
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["return_code"], 0)
            self.assertEqual(result["validation"][0]["status"], "passed")
            run_dir = root / "artifacts/runs/managed_001"
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["execution"]["return_code"], 0)
            self.assertTrue((run_dir / "stdout.log").is_file())
            self.assertTrue((run_dir / "validation/external_response.json").is_file())

    def test_failed_command_is_recorded_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_managed_command(
                root=Path(tmp),
                run_id="managed_failed",
                kind="smoke",
                command=[sys.executable, "-c", "raise SystemExit(3)"],
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["return_code"], 3)
            manifest = json.loads(
                (Path(tmp) / "artifacts/runs/managed_failed/run_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "failed")

    def test_output_spec_requires_kind_and_path(self):
        self.assertEqual(
            parse_output_spec("external_response=file.json"),
            ("external_response", "file.json", None),
        )
        self.assertEqual(
            parse_output_spec("external_response[276]=file.json"),
            ("external_response", "file.json", "276"),
        )
        with self.assertRaises(ValueError):
            parse_output_spec("file.json")


if __name__ == "__main__":
    unittest.main()

