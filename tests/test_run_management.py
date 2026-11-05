import json
import tempfile
import unittest
from pathlib import Path

from project_ops.run_management import (
    initialize_run,
    list_runs,
    update_run_status,
)


class RunManagementTests(unittest.TestCase):
    def test_initialize_run_records_command_conditions_and_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = initialize_run(
                root=root,
                run_id="smoke_001",
                kind="single_case",
                command=["python", "scripts/run_single_v2_case.py", "--patient-id", "1"],
                purpose="contract smoke",
                entrypoint="scripts/run_single_v2_case.py",
                dataset={"dataset_id": "fixture", "version": "1"},
                model={"name": "gpt-test"},
                pipeline={"evidence_mode": "v2"},
                metadata={"dry_run": True},
            )

            run_dir = root / "artifacts/runs/smoke_001"
            self.assertEqual(manifest["status"], "initialized")
            self.assertEqual(manifest["command_text"], "python scripts/run_single_v2_case.py --patient-id 1")
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertEqual((run_dir / "command.txt").read_text(), manifest["command_text"] + "\n")
            registry = json.loads(
                (root / "local_artifacts/catalog/run_registry.json").read_text()
            )
            self.assertEqual(registry["runs"][0]["run_id"], "smoke_001")
            self.assertEqual(registry["runs"][0]["dataset"]["dataset_id"], "fixture")

    def test_status_update_is_reflected_in_manifest_and_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_run(root=root, run_id="run_a", kind="batch", command=["runner"])
            updated = update_run_status(root / "artifacts/runs/run_a", "completed")

            self.assertEqual(updated["status"], "completed")
            self.assertIn("finished_at", updated)
            completed = list_runs(root / "local_artifacts/catalog/run_registry.json", "completed")
            self.assertEqual([item["run_id"] for item in completed], ["run_a"])

    def test_run_id_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                initialize_run(root=Path(tmp), run_id="../unsafe", kind="test", command=[])


if __name__ == "__main__":
    unittest.main()

