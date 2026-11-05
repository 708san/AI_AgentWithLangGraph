import json
import tempfile
import unittest
from pathlib import Path

from project_ops.legacy_import import discover_legacy_runs, import_legacy_root


class LegacyImportTests(unittest.TestCase):
    def test_import_registers_existing_response_without_copying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "old_runs"
            source_dir = source_root / "patient_1"
            results = source_dir / "results"
            results.mkdir(parents=True)
            (source_dir / "run_config.json").write_text(
                json.dumps({"model": "fixture", "evidence_mode": "v2"}),
                encoding="utf-8",
            )
            response = results / "external_response.json"
            response.write_text(
                json.dumps({"schema_version": "legacy-v1", "finalDiagnosis": {"ans": [{"rank": 1}]}}),
                encoding="utf-8",
            )
            self.assertEqual(discover_legacy_runs(source_root), [source_dir.resolve()])
            imported = import_legacy_root(root=root, source_root=source_root)
            self.assertEqual(len(imported), 1)
            self.assertEqual(imported[0]["output_count"], 1)
            run_dir = root / "artifacts/runs" / imported[0]["run_id"]
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "imported")
            self.assertEqual(manifest["metadata"]["source_dir"], str(source_dir.resolve()))
            self.assertEqual(manifest["outputs"][0]["path"], str(response.resolve()))
            self.assertTrue((run_dir / "validation").is_dir())
            self.assertFalse((run_dir / "results").exists())


if __name__ == "__main__":
    unittest.main()

