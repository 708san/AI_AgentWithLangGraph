import json
import tempfile
import unittest
from pathlib import Path

from project_ops.catalog import build_catalog, write_catalog


class ProjectCatalogTests(unittest.TestCase):
    def test_catalog_classifies_scripts_data_and_excludes_credentials_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts" / "run_case.py").write_text(
                '"""Run one case."""\nimport argparse\n\n'
                'if __name__ == "__main__":\n    pass\n',
                encoding="utf-8",
            )
            (root / "Data").mkdir()
            (root / "Data" / "cases.tsv").write_text("case_id\n1\n", encoding="utf-8")
            (root / "scripts" / "data").mkdir()
            (root / "scripts" / "data" / "benchmark.json").write_text("{}", encoding="utf-8")
            (root / "scripts" / "README.md").write_text("Scripts", encoding="utf-8")
            (root / "tryToolsForNewData").mkdir()
            (root / "tryToolsForNewData" / "experiment.json").write_text("{}", encoding="utf-8")
            (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
            (root / "local_artifacts").mkdir()
            (root / "local_artifacts" / "generated.json").write_text("{}", encoding="utf-8")
            (root / "testCode/phenopacket_image_case/runs/case_1").mkdir(parents=True)
            (root / "testCode/phenopacket_image_case/runs/case_1/result.json").write_text(
                "{}", encoding="utf-8"
            )

            catalog = build_catalog(root, include_hash=True)

            paths = {item["path"] for item in catalog["files"]}
            self.assertIn("scripts/run_case.py", paths)
            self.assertIn("Data/cases.tsv", paths)
            self.assertNotIn(".env", paths)
            self.assertNotIn("local_artifacts/generated.json", paths)
            self.assertNotIn(
                "testCode/phenopacket_image_case/runs/case_1/result.json",
                paths,
            )
            self.assertEqual(catalog["summary"]["script_count"], 1)
            self.assertEqual(catalog["summary"]["data_file_count"], 2)
            categories = {item["path"]: item["category"] for item in catalog["files"]}
            self.assertEqual(categories["scripts/data/benchmark.json"], "data")
            self.assertEqual(categories["scripts/README.md"], "documentation")
            experiment = next(
                item for item in catalog["files"]
                if item["path"] == "tryToolsForNewData/experiment.json"
            )
            self.assertNotIn("sha256", experiment)
            script = catalog["scripts"][0]
            self.assertEqual(script["module_doc"], "Run one case.")
            self.assertTrue(script["has_main_guard"])
            self.assertTrue(script["uses_argparse"])
            self.assertIn("sha256", script)

    def test_catalog_writes_json_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "catalog" / "project_catalog.json"
            write_catalog(output, {"schema_version": 1, "files": []})

            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], 1)
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()

