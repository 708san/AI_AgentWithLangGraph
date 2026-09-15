import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation import runner


class RunnerTests(unittest.TestCase):
    def fixture(self, root):
        benchmark = root / "cases.json"
        benchmark.write_text(json.dumps({"cases": [{"case_id": "a", "patient_id": "1",
            "input": {"present_hpo_list": ["HP:0001250"]},
            "expected_output": {"omim_id": "123456"}}]}))
        return benchmark

    def test_dry_run_never_initializes_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            for mode, factory in [("zero-shot", "zero_shot_executor"), ("tentative", "tentative_executor")]:
                with patch.object(runner, factory) as mocked:
                    self.assertEqual(runner.main(mode, ["--benchmark", str(path), "--dry-run"]), 0)
                    mocked.assert_not_called()

    def test_repeats_save_and_score_without_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            output = root / "run"
            calls = []
            def execute(inputs, record, checkpoint):
                self.assertNotIn("expected_output", inputs)
                self.assertNotIn("expected_output", record)
                calls.append(inputs)
                record["zeroShotRaw"] = {"ans": [{"rank": len(calls), "OMIM_id": "123456"}]}
                record["prompts"] = {"zeroShot": "test prompt"}
                checkpoint()
            with patch.object(runner, "zero_shot_executor", return_value=execute):
                self.assertEqual(runner.main("zero-shot", ["--benchmark", str(path), "--output-dir", str(output), "--repeats", "2"]), 0)
            self.assertEqual(len(calls), 2)
            reports = [json.loads((output / f"repeat_{n:03d}/evaluation.json").read_text()) for n in (1, 2)]
            self.assertEqual(reports[0]["summary"]["zeroShotRaw"]["top_k"]["1"]["hits"], 1)
            self.assertEqual(reports[1]["summary"]["zeroShotRaw"]["top_k"]["1"]["hits"], 0)

    def test_partial_output_survives_case_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            def execute(inputs, record, checkpoint):
                record["zeroShotRaw"] = {"ans": []}
                checkpoint()
                raise RuntimeError("mock failure")
            with patch.object(runner, "tentative_executor", return_value=execute):
                code = runner.main("tentative", ["--benchmark", str(path), "--output-dir", str(root / "out")])
            self.assertEqual(code, 1)
            record = json.loads((root / "out/repeat_001/predictions/a.json").read_text())
            self.assertEqual(record["zeroShotRaw"], {"ans": []})
            self.assertEqual(record["status"], "error")

    def test_packaged_image_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "images").mkdir()
            image = root / "images/a.jpg"
            image.write_bytes(b"test")
            self.assertEqual(runner.resolve_image("images/a.jpg", root / "data/cases.json"), str(image.resolve()))


if __name__ == "__main__":
    unittest.main()
