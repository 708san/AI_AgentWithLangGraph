import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.evaluate import compare, evaluate, load_cases, main, omim_id, score_stage


class EvaluationTests(unittest.TestCase):
    def test_id_matching_is_namespace_safe(self):
        self.assertEqual(omim_id("omim: 123456"), "OMIM:123456")
        self.assertIsNone(omim_id("MONDO:123456"))
        self.assertIsNone(omim_id("disease 123456"))

    def test_declared_ranks_and_missing_ids(self):
        output = {"ans": [{"rank": 7, "OMIM_id": "123456"},
                          {"rank": 2, "OMIM_id": None}]}
        row = score_stage(output, "OMIM:123456", [1, 5, 10])
        self.assertEqual(row["hits"], {"1": False, "5": False, "10": True})
        self.assertEqual(row["reciprocal_rank"], 1 / 7)
        self.assertEqual(row["invalid_omim_count"], 1)

    def test_invalid_and_duplicate_rank(self):
        for rank in (None, 0, -1, "1", True):
            self.assertEqual(score_stage({"ans": [{"rank": rank}]}, "OMIM:123456", [1])["status"], "invalid_rank")
        self.assertEqual(score_stage({"ans": [{"rank": 1}, {"rank": 1}]}, "OMIM:123456", [1])["status"], "duplicate_rank")

    def test_duplicate_disease_uses_best_rank(self):
        row = score_stage({"ans": [{"rank": 5, "OMIM_id": "123456"}, {"rank": 2, "omim_id": "123456"}]}, "OMIM:123456", [3])
        self.assertEqual(row["correct_rank"], 2)
        self.assertEqual(row["duplicate_omim_count"], 1)

    def test_missing_cases_stay_in_denominator_and_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [{"case_id": c, "expected_output": {"omim_id": "123456"}} for c in ("a", "b")]
            baseline = evaluate(cases, root, ["zeroShotResult"], [1])
            (root / "a.json").write_text(json.dumps({"zeroShotResult": {"ans": [{"rank": 1, "OMIM_id": "123456"}]}}))
            current = evaluate(cases, root, ["zeroShotResult"], [1])
            self.assertEqual(current["summary"]["zeroShotResult"]["top_k"]["1"]["rate"], 0.5)
            self.assertEqual(compare(current, baseline, ["zeroShotResult"], [1])["zeroShotResult"]["top_k"]["1"]["gained"], ["a"])

    def test_cli_bare_output_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / "cases.json"
            cases.write_text(json.dumps({"cases": [{"case_id": "a", "expected_output": {"omim_id": "123456"}}]}))
            predictions = root / "predictions"
            predictions.mkdir()
            (predictions / "a.json").write_text('{"ans": [{"rank": 1, "omim_id": "123456"}]}')
            output = root / "report.json"
            args = ["--cases", str(cases), "--predictions-dir", str(predictions), "--bare-stage", "zeroShotRaw", "--output", str(output)]
            self.assertEqual(main(args), 0)
            original = output.read_bytes()
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(original, output.read_bytes())

    def test_invalid_files_and_ambiguous_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = {"case_id": "a", "patient_id": "1", "expected_output": {"omim_id": "123456"}}
            (root / "a.json").write_text('{')
            result = evaluate([case], root, ["tentativeDiagnosis"], [1])
            self.assertTrue(result["cases"][0]["stages"]["tentativeDiagnosis"]["status"].startswith("invalid_file"))
            (root / "1.json").write_text('{}')
            result = evaluate([case], root, ["tentativeDiagnosis"], [1])
            self.assertEqual(result["cases"][0]["stages"]["tentativeDiagnosis"]["status"], "ambiguous_files")

    def test_labels_cannot_be_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.json"
            path.write_text('{"cases": [{"case_id": "a", "expected_output": {}}]}')
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_report_preserves_outputs_after_source_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [{"case_id": "a", "expected_output": {
                "omim_id": "123456", "disease_name": "Expected disease"}}]
            payload = {"prompt": "Original prompt", "model": "example-model",
                       "zeroShotResult": {"ans": [{"rank": 3, "OMIM_id": "123456", "disease_name": "Original name"}]},
                       "tentativeDiagnosis": {"ans": [{"rank": "invalid", "description": "元の支持理由", "custom": [1, 2]}], "reference": "Original source"}}
            path = root / "a.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            baseline = evaluate(cases, root, ["zeroShotResult", "tentativeDiagnosis"], [1, 5])
            report = evaluate(cases, root, ["zeroShotResult", "tentativeDiagnosis"], [1, 5])
            report["baseline"] = baseline
            saved = root / "report.json"
            saved.write_text(json.dumps(report), encoding="utf-8")
            path.unlink()
            restored = json.loads(saved.read_text())
            for section in (restored, restored["baseline"]):
                row = section["cases"][0]
                self.assertEqual(row["source_payload"], payload)
                self.assertEqual(row["expected_output"], cases[0]["expected_output"])
                self.assertEqual(row["stages"]["tentativeDiagnosis"]["status"], "invalid_rank")
                for stage in ("zeroShotResult", "tentativeDiagnosis"):
                    self.assertEqual(row["stages"][stage]["output"], payload[stage])

    def test_bare_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {"ans": [{"rank": 2, "OMIM_id": None, "disease_name": "Unmapped"}]}
            (root / "a.json").write_text(json.dumps(payload))
            report = evaluate([{"case_id": "a", "expected_output": {"omim_id": "123456"}}], root, ["zeroShotRaw"], [1], "zeroShotRaw")
            self.assertEqual(report["cases"][0]["stages"]["zeroShotRaw"]["output"], payload)


if __name__ == "__main__":
    unittest.main()
