import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from testCode.phenopacket_image_case.run_all_cases import (
    _case_is_complete,
    _load_rows,
    _prepare_case_for_run,
    _run_one_case,
)
from testCode.phenopacket_image_case.run_one_case import _validate_v2_public_response


class BatchCaseRunnerTests(unittest.TestCase):
    def test_load_rows_keeps_unique_patient_order_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata.tsv"
            metadata.write_text(
                "patient_id\timage_id\n"
                "2\timg-2a\n"
                "1\timg-1\n"
                "2\timg-2b\n",
                encoding="utf-8",
            )
            order, rows = _load_rows(metadata)
            self.assertEqual(order, ["2", "1"])
            self.assertEqual([row["image_id"] for row in rows["2"]], ["img-2a", "img-2b"])

    def test_case_is_complete_requires_matching_config_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "results").mkdir()
            (case_dir / "results" / "public_response.json").write_text("{}", encoding="utf-8")
            (case_dir / "results" / "finalDiagnosis.json").write_text("{}", encoding="utf-8")
            expected = {"patient_id": "1", "model": "gpt-4o"}
            (case_dir / "run_config.json").write_text(json.dumps(expected), encoding="utf-8")
            (case_dir / "execution_summary.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (case_dir / "results" / "output_contract.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            self.assertTrue(_case_is_complete(case_dir, expected))
            self.assertFalse(_case_is_complete(case_dir, {**expected, "model": "other"}))

            (case_dir / "execution_summary.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            self.assertFalse(_case_is_complete(case_dir, expected))

    def test_prepare_case_for_run_clears_stale_outputs_and_marks_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            case_dir = output_root / "patient_42"
            case_dir.mkdir()
            expected = {
                "runner": "phenopacket_image_case_v2",
                "patient_id": "42",
                "model": "gpt-4o",
            }
            (case_dir / "run_config.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            (case_dir / "execution_summary.json").write_text(
                json.dumps({"status": "failed", "patient_id": "42"}), encoding="utf-8"
            )
            for folder in ("input", "logs", "results", "node_results"):
                artifact_dir = case_dir / folder
                artifact_dir.mkdir()
                (artifact_dir / "stale.json").write_text("{}", encoding="utf-8")
            (case_dir / "launcher.log").write_text("old attempt", encoding="utf-8")
            (case_dir / "notes.txt").write_text("keep unrelated file", encoding="utf-8")

            _prepare_case_for_run(
                case_dir,
                output_root=output_root,
                patient_id="42",
                expected_config=expected,
            )

            for folder in ("input", "logs", "results", "node_results"):
                self.assertFalse((case_dir / folder).exists())
            self.assertFalse((case_dir / "launcher.log").exists())
            self.assertEqual((case_dir / "notes.txt").read_text(encoding="utf-8"), "keep unrelated file")
            self.assertEqual(json.loads((case_dir / "run_config.json").read_text()), expected)
            self.assertEqual(
                json.loads((case_dir / "execution_summary.json").read_text())["status"],
                "running",
            )

    def test_prepare_case_for_run_refuses_unrecognized_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            case_dir = output_root / "patient_42"
            case_dir.mkdir()
            marker = case_dir / "user_data.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                _prepare_case_for_run(
                    case_dir,
                    output_root=output_root,
                    patient_id="42",
                    expected_config={"patient_id": "42"},
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_run_one_case_terminates_child_when_batch_is_interrupted(self):
        class InterruptedProcess:
            terminated = False

            def wait(self, timeout=None):
                if timeout is None:
                    raise KeyboardInterrupt
                return -15

            def terminate(self):
                self.terminated = True

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "patient_42"
            process = InterruptedProcess()
            args = SimpleNamespace(
                metadata="metadata.tsv",
                image_dir="images",
                output_root=tmp,
                model="gpt-4o",
                max_flow_iterations=1,
                pubmed_max_workers=2,
                evidence_max_workers=2,
                use_phenobrain=False,
                use_absent_hpo=False,
            )
            with patch(
                "testCode.phenopacket_image_case.run_all_cases.subprocess.Popen",
                return_value=process,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    _run_one_case(patient_id="42", case_dir=case_dir, args=args)
            self.assertTrue(process.terminated)

    def test_short_description_is_a_warning_not_a_pipeline_failure(self):
        contract = _validate_v2_public_response({
            "evidence_mode": "v2",
            "validation_status": "partial",
            "references": [],
            "tentativeDiagnosis": {"ans": [], "references": []},
            "finalDiagnosis": {
                "ans": [{
                    "candidate_id": "OMIM:123456",
                    "description": "Short but valid provisional text.",
                    "description_en": "Short but valid provisional text.",
                    "description_ja": "短い暫定説明。",
                    "literature_status": "not_retrieved",
                    "citation_numbers": [],
                }],
                "references": [],
            },
        })
        self.assertTrue(contract["passed"])
        self.assertEqual(contract["errors"], [])
        self.assertTrue(contract["warnings"])


if __name__ == "__main__":
    unittest.main()
