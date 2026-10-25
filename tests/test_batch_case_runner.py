import json
import tempfile
import unittest
from pathlib import Path

from testCode.phenopacket_image_case.run_all_cases import (
    _case_is_complete,
    _load_rows,
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
