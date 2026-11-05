import json
import tempfile
import unittest
from pathlib import Path

from project_ops.compatibility import compare_artifact_directories


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CompatibilityTests(unittest.TestCase):
    def _make_pair(self) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        baseline = root / "baseline"
        candidate = root / "candidate"
        external = {
            "finalDiagnosis": {
                "ans": [{"rank": 1, "disease_name": "Example disease"}]
            }
        }
        _write(baseline / "external_response.json", external)
        _write(candidate / "external_response.json", external)
        _write(
            baseline / "public_response.json",
            {"imagePath": "/old/location/face.png", "finalDiagnosis": external["finalDiagnosis"]},
        )
        _write(
            candidate / "public_response.json",
            {"imagePath": "/new/location/face.png", "finalDiagnosis": external["finalDiagnosis"]},
        )
        return baseline, candidate, temporary

    def test_same_fixed_fixture_ignores_run_location_metadata(self):
        baseline, candidate, temporary = self._make_pair()
        try:
            result = compare_artifact_directories(
                baseline,
                candidate,
                profile={
                    "profile_id": "phenopacket.single_v2",
                    "compatibility": {
                        "required_artifacts": ["external_response"],
                        "optional_artifacts": ["public_response"],
                    },
                },
            )
        finally:
            temporary.cleanup()
        self.assertEqual(result["status"], "equivalent")
        self.assertEqual(result["artifacts"]["external_response"]["status"], "equivalent")

    def test_external_response_change_is_reported_without_freezing_schema(self):
        baseline, candidate, temporary = self._make_pair()
        try:
            payload = json.loads((candidate / "external_response.json").read_text(encoding="utf-8"))
            payload["finalDiagnosis"]["ans"][0]["rank"] = 2
            _write(candidate / "external_response.json", payload)
            result = compare_artifact_directories(
                baseline,
                candidate,
                profile={
                    "profile_id": "phenopacket.single_v2",
                    "compatibility": {"required_artifacts": ["external_response"]},
                },
            )
        finally:
            temporary.cleanup()
        self.assertEqual(result["status"], "changed")
        self.assertIn("external_response", result["summary"]["changed"])
        self.assertTrue(result["artifacts"]["external_response"]["differences"])

    def test_missing_required_artifact_is_incomplete(self):
        baseline, candidate, temporary = self._make_pair()
        try:
            (candidate / "external_response.json").unlink()
            result = compare_artifact_directories(
                baseline,
                candidate,
                profile={
                    "profile_id": "phenopacket.single_v2",
                    "compatibility": {"required_artifacts": ["external_response"]},
                },
            )
        finally:
            temporary.cleanup()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["summary"]["required_missing"], ["external_response"])


if __name__ == "__main__":
    unittest.main()
