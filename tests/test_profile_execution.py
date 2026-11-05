import sys
import unittest
from pathlib import Path

from project_ops.profile_execution import build_profile_command


class ProfileExecutionTests(unittest.TestCase):
    def test_profile_command_uses_current_python_and_existing_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        command = build_profile_command(
            root,
            "phenopacket.single_v2",
            ["--help"],
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[-1], "--help")
        self.assertTrue(command[1].endswith("scripts/run_single_v2_case.py"))

    def test_missing_profile_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(KeyError):
            build_profile_command(root, "missing.profile", [])


if __name__ == "__main__":
    unittest.main()

