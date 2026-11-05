import unittest
from pathlib import Path

from project_ops.profiles import get_profile, list_profiles


class ProfileTests(unittest.TestCase):
    def test_profiles_cover_existing_execution_paths(self):
        root = Path(__file__).resolve().parents[1]
        profiles = list_profiles(root)
        ids = {profile["profile_id"] for profile in profiles}
        self.assertIn("phenopacket.single_v2", ids)
        self.assertIn("image_case.batch", ids)
        self.assertIn("evaluation.tentative", ids)
        self.assertIn("analysis.saved_results", ids)

    def test_profile_lookup_is_explicit(self):
        root = Path(__file__).resolve().parents[1]
        profile = get_profile(root, "image_case.batch")
        self.assertEqual(profile["entrypoint"], "testCode/phenopacket_image_case/run_all_cases.py")
        with self.assertRaises(KeyError):
            get_profile(root, "does-not-exist")


if __name__ == "__main__":
    unittest.main()

