import unittest
from unittest.mock import patch

from agent.state.state_types import ZeroShotFormat, ZeroShotOutput
from agent.tools import diseaseNormalize


class ZeroShotNormalizationTests(unittest.TestCase):
    def _state(self, *items):
        return {"zeroShotResult": ZeroShotOutput(ans=list(items))}

    def test_unconfirmed_candidates_are_excluded_and_remaining_rows_are_renumbered(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Known syndrome", rank=1, OMIM_id="123456"),
            ZeroShotFormat(disease_name="Unknown syndrome", rank=2, OMIM_id=None),
            ZeroShotFormat(disease_name="Known syndrome two", rank=5, OMIM_id="654321"),
        ])
        state = {"zeroShotResult": output}
        with patch.object(diseaseNormalize, "known_disease_name", side_effect=lambda value: {
            "OMIM:123456": "KNOWN SYNDROME",
            "OMIM:654321": "KNOWN SYNDROME TWO",
        }.get(value)), patch.object(
            diseaseNormalize, "disease_normalize", return_value=(None, None, 0.0)
        ), patch.object(
            diseaseNormalize, "names_compatible", return_value=True
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual([item.rank for item in result.ans], [1, 2])
        self.assertEqual(
            [item.disease_name for item in result.ans],
            ["KNOWN SYNDROME", "KNOWN SYNDROME TWO"],
        )
        validation = {item["disease_name"]: item for item in state["zeroShotValidation"]}
        self.assertTrue(validation["Unknown syndrome"]["excluded"])
        self.assertEqual(validation["Unknown syndrome"]["exclusion_status"], "unconfirmed_identity")
        self.assertIsNone(validation["Unknown syndrome"]["display_rank"])
        self.assertEqual(validation["Known syndrome two"]["display_rank"], 2)

    def test_known_omim_id_wins_over_mismatched_generated_name(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Wrong generated label", rank=4, OMIM_id="123456"),
        ])
        state = {"zeroShotResult": output}
        with patch.object(diseaseNormalize, "known_disease_name", return_value="CANONICAL SYNDROME"), patch.object(
            diseaseNormalize, "names_compatible", return_value=False
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual(result.ans[0].disease_name, "CANONICAL SYNDROME")
        self.assertEqual(result.ans[0].OMIM_id, "OMIM:123456")
        self.assertEqual(result.ans[0].rank, 1)
        self.assertEqual(state["zeroShotValidation"][0]["status"], "mismatch")
        self.assertFalse(state["zeroShotValidation"][0]["excluded"])

    def test_name_can_resolve_when_omim_is_missing(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Resolvable syndrome", rank=7, OMIM_id=None),
        ])
        state = {"zeroShotResult": output}
        with patch.object(
            diseaseNormalize, "disease_normalize",
            return_value=("654321", "RESOLVED SYNDROME", 0.99),
        ), patch.object(
            diseaseNormalize, "known_disease_name", return_value=None
        ), patch.object(
            diseaseNormalize, "names_compatible", return_value=True
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual(result.ans[0].OMIM_id, "OMIM:654321")
        self.assertEqual(result.ans[0].rank, 1)
        self.assertFalse(state["zeroShotValidation"][0]["excluded"])

    def test_resolvable_name_wins_when_supplied_omim_is_unknown(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Resolvable syndrome", rank=2, OMIM_id="999999"),
        ])
        state = {"zeroShotResult": output}
        with patch.object(
            diseaseNormalize, "known_disease_name", return_value=None
        ), patch.object(
            diseaseNormalize, "disease_normalize",
            return_value=("654321", "RESOLVED SYNDROME", 0.98),
        ), patch.object(
            diseaseNormalize, "names_compatible", return_value=True
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual(result.ans[0].OMIM_id, "OMIM:654321")
        self.assertEqual(result.ans[0].disease_name, "RESOLVED SYNDROME")
        self.assertEqual(state["zeroShotValidation"][0]["status"], "normalized")
        self.assertFalse(state["zeroShotValidation"][0]["excluded"])


if __name__ == "__main__":
    unittest.main()
