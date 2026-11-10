import unittest
from unittest.mock import patch

from agent.state.state_types import ZeroShotFormat, ZeroShotOutput
from agent.tools import diseaseNormalize


class ZeroShotNormalizationTests(unittest.TestCase):
    def _state(self, *items):
        return {"zeroShotResult": ZeroShotOutput(ans=list(items))}

    def test_unconfirmed_candidate_is_retained_and_rows_are_renumbered(self):
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
            diseaseNormalize, "disease_name_matches_omim", return_value=True
        ), patch.object(
            diseaseNormalize, "resolve_known_disease_identity", return_value=(None, None, False)
        ), patch.object(
            diseaseNormalize, "disease_normalize_many", return_value=[(None, None, 0.0)]
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual([item.rank for item in result.ans], [1, 2, 3])
        self.assertEqual(
            [item.disease_name for item in result.ans],
            ["Known syndrome", "Unknown syndrome", "Known syndrome two"],
        )
        self.assertIsNone(result.ans[1].OMIM_id)
        validation = {item["disease_name"]: item for item in state["zeroShotValidation"]}
        self.assertFalse(validation["Unknown syndrome"]["excluded"])
        self.assertEqual(validation["Unknown syndrome"]["display_rank"], 2)
        self.assertEqual(validation["Known syndrome two"]["display_rank"], 3)

    def test_uniquely_resolved_name_corrects_conflicting_omim(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Wrong generated label", rank=4, OMIM_id="123456"),
        ])
        state = {"zeroShotResult": output}
        with patch.object(diseaseNormalize, "known_disease_name", side_effect=lambda value: {
            "OMIM:123456": "ID SYNDROME",
            "OMIM:654321": "NAME SYNDROME",
        }.get(value)), patch.object(
            diseaseNormalize, "resolve_known_disease_identity",
            return_value=("OMIM:654321", "NAME SYNDROME", False),
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual(len(result.ans), 1)
        self.assertEqual(result.ans[0].disease_name, "Wrong generated label")
        self.assertEqual(result.ans[0].OMIM_id, "OMIM:654321")
        self.assertEqual(state["zeroShotValidation"][0]["status"], "name_authoritative_id_corrected")
        self.assertFalse(state["zeroShotValidation"][0]["excluded"])

    def test_name_can_resolve_when_omim_is_missing(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Resolvable syndrome", rank=7, OMIM_id=None),
        ])
        state = {"zeroShotResult": output}
        with patch.object(
            diseaseNormalize, "resolve_known_disease_identity",
            return_value=("OMIM:654321", "RESOLVED SYNDROME", False),
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual(result.ans[0].OMIM_id, "OMIM:654321")
        self.assertEqual(result.ans[0].rank, 1)
        self.assertEqual(result.ans[0].disease_name, "Resolvable syndrome")
        self.assertFalse(state["zeroShotValidation"][0]["excluded"])

    def test_resolvable_name_wins_when_supplied_omim_is_unknown(self):
        output = ZeroShotOutput(ans=[
            ZeroShotFormat(disease_name="Resolvable syndrome", rank=2, OMIM_id="999999"),
        ])
        state = {"zeroShotResult": output}
        with patch.object(
            diseaseNormalize, "known_disease_name", return_value=None
        ), patch.object(
            diseaseNormalize, "resolve_known_disease_identity",
            return_value=("OMIM:654321", "RESOLVED SYNDROME", False),
        ):
            result = diseaseNormalize.normalize_zeroshot_results(state)

        self.assertEqual(result.ans[0].OMIM_id, "OMIM:654321")
        self.assertEqual(result.ans[0].disease_name, "Resolvable syndrome")
        self.assertEqual(state["zeroShotValidation"][0]["status"], "exact_name_match")
        self.assertFalse(state["zeroShotValidation"][0]["excluded"])


if __name__ == "__main__":
    unittest.main()
