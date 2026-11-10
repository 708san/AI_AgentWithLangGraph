import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agent.utils.language_quality import (
    LanguageRepairOutput,
    has_obvious_language_mixing,
    repair_obvious_language_mixing,
)


class LanguageQualityTests(unittest.TestCase):
    def test_english_field_flags_appended_japanese_translation(self):
        self.assertTrue(has_obvious_language_mixing(
            "The evidence supports this candidate.\nこの根拠はこの候補を支持します。",
            "en",
        ))

    def test_japanese_field_flags_appended_english_translation(self):
        self.assertTrue(has_obvious_language_mixing(
            "この根拠は候補を支持します。 The evidence supports this candidate.",
            "ja",
        ))

    def test_japanese_field_allows_identifiers_and_known_disease_name(self):
        self.assertFalse(has_obvious_language_mixing(
            "Example syndrome（OMIM:123456）はNIPBL遺伝子と関連します。",
            "ja",
            known_names=["Example syndrome"],
        ))

    def test_english_field_allows_known_japanese_disease_name(self):
        self.assertFalse(has_obvious_language_mixing(
            "マルファン症候群 is considered with the reported findings.",
            "en",
            known_names=["マルファン症候群"],
        ))

    def test_repair_batches_mixed_fields_and_preserves_evidence_markers(self):
        item = SimpleNamespace(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            description_en=(
                "The evidence supports the candidate. "
                "この根拠は候補を支持します。 [[EVIDENCE:EVD-1]]"
            ),
            description_ja="この根拠は候補を支持します。[[EVIDENCE:EVD-1]]",
        )
        result = LanguageRepairOutput(repairs=[{
            "identity": "OMIM:123456",
            "field": "description_en",
            "text": "The evidence supports the candidate. [[EVIDENCE:EVD-1]]",
        }])
        runnable = Mock()
        runnable.invoke.return_value = result
        llm = Mock()
        llm.get_structured_llm.return_value = runnable

        repaired = repair_obvious_language_mixing(
            llm,
            [item],
            (("description_en", "description_ja"),),
        )

        self.assertIs(repaired[0], item)
        self.assertEqual(item.description_en, result.repairs[0].text)
        llm.get_structured_llm.assert_called_once()
        runnable.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
