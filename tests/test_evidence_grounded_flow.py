import inspect
import unittest

from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
from agent.state.state_types import (
    DiagnosisFormatV2,
    DiagnosisOutputV2,
    EvidenceDraft,
    EvidenceExtractionOutput,
    EvidenceItem,
    MergedDiseaseCandidateV2,
    SourceRecord,
)
from agent.tools.candidate_metadata import candidate_id_for, to_v2_candidate
from agent.tools.evidence_extraction import (
    _select_relevant_chunks,
    _source_chunks,
    extract_evidence_for_candidates,
)
from agent.tools.evidence_validation import validate_v2_output
from agent.tools.literature_search import (
    build_literature_queries,
    matches_candidate_identity,
)
from agent.utils.response_serializer import serialize_node_result, serialize_public_state


class TestEvidenceGroundedFlow(unittest.TestCase):
    def setUp(self):
        self.candidate = MergedDiseaseCandidateV2(
            candidate_id="OMIM:123456",
            disease_name="Example syndrome",
            OMIM_id="OMIM:123456",
            gene_symbols=["EXG1"],
            synonyms=["Example disease"],
            phenotype_labels=["Hypotonia"],
            consensus_count=2,
            best_rank=1,
        )
        self.source = SourceRecord(
            source_id="SRC-1",
            source_type="literature",
            source_name="PubMed",
            title="Example syndrome and EXG1",
            identifier="PMID:12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            candidate_ids=["OMIM:123456"],
            content_type="abstract",
            raw_content="Example syndrome is associated with hypotonia and EXG1 variants.",
            citation_allowed=True,
        )
        self.evidence = EvidenceItem(
            evidence_id="EVD-1",
            candidate_id="OMIM:123456",
            source_id="SRC-1",
            evidence_type="core_phenotype",
            claim_en="The disease is associated with hypotonia.",
            claim_ja="この疾患は筋緊張低下と関連します。",
            excerpt="Example syndrome is associated with hypotonia",
            support="supports",
            confidence="high",
        )

    def test_candidate_ids_and_queries(self):
        self.assertEqual(candidate_id_for("Example syndrome", "123456"), "OMIM:123456")
        candidate = to_v2_candidate({
            "disease_name": "Example syndrome",
            "OMIM_id": "123456",
            "tool_rankings": [{"tool": "GestaltMatcher", "rank": 1, "rank_source": "api"}],
        })
        self.assertEqual(candidate.candidate_id, "OMIM:123456")
        self.assertEqual(candidate.tool_supports[0].rank_source, "api")
        self.assertTrue(any("Example syndrome" in query for query in build_literature_queries(self.candidate, ["Hypotonia"])))

    def test_literature_identity_rejects_conflicting_subtype(self):
        cdls1 = self.candidate.model_copy(update={
            "disease_name": "Cornelia de Lange syndrome 1; CDLS1",
        })
        self.assertFalse(matches_candidate_identity({
            "title": "First Cornelia de Lange Syndrome Type 4 caused by RAD21 deletion",
            "abstract": "This is a report of CDLS4.",
        }, cdls1))
        self.assertTrue(matches_candidate_identity({
            "title": "Cornelia de Lange syndrome: clinical features",
            "abstract": "A clinical review of Cornelia de Lange syndrome.",
        }, cdls1))

    def test_chunk_selection_reaches_late_full_text(self):
        source = self.source.model_copy(update={
            "raw_content": (
                "Introduction\n\n" + ("General background without the target terms. " * 180)
                + "\n\nResults\n\nNIPBL variants were identified in patients with developmental delay."
            )
        })
        self.assertGreater(len(_source_chunks(source)), 1)
        selected = _select_relevant_chunks(self.candidate, [source], ["Global developmental delay"])
        self.assertLessEqual(len(selected), 10)
        self.assertTrue(any("NIPBL variants" in chunk.text for chunk in selected))

    def test_evidence_contains_only_claim_and_sourceable_excerpt(self):
        class Structured:
            def invoke(self, _messages):
                return EvidenceExtractionOutput(evidence=[EvidenceDraft(
                    source_id="SRC-1",
                    evidence_type="core_phenotype",
                    claim_en="The disease is associated with hypotonia.",
                    claim_ja="この疾患は筋緊張低下と関連します。",
                    excerpt="Example syndrome is associated with hypotonia",
                    support="supports",
                    confidence="high",
                )])

        class FakeLLM:
            def get_structured_llm(self, _model):
                return Structured()

        evidence = extract_evidence_for_candidates(
            [self.candidate], [self.source], {"HP:1": "Hypotonia"}, {}, FakeLLM()
        )
        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertEqual(item.excerpt, "Example syndrome is associated with hypotonia")
        self.assertFalse(hasattr(item, "impact"))
        self.assertFalse(hasattr(item, "relationship"))
        self.assertFalse(hasattr(item, "patient_matches"))
        self.assertFalse(hasattr(item, "limitations"))

    def test_validator_needs_only_literature_evidence(self):
        diagnosis = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="OMIM:123456",
            rank=1,
            description_en="The patient finding is consistent with the reported phenotype.",
            description_ja="患者所見は文献に記載された表現型と整合します。",
            evidence_ids=["EVD-1"],
            literature_status="supported",
        )])
        validated = validate_v2_output(diagnosis, [self.candidate], [self.source], [self.evidence])
        self.assertEqual(validated.validation_status, "passed")
        self.assertEqual(validated.references[0].identifier, "PMID:12345678")

    def test_public_response_has_simple_numbered_references(self):
        diagnosis = DiagnosisOutputV2(
            ans=[DiagnosisFormatV2(
                candidate_id="OMIM:123456",
                disease_name_en="Example syndrome",
                disease_name_ja="Example syndrome",
                OMIM_id="123456",
                rank=1,
                description_en="The patient finding is consistent with the reported phenotype.",
                description_ja="患者所見は文献に記載された表現型と整合します。",
                evidence_ids=["EVD-1"],
                literature_status="supported",
            )],
            evidence=[self.evidence],
        )
        result = serialize_public_state({
            "evidence_mode": "v2",
            "sources": [self.source],
            "evidence": [self.evidence],
            "tentativeDiagnosis": diagnosis,
            "reflection": None,
            "finalDiagnosis": diagnosis,
        })
        item = result["finalDiagnosis"]["ans"][0]
        reference = result["finalDiagnosis"]["references"][0]
        self.assertEqual(item["citation_numbers"], [1])
        self.assertIn("[1]", item["description_en"])
        self.assertEqual(reference["number"], 1)
        self.assertEqual(reference["title"], self.source.title)
        self.assertEqual(reference["url"], self.source.url)
        self.assertEqual(reference["evidence"][0]["excerpt"], self.evidence.excerpt)
        self.assertNotIn("raw_content", result["sources"][0])
        self.assertNotIn("reasoning", item)
        self.assertNotIn("reference_catalog", result)
        self.assertNotIn("reference_records", result["finalDiagnosis"])

    def test_node_result_uses_same_simple_projection(self):
        node_result = serialize_node_result(
            "finalDiagnosisNode",
            {"finalDiagnosis": DiagnosisOutputV2(
                ans=[DiagnosisFormatV2(
                    candidate_id="OMIM:123456",
                    disease_name_en="Example syndrome",
                    disease_name_ja="Example syndrome",
                    OMIM_id="123456",
                    rank=1,
                    description_en="Supported by the cited literature.",
                    description_ja="引用文献により支持されます。",
                    evidence_ids=["EVD-1"],
                    literature_status="supported",
                )],
                evidence=[self.evidence],
            )},
            {"evidence_mode": "v2", "sources": [self.source], "evidence": [self.evidence]},
        )
        payload = node_result["result"]["finalDiagnosis"]
        self.assertEqual(payload["ans"][0]["citation_numbers"], [1])
        self.assertEqual(payload["references"][0]["number"], 1)

    def test_v2_default_and_original_node_order_are_preserved(self):
        self.assertEqual(inspect.signature(RareDiseaseDiagnosisPipeline.run).parameters["evidence_mode"].default, "v2")
        source = inspect.getsource(RareDiseaseDiagnosisPipeline._build_graph)
        self.assertLess(source.find('("candidateMetadataNode", "createDiagnosisNode")'), source.find('("diseaseNormalizeNode", "literatureSearchNode")'))
        self.assertLess(source.find('("literatureSearchNode", "evidenceExtractionNode")'), source.find('("evidenceExtractionNode", "reflectionNode")'))
        self.assertLess(source.find('("evidenceExtractionNode", "reflectionNode")'), source.find('("finalDiagnosisNode", "diseaseNormalizeForFinalNode")'))
